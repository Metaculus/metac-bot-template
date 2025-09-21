# dashboard/streamlit_app.py
# -----------------------------------------------------------
# Spagbot Performance Dashboard (Streamlit)
#
# GOAL (non-coder friendly):
# - Always-on, web-based dashboard for collaborators.
# - Shows both "Key" (headline) and "Refined" performance metrics.
# - Reads forecasts/logs directly from your GitHub repo or a parquet snapshot.
#
# HOW TO USE:
# 1) Put this file in your repo at: dashboard/streamlit_app.py
# 2) (Option A: simplest) Point RAW_CSV_URL at your repo's raw forecasts.csv.
#    Example raw URL looks like:
#    https://raw.githubusercontent.com/<org_or_user>/<repo>/main/forecast_logs/forecasts.csv
# 3) (Option B: faster) Use the optional GitHub Action below to create a parquet snapshot
#    at dashboard/data/forecasts.parquet and set USE_PARQUET=True.
# 4) Deploy on Streamlit Community Cloud; set no secrets unless your data is private.
#
# NOTES:
# - The app tries to "auto-map" your column names to common internal names,
#   so it should work even if your CSV headers are slightly different.
# - Everything is heavily commented; you can skim the top and ignore the rest.
# -----------------------------------------------------------

from __future__ import annotations
import io
import os
import sys
import math
import json
import time
import textwrap
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# USER SETTINGS (edit these)
# -----------------------------
import os
from pathlib import Path

# Build paths relative to THIS file, so it works no matter the working directory.
APP_DIR = Path(__file__).resolve().parent

# Option B (faster): read from a parquet snapshot that lives next to this app:
# Expected location: <APP_DIR>/data/forecasts.parquet
USE_PARQUET = True  # you can also make this env-driven if you like
PARQUET_PATH = APP_DIR / "data" / "forecasts.parquet"

# Option A (fallback): raw CSV URL (used only if parquet isn't found or USE_PARQUET is False)
RAW_CSV_URL = os.getenv(
    "SPAGBOT_RAW_CSV_URL",
    "https://raw.githubusercontent.com/<ORG_OR_USER>/<REPO>/main/forecast_logs/forecasts.csv"
)

# If your repo is private and you want to read the raw URL with auth:
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# -----------------------------
# Column name auto-mapping
# -----------------------------
# We try to standardize to these internal names so the rest of the app is simple:
#   "qid", "pid", "question_title", "question_type",
#   "created_at", "submitted_at", "run_id",
#   "model", "ensemble", "forecast", "forecast_hi", "forecast_lo",
#   "outcome", "cdf_json", "mcq_vector_json",
#   "cost_usd", "tokens_in", "tokens_out", "tag", "lead_time_hours"
#
# We'll search your CSV for several possibilities per field:
COLUMN_ALIAS_MAP: Dict[str, List[str]] = {
    "qid": ["qid", "question_id", "metaculus_qid", "QID"],
    "pid": ["pid", "post_id", "metaculus_post_id", "PID"],
    "question_title": ["title", "question_title", "name"],
    "question_type": ["question_type", "qtype", "type"],
    "created_at": ["created_at", "timestamp", "run_timestamp", "ist_timestamp", "time"],
    "submitted_at": ["submitted_at", "submit_time", "metaculus_submit_ts"],
    "run_id": ["run_id", "git_sha", "run_sha", "commit_sha"],
    "model": ["model", "provider", "engine", "llm"],
    "ensemble": ["ensemble", "is_ensemble", "ensemble_name"],
    "forecast": ["forecast", "prob_yes", "p_yes", "p", "final_forecast"],
    "forecast_hi": ["forecast_hi", "p_hi", "upper", "hi"],
    "forecast_lo": ["forecast_lo", "p_lo", "lower", "lo"],
    "outcome": ["outcome", "resolved", "result", "y_true"],
    "cdf_json": ["cdf_json", "numeric_cdf_json", "cdf", "cdf_payload"],
    "mcq_vector_json": ["mcq_vector_json", "probability_yes_per_category", "mcq_probs", "mcq_json"],
    "cost_usd": ["cost_usd", "usd_cost", "cost"],
    "tokens_in": ["tokens_in", "prompt_tokens"],
    "tokens_out": ["tokens_out", "completion_tokens"],
    "tag": ["tag", "topic", "category"],
    "lead_time_hours": ["lead_time_hours", "lead_hours", "lead_time"]
}

# -----------------------------
# Utility: map columns
# -----------------------------
def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Create a new dict of standardized columns
    col_map: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for std, options in COLUMN_ALIAS_MAP.items():
        for opt in options:
            if opt.lower() in lower_cols:
                col_map[std] = lower_cols[opt.lower()]
                break
    # Rename where possible
    df = df.rename(columns={v: k for k, v in col_map.items()})
    return df

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    # Option B: Parquet (fast)
    if USE_PARQUET and PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        return _auto_map_columns(df)
    ...

    # Option A: raw CSV from GitHub
    if not RAW_CSV_URL or RAW_CSV_URL.startswith("https://raw.githubusercontent.com/<"):
        st.warning("Please set SPAGBOT_RAW_CSV_URL env var to your repo's raw forecasts.csv URL.")
        return pd.DataFrame()

    # If a token is provided and the URL is a GitHub raw URL, use authenticated request
    try:
        import requests
        headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        resp = requests.get(RAW_CSV_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.text
        df = pd.read_csv(io.StringIO(data))
        return _auto_map_columns(df)
    except Exception as e:
        st.error(f"Failed to fetch CSV: {e}")
        return pd.DataFrame()

# -----------------------------
# Metrics helpers (binary focus with graceful fallback)
# -----------------------------
def safe_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def brier_score(p: pd.Series, y: pd.Series) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    if mask.sum() == 0:
        return None
    return float(np.mean((p[mask] - y[mask]) ** 2))

def log_loss(p: pd.Series, y: pd.Series, eps: float = 1e-15) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    if mask.sum() == 0:
        return None
    p_clip = np.clip(p[mask].values, eps, 1 - eps)
    yv = y[mask].values
    ll = -np.mean(yv * np.log(p_clip) + (1 - yv) * np.log(1 - p_clip))
    return float(ll)

def hit_rate(p: pd.Series, y: pd.Series, threshold: float = 0.5) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    if mask.sum() == 0:
        return None
    preds = (p[mask] >= threshold).astype(int)
    return float((preds.values == y[mask].values).mean())

def calibration_bins(p: pd.Series, y: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    if mask.sum() == 0:
        return pd.DataFrame(columns=["bin_mid", "pred_mean", "obs_rate", "count"])
    p, y = p[mask], y[mask]
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    rows = []
    for i in range(n_bins):
        sel = idx == i
        if sel.sum() == 0:
            rows.append({"bin_mid": (bins[i] + bins[i+1]) / 2, "pred_mean": np.nan, "obs_rate": np.nan, "count": 0})
        else:
            rows.append({
                "bin_mid": (bins[i] + bins[i+1]) / 2,
                "pred_mean": float(p[sel].mean()),
                "obs_rate": float(y[sel].mean()),
                "count": int(sel.sum())
            })
    return pd.DataFrame(rows)

def sharpness(p: pd.Series) -> Optional[float]:
    # Sharpness ~ spread away from 0.5; higher is "sharper"
    p = safe_float_series(p)
    if p.notna().sum() == 0:
        return None
    return float(np.mean(np.abs(p - 0.5)))

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Spagbot Dashboard", layout="wide")

st.title("🔮 Spagbot Performance Dashboard")
st.caption("Always-on view of key + refined metrics for Metaculus tournament and cup submissions.")

df = load_data()
if df.empty:
    st.stop()

# Normalize some types
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
if "submitted_at" in df.columns:
    df["submitted_at"] = pd.to_datetime(df["submitted_at"], errors="coerce")

# Identify usability
has_outcome = "outcome" in df.columns and df["outcome"].notna().any()
has_model   = "model" in df.columns
has_type    = "question_type" in df.columns

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    # date range
    if "created_at" in df.columns and df["created_at"].notna().any():
        min_d = pd.to_datetime(df["created_at"].min())
        max_d = pd.to_datetime(df["created_at"].max())
        d1, d2 = st.date_input("Date range", value=(min_d.date(), max_d.date()))
        mask_date = (df["created_at"].dt.date >= d1) & (df["created_at"].dt.date <= d2)
        df = df[mask_date]

    # question type
    if has_type:
        types = sorted([str(x) for x in df["question_type"].dropna().unique()])
        if types:
            sel_types = st.multiselect("Question types", types, default=types)
            df = df[df["question_type"].astype(str).isin(sel_types)]

    # model
    if has_model:
        models = sorted([str(x) for x in df["model"].dropna().unique()])
        if models:
            sel_models = st.multiselect("Models", models, default=models)
            df = df[df["model"].astype(str).isin(sel_models)]

    # tag/topic if present
    if "tag" in df.columns:
        tags = sorted([str(x) for x in df["tag"].dropna().unique()])
        show_tag = st.checkbox("Filter by tag/topic", value=False)
        if show_tag and tags:
            sel_tags = st.multiselect("Tags", tags, default=tags[:10])
            df = df[df["tag"].astype(str).isin(sel_tags)]

st.markdown("### Key metrics")

# Choose ensemble-only vs all
show_ensemble_only = st.toggle("Show ENSEMBLE forecasts only (if flagged)", value=False)
df_key = df.copy()
if show_ensemble_only and "ensemble" in df.columns:
    df_key = df_key[df_key["ensemble"].astype(str).str.lower().isin(["1", "true", "yes", "ensemble"])]

# Identify forecast + outcome columns
p_col = "forecast" if "forecast" in df_key.columns else None
y_col = "outcome" if "outcome" in df_key.columns else None

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    brier = brier_score(df_key[p_col], df_key[y_col]) if (p_col and y_col) else None
    st.metric("Brier (binary)", f"{brier:.3f}" if brier is not None else "—")

with col2:
    ll = log_loss(df_key[p_col], df_key[y_col]) if (p_col and y_col) else None
    st.metric("Log loss", f"{ll:.3f}" if ll is not None else "—")

with col3:
    hr = hit_rate(df_key[p_col], df_key[y_col]) if (p_col and y_col) else None
    st.metric("Hit rate @0.5", f"{100*hr:.1f}%" if hr is not None else "—")

with col4:
    sh = sharpness(df_key[p_col]) if p_col else None
    st.metric("Sharpness (|p-0.5|)", f"{sh:.3f}" if sh is not None else "—")

with col5:
    n_sub = int(df_key[p_col].notna().sum()) if p_col else len(df_key)
    st.metric("# forecasts", f"{n_sub}")

with col6:
    last_ts = None
    for c in ["submitted_at", "created_at"]:
        if c in df_key.columns and df_key[c].notna().any():
            last_ts = pd.to_datetime(df_key[c].max())
            break
    st.metric("Last update", last_ts.strftime("%Y-%m-%d %H:%M") if last_ts is not None else "—")

st.divider()

# Tabs for refined analysis
tab_cal, tab_models, tab_questions, tab_costs, tab_runs = st.tabs(
    ["Calibration & Sharpness", "Per-Model & Type", "Per-Question drilldown", "Costs/Tokens", "Run Stability"]
)

with tab_cal:
    st.subheader("Reliability (Calibration) curve")
    if p_col and y_col and df_key[y_col].notna().any():
        bin_df = calibration_bins(df_key[p_col], df_key[y_col], n_bins=10)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bin_df["bin_mid"], y=bin_df["obs_rate"],
            mode="lines+markers", name="Observed",
            hovertext=[f"n={c}" for c in bin_df["count"]]
        ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            name="Perfect calibration", line=dict(dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Forecast probability",
            yaxis_title="Observed frequency",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Bubbles show the observed resolution per probability bin; dashed line is perfect calibration.")
    else:
        st.info("Need binary outcomes to show calibration.")

    st.subheader("Forecast distribution (sharpness)")
    if p_col:
        fig2 = px.histogram(df_key, x=p_col, nbins=20, height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No forecast column detected.")

with tab_models:
    c1, c2 = st.columns(2)
    with c1:
        if has_model:
            st.markdown("**Model-wise performance**")
            if y_col and p_col:
                g = df_key.dropna(subset=[p_col, y_col]).groupby("model").apply(
                    lambda s: pd.Series({
                        "n": len(s),
                        "brier": brier_score(s[p_col], s[y_col]),
                        "log_loss": log_loss(s[p_col], s[y_col]),
                        "hit@0.5": hit_rate(s[p_col], s[y_col])
                    })
                ).reset_index()
                st.dataframe(g.sort_values("brier", ascending=True), use_container_width=True)
            else:
                st.info("Need outcomes for per-model performance.")

    with c2:
        if has_type:
            st.markdown("**By question type**")
            if y_col and p_col:
                g2 = df_key.dropna(subset=[p_col, y_col]).groupby("question_type").apply(
                    lambda s: pd.Series({
                        "n": len(s),
                        "brier": brier_score(s[p_col], s[y_col]),
                        "log_loss": log_loss(s[p_col], s[y_col]),
                        "hit@0.5": hit_rate(s[p_col], s[y_col])
                    })
                ).reset_index()
                st.dataframe(g2.sort_values("brier", ascending=True), use_container_width=True)
            else:
                st.info("Need outcomes for per-type performance.")

with tab_questions:
    st.markdown("**Questions**")
    # Build an identifier for selection
    if "qid" in df_key.columns:
        opts = (
            df_key[["qid", "question_title"]]
            .drop_duplicates()
            .sort_values("qid")
        )
        display = [f"Q{int(q)} – {str(t)[:80]}" for q, t in zip(opts["qid"], opts["question_title"].fillna(""))]
        chosen = st.selectbox("Pick a question:", display)
        if chosen:
            qid = int(chosen.split("–")[0].strip().replace("Q", "").strip())
            dd = df_key[df_key["qid"] == qid].copy()
            st.write(f"### Q{qid} — {dd['question_title'].dropna().iloc[0] if dd['question_title'].notna().any() else ''}")

            # Time series of forecasts
            if "created_at" in dd.columns and p_col in dd.columns:
                dd = dd.sort_values("created_at")
                fig = px.line(dd, x="created_at", y=p_col, color=("model" if has_model else None),
                              markers=True, title="Forecast history")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Show raw rows for transparency
            st.markdown("**All rows for this question**")
            st.dataframe(dd, use_container_width=True)
    else:
        st.info("No QID column found; cannot build per-question drilldown.")

with tab_costs:
    st.subheader("Cost & tokens")
    # If cost info exists
    if "cost_usd" in df_key.columns:
        fig = px.box(df_key.dropna(subset=["cost_usd"]), y="cost_usd", points="all", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Total cost (visible rows): ${df_key['cost_usd'].fillna(0).sum():.2f}")
    else:
        st.info("No cost_usd column found.")

    c1, c2 = st.columns(2)
    with c1:
        if "tokens_in" in df_key.columns:
            st.bar_chart(df_key["tokens_in"].dropna())
        else:
            st.info("No tokens_in column.")
    with c2:
        if "tokens_out" in df_key.columns:
            st.bar_chart(df_key["tokens_out"].dropna())
        else:
            st.info("No tokens_out column.")

with tab_runs:
    st.subheader("Stability across runs/commits")
    # We estimate stability by grouping on (qid, run_id) when available
    if "run_id" in df_key.columns and "qid" in df_key.columns and p_col in df_key.columns:
        g = (
            df_key.dropna(subset=[p_col])
            .groupby(["qid", "run_id"])[p_col]
            .mean()
            .reset_index()
        )
        # Spread of orecasts per Q across runs
        spread = g.groupby("qid")[p_col].agg(["count", "std", "min", "max"]).reset_index()
        st.dataframe(spread.sort_values("std", ascending=False), use_container_width=True)
        st.caption("Tip: lower std suggests more stable forecasts for a question across runs.")
    else:
        st.info("Need run_id and qid to compute cross-run stability.")

st.divider()
st.markdown("**Data source**")
st.code(f"Parquet: {USE_PARQUET} | RAW_CSV_URL: {RAW_CSV_URL}", language="text")
st.caption("If parquet is enabled, the app reads dashboard/data/forecasts.parquet. Otherwise, it fetches forecasts.csv from your repo.")
