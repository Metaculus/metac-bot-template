# Dashboard/streamlit_app.py
# =============================================================================
# Spagbot Performance Dashboard (Streamlit)
# -----------------------------------------------------------------------------
# - Loads forecasts from a Parquet snapshot (fast) OR from a raw CSV URL.
# - Auto-detects/renames your repo's columns to a standard schema.
# - Headline + refined metrics for *all permutations* (binary, numeric, MCQ).
# - New tabs:
#     • Numeric permutations: pinball loss (q=0.1/0.5/0.9), 80% coverage, sharpness
#     • MCQ permutations: cross-entropy (log-loss), multiclass Brier, Top-1 accuracy
# - Downloads: current view CSV/Parquet; repo parquet; human logs ZIP.
# =============================================================================

from __future__ import annotations

import io
import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Configuration knobs (edit via env/secrets; no code edits required later)
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# Toggle parquet vs. CSV by environment variable/secret:
#   - In Streamlit Cloud: Settings → Secrets:  USE_PARQUET = true
USE_PARQUET = os.getenv("USE_PARQUET", "true").lower() in {"1", "true", "yes"}

# Parquet path (local to repo)
PARQUET_PATH = APP_DIR / "data" / "forecasts.parquet"

# CSV fallback (used if USE_PARQUET is false or parquet is missing)
RAW_CSV_URL = os.getenv(
    "SPAGBOT_RAW_CSV_URL",
    "https://raw.githubusercontent.com/<ORG_OR_USER>/<REPO>/main/forecast_logs/forecasts.csv",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # only needed for private CSVs


# -----------------------------------------------------------------------------
# Column auto-mapping
# -----------------------------------------------------------------------------
# We map your actual column names to a standard schema so downstream code
# doesn't need to special-case every name.
#
# Based on your column list, we cover:
#   - IDs/titles/times
#   - binary forecast permutations (binary_prob__*)
#   - outcomes (resolved*, resolved_outcome_label)
#   - costs (cost_usd__total, cost_usd__research, cost_usd__classifier)
#   - numeric and MCQ permutations are handled dynamically by prefix
#
# Standard schema keys we use in the app below:
#   qid, qtitle, qtype, created_at, closes_at, resolves_at,
#   run_id, run_time, purpose, tournament_id,
#   outcome (binary 0/1), outcome_label (text), resolved_value (numeric),
#   cost_usd_total, cost_usd_research, cost_usd_classifier
#
COLUMN_ALIAS_MAP: Dict[str, List[str]] = {
    # question identifiers / labels
    "qid": ["question_id", "qid", "metaculus_qid"],
    "qtitle": ["question_title", "title", "name"],
    "qtype": ["question_type", "qtype", "type"],

    # times
    "created_at": ["run_time_iso", "created_time_iso", "created_at"],
    "closes_at": ["closes_time_iso"],
    "resolves_at": ["resolves_time_iso"],

    # run metadata
    "run_id": ["run_id"],
    "purpose": ["purpose"],
    "git_sha": ["git_sha"],
    "tournament_id": ["tournament_id"],

    # outcomes / resolution
    "outcome_label": ["resolved_outcome_label"],
    "resolved_flag": ["resolved"],  # True/False
    "resolved_value": ["resolved_value"],  # numeric outcomes for numeric questions

    # costs (present in your data)
    "cost_usd_total": ["cost_usd__total", "cost_usd_total", "cost_usd"],
    "cost_usd_research": ["cost_usd__research"],
    "cost_usd_classifier": ["cost_usd__classifier"],
}

# Binary forecast permutations (column -> human-friendly label)
BINARY_PERM_LABELS: Dict[str, str] = {
    "binary_prob__ensemble": "Ensemble (default)",
    "binary_prob__ensemble_no_gtmc1": "Ensemble (no GTMC1)",
    "binary_prob__ensemble_uniform_weights": "Ensemble (uniform weights)",
    "binary_prob__ensemble_no_bmc_no_gtmc1": "No BMC + no GTMC1 (simple avg)",
    "binary_prob__ensemble_no_research": "Ensemble (no research)",
    "binary_prob__ensemble_no_research_no_gtmc1": "Ensemble (no research, no GTMC1)",
    "binary_prob__ensemble_no_research_uniform_weights": "Ensemble (no research, uniform)",
    "binary_prob__ensemble_no_research_no_bmc_no_gtmc1": "No research + no BMC + no GTMC1",
    "binary_prob__OpenRouter-Default": "OpenRouter-Default (single model)",
}

# Prefixes for numeric permutations: we expect sets of p10/p50/p90 under the same suffix
NUMERIC_PREFIXES = ["numeric_p10__", "numeric_p50__", "numeric_p90__"]

# Prefix for MCQ permutations (prob vectors as JSON of label->prob)
MCQ_PREFIX = "mcq_json__"


# -----------------------------------------------------------------------------
# Helpers: load + normalize data
# -----------------------------------------------------------------------------
def _first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """Return the first column in `names` that exists in df (case-sensitive)."""
    for n in names:
        if n in df.columns:
            return n
    return None


def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create standard columns (qid, outcome, cost_usd_total, etc.) if present."""
    out = df.copy()

    # Map straightforward aliases (we keep originals too)
    for std, options in COLUMN_ALIAS_MAP.items():
        name = _first_present(out, options)
        if name and std not in out.columns:
            out[std] = out[name]

    # Ensure datetime for key times (best effort)
    for tcol in ["created_at", "closes_at", "resolves_at"]:
        if tcol in out.columns:
            out[tcol] = pd.to_datetime(out[tcol], errors="coerce")

    # Derive a binary `outcome` from outcome_label for binary questions (YES/NO).
    if "outcome_label" in out.columns:
        lbl = out["outcome_label"].astype(str).str.strip().str.lower()
        out["outcome"] = np.where(
            lbl.isin(["yes", "true", "1"]),
            1.0,
            np.where(lbl.isin(["no", "false", "0"]), 0.0, np.nan),
        )

    # resolved_value stays as-is for numeric questions (float)
    if "resolved_value" in out.columns:
        out["resolved_value"] = pd.to_numeric(out["resolved_value"], errors="coerce")

    return out


@st.cache_data(show_spinner=True)
def load_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _auto_map_columns(df)


@st.cache_data(show_spinner=True)
def load_from_csv(url: str, token: str = "") -> pd.DataFrame:
    import requests
    headers = {"Authorization": f"token {token}"} if token else {}
    resp = requests.get(url, headers=headers, timeout=45)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return _auto_map_columns(df)


def load_data() -> pd.DataFrame:
    """Load according to mode; show helpful guidance if missing."""
    if USE_PARQUET:
        if PARQUET_PATH.exists():
            return load_from_parquet(PARQUET_PATH)
        else:
            st.warning(f"USE_PARQUET=True but file not found at: {PARQUET_PATH}")
            st.info("Falling back to CSV (RAW_CSV_URL) if configured.")

    if RAW_CSV_URL and "raw.githubusercontent.com/<ORG_OR_USER>/<REPO>" not in RAW_CSV_URL:
        try:
            return load_from_csv(RAW_CSV_URL, GITHUB_TOKEN)
        except Exception as e:
            st.error(f"Failed to fetch CSV from RAW_CSV_URL.\n{e}")

    st.error(
        "No data source available.\n"
        "Either place a parquet at 'Dashboard/data/forecasts.parquet' and set USE_PARQUET=true,\n"
        "or set SPAGBOT_RAW_CSV_URL (and GITHUB_TOKEN for private repos)."
    )
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------
def safe_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

# ---- Binary (already in your previous version) ----
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

def sharpness(p: pd.Series) -> Optional[float]:
    p = safe_float_series(p)
    if p.notna().sum() == 0:
        return None
    return float(np.mean(np.abs(p - 0.5)))

# ---- Numeric (new): quantile/pinball loss, coverage, sharpness ----
def pinball_loss(y_true: np.ndarray, qhat: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for a single quantile q in (0,1)."""
    diff = y_true - qhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def numeric_metrics_from_quantiles(y: pd.Series, p10: pd.Series, p50: pd.Series, p90: pd.Series) -> Dict[str, Optional[float]]:
    """Compute average pinball loss at q=0.1,0.5,0.9; 80% coverage; and interval width."""
    yv = safe_float_series(y).values
    q10 = safe_float_series(p10).values
    q50 = safe_float_series(p50).values
    q90 = safe_float_series(p90).values
    mask = np.isfinite(yv) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
    if mask.sum() == 0:
        return {"pinball_avg": None, "coverage80": None, "interval_width": None, "n": 0}

    yv, q10, q50, q90 = yv[mask], q10[mask], q50[mask], q90[mask]
    # Enforce sensible order (in case of minor numeric issues)
    q10, q50, q90 = np.minimum(q10, q50), np.clip(q50, np.minimum(q10, q50), np.maximum(q10, q90)), np.maximum(q50, q90)

    l10 = pinball_loss(yv, q10, 0.1)
    l50 = pinball_loss(yv, q50, 0.5)
    l90 = pinball_loss(yv, q90, 0.9)
    coverage = float(np.mean((yv >= q10) & (yv <= q90)))
    width = float(np.mean(q90 - q10))
    return {"pinball_avg": (l10 + l50 + l90) / 3.0, "coverage80": coverage, "interval_width": width, "n": int(mask.sum())}

# ---- MCQ (new): cross-entropy, multiclass Brier, top-1 ----
def parse_mcq_json(col: pd.Series) -> List[Optional[Dict[str, float]]]:
    """Parse a column of JSON strings -> list of dict(label->prob)."""
    out: List[Optional[Dict[str, float]]] = []
    for x in col.fillna("").astype(str):
        x = x.strip()
        if not x or x.lower() == "none" or x == "nan":
            out.append(None)
            continue
        try:
            d = json.loads(x)
            if not isinstance(d, dict):
                out.append(None); continue
            # Normalize (just in case)
            total = float(sum(float(v) for v in d.values()))
            if total > 0:
                d = {k: float(v) / total for k, v in d.items()}
            out.append(d)
        except Exception:
            out.append(None)
    return out

def mcq_metrics(y_label: pd.Series, probs_list: List[Optional[Dict[str, float]]]) -> Dict[str, Optional[float]]:
    """Compute cross-entropy (NLL), multiclass Brier, top-1 accuracy."""
    y = y_label.fillna("").astype(str).str.strip()
    ce_vals, brier_vals, top1_vals = [], [], []
    n = 0
    for truth, probs in zip(y, probs_list):
        if not truth or probs is None:
            continue
        # Target distribution one-hot over the observed label
        p = float(probs.get(truth, 0.0))
        # Cross-entropy (negative log likelihood); epsilon for stability
        eps = 1e-15
        ce_vals.append(-np.log(max(p, eps)))
        # Multiclass Brier: sum_i (p_i - y_i)^2 ; here only y_truth=1, others 0
        b = (1 - p) ** 2 + sum((pv - 0.0) ** 2 for k, pv in probs.items() if k != truth)
        brier_vals.append(b)
        # Top-1 accuracy: argmax matches truth?
        top_label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        top1_vals.append(1.0 if top_label == truth else 0.0)
        n += 1
    if n == 0:
        return {"cross_entropy": None, "brier_mc": None, "top1_acc": None, "n": 0}
    return {
        "cross_entropy": float(np.mean(ce_vals)),
        "brier_mc": float(np.mean(brier_vals)),
        "top1_acc": float(np.mean(top1_vals)),
        "n": n,
    }


# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Spagbot Dashboard", layout="wide")
st.title("🔮 Spagbot Performance Dashboard")
st.caption("Always-on key + refined metrics for the Metaculus AI benchmarks (tournament + cup).")

with st.sidebar:
    st.header("Controls")
    show_debug = st.checkbox("Show data-source debug", value=False)
    # simple filters
    date_filter_on = st.checkbox("Filter by created date", value=False)

df = load_data()
if df.empty:
    st.stop()

# Basic normalization for filters / displays
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Identify permutations present in df
binary_perm_cols = [c for c in df.columns if c.startswith("binary_prob__")]
# For numeric, build a dict of variants -> (p10,p50,p90 columns) if present
numeric_variants: Dict[str, Dict[str, str]] = {}
for pref in NUMERIC_PREFIXES:
    for c in df.columns:
        if c.startswith(pref):
            suffix = c[len(pref):]  # e.g., "ensemble", "ensemble_no_gtmc1"
            numeric_variants.setdefault(suffix, {})
            numeric_variants[suffix][pref] = c
# Keep only variants with at least p10 & p90 (p50 optional but preferred)
numeric_variants = {
    k: v for k, v in numeric_variants.items()
    if ("numeric_p10__" in v and "numeric_p90__" in v)
}

# MCQ variants from mcq_json__*
mcq_variants = [c for c in df.columns if c.startswith(MCQ_PREFIX)]

# Pretty labels for binary permutations
perm_labels = {c: (c in BINARY_PERM_LABELS and BINARY_PERM_LABELS[c]) or c.replace("binary_prob__", "").replace("_", " ") for c in binary_perm_cols}

# Outcome availability (binary)
has_outcome = "outcome" in df.columns and df["outcome"].notna().any()

# Optional date filter
if date_filter_on and "created_at" in df.columns and df["created_at"].notna().any():
    min_d = pd.to_datetime(df["created_at"].min()).date()
    max_d = pd.to_datetime(df["created_at"].max()).date()
    d1, d2 = st.sidebar.date_input("Date range", value=(min_d, max_d))
    mask = (df["created_at"].dt.date >= d1) & (df["created_at"].dt.date <= d2)
    df = df[mask]

# -----------------------------------------------------------------------------
# Key Metrics (default to ensemble if present)
# -----------------------------------------------------------------------------
st.markdown("## Key metrics")

# Choose which permutation powers the headline (defaults to ensemble)
default_perm = "binary_prob__ensemble" if "binary_prob__ensemble" in perm_labels else (binary_perm_cols[0] if binary_perm_cols else None)
p_col = st.selectbox(
    "Which permutation to treat as the headline forecast?",
    options=[default_perm] + [c for c in binary_perm_cols if c != default_perm] if default_perm else binary_perm_cols,
    format_func=lambda c: perm_labels.get(c, c),
) if binary_perm_cols else None

# Compute headline metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
if p_col is not None:
    with c1:
        brier = brier_score(df[p_col], df["outcome"]) if has_outcome else None
        st.metric("Brier (binary)", f"{brier:.3f}" if brier is not None else "—")
    with c2:
        ll = log_loss(df[p_col], df["outcome"]) if has_outcome else None
        st.metric("Log loss", f"{ll:.3f}" if ll is not None else "—")
    with c3:
        hr = hit_rate(df[p_col], df["outcome"]) if has_outcome else None
        st.metric("Hit rate @0.5", f"{100*hr:.1f}%" if hr is not None else "—")
    with c4:
        sh = sharpness(df[p_col])
        st.metric("Sharpness (|p-0.5|)", f"{sh:.3f}" if sh is not None else "—")
else:
    for col in (c1, c2, c3, c4):
        with col:
            st.metric("—", "—")

with c5:
    n_sub = int(df[p_col].notna().sum()) if p_col else len(df)
    st.metric("forecasts", f"{n_sub}")

with c6:
    last_ts = None
    for c in ["created_at", "resolves_at", "closes_at"]:
        if c in df.columns and df[c].notna().any():
            last_ts = pd.to_datetime(df[c].max())
            break
    st.metric("Last update", last_ts.strftime("%Y-%m-%d %H:%M") if last_ts is not None else "—")

st.divider()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_cal, tab_compare, tab_numeric, tab_mcq, tab_questions, tab_costs, tab_runs, tab_downloads = st.tabs(
    [
        "Calibration & Sharpness",
        "Compare permutations (binary)",
        "Numeric permutations",
        "MCQ permutations",
        "Per-Question drilldown",
        "Costs/Tokens",
        "Run Stability",
        "Downloads",
    ]
)

# --- Tab: Calibration & sharpness for the selected permutation ---
with tab_cal:
    st.subheader("Reliability (Calibration) curve")
    if p_col is not None and has_outcome:
        bins = np.linspace(0, 1, 11)
        p = pd.to_numeric(df[p_col], errors="coerce")
        y = pd.to_numeric(df["outcome"], errors="coerce")
        mask = p.notna() & y.notna()
        if mask.sum() > 0:
            idx = np.digitize(p[mask], bins) - 1
            rows = []
            for i in range(10):
                sel = idx == i
                if sel.sum() == 0:
                    rows.append({"bin_mid": 0.05 + 0.1 * i, "pred_mean": np.nan, "obs_rate": np.nan, "count": 0})
                else:
                    rows.append({
                        "bin_mid": 0.05 + 0.1 * i,
                        "pred_mean": float(p[mask][sel].mean()),
                        "obs_rate": float(y[mask][sel].mean()),
                        "count": int(sel.sum()),
                    })
            bdf = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bdf["bin_mid"], y=bdf["obs_rate"],
                mode="lines+markers", name="Observed",
                hovertext=[f"n={c}" for c in bdf["count"]]
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")
            ))
            fig.update_layout(
                xaxis_title="Forecast probability",
                yaxis_title="Observed frequency",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resolved rows yet for calibration.")
    else:
        st.info("Need binary outcomes and a forecast column to show calibration.")

    st.subheader("Forecast distribution (sharpness)")
    if p_col is not None:
        fig2 = px.histogram(df.dropna(subset=[p_col]), x=p_col, nbins=20, height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No forecast column detected.")

# --- Tab: Compare permutations (binary) ---
with tab_compare:
    st.subheader("Headline metrics by permutation (binary)")
    if not binary_perm_cols:
        st.info("No binary forecast columns (binary_prob__*) found.")
    else:
        rows = []
        for col in binary_perm_cols:
            row = {"perm_column": col, "perm_label": perm_labels.get(col, col)}
            p = pd.to_numeric(df[col], errors="coerce")
            if has_outcome:
                y = pd.to_numeric(df["outcome"], errors="coerce")
                row["Brier"] = brier_score(p, y)
                row["LogLoss"] = log_loss(p, y)
                row["Hit@0.5"] = hit_rate(p, y)
            else:
                row["Brier"] = None
                row["LogLoss"] = None
                row["Hit@0.5"] = None
            row["Sharpness|p-0.5|"] = sharpness(p)
            row["#Forecasts"] = int(p.notna().sum())
            rows.append(row)
        mdf = pd.DataFrame(rows)
        if has_outcome and mdf["Brier"].notna().any():
            mdf = mdf.sort_values("Brier", na_position="last")
        st.dataframe(
            mdf[["perm_label", "Brier", "LogLoss", "Hit@0.5", "Sharpness|p-0.5|", "#Forecasts"]],
            use_container_width=True
        )
        st.caption("Tip: Click column headers to sort. Lower Brier/LogLoss is better; higher Hit@0.5 is better.")

# --- Tab: Numeric permutations (NEW) ---
with tab_numeric:
    st.subheader("Numeric permutations (q10/q50/q90) — pinball loss, coverage, sharpness")
    if not numeric_variants:
        st.info("No numeric permutations found (expected columns like numeric_p10__/numeric_p50__/numeric_p90__*).")
    elif "resolved_value" not in df.columns or df["resolved_value"].notna().sum() == 0:
        st.info("No numeric outcomes present (resolved_value) to evaluate numeric forecasts.")
    else:
        rows = []
        for variant, cols in numeric_variants.items():
            p10_col = cols.get("numeric_p10__")
            p50_col = cols.get("numeric_p50__")
            p90_col = cols.get("numeric_p90__")
            # If p50 missing, approximate with midpoint (just for pinball @0.5)
            if p50_col is None and p10_col and p90_col:
                df["_mid_tmp_"] = (pd.to_numeric(df[p10_col], errors="coerce") + pd.to_numeric(df[p90_col], errors="coerce")) / 2.0
                p50_col = "_mid_tmp_"

            metrics = numeric_metrics_from_quantiles(
                y=df["resolved_value"],
                p10=df[p10_col] if p10_col else pd.Series(dtype=float),
                p50=df[p50_col] if p50_col else pd.Series(dtype=float),
                p90=df[p90_col] if p90_col else pd.Series(dtype=float),
            )
            rows.append({
                "variant": variant,
                "Pinball(0.1/0.5/0.9) avg": metrics["pinball_avg"],
                "Coverage 80%": metrics["coverage80"],
                "Interval width (P90-P10)": metrics["interval_width"],
                "#Resolved": metrics["n"],
            })
        ndf = pd.DataFrame(rows)
        # Sort by pinball if available
        if ndf["Pinball(0.1/0.5/0.9) avg"].notna().any():
            ndf = ndf.sort_values("Pinball(0.1/0.5/0.9) avg", na_position="last")
        st.dataframe(ndf, use_container_width=True)

        # Optional visualization for a chosen variant
        chosen_var = st.selectbox("Variant to visualize", [r["variant"] for r in rows if r["#Resolved"] > 0] or [])
        if chosen_var:
            cols = numeric_variants[chosen_var]
            p10_col, p50_col, p90_col = cols.get("numeric_p10__"), cols.get("numeric_p50__"), cols.get("numeric_p90__")
            if p50_col is None and p10_col and p90_col:
                df["_mid_tmp_"] = (pd.to_numeric(df[p10_col], errors="coerce") + pd.to_numeric(df[p90_col], errors="coerce")) / 2.0
                p50_col = "_mid_tmp_"
            dplot = df.dropna(subset=["resolved_value", p10_col, p90_col]).copy()
            dplot["inside_80"] = ((dplot["resolved_value"] >= pd.to_numeric(dplot[p10_col], errors="coerce")) &
                                  (dplot["resolved_value"] <= pd.to_numeric(dplot[p90_col], errors="coerce"))).astype(int)
            fig3 = px.scatter(
                dplot,
                x="resolved_value",
                y=p50_col if p50_col else p10_col,
                color="inside_80",
                title=f"Resolved vs. central estimate — {chosen_var}",
                labels={"inside_80": "Inside 80% interval"},
            )
            fig3.update_layout(height=420)
            st.plotly_chart(fig3, use_container_width=True)

# --- Tab: MCQ permutations (NEW) ---
with tab_mcq:
    st.subheader("MCQ permutations — cross-entropy, multiclass Brier, Top-1")
    if not mcq_variants:
        st.info("No MCQ permutations found (expected columns like mcq_json__*).")
    elif "outcome_label" not in df.columns or df["outcome_label"].notna().sum() == 0:
        st.info("No resolved outcome labels present to evaluate MCQ forecasts.")
    else:
        rows = []
        for col in mcq_variants:
            probs_list = parse_mcq_json(df[col])
            m = mcq_metrics(df["outcome_label"], probs_list)
            rows.append({
                "variant": col[len(MCQ_PREFIX):],  # strip prefix for readability
                "Cross-entropy": m["cross_entropy"],
                "Brier (MCQ)": m["brier_mc"],
                "Top-1 acc": m["top1_acc"],
                "#Resolved": m["n"],
            })
        mdf = pd.DataFrame(rows)
        # Sort by cross-entropy if available
        if mdf["Cross-entropy"].notna().any():
            mdf = mdf.sort_values("Cross-entropy", na_position="last")
        st.dataframe(mdf, use_container_width=True)
        st.caption("Lower is better for Cross-entropy and Brier; higher is better for Top-1 accuracy.")

# --- Tab: Per-Question drilldown ---
with tab_questions:
    st.subheader("Questions")
    if "qid" in df.columns:
        # Build the selector
        opts = (
            df[["qid", "qtitle"]]
            .dropna(subset=["qid"])
            .drop_duplicates()
            .sort_values("qid")
        )
        display = [f"Q{int(q)} – {str(t)[:80]}" for q, t in zip(opts["qid"], opts["qtitle"].fillna(""))]
        chosen = st.selectbox("Pick a question:", display if display else ["(no questions found)"])
        if display:
            qid = int(chosen.split("–")[0].strip().replace("Q", "").strip())
            dd = df[df["qid"] == qid].copy()
            title = dd["qtitle"].dropna().iloc[0] if dd["qtitle"].notna().any() else "(no title)"
            st.write(f"### Q{qid} — {title}")

            # Binary: plot chosen permutation over time
            if binary_perm_cols:
                chosen_perm = st.selectbox(
                    "Binary permutation to plot",
                    binary_perm_cols,
                    index=max(0, binary_perm_cols.index(p_col)) if p_col in binary_perm_cols else 0,
                    format_func=lambda c: perm_labels.get(c, c),
                )
                if "created_at" in dd.columns and chosen_perm in dd.columns:
                    dd = dd.sort_values("created_at")
                    fig = px.line(dd, x="created_at", y=chosen_perm, markers=True, title="Forecast history (binary)")
                    fig.update_layout(height=360)
                    st.plotly_chart(fig, use_container_width=True)

            # Numeric: show interval width across runs (if present)
            some_numeric = [v for v in numeric_variants if "numeric_p10__" in numeric_variants[v] and "numeric_p90__" in numeric_variants[v]]
            if some_numeric:
                nv = st.selectbox("Numeric variant", some_numeric)
                p10c = numeric_variants[nv]["numeric_p10__"]; p90c = numeric_variants[nv]["numeric_p90__"]
                if "created_at" in dd.columns and p10c in dd.columns and p90c in dd.columns:
                    dd = dd.sort_values("created_at")
                    dd["_interval_width"] = pd.to_numeric(dd[p90c], errors="coerce") - pd.to_numeric(dd[p10c], errors="coerce")
                    figw = px.line(dd, x="created_at", y="_interval_width", markers=True, title="Interval width (P90-P10)")
                    figw.update_layout(height=300)
                    st.plotly_chart(figw, use_container_width=True)

            # MCQ: show top-1 prob of the eventual label (when known)
            if mcq_variants and "outcome_label" in dd.columns and dd["outcome_label"].notna().any():
                mc = st.selectbox("MCQ variant", [c[len(MCQ_PREFIX):] for c in mcq_variants])
                col = MCQ_PREFIX + mc
                probs = parse_mcq_json(dd[col])
                truth = dd["outcome_label"].fillna("").astype(str)
                top_true_prob = []
                for t, pr in zip(truth, probs):
                    if not t or pr is None:
                        top_true_prob.append(np.nan)
                    else:
                        top_true_prob.append(pr.get(t, 0.0))
                dd["_top_true_prob"] = top_true_prob
                if "created_at" in dd.columns:
                    dd = dd.sort_values("created_at")
                figm = px.line(dd, x="created_at" if "created_at" in dd.columns else dd.index, y="_top_true_prob",
                               markers=True, title="Probability assigned to eventual true label")
                figm.update_layout(height=300)
                st.plotly_chart(figm, use_container_width=True)

            st.markdown("**All rows for this question**")
            st.dataframe(dd, use_container_width=True)
    else:
        st.info("No question id column found (expected 'question_id' → 'qid').")

# --- Tab: Costs/Tokens ---
with tab_costs:
    st.subheader("Cost & tokens")

    # Cost columns from your schema
    has_total = "cost_usd_total" in df.columns and df["cost_usd_total"].notna().any()
    has_research = "cost_usd_research" in df.columns and df["cost_usd_research"].notna().any()
    has_classifier = "cost_usd_classifier" in df.columns and df["cost_usd_classifier"].notna().any()

    if has_total or has_research or has_classifier:
        cols = []
        if has_total: cols.append("cost_usd_total")
        if has_research: cols.append("cost_usd_research")
        if has_classifier: cols.append("cost_usd_classifier")

        melted = df[cols].melt(var_name="cost_type", value_name="usd").dropna()
        melted["cost_type"] = melted["cost_type"].map({
            "cost_usd_total": "Total",
            "cost_usd_research": "Research",
            "cost_usd_classifier": "Classifier",
        }).fillna(melted["cost_type"])

        fig = px.box(melted, x="cost_type", y="usd", points="all", title="Cost distribution (USD)")
        st.plotly_chart(fig, use_container_width=True)

        totals = melted.groupby("cost_type")["usd"].sum().reset_index().rename(columns={"usd": "sum_usd"})
        st.dataframe(totals, use_container_width=True)
    else:
        st.info("No cost columns found (expected any of cost_usd__total / __research / __classifier).")

    st.caption("Token columns were not in your parquet; this tab shows cost distributions only for now.")

# --- Tab: Run Stability (variation across run_id for a question) ---
with tab_runs:
    st.subheader("Stability across runs/commits")
    # Use the currently selected binary permutation if available
    if "run_id" in df.columns and "qid" in df.columns and (p_col is not None):
        g = (
            df.dropna(subset=[p_col])
              .groupby(["qid", "run_id"])[p_col]
              .mean()
              .reset_index()
        )
        spread = g.groupby("qid")[p_col].agg(["count", "std", "min", "max"]).reset_index()
        st.dataframe(spread.sort_values("std", ascending=False), use_container_width=True)
        st.caption("Lower std suggests more stable forecasts per question across runs.")
    else:
        st.info("Need run_id, qid, and a chosen permutation to compute cross-run stability.")

# --- Tab: Downloads (CSV/Parquet + human logs ZIP) ---
with tab_downloads:
    st.subheader("Download data")
    # 1) Current dataframe as CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download current view as CSV", data=csv_bytes, file_name="spagbot_forecasts_view.csv", mime="text/csv")

    # 2) Current dataframe as Parquet
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        st.download_button("Download current view as Parquet", data=buf.getvalue(), file_name="spagbot_forecasts_view.parquet", mime="application/octet-stream")
    except Exception:
        st.info("Parquet export unavailable (missing pyarrow). Install pyarrow to enable Parquet downloads.")

    # 3) Offer the raw repo parquet too, if present
    raw_parquet_path = PARQUET_PATH if PARQUET_PATH.exists() else None
    if raw_parquet_path:
        raw_bytes = raw_parquet_path.read_bytes()
        st.download_button("Download repo parquet (as-is)", data=raw_bytes, file_name="forecasts.parquet", mime="application/octet-stream")

    # 4) Human logs ZIP (from logs/ and forecast_logs/) if present
    st.markdown("### Human logs ZIP")
    repo_root = APP_DIR.parent
    candidate_dirs = [repo_root / "logs", repo_root / "forecast_logs"]
    found_any = any(d.exists() and any(d.iterdir()) for d in candidate_dirs)

    if found_any:
        bufzip = io.BytesIO()
        with zipfile.ZipFile(bufzip, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in candidate_dirs:
                if d.exists():
                    for p in d.rglob("*"):
                        if p.is_file():
                            arcname = p.relative_to(repo_root).as_posix()
                            zf.write(p, arcname)
        st.download_button(
            "Download human logs (logs/ & forecast_logs/) as ZIP",
            data=bufzip.getvalue(),
            file_name="spagbot_human_logs.zip",
            mime="application/zip"
        )
    else:
        st.info("No files found under logs/ or forecast_logs/ to package.")

# -----------------------------------------------------------------------------
# Footer diagnostics
# -----------------------------------------------------------------------------
st.divider()
st.markdown("**Data source**")
st.code(
    f"USE_PARQUET={USE_PARQUET} | PARQUET_PATH={PARQUET_PATH} | RAW_CSV_URL={RAW_CSV_URL}",
    language="text",
)

with st.expander("Detected columns / quick debug"):
    st.write("First 20 columns:", list(df.columns)[:20])
    st.write("Binary permutations:", [c for c in df.columns if c.startswith('binary_prob__')][:20])
    st.write("Numeric variants:", {k: list(v.keys()) for k, v in numeric_variants.items()})
    st.write("MCQ variants:", mcq_variants)