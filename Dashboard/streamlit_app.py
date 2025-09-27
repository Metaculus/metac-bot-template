# Dashboard/streamlit_app.py
# =============================================================================
# Spagbot Performance Dashboard (resilient + cache-clear)
# - Robust to missing qid/qtitle (derives from question_url when needed)
# - Keeps data fresh: parquet vs CSV freshness checks + optional local CSV
# - Data-source banner (path/URL, modified time, row count)
# - Sidebar toggles: Clear cache, Prefer CSV if newer, Date filter
# - Binary, numeric (pinball/coverage/sharpness), MCQ, costs, runs, downloads
# - Uses use_container_width=True (broad Streamlit compatibility)
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# --- load Streamlit secrets -> environment (for persistent local/cloud config)
try:
    if "env" in st.secrets:
        for k, v in st.secrets["env"].items():
            os.environ.setdefault(k, str(v))
except Exception:
    pass

# -----------------------------------------------------------------------------
# App paths / Config
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent  # assumes Dashboard/ at repo root
USE_PARQUET = os.getenv("USE_PARQUET", "true").lower() == "true"
PARQUET_PATH = REPO_ROOT / "Dashboard" / "data" / "forecasts.parquet"

# Raw CSV fallback (GitHub). Override if your repo path differs or is private.
RAW_CSV_URL = os.getenv(
    "SPAGBOT_RAW_CSV_URL",
    "https://raw.githubusercontent.com/kwyjad/Spagbot_metac-bot/main/forecast_logs/forecasts.csv",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # only needed if repo is private

# Optional explicit local CSV path override (e.g., SPAGBOT_LOCAL_CSV_PATH=forecasts.csv)
LOCAL_CSV_OVERRIDE = os.getenv("SPAGBOT_LOCAL_CSV_PATH", "").strip()

# -----------------------------------------------------------------------------
# Column mapping + labels
# -----------------------------------------------------------------------------
COLUMN_ALIAS_MAP: Dict[str, List[str]] = {
    "qid": ["question_id", "qid", "metaculus_qid"],
    "qtitle": ["question_title", "title", "name"],
    "qtype": ["question_type", "qtype", "type"],
    "created_at": ["run_time_iso", "created_time_iso", "created_at"],
    "closes_at": ["closes_time_iso"],
    "resolves_at": ["resolves_time_iso"],
    "run_id": ["run_id"],
    "purpose": ["purpose"],
    "git_sha": ["git_sha"],
    "tournament_id": ["tournament_id"],
    "outcome_label": ["resolved_outcome_label"],
    "resolved_flag": ["resolved"],
    "resolved_value": ["resolved_value"],
    "cost_usd_total": ["cost_usd__total", "cost_usd_total", "cost_usd"],
    "cost_usd_research": ["cost_usd__research"],
    "cost_usd_classifier": ["cost_usd__classifier"],
}

# Labels for binary permutations (extend as needed)
BINARY_PERM_LABELS: Dict[str, str] = {
    # "binary_prob__ensemble": "Ensemble",
    # "binary_prob__bayes": "Bayes",
}

# Family prefixes
NUMERIC_PREFIXES = [
    "numeric_p10__", "numeric_p50__", "numeric_p90__",
    "numeric_mu__", "numeric_sigma__"
]
MCQ_PREFIX = "mcq_prob__"   # for MCQ detection

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _http_get(url: str, token: str = "") -> bytes:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.content

def _first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _parse_qid_from_url(url: str) -> Optional[int]:
    # e.g., https://www.metaculus.com/questions/39562/some-slug/
    m = re.search(r"/questions/(\d+)/", url)
    return int(m.group(1)) if m else None

def _title_from_slug(url: str) -> Optional[str]:
    m = re.search(r"/questions/\d+/([^/]+)/?", url)
    return m.group(1).replace("-", " ").title() if m else None

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Map aliases to standardized names
    for std, options in COLUMN_ALIAS_MAP.items():
        name = _first_present(out, options)
        if name and std not in out.columns:
            out[std] = out[name]

    # Datetimes
    for tcol in ["created_at", "closes_at", "resolves_at"]:
        if tcol in out.columns:
            out[tcol] = pd.to_datetime(out[tcol], errors="coerce")

    # Derive outcome (binary 0/1) from label if present
    if "outcome_label" in out.columns:
        lab = out["outcome_label"].astype(str).str.lower()
        out["outcome"] = np.where(lab.eq("yes"), 1.0,
                           np.where(lab.eq("no"), 0.0, np.nan))

    # Force qid numeric so drilldown never misses rows on dtype mismatch
    if "qid" in out.columns:
        out["qid"] = pd.to_numeric(out["qid"], errors="coerce").astype("Int64")

    # Derive qid/qtitle from question_url when missing
    if "qid" not in out.columns and "question_url" in out.columns:
        out["qid"] = [_parse_qid_from_url(str(u)) for u in out["question_url"].fillna("")]
        if out["qid"].isna().all():
            out.drop(columns=["qid"], errors="ignore", inplace=True)
        else:
            out["qid"] = pd.to_numeric(out["qid"], errors="coerce").astype("Int64")

    if "qtitle" not in out.columns:
        name = _first_present(out, ["question_title", "title", "name"])
        if name:
            out["qtitle"] = out[name]
        elif "question_url" in out.columns:
            t = [_title_from_slug(str(u)) for u in out["question_url"].fillna("")]
            if any(bool(x) for x in t):
                out["qtitle"] = t

    # Ensure numeric types for forecast-like columns (avoids blank plots)
    forecast_like_cols = [
        c for c in out.columns
        if c.startswith("binary_prob__")
        or any(c.startswith(pref) for pref in NUMERIC_PREFIXES)
    ]
    for c in forecast_like_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def _load_parquet(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_parquet(path)
    meta = {
        "source": "parquet",
        "path": str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows": str(len(df)),
    }
    return _auto_map_columns(df), meta

def _load_csv_url(url: str, token: str = "") -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = _http_get(url, token)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    meta = {
        "source": "csv-url",
        "path": url,
        "mtime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows": str(len(df)),
    }
    return _auto_map_columns(df), meta

def _load_csv_local(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    meta = {
        "source": "csv-local",
        "path": str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows": str(len(df)),
    }
    return _auto_map_columns(df), meta

def _csv_seems_newer_than_parquet(pq_meta_rows: int) -> Tuple[bool, Optional[Tuple[pd.DataFrame, Dict[str, str]]]]:
    """
    Heuristic freshness: if any available CSV (local or URL) has strictly more rows than parquet,
    return (True, (df, meta)) for the freshest CSV to use immediately.
    """
    candidates: List[Tuple[pd.DataFrame, Dict[str, str]]] = []
    # Local override path
    if LOCAL_CSV_OVERRIDE:
        p = (REPO_ROOT / LOCAL_CSV_OVERRIDE).resolve()
        if p.exists():
            try:
                candidates.append(_load_csv_local(p))
            except Exception:
                pass
    # Standard local locations inside repo
    for rel in ["forecasts.csv", "forecast_logs/forecasts.csv"]:
        p = (REPO_ROOT / rel).resolve()
        if p.exists():
            try:
                candidates.append(_load_csv_local(p))
            except Exception:
                pass
    # URL fallback
    if RAW_CSV_URL:
        try:
            candidates.append(_load_csv_url(RAW_CSV_URL, GITHUB_TOKEN))
        except Exception:
            pass

    if not candidates:
        return False, None

    # Pick the CSV with the most rows
    best = max(candidates, key=lambda tup: len(tup[0]))
    newer = len(best[0]) > pq_meta_rows
    return newer, (best[0], best[1]) if newer else (False, None)

def load_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    prefer_csv = st.sidebar.checkbox("Prefer CSV if it’s newer than parquet", value=False)

    # 1) Parquet path present?
    if USE_PARQUET and PARQUET_PATH.exists():
        df_pq, meta_pq = _load_parquet(PARQUET_PATH)
        if prefer_csv:
            try:
                pq_rows = int(meta_pq.get("rows", "0"))
            except Exception:
                pq_rows = len(df_pq)
            newer, best_csv = _csv_seems_newer_than_parquet(pq_rows)
            if newer and best_csv:
                return best_csv
        return df_pq, meta_pq

    # 2) No parquet — try local CSVs, then URL
    if LOCAL_CSV_OVERRIDE:
        p = (REPO_ROOT / LOCAL_CSV_OVERRIDE).resolve()
        if p.exists():
            try: return _load_csv_local(p)
            except Exception as e: st.error(f"Failed reading local CSV at {p}: {e}")

    for rel in ["forecasts.csv", "forecast_logs/forecasts.csv"]:
        p = (REPO_ROOT / rel).resolve()
        if p.exists():
            try: return _load_csv_local(p)
            except Exception as e: st.error(f"Failed reading local CSV at {p}: {e}")

    if RAW_CSV_URL:
        try:
            return _load_csv_url(RAW_CSV_URL, GITHUB_TOKEN)
        except Exception as e:
            st.error(f"Failed to fetch CSV from RAW_CSV_URL.\n{e}")

    st.error(
        "No data source available.\n"
        "Place a parquet at 'Dashboard/data/forecasts.parquet' (and set USE_PARQUET=true), "
        "or provide a CSV locally (forecasts.csv / forecast_logs/forecasts.csv), "
        "or set SPAGBOT_RAW_CSV_URL to a raw CSV URL."
    )
    return pd.DataFrame(), {"source": "none", "path": "—", "mtime": "—", "rows": "0"}

# -----------------------------------------------------------------------------
# Human logs helper
# -----------------------------------------------------------------------------
def _find_human_logs_for_question(dfq: pd.DataFrame, qid: Optional[int]) -> List[Path]:
    """
    Look for .md reasoning logs related to this question under forecast_logs/ and logs/.
    Heuristics:
      - Filename contains the QID (e.g., ...Q39562..., ...qid_39562...)
      - If run_id is present, also look for files containing the run_id
    Returns most recent first.
    """
    roots = [REPO_ROOT / "forecast_logs", REPO_ROOT / "logs"]
    patterns = []
    if qid is not None:
        patterns.append(str(qid))
        patterns.append(f"Q{qid}")
        patterns.append(f"qid_{qid}")
    # add run_id hints if available
    if "run_id" in dfq.columns and dfq["run_id"].notna().any():
        run_ids = sorted(set(str(x) for x in dfq["run_id"].dropna().astype(str)))
        patterns += run_ids

    found: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            name = p.name.lower()
            if any(s.lower() in name for s in patterns) or (qid is None and p.is_file()):
                found.append(p)

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Spagbot Performance", layout="wide")
st.title("Spagbot Performance Dashboard")

# Cache clear
if st.sidebar.button("Clear cache & reload", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Data load
df, meta = load_data()

# Source banner
st.info(
    f"**Source:** {meta.get('source','?')}  |  **Path/URL:** {meta.get('path','?')}  |  "
    f"**Modified:** {meta.get('mtime','?')}  |  **Rows:** {meta.get('rows','?')}"
)

# Normalize created_at
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Optional date filter (checkbox-controlled)
date_filter_on = st.sidebar.checkbox("Enable date filter", value=False)
if date_filter_on and "created_at" in df.columns and df["created_at"].notna().any():
    d1 = pd.to_datetime(df["created_at"].min()).date()
    d2 = pd.to_datetime(df["created_at"].max()).date()
    r1, r2 = st.sidebar.date_input("Date range", value=(d1, d2))
    mask = (df["created_at"].dt.date >= r1) & (df["created_at"].dt.date <= r2)
    df = df[mask]

# Detect permutations
binary_perm_cols = [c for c in df.columns if c.startswith("binary_prob__")]

numeric_variants: Dict[str, Dict[str, str]] = {}
for pref in NUMERIC_PREFIXES:
    for c in df.columns:
        if c.startswith(pref):
            suffix = c[len(pref):]
            numeric_variants.setdefault(suffix, {})
            numeric_variants[suffix][pref] = c
numeric_variants = {k: v for k, v in numeric_variants.items() if ("numeric_p10__" in v and "numeric_p90__" in v)}

mcq_variants = [c for c in df.columns if c.startswith(MCQ_PREFIX)]
perm_labels = {c: (BINARY_PERM_LABELS.get(c) or c.replace("binary_prob__", "").replace("_", " ")) for c in binary_perm_cols}

has_outcome = "outcome" in df.columns and df["outcome"].notna().any()

# --- Key metrics
st.markdown("## Key metrics")
default_perm = "binary_prob__ensemble" if "binary_prob__ensemble" in perm_labels else (binary_perm_cols[0] if binary_perm_cols else None)
p_col = st.selectbox(
    "Headline permutation",
    options=([default_perm] + [c for c in binary_perm_cols if c != default_perm]) if default_perm else binary_perm_cols,
    format_func=lambda c: perm_labels.get(c, c),
) if binary_perm_cols else None

c1, c2, c3, c4, c5, c6 = st.columns(6)
def safe_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)
def brier_score(p: pd.Series, y: pd.Series) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    return float(np.mean((p[mask] - y[mask]) ** 2)) if mask.sum() else None
def log_loss(p: pd.Series, y: pd.Series, eps: float = 1e-15) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    if not mask.sum():
        return None
    pv = np.clip(p[mask].values, eps, 1 - eps)
    yv = y[mask].values
    return float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
def hit_rate(p: pd.Series, y: pd.Series, threshold: float = 0.5) -> Optional[float]:
    p, y = safe_float_series(p), safe_float_series(y)
    mask = p.notna() & y.notna()
    return float(((p[mask] >= threshold).astype(int).values == y[mask].values).mean()) if mask.sum() else None
def sharpness(p: pd.Series) -> Optional[float]:
    p = safe_float_series(p)
    return float(np.mean(np.abs(p - 0.5))) if p.notna().sum() else None

total_rows = len(df)
sel_non_null = int(pd.to_numeric(df[p_col], errors="coerce").notna().sum()) if p_col else total_rows
with c1: st.metric("Brier (binary)", f"{brier_score(df[p_col], df['outcome']):.3f}" if has_outcome and p_col else "—")
with c2: st.metric("Log loss", f"{log_loss(df[p_col], df['outcome']):.3f}" if has_outcome and p_col else "—")
with c3: st.metric("Hit rate @0.5", f"{100*hit_rate(df[p_col], df['outcome']):.1f}%" if has_outcome and p_col else "—")
with c4: st.metric("Sharpness (|p-0.5|)", f"{sharpness(df[p_col]):.3f}" if p_col in df.columns else "—")
with c5: st.metric("forecasts", f"{sel_non_null} / {total_rows}")
with c6:
    last_ts = None
    for c in ["created_at", "resolves_at", "closes_at"]:
        if c in df.columns and df[c].notna().any():
            last_ts = pd.to_datetime(df[c].max()); break
    st.metric("Last update", last_ts.strftime("%Y-%m-%d %H:%M") if last_ts is not None else "—")

st.divider()

tab_cal, tab_compare, tab_numeric, tab_mcq, tab_questions, tab_costs, tab_runs, tab_downloads = st.tabs(
    ["Calibration & Sharpness", "Compare permutations (binary)", "Numeric permutations", "MCQ permutations",
     "Per-Question drilldown", "Costs/Tokens", "Run Stability", "Downloads"]
)

# Calibration ------------------------------------------------------------------
with tab_cal:
    st.subheader("Reliability (Calibration) curve")
    if p_col is not None and has_outcome:
        bins = np.linspace(0, 1, 11)
        p = pd.to_numeric(df[p_col], errors="coerce"); y = pd.to_numeric(df["outcome"], errors="coerce")
        mask = p.notna() & y.notna()
        if mask.sum() > 0:
            idx = np.digitize(p[mask], bins) - 1
            rows = []
            for i in range(10):
                sel = idx == i
                rows.append({
                    "bin_mid": 0.05 + 0.1 * i,
                    "pred_mean": float(p[mask][sel].mean()) if sel.sum() else np.nan,
                    "obs_rate": float(y[mask][sel].mean()) if sel.sum() else np.nan,
                    "count": int(sel.sum()),
                })
            bdf = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bdf["bin_mid"], y=bdf["obs_rate"], mode="lines+markers", name="Observed",
                                     hovertext=[f"n={c}" for c in bdf["count"]]))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="Forecast probability", yaxis_title="Observed frequency", height=430)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resolved rows yet for calibration.")
    else:
        st.info("Need binary outcomes and a forecast column to show calibration.")

    st.subheader("Forecast distribution (sharpness)")
    if p_col is not None:
        st.plotly_chart(px.histogram(df.dropna(subset=[p_col]), x=p_col, nbins=20, height=320), use_container_width=True)
    else:
        st.info("No forecast column detected.")

# Compare permutations (binary) ------------------------------------------------
with tab_compare:
    st.subheader("Headline metrics by permutation (binary)")
    if not binary_perm_cols:
        st.info("No binary forecast columns (binary_prob__*) found.")
    else:
        rows = []
        for col in binary_perm_cols:
            row = {"Permutation": perm_labels.get(col, col)}
            p = pd.to_numeric(df[col], errors="coerce")
            row["Brier"] = brier_score(p, df["outcome"]) if has_outcome else None
            row["LogLoss"] = log_loss(p, df["outcome"]) if has_outcome else None
            row["Hit@0.5"] = hit_rate(p, df["outcome"]) if has_outcome else None
            row["Sharpness|p-0.5|"] = sharpness(p)
            row["#Forecasts"] = int(p.notna().sum())
            rows.append(row)
        mdf = pd.DataFrame(rows)
        if has_outcome and mdf["Brier"].notna().any():
            mdf = mdf.sort_values("Brier", na_position="last")
        st.dataframe(mdf, use_container_width=True)

# Numeric permutations ---------------------------------------------------------
def pinball_loss(y_true: np.ndarray, qhat: np.ndarray, q: float) -> float:
    diff = y_true - qhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def numeric_metrics_from_quantiles(y: pd.Series, p10: pd.Series, p50: pd.Series, p90: pd.Series) -> Dict[str, Optional[float]]:
    yv = safe_float_series(y).values
    q10 = safe_float_series(p10).values
    q50 = safe_float_series(p50).values
    q90 = safe_float_series(p90).values
    mask = np.isfinite(yv) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
    if mask.sum() == 0:
        return {"pinball_avg": None, "coverage80": None, "interval_width": None, "n": 0}
    yv, q10, q50, q90 = yv[mask], q10[mask], q50[mask], q90[mask]
    q10, q90 = np.minimum(q10, q90), np.maximum(q10, q90)
    q50 = np.clip(q50, q10, q90)
    l10 = pinball_loss(yv, q10, 0.1)
    l50 = pinball_loss(yv, q50, 0.5)
    l90 = pinball_loss(yv, q90, 0.9)
    coverage = float(np.mean((yv >= q10) & (yv <= q90)))
    width = float(np.mean(q90 - q10))
    return {"pinball_avg": (l10 + l50 + l90) / 3.0, "coverage80": coverage, "interval_width": width, "n": int(mask.sum())}

with tab_numeric:
    st.subheader("Numeric permutations (q10/q50/q90)")
    if not numeric_variants:
        st.info("No numeric permutations found.")
    elif "resolved_value" not in df.columns or df["resolved_value"].notna().sum() == 0:
        st.info("No numeric outcomes present (resolved_value).")
    else:
        rows = []
        for variant, cols in numeric_variants.items():
            p10_col = cols.get("numeric_p10__"); p50_col = cols.get("numeric_p50__"); p90_col = cols.get("numeric_p90__")
            if p50_col is None and p10_col and p90_col:
                df["_mid_tmp_"] = (pd.to_numeric(df[p10_col], errors="coerce") + pd.to_numeric(df[p90_col], errors="coerce")) / 2.0
                p50_col = "_mid_tmp_"
            metrics = numeric_metrics_from_quantiles(df["resolved_value"], df[p10_col], df[p50_col], df[p90_col])
            rows.append({
                "Variant": variant,
                "Pinball avg (0.1/0.5/0.9)": metrics["pinball_avg"],
                "Coverage 80%": metrics["coverage80"],
                "Interval width": metrics["interval_width"],
                "#Resolved": metrics["n"],
            })
        ndf = pd.DataFrame(rows)
        if ndf["Pinball avg (0.1/0.5/0.9)"].notna().any():
            ndf = ndf.sort_values("Pinball avg (0.1/0.5/0.9)", na_position="last")
        st.dataframe(ndf, use_container_width=True)

# MCQ permutations -------------------------------------------------------------
def parse_mcq_json(col: pd.Series) -> List[Optional[Dict[str, float]]]:
    out: List[Optional[Dict[str, float]]] = []
    for x in col.fillna("").astype(str):
        x = x.strip()
        if not x or x.lower() == "none" or x == "nan":
            out.append(None); continue
        try:
            d = json.loads(x)
            if not isinstance(d, dict):
                out.append(None); continue
            total = float(sum(float(v) for v in d.values()))
            if total > 0:
                d = {k: float(v) / total for k, v in d.items()}
            out.append(d)
        except Exception:
            out.append(None)
    return out

def mcq_metrics(y_label: pd.Series, probs_list: List[Optional[Dict[str, float]]]) -> Dict[str, Optional[float]]:
    y = y_label.fillna("").astype(str).str.strip()
    ce_vals, brier_vals, top1_vals = [], [], []
    n = 0
    for truth, probs in zip(y, probs_list):
        if not truth or probs is None:
            continue
        p = float(probs.get(truth, 0.0))
        eps = 1e-15
        ce_vals.append(-np.log(max(p, eps)))
        b = (1 - p) ** 2 + sum((pv - 0.0) ** 2 for k, pv in probs.items() if k != truth)
        brier_vals.append(b)
        top_label = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        top1_vals.append(1.0 if top_label == truth else 0.0)
        n += 1
    if n == 0:
        return {"cross_entropy": None, "brier_mc": None, "top1_acc": None, "n": 0}
    return {"cross_entropy": float(np.mean(ce_vals)), "brier_mc": float(np.mean(brier_vals)), "top1_acc": float(np.mean(top1_vals)), "n": n}

with tab_mcq:
    st.subheader("MCQ permutations")
    if not mcq_variants:
        st.info("No MCQ permutations found.")
    elif "outcome_label" not in df.columns or df["outcome_label"].notna().sum() == 0:
        st.info("No resolved outcome labels (needed for MCQ evaluation).")
    else:
        rows = []
        for col in mcq_variants:
            probs_list = parse_mcq_json(df[col])
            m = mcq_metrics(df["outcome_label"], probs_list)
            rows.append({
                "Variant": col[len(MCQ_PREFIX):],
                "Cross-entropy": m["cross_entropy"],
                "Brier (MCQ)": m["brier_mc"],
                "Top-1 acc": m["top1_acc"],
                "#Resolved": m["n"],
            })
        mdf = pd.DataFrame(rows)
        if mdf["Cross-entropy"].notna().any():
            mdf = mdf.sort_values("Cross-entropy", na_position="last")
        st.dataframe(mdf, use_container_width=True)

# Per-Question drilldown (robust to missing qtitle) ---------------------------
with tab_questions:
    st.subheader("Questions")

    # Try to ensure qid/qtitle exist from question_url when missing
    if "qid" not in df.columns and "question_url" in df.columns and df["question_url"].notna().any():
        derived = df["question_url"].fillna("").astype(str).map(_parse_qid_from_url)
        if derived.notna().any():
            df["qid"] = derived
    if "qtitle" not in df.columns and "question_url" in df.columns and df["question_url"].notna().any():
        derived_t = df["question_url"].fillna("").astype(str).map(_title_from_slug)
        if derived_t.notna().any():
            df["qtitle"] = derived_t

    has_qid = "qid" in df.columns and df["qid"].notna().any()

    if has_qid:
        # Build per-question recency (use max created_at where available)
        df_tmp = df[df["qid"].notna()].copy()
        if "created_at" in df_tmp.columns:
            # max created_at per qid
            recency = df_tmp.groupby("qid")["created_at"].max().reset_index().rename(columns={"created_at":"_latest_ts"})
            opts = (
                df_tmp[["qid", "qtitle"]].drop_duplicates()
                .merge(recency, on="qid", how="left")
                .sort_values(["_latest_ts", "qid"], ascending=[False, False])
            )
        else:
            opts = (
                df_tmp[["qid", "qtitle"]].drop_duplicates()
                .sort_values("qid", ascending=False)   # fallback: newest-ish by larger qid
            )

        # Human-friendly display labels
        if "qtitle" in opts.columns and opts["qtitle"].notna().any():
            display = [
                f"Q{int(q)} – {str(t)[:80]}"
                for q, t in zip(opts["qid"], opts["qtitle"].fillna(""))
            ]
        else:
            display = [f"Q{int(q)}" for q in opts["qid"]]

        chosen = st.selectbox("Pick a question:", display if len(display) else ["(no questions found)"])
        if display:
            import re as _re
            m = _re.search(r"Q(\d+)", chosen)
            qid = int(m.group(1)) if m else None

            # Subset rows for this question
            dd = df[df["qid"] == qid].copy()

            # Title derived from the subset itself (avoids mismatch df vs dd)
            has_qtitle_dd = "qtitle" in dd.columns and dd["qtitle"].notna().any()
            title = dd["qtitle"].dropna().iloc[0] if has_qtitle_dd else "(no title)"
            st.write(f"### Q{qid} — {title}")

            # Prefer binary plot if this question actually has non-null binary series
            available_binaries = []
            if "created_at" in dd.columns:
                for c in binary_perm_cols:
                    if c in dd.columns and pd.to_numeric(dd[c], errors="coerce").notna().any():
                        available_binaries.append(c)

            if available_binaries:
                chosen_perm = st.selectbox(
                    "Binary permutation to plot",
                    available_binaries,
                    format_func=lambda c: perm_labels.get(c, c)
                )
                dd = dd.sort_values("created_at")
                yvals = pd.to_numeric(dd[chosen_perm], errors="coerce")
                mask = yvals.notna() & dd["created_at"].notna()
                fig = px.line(
                    dd.loc[mask],
                    x="created_at",
                    y=yvals.loc[mask],
                    markers=True,
                    title="Forecast history (binary)"
                )
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # --- FIX: robust numeric presence check (no TypeError) ---
                num_pfx = ("numeric_p10__", "numeric_p50__", "numeric_p90__")
                has_numeric = any(c.startswith(num_pfx) for c in dd.columns)

                if has_numeric and "created_at" in dd.columns:
                    # group available variants for which we have any numeric data
                    variants = {}
                    for c in dd.columns:
                        for pref in num_pfx:
                            if c.startswith(pref):
                                var = c[len(pref):]
                                variants.setdefault(var, {})[pref] = c
                    # choose 'ensemble' if present, else any variant with data
                    preferred = "ensemble" if "ensemble" in variants else (next(iter(variants.keys())) if variants else None)
                    if preferred:
                        cols = variants.get(preferred, {})
                        p10 = pd.to_numeric(dd.get(cols.get("numeric_p10__")), errors="coerce") if "numeric_p10__" in cols else None
                        p50 = pd.to_numeric(dd.get(cols.get("numeric_p50__")), errors="coerce") if "numeric_p50__" in cols else None
                        p90 = pd.to_numeric(dd.get(cols.get("numeric_p90__")), errors="coerce") if "numeric_p90__" in cols else None

                        idx = dd["created_at"].notna()
                        if p50 is not None and p50.notna().any():
                            fign = go.Figure()
                            if p10 is not None and p90 is not None and p10.notna().any() and p90.notna().any():
                                fign.add_traces([
                                    go.Scatter(x=dd.loc[idx, "created_at"], y=p10.loc[idx], name="q10", mode="lines+markers"),
                                    go.Scatter(x=dd.loc[idx, "created_at"], y=p50.loc[idx], name="q50 (median)", mode="lines+markers"),
                                    go.Scatter(x=dd.loc[idx, "created_at"], y=p90.loc[idx], name="q90", mode="lines+markers"),
                                ])
                            else:
                                fign.add_trace(go.Scatter(x=dd.loc[idx, "created_at"], y=p50.loc[idx], name="q50 (median)", mode="lines+markers"))
                            fign.update_layout(title=f"Numeric forecast history ({preferred})", height=360)
                            st.plotly_chart(fign, use_container_width=True)
                        else:
                            st.info("No numeric forecasts found for this question.")
                    else:
                        st.info("No numeric variants detected for this question.")
                else:
                    st.info("This question has no binary or numeric series to plot yet.")

            # Full table for this question
            st.markdown("**All rows for this question**")
            st.dataframe(dd, use_container_width=True)

            # --- Human logs (model reasoning) ---
            st.markdown("**Human logs (model reasoning)**")
            try:
                qid_int = int(qid) if qid is not None else None
            except Exception:
                qid_int = None

            log_paths = _find_human_logs_for_question(dd, qid_int)

            if not log_paths:
                st.info("No .md human logs found for this question in forecast_logs/ or logs/.")
            else:
                for p in log_paths[:8]:
                    with st.expander(p.name, expanded=False):
                        try:
                            st.markdown(p.read_text(encoding="utf-8", errors="ignore"))
                        except Exception as e:
                            st.warning(f"Could not read {p.name}: {e}")

    else:
        st.info("Couldn’t find a question identifier. Expected `question_id` (→ qid) or a parsable `question_url`.")

# Costs -----------------------------------------------------------------------
with tab_costs:
    st.subheader("Cost & tokens")
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
            "cost_usd_total": "Total", "cost_usd_research": "Research", "cost_usd_classifier": "Classifier"
        }).fillna(melted["cost_type"])
        st.plotly_chart(px.box(melted, x="cost_type", y="usd", points="all", title="Cost distribution (USD)"), use_container_width=True)
        st.dataframe(melted.groupby("cost_type")["usd"].sum().reset_index().rename(columns={"usd":"sum_usd"}), use_container_width=True)
    else:
        st.info("No cost columns found (expected any of cost_usd__total / __research / __classifier).")
    st.caption("Token columns not present in this parquet; showing cost distributions only.")

# Run stability ---------------------------------------------------------------
with tab_runs:
    st.subheader("Stability across runs/commits")
    if "run_id" in df.columns and "qid" in df.columns and p_col:
        g = df.dropna(subset=[p_col]).groupby(["qid", "run_id"])[p_col].mean().reset_index()
        spread = g.groupby("qid")[p_col].agg(["count", "std", "min", "max"]).reset_index()
        st.dataframe(spread.sort_values("std", ascending=False), use_container_width=True)
    else:
        st.info("Need run_id, qid, and a chosen permutation to compute cross-run stability.")

# Downloads -------------------------------------------------------------------
with tab_downloads:
    st.subheader("Download data")
    st.download_button("Download current view as CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="spagbot_forecasts_view.csv", mime="text/csv")
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq  # noqa
        buf = io.BytesIO(); df.to_parquet(buf, index=False)
        st.download_button("Download current view as Parquet", data=buf.getvalue(),
                           file_name="spagbot_forecasts_view.parquet", mime="application/octet-stream")
    except Exception:
        st.info("Parquet export unavailable (missing pyarrow).")

    raw_parquet_path = PARQUET_PATH if PARQUET_PATH.exists() else None
    if raw_parquet_path:
        st.download_button("Download repo parquet (as-is)", data=raw_parquet_path.read_bytes(),
                           file_name="forecasts.parquet", mime="application/octet-stream")

    # Human logs ZIP
    candidate_dirs = [REPO_ROOT / "logs", REPO_ROOT / "forecast_logs"]
    found_any = any(d.exists() and any(d.iterdir()) for d in candidate_dirs)
    if found_any:
        bufzip = io.BytesIO()
        with zipfile.ZipFile(bufzip, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in candidate_dirs:
                if d.exists():
                    for p in d.rglob("*"):
                        if p.is_file():
                            zf.write(p, p.relative_to(REPO_ROOT).as_posix())
        st.download_button("Download human logs (logs/ & forecast_logs/) as ZIP",
                           data=bufzip.getvalue(), file_name="spagbot_human_logs.zip", mime="application/zip")
    else:
        st.info("No files found under logs/ or forecast_logs/ to package.")

# Debug expander --------------------------------------------------------------
st.divider()
with st.expander("Detected columns / quick debug"):
    st.write("USE_PARQUET:", USE_PARQUET)
    st.write("PARQUET_PATH:", str(PARQUET_PATH))
    st.write("RAW_CSV_URL:", RAW_CSV_URL)
    st.write("LOCAL_CSV_OVERRIDE:", LOCAL_CSV_OVERRIDE or "—")
    st.write("First 25 columns:", list(df.columns)[:25])
    st.write("Binary permutations:", [c for c in df.columns if c.startswith('binary_prob__')])
    st.write("Numeric variants:", list(numeric_variants.keys()))
    st.write("MCQ variants:", [c for c in df.columns if c.startswith(MCQ_PREFIX)])