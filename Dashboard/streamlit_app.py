# Dashboard/streamlit_app.py
# =============================================================================
# Spagbot Performance Dashboard (resilient + cache-clear)
# - Robust to missing qid/qtitle (derives from question_url when needed)
# - Data-source banner (path/URL, modified time, row count)
# - Sidebar "Clear cache & reload" button
# - Binary, numeric (pinball/coverage/sharpness), MCQ, costs, runs, downloads
# =============================================================================

from __future__ import annotations

import io
import os
import re
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

# Prefer parquet if present; can be overridden with env
USE_PARQUET = os.getenv("USE_PARQUET", "true").lower() in {"1", "true", "yes"}
PARQUET_PATH = APP_DIR / "data" / "forecasts.parquet"

# Fallback CSV (raw GitHub). Default points to your repo; can be overridden.
RAW_CSV_URL = os.getenv(
    "SPAGBOT_RAW_CSV_URL",
    "https://raw.githubusercontent.com/kwyjad/Spagbot_metac-bot/main/forecast_logs/forecasts.csv",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # only needed if repo is private

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

NUMERIC_PREFIXES = ["numeric_p10__", "numeric_p50__", "numeric_p90__"]
MCQ_PREFIX = "mcq_json__"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def _parse_qid_from_url(url: str) -> Optional[int]:
    m = re.search(r"/questions?/(\d+)", url)
    if not m:
        m = re.search(r"[?&]q=(\d+)", url)
    try:
        return int(m.group(1)) if m else None
    except Exception:
        return None

def _title_from_slug(url: str) -> Optional[str]:
    m = re.search(r"/questions?/\d+-([a-z0-9\-]+)", url.lower())
    if not m:
        return None
    slug = m.group(1).replace("-", " ").strip()
    return slug or None

def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Map aliases
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
        lbl = out["outcome_label"].astype(str).str.strip().str.lower()
        out["outcome"] = np.where(
            lbl.isin(["yes", "true", "1"]), 1.0,
            np.where(lbl.isin(["no", "false", "0"]), 0.0, np.nan)
        )

def _find_human_logs_for_question(dd: pd.DataFrame, qid: Optional[int]) -> List[Path]:
    """
    Return markdown logs related to this question (by content scan).
    Looks under:
      - <repo>/forecast_logs/runs/*.md
      - <repo>/forecast_logs/**/*.md
      - <repo>/logs/**/*.md
    Matches if the file CONTENT (first ~64 KB) contains:
      - 'Q{qid}'          (e.g., 'Q39562')
      - '/questions/{qid}' in a URL
      - 'question_id: {qid}' (yaml-ish)
    Also tries run_id matches if filenames include a run id substring.
    Results are de-duplicated, newest-first.
    """
    if qid is None:
        return []

    repo_root = APP_DIR.parent
    search_roots = [
        repo_root / "forecast_logs" / "runs",
        repo_root / "forecast_logs",
        repo_root / "logs",
    ]

    # Collect candidate files
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        # direct .md in folder
        candidates.extend([p for p in root.glob("*.md") if p.is_file()])
        # recurse
        candidates.extend([p for p in root.rglob("*.md") if p.is_file()])

    # Optional: match by run_id substring if present in dd
    run_id_strs: List[str] = []
    if "run_id" in dd.columns and dd["run_id"].notna().any():
        run_id_strs = [str(x) for x in dd["run_id"].dropna().astype(str).unique().tolist()]

    # Precompile quick patterns
    qid_str = str(qid)
    pat_qid     = re.compile(rf"\bQ{re.escape(qid_str)}\b", re.IGNORECASE)
    pat_url_qid = re.compile(rf"/questions?/{re.escape(qid_str)}\b")
    pat_yaml_q  = re.compile(rf"\bquestion[_\- ]?id\s*:\s*{re.escape(qid_str)}\b", re.IGNORECASE)

    matched: List[Path] = []
    for p in candidates:
        try:
            # Fast path: filename contains some run_id
            name = p.name
            if any(rid in name for rid in run_id_strs):
                matched.append(p)
                continue

            # Content sniff (limit read to ~64KB for speed)
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(64 * 1024)

            if pat_qid.search(text) or pat_url_qid.search(text) or pat_yaml_q.search(text):
                matched.append(p)
                continue
        except Exception:
            # ignore unreadable files
            pass

    # De-duplicate and sort newest-first
    unique: Dict[str, Path] = {}
    for p in sorted(matched, key=lambda x: x.stat().st_mtime, reverse=True):
        unique.setdefault(p.resolve().as_posix(), p)
    return list(unique.values())


    # Numeric resolution
    if "resolved_value" in out.columns:
        out["resolved_value"] = pd.to_numeric(out["resolved_value"], errors="coerce")

    # Derive qid/qtitle from question_url when missing
    if "qid" not in out.columns and "question_url" in out.columns:
        out["qid"] = [ _parse_qid_from_url(str(u)) for u in out["question_url"].fillna("") ]
        if out["qid"].isna().all():
            out.drop(columns=["qid"], errors="ignore", inplace=True)

    if "qtitle" not in out.columns:
        name = _first_present(out, ["question_title", "title", "name"])
        if name:
            out["qtitle"] = out[name]
        elif "question_url" in out.columns:
            t = [ _title_from_slug(str(u)) for u in out["question_url"].fillna("") ]
            if any(bool(x) for x in t):
                out["qtitle"] = t

    # --- FORCE NUMERIC for forecast columns ---
    forecast_like_cols = [
        c for c in out.columns
        if c.startswith("binary_prob__")
        or any(c.startswith(pref) for pref in NUMERIC_PREFIXES)
    ]
    for c in forecast_like_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

@st.cache_data(show_spinner=True)
def load_from_parquet(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_parquet(path)
    meta = {
        "source": "parquet",
        "path": str(path),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows": str(len(df)),
    }
    return _auto_map_columns(df), meta

@st.cache_data(show_spinner=True)
def load_from_csv(url: str, token: str = "") -> Tuple[pd.DataFrame, Dict[str, str]]:
    import requests
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(url, headers=headers, timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    meta = {
        "source": "csv",
        "path": url,
        "mtime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "rows": str(len(df)),
    }
    return _auto_map_columns(df), meta

def load_data() -> Tuple[pd.DataFrame, Dict[str, str]]:
    if USE_PARQUET and PARQUET_PATH.exists():
        return load_from_parquet(PARQUET_PATH)

    # Fallback to CSV from GitHub if configured or default works
    if RAW_CSV_URL:
        try:
            return load_from_csv(RAW_CSV_URL, GITHUB_TOKEN)
        except Exception as e:
            st.error(f"Failed to fetch CSV from RAW_CSV_URL.\n{e}")

    st.error(
        "No data source available.\n"
        "Place a parquet at 'Dashboard/data/forecasts.parquet' (and set USE_PARQUET=true), "
        "or set SPAGBOT_RAW_CSV_URL to a raw CSV URL."
    )
    return pd.DataFrame(), {"source": "none", "path": "-", "mtime": "-", "rows": "0"}

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Spagbot Dashboard", layout="wide")
st.title("🔮 Spagbot Performance Dashboard")

# Sidebar controls (with Cache clear)
with st.sidebar:
    st.header("Controls")
    if st.button("🔁 Clear cache & reload", help="If the banner looks stale, click me"):
        st.cache_data.clear()
        st.rerun()
    date_filter_on = st.checkbox("Filter by created date", value=False)

df, meta = load_data()
if df.empty:
    st.stop()

# Expand perm labels to include *all* binary_prob__* columns dynamically
binary_perm_cols = [c for c in df.columns if c.startswith("binary_prob__")]
for c in binary_perm_cols:
    if c not in BINARY_PERM_LABELS:
        BINARY_PERM_LABELS[c] = c.replace("binary_prob__", "").replace("_", " ")

# Banner shows where the data came from
st.info(
    f"**Source:** {meta.get('source','?')}  |  **Path/URL:** {meta.get('path','?')}  |  "
    f"**Modified:** {meta.get('mtime','?')}  |  **Rows:** {meta.get('rows','?')}"
)


# Normalize created_at
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Optional date filter
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
    return float(( (p[mask] >= threshold).astype(int).values == y[mask].values ).mean()) if mask.sum() else None
def sharpness(p: pd.Series) -> Optional[float]:
    p = safe_float_series(p)
    return float(np.mean(np.abs(p - 0.5))) if p.notna().sum() else None

if p_col is not None:
    with c1: st.metric("Brier (binary)", f"{brier_score(df[p_col], df['outcome']):.3f}" if has_outcome else "—")
    with c2: st.metric("Log loss", f"{log_loss(df[p_col], df['outcome']):.3f}" if has_outcome else "—")
    with c3: st.metric("Hit rate @0.5", f"{100*hit_rate(df[p_col], df['outcome']):.1f}%" if has_outcome else "—")
    with c4: st.metric("Sharpness (|p-0.5|)", f"{sharpness(df[p_col]):.3f}" if p_col in df.columns else "—")
else:
    for col in (c1, c2, c3, c4): 
        with col: st.metric("—", "—")

with c5: st.metric("forecasts", f"{int(df[p_col].notna().sum()) if p_col else len(df)}")
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
        # Build selector safely (do not assume qtitle exists)
        has_qtitle_all = "qtitle" in df.columns and df["qtitle"].notna().any()
        select_cols = ["qid"] + (["qtitle"] if has_qtitle_all else [])
        opts = (
            df.loc[df["qid"].notna(), select_cols]
              .drop_duplicates()
              .sort_values("qid")
        )

        # Human-friendly display labels
        if has_qtitle_all:
            display = [f"Q{int(q)} – {str(t)[:80]}" for q, t in zip(opts["qid"], opts["qtitle"].fillna(""))]
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

            # Binary forecast history plot (numeric-safe)
            if binary_perm_cols and "created_at" in dd.columns:
                chosen_perm = st.selectbox(
                    "Binary permutation to plot",
                    binary_perm_cols,
                    format_func=lambda c: perm_labels.get(c, c)
                )
                dd = dd.sort_values("created_at")
                yvals = pd.to_numeric(dd[chosen_perm], errors="coerce")
                mask = yvals.notna() & dd["created_at"].notna()
                if mask.any():
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
                    st.info("No valid numeric values to plot for this permutation.")

            # Full table for this question
            st.markdown("**All rows for this question**")
            st.dataframe(dd, use_container_width=True)

            # --- Human logs (model reasoning) ---
            st.markdown("**Human logs (model reasoning)**")
            try:
                qid_int = int(qid) if qid is not None else None
            except Exception:
                qid_int = None

            # Requires the helper added in Utilities:
            # _find_human_logs_for_question(dd: pd.DataFrame, qid: Optional[int]) -> List[Path]
            log_paths = _find_human_logs_for_question(dd, qid_int)

            if not log_paths:
                st.info("No .md human logs found for this question in forecast_logs/ or logs/.")
            else:
                # Show up to the most recent 8 logs (adjust as desired)
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
                            zf.write(p, p.relative_to(repo_root).as_posix())
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
    st.write("First 25 columns:", list(df.columns)[:25])
    st.write("Binary permutations:", [c for c in df.columns if c.startswith('binary_prob__')])
    st.write("Numeric variants:", list(numeric_variants.keys()))
    st.write("MCQ variants:", [c for c in df.columns if c.startswith(MCQ_PREFIX)])
