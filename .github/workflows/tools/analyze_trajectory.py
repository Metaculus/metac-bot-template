#!/usr/bin/env python3
"""
analyze_trajectory.py
---------------------

Purpose:
- 1) Export per-bin calibration tables to CSV for charting (binary, MCQ, numeric/discrete),
     matching the same logic you use in update_calibration.py (deciles, ECE bins, PIT-lite).
- 2) Export simple "trajectory" features per question from forecasts.csv (how forecasts evolved
     before resolution): number of updates, time span, volatility, linear trend, last-24h change.

This script DOES NOT modify prompts or advice. It only emits CSVs under:
  data/calibration_exports/
    - binary_deciles.csv
    - binary_summary.csv
    - mcq_bins.csv
    - mcq_summary.csv
    - numeric_pit_bins.csv
    - numeric_summary.csv
    - trajectories.csv

You can open these directly in Excel/Google Sheets for plotting reliability, ECE, or coverage charts.

Assumptions / Inputs:
- We assume you already log forecasts to forecasts.csv (you said this is operational).
- We assume you already have a way to know question resolution/outcome and resolution time.
  If your update_calibration.py already reads a resolutions CSV or pulls via API, you can point this
  script at the SAME source. By default below we assume a local 'data/resolutions.csv'.

If your column names differ, EDIT the “EDIT THESE IF NEEDED” block below.

Author: Spagbot helper, Aug 2025.
"""
from __future__ import annotations
import os
import math
import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import numpy as np

# -------------------------
# EDIT THESE IF NEEDED
# -------------------------
# Path to your main forecast log
FORECASTS_CSV = "forecasts.csv"

# Path to resolutions file (question_id, question_type, resolved, resolution_time_utc, outcome columns).
# If your update_calibration.py already emits a canonical resolutions CSV, point to that here.
RESOLUTIONS_CSV = "data/resolutions.csv"

# Expected columns in forecasts.csv:
COL_QUESTION_ID   = "question_id"
COL_QUESTION_TYPE = "question_type"   # values like "binary", "mcq", "numeric" / "discrete"
COL_CREATED_AT    = "created_at_utc"  # ISO8601 or "YYYY-MM-DD HH:MM:SS" in UTC
COL_USER_PROB     = "probability"     # for binary: single prob in [0,1]
COL_PROBS_VECTOR  = "probs"           # for MCQ: JSON-like list string "[0.1,0.2,...]"
COL_NUMERIC_P10   = "p10"             # numeric/discrete predictive p10
COL_NUMERIC_P50   = "p50"
COL_NUMERIC_P90   = "p90"

# In resolutions.csv:
R_QUESTION_ID     = "question_id"
R_TYPE            = "question_type"
R_RESOLVED        = "resolved"            # True/False
R_RESOLUTION_TIME = "resolution_time_utc" # ISO8601
R_OUTCOME_BIN     = "outcome_binary"      # 0/1 for binary
R_OUTCOME_MCQ     = "outcome_index"       # integer 0..K-1 for MCQ
R_OUTCOME_NUMERIC = "outcome_value"       # numeric/discrete resolved value (if available)

# Number of bins
BINARY_DECILES = 10
MCQ_BINS = 10  # ECE bins on top-1 probability
# -------------------------


def _parse_dt(x: str) -> Optional[datetime]:
    if pd.isna(x) or x is None or str(x).strip() == "":
        return None
    try:
        # Try flexible parser
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        return None


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_list(s: str) -> Optional[np.ndarray]:
    if pd.isna(s):
        return None
    st = str(s).strip()
    if not st:
        return None
    try:
        # Accept Python/JSON-like list
        arr = eval(st, {"__builtins__": {}}, {})  # safe-ish for simple literals
        return np.array(arr, dtype=float)
    except Exception:
        return None


def _time_filter_before_resolution(df_f: pd.DataFrame, df_r: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only forecasts strictly before each question's resolution time.
    If question not resolved, we drop it (cannot use for calibration).
    """
    df = df_f.merge(
        df_r[[R_QUESTION_ID, R_RESOLUTION_TIME, R_RESOLVED]],
        left_on=COL_QUESTION_ID, right_on=R_QUESTION_ID, how="left"
    )
    df = df[df[R_RESOLVED] == True].copy()
    df["res_time"] = df[R_RESOLUTION_TIME].apply(_parse_dt)
    df["f_time"] = df[COL_CREATED_AT].apply(_parse_dt)
    df = df[~df["res_time"].isna() & ~df["f_time"].isna()]
    df = df[df["f_time"] < df["res_time"]].copy()
    return df


def _final_forecast_pre_resolution(df_q: pd.DataFrame) -> pd.Series:
    """
    For a single question's time-filtered rows, return the *last* forecast before resolution.
    """
    return df_q.sort_values("f_time").iloc[-1]


def _brier_binary(p: float, y: int) -> float:
    return (p - y) ** 2


def _crps_from_quantiles(y: float, q10: float, q50: float, q90: float) -> float:
    """
    CRPS-lite: approximate CRPS using three quantiles (p10,p50,p90).
    There are different approximations; this one is a simple piecewise that is
    monotone with coverage errors. It's not an exact CRPS, but correlates well.
    """
    # If you later store full CDF or many quantiles, replace this with a proper CRPS.
    # Simple approach: penalize distance to median, with extra penalty when y lies outside [p10,p90].
    base = abs(y - q50)
    lo_pen = max(0.0, q10 - y)
    hi_pen = max(0.0, y - q90)
    return base + 0.5 * (lo_pen + hi_pen)


def _lin_regress(x: np.ndarray, y: np.ndarray) -> float:
    """
    Return slope of y ~ a + b*x. Used for trajectory trend.
    """
    if len(x) < 2:
        return 0.0
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # beta = [a, b]
    return float(beta[1])


def export_calibration_and_trajectories():
    out_dir = "data/calibration_exports"
    _ensure_dir(out_dir)

    # --- Load inputs
    if not os.path.exists(FORECASTS_CSV):
        print(f"[warn] {FORECASTS_CSV} not found. Nothing to do.")
        return
    if not os.path.exists(RESOLUTIONS_CSV):
        print(f"[warn] {RESOLUTIONS_CSV} not found. Nothing to do.")
        return

    df_f = pd.read_csv(FORECASTS_CSV)
    df_r = pd.read_csv(RESOLUTIONS_CSV)

    # Normalize booleans
    if R_RESOLVED in df_r.columns:
        df_r[R_RESOLVED] = df_r[R_RESOLVED].astype(str).str.lower().isin(["1","true","yes"])

    # Keep only forecasts made BEFORE resolution
    df_f = _time_filter_before_resolution(df_f, df_r)

    # Split by type (based on your logging)
    df_bin = df_f[df_f[COL_QUESTION_TYPE].str.lower().eq("binary")].copy()
    df_mcq = df_f[df_f[COL_QUESTION_TYPE].str.lower().eq("mcq")].copy()
    df_num = df_f[df_f[COL_QUESTION_TYPE].str.lower().isin(["numeric","discrete"])].copy()

    # --- BINARY: take final pre-resolution forecast per question
    bin_rows = []
    for qid, g in df_bin.groupby(COL_QUESTION_ID):
        r = _final_forecast_pre_resolution(g)
        p = float(r[COL_USER_PROB]) if COL_USER_PROB in r else np.nan
        # outcome
        y = df_r.loc[df_r[R_QUESTION_ID]==qid, R_OUTCOME_BIN]
        y = int(y.iloc[0]) if len(y)>0 and not pd.isna(y.iloc[0]) else None
        if y is None or pd.isna(p):
            continue
        bin_rows.append({"question_id": qid, "p": p, "y": y})

    dfb = pd.DataFrame(bin_rows)
    if not dfb.empty:
        # Decile binning
        dfb["bin"] = np.clip((dfb["p"] * BINARY_DECILES).astype(int), 0, BINARY_DECILES-1)
        grp = dfb.groupby("bin", as_index=False).agg(
            count=("y","size"),
            mean_p=("p","mean"),
            emp_rate=("y","mean"),
            brier=("p", lambda x: np.nan) # placeholder
        )
        # Per-bin Brier: average over items in the bin
        briers = []
        for b, g in dfb.groupby("bin"):
            briers.append({"bin": b, "brier": float(np.mean([(row["p"]-row["y"])**2 for _,row in g.iterrows()]))})
        grp = grp.drop(columns=["brier"]).merge(pd.DataFrame(briers), on="bin", how="left")

        # Reliability gap per bin (abs difference)
        grp["abs_calib_gap"] = (grp["emp_rate"] - grp["mean_p"]).abs()

        grp.to_csv(os.path.join(out_dir, "binary_deciles.csv"), index=False)

        # Overall Brier + a compact summary
        overall_brier = float(np.mean([(row["p"]-row["y"])**2 for _,row in dfb.iterrows()]))
        pd.DataFrame([{
            "n_questions": int(dfb.shape[0]),
            "overall_brier": overall_brier
        }]).to_csv(os.path.join(out_dir, "binary_summary.csv"), index=False)

    # --- MCQ: final pre-resolution top-1 prob & outcome index
    mcq_rows = []
    for qid, g in df_mcq.groupby(COL_QUESTION_ID):
        r = _final_forecast_pre_resolution(g)
        probs = _safe_list(r.get(COL_PROBS_VECTOR, ""))
        true_idx = df_r.loc[df_r[R_QUESTION_ID]==qid, R_OUTCOME_MCQ]
        true_idx = int(true_idx.iloc[0]) if len(true_idx)>0 and not pd.isna(true_idx.iloc[0]) else None
        if probs is None or true_idx is None or true_idx<0 or true_idx>=len(probs):
            continue
        top1 = float(np.max(probs))
        pred_idx = int(np.argmax(probs))
        correct = int(pred_idx == true_idx)
        # multiclass Brier (sum over classes (p_k - y_k)^2)
        yvec = np.zeros_like(probs)
        yvec[true_idx] = 1.0
        mc_brier = float(np.sum((probs - yvec)**2))
        mcq_rows.append({
            "question_id": qid,
            "top1_prob": top1,
            "pred_idx": pred_idx,
            "true_idx": true_idx,
            "correct": correct,
            "mc_brier": mc_brier
        })
    dfm = pd.DataFrame(mcq_rows)
    if not dfm.empty:
        # ECE-style bins on top-1 prob
        dfm["bin"] = np.clip((dfm["top1_prob"] * MCQ_BINS).astype(int), 0, MCQ_BINS-1)
        grp = dfm.groupby("bin", as_index=False).agg(
            count=("correct","size"),
            mean_top1=("top1_prob","mean"),
            acc=("correct","mean"),
            mean_mc_brier=("mc_brier","mean")
        )
        grp["abs_calib_gap"] = (grp["acc"] - grp["mean_top1"]).abs()
        grp.to_csv(os.path.join(out_dir, "mcq_bins.csv"), index=False)

        ece = float(np.average(grp["abs_calib_gap"], weights=grp["count"]))
        pd.DataFrame([{
            "n_questions": int(dfm.shape[0]),
            "ece_top1": ece,
            "avg_mc_brier": float(dfm["mc_brier"].mean())
        }]).to_csv(os.path.join(out_dir, "mcq_summary.csv"), index=False)

    # --- Numeric/Discrete: PIT-lite coverage using p10/p50/p90
    num_rows = []
    # Join outcome (if available)
    df_num = df_num.merge(
        df_r[[R_QUESTION_ID, R_OUTCOME_NUMERIC]], left_on=COL_QUESTION_ID, right_on=R_QUESTION_ID, how="left"
    )
    for qid, g in df_num.groupby(COL_QUESTION_ID):
        r = _final_forecast_pre_resolution(g)
        try:
            q10 = float(r[COL_NUMERIC_P10]); q50 = float(r[COL_NUMERIC_P50]); q90 = float(r[COL_NUMERIC_P90])
        except Exception:
            continue
        y = g[R_OUTCOME_NUMERIC].iloc[0]
        if pd.isna(y):
            # If you don’t have numeric outcomes, you’ll still get the exported quantiles.
            y = np.nan
        else:
            y = float(y)
        # Coverage flags
        in_10_90 = (not pd.isna(y)) and (q10 <= y <= q90)
        below_10  = (not pd.isna(y)) and (y < q10)
        above_90  = (not pd.isna(y)) and (y > q90)
        crps = _crps_from_quantiles(y, q10, q50, q90) if not pd.isna(y) else np.nan
        num_rows.append({
            "question_id": qid, "p10": q10, "p50": q50, "p90": q90, "y": y,
            "covered_10_90": int(in_10_90), "below_p10": int(below_10), "above_p90": int(above_90),
            "crps_lite": crps
        })
    dfn = pd.DataFrame(num_rows)
    if not dfn.empty:
        # Export row-wise detail (good for scatter/interval plots)
        dfn.to_csv(os.path.join(out_dir, "numeric_pit_bins.csv"), index=False)
        # Summary rates
        cov = float(dfn["covered_10_90"].mean()) if "covered_10_90" in dfn else float("nan")
        below = float(dfn["below_p10"].mean()) if "below_p10" in dfn else float("nan")
        above = float(dfn["above_p90"].mean()) if "above_p90" in dfn else float("nan")
        crps_mean = float(dfn["crps_lite"].mean(skipna=True))
        pd.DataFrame([{
            "n_questions": int(dfn.shape[0]),
            "p10_p90_coverage": cov,   # target ~0.80 if p10/p90
            "below_p10_rate": below,   # target ~0.10
            "above_p90_rate": above,   # target ~0.10
            "crps_lite_mean": crps_mean
        }]).to_csv(os.path.join(out_dir, "numeric_summary.csv"), index=False)

    # --- Trajectories: how the forecast evolved before resolution
    traj_rows = []
    for qid, g in df_f.groupby(COL_QUESTION_ID):
        g = g.sort_values("f_time").copy()
        t0 = g["f_time"].iloc[0]
        tN = g["f_time"].iloc[-1]
        duration_days = (tN - t0).total_seconds() / 86400.0

        # Build a single scalar series “f(t)” for trajectory:
        # - Binary: use probability
        # - MCQ: use max class probability (top-1)
        # - Numeric: use median p50 (normalize via rank isn’t necessary for simple stats)
        typ = str(g[COL_QUESTION_TYPE].iloc[0]).lower()
        if typ == "binary":
            s = g[COL_USER_PROB].astype(float).to_numpy()
        elif typ == "mcq":
            s = g[COL_PROBS_VECTOR].apply(_safe_list).apply(lambda a: np.max(a) if a is not None else np.nan).astype(float).to_numpy()
        else:
            s = g[COL_NUMERIC_P50].astype(float).to_numpy()

        s = s[~np.isnan(s)]
        if len(s) == 0:
            continue

        # Volatility: standard deviation of the series
        vol = float(np.std(s))

        # Trend: slope vs time index (0..n-1)
        x = np.arange(len(s), dtype=float)
        slope = _lin_regress(x, s)

        # Last-24h change
        # Define a 24h cutoff before final forecast
        cutoff = tN - timedelta(hours=24)
        s_last = g[g["f_time"] >= cutoff]
        if len(s_last) >= 2:
            if typ == "binary":
                s_last_vals = s_last[COL_USER_PROB].astype(float).to_numpy()
            elif typ == "mcq":
                s_last_vals = s_last[COL_PROBS_VECTOR].apply(_safe_list).apply(lambda a: np.max(a) if a is not None else np.nan).astype(float).to_numpy()
            else:
                s_last_vals = s_last[COL_NUMERIC_P50].astype(float).to_numpy()
            last_24h_change = float(s_last_vals[-1] - s_last_vals[0])
        else:
            last_24h_change = 0.0

        traj_rows.append({
            "question_id": qid,
            "question_type": typ,
            "n_updates": int(len(g)),
            "span_days": duration_days,
            "volatility": vol,
            "trend_slope": slope,
            "last_24h_change": last_24h_change
        })

    dft = pd.DataFrame(traj_rows)
    if not dft.empty:
        dft.to_csv(os.path.join(out_dir, "trajectories.csv"), index=False)

    print(f"[ok] Exports written under {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export calibration bin CSVs + simple trajectory stats.")
    _ = parser.parse_args()
    export_calibration_and_trajectories()
