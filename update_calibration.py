#!/usr/bin/env python3
"""
update_calibration.py (v2)

Builds a concise calibration note for Spagbot from your forecasts.csv, now with:
- Time filtering: only forecasts submitted BEFORE resolution are used.
- Binary: decile reliability + Brier.
- MCQ: Top-1 reliability (ECE & per-bin advice) + multiclass Brier.
- Numeric: PIT-lite coverage checks (p10/p50/p90) + CRPS (MC approximation).

Output: data/calibration_advice.txt
Safe if nothing is resolved yet (writes a neutral note).
"""

from __future__ import annotations
import csv
import os
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import dotenv
dotenv.load_dotenv()

from forecasting_tools import MetaculusApi  # provided by template


CSV_PATH = "forecasts.csv"
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "calibration_advice.txt")

# ---------- utilities ----------

def _parse_iso(s: str) -> Optional[datetime]:
    """Parse ISO8601 strings (with or without 'Z'). Return timezone-aware UTC when possible."""
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _decile_bin01(p: float) -> Tuple[int, int]:
    """Return (lo, hi) decile bounds on [0,1] as percentage ints."""
    pct = max(0.0, min(100.0, float(p) * 100.0))
    lo = int(pct // 10) * 10
    hi = min(lo + 10, 100)
    return lo, hi

def _read_csv_rows(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def _get_resolution_dt(mq) -> Optional[datetime]:
    aj = getattr(mq, "api_json", {}) or {}
    # Try actual_resolve_time (most reliable)
    s = aj.get("question", {}).get("actual_resolve_time") or aj.get("actual_resolve_time")
    if not s:
        s = aj.get("scheduled_resolve_time") or aj.get("scheduled_close_time")
    if isinstance(s, str):
        return _parse_iso(s)
    # Try attribute
    attr = getattr(mq, "actual_resolution_time", None) or getattr(mq, "scheduled_resolution_time", None)
    if isinstance(attr, datetime):
        # assume already tz-aware or UTC-like
        return (attr if attr.tzinfo else attr.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    return None

def _resolve_binary_outcome(mq) -> Optional[int]:
    """Return 1 for YES, 0 for NO, None if unresolved or unknown."""
    aj = getattr(mq, "api_json", {}) or {}
    # check resolved flag
    if not aj.get("resolved", False) and getattr(mq, "actual_resolution_time", None) is None:
        return None
    val = aj.get("resolution", None)
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val >= 0.5 else 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("yes", "true", "1"):
            return 1
        if s in ("no", "false", "0"):
            return 0
    s = aj.get("resolution_string") or getattr(mq, "resolution_string", None)
    if isinstance(s, str):
        s = s.strip().lower()
        if "yes" in s:
            return 1
        if "no" in s:
            return 0
    return None

def _mcq_resolved_option(mq) -> Optional[str]:
    """Return the resolved option name for MCQ, if available."""
    aj = getattr(mq, "api_json", {}) or {}
    if not aj.get("resolved", False) and getattr(mq, "actual_resolution_time", None) is None:
        return None
    # Try aj['resolution'] as string option
    val = aj.get("resolution", None)
    if isinstance(val, str) and val.strip():
        return val.strip()
    # Try resolution_string
    rs = aj.get("resolution_string") or ""
    if isinstance(rs, str) and rs.strip():
        return rs.strip()
    # Fallback: Metaculus sometimes stores as index; attempt to map if options known
    opts = aj.get("question", {}).get("options") or aj.get("options")
    if isinstance(val, (int, float)) and isinstance(opts, list):
        idx = int(val)
        if 0 <= idx < len(opts):
            return str(opts[idx])
    return None

def _collect_mcq_probs(row: dict) -> Dict[str, float]:
    """
    Collect MCQ probabilities from row.
    Strategy:
      - Any column starting with 'MCQ_' is treated as an option prob.
      - Otherwise, try columns that exactly match option names you used as headers.
    """
    out: Dict[str, float] = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        key = str(k)
        if key.startswith("MCQ_"):
            name = key[len("MCQ_") :].strip()
            try:
                out[name] = float(v)
            except Exception:
                pass
    # If none found and there are bare option headers, try to float them
    if not out:
        for k, v in row.items():
            if k in ("Binary_Prob", "Type", "RunTime", "URL", "Question", "QuestionID", "RunID",
                     "CostUSD", "Minutes", "Explanation_Short", "Model", "Extra_Prompt"):
                continue
            try:
                fv = float(v)
                if 0.0 <= fv <= 1.0:
                    out[k] = fv
            except Exception:
                continue
    # Normalize if needed (tolerate small rounding)
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] = max(0.0, min(1.0, out[k] / s))
    return out

def _numeric_percentiles(row: dict) -> Optional[Dict[str, float]]:
    """
    Return dict with percentile keys among {'P01','P10','P25','P50','P75','P90','P99'} if available.
    Supports variants like 'Num_P10' or 'P10' (case-insensitive).
    """
    want = ["P01", "P10", "P25", "P50", "P75", "P90", "P99"]
    found: Dict[str, float] = {}
    low = {k.lower(): k for k in want}
    for k, v in row.items():
        if v is None or v == "":
            continue
        kk = k.lower().replace("num_", "").replace("numeric_", "").strip()
        kk = kk.upper()
        if kk in want:
            try:
                found[kk] = float(v)
            except Exception:
                pass
    return found or None

# ---------- metrics helpers ----------

@dataclass
class BinStat:
    n: int = 0
    sum_pred: float = 0.0
    sum_true: float = 0.0

def _ece_from_bins(bins: Dict[Tuple[int,int], BinStat]) -> float:
    """Expected Calibration Error across bins, weighted by bin counts."""
    total = sum(b.n for b in bins.values())
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in bins.values():
        if b.n == 0:
            continue
        avg_p = b.sum_pred / b.n
        freq = b.sum_true / b.n
        ece += (b.n / total) * abs(avg_p - freq)
    return ece

def _brier_binary(p: float, y: int) -> float:
    return (p - y) ** 2

def _brier_multiclass(probs: Dict[str, float], true_label: str) -> float:
    s = 0.0
    for opt, p in probs.items():
        o = 1.0 if opt == true_label else 0.0
        s += (p - o) ** 2
    return s

def _sample_from_percentiles(pcts: Dict[str, float], n_samples: int = 2000) -> List[float]:
    """
    Build a piecewise-linear quantile function from provided percentiles
    and sample by inverse transform.
    """
    if not pcts:
        return []
    pts = []
    # Add 0 and 1 crude tails from 1%/99% if absent
    grid = []
    def get(k, default):
        return pcts.get(k, default)
    # define quantile points as (u, x)
    mapping = [
        (0.01, get("P01", get("P10", get("P25", get("P50", 0.0)))-abs(get("P50", 0.0))*0.5)),
        (0.10, get("P10", get("P25", get("P50", 0.0)))),
        (0.25, get("P25", get("P50", 0.0))),
        (0.50, get("P50", 0.0)),
        (0.75, get("P75", get("P50", 0.0))),
        (0.90, get("P90", get("P75", get("P50", 0.0)))),
        (0.99, get("P99", get("P90", get("P75", get("P50", 0.0))))),
    ]
    mapping.sort(key=lambda t: t[0])
    # Ensure non-decreasing x
    xs = [x for _, x in mapping]
    for i in range(1, len(xs)):
        xs[i] = max(xs[i], xs[i-1])
    mapping = [(u, xs[i]) for i, (u, _) in enumerate(mapping)]
    # Sample
    out = []
    for _ in range(n_samples):
        u = random.random()
        # find segment
        for i in range(len(mapping)-1):
            u0, x0 = mapping[i]
            u1, x1 = mapping[i+1]
            if u0 <= u <= u1:
                # linear interpolation
                if u1 == u0:
                    x = x0
                else:
                    t = (u - u0) / (u1 - u0)
                    x = x0 + t * (x1 - x0)
                out.append(x)
                break
        else:
            out.append(mapping[-1][1])
    return out

def _crps_mc(pcts: Dict[str, float], x: float, n_samples: int = 2000) -> Optional[float]:
    """
    Monte Carlo CRPS using identity:
      CRPS(F, x) = E|Y - x| - 0.5 E|Y - Y'|
    where Y, Y' ~ F.
    We approximate Y via sampling from piecewise-linear inverse CDF from percentiles.
    """
    if not pcts:
        return None
    ys = _sample_from_percentiles(pcts, n_samples=n_samples)
    if not ys:
        return None
    # E|Y - x|
    m1 = sum(abs(y - x) for y in ys) / len(ys)
    # 0.5 E|Y - Y'|
    # sample a small subset for speed
    k = min(1000, len(ys))
    idxs = random.sample(range(len(ys)), k)
    pairs = random.sample(range(len(ys)), k)
    m2 = sum(abs(ys[i] - ys[j]) for i, j in zip(idxs, pairs)) / k
    return m1 - 0.5 * m2

# ---------- main analysis ----------

def build_calibration_note() -> str:
    rows = _read_csv_rows(CSV_PATH)
    if not rows:
        return ("No forecast history found yet. Keep using conservative updates. "
                "Re-run the calibration updater after questions begin resolving.")

    # Group rows by URL to fetch each Metaculus question once
    by_url: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        url = (r.get("URL") or "").strip()
        if url:
            by_url[url].append(r)

    # Fetch question objects & resolution info
    qinfo: Dict[str, dict] = {}
    for url in by_url.keys():
        try:
            mq = MetaculusApi.get_question_by_url(url)
        except Exception:
            continue
        res_dt = _get_resolution_dt(mq)
        qtype = getattr(mq, "question_type", None) or (mq.api_json.get("question", {}).get("type") if getattr(mq, "api_json", None) else None)
        qinfo[url] = {"mq": mq, "resolved_at": res_dt, "type_hint": qtype}

    # Containers
    # Binary
    bin_bins: Dict[Tuple[int,int], BinStat] = defaultdict(BinStat)
    bin_brier_sum, bin_brier_n = 0.0, 0

    # MCQ
    top1_bins: Dict[Tuple[int,int], BinStat] = defaultdict(BinStat)  # sum_true = #correct
    mcq_brier_sum, mcq_brier_n = 0.0, 0

    # Numeric coverage
    cnt_p10, cnt_p50, cnt_p90, n_num = 0, 0, 0, 0
    crps_sum, crps_n = 0.0, 0

    # Iterate each URL, but only use forecasts submitted BEFORE resolution
    for url, rlist in by_url.items():
        info = qinfo.get(url)
        if not info:
            continue
        res_dt = info["resolved_at"]
        if not res_dt:
            # unresolved — skip for calibration
            continue
        # keep only rows with RunTime < res_dt
        prior_rows = []
        for r in rlist:
            rt = _parse_iso(r.get("RunTime", ""))
            if rt and rt < res_dt:
                prior_rows.append(r)
        if not prior_rows:
            continue
        # Use the last forecast before resolution
        row = sorted(prior_rows, key=lambda x: _parse_iso(x.get("RunTime","")) or datetime.min.replace(tzinfo=timezone.utc))[-1]
        rtype = (row.get("Type") or "").strip().lower()

        mq = info["mq"]

        # ---------- Binary ----------
        if rtype == "binary":
            p_str = row.get("Binary_Prob", "")
            try:
                p = float(p_str)
            except Exception:
                continue
            y = _resolve_binary_outcome(mq)
            if y is None:
                continue
            lo, hi = _decile_bin01(p)
            b = bin_bins[(lo, hi)]
            b.n += 1
            b.sum_pred += p
            b.sum_true += y
            bin_brier_sum += _brier_binary(p, y)
            bin_brier_n += 1

        # ---------- MCQ ----------
        elif rtype in ("multiple_choice", "mcq", "multichoice"):
            probs = _collect_mcq_probs(row)
            if not probs:
                continue
            true_label = _mcq_resolved_option(mq)
            if not true_label:
                continue
            # top-1 calibration
            top_label = max(probs.items(), key=lambda kv: kv[1])[0]
            top_p = probs[top_label]
            lo, hi = _decile_bin01(top_p)
            b = top1_bins[(lo, hi)]
            b.n += 1
            b.sum_pred += top_p
            b.sum_true += (1.0 if top_label == true_label else 0.0)
            # multiclass Brier
            mcq_brier_sum += _brier_multiclass(probs, true_label)
            mcq_brier_n += 1

        # ---------- Numeric ----------
        elif rtype in ("numeric", "discrete", "continuous"):
            pcts = _numeric_percentiles(row)
            if not pcts:
                continue
            # Try to parse realized value
            aj = getattr(mq, "api_json", {}) or {}
            realized = aj.get("resolution") or aj.get("resolved_value") or aj.get("resolution_value")
            if realized is None:
                realized = aj.get("question", {}).get("resolution")
            try:
                x = float(realized)
            except Exception:
                continue
            n_num += 1
            # coverage checks
            def getp(k, default=None):
                return pcts.get(k, default)
            p10, p50, p90 = getp("P10"), getp("P50"), getp("P90")
            if p10 is not None and x <= p10: cnt_p10 += 1
            if p50 is not None and x <= p50: cnt_p50 += 1
            if p90 is not None and x <= p90: cnt_p90 += 1
            # CRPS (MC approx)
            cr = _crps_mc(pcts, x, n_samples=2000)
            if cr is not None and math.isfinite(cr):
                crps_sum += cr
                crps_n += 1

    # ---------- Assemble advice text ----------
    lines: List[str] = []

    # Binary section
    lines.append("BINARY CALIBRATION")
    if bin_brier_n == 0:
        lines.append("- No resolved binary questions (with pre-resolution forecasts) yet.")
        lines.append("- Advice: keep using base rates + small evidence-weighted updates; avoid big swings.")
    else:
        brier_avg = bin_brier_sum / bin_brier_n
        lines.append(f"- Sample size: {bin_brier_n} resolved forecasts (pre-resolution).")
        lines.append(f"- Mean Brier: {brier_avg:.3f} (lower is better).")
        # per-decile advice
        lines.append("- Decile reliability (observed YES vs mean predicted):")
        for (lo, hi) in sorted(bin_bins.keys()):
            b = bin_bins[(lo, hi)]
            if b.n == 0: continue
            avg_p = b.sum_pred / b.n
            freq = b.sum_true / b.n
            gap = freq - avg_p  # + => underconfident
            hint = "No strong adjustment."
            if b.n >= 5 and abs(gap) >= 0.05:
                hint = "UNDERconfident → nudge up." if gap > 0 else "OVERconfident → nudge down."
            lines.append(f"  {lo:02d}–{hi:02d}%: observed={freq:.0%}, avg_pred={avg_p:.0%}, n={b.n}. {hint}")

    lines.append("")  # spacer

    # MCQ section
    lines.append("MULTIPLE-CHOICE CALIBRATION (Top-1)")
    if mcq_brier_n == 0:
        lines.append("- No resolved MCQ questions (with pre-resolution forecasts) yet.")
    else:
        # ECE
        ece = _ece_from_bins(top1_bins)
        lines.append(f"- Sample size: {sum(b.n for b in top1_bins.values())} resolved (top-1).")
        if math.isnan(ece):
            lines.append("- ECE: n/a (insufficient bin coverage).")
        else:
            lines.append(f"- ECE (Expected Calibration Error): {ece:.3f} (lower is better).")
        # Per-bin advice
        lines.append("- Top-1 reliability by confidence bin:")
        for (lo, hi) in sorted(top1_bins.keys()):
            b = top1_bins[(lo, hi)]
            if b.n == 0: continue
            avg_p = b.sum_pred / b.n
            acc = b.sum_true / b.n
            gap = acc - avg_p  # + => underconfident
            hint = "No strong adjustment."
            if b.n >= 10 and abs(gap) >= 0.05:
                hint = "UNDERconfident → nudge up." if gap > 0 else "OVERconfident → nudge down."
            lines.append(f"  {lo:02d}–{hi:02d}%: acc={acc:.0%}, mean_top1={avg_p:.0%}, n={b.n}. {hint}")
        # Multiclass Brier
        lines.append(f"- Mean multiclass Brier: {mcq_brier_sum / mcq_brier_n:.3f} (lower is better).")

    lines.append("")

    # Numeric section
    lines.append("NUMERIC CALIBRATION (PIT-lite + CRPS)")
    if n_num == 0:
        lines.append("- No resolved numeric questions (with pre-resolution forecasts) yet.")
        lines.append("- When numeric results exist, we’ll check p10/p50/p90 coverage and CRPS.")
    else:
        def pct(x, n):
            return (x / n) * 100.0 if n > 0 else float("nan")
        lines.append(f"- Sample size: {n_num} resolved numeric questions.")
        lines.append(f"- Truth ≤ p10: {pct(cnt_p10, n_num):.0f}% (target 10%).")
        lines.append(f"- Truth ≤ p50: {pct(cnt_p50, n_num):.0f}% (target 50%).")
        lines.append(f"- Truth ≤ p90: {pct(cnt_p90, n_num):.0f}% (target 90%).")
        # Directional advice
        adv = []
        if abs(cnt_p10 / n_num - 0.10) >= 0.05:
            adv.append("Lower tail too narrow" if cnt_p10 / n_num > 0.10 else "Lower tail too wide")
        if abs(cnt_p50 / n_num - 0.50) >= 0.08:
            adv.append("Medians biased high" if cnt_p50 / n_num > 0.50 else "Medians biased low")
        if abs(cnt_p90 / n_num - 0.90) >= 0.05:
            adv.append("Upper tail too light" if cnt_p90 / n_num > 0.90 else "Upper tail too heavy")
        if adv:
            lines.append("- Advice: " + "; ".join(adv) + ".")
        else:
            lines.append("- Advice: intervals broadly well calibrated at p10/p50/p90.")

        if crps_n > 0:
            lines.append(f"- Mean CRPS: {crps_sum / crps_n:.3f} (lower is better).")
        else:
            lines.append("- CRPS: n/a (not enough complete percentile sets to estimate)")

    lines.append("")
    lines.append("General takeaway: apply **small** nudges only where gaps are consistent with decent sample sizes (≥10).")

    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    note = build_calibration_note()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(note.strip() + "\n")
    print(f"✅ Wrote calibration advice to: {OUT_PATH}")
    print("\n--- Preview ---\n" + note)


if __name__ == "__main__":
    main()
