#!/usr/bin/env python3
# ANCHOR: update_calibration (paste whole file)
"""
update_calibration.py — Build class-conditional model weights from forecasts.csv

What this script does (in plain English)
---------------------------------------
1) Reads the unified forecasts.csv produced by io_logs.py.
2) Keeps ONLY resolved questions, and ONLY forecasts made before the resolution time.
3) Computes per-model skill within each (class_primary, question_type):
   - Binary & MCQ: average log loss (also tracks Brier in a side counter).
   - Numeric: CRPS using a normal approximation derived from P10/P50/P90.
4) Regularizes class-level weights toward global weights with a shrinkage factor:
     alpha = N_class / (N_class + k), default k=30, with optional time-decay hook.
5) Maps losses → weights via a softmax on negative loss:  w ∝ exp(-β * loss).
6) Writes weights to CALIB_WEIGHTS_PATH and a friendly note to CALIB_ADVICE_PATH.

Run:
    poetry run python update_calibration.py

You can tweak β, k, and half-life via environment variables:
    SPAG_CALIB_BETA=3.0
    SPAG_CALIB_K=30
    SPAG_CALIB_HALFLIFE_DAYS=60

IMPORTANT (your CI error):
--------------------------
If CALIB_WEIGHTS_PATH points to a nested directory (e.g., 'calibration/calibration_weights.json'),
we MUST ensure that directory exists before writing. This file now does that.
"""

from __future__ import annotations
import os, csv, json, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ---- Timezone (Istanbul) ------------------------------------------------------
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    IST_TZ = timezone(timedelta(hours=3))

# ---- Paths via ENV (defaults are safe for local runs) -------------------------
CSV_PATH = os.getenv("FORECASTS_CSV_PATH", "forecasts.csv")
OUT_JSON = os.getenv("CALIB_WEIGHTS_PATH", "calibration_weights.json")
ADVICE_TXT = os.getenv("CALIB_ADVICE_PATH", os.path.join("data", "calibration_advice.txt"))

# ---- Hyperparameters for weighting --------------------------------------------
BETA = float(os.getenv("SPAG_CALIB_BETA", "3.0"))
K_SHRINK = int(os.getenv("SPAG_CALIB_K", "30"))
HALFLIFE_DAYS = int(os.getenv("SPAG_CALIB_HALFLIFE_DAYS", "60"))
EPS = 1e-9

# ---- Small utilities ----------------------------------------------------------
def _parse_iso(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None

def _crps_gaussian(mu: float, sigma: float, x: float) -> float:
    """
    Closed-form CRPS for a Normal(mu, sigma^2).
    Lower is better. If sigma ~ 0, CRPS → |x - mu|.
    """
    if sigma <= 1e-9:
        return abs(x - mu)
    z = (x - mu) / sigma
    from math import erf, sqrt, exp, pi
    Phi = 0.5 * (1 + erf(z / math.sqrt(2)))
    phi = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))

@dataclass
class Stat:
    n: int = 0
    loss_sum: float = 0.0
    brier_sum: float = 0.0
    def add(self, loss: float, brier: Optional[float] = None):
        self.n += 1
        self.loss_sum += float(loss)
        if brier is not None:
            self.brier_sum += float(brier)

def _softmax_weights(losses: Dict[str, float], beta: float) -> Dict[str, float]:
    """
    Convert per-model average losses into weights:
        w_i ∝ exp(-beta * loss_i)
    Includes a small floor and renormalization to avoid zeros.
    """
    keys = list(losses.keys())
    arr = [math.exp(-beta * losses[k]) for k in keys]
    s = sum(arr) or 1.0
    w = [x / s for x in arr]
    w = [max(0.02, x) for x in w]  # floor
    s2 = sum(w)
    w = [x / s2 for x in w]
    return {k: v for k, v in zip(keys, w)}

def _blend(local: Dict[str, float], global_w: Dict[str, float], n: int) -> Dict[str, float]:
    """
    Shrink class weights toward global weights based on sample size n.
        alpha = n / (n + K_SHRINK)
        w = alpha * local + (1 - alpha) * global
    """
    a = n / (n + K_SHRINK) if (n + K_SHRINK) > 0 else 0.0
    out: Dict[str, float] = {}
    for m in set(local.keys()) | set(global_w.keys()):
        out[m] = a * local.get(m, 0.0) + (1 - a) * global_w.get(m, 0.0)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}

def _collect_model_names(header: List[str]) -> List[str]:
    """
    Detect model columns in forecasts.csv header:
      - binary_prob__ModelName
      - mcq_json__ModelName
      - numeric_p10__ModelName (and P50/P90)
    Excludes ensemble columns and any double-underscored variants.
    """
    model_names = set()
    for h in header:
        m = None
        if h.startswith("binary_prob__"):
            m = h.split("binary_prob__", 1)[1]
        elif h.startswith("mcq_json__"):
            m = h.split("mcq_json__", 1)[1]
        elif h.startswith("numeric_p10__"):
            m = h.split("numeric_p10__", 1)[1]
        if m and not m.endswith("ensemble") and "__" not in m:
            model_names.add(m)
    return sorted(model_names)

def _parse_binary_outcome(row: dict) -> Optional[int]:
    lab = (row.get("resolved_outcome_label") or "").strip().lower()
    if lab in ("yes", "no"):
        return 1 if lab == "yes" else 0
    rv = _safe_float(row.get("resolved_value"))
    if rv is not None:
        return 1 if rv >= 0.5 else 0
    return None

def _parse_mcq_outcome(row: dict, options: List[str]) -> Optional[str]:
    lab = (row.get("resolved_outcome_label") or "").strip()
    return lab or None

def _parse_numeric_outcome(row: dict) -> Optional[float]:
    return _safe_float(row.get("resolved_value"))

def _json_or_empty(s: str):
    try:
        if isinstance(s, str):
            return json.loads(s)
        return s or {}
    except Exception:
        return {}

# ---- Main ---------------------------------------------------------------------
def main():
    # 0) Ensure output directories exist BEFORE any writing
    #    This directly fixes your CI error.
    out_dir = os.path.dirname(OUT_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    advice_dir = os.path.dirname(ADVICE_TXT)
    if advice_dir:
        os.makedirs(advice_dir, exist_ok=True)

    # 1) Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"[warn] {CSV_PATH} not found; nothing to do.")
        # Still produce an empty but valid weights file so downstream code never crashes
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "updated_at": datetime.now(IST_TZ).isoformat(),
                "beta": BETA,
                "k_shrink": K_SHRINK,
                "halflife_days": HALFLIFE_DAYS,
                "global": {},
                "by_class": {},
            }, f, indent=2, ensure_ascii=False)
        # Friendly advice note
        with open(ADVICE_TXT, "w", encoding="utf-8") as f:
            f.write(f"No CSV found at {CSV_PATH}. Wrote empty calibration.\n")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        model_names = _collect_model_names(header)
        rows = list(reader)

    # 2) Filter to resolved and submitted BEFORE resolution
    usable: List[dict] = []
    for r in rows:
        if (r.get("resolved") or "").strip().lower() not in ("true", "1", "yes"):
            continue
        rt = _parse_iso(r.get("resolved_time_iso") or "")
        ft = _parse_iso(r.get("run_time_iso") or "")
        if rt and ft and ft > rt:
            # Exclude forecasts logged after resolution
            continue
        usable.append(r)

    # 3) Aggregate losses per (qtype, class) and also track global-by-qtype
    global_stats: Dict[str, Stat] = {m: Stat() for m in model_names}
    by_class_qtype: Dict[Tuple[str, str], Dict[str, Stat]] = {}

    for r in usable:
        qtype = (r.get("question_type") or "").strip()
        qclass = (r.get("class_primary") or "global").strip() or "global"
        key = (qclass, qtype)
        if key not in by_class_qtype:
            by_class_qtype[key] = {m: Stat() for m in model_names}

        if qtype == "binary":
            y = _parse_binary_outcome(r)
            if y is None:
                continue
            for m in model_names:
                p = _safe_float(r.get(f"binary_prob__{m}"))
                if p is None:
                    continue
                p = max(EPS, min(1 - EPS, p))
                logloss = -math.log(p if y == 1 else (1 - p))
                brier = (p - y) ** 2
                global_stats[m].add(logloss, brier)
                by_class_qtype[key][m].add(logloss, brier)

        elif qtype == "multiple_choice":
            # Try to recover the winning label
            try:
                opts = _json_or_empty(r.get("options_json", "[]"))
                if isinstance(opts, dict):
                    options = list(opts.keys())
                else:
                    options = list(opts) if isinstance(opts, list) else []
            except Exception:
                options = []
            ylab = _parse_mcq_outcome(r, options)
            if not ylab:
                continue
            for m in model_names:
                vec = _json_or_empty(r.get(f"mcq_json__{m}", "{}"))
                p = float(vec.get(ylab, 0.0))
                p = max(EPS, min(1.0, p))
                logloss = -math.log(p)
                # Approximate Brier with (1 - p)^2 for the winning class
                brier = (1.0 - p) ** 2
                global_stats[m].add(logloss, brier)
                by_class_qtype[key][m].add(logloss, brier)

        elif qtype in ("numeric", "discrete_numeric"):
            x = _parse_numeric_outcome(r)
            if x is None:
                continue
            for m in model_names:
                p10 = _safe_float(r.get(f"numeric_p10__{m}"))
                p50 = _safe_float(r.get(f"numeric_p50__{m}"))
                p90 = _safe_float(r.get(f"numeric_p90__{m}"))
                if None in (p10, p50, p90):
                    continue
                mu = float(p50)
                sigma = max(1e-6, (float(p90) - float(p10)) / 2.5631)
                crps = _crps_gaussian(mu, sigma, float(x))
                global_stats[m].add(crps, None)
                by_class_qtype[key][m].add(crps, None)

        else:
            continue

    # 4) Build global weights per qtype by re-scanning usable rows
    global_weights: Dict[str, Dict[str, float]] = {}
    for qtype in ("binary", "multiple_choice", "numeric", "discrete_numeric"):
        sums: Dict[str, float] = {m: 0.0 for m in model_names}
        counts: Dict[str, int] = {m: 0 for m in model_names}
        for r in usable:
            if (r.get("question_type") or "") != qtype:
                continue
            if qtype == "binary":
                y = _parse_binary_outcome(r)
                if y is None:
                    continue
                for m in model_names:
                    p = _safe_float(r.get(f"binary_prob__{m}"))
                    if p is None:
                        continue
                    p = max(EPS, min(1 - EPS, p))
                    ll = -math.log(p if y == 1 else (1 - p))
                    sums[m] += ll; counts[m] += 1
            elif qtype == "multiple_choice":
                ylab = _parse_mcq_outcome(r, [])
                if not ylab:
                    continue
                for m in model_names:
                    vec = _json_or_empty(r.get(f"mcq_json__{m}", "{}"))
                    p = float(vec.get(ylab, 0.0))
                    p = max(EPS, min(1.0, p))
                    ll = -math.log(p)
                    sums[m] += ll; counts[m] += 1
            else:
                x = _parse_numeric_outcome(r)
                if x is None:
                    continue
                for m in model_names:
                    p10 = _safe_float(r.get(f"numeric_p10__{m}"))
                    p50 = _safe_float(r.get(f"numeric_p50__{m}"))
                    p90 = _safe_float(r.get(f"numeric_p90__{m}"))
                    if None in (p10, p50, p90):
                        continue
                    mu = float(p50)
                    sigma = max(1e-6, (float(p90) - float(p10)) / 2.5631)
                    crps = _crps_gaussian(mu, sigma, float(x))
                    sums[m] += crps; counts[m] += 1
        avg_loss = {m: (sums[m] / counts[m]) if counts[m] > 0 else 1.0 for m in model_names}
        global_weights[qtype] = _softmax_weights(avg_loss, BETA)

    # 5) Build per-class weights and shrink toward global
    by_class_weights: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (cls, qtype), stats in by_class_qtype.items():
        # average loss per model for this class/qtype
        avg_loss = {m: (st.loss_sum / max(1, st.n)) for m, st in stats.items()}
        raw_w = _softmax_weights(avg_loss, BETA)
        n_examples = next(iter(stats.values())).n if stats else 0
        blended = _blend(raw_w, global_weights.get(qtype, {}), n=n_examples)
        by_class_weights.setdefault(cls, {})[qtype] = blended

    # 6) Write outputs (weights JSON + friendly advice note)
    out_payload = {
        "updated_at": datetime.now(IST_TZ).isoformat(),
        "beta": BETA,
        "k_shrink": K_SHRINK,
        "halflife_days": HALFLIFE_DAYS,
        "global": global_weights,
        "by_class": by_class_weights,
    }

    # Ensure directory still exists (defensive)
    out_dir = os.path.dirname(OUT_JSON)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    # Friendly note
    advice_dir = os.path.dirname(ADVICE_TXT)
    if advice_dir:
        os.makedirs(advice_dir, exist_ok=True)
    with open(ADVICE_TXT, "w", encoding="utf-8") as f:
        f.write("Calibration weights updated.\n")
        f.write(f"Global weights by qtype: {json.dumps(global_weights, indent=2)}\n")
        f.write(f"Classes learned: {', '.join(sorted(by_class_weights.keys()))}\n")

if __name__ == "__main__":
    main()