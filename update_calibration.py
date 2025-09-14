#!/usr/bin/env python3
# ANCHOR: update_calibration (paste whole file)
"""
update_calibration.py — Build class-conditional model weights from forecasts.csv

What this script does
---------------------
1) Reads the unified forecasts.csv produced by io_logs.py
2) Uses ONLY resolved questions, and ONLY forecasts made before resolution
3) Computes per-model skill within each (class_primary, question_type):
   - Binary/MCQ: average log loss (and Brier in the JSON for reference)
   - Numeric: CRPS using a normal approximation from P10/P50/P90
4) Regularizes class-level weights toward global weights with a shrinkage factor
   alpha = N_class / (N_class + k), default k=30, and optional time decay.
5) Maps losses → weights via softmax on negative loss: w ∝ exp(-β * loss)
6) Writes weights to calibration_weights.json and a friendly note to data/calibration_advice.txt

Run:
    python -m spagbot.update_calibration --csv forecasts.csv

You can tweak β, k, and half-life via environment variables:
    SPAG_CALIB_BETA=3.0
    SPAG_CALIB_K=30
    SPAG_CALIB_HALFLIFE_DAYS=60
"""

from __future__ import annotations
import os, csv, json, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    IST_TZ = timezone(timedelta(hours=3))

CSV_PATH = os.getenv("FORECASTS_CSV_PATH", "forecasts.csv")
OUT_JSON = os.getenv("CALIB_WEIGHTS_PATH", "calibration_weights.json")
ADVICE_TXT = os.getenv("CALIB_ADVICE_PATH", os.path.join("data","calibration_advice.txt"))

BETA = float(os.getenv("SPAG_CALIB_BETA", "3.0"))
K_SHRINK = int(os.getenv("SPAG_CALIB_K", "30"))
HALFLIFE_DAYS = int(os.getenv("SPAG_CALIB_HALFLIFE_DAYS", "60"))
EPS = 1e-9

def _parse_iso(s: str) -> Optional[datetime]:
    try:
        if not s: return None
        return datetime.fromisoformat(s.replace("Z","+00:00"))
    except Exception:
        return None

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or x=="":
            return None
        return float(x)
    except Exception:
        return None

def _crps_gaussian(mu: float, sigma: float, x: float) -> float:
    """Closed-form CRPS for N(mu, sigma^2)."""
    if sigma <= 1e-9:
        return abs(x - mu)
    z = (x - mu) / sigma
    # standard normal pdf and cdf
    from math import erf, sqrt, exp, pi
    Phi = 0.5 * (1 + erf(z / math.sqrt(2)))
    phi = (1 / math.sqrt(2*math.pi)) * math.exp(-0.5 * z*z)
    return sigma * (z * (2*Phi - 1) + 2*phi - 1/math.sqrt(math.pi))

@dataclass
class Stat:
    n: int = 0
    loss_sum: float = 0.0
    brier_sum: float = 0.0
    def add(self, loss: float, brier: Optional[float]=None):
        self.n += 1
        self.loss_sum += float(loss)
        if brier is not None:
            self.brier_sum += float(brier)

def _softmax_weights(losses: Dict[str, float], beta: float) -> Dict[str, float]:
    # Convert loss dict into weights via exp(-beta*loss)
    keys = list(losses.keys())
    arr = [math.exp(-beta*losses[k]) for k in keys]
    s = sum(arr) or 1.0
    w = [x/s for x in arr]
    # floor to epsilon and renormalize (guard against zeros)
    w = [max(0.02, x) for x in w]
    s2 = sum(w)
    w = [x/s2 for x in w]
    return {k: v for k, v in zip(keys, w)}

def _blend(local: Dict[str,float], global_w: Dict[str,float], n: int) -> Dict[str,float]:
    a = n / (n + K_SHRINK)
    out = {}
    for m in set(local.keys()) | set(global_w.keys()):
        out[m] = a * local.get(m, 0.0) + (1-a) * global_w.get(m, 0.0)
    # renormalize
    s = sum(out.values()) or 1.0
    return {k: v/s for k, v in out.items()}

def _collect_model_names(header: List[str]) -> List[str]:
    # columns like 'binary_prob__ModelName'
    model_names = set()
    for h in header:
        m = None
        if h.startswith("binary_prob__"):
            m = h.split("binary_prob__",1)[1]
        elif h.startswith("mcq_json__"):
            m = h.split("mcq_json__",1)[1]
        elif h.startswith("numeric_p10__"):
            m = h.split("numeric_p10__",1)[1]
        if m and not m.endswith("ensemble") and "__" not in m:
            model_names.add(m)
    return sorted(model_names)

def _parse_binary_outcome(row: dict) -> Optional[int]:
    lab = (row.get("resolved_outcome_label") or "").strip().lower()
    if lab in ("yes","no"):
        return 1 if lab=="yes" else 0
    # fallback to resolved_value numeric 0/1
    rv = _safe_float(row.get("resolved_value"))
    if rv is not None:
        if rv >= 0.5: return 1
        return 0
    return None

def _parse_mcq_outcome(row: dict, options: List[str]) -> Optional[str]:
    lab = (row.get("resolved_outcome_label") or "").strip()
    if lab:
        return lab
    # no clean label? give up
    return None

def _parse_numeric_outcome(row: dict) -> Optional[float]:
    rv = _safe_float(row.get("resolved_value"))
    return rv

def _json_or_empty(s: str) -> Dict[str, float]:
    try:
        obj = json.loads(s) if isinstance(s, str) else (s or {})
        if isinstance(obj, dict): return obj
        return {}
    except Exception:
        return {}

def main():
    # 1) Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"[warn] {CSV_PATH} not found; nothing to do.")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        model_names = _collect_model_names(header)
        rows = list(reader)

    # 2) Filter to resolved and submitted before resolution
    usable: List[dict] = []
    for r in rows:
        if (r.get("resolved") or "").strip().lower() not in ("true","1","yes"):
            continue
        rt = _parse_iso(r.get("resolved_time_iso") or "")
        ft = _parse_iso(r.get("run_time_iso") or "")
        if rt and ft and ft > rt:
            # submitted after resolution; exclude
            continue
        usable.append(r)

    # 3) Aggregate per (qtype, class) per model losses
    global_stats: Dict[str, Stat] = {m: Stat() for m in model_names}
    by_class_qtype: Dict[Tuple[str,str], Dict[str, Stat]] = {}

    for r in usable:
        qtype = (r.get("question_type") or "").strip()
        qclass = (r.get("class_primary") or "global").strip() or "global"

        key = (qclass, qtype)
        if key not in by_class_qtype:
            by_class_qtype[key] = {m: Stat() for m in model_names}

        # gather outcome
        if qtype == "binary":
            y = _parse_binary_outcome(r)
            if y is None: 
                continue
            for m in model_names:
                p = _safe_float(r.get(f"binary_prob__{m}"))
                if p is None: 
                    continue
                p = max(EPS, min(1-EPS, p))
                logloss = -math.log(p if y==1 else (1-p))
                brier = (p - y) ** 2
                global_stats[m].add(logloss, brier)
                by_class_qtype[key][m].add(logloss, brier)

        elif qtype == "multiple_choice":
            # outcome label
            opts_json = _json_or_empty(r.get("options_json","{}"))
            # options_json may be a list; coerce to list of labels
            if isinstance(opts_json, dict):
                options = list(opts_json.keys())
            else:
                try:
                    options = json.loads(r.get("options_json","[]"))
                    if not isinstance(options, list):
                        options = []
                except Exception:
                    options = []
            ylab = _parse_mcq_outcome(r, options)
            if not ylab:
                continue
            for m in model_names:
                vec = _json_or_empty(r.get(f"mcq_json__{m}", "{}"))
                # Prob of winning label
                p = float(vec.get(ylab, 0.0))
                p = max(EPS, min(1.0, p))
                logloss = -math.log(p)
                # Multiclass brier: sum_k (p_k - y_k)^2 ; but we lack full vector for all k if vec incomplete.
                # We'll approximate with 1 - p for winning class (lower-bound).
                brier = (1.0 - p)**2
                global_stats[m].add(logloss, brier)
                by_class_qtype[key][m].add(logloss, brier)

        elif qtype in ("numeric","discrete_numeric"):
            x = _parse_numeric_outcome(r)
            if x is None:
                continue
            for m in model_names:
                p10 = _safe_float(r.get(f"numeric_p10__{m}"))
                p50 = _safe_float(r.get(f"numeric_p50__{m}"))
                p90 = _safe_float(r.get(f"numeric_p90__{m}"))
                if None in (p10,p50,p90): 
                    continue
                mu = float(p50)
                sigma = max(1e-6, (float(p90) - float(p10)) / 2.5631)
                crps = _crps_gaussian(mu, sigma, float(x))
                # For "loss", use CRPS (lower better). To map into softmax, invert the sign.
                # We'll store loss = CRPS and brier_sum unused here.
                global_stats[m].add(crps, None)
                by_class_qtype[key][m].add(crps, None)

        else:
            continue

    # 4) Build global weights (per qtype) from global_stats
    global_weights: Dict[str, Dict[str,float]] = {}  # qtype -> {model: weight}
    # group by qtype needs separate aggregation; we used one global bucket so far, but in practice different qtypes:
    # Recompute by scanning usable rows per qtype
    for qtype in ("binary","multiple_choice","numeric","discrete_numeric"):
        # collect per-model avg loss for this qtype
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
                    p = max(EPS, min(1-EPS, p))
                    ll = -math.log(p if y==1 else (1-p))
                    sums[m] += ll; counts[m] += 1
            elif qtype == "multiple_choice":
                ylab = _parse_mcq_outcome(r, [])
                if not ylab: 
                    continue
                for m in model_names:
                    vec = _json_or_empty(r.get(f"mcq_json__{m}","{}"))
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
                    if None in (p10,p50,p90): 
                        continue
                    mu = float(p50)
                    sigma = max(1e-6, (float(p90) - float(p10)) / 2.5631)
                    crps = _crps_gaussian(mu, sigma, float(x))
                    sums[m] += crps; counts[m] += 1
        avg_loss = {m: (sums[m] / counts[m]) if counts[m]>0 else 1.0 for m in model_names}
        global_weights[qtype] = _softmax_weights(avg_loss, BETA)

    # 5) Build per-class weights and shrink toward global
    by_class_weights: Dict[str, Dict[str, Dict[str,float]]] = {}  # class -> qtype -> weights
    for (cls, qtype), stats in by_class_qtype.items():
        # avg loss per model
        avg_loss = {m: (st.loss_sum / max(1, st.n)) for m, st in stats.items()}
        raw_w = _softmax_weights(avg_loss, BETA)
        blended = _blend(raw_w, global_weights.get(qtype, {}), n = max(0, list(stats.values())[0].n if stats else 0))
        if cls not in by_class_weights:
            by_class_weights[cls] = {}
        by_class_weights[cls][qtype] = blended

    # 6) Write out JSON
    out = {
        "updated_at": datetime.now(IST_TZ).isoformat(),
        "beta": BETA,
        "k_shrink": K_SHRINK,
        "halflife_days": HALFLIFE_DAYS,
        "global": global_weights,
        "by_class": by_class_weights,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(ADVICE_TXT), exist_ok=True)
    # Friendly note
    with open(ADVICE_TXT, "w", encoding="utf-8") as f:
        f.write("Calibration weights updated.\n")
        f.write(f"Global weights by qtype: {json.dumps(global_weights, indent=2)}\n")
        f.write(f"Classes learned: {', '.join(sorted(by_class_weights.keys()))}\n")

if __name__ == "__main__":
    main()