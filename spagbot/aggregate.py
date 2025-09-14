# ANCHOR: aggregate (paste whole file)
from __future__ import annotations
import math, numpy as np
from typing import List, Tuple, Dict, Optional, Any

from .ensemble import EnsembleResult, sanitize_mcq_vector, MemberOutput
from . import bayes_mc as BMC

# ---------------- Binary ----------------

def aggregate_binary(
    ensemble_res: EnsembleResult,
    gtmc1_signal: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Bayesian aggregation for binary questions using bayes_mc.
    """
    evidences = []
    # 1. LLM Ensemble Evidence
    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, (float, int)):
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.BinaryEvidence(p=float(m.parsed), w=w))

    # 2. GTMC1 Evidence (if available and valid)
    if gtmc1_signal and "exceedance_ge_50" in gtmc1_signal:
        try:
            p_gtmc = float(gtmc1_signal["exceedance_ge_50"])
            # Weight for GTMC1 can be tuned; 1.5 is a reasonable default
            evidences.append(BMC.BinaryEvidence(p=p_gtmc, w=1.5))
        except (ValueError, TypeError):
            pass

    # Weak prior to let evidence dominate
    prior = BMC.BinaryPrior(alpha=0.1, beta=0.1)

    if not evidences:
        return 0.5, {"method": "empty_evidence", "mean": 0.5}

    bmc_summary = BMC.update_binary_with_mc(prior, evidences)
    # FIX: Remove the non-serializable numpy array before returning
    bmc_summary.pop("samples", None)
    
    final_prob = float(bmc_summary.get("mean", 0.5))
    
    return final_prob, bmc_summary

# ---------------- MCQ ----------------

def aggregate_mcq(
    ensemble_res: EnsembleResult,
    n_options: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Bayesian aggregation for MCQ using bayes_mc.
    """
    evidences = []
    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, list) and len(m.parsed) == n_options:
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.MCQEvidence(probs=m.parsed, w=w))
            
    if not evidences:
        # Fallback to uniform if no valid evidence
        vec = [1.0 / n_options] * n_options
        return vec, {"method": "empty_evidence", "mean": vec}

    # Weak prior
    prior = BMC.DirichletPrior(alphas=[0.1] * n_options)
    bmc_summary = BMC.update_mcq_with_mc(prior, evidences)
    
    # FIX: Remove the non-serializable numpy array before returning
    bmc_summary.pop("samples", None)

    final_vector = bmc_summary.get("mean", [1.0 / n_options] * n_options)
    
    return sanitize_mcq_vector(final_vector), bmc_summary

# ---------------- Numeric / Discrete ----------------

def aggregate_numeric(
    ensemble_res: EnsembleResult,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Numeric aggregation via a Normal Mixture Model in bayes_mc.
    Each LLM's forecast is treated as a separate distribution.
    """
    evidences = []
    
    def _coerce_p10_p50_p90(d: Dict[str, float]) -> Tuple[float, float, float]:
        p10 = float(d.get("P10", 10.0))
        p90 = float(d.get("P90", 90.0))
        if "P50" in d:
            p50 = float(d["P50"])
        else:
            p50 = 0.5 * (p10 + p90)
        # Ensure order
        p10, p50, p90 = sorted([p10, p50, p90])
        return p10, p50, p90

    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, dict):
            p10, p50, p90 = _coerce_p10_p50_p90(m.parsed)
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.NumericEvidence(p10=p10, p50=p50, p90=p90, w=w))

    if not evidences:
        # Degenerate default
        quantiles = {"P10": 10.0, "P50": 50.0, "P90": 90.0}
        return quantiles, {"method": "empty_evidence", "p10": 10.0, "p50": 50.0, "p90": 90.0}

    bmc_summary = BMC.update_numeric_with_mc(evidences)
    
    # FIX: Remove the non-serializable numpy array before returning
    bmc_summary.pop("samples", None)

    quantiles = {
        "P10": bmc_summary.get("p10"),
        "P50": bmc_summary.get("p50"),
        "P90": bmc_summary.get("p90"),
    }
    
    return quantiles, bmc_summary