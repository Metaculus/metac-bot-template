#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayes_mc.py
-----------
Monte Carlo layer for Bayesian reasoning in Spagbot.
VERSION: 2.0 (2025-09-15)

This module:
- Defines light-weight Bayesian updaters for Binary (Beta), MCQ (Dirichlet), and Numeric (Normal Mixture).
- Accepts multiple "evidence sources" (e.g., LLMs), each with a forecast + confidence weight w.
- Produces posterior/posterior-predictive Monte Carlo samples and summary statistics.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import math
import numpy as np

# -----------------------------
# Common configuration
# -----------------------------
DEFAULT_SAMPLES = 20000
DEFAULT_SEED    = 42

# -----------------------------
# Utilities
# -----------------------------
def apply_calibration_weight(raw_weight: float, question_type: str, top_prob: float) -> float:
    """
    Calibrated weight for external signals when explicit calibration data is sparse.
    Intuition: shrink weight near extremes to avoid overconfident pull.
      - At p=0.5 → keep full weight.
      - At p≈0 or 1 → shrink toward a floor (e.g., 0.2).
    """
    try:
        raw_weight = float(raw_weight)
        if question_type == "binary":
            penalty = 1.0 - 4.0 * (float(top_prob) - 0.5) ** 2  # parabola: 1 @0.5, 0 @0/1
            return float(max(0.2, min(1.0, raw_weight * max(0.0, penalty))))
        return float(max(0.2, min(1.0, raw_weight)))
    except Exception:
        return float(max(0.2, min(1.0, raw_weight)))

# -----------------------------
# Data containers
# -----------------------------
@dataclass
class BinaryEvidence:
    p: float
    w: float = 1.0

@dataclass
class MCQEvidence:
    probs: List[float]
    w: float = 1.0

@dataclass
class NumericEvidence:
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    w: float = 1.0
    samples: Optional[np.ndarray] = None

# -----------------------------
# Binary: Beta-Binomial
# -----------------------------
@dataclass
class BinaryPrior:
    alpha: float = 1.0
    beta: float = 1.0

def update_binary_with_mc(
    prior: BinaryPrior,
    evidences: List[BinaryEvidence],
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED
) -> Dict[str, object]:
    a, b = float(prior.alpha), float(prior.beta)
    for ev in evidences:
        p = min(max(float(ev.p), 1e-6), 1-1e-6)
        w = max(float(ev.w), 0.0)
        a += w * p
        b += w * (1.0 - p)
    rng = np.random.default_rng(seed)
    theta = rng.beta(a, b, size=n_samples)

    mean = float(theta.mean())
    p10, p50, p90 = np.percentile(theta, [10, 50, 90])
    return {
        "samples": theta,
        "mean": mean,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "posterior_alpha": a,
        "posterior_beta": b
    }

# -----------------------------
# MCQ: Dirichlet-Multinomial
# -----------------------------
@dataclass
class DirichletPrior:
    alphas: List[float]

def update_mcq_with_mc(
    prior: DirichletPrior,
    evidences: List[MCQEvidence],
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED
) -> Dict[str, object]:
    alpha = np.array(prior.alphas, dtype=float)
    K = alpha.size
    for ev in evidences:
        p = np.array(ev.probs, dtype=float)
        if p.size != K:
            raise ValueError(f"Evidence probs size {p.size} != prior K={K}")
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum()
        w = max(float(ev.w), 0.0)
        alpha += w * p

    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(alpha, size=n_samples)   # (n_samples, K)

    mean = phi.mean(axis=0)
    p10 = np.percentile(phi, 10, axis=0)
    p50 = np.percentile(phi, 50, axis=0)
    p90 = np.percentile(phi, 90, axis=0)
    return {
        "samples": phi,
        "mean": mean.tolist(),
        "p10": p10.tolist(),
        "p50": p50.tolist(),
        "p90": p90.tolist(),
        "posterior_alpha": alpha.tolist()
    }

# -----------------------------
# Numeric/Discrete: Mixture of Normals from quantiles
# -----------------------------

def _fit_normal_from_q(q10: float, q50: float, q90: float) -> Tuple[float, float]:
    mu = q50
    z90 = 1.2815515655446004
    sigma = (q90 - q10) / (2.0 * z90)
    return mu, max(sigma, 1e-6)

def update_numeric_with_mc(
    evidences: List[NumericEvidence],
    prior: Optional[Any] = None,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    weights = [max(ev.w, 0.0) for ev in evidences]
    total_w = sum(weights) if sum(weights) > 0 else 1.0
    out_samples = []

    for ev, w in zip(evidences, weights):
        ni = max(0, int(round(n_samples * (w / total_w))))
        if ni == 0:
            continue
        if ev.p10 is not None and ev.p50 is not None and ev.p90 is not None:
            q10, q50, q90 = sorted([ev.p10, ev.p50, ev.p90])
            mu, sigma = _fit_normal_from_q(q10, q50, q90)
            samples = rng.normal(mu, sigma, size=ni)
            out_samples.append(samples)
        elif ev.samples is not None and len(ev.samples) > 0:
            idx = rng.integers(0, len(ev.samples), size=ni)
            out_samples.append(np.asarray(ev.samples)[idx])

    if not out_samples:
        fallback = rng.normal(0.0, 10.0, size=n_samples)
        out_samples = [fallback]

    final_samples = np.concatenate(out_samples)
    if final_samples.size > n_samples:
        idx = rng.choice(final_samples.size, n_samples, replace=False)
        final_samples = final_samples[idx]

    mean = float(np.mean(final_samples))
    p10, p50, p90 = np.percentile(final_samples, [10, 50, 90])
    return {
        "samples": final_samples,
        "mean": mean,
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90)
    }
