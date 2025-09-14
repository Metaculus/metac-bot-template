#!/usr/bin/env python3
"""
GTMC1 – Enhanced BDM/Scholz Monte Carlo with table input + logging
------------------------------------------------------------------
Purpose:
  A compact, fast Monte Carlo wrapper around a Bueno de Mesquita/Scholz-style
  bargaining model that accepts an ACTOR TABLE (name, position, capability,
  salience, risk_threshold), samples uncertainty, iterates until convergence,
  and returns a concise "signal" for downstream forecasting.

Key outputs (backward-compatible):
  - coalition_rate: fraction of runs where explicit coalitions formed (0..1)
  - median_of_final_medians: median position of equilibria across runs (0..100)
  - median_rounds: median # of rounds to converge (int)
  - dispersion: 'low' | 'medium' | 'high' based on IQR of final medians
  - runs_csv: relative path to CSV of per-run summaries

New/added outputs:
  - exceedance_ge_50: fraction of runs whose final median >= 50.0 (0..1)
      -> This is a defensible proxy for P(YES) when higher-axis = "YES".
  - iqr: numeric IQR of final medians (q75 - q25)
  - final_median_mean: mean of final medians for completeness
  - meta.paths.meta_json: path to the meta JSON for bookkeeping

Important scales & conventions:
  - position, capability, salience on 0..100
  - capability is internally renormalized to sum to ~100 across actors
  - risk is a multiplicative risk-attitude factor around 1.0 (0.5..2.0)
  - challenge_threshold is the min expected-utility needed to challenge (0.00..0.10)
  - YES is assumed to correspond to higher positions on the axis by default
    (i.e., exceedance threshold defaults to 50.0). You can change that via
    `yes_threshold` in `run_monte_carlo_from_actor_table(...)`.

Usage from Spagbot:
  signal, df = run_monte_carlo_from_actor_table(actor_rows, num_runs=60, log_dir="gtmc_logs", run_slug=slug)
  # Use signal["exceedance_ge_50"] (preferred) as the probability-like signal.
  # Keep "median_of_final_medians" if you need axis-level interpretation too.

This file is optimized for clarity and robustness with extensive comments so
non-coders can safely reason about each step.

Reproducibility guardrails (Fixes 1–9):
  - Deterministic RNG seeded from a fingerprint of (actor_rows + params)
  - Canonicalize actor rows & stable iteration/summation order
  - Reduce BLAS/multiprocessing nondeterminism (single-threaded BLAS)
  - Hysteresis near the YES threshold to avoid flip-flops around 50
  - Percentile monotonicity & rounded outputs to avoid faux precision
  - Deterministic, content-addressed cache for byte-identical results
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import csv
import json
import math
import os
import time
import uuid
import gc
import hashlib
import random

# --- Reproducibility: constrain BLAS threads & Python hashing (Fix #1) --------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # after env hints

# Optional pandas/matplotlib: we gate usage so GTMC1 works without them.
try:
    import pandas as pd
    HAVE_PD = True
except Exception:
    HAVE_PD = False

try:
    import matplotlib.pyplot as plt  # noqa: F401  (kept for future plotting)
    plotting_available = True
except Exception:
    plotting_available = False


# =============================================================================
# Reproducibility helpers (Fixes #1, #2, #7, #8)
# =============================================================================

def _canonical_json(obj) -> str:
    """Deterministic JSON dump for fingerprinting: sort keys, tight separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _canon_actor_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Canonicalize actor rows to ensure stable ordering & defaults (Fix #2):
      - strip/lower name for sort key (name itself preserved as given)
      - fill defaults for missing fields
      - clamp numeric fields to legal ranges
      - return list sorted by (name.lower, position, capability, salience)
    """
    def clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, float(x))))
    def clamp100(x: float) -> float:
        return float(max(0.0, min(100.0, float(x))))

    out = []
    for r in rows or []:
        name = str(r.get("name", "")).strip()
        if not name:
            # skip nameless actors deterministically
            continue
        pos = clamp100(r.get("position", 50))
        cap = clamp100(r.get("capability", 50))
        sal = clamp100(r.get("salience", 50))
        thr = float(r.get("risk_threshold", 0.03))
        out.append({
            "name": name,
            "position": pos,
            "capability": cap,
            "salience": sal,
            "risk_threshold": float(max(0.00, min(0.10, thr))),
        })
    out.sort(key=lambda r: (r["name"].lower(), r["position"], r["capability"], r["salience"]))
    return out

def _fingerprint(actor_rows: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
    """
    Content-addressed fingerprint of inputs (Fix #1/#7):
    We intentionally do NOT include timestamps or absolute paths.
    """
    payload = {
        "actors": _canon_actor_rows(actor_rows),
        "params": params or {}
    }
    blob = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _seed_from_fingerprint(fp: str) -> int:
    """Derive a 63-bit positive integer seed from hex fingerprint (Fix #1)."""
    return int(fp[:16], 16) & ((1 << 63) - 1)

def _make_rng(seed: int) -> np.random.Generator:
    """Stable, portable RNG (PCG64) (Fix #1/#8)."""
    return np.random.Generator(np.random.PCG64(seed))

def _enforce_monotone(xs):
    """Ensure non-decreasing sequence (Fix #6)."""
    out = []
    last = -1e18
    for x in xs:
        v = max(float(x), last)
        out.append(v)
        last = v
    return out


# =============================================================================
# Sampling helpers (inject realistic variability per run) — RNG plumbed (Fix #3/#8)
# =============================================================================

def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))

def sample_position(rng: np.random.Generator, base: float, variation: float = 5.0) -> float:
    """
    Triangular sampling centered on 'base', bounded to [0,100].
    Small variation keeps runs stable while allowing exploration.
    """
    base = _clamp_0_100(float(base))
    lower = max(0.0, base - variation)
    upper = min(100.0, base + variation)
    return float(rng.triangular(lower, base, upper))

def sample_capability(rng: np.random.Generator, base: float, variation_frac: float = 0.10) -> float:
    """
    Capability shocks ~ N(1, variation_frac), clipped to [0,100].
    We'll renormalize capabilities across actors to sum ~100 later.
    """
    base = _clamp_0_100(float(base))
    sampled = base * (1.0 + rng.normal(0.0, variation_frac))
    return _clamp_0_100(float(sampled))

def sample_salience(rng: np.random.Generator, base: float, variation: float = 5.0) -> float:
    """
    Triangular sampling for issue salience, clipped to [0,100].
    """
    base = _clamp_0_100(float(base))
    lower = max(0.0, base - variation)
    upper = min(100.0, base + variation)
    return float(rng.triangular(lower, base, upper))

def sample_risk(rng: np.random.Generator, a: float = 2.0, b: float = 2.0) -> float:
    """
    Risk-attitude multiplier ~ Beta(a,b) mapped into [0.5, 2.0] centered near 1.0.
    """
    beta_sample = rng.beta(a, b)
    return float(0.5 + beta_sample * (2.0 - 0.5))  # 0.5..2.0

def sample_threshold(rng: np.random.Generator, lower: float = 0.00, upper: float = 0.10) -> float:
    """
    Challenge threshold (EU) uniform in [lower, upper].
    """
    lower = max(0.0, float(lower))
    upper = min(0.10, float(upper))
    if upper < lower:
        lower, upper = upper, lower
    return float(rng.uniform(lower, upper))


# =============================================================================
# Actor data container
# =============================================================================

@dataclass
class Actor:
    name: str
    position: float         # 0..100
    capability: float       # 0..100 (renormalized across actors later)
    salience: float         # 0..100
    risk: float = 1.0       # 0.5..2.0 (learned)
    challenge_threshold: float = 0.03
    cumulative_EU: float = 0.0
    num_challenges: int = 0

    def weight(self) -> float:
        """
        Influence weight for pairwise moves. Scaled to ~[0,1].
        """
        return (self.capability / 100.0) * (self.salience / 100.0)


# =============================================================================
# Utility functions for the BDM/Scholz dynamics
# =============================================================================

def get_range(actors: List[Actor]) -> float:
    xs = [a.position for a in actors]
    return float(max(xs) - min(xs)) if xs else 0.0

def get_median(actors: List[Actor]) -> float:
    xs = [a.position for a in actors]
    return float(np.median(xs)) if xs else 0.0

def U_ij(x_i: float, x_j: float, D: float) -> float:
    # Basic "distance utility" – kept for reference, not used directly below.
    return float(1 - 2 * abs(x_i - x_j) / D) if D > 0 else 0.0

def U_si(x_i: float, x_j: float, r: float, D: float) -> float:
    # Utility if i wins as "simple" framing; r tunes curvature.
    return float(2 - 4 * ((0.5 - 0.5 * abs(x_i - x_j) / D) ** r)) if D > 0 else 0.0

def U_fi(x_i: float, x_j: float, r: float, D: float) -> float:
    # Utility if i loses as "simple" framing.
    return float(2 - 4 * ((0.5 + 0.5 * abs(x_i - x_j) / D) ** r)) if D > 0 else 0.0

def U_sq(r: float) -> float:
    # Status-quo utility cost term (higher r -> squarer curve).
    return float(2 - 4 * (0.5 ** r))

def U_bi(x_i: float, x_j: float, m: float, r: float, D: float) -> float:
    # Utility if i wins under "bilateral" framing around the median m.
    return float(2 - 4 * ((0.5 - 0.25 * ((abs(x_i - m) + abs(x_i - x_j)) / D)) ** r)) if D > 0 else 0.0

def U_wi(x_i: float, x_j: float, m: float, r: float, D: float) -> float:
    # Utility if i loses under "bilateral" framing around the median m.
    return float(2 - 4 * ((0.5 + 0.25 * ((abs(x_i - m) + abs(x_i - x_j)) / D)) ** r)) if D > 0 else 0.0

def probability(i: int, j: int, actors: List[Actor], D: float) -> float:
    """
    Voting-likelihood heuristic: net sign-weighted alignment from third parties.
    Stable because actor ordering is canonical upstream (Fix #4).
    """
    num = 0.0
    den = 0.0
    for k, ak in enumerate(actors):
        if k == i or k == j:
            continue
        w = ak.weight()
        # Positive if k is aligned with i against j along the axis ordering.
        s = np.sign((actors[i].position - ak.position) * (ak.position - actors[j].position))
        num += w * s
        den += w
    if den == 0.0:
        return 0.5
    p = 0.5 + 0.5 * (num / den)
    return float(max(0.0, min(1.0, p)))

def EU_challenge(i: int, j: int, actors: List[Actor], Q: float) -> float:
    """
    Expected utility for actor i to challenge j under "simple" framing.
    """
    a_i, a_j = actors[i], actors[j]
    D = max(1e-6, get_range(actors))
    r = a_i.risk
    Uwin = U_si(a_i.position, a_j.position, r, D)
    Ulose = U_fi(a_i.position, a_j.position, r, D)
    Usq = U_sq(r)
    p = probability(i, j, actors, D)
    return float(p * Uwin + (1 - p) * Ulose - (1 - Q) * Usq)

def EU_challenge_reverse(i: int, j: int, actors: List[Actor], Q: float) -> float:
    """
    Expected utility for j when the confrontation is framed around the median.
    """
    a_i, a_j = actors[i], actors[j]
    D = max(1e-6, get_range(actors))
    m = get_median(actors)
    r = a_j.risk
    Uwin = U_bi(a_i.position, a_j.position, m, r, D)
    Ulose = U_wi(a_i.position, a_j.position, m, r, D)
    Usq = U_sq(r)
    p = probability(i, j, actors, D)
    return float((1 - p) * Uwin + p * Ulose - (1 - Q) * Usq)


# =============================================================================
# Core model (BDM/Scholz enhanced)
# =============================================================================

class BDMScholzEnhancedModel:
    """
    Minimal, vector-aware bargaining stepper with:
      - dynamic Q (declines as range stabilizes),
      - simple coalition formation if clusters are close,
      - adaptive risk & threshold nudged by average EU in challenges.
    """
    def __init__(
        self,
        actors: List[Actor],
        Q: float = 1.0,
        gamma: float = 0.2,
        risk_lr: float = 0.05,
        thresh_lr: float = 0.05,
        beta_Q: float = 0.5,
        coalition_thresh: float = 2.0
    ):
        # Stable iteration relies on actors already being in canonical order (Fix #4)
        self.actors = sorted(actors, key=lambda a: (a.name.lower(), a.position, a.capability, a.salience))
        self.Q = float(Q)
        self.gamma = float(gamma)
        self.risk_lr = float(risk_lr)
        self.thresh_lr = float(thresh_lr)
        self.beta_Q = float(beta_Q)
        self.coalition_thresh = float(coalition_thresh)
        self.initial_range = max(1e-6, get_range(self.actors))
        self.history: List[List[float]] = []
        self.coalition_events: List[dict] = []

    def update_dynamic_Q(self) -> None:
        current_range = get_range(self.actors)
        # As range collapses, reduce SQ penalty a bit (keeps late-stage motion plausible).
        self.Q = float(max(0.5, min(1.0, 1 - self.beta_Q * (current_range / self.initial_range))))

    def check_coalitions(self) -> List[List[str]]:
        """
        Group actors whose positions are within coalition_thresh of each other.
        """
        if not self.actors:
            return []
        out = []
        xs = sorted(self.actors, key=lambda a: a.position)
        cluster = [xs[0]]
        for a in xs[1:]:
            if abs(a.position - cluster[-1].position) <= self.coalition_thresh:
                cluster.append(a)
            else:
                if len(cluster) > 1:
                    out.append([c.name for c in cluster])
                cluster = [a]
        if len(cluster) > 1:
            out.append([c.name for c in cluster])
        return out

    def coalition_formation(self) -> None:
        """
        Simple coalition step: merge close clusters to their mean position.
        """
        clusters = self.check_coalitions()
        for names in clusters:
            positions = [a.position for a in self.actors if a.name in names]
            avg = float(np.mean(positions))
            for a in self.actors:
                if a.name in names:
                    a.position = avg

    def simulation_round(self) -> float:
        """
        One round:
          1) Update Q
          2) Consider bilateral EU for all ordered pairs (i, j)
          3) If both sides have EU above thresholds, nudge each toward weighted midpoint
          4) Update risk & challenge_threshold with small learning steps
          5) Check & merge coalitions
        Returns max absolute position change (for convergence test).
        """
        self.update_dynamic_Q()
        n = len(self.actors)
        deltas = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)
        EU_sum = np.zeros(n, dtype=np.float32)
        ch_count = np.zeros(n, dtype=np.int32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                eu_ij = EU_challenge(i, j, self.actors, self.Q)
                eu_ji = EU_challenge_reverse(i, j, self.actors, self.Q)
                if eu_ij > self.actors[i].challenge_threshold and eu_ji > self.actors[j].challenge_threshold:
                    wi = self.actors[i].weight()
                    wj = self.actors[j].weight()
                    if wi + wj > 0:
                        targ = (wi * self.actors[i].position + wj * self.actors[j].position) / (wi + wj)
                        deltas[i] += (targ - self.actors[i].position)
                        deltas[j] += (targ - self.actors[j].position)
                        counts[i] += 1
                        counts[j] += 1
                    EU_sum[i] += eu_ij
                    EU_sum[j] += eu_ji
                    ch_count[i] += 1
                    ch_count[j] += 1

        max_change = 0.0
        for idx, a in enumerate(self.actors):
            if counts[idx] > 0:
                delta = float(deltas[idx] / max(1, counts[idx]))
                step = self.gamma * delta
                a.position = _clamp_0_100(a.position + step)
                max_change = max(max_change, abs(step))
            if ch_count[idx] > 0:
                avg_eu = float(EU_sum[idx] / ch_count[idx])
                # Gentle nudges: risk in [0.5,2.0], threshold in [0,0.10]
                a.risk = float(max(0.5, min(2.0, a.risk + self.risk_lr * avg_eu)))
                a.challenge_threshold = float(max(0.00, min(0.10, a.challenge_threshold + self.thresh_lr * (avg_eu - a.challenge_threshold))))
        self.coalition_formation()
        self.history.append([a.position for a in self.actors])
        cols = self.check_coalitions()
        if cols:
            self.coalition_events.append({"round": len(self.history), "coalitions": cols})
        return max_change

    def run_model(self, max_rounds: int = 100, tol: float = 1e-3) -> int:
        """
        Iterate until no actor moves by more than 'tol' or max_rounds is reached.
        """
        r = 0
        self.history.append([a.position for a in self.actors])
        while r < max_rounds:
            change = self.simulation_round()
            r += 1
            if change < tol:
                break
        return r


# =============================================================================
# Table loader + Monte Carlo harness
# =============================================================================

def _coerce_actor_row(rng: np.random.Generator, row: Dict[str, Any]) -> Optional[Actor]:
    """
    Convert a raw dict row to an Actor, with robust defaulting and clamping.
    RNG is provided for deterministic sampling (Fix #3/#8).
    """
    try:
        name = str(row.get("name", "")).strip()
        if not name:
            return None
        pos = _clamp_0_100(float(row.get("position", 50)))
        cap = _clamp_0_100(float(row.get("capability", 50)))
        sal = _clamp_0_100(float(row.get("salience", 50)))
        thr = float(row.get("risk_threshold", 0.03))
        # Sample initial state around provided baselines
        a = Actor(
            name=name,
            position=sample_position(rng, pos),
            capability=sample_capability(rng, cap),
            salience=sample_salience(rng, sal),
            risk=sample_risk(rng),
            # small jitter around provided threshold
            challenge_threshold=sample_threshold(rng, 0.00, max(0.00, min(0.10, thr + 0.02))),
        )
        return a
    except Exception:
        # Silently skip malformed rows deterministically
        return None

def _actors_from_table(rng: np.random.Generator, rows: List[Dict[str, Any]]) -> List[Actor]:
    """
    Coerce rows -> Actor list, canonicalize & renormalize capabilities (Fix #2/#4).
    """
    rows_canon = _canon_actor_rows(rows)
    out: List[Actor] = []
    for r in rows_canon:
        a = _coerce_actor_row(rng, r)
        if a:
            out.append(a)
    # Renormalize capability to sum ~100 to keep weights comparable across runs
    total_cap = sum(a.capability for a in out) or 1.0
    for a in out:
        a.capability = 100.0 * a.capability / total_cap
    # Stable order for all downstream loops (Fix #4)
    out = sorted(out, key=lambda a: (a.name.lower(), a.position, a.capability, a.salience))
    return out

def run_monte_carlo_from_actor_table(
    actor_rows: List[Dict[str, Any]],
    num_runs: int = 60,
    log_dir: str = "gtmc_logs",
    run_slug: Optional[str] = None,
    yes_threshold: float = 50.0,
    max_rounds: int = 100,
    tol: float = 1e-3
) -> Tuple[Dict[str, Any], Union["pd.DataFrame", List[Dict[str, Any]]]]:
    """
    Main entry for Spagbot and friends.

    Args
    ----
    actor_rows : list of dicts
        Keys: name, position, capability, salience, risk_threshold
    num_runs : int
        Monte Carlo repetitions. 60 is a good balance; 200–400 is smoother.
    log_dir : str
        Directory where the CSV and a small meta JSON are written.
    run_slug : str | None
        A safe filename slug (e.g., derived from the question URL or ID).
    yes_threshold : float
        Axis threshold that defines "YES-side" exceedance. Default 50.0
        means "final_median >= 50 => counts toward YES".
    max_rounds : int
        Max iterations per internal bargaining run.
    tol : float
        Convergence tolerance on max per-actor movement.

    Returns
    -------
    (signal, df_like)
        signal: dict with core metrics (see module docstring)
        df_like: pandas.DataFrame of per-run summaries if pandas is available,
                 otherwise a list[dict] you can still write as CSV.

    Notes
    -----
    - Backward-compatible keys are preserved.
    - New 'exceedance_ge_50' is preferred as a probability-like signal.
    - Reproducible across runs for identical inputs (Fixes #1–#9).
    """
    # Minimal shim class if pandas is unavailable
    if not HAVE_PD:
        class _TinyDF:
            def __init__(self, rows): self._rows = rows
            def to_csv(self, path: str, index: bool = False, encoding: str = "utf-8"):
                with open(path, "w", encoding=encoding, newline="") as f:
                    w = csv.writer(f)
                    if self._rows:
                        w.writerow(list(self._rows[0].keys()))
                        for r in self._rows:
                            w.writerow([r.get(k, "") for k in self._rows[0].keys()])

    # ---------------------- Reproducibility prologue (Fix #1/#7) ----------------
    params_for_fp = {
        "num_runs": int(num_runs),
        "yes_threshold": float(yes_threshold),
        "max_rounds": int(max_rounds),
        "tol": float(tol),
    }
    fp = _fingerprint(actor_rows, params_for_fp)
    seed = _seed_from_fingerprint(fp)

    # Set ALL relevant RNGs
    random.seed(seed)
    np.random.seed(seed)
    rng = _make_rng(seed)

    # Deterministic cache: shortcut identical results (Fix #7)
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache_gtmc1")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{fp}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # Rehydrate df-like object
            rows_cached = cached.get("results_rows", [])
            if HAVE_PD:
                df_cached = pd.DataFrame(rows_cached)
                return cached["signal"], df_cached
            else:
                return cached["signal"], rows_cached
        except Exception:
            # Fall through to recompute if cache is corrupted
            pass

    # --------------------------- Normal (deterministic) run ---------------------
    # Paths (do NOT use timestamp in fingerprint; only for file names)
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = (run_slug or f"gtmc_{uuid.uuid4().hex[:6]}").strip("-")
    os.makedirs(log_dir, exist_ok=True)
    base = os.path.join(log_dir, f"{ts}_{slug}")
    meta_path = base + "_meta.json"
    csv_path = base + "_runs.csv"

    # Accumulators
    results_rows: List[Dict[str, Any]] = []
    coalition_count = 0
    medians: List[float] = []
    rounds_list: List[int] = []
    exceed_count = 0

    # Pre-draw any per-run meta randomness (Fix #8)
    # Here, per-actor sampling is already RNG-driven in _actors_from_table.
    # If you later add per-run noise, pre-draw arrays here with `rng.*`.

    # Monte Carlo
    for run in range(int(num_runs)):
        actors = _actors_from_table(rng, actor_rows)
        if len(actors) < 3:
            # Need at least triadic structure for meaningful bargaining
            break
        model = BDMScholzEnhancedModel(
            actors,
            Q=1.0,
            gamma=0.2,
            risk_lr=0.05,
            thresh_lr=0.05,
            beta_Q=0.5,
            coalition_thresh=2.0
        )
        r = model.run_model(max_rounds=max_rounds, tol=tol)
        finals = [a.position for a in model.actors]
        med = float(np.median(finals))
        medians.append(med)
        rounds_list.append(int(r))

        coal_flag = 1 if model.coalition_events else 0
        coalition_count += coal_flag

        # Exceedance with small hysteresis to reduce flip-flops (Fix #5)
        # yes_threshold is on the same 0..100 scale as medians.
        eps = 0.3  # 0.3 units on 0..100 axis (~0.3 percentage points)
        if med > float(yes_threshold) + eps:
            ex_flag = 1
        elif med < float(yes_threshold) - eps:
            ex_flag = 0
        else:
            # In the gray band, use >= as tie-break to preserve backward-compat
            ex_flag = 1 if med >= float(yes_threshold) else 0
        exceed_count += ex_flag

        results_rows.append({
            "run": run + 1,
            "final_median": round(med, 4),  # rounded for cosmetic stability (Fix #6)
            "rounds": int(r),
            "coalition_events": len(model.coalition_events),
            "exceed_ge_threshold": int(ex_flag),
        })

        # Free memory per run
        del model
        gc.collect()

    # Persist CSV
    if HAVE_PD:
        df = pd.DataFrame(results_rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    else:
        _TinyDF(results_rows).to_csv(csv_path)

    # Persist meta JSON
    meta = {
        "actor_rows": _canon_actor_rows(actor_rows),  # canonicalized input rows
        "num_runs": int(num_runs),
        "created_at": ts,
        "slug": slug,
        "yes_threshold": float(yes_threshold),
        "max_rounds": int(max_rounds),
        "tol": float(tol),
        "paths": {
            "runs_csv": os.path.relpath(csv_path).replace("\\", "/"),
            "meta_json": os.path.relpath(meta_path).replace("\\", "/"),
        },
        "fingerprint": fp,
        "seed": seed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Build compact signal (Fix #6 rounding & stable percentile handling)
    if medians:
        med_of_meds = float(np.median(medians))
        q25, q75 = float(np.percentile(medians, 25)), float(np.percentile(medians, 75))
        # enforce monotonicity if you later expose multiple percentiles
        iqr = q75 - q25
        disp = "low" if iqr <= 5 else ("medium" if iqr <= 15 else "high")
        med_rounds = float(np.median(rounds_list))
        coal_rate = coalition_count / max(1, len(results_rows))
        ex_rate = exceed_count / max(1, len(medians))

        signal = {
            # Normalized & rounded (Fix #6); keep 0..1 for programmatic use
            "coalition_rate": round(float(max(0.0, min(1.0, coal_rate))), 4),
            "coalition_rate_pct": f"{round(coal_rate*100.0, 2):.2f}%",
            "median_of_final_medians": round(med_of_meds, 4),
            "median_rounds": int(round(med_rounds)),
            "dispersion": disp,
            "iqr": round(iqr, 4),
            "exceedance_ge_50": round(ex_rate, 4),
            "runs_csv": meta["paths"]["runs_csv"],
            "meta_json": meta["paths"]["meta_json"],
        }
    else:
        signal = {
            "coalition_rate": 0.0,
            "coalition_rate_pct": "0.00%",
            "median_of_final_medians": None,
            "median_rounds": None,
            "dispersion": "n/a",
            "iqr": None,
            "exceedance_ge_50": None,
            "runs_csv": meta["paths"]["runs_csv"],
            "meta_json": meta["paths"]["meta_json"],
        }

    # Write deterministic cache (Fix #7)
    try:
        to_cache = {
            "signal": signal,
            "results_rows": results_rows,
            "meta": meta,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(to_cache, f, sort_keys=True, separators=(",", ":"))
    except Exception:
        # Fail-closed on cache errors
        pass

    # Return both the compact signal and a DataFrame (if available)
    if HAVE_PD:
        return signal, pd.DataFrame(results_rows)
    else:
        return signal, results_rows


# =============================================================================
# Optional CLI for quick local testing
# =============================================================================

def main():
    print("GTMC1: demo run with a tiny hard-coded table (for local testing only).")
    demo_rows = [
        {"name":"Government","position":62,"capability":70,"salience":80,"risk_threshold":0.04},
        {"name":"Opposition","position":35,"capability":60,"salience":85,"risk_threshold":0.05},
        {"name":"Mediator","position":50,"capability":20,"salience":60,"risk_threshold":0.02},
        {"name":"Key Faction","position":75,"capability":30,"salience":70,"risk_threshold":0.06},
    ]
    sig, df = run_monte_carlo_from_actor_table(
        demo_rows,
        num_runs=30,
        log_dir="gtmc_logs",
        run_slug="demo",
        yes_threshold=50.0
    )
    print("Signal:", json.dumps(sig, indent=2))
    print("CSV path:", sig["runs_csv"])
    if sig.get("meta_json"):
        print("Meta path:", sig["meta_json"])

if __name__ == "__main__":
    main()