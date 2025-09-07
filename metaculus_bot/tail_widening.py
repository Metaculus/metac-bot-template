"""
Tail widening utilities for numeric declared percentiles.

Implements transform-space scaling around the median to fatten tails while preserving
bounds and monotonicity. Optionally enforces a span floor at the extreme tail percentiles.
"""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.cluster_processing import ensure_strictly_increasing_bounded

logger = logging.getLogger(__name__)


def _sigmoid(y: float) -> float:
    return 1.0 / (1.0 + math.exp(-y))


def _logit(u: float) -> float:
    return math.log(u / (1.0 - u))


def _choose_transform(
    question: NumericQuestion,
    eps: float,
):
    """
    Choose forward/inverse transforms based on bound semantics.

    - Both bounds closed: use bounded logit on normalized u.
    - Lower closed, upper open: lower-bounded → log(x - L + eps).
    - Lower open, upper closed: upper-bounded → -log(U - x + eps) (monotone in x).
    - Both open: identity.
    """

    L = float(getattr(question, "lower_bound", 0.0))
    U = float(getattr(question, "upper_bound", 0.0))
    open_low = bool(getattr(question, "open_lower_bound", False))
    open_up = bool(getattr(question, "open_upper_bound", False))

    if not open_low and not open_up:
        # Bounded both sides → normalize and use logit
        rng = max(U - L, eps)

        def fwd(x: float) -> float:
            u = (x - L) / rng
            u = max(1e-12, min(1.0 - 1e-12, u))
            return _logit(u)

        def inv(y: float) -> float:
            u = _sigmoid(y)
            return L + rng * u

        return fwd, inv

    if not open_low and open_up:
        # Lower-bounded → y = log(x - L + eps)
        def fwd(x: float) -> float:
            return math.log(max(x - L + eps, 1e-18))

        def inv(y: float) -> float:
            return L - eps + math.exp(y)

        return fwd, inv

    if open_low and not open_up:
        # Upper-bounded → y = -log(U - x + eps), monotone increasing in x
        def fwd(x: float) -> float:
            return -math.log(max(U - x + eps, 1e-18))

        def inv(y: float) -> float:
            return U + eps - math.exp(-y)

        return fwd, inv

    # Both open → identity
    return (lambda x: x), (lambda y: y)


def _tail_weight(p: float, tail_start: float) -> float:
    """
    Linear tail ramp weight in [0,1]: 0 at center, 1 at deepest tails.
    No widening for p in [0.5 - (0.5 - tail_start), 0.5 + (0.5 - tail_start)].
    """
    t = abs(p - 0.5)
    no_widen_zone = 0.5 - tail_start
    if t <= no_widen_zone:
        return 0.0
    # Linear ramp from no_widen_zone → 0.5
    return min(1.0, (t - no_widen_zone) / tail_start)


def widen_declared_percentiles(
    percentile_list: List[Percentile],
    question: NumericQuestion,
    *,
    k_tail: float = 1.25,
    tail_start: float = 0.2,
    span_floor_gamma: float = 1.0,
) -> List[Percentile]:
    """
    Widen tails by scaling distances from the median in a transformed space.

    Parameters
    ----------
    percentile_list: list of Percentile (assumed sorted by percentile and strictly increasing values)
    question: NumericQuestion with bound semantics
    k_tail: maximum stretch factor at deepest tails in transformed space (>=1.0)
    tail_start: tail ramp start (fraction of percentile distance from the median)
    span_floor_gamma: enforce (p05 - p02.5) >= gamma*(p10 - p05) and (p97.5 - p95) >= gamma*(p95 - p90)
    """

    # If no percentiles, or both widening and span-floor disabled, bail out
    if not percentile_list or (k_tail <= 1.0 and span_floor_gamma <= 0.0):
        return percentile_list

    L = float(getattr(question, "lower_bound", -np.inf))
    U = float(getattr(question, "upper_bound", np.inf))
    rng = max(U - L, 1e-12)

    eps = max(1e-12, rng * 1e-12)
    fwd, inv = _choose_transform(question, eps)

    # Build arrays in percentile order
    p_vals = np.array([float(p.percentile) for p in percentile_list], dtype=float)
    x_vals = np.array([float(p.value) for p in percentile_list], dtype=float)

    # Transform and find median in transformed space
    y_vals = np.array([fwd(x) for x in x_vals], dtype=float)

    # Locate median index by p=0.5; if not present, interpolate y_median between neighbors
    if any(abs(p - 0.5) < 1e-12 for p in p_vals):
        y_m = float(y_vals[np.argmin(np.abs(p_vals - 0.5))])
    else:
        # Simple linear interpolation in p-space for y
        y_m = float(np.interp(0.5, p_vals, y_vals))

    # Apply tail ramp widening
    k_delta = max(0.0, k_tail - 1.0)
    widened_y = []
    for p, y in zip(p_vals, y_vals):
        w = _tail_weight(p, tail_start)
        k_eff = 1.0 + k_delta * w
        widened_y.append(y_m + k_eff * (y - y_m))
    widened_y_arr = np.array(widened_y, dtype=float)

    # Inverse transform back to x-space (or keep original if k_tail<=1)
    if k_tail > 1.0:
        widened_x = np.array([inv(y) for y in widened_y_arr], dtype=float)
    else:
        widened_x = x_vals.copy()

    # Clamp to numeric bounds
    widened_x = np.clip(widened_x, L, U)

    # Enforce span floors at extremes if available
    # Require presence of 2.5, 5, 10 and 90, 95, 97.5
    def _find_index(target: float) -> int | None:
        idxs = np.where(np.isclose(p_vals, target, atol=5e-6))[0]
        return int(idxs[0]) if len(idxs) else None

    i025 = _find_index(0.025)
    i05 = _find_index(0.05)
    i10 = _find_index(0.10)
    i90 = _find_index(0.90)
    i95 = _find_index(0.95)
    i975 = _find_index(0.975)

    if span_floor_gamma > 0 and None not in (i025, i05, i10):
        inner = max(0.0, widened_x[i10] - widened_x[i05])
        target_span = span_floor_gamma * inner
        current = widened_x[i05] - widened_x[i025]
        if target_span > current + 1e-15:
            widened_x[i025] = max(L, widened_x[i05] - target_span)

    if span_floor_gamma > 0 and None not in (i90, i95, i975):
        inner = max(0.0, widened_x[i95] - widened_x[i90])
        target_span = span_floor_gamma * inner
        current = widened_x[i975] - widened_x[i95]
        if target_span > current + 1e-15:
            widened_x[i975] = min(U, widened_x[i95] + target_span)

    # Ensure outer spans do not shrink relative to baseline when widening is requested
    if k_tail > 1.0:
        if None not in (i025, i05):
            base_low_outer = x_vals[i05] - x_vals[i025]
            new_low_outer = widened_x[i05] - widened_x[i025]
            if new_low_outer + 1e-15 < base_low_outer:
                widened_x[i025] = max(L, widened_x[i05] - base_low_outer)
        if None not in (i95, i975):
            base_up_outer = x_vals[i975] - x_vals[i95]
            new_up_outer = widened_x[i975] - widened_x[i95]
            if new_up_outer + 1e-15 < base_up_outer:
                widened_x[i975] = min(U, widened_x[i95] + base_up_outer)

    # Ensure strictly increasing and within bounds
    updated = widened_x.tolist()

    # A final gentle pass to guarantee strict monotonicity and bound proximity
    range_size = rng
    updated = ensure_strictly_increasing_bounded(updated, question, range_size)

    # For open bounds, nudge tails away from exact bounds to avoid near-duplicates against range
    open_low = bool(getattr(question, "open_lower_bound", False))
    open_up = bool(getattr(question, "open_upper_bound", False))
    # Use a modest floor to avoid unit-mismatch detector (relative to range)
    value_floor = max(range_size * 1e-6, 1e-8)
    if open_low:
        updated[0] = max(updated[0], L + value_floor)
        if len(updated) >= 2:
            updated[1] = max(updated[1], updated[0] + value_floor)
    if open_up:
        updated[-1] = min(updated[-1], U - value_floor)
        if len(updated) >= 2:
            updated[-2] = min(updated[-2], updated[-1] - value_floor)
    # Re-ensure monotonic after nudging
    updated = ensure_strictly_increasing_bounded(updated, question, range_size)

    # Final safety: clamp and enforce monotonicity within [L, U] for semi-bounded cases
    updated = np.clip(updated, L, U).tolist()
    # Build a small spacing schedule so we never exceed U while preserving order
    if len(updated) >= 2:
        # Forward pass: ensure each is at least value_floor above previous
        for i in range(1, len(updated)):
            min_allowed = updated[i - 1] + value_floor
            max_allowed = U - value_floor * (len(updated) - 1 - i) if open_up else U
            if updated[i] < min_allowed:
                updated[i] = min(max_allowed, min_allowed)
        # Backward pass: ensure each is at most value_floor below next
        for i in range(len(updated) - 2, -1, -1):
            max_allowed = updated[i + 1] - value_floor
            min_allowed = L + value_floor * i if open_low else L
            if updated[i] > max_allowed:
                updated[i] = max(min_allowed, max_allowed)

    # Rebuild Percentile objects preserving original percentiles
    result: list[Percentile] = [Percentile(value=float(v), percentile=float(p)) for v, p in zip(updated, p_vals)]

    try:
        # Quick sanity: monotone increasing
        deltas = np.diff([pp.value for pp in result])
        if not np.all(deltas > -1e-12):
            logger.warning(
                "Tail widening produced non-monotone sequence; enforced correction applied | Q=%s",
                getattr(question, "id_of_question", None),
            )
    except Exception:
        pass

    return result
