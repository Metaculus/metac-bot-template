"""
PCHIP-based CDF construction for robust numeric forecasting.

Based on the battle-tested implementation from panchul (Q2 2025 competition winner).
Provides smooth, monotonic CDF construction with strict constraints enforcement.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


def _safe_cdf_bounds(cdf: np.ndarray, open_lower: bool, open_upper: bool, min_step: float) -> np.ndarray:
    """
    Ensure CDF respects Metaculus boundary constraints:
    • For *open* bounds: cdf[0] ≥ 0.001, cdf[-1] ≤ 0.999
    • No single step may exceed 0.59
    """
    # Pin tails to legal open-bound limits
    if open_lower:
        cdf[0] = max(cdf[0], 0.001)
    if open_upper:
        cdf[-1] = min(cdf[-1], 0.999)

    # Enforce the 0.59 maximum step rule
    big_jumps = np.where(np.diff(cdf) > 0.59)[0]
    for idx in big_jumps:
        excess = cdf[idx + 1] - cdf[idx] - 0.59
        # Spread the excess evenly over the remaining points
        span = len(cdf) - idx - 1
        if span > 0:
            cdf[idx + 1 :] -= excess * np.linspace(1, 0, span)
        # Re-monotonise
        cdf[idx + 1 :] = np.maximum.accumulate(cdf[idx + 1 :])

    return cdf


def enforce_strict_increasing(percentile_dict: Dict[Union[int, float], float]) -> Dict[Union[int, float], float]:
    """Ensure strictly increasing values by adding tiny jitter if necessary."""
    sorted_items = sorted(percentile_dict.items())
    last_val = -float("inf")
    new_dict = {}

    for p, v in sorted_items:
        if v <= last_val:
            v = last_val + 1e-8  # Add a tiny epsilon
        new_dict[p] = v
        last_val = v

    return new_dict


def generate_pchip_cdf(
    percentile_values: Dict[Union[int, float], float],
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: Optional[float] = None,
    *,
    min_step: float = 5.0e-5,
    num_points: int = 201,
    question_id: Optional[Union[int, str]] = None,
    question_url: Optional[str] = None,
) -> tuple[List[float], bool]:
    """
    Generate a robust continuous CDF using PCHIP interpolation with strict constraint enforcement.

    Based on the panchul implementation with enhancements for robustness.

    Args:
        percentile_values: Dictionary mapping percentiles (0-100) to values
        open_upper_bound: Whether the upper bound is open
        open_lower_bound: Whether the lower bound is open
        upper_bound: Maximum possible value
        lower_bound: Minimum possible value
        zero_point: Reference point for non-linear scaling (optional)
        min_step: Minimum step size between adjacent CDF points (default: 5.0e-5)
        num_points: Number of points in the output CDF (default: 201)
        question_id: Optional question identifier for logging context
        question_url: Optional question URL for logging context

    Returns:
        Tuple of (CDF values, aggressive_enforcement_used) where:
        - CDF values: List of probability values with strictly enforced monotonicity and step size
        - aggressive_enforcement_used: True if aggressive step enforcement was required

    Raises:
        ValueError: If input validation fails
        RuntimeError: If constraint enforcement fails
    """
    # Validate inputs
    if not percentile_values:
        raise ValueError("Empty percentile values dictionary")

    if upper_bound <= lower_bound:
        raise ValueError(f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})")

    if zero_point is not None:
        if abs(zero_point - lower_bound) < 1e-6 or abs(zero_point - upper_bound) < 1e-6:
            raise ValueError(f"zero_point ({zero_point}) too close to bounds [{lower_bound}, {upper_bound}]")

    # Clean and validate percentile values
    pv = {}
    for k, v in percentile_values.items():
        try:
            k_float = float(k)
            v_float = float(v)

            if not (0 < k_float < 100):
                continue  # Skip invalid percentiles

            if not np.isfinite(v_float):
                continue  # Skip non-finite values

            pv[k_float] = v_float
        except (ValueError, TypeError):
            continue  # Skip non-numeric entries

    if len(pv) < 2:
        raise ValueError(f"Need at least 2 valid percentile points (got {len(pv)})")

    # Handle duplicate values by adding small offsets
    # First, sort all items to process in order
    sorted_items = sorted(pv.items())
    last_value = -float("inf")

    for k, v in sorted_items:
        if v <= last_value:
            # Add a small epsilon to ensure strictly increasing
            v = last_value + 1e-9
        pv[k] = v
        last_value = v

    # Create arrays of percentiles and values
    percentiles, values = zip(*sorted(pv.items()))
    percentiles = np.array(percentiles) / 100.0  # Convert to [0,1] range
    values = np.array(values)

    # Check if values are strictly increasing after de-duplication
    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de-duplication")

    # Add boundary points if needed
    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values = np.insert(values, 0, lower_bound)

    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values = np.append(values, upper_bound)

    # Determine if log scaling is appropriate (all values positive and lower bound > 0)
    use_log = np.all(values > 0) and zero_point is None and lower_bound > 0
    x_vals = np.log(values) if use_log else values

    # Create interpolator with fallback
    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception as e:
        # Fallback to linear interpolation
        print(f"PchipInterpolator failed ({str(e)}), falling back to linear interpolation")
        spline = lambda x: np.interp(x, x_vals, percentiles)

    # Generate evaluation grid based on zero_point
    def create_grid(num_points: int) -> np.ndarray:
        t = np.linspace(0, 1, num_points)

        if zero_point is None:
            # Linear grid
            return lower_bound + (upper_bound - lower_bound) * t
        else:
            # Non-linear grid based on zero_point
            ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
            # Handle potential numerical issues
            if abs(ratio - 1.0) < 1e-10:
                return lower_bound + (upper_bound - lower_bound) * t
            else:
                return np.array(
                    [lower_bound + (upper_bound - lower_bound) * ((ratio**tt - 1) / (ratio - 1)) for tt in t]
                )

    # Generate the grid and evaluate
    cdf_x = create_grid(num_points)

    # Handle log transformation for evaluation
    eval_x = np.log(cdf_x) if use_log else cdf_x

    # Clamp values to avoid extrapolation issues
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])

    # Generate initial CDF values and clamp to [0,1]
    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)

    # Ensure monotonicity (non-decreasing)
    cdf_y = np.maximum.accumulate(cdf_y)

    # Set boundary values if bounds are closed
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0

    # Strict enforcement of minimum step size with iterative approach
    def enforce_min_steps(y_values: np.ndarray, min_step_size: float) -> np.ndarray:
        """Enforce minimum step size between adjacent points"""
        result = y_values.copy()

        # First pass: enforce minimum steps
        for i in range(1, len(result)):
            if result[i] < result[i - 1] + min_step_size:
                result[i] = min(result[i - 1] + min_step_size, 1.0)

        # Second pass: ensure we don't exceed 1.0
        if result[-1] > 1.0:
            # If we've exceeded 1.0 before the end, rescale the steps
            overflow_idx = np.where(result > 1.0)[0]
            if len(overflow_idx) > 0:
                overflow_idx = overflow_idx[0]
                steps_remaining = len(result) - overflow_idx

                for i in range(overflow_idx, len(result)):
                    t = (i - overflow_idx) / max(1, steps_remaining - 1)
                    result[i] = min(1.0, result[overflow_idx - 1] + (1.0 - result[overflow_idx - 1]) * t)

                # Final check for minimum steps
                for i in range(overflow_idx, len(result)):
                    if i > overflow_idx and result[i] < result[i - 1] + min_step_size:
                        result[i] = result[i - 1] + min_step_size
                        if result[i] > 1.0:
                            # If we exceed 1.0 again, cap at 1.0 and adjust previous values
                            result[i] = 1.0
                            # Backtrack and redistribute
                            for j in range(i - 1, overflow_idx - 1, -1):
                                max_allowed = result[j + 1] - min_step_size
                                if result[j] > max_allowed:
                                    result[j] = max_allowed

        return result

    # Apply strict step enforcement
    cdf_y = enforce_min_steps(cdf_y, min_step)

    # Apply boundary constraints and max jump rules
    cdf_y = _safe_cdf_bounds(cdf_y, open_lower_bound, open_upper_bound, min_step)

    # Check if we have enough room for minimum steps
    required_range = (len(cdf_y) - 1) * min_step
    available_range = cdf_y[-1] - cdf_y[0]

    # Double-check minimum step size requirement
    steps = np.diff(cdf_y)
    aggressive_enforcement_used = False
    if np.any(steps < min_step):
        aggressive_enforcement_used = True
        # Log detailed context before aggressive enforcement
        violated_steps = np.sum(steps < min_step)
        min_violated_step = np.min(steps)
        violation_percentage = 100.0 * violated_steps / len(steps)

        logger.warning(
            "PCHIP minimum step enforcement required for Q %s | URL %s | violated_steps=%d/%d (%.1f%%) | min_step_found=%.8f | min_step_required=%.8f | available_range=%.6f | required_range=%.6f",
            question_id or "N/A",
            question_url or "N/A",
            violated_steps,
            len(steps),
            violation_percentage,
            min_violated_step,
            min_step,
            available_range,
            required_range,
        )

        # Create a strictly monotonic sequence
        if not open_lower_bound:
            start_val = 0.0
        else:
            start_val = cdf_y[0]

        if not open_upper_bound:
            end_val = 1.0
        else:
            end_val = min(cdf_y[-1], 1.0)

        available_range = end_val - start_val
        # Ensure we have enough room for all steps
        required_range = (len(cdf_y) - 1) * min_step

        if required_range > available_range:
            # We don't have enough room for minimum steps
            raise ValueError(
                f"Cannot satisfy minimum step requirement: need {required_range:.6f} "
                f"but only have {available_range:.6f} available in CDF range"
            )

        # Create a new CDF with exactly min_step between points where needed
        # and distribute remaining range proportionally
        new_cdf = np.zeros_like(cdf_y)
        new_cdf[0] = start_val

        # Get the shape from original CDF but enforce minimum steps
        if len(cdf_y) > 2:
            # Calculate normalized shape from original CDF
            orig_shape = np.diff(cdf_y)
            orig_shape = np.maximum(orig_shape, min_step)  # Enforce minimum
            orig_shape = orig_shape / np.sum(orig_shape)  # Normalize

            # Allocate the available range according to shape but ensure minimum steps
            remaining = available_range - (len(cdf_y) - 1) * min_step
            extra_steps = remaining * orig_shape

            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i - 1] + min_step + extra_steps[i - 1]
        else:
            # Simple linear spacing if original shape is unavailable
            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i - 1] + (available_range / (len(new_cdf) - 1))

        # Final validation
        if np.any(np.diff(new_cdf) < min_step - 1e-10):
            raise RuntimeError("Internal error: Step size enforcement failed")

        cdf_y = new_cdf

        # Log successful aggressive enforcement
        new_steps = np.diff(cdf_y)
        new_min_step = np.min(new_steps)
        new_max_step = np.max(new_steps)
        total_range_redistributed = available_range

        logger.info(
            "PCHIP aggressive enforcement completed for Q %s | URL %s | new_min_step=%.8f | new_max_step=%.8f | total_range_redistributed=%.6f | shape_preserved=True",
            question_id or "N/A",
            question_url or "N/A",
            new_min_step,
            new_max_step,
            total_range_redistributed,
        )

    # Final checks
    if np.any(np.diff(cdf_y) < min_step - 1e-10):
        problematic_indices = np.where(np.diff(cdf_y) < min_step - 1e-10)[0]
        error_msg = (
            f"Failed to enforce minimum step size at indices: {problematic_indices}, "
            f"values: {np.diff(cdf_y)[problematic_indices]}"
        )
        raise RuntimeError(error_msg)

    if not open_lower_bound and abs(cdf_y[0]) > 1e-10:
        raise RuntimeError(f"Failed to enforce lower bound: {cdf_y[0]}")

    if not open_upper_bound and abs(cdf_y[-1] - 1.0) > 1e-10:
        raise RuntimeError(f"Failed to enforce upper bound: {cdf_y[-1]}")

    return cdf_y.tolist(), aggressive_enforcement_used


def percentiles_to_pchip_format(percentiles: List) -> Dict[float, float]:
    """
    Convert forecasting-tools Percentile objects to PCHIP input format.

    Args:
        percentiles: List of Percentile objects with .percentile and .value attributes

    Returns:
        Dictionary mapping percentile (0-100) to value
    """
    result = {}
    for p in percentiles:
        percentile_key = p.percentile * 100  # Convert from [0,1] to [0,100]
        result[percentile_key] = p.value
    return result
