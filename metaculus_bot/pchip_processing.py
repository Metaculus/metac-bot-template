from __future__ import annotations

"""
PCHIP CDF processing utilities for numeric forecasting.

Extracted from main.py to improve testability and maintainability.
Contains logic for PCHIP CDF generation, validation, and smoothing.
"""

import logging
from typing import List, Optional

import numpy as np
from forecasting_tools import NumericDistribution
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP, NUM_RAMP_K_FACTOR

from .numeric_config import CDF_RAMP_K_FACTOR, MAX_CDF_PROB_STEP, MIN_CDF_PROB_STEP, PCHIP_CDF_POINTS

logger = logging.getLogger(__name__)


# Module-level counters for PCHIP enforcement statistics (per run)
_pchip_stats = {
    "total_attempts": 0,
    "successful_without_enforcement": 0,
    "required_aggressive_enforcement": 0,
    "failed_entirely": 0,
}


def reset_pchip_stats() -> None:
    """Reset PCHIP statistics counters (call at start of each run)."""
    global _pchip_stats
    _pchip_stats = {
        "total_attempts": 0,
        "successful_without_enforcement": 0,
        "required_aggressive_enforcement": 0,
        "failed_entirely": 0,
    }


def get_pchip_stats() -> dict:
    """Get current PCHIP statistics."""
    return _pchip_stats.copy()


def log_pchip_summary() -> None:
    """Log comprehensive PCHIP enforcement statistics."""
    stats = get_pchip_stats()
    if stats["total_attempts"] == 0:
        logger.info("PCHIP Summary: No PCHIP attempts in this run")
        return

    success_rate = 100.0 * stats["successful_without_enforcement"] / stats["total_attempts"]
    enforcement_rate = 100.0 * stats["required_aggressive_enforcement"] / stats["total_attempts"]
    failure_rate = 100.0 * stats["failed_entirely"] / stats["total_attempts"]

    logger.info(
        "PCHIP Summary | total_attempts=%d | successful_without_enforcement=%d (%.1f%%) | required_aggressive_enforcement=%d (%.1f%%) | failed_entirely=%d (%.1f%%)",
        stats["total_attempts"],
        stats["successful_without_enforcement"],
        success_rate,
        stats["required_aggressive_enforcement"],
        enforcement_rate,
        stats["failed_entirely"],
        failure_rate,
    )


def generate_pchip_cdf_with_smoothing(
    percentile_list: List[Percentile], question: NumericQuestion, zero_point: Optional[float]
) -> tuple[List[float], bool, bool]:
    """
    Generate PCHIP CDF with optional ramp smoothing.

    Args:
        percentile_list: List of percentiles to process
        question: NumericQuestion with bounds information
        zero_point: Zero point for log scaling (None for linear)

    Returns:
        Tuple of (pchip_cdf_values, smoothing_applied, aggressive_enforcement_used)

    Raises:
        ValueError: If PCHIP CDF generation fails validation
    """
    from metaculus_bot.pchip_cdf import generate_pchip_cdf, percentiles_to_pchip_format

    # Track attempt
    global _pchip_stats
    _pchip_stats["total_attempts"] += 1

    # Convert percentiles to PCHIP input format
    pchip_percentiles = percentiles_to_pchip_format(percentile_list)

    try:
        # Generate robust CDF using PCHIP interpolation
        pchip_cdf, aggressive_enforcement_used = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=zero_point,
            min_step=NUM_MIN_PROB_STEP,
            num_points=PCHIP_CDF_POINTS,
            question_id=getattr(question, "id_of_question", None),
            question_url=getattr(question, "page_url", None),
        )

        # Track success type
        if aggressive_enforcement_used:
            _pchip_stats["required_aggressive_enforcement"] += 1
        else:
            _pchip_stats["successful_without_enforcement"] += 1

    except (ValueError, RuntimeError) as e:
        _pchip_stats["failed_entirely"] += 1
        raise

    # Apply probability-side ramp smoothing if needed
    smoothing_applied = False
    try:
        smoothing_applied = _apply_ramp_smoothing(pchip_cdf, question)
    except Exception as smooth_e:
        logger.error("Ramp smoothing skipped due to error: %s", smooth_e)

    # Validate the generated CDF
    _validate_pchip_cdf(pchip_cdf, question)

    # Log success
    _log_pchip_success(pchip_cdf, question, smoothing_applied)

    return pchip_cdf, smoothing_applied, aggressive_enforcement_used


def _apply_ramp_smoothing(pchip_cdf: List[float], question: NumericQuestion) -> bool:
    """
    Apply ramp smoothing to enforce minimum step size.

    Args:
        pchip_cdf: CDF values to potentially smooth (modified in place)
        question: NumericQuestion for bounds information

    Returns:
        True if smoothing was applied, False otherwise
    """
    diffs_before = np.diff(pchip_cdf)
    min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0

    if min_delta_before < NUM_MIN_PROB_STEP:
        # Apply ramp smoothing
        ramp = np.linspace(0.0, NUM_MIN_PROB_STEP * NUM_RAMP_K_FACTOR, len(pchip_cdf))
        pchip_cdf[:] = np.maximum.accumulate(np.array(pchip_cdf) + ramp).tolist()

        # Re-pin endpoints to respect open/closed bounds semantics
        if not question.open_lower_bound:
            pchip_cdf[0] = 0.0
        else:
            pchip_cdf[0] = max(pchip_cdf[0], 0.001)
        if not question.open_upper_bound:
            pchip_cdf[-1] = 1.0
        else:
            pchip_cdf[-1] = min(pchip_cdf[-1], 0.999)

        # Log the smoothing
        diffs_after = np.diff(pchip_cdf)
        min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
        logger.warning(
            "CDF ramp smoothing for Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f | k_factor=%.1f",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            min_delta_before,
            min_delta_after,
            NUM_RAMP_K_FACTOR,
        )
        return True

    return False


def _validate_pchip_cdf(pchip_cdf: List[float], question: NumericQuestion) -> None:
    """
    Validate PCHIP CDF meets all requirements.

    Args:
        pchip_cdf: CDF values to validate
        question: NumericQuestion for bounds validation

    Raises:
        ValueError: If any validation fails
    """
    # Check point count
    if len(pchip_cdf) != PCHIP_CDF_POINTS:
        raise ValueError(f"PCHIP CDF has {len(pchip_cdf)} points, expected {PCHIP_CDF_POINTS}")

    # Check probability range
    if not all(0.0 <= p <= 1.0 for p in pchip_cdf):
        invalid_probs = [p for p in pchip_cdf if not (0.0 <= p <= 1.0)]
        raise ValueError(f"PCHIP CDF contains invalid probabilities outside [0,1]: {invalid_probs}")

    # Check monotonicity
    if not all(a <= b for a, b in zip(pchip_cdf[:-1], pchip_cdf[1:])):
        raise ValueError("PCHIP CDF is not monotonic")

    # Check minimum step requirement
    min_step = np.min(np.diff(pchip_cdf))
    if min_step < NUM_MIN_PROB_STEP - 1e-10:
        raise ValueError(f"PCHIP CDF violates minimum step requirement: {min_step:.8f} < 5e-5")

    # Check maximum step requirement
    max_step = np.max(np.diff(pchip_cdf))
    if max_step > NUM_MAX_STEP + 1e-6:
        raise ValueError(f"PCHIP CDF violates maximum step requirement: {max_step:.8f} > 0.59")

    # Check boundary conditions
    if not question.open_lower_bound and abs(pchip_cdf[0]) > 1e-6:
        raise ValueError(f"PCHIP CDF closed lower bound violation: {pchip_cdf[0]} != 0.0")

    if not question.open_upper_bound and abs(pchip_cdf[-1] - 1.0) > 1e-6:
        raise ValueError(f"PCHIP CDF closed upper bound violation: {pchip_cdf[-1]} != 1.0")

    if question.open_lower_bound and pchip_cdf[0] < 0.001:
        raise ValueError(f"PCHIP CDF open lower bound violation: {pchip_cdf[0]} < 0.001")

    if question.open_upper_bound and pchip_cdf[-1] > 0.999:
        raise ValueError(f"PCHIP CDF open upper bound violation: {pchip_cdf[-1]} > 0.999")


def _log_pchip_success(pchip_cdf: List[float], question: NumericQuestion, smoothing_applied: bool) -> None:
    """
    Log successful PCHIP CDF generation.

    Args:
        pchip_cdf: Generated CDF values
        question: NumericQuestion for logging context
        smoothing_applied: Whether smoothing was applied
    """
    min_step = np.min(np.diff(pchip_cdf))
    max_step = np.max(np.diff(pchip_cdf))

    logger.info(
        "PCHIP OK for Q %s | points=%d | min_step=%.8f | max_step=%.8f | smoothing=%s | open_bounds=(%s,%s)",
        getattr(question, "id_of_question", "N/A"),
        len(pchip_cdf),
        min_step,
        max_step,
        smoothing_applied,
        question.open_lower_bound,
        question.open_upper_bound,
    )


def create_pchip_numeric_distribution(
    pchip_cdf: List[float], percentile_list: List[Percentile], question: NumericQuestion, zero_point: Optional[float]
) -> NumericDistribution:
    """
    Create a custom NumericDistribution that uses PCHIP CDF.

    Args:
        pchip_cdf: Generated PCHIP CDF values
        percentile_list: Original percentile list
        question: NumericQuestion with bounds
        zero_point: Zero point for scaling

    Returns:
        NumericDistribution with PCHIP CDF
    """

    class PchipNumericDistribution(NumericDistribution):
        def __init__(self, pchip_cdf_values, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pchip_cdf_values = pchip_cdf_values

        @property
        def cdf(self) -> list[Percentile]:
            """Return PCHIP-generated CDF as Percentile objects."""
            # Create the value axis (201 points from lower to upper bound)
            x_vals = np.linspace(self.lower_bound, self.upper_bound, len(self._pchip_cdf_values))

            # Create Percentile objects with correct mapping:
            # _pchip_cdf_values contains the probability values (0-1)
            # x_vals contains the corresponding question values
            return [
                Percentile(percentile=prob_val, value=question_val)
                for question_val, prob_val in zip(x_vals, self._pchip_cdf_values)
            ]

    return PchipNumericDistribution(
        pchip_cdf_values=pchip_cdf,
        declared_percentiles=percentile_list,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=zero_point,
        cdf_size=getattr(question, "cdf_size", None),
    )


def create_fallback_numeric_distribution(
    percentile_list: List[Percentile], question: NumericQuestion, zero_point: Optional[float]
) -> NumericDistribution:
    """
    Create fallback NumericDistribution when PCHIP fails.

    Args:
        percentile_list: List of percentiles
        question: NumericQuestion with bounds
        zero_point: Zero point for scaling

    Returns:
        Standard NumericDistribution
    """
    return NumericDistribution(
        declared_percentiles=percentile_list,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=zero_point,
        cdf_size=getattr(question, "cdf_size", None),
    )
