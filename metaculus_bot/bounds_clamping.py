from __future__ import annotations

"""
Bounds clamping utilities for numeric percentile processing.

Extracted from main.py to improve testability and maintainability.
Contains logic for clamping values to bounds with appropriate tolerances.
"""

import logging
from typing import List, Tuple

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from .numeric_config import BOUNDARY_SAFETY_MARGIN, MIN_BOUNDARY_DISTANCE

logger = logging.getLogger(__name__)


def calculate_bounds_buffer(question: NumericQuestion) -> float:
    """
    Calculate buffer for bounds clamping based on question range.

    Args:
        question: NumericQuestion to calculate buffer for

    Returns:
        Buffer size for bounds clamping
    """
    range_size = question.upper_bound - question.lower_bound
    return 1.0 if range_size > 100 else range_size * BOUNDARY_SAFETY_MARGIN


def clamp_values_to_bounds(
    modified_values: List[float], percentile_list: List[Percentile], question: NumericQuestion, buffer: float
) -> Tuple[List[float], bool]:
    """
    Clamp values to bounds if they violate by small amounts.

    Args:
        modified_values: List of values to clamp (modified in place)
        percentile_list: Original percentile list for logging context
        question: NumericQuestion with bounds information
        buffer: Buffer size for clamping tolerance

    Returns:
        Tuple of (modified_values, corrections_made)
    """
    corrections_made = False

    for i in range(len(modified_values)):
        original_value = modified_values[i]

        # Check lower bound violation
        if not question.open_lower_bound and modified_values[i] < question.lower_bound:
            if question.lower_bound - modified_values[i] <= buffer:
                modified_values[i] = question.lower_bound + buffer
                corrections_made = True
                logger.debug(
                    f"Clamped lower: percentile {percentile_list[i].percentile} value {original_value} -> {modified_values[i]}"
                )
            else:
                raise ValueError(
                    f"Value {original_value} too far below lower bound {question.lower_bound} (tolerance: {buffer})"
                )

        # Check upper bound violation
        if not question.open_upper_bound and modified_values[i] > question.upper_bound:
            if modified_values[i] - question.upper_bound <= buffer:
                modified_values[i] = question.upper_bound - buffer
                corrections_made = True
                logger.debug(
                    f"Clamped upper: percentile {percentile_list[i].percentile} value {original_value} -> {modified_values[i]}"
                )
            else:
                raise ValueError(
                    f"Value {original_value} too far above upper bound {question.upper_bound} (tolerance: {buffer})"
                )

    return modified_values, corrections_made


def log_heavy_clamping_diagnostics(
    modified_values: List[float], original_values: List[float], question: NumericQuestion, buffer: float
) -> None:
    """
    Log diagnostics if too many values were clamped to bounds.

    Args:
        modified_values: Final processed values
        original_values: Original input values
        question: NumericQuestion for context
        buffer: Buffer size used for clamping
    """
    if not original_values:
        return

    tol = MIN_BOUNDARY_DISTANCE
    clamped_lower = sum(
        1 for v in modified_values if not question.open_lower_bound and v <= question.lower_bound + buffer + tol
    )
    clamped_upper = sum(
        1 for v in modified_values if not question.open_upper_bound and v >= question.upper_bound - buffer - tol
    )

    # Warn if more than 50% of values were clamped to either bound
    if clamped_lower / len(original_values) > 0.5 or clamped_upper / len(original_values) > 0.5:
        logger.warning(
            "Heavy bound clamping for Q %s | URL %s | clamped_to_lower=%d%% | clamped_to_upper=%d%% | bounds=[%s, %s]",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            int(100 * clamped_lower / len(original_values)),
            int(100 * clamped_upper / len(original_values)),
            question.lower_bound,
            question.upper_bound,
        )


def log_corrections_summary(
    modified_values: List[float], original_values: List[float], question: NumericQuestion, corrections_made: bool
) -> None:
    """
    Log summary of corrections made to the distribution.

    Args:
        modified_values: Final processed values
        original_values: Original input values
        question: NumericQuestion for context
        corrections_made: Whether bounds corrections were made
    """
    if corrections_made or any(v != orig for v, orig in zip(modified_values, original_values)):
        logger.warning(f"Corrected numeric distribution for question {getattr(question, 'id_of_question', 'N/A')}")


def log_cluster_spreading_summary(
    modified_values: List[float],
    original_values: List[float],
    question: NumericQuestion,
    clusters_applied: int,
    spread_delta: float,
    count_like: bool,
) -> None:
    """
    Log summary of cluster spreading operations.

    Args:
        modified_values: Final processed values after spreading
        original_values: Original values before spreading
        question: NumericQuestion for context
        clusters_applied: Number of clusters that were spread
        spread_delta: Delta used for spreading
        count_like: Whether distribution was detected as count-like
    """
    if clusters_applied > 0:
        # Compute pre- and post-spread min deltas for logging
        pre_deltas = [b - a for a, b in zip(original_values, original_values[1:])]
        post_deltas = [b - a for a, b in zip(modified_values, modified_values[1:])]

        min_value_delta_before = min(pre_deltas) if pre_deltas else float("inf")
        min_value_delta_after = min(post_deltas) if post_deltas else float("inf")

        logger.warning(
            "Cluster spread applied for Q %s | URL %s | clusters=%d | delta_used=%.6g | min_value_delta_before=%.6g | min_value_delta_after=%.6g | count_like=%s",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            clusters_applied,
            spread_delta,
            min_value_delta_before,
            min_value_delta_after,
            count_like,
        )
