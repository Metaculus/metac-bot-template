"""
Cluster detection and spreading utilities for numeric percentile processing.

Extracted from main.py to improve testability and maintainability.
Contains logic for detecting clusters of near-equal values and spreading them
to ensure strictly increasing sequences.
"""

import logging
from typing import List, Tuple

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_SPREAD_DELTA_MULT, NUM_VALUE_EPSILON_MULT

from .numeric_config import (
    CLUSTER_DETECTION_ATOL,
    CLUSTER_SPREAD_BASE_DELTA,
    COUNT_LIKE_DELTA_MULTIPLIER,
    COUNT_LIKE_THRESHOLD,
    MIN_BOUNDARY_DISTANCE,
    STRICT_ORDERING_EPSILON,
)

logger = logging.getLogger(__name__)


def detect_count_like_pattern(values: List[float]) -> bool:
    """
    Detect if all values are near integers (count-like pattern).

    Args:
        values: List of numeric values to check

    Returns:
        True if all values are near integers, False otherwise
    """
    try:
        if not values:
            return False
        return all(abs(v - round(v)) <= COUNT_LIKE_THRESHOLD for v in values)
    except Exception:
        return False


def compute_cluster_parameters(
    range_size: float, count_like: bool, span: float | None = None
) -> Tuple[float, float, float]:
    """
    Compute parameters for cluster detection and spreading.

    Args:
        range_size: Range of the question (upper_bound - lower_bound)
        count_like: Whether values follow a count-like pattern

    Returns:
        Tuple of (value_eps, base_delta, spread_delta)
    """
    value_eps = max(range_size * NUM_VALUE_EPSILON_MULT, CLUSTER_DETECTION_ATOL)
    base_delta = max(range_size * NUM_SPREAD_DELTA_MULT, CLUSTER_SPREAD_BASE_DELTA)
    # Prefer a spread relative to the raw span when available to avoid range-driven explosions
    if span is not None and span > 0:
        span_based = max(0.02 * span, CLUSTER_SPREAD_BASE_DELTA)
    else:
        span_based = base_delta
    spread_delta = max(base_delta, span_based, COUNT_LIKE_DELTA_MULTIPLIER if count_like else base_delta)
    return value_eps, base_delta, spread_delta


def apply_cluster_spreading(
    modified_values: List[float],
    question: NumericQuestion,
    value_eps: float,
    spread_delta: float,
    range_size: float,
) -> Tuple[List[float], int]:
    """
    Apply cluster spreading to ensure strictly increasing values.

    Args:
        modified_values: List of values to process (modified in place)
        question: NumericQuestion with bounds information
        value_eps: Epsilon for detecting clusters
        spread_delta: Delta to use for spreading clusters
        range_size: Range of the question

    Returns:
        Tuple of (modified_values, clusters_applied_count)
    """
    clusters_applied = 0
    i = 0

    while i < len(modified_values) - 1:
        j = i
        # Grow cluster while adjacent values within epsilon
        while j + 1 < len(modified_values) and abs(modified_values[j + 1] - modified_values[j]) <= value_eps:
            j += 1

        if j > i:
            # We have a cluster from i..j inclusive
            clusters_applied += 1
            k = j - i + 1

            # Base center value: mean of the cluster
            center = float(np.mean(modified_values[i : j + 1]))

            # Offsets: symmetric around center
            # Example for k=3: -d, 0, +d; for k=4: -1.5d, -0.5d, +0.5d, +1.5d
            offsets = [((idx - (k - 1) / 2.0) * spread_delta) for idx in range(k)]
            new_vals = [center + off for off in offsets]

            # Enforce bounds softly during spread to avoid later large clamps
            tiny = max(MIN_BOUNDARY_DISTANCE * range_size, CLUSTER_DETECTION_ATOL)
            if not question.open_lower_bound:
                new_vals = [max(v, question.lower_bound + tiny) for v in new_vals]
            if not question.open_upper_bound:
                new_vals = [min(v, question.upper_bound - tiny) for v in new_vals]

            # Apply while preserving non-decreasing relation to neighbors
            # If previous value exists and is >= first new, shift all up minimally
            if i - 1 >= 0 and new_vals[0] <= modified_values[i - 1]:
                shift = (modified_values[i - 1] + max(STRICT_ORDERING_EPSILON, value_eps)) - new_vals[0]
                new_vals = [v + shift for v in new_vals]

            # If next value exists and last new exceeds it, compress offsets
            if j + 1 < len(modified_values) and new_vals[-1] >= modified_values[j + 1]:
                # Compress spread to fit in available gap
                available = max(
                    modified_values[j + 1] - (new_vals[0]),
                    max(value_eps, STRICT_ORDERING_EPSILON),
                )
                if k > 1:
                    step = available / k
                    new_vals = [new_vals[0] + step * idx for idx in range(k)]

            # Assign new values
            for t in range(k):
                modified_values[i + t] = new_vals[t]

            i = j + 1
        else:
            i += 1

    return modified_values, clusters_applied


def apply_jitter_for_duplicates(
    modified_values: List[float],
    question: NumericQuestion,
    range_size: float,
    percentile_list: List[Percentile],
) -> List[float]:
    """
    Apply jitter to eliminate any remaining duplicate values.

    Args:
        modified_values: List of values to process (modified in place)
        question: NumericQuestion with bounds information
        range_size: Range of the question
        percentile_list: Original percentile list for logging

    Returns:
        Modified list of values
    """
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
            epsilon = max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)
            target = modified_values[i - 1] + epsilon

            if not question.open_upper_bound:
                target = min(target, question.upper_bound - epsilon)

            # Increase if possible; otherwise allow equality (PCHIP will handle de-dup)
            new_val = max(modified_values[i], target)

            # Also respect lower bound on closed lower
            if not question.open_lower_bound:
                new_val = max(new_val, question.lower_bound + epsilon)

            modified_values[i] = new_val
            logger.debug(
                f"Applied jitter: percentile {percentile_list[i].percentile} value {modified_values[i]} -> {new_val}"
            )

    return modified_values


def ensure_strictly_increasing_bounded(
    modified_values: List[float], question: NumericQuestion, range_size: float
) -> List[float]:
    """
    Final pass to ensure all values are strictly increasing within bounds.

    Args:
        modified_values: List of values to process (modified in place)
        question: NumericQuestion with bounds information
        range_size: Range of the question

    Returns:
        Modified list of values
    """
    epsilon = max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)

    # Re-ensure increasing after clamping, bounded (left-to-right)
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
            target = modified_values[i - 1] + epsilon
            if not question.open_upper_bound:
                target = min(target, question.upper_bound - epsilon)
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = max(modified_values[i], target)

    # Additional pass (right-to-left) to make room near closed upper bound
    # If upper bound is closed and strict increase is capped, slide earlier values down by epsilon
    for i in range(len(modified_values) - 2, -1, -1):
        if modified_values[i] >= modified_values[i + 1]:
            target = modified_values[i + 1] - epsilon
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = min(modified_values[i], target)

    return modified_values
