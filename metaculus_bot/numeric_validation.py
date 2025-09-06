"""
Percentile validation and processing utilities for numeric forecasting.

Extracted from main.py to improve testability and maintainability.
Contains validation logic for percentile sets and value processing.
"""

import logging
from typing import List

from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from .numeric_config import EXPECTED_PERCENTILE_COUNT, STANDARD_PERCENTILES

logger = logging.getLogger(__name__)


def validate_percentile_count_and_values(percentile_list: List[Percentile]) -> None:
    """
    Validate that we have exactly the expected number of percentiles with the correct values.

    Args:
        percentile_list: List of Percentile objects to validate

    Raises:
        ValidationError: If percentile count or values don't match expectations
    """
    expected_percentiles = set(STANDARD_PERCENTILES)

    # Check count
    if len(percentile_list) != EXPECTED_PERCENTILE_COUNT:
        raise ValidationError.from_exception_data(
            "NumericDistribution",
            [
                {
                    "type": "value_error",
                    "loc": ("declared_percentiles",),
                    "input": percentile_list,
                    "ctx": {
                        "error": f"Expected {EXPECTED_PERCENTILE_COUNT} declared percentiles (5,10,20,40,60,80,90,95), got {len(percentile_list)}.",
                    },
                }
            ],
        )

    # Check values with tolerance for rounding
    actual_percentiles = {round(p.percentile, 6) for p in percentile_list}
    expected_rounded = {round(p, 6) for p in expected_percentiles}
    if actual_percentiles != expected_rounded:
        raise ValidationError.from_exception_data(
            "NumericDistribution",
            [
                {
                    "type": "value_error",
                    "loc": ("declared_percentiles",),
                    "input": percentile_list,
                    "ctx": {
                        "error": f"Expected percentile set {{5,10,20,40,60,80,90,95}}, got {sorted(p.percentile * 100 for p in percentile_list)}.",
                    },
                }
            ],
        )


def sort_percentiles_by_value(percentile_list: List[Percentile]) -> List[Percentile]:
    """
    Sort percentiles by percentile value to ensure proper order.

    Args:
        percentile_list: List of Percentile objects to sort

    Returns:
        Sorted list of percentiles
    """
    return sorted(percentile_list, key=lambda p: p.percentile)


def filter_to_standard_percentiles(percentile_list: List[Percentile]) -> List[Percentile]:
    """Keep only the standard 8 percentiles {5,10,20,40,60,80,90,95}.

    If extras like 50th percentile are present, drop them before validation.
    If duplicates occur (same percentile repeated), keep the first occurrence.
    """
    allowed = {round(p, 6) for p in STANDARD_PERCENTILES}
    seen: set[float] = set()
    filtered: List[Percentile] = []
    for p in percentile_list:
        key = round(float(p.percentile), 6)
        if key in allowed and key not in seen:
            filtered.append(p)
            seen.add(key)
    return filtered


def check_discrete_question_properties(question, cdf_points: int) -> tuple[bool, bool]:
    """
    Check if a question is discrete and determine zero_point handling.

    Args:
        question: NumericQuestion object
        cdf_points: Number of points expected in CDF (e.g., 201)

    Returns:
        Tuple of (is_discrete, should_force_zero_point_none)
    """
    cdf_size = getattr(question, "cdf_size", None)
    is_discrete = cdf_size is not None and cdf_size != cdf_points
    zero_point = getattr(question, "zero_point", None)

    force_zero_point_none = False

    if is_discrete and zero_point is not None:
        logger.debug(
            f"Question {getattr(question, 'id_of_question', 'N/A')}: Forcing zero_point=None for discrete question"
        )
        force_zero_point_none = True
    elif zero_point is not None and zero_point == question.lower_bound:
        logger.warning(
            f"Question {getattr(question, 'id_of_question', 'N/A')}: zero_point ({zero_point}) is equal to lower_bound "
            f"({question.lower_bound}). Forcing linear scale for CDF generation."
        )
        force_zero_point_none = True

    return is_discrete, force_zero_point_none
