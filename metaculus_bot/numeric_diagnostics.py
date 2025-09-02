from __future__ import annotations

"""
Diagnostic utilities for numeric forecasting.

Extracted from main.py to improve testability and maintainability.
Contains logging and diagnostic functions for numeric predictions.
"""

import logging
from typing import Any, List, Optional

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

logger = logging.getLogger(__name__)


def log_cdf_diagnostics_on_error(prediction: Any, question: NumericQuestion, error: Exception) -> None:
    """
    Log rich diagnostics when CDF construction fails.

    Args:
        prediction: NumericDistribution object that failed CDF construction
        question: NumericQuestion for context
        error: Exception that occurred during CDF construction
    """
    try:
        declared = getattr(prediction, "declared_percentiles", [])
        bounds = {
            "lower_bound": getattr(question, "lower_bound", None),
            "upper_bound": getattr(question, "upper_bound", None),
            "open_lower_bound": getattr(question, "open_lower_bound", None),
            "open_upper_bound": getattr(question, "open_upper_bound", None),
            "zero_point": getattr(question, "zero_point", None),
            "cdf_size": getattr(question, "cdf_size", None),
        }
        vals = [float(p.value) for p in declared]
        prcs = [float(p.percentile) for p in declared]
        deltas_val = [b - a for a, b in zip(vals, vals[1:])]
        deltas_pct = [b - a for a, b in zip(prcs, prcs[1:])]

        logger.error(
            "Numeric CDF spacing assertion for Q %s | URL %s | error=%s\n"
            "Bounds=%s\n"
            "Declared percentiles (p%% -> v): %s\n"
            "Value deltas: %s | Percentile deltas: %s",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            error,
            bounds,
            [(p, v) for p, v in zip(prcs, vals)],
            deltas_val,
            deltas_pct,
        )
    except Exception as log_e:
        logger.error("Failed logging numeric CDF diagnostics: %s", log_e)


def validate_cdf_construction(prediction: Any, question: NumericQuestion) -> None:
    """
    Validate CDF construction for non-PCHIP distributions.

    Args:
        prediction: NumericDistribution to validate
        question: NumericQuestion for diagnostic context

    Raises:
        AssertionError, ZeroDivisionError: If CDF validation fails
    """
    # Skip CDF validation for PCHIP distributions since they enforce constraints internally
    if hasattr(prediction, "_pchip_cdf_values"):
        logger.debug(
            f"Question {getattr(question, 'id_of_question', 'N/A')}: Skipping CDF validation for PCHIP distribution"
        )
        return

    try:
        # Force CDF construction to surface any issues
        _ = prediction.cdf
    except (AssertionError, ZeroDivisionError) as e:
        log_cdf_diagnostics_on_error(prediction, question, e)
        raise


def log_final_prediction(prediction: Any, question: NumericQuestion) -> None:
    """
    Log the final prediction for debugging purposes.

    Args:
        prediction: NumericDistribution with final prediction
        question: NumericQuestion for context
    """
    logger.info(f"Forecasted URL {question.page_url} as {prediction.declared_percentiles}")


def log_pchip_fallback(question: NumericQuestion, error: Exception) -> None:
    """
    Log when PCHIP CDF construction fails and fallback is used.

    Args:
        question: NumericQuestion for context
        error: Exception that caused fallback
    """
    logger.warning(
        f"Question {getattr(question, 'id_of_question', 'N/A')}: PCHIP CDF construction failed ({str(error)}), "
        "falling back to forecasting-tools default"
    )
