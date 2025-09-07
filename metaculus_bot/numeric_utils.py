"""Numeric aggregation and helper utilities used by TemplateForecaster.

This module centralises logic for combining numeric forecasts and constructing
user-friendly bound messages so that the core forecaster class stays small.
"""

import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN, NUM_MIN_PROB_STEP, NUM_RAMP_K_FACTOR
from metaculus_bot.pchip_cdf import generate_pchip_cdf
from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

__all__ = [
    "aggregate_numeric",
    "aggregate_binary_mean",
    "bound_messages",
    "clamp_and_renormalize_mc",
]


logger = logging.getLogger(__name__)


def aggregate_binary_mean(predictions: Sequence[float]) -> float:
    """Return the mean of *binary* forecasts rounded to three decimals.

    This matches the old behaviour from `TemplateForecaster`.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")

    mean_prediction = sum(predictions) / len(predictions)
    return round(mean_prediction, 3)


# TODO: modularize this function; split mean/median logic is gross
async def aggregate_numeric(
    predictions: Sequence[NumericDistribution],
    question: NumericQuestion,
    method: str | Literal["mean", "median"] = "mean",
) -> NumericDistribution:
    """Aggregate numeric distributions by mean or median.

    Parameters
    ----------
    predictions
        List of `NumericDistribution` objects as produced by individual LLMs.
    question
        The original `NumericQuestion` – needed for bounds metadata.
    method
        "mean" (default) or "median" to pick aggregation strategy.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of numeric predictions")

    if method == "median":
        # ---- Median aggregation (robust ensemble on common x-grid) ----
        numeric_predictions = list(predictions)

        cdfs_as_dfs = [pd.DataFrame([p.model_dump() for p in pred.cdf]) for pred in numeric_predictions]
        combined_cdf = pd.concat(cdfs_as_dfs, ignore_index=True)
        median_series = combined_cdf.groupby("value", sort=True)["percentile"].median()

        x_vals = median_series.index.to_numpy(dtype=float)
        p_vals = median_series.to_numpy(dtype=float)

        # Clamp and monotonic
        p_vals = np.clip(p_vals, 0.0, 1.0)
        p_vals = np.maximum.accumulate(p_vals)

        # Re-pin endpoints
        if question.open_lower_bound:
            p_vals[0] = max(p_vals[0], 0.001)
        else:
            p_vals[0] = 0.0
        if question.open_upper_bound:
            p_vals[-1] = min(p_vals[-1], 0.999)
        else:
            p_vals[-1] = 1.0

        # Determine target size and min-step
        target_cdf_size = int(getattr(question, "cdf_size", len(x_vals)) or len(x_vals))
        is_discrete = target_cdf_size != len(x_vals)
        min_step_required = max(NUM_MIN_PROB_STEP, 0.01 / max(1, (target_cdf_size - 1)))

        if not is_discrete:
            # Smooth if needed
            diffs_before = np.diff(p_vals)
            min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0
            if min_delta_before < min_step_required:
                ramp = np.linspace(0.0, min_step_required * NUM_RAMP_K_FACTOR, len(p_vals))
                p_vals = np.maximum.accumulate(p_vals + ramp)
                # Re-pin endpoints after smoothing
                if question.open_lower_bound:
                    p_vals[0] = max(p_vals[0], 0.001)
                else:
                    p_vals[0] = 0.0
                if question.open_upper_bound:
                    p_vals[-1] = min(p_vals[-1], 0.999)
                else:
                    p_vals[-1] = 1.0

                diffs_after = np.diff(p_vals)
                min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
                logger.warning(
                    "Ensemble CDF ramp smoothing (median) | Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f",
                    getattr(question, "id_of_question", None),
                    getattr(question, "page_url", None),
                    min_delta_before,
                    min_delta_after,
                )
            declared_percentiles = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_vals, p_vals)]
            return create_pchip_numeric_distribution(
                pchip_cdf=list(map(float, p_vals)),
                percentile_list=declared_percentiles,
                question=question,
                zero_point=question.zero_point,
            )

        # Discrete: resample and enforce min-step
        logger.info(
            "Discrete aggregation detected (median) | Q %s | URL %s | target_cdf_size=%d | min_step_required=%.8f",
            getattr(question, "id_of_question", None),
            getattr(question, "page_url", None),
            target_cdf_size,
            min_step_required,
        )

        percentile_values = {float(prob * 100.0): float(val) for val, prob in zip(x_vals, p_vals)}
        pchip_cdf_values, _ = generate_pchip_cdf(
            percentile_values=percentile_values,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
            min_step=min_step_required,
            num_points=target_cdf_size,
            question_id=getattr(question, "id_of_question", None),
            question_url=getattr(question, "page_url", None),
        )

        x_disc = np.linspace(question.lower_bound, question.upper_bound, target_cdf_size)
        declared_percentiles = [
            Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_disc, pchip_cdf_values)
        ]
        return create_pchip_numeric_distribution(
            pchip_cdf=list(map(float, pchip_cdf_values)),
            percentile_list=declared_percentiles,
            question=question,
            zero_point=question.zero_point,
        )

    if method != "mean":
        raise ValueError(f"Invalid aggregation method: {method}")

    # ---- Mean aggregation (robust ensemble on common x-grid) ----
    numeric_predictions = list(predictions)

    # Build a combined dataframe of all model CDFs (each row: {value, percentile})
    cdfs_as_dfs = [pd.DataFrame([p.model_dump() for p in pred.cdf]) for pred in numeric_predictions]
    combined_cdf = pd.concat(cdfs_as_dfs, ignore_index=True)

    # Average the probability at each x, preserving the shared x-grid order
    mean_series = combined_cdf.groupby("value", sort=True)["percentile"].mean()
    x_vals = mean_series.index.to_numpy(dtype=float)
    p_vals = mean_series.to_numpy(dtype=float)

    # Validate and enforce probability-side constraints at the ensemble level
    # 1) Clamp to [0,1]
    p_vals = np.clip(p_vals, 0.0, 1.0)

    # 2) Ensure monotonic non-decreasing
    p_vals = np.maximum.accumulate(p_vals)

    # 3) Re-pin endpoints according to open/closed bound semantics
    if question.open_lower_bound:
        p_vals[0] = max(p_vals[0], 0.001)
    else:
        p_vals[0] = 0.0
    if question.open_upper_bound:
        p_vals[-1] = min(p_vals[-1], 0.999)
    else:
        p_vals[-1] = 1.0

    # 4) Determine target CDF size and min-step based on discrete/continuous
    target_cdf_size = int(getattr(question, "cdf_size", len(x_vals)) or len(x_vals))
    is_discrete = target_cdf_size != len(x_vals)
    # Metaculus backend effectively enforces total available probability range >= 0.01.
    # Therefore per-step minimum is ~0.01/(N-1). Use at least NUM_MIN_PROB_STEP for continuous.
    min_step_required = max(NUM_MIN_PROB_STEP, 0.01 / max(1, (target_cdf_size - 1)))

    if not is_discrete:
        # Continuous path: optional gentle ramp smoothing if needed
        diffs_before = np.diff(p_vals)
        min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0
        if min_delta_before < min_step_required:
            ramp = np.linspace(0.0, min_step_required * NUM_RAMP_K_FACTOR, len(p_vals))
            p_vals = np.maximum.accumulate(p_vals + ramp)
            # Re-pin endpoints after smoothing
            if question.open_lower_bound:
                p_vals[0] = max(p_vals[0], 0.001)
            else:
                p_vals[0] = 0.0
            if question.open_upper_bound:
                p_vals[-1] = min(p_vals[-1], 0.999)
            else:
                p_vals[-1] = 1.0

            diffs_after = np.diff(p_vals)
            min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
            logger.warning(
                "Ensemble CDF ramp smoothing | Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f",
                getattr(question, "id_of_question", None),
                getattr(question, "page_url", None),
                min_delta_before,
                min_delta_after,
            )
        # Declared percentiles: full 201-point grid
        declared_percentiles: list[Percentile] = [
            Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_vals, p_vals)
        ]
        # Return PCHIP-backed distribution using the continuous-length CDF
        return create_pchip_numeric_distribution(
            pchip_cdf=list(map(float, p_vals)),
            percentile_list=declared_percentiles,
            question=question,
            zero_point=question.zero_point,
        )

    # Discrete path: resample to question.cdf_size and strictly enforce min-step
    logger.info(
        "Discrete aggregation detected | Q %s | URL %s | target_cdf_size=%d | min_step_required=%.8f",
        getattr(question, "id_of_question", None),
        getattr(question, "page_url", None),
        target_cdf_size,
        min_step_required,
    )

    # Build percentile mapping from ensemble CDF (probability→value), scaled to [0,100] keys
    percentile_values = {float(prob * 100.0): float(val) for val, prob in zip(x_vals, p_vals)}

    # Use the robust PCHIP generator to resample to the discrete grid and enforce min-step
    pchip_cdf_values, _ = generate_pchip_cdf(
        percentile_values=percentile_values,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=question.zero_point,
        min_step=min_step_required,
        num_points=target_cdf_size,
        question_id=getattr(question, "id_of_question", None),
        question_url=getattr(question, "page_url", None),
    )

    # Declared percentiles: full discrete grid after enforcement
    x_disc = np.linspace(question.lower_bound, question.upper_bound, target_cdf_size)
    declared_percentiles = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_disc, pchip_cdf_values)]

    return create_pchip_numeric_distribution(
        pchip_cdf=list(map(float, pchip_cdf_values)),
        percentile_list=declared_percentiles,
        question=question,
        zero_point=question.zero_point,
    )


def bound_messages(question: NumericQuestion) -> tuple[str, str]:
    """Return upper & lower bound helper messages for numeric prompts.

    For discrete questions, if nominal bounds are missing, derive them using half-step logic.
    """

    nominal_upper = getattr(question, "nominal_upper_bound", None)
    nominal_lower = getattr(question, "nominal_lower_bound", None)

    # For discrete questions, if we don't have nominal bounds, derive them using half-step logic
    cdf_size = getattr(question, "cdf_size", None)
    if nominal_upper is None and nominal_lower is None and cdf_size is not None and cdf_size != 201:
        # This is likely a discrete question - derive nominal bounds from half-step
        step = (question.upper_bound - question.lower_bound) / (cdf_size - 1)
        nominal_upper = question.upper_bound - step / 2
        nominal_lower = question.lower_bound + step / 2

    upper_bound_number = nominal_upper if nominal_upper is not None else question.upper_bound
    lower_bound_number = nominal_lower if nominal_lower is not None else question.lower_bound

    if question.open_upper_bound:
        upper_bound_message = f"Practical upper bound / display range is {upper_bound_number}."
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound_number}."

    if question.open_lower_bound:
        lower_bound_message = f"Practical lower bound / display range is {lower_bound_number}."
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound_number}."
    return upper_bound_message, lower_bound_message


def clamp_and_renormalize_mc(
    predicted_option_list: PredictedOptionList,
) -> PredictedOptionList:
    """Clamp MC option probabilities and renormalize in-place.

    - Clamps each option probability to [MC_PROB_MIN, MC_PROB_MAX].
    - Renormalizes so that probabilities sum to 1.0 (if total > 0).
    - Returns the same `PredictedOptionList` for convenience.
    """
    # Clamp
    for option in predicted_option_list.predicted_options:
        option.probability = max(MC_PROB_MIN, min(MC_PROB_MAX, option.probability))

    # Renormalize
    total_prob = sum(option.probability for option in predicted_option_list.predicted_options)
    if total_prob > 0:
        for option in predicted_option_list.predicted_options:
            option.probability /= total_prob

    return predicted_option_list
