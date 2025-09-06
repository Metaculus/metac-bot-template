import statistics
from enum import Enum
from typing import Sequence

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption


class AggregationStrategy(Enum):
    """Supported aggregation strategies for ensemble predictions."""

    MEAN = "mean"
    MEDIAN = "median"
    STACKING = "stacking"
    # Future: GEOMETRIC_MEAN = "geometric_mean"
    # Future: TRIMMED_MEAN = "trimmed_mean"


def aggregate_binary_median(predictions: Sequence[float]) -> float:
    """Return the median of binary forecasts rounded to three decimals.

    Parameters
    ----------
    predictions : Sequence[float]
        Binary predictions from multiple models (values between 0 and 1)

    Returns
    -------
    float
        Median prediction rounded to 3 decimal places

    Raises
    ------
    ValueError
        If predictions list is empty
    """
    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")

    median_prediction = statistics.median(predictions)
    return round(median_prediction, 3)


def aggregate_multiple_choice_mean(
    predictions: Sequence[PredictedOptionList],
) -> PredictedOptionList:
    """Aggregate multiple choice predictions using mean pooling across option probabilities.

    This extracts the logic that was previously handled by the framework's
    ForecastBot._aggregate_predictions() method.

    Parameters
    ----------
    predictions : Sequence[PredictedOptionList]
        Multiple choice predictions from different models

    Returns
    -------
    PredictedOptionList
        Aggregated prediction with mean probabilities across all options

    Raises
    ------
    ValueError
        If predictions list is empty or options don't match across predictions
    """
    if not predictions:
        raise ValueError("Cannot aggregate empty list of multiple choice predictions")

    # Verify all predictions have the same options
    first_options = {opt.option_name for opt in predictions[0].predicted_options}
    for pred in predictions[1:]:
        pred_options = {opt.option_name for opt in pred.predicted_options}
        if pred_options != first_options:
            raise ValueError("All predictions must have the same option names")

    # Aggregate probabilities by option name
    option_probabilities: dict[str, list[float]] = {}
    for pred in predictions:
        for option in pred.predicted_options:
            if option.option_name not in option_probabilities:
                option_probabilities[option.option_name] = []
            option_probabilities[option.option_name].append(option.probability)

    # Calculate mean probabilities
    aggregated_options = []
    for option_name, probs in option_probabilities.items():
        mean_prob = sum(probs) / len(probs)
        aggregated_options.append(PredictedOption(option_name=option_name, probability=mean_prob))

    # Renormalize to ensure probabilities sum to 1.0
    total_prob = sum(opt.probability for opt in aggregated_options)
    if total_prob > 0:
        for option in aggregated_options:
            option.probability /= total_prob

    return PredictedOptionList(predicted_options=aggregated_options)


def aggregate_multiple_choice_median(
    predictions: Sequence[PredictedOptionList],
) -> PredictedOptionList:
    """Aggregate multiple choice predictions using median pooling across option probabilities.

    Parameters
    ----------
    predictions : Sequence[PredictedOptionList]
        Multiple choice predictions from different models

    Returns
    -------
    PredictedOptionList
        Aggregated prediction with median probabilities across all options

    Raises
    ------
    ValueError
        If predictions list is empty or options don't match across predictions
    """
    if not predictions:
        raise ValueError("Cannot aggregate empty list of multiple choice predictions")

    # Verify all predictions have the same options (same logic as mean)
    first_options = {opt.option_name for opt in predictions[0].predicted_options}
    for pred in predictions[1:]:
        pred_options = {opt.option_name for opt in pred.predicted_options}
        if pred_options != first_options:
            raise ValueError("All predictions must have the same option names")

    # Aggregate probabilities by option name
    option_probabilities: dict[str, list[float]] = {}
    for pred in predictions:
        for option in pred.predicted_options:
            if option.option_name not in option_probabilities:
                option_probabilities[option.option_name] = []
            option_probabilities[option.option_name].append(option.probability)

    # Calculate median probabilities
    aggregated_options = []
    for option_name, probs in option_probabilities.items():
        median_prob = statistics.median(probs)
        aggregated_options.append(PredictedOption(option_name=option_name, probability=median_prob))

    # Renormalize to ensure probabilities sum to 1.0
    total_prob = sum(opt.probability for opt in aggregated_options)
    if total_prob > 0:
        for option in aggregated_options:
            option.probability /= total_prob

    return PredictedOptionList(predicted_options=aggregated_options)
