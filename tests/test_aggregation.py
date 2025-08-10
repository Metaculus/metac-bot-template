from datetime import datetime
from typing import cast

import numpy as np
import pytest
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_numeric_aggregation_configurable():
    """
    Tests that the aggregation method for numeric questions can be configured
    and correctly applies either mean or median as specified.
    """
    # 1. Arrange
    # Create a mock numeric question to provide context for the aggregation.
    question = NumericQuestion(
        id_of_question=1,
        id_of_post=1,
        page_url="https://www.metaculus.com/questions/1/test",
        question_text="Test question for numeric aggregation?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=datetime.fromisoformat("2023-01-01T00:00:00"),
        close_time=datetime.fromisoformat("2025-01-01T00:00:00"),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )

    # Create two different numeric distributions to be aggregated.
    # `dist1` is a uniform distribution.
    # `dist2` is skewed towards the upper bound.
    x_axis = [p * 100 for p in np.linspace(0, 1, 11) if 0 < p < 1]
    dist1_percentiles = [Percentile(value=v, percentile=p) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]
    dist2_percentiles = [Percentile(value=v, percentile=p**0.5) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]

    # Third distribution: quadratic skew towards lower bound.
    dist3_percentiles = [Percentile(value=v, percentile=p**2) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]

    common_args = {
        "open_lower_bound": question.open_lower_bound,
        "open_upper_bound": question.open_upper_bound,
        "lower_bound": question.lower_bound,
        "upper_bound": question.upper_bound,
        "zero_point": question.zero_point,
    }

    pred1 = NumericDistribution(declared_percentiles=dist1_percentiles, **common_args)
    pred2 = NumericDistribution(declared_percentiles=dist2_percentiles, **common_args)
    pred3 = NumericDistribution(declared_percentiles=dist3_percentiles, **common_args)
    predictions: list[PredictionTypes] = [pred1, pred2, pred3]

    # Initialize two forecaster instances with different aggregation methods.
    llms_min = {"default": "mock", "parser": "mock", "researcher": "mock", "summarizer": "mock"}
    forecaster_mean = TemplateForecaster(llms=llms_min, numeric_aggregation_method="mean")
    forecaster_median = TemplateForecaster(llms=llms_min, numeric_aggregation_method="median")

    # 2. Act
    # Run the aggregation for both the 'mean' and 'median' configurations.
    mean_agg_result_uncast = await forecaster_mean._aggregate_predictions(predictions, question)
    median_agg_result_uncast = await forecaster_median._aggregate_predictions(predictions, question)
    mean_agg_result = cast(NumericDistribution, mean_agg_result_uncast)
    median_agg_result = cast(NumericDistribution, median_agg_result_uncast)

    # 3. Assert
    # Manually calculate the expected CDFs to verify the aggregation logic.
    expected_mean_cdf_percentiles = np.mean(
        [
            [p.percentile for p in pred1.cdf],
            [p.percentile for p in pred2.cdf],
            [p.percentile for p in pred3.cdf],
        ],
        axis=0,
    )
    expected_median_cdf_percentiles = np.median(
        [
            [p.percentile for p in pred1.cdf],
            [p.percentile for p in pred2.cdf],
            [p.percentile for p in pred3.cdf],
        ],
        axis=0,
    )

    # Extract the percentile values from the results.
    result_mean_percentiles = [p.percentile for p in mean_agg_result.declared_percentiles]
    result_median_percentiles = [p.percentile for p in median_agg_result.declared_percentiles]

    # Verify that the aggregated distributions match the expected values.
    assert np.allclose(result_mean_percentiles, expected_mean_cdf_percentiles)
    assert np.allclose(result_median_percentiles, expected_median_cdf_percentiles)

    # Ensure the two aggregation methods produce different results.
    assert not np.allclose(result_mean_percentiles, result_median_percentiles)
