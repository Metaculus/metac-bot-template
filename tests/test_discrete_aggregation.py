import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric_utils import aggregate_numeric


@pytest.mark.asyncio
async def test_discrete_mean_aggregation_cdf_size_and_min_step():
    """Mean aggregation returns exactly cdf_size values with required min step for discrete."""
    question = NumericQuestion(
        id_of_question=123,
        id_of_post=123,
        page_url="https://example.com/q/123",
        question_text="Discrete count question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )

    # Three simple input distributions with modestly different shapes
    decl_a = [
        Percentile(value=0.0, percentile=0.10),
        Percentile(value=3.0, percentile=0.50),
        Percentile(value=7.0, percentile=0.90),
    ]
    decl_b = [
        Percentile(value=1.0, percentile=0.20),
        Percentile(value=4.0, percentile=0.55),
        Percentile(value=6.5, percentile=0.85),
    ]
    decl_c = [
        Percentile(value=0.5, percentile=0.15),
        Percentile(value=3.5, percentile=0.48),
        Percentile(value=7.2, percentile=0.92),
    ]

    dist_a = NumericDistribution(declared_percentiles=decl_a, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=decl_b, **question.model_dump())
    dist_c = NumericDistribution(declared_percentiles=decl_c, **question.model_dump())

    agg = await aggregate_numeric([dist_a, dist_b, dist_c], question, "mean")

    cdf = agg.cdf
    probs = np.array([p.percentile for p in cdf], dtype=float)
    values = np.array([p.value for p in cdf], dtype=float)

    assert len(cdf) == question.cdf_size
    required_min_step = 0.01 / (question.cdf_size - 1)
    diffs = np.diff(probs)
    assert np.all(diffs >= required_min_step - 1e-12)
    # Endpoints pinned for closed bounds
    assert abs(probs[0] - 0.0) <= 1e-12
    assert abs(probs[-1] - 1.0) <= 1e-12
    # Value axis is the evenly spaced discrete grid
    expected_values = np.linspace(question.lower_bound, question.upper_bound, question.cdf_size)
    assert np.allclose(values, expected_values)


@pytest.mark.asyncio
async def test_discrete_median_aggregation_open_upper():
    """Median aggregation handles open upper bound and discrete min step."""
    question = NumericQuestion(
        id_of_question=456,
        id_of_post=456,
        page_url="https://example.com/q/456",
        question_text="Discrete count question (open upper)",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )

    decl_a = [
        Percentile(value=0.0, percentile=0.05),
        Percentile(value=3.0, percentile=0.45),
        Percentile(value=7.0, percentile=0.88),
    ]
    decl_b = [
        Percentile(value=1.0, percentile=0.18),
        Percentile(value=4.0, percentile=0.52),
        Percentile(value=6.8, percentile=0.87),
    ]
    dist_a = NumericDistribution(declared_percentiles=decl_a, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=decl_b, **question.model_dump())

    agg = await aggregate_numeric([dist_a, dist_b], question, "median")

    cdf = agg.cdf
    probs = np.array([p.percentile for p in cdf], dtype=float)
    values = np.array([p.value for p in cdf], dtype=float)

    assert len(cdf) == question.cdf_size
    required_min_step = 0.01 / (question.cdf_size - 1)
    diffs = np.diff(probs)
    assert np.all(diffs >= required_min_step - 1e-12)
    # Endpoint semantics: closed lower = 0.0, open upper ≤ 0.999
    assert abs(probs[0] - 0.0) <= 1e-12
    assert probs[-1] <= 0.999 + 1e-12
    expected_values = np.linspace(question.lower_bound, question.upper_bound, question.cdf_size)
    assert np.allclose(values, expected_values)
