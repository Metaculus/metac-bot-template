import pytest
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList

from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    aggregate_binary_median,
    aggregate_multiple_choice_mean,
    aggregate_multiple_choice_median,
)


def test_aggregate_binary_median():
    """Test binary median aggregation."""
    # Test basic median
    assert aggregate_binary_median([0.3, 0.5, 0.7]) == 0.5

    # Test even number of predictions
    assert aggregate_binary_median([0.2, 0.4, 0.6, 0.8]) == 0.5

    # Test rounding
    assert aggregate_binary_median([0.333, 0.334, 0.335]) == 0.334

    # Test single prediction
    assert aggregate_binary_median([0.75]) == 0.75

    # Test empty list raises error
    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregate_binary_median([])


def test_aggregate_multiple_choice_mean():
    """Test multiple choice mean aggregation."""
    # Create test predictions
    pred1 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.6),
            PredictedOption(option_name="B", probability=0.4),
        ]
    )
    pred2 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.8),
            PredictedOption(option_name="B", probability=0.2),
        ]
    )

    result = aggregate_multiple_choice_mean([pred1, pred2])

    # Check that probabilities are averaged correctly
    assert len(result.predicted_options) == 2

    option_a = next(opt for opt in result.predicted_options if opt.option_name == "A")
    option_b = next(opt for opt in result.predicted_options if opt.option_name == "B")

    assert option_a.probability == pytest.approx(0.7)  # (0.6 + 0.8) / 2
    assert option_b.probability == pytest.approx(0.3)  # (0.4 + 0.2) / 2

    # Check normalization
    total_prob = sum(opt.probability for opt in result.predicted_options)
    assert total_prob == pytest.approx(1.0)


def test_aggregate_multiple_choice_median():
    """Test multiple choice median aggregation."""
    # Create test predictions with 3 models for clear median
    pred1 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.5),
            PredictedOption(option_name="B", probability=0.5),
        ]
    )
    pred2 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.7),
            PredictedOption(option_name="B", probability=0.3),
        ]
    )
    pred3 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.9),
            PredictedOption(option_name="B", probability=0.1),
        ]
    )

    result = aggregate_multiple_choice_median([pred1, pred2, pred3])

    # Check that probabilities use median correctly
    option_a = next(opt for opt in result.predicted_options if opt.option_name == "A")
    option_b = next(opt for opt in result.predicted_options if opt.option_name == "B")

    # Medians: A=[0.5, 0.7, 0.9] -> 0.7, B=[0.5, 0.3, 0.1] -> 0.3
    assert option_a.probability == pytest.approx(0.7)
    assert option_b.probability == pytest.approx(0.3)

    # Check normalization
    total_prob = sum(opt.probability for opt in result.predicted_options)
    assert total_prob == pytest.approx(1.0)


def test_mismatched_options_error():
    """Test that mismatched options raise an error."""
    pred1 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.6),
            PredictedOption(option_name="B", probability=0.4),
        ]
    )
    pred2 = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.8),
            PredictedOption(option_name="C", probability=0.2),  # Different option!
        ]
    )

    with pytest.raises(ValueError, match="All predictions must have the same option names"):
        aggregate_multiple_choice_mean([pred1, pred2])

    with pytest.raises(ValueError, match="All predictions must have the same option names"):
        aggregate_multiple_choice_median([pred1, pred2])


def test_empty_list_errors():
    """Test that empty prediction lists raise appropriate errors."""
    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregate_multiple_choice_mean([])

    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregate_multiple_choice_median([])


def test_aggregation_strategy_enum():
    """Test that AggregationStrategy enum has expected values."""
    assert AggregationStrategy.MEAN.value == "mean"
    assert AggregationStrategy.MEDIAN.value == "median"

    # Ensure we have the expected strategies
    strategies = [s.value for s in AggregationStrategy]
    assert "mean" in strategies
    assert "median" in strategies
