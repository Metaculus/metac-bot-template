#!/usr/bin/env python3
"""Test real ensemble prediction aggregation."""

from metaculus_bot.correlation_analysis import CorrelationAnalyzer


def test_binary_prediction_aggregation():
    """Test that binary prediction aggregation works correctly for mean vs median."""
    analyzer = CorrelationAnalyzer()

    # Test 3-model aggregation
    individual_preds = {"model_a": 0.4, "model_b": 0.8, "model_c": 0.5}
    models = ["model_a", "model_b", "model_c"]

    mean_result = analyzer._aggregate_predictions(individual_preds, models, "binary", "mean")
    median_result = analyzer._aggregate_predictions(individual_preds, models, "binary", "median")

    expected_mean = (0.4 + 0.8 + 0.5) / 3  # 0.567
    expected_median = 0.5  # median of [0.4, 0.5, 0.8]

    assert abs(mean_result - expected_mean) < 0.001, f"Expected {expected_mean}, got {mean_result}"
    assert abs(median_result - expected_median) < 0.001, f"Expected {expected_median}, got {median_result}"
    assert mean_result != median_result, "Mean and median should be different for 3 models"


def test_binary_prediction_aggregation_two_models():
    """Test that 2-model aggregation produces identical mean and median."""
    analyzer = CorrelationAnalyzer()

    individual_preds = {"model_a": 0.4, "model_b": 0.8}
    models = ["model_a", "model_b"]

    mean_result = analyzer._aggregate_predictions(individual_preds, models, "binary", "mean")
    median_result = analyzer._aggregate_predictions(individual_preds, models, "binary", "median")

    expected = (0.4 + 0.8) / 2  # 0.6

    assert abs(mean_result - expected) < 0.001, f"Expected {expected}, got {mean_result}"
    assert abs(median_result - expected) < 0.001, f"Expected {expected}, got {median_result}"
    assert mean_result == median_result, "Mean and median should be identical for 2 models"


def test_baseline_score_calculation():
    """Test that baseline score calculation works for binary predictions."""
    analyzer = CorrelationAnalyzer()

    # Test with different prediction and community values
    prediction = 0.7
    community_pred = 0.6

    score = analyzer._calculate_baseline_score(prediction, community_pred, "binary")

    assert score is not None, "Should return a score for valid inputs"
    assert isinstance(score, (int, float)), "Score should be numeric"

    # Test that different predictions produce different scores
    score2 = analyzer._calculate_baseline_score(0.5, community_pred, "binary")
    assert score != score2, "Different predictions should produce different scores"
