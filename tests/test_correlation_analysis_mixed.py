"""Tests for mixed question type correlation analysis improvements."""

from unittest.mock import Mock

import pytest

from metaculus_bot.correlation_analysis import CorrelationAnalyzer, ModelPrediction


def test_extract_prediction_components_binary():
    """Test extraction of binary prediction components."""
    analyzer = CorrelationAnalyzer()

    # Mock binary prediction
    report = Mock()
    report.prediction = 0.75

    q_type, components = analyzer._extract_prediction_components(report)
    assert q_type == "binary"
    assert components == [0.75]


def test_extract_prediction_components_numeric():
    """Test extraction of numeric prediction components."""
    analyzer = CorrelationAnalyzer()

    # Mock numeric prediction with percentiles
    report = Mock()
    report.prediction = Mock()

    # Create mock percentiles
    percentiles = []
    for p in [10, 20, 40, 60, 80, 90]:
        mock_percentile = Mock()
        mock_percentile.percentile = p
        mock_percentile.value = p * 10  # Values: 100, 200, 400, 600, 800, 900
        percentiles.append(mock_percentile)

    report.prediction.declared_percentiles = percentiles
    # Make sure it doesn't get misclassified as multiple choice
    report.prediction.predicted_options = None

    q_type, components = analyzer._extract_prediction_components(report)
    assert q_type == "numeric"
    assert components == [100.0, 200.0, 400.0, 600.0, 800.0, 900.0]


def test_extract_prediction_components_multiple_choice():
    """Test extraction of multiple choice prediction components."""
    analyzer = CorrelationAnalyzer()

    # Mock multiple choice prediction
    report = Mock()
    report.prediction = Mock()

    # Create mock options
    options = []
    for i, (option, prob) in enumerate([("Option A", 0.4), ("Option B", 0.3), ("Option C", 0.3)]):
        mock_option = Mock()
        mock_option.option = option
        mock_option.probability = prob
        options.append(mock_option)

    report.prediction.predicted_options = options
    # Make sure other attributes don't interfere
    report.prediction.declared_percentiles = None
    report.prediction.median = None

    q_type, components = analyzer._extract_prediction_components(report)
    assert q_type == "multiple_choice"
    # Should be sorted by option name
    assert len(components) == 3
    assert all(isinstance(c, float) for c in components)


def test_has_mixed_question_types():
    """Test detection of mixed question types."""
    analyzer = CorrelationAnalyzer()

    # Mock benchmarks with different question types
    binary_report = Mock()
    binary_report.prediction = 0.5

    numeric_report = Mock()
    numeric_report.prediction = Mock()
    numeric_report.prediction.declared_percentiles = None  # No percentiles, should use median fallback
    numeric_report.prediction.median = 100.0

    mc_report = Mock()
    mc_report.prediction = Mock()
    mc_option = Mock()
    mc_option.option = "A"
    mc_option.probability = 1.0
    mc_report.prediction.predicted_options = [mc_option]
    mc_report.prediction.declared_percentiles = None
    mc_report.prediction.median = None

    # Test with only binary (should be False)
    benchmark1 = Mock()
    benchmark1.forecast_reports = [binary_report]
    analyzer.benchmarks = [benchmark1]
    assert not analyzer._has_mixed_question_types()

    # Test with binary and numeric (should be True)
    benchmark2 = Mock()
    benchmark2.forecast_reports = [numeric_report]
    analyzer.benchmarks = [benchmark1, benchmark2]
    assert analyzer._has_mixed_question_types()

    # Test with all three types (should be True)
    benchmark3 = Mock()
    benchmark3.forecast_reports = [mc_report]
    analyzer.benchmarks = [benchmark1, benchmark2, benchmark3]
    assert analyzer._has_mixed_question_types()


def test_get_question_type_breakdown():
    """Test question type counting."""
    analyzer = CorrelationAnalyzer()

    # Create mock reports
    binary_reports = [Mock() for _ in range(3)]
    for report in binary_reports:
        report.prediction = 0.5

    numeric_reports = [Mock() for _ in range(2)]
    for report in numeric_reports:
        report.prediction = Mock()
        report.prediction.declared_percentiles = None  # No percentiles, should use median fallback
        report.prediction.median = 100.0
        report.prediction.predicted_options = None  # Make sure it's not misclassified as multiple choice

    # Create mock benchmark
    benchmark = Mock()
    benchmark.forecast_reports = binary_reports + numeric_reports
    analyzer.benchmarks = [benchmark]

    breakdown = analyzer._get_question_type_breakdown()
    assert breakdown["binary"] == 3
    assert breakdown["numeric"] == 2
    assert len(breakdown) == 2
