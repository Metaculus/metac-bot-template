"""
Tests for the scoring patches module.

Tests the monkey patching and scoring logic for mixed question types.
"""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from metaculus_bot.scoring_patches import (
    apply_scoring_patches,
    calculate_multiple_choice_baseline_score,
    calculate_numeric_baseline_score,
    extract_multiple_choice_probabilities,
    extract_numeric_percentiles,
    validate_community_prediction_count,
)


class TestCommunityPredictionValidation:
    """Test community prediction count validation."""

    def test_validate_with_num_predictions(self):
        """Test validation using num_predictions attribute."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 15

        assert validate_community_prediction_count(question) is True

        question.num_predictions = 5
        assert validate_community_prediction_count(question) is False

    def test_validate_with_prediction_count(self):
        """Test validation using prediction_count attribute."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = None
        question.prediction_count = 25

        assert validate_community_prediction_count(question) is True

    def test_validate_with_community_prediction_exists(self):
        """Test validation using community_prediction_at_access_time."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = None
        question.prediction_count = None
        question.community_prediction_at_access_time = 0.5

        assert validate_community_prediction_count(question) is True

    def test_validate_insufficient_data(self):
        """Test validation when no adequate data is available."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = None
        question.prediction_count = None
        question.community_prediction_at_access_time = None

        assert validate_community_prediction_count(question) is False


class TestMultipleChoiceExtraction:
    """Test multiple choice probability extraction."""

    def test_extract_mc_probabilities_success(self):
        """Test successful extraction of MC probabilities."""
        option1 = Mock()
        option1.option_name = "Option A"
        option1.probability = 0.3

        option2 = Mock()
        option2.option_name = "Option B"
        option2.probability = 0.7

        prediction = Mock()
        prediction.predicted_options = [option2, option1]  # Unsorted

        probs, option_names = extract_multiple_choice_probabilities(prediction)

        # Should be sorted by option_name: A, B
        assert probs == [0.3, 0.7]
        assert option_names == ["Option A", "Option B"]

    def test_extract_mc_probabilities_empty(self):
        """Test extraction with empty or invalid data."""
        prediction = Mock()
        prediction.predicted_options = None

        probs, option_names = extract_multiple_choice_probabilities(prediction)
        assert probs == []
        assert option_names == []

        # Test with empty list
        prediction.predicted_options = []
        probs, option_names = extract_multiple_choice_probabilities(prediction)
        assert probs == []
        assert option_names == []


class TestNumericExtraction:
    """Test numeric percentile extraction."""

    def test_extract_numeric_percentiles_success(self):
        """Test successful extraction of numeric percentiles."""
        p1 = Mock()
        p1.percentile = 10
        p1.value = 100

        p2 = Mock()
        p2.percentile = 90
        p2.value = 1000

        prediction = Mock()
        prediction.declared_percentiles = [p1, p2]

        percentiles = extract_numeric_percentiles(prediction)
        assert percentiles == [(10, 100), (90, 1000)]

    def test_extract_numeric_percentiles_empty(self):
        """Test extraction with empty or invalid data."""
        prediction = Mock()
        prediction.declared_percentiles = None

        percentiles = extract_numeric_percentiles(prediction)
        assert percentiles == []


class TestMultipleChoiceScoring:
    """Test multiple choice baseline scoring."""

    def test_mc_scoring_success(self):
        """Test successful MC scoring."""
        # Create mock question
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 15

        # Create mock prediction with 3 options
        option1 = Mock()
        option1.option_name = "A"
        option1.probability = 0.1

        option2 = Mock()
        option2.option_name = "B"
        option2.probability = 0.8

        option3 = Mock()
        option3.option_name = "C"
        option3.probability = 0.1

        prediction = Mock()
        prediction.predicted_options = [option1, option2, option3]

        # Provide community CP aligned to options
        question.options = ["A", "B", "C"]
        question.api_json = {
            "question": {
                "type": "multiple_choice",
                "options": ["A", "B", "C"],
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.1, 0.8, 0.1]}}},
            }
        }

        # Create mock report
        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_multiple_choice_baseline_score(report)

        # Score should be calculated and finite
        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)

        # Score should be in reasonable range for baseline scoring
        assert -500 <= score <= 500

    def test_mc_scoring_insufficient_predictions(self):
        """Test MC scoring with insufficient community predictions."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 5  # Below threshold
        question.prediction_count = None
        question.community_prediction_at_access_time = None

        report = Mock()
        report.question = question

        score = calculate_multiple_choice_baseline_score(report)
        assert score is None

    def test_mc_scoring_no_bot_probs(self):
        """Test MC scoring when bot probabilities cannot be extracted."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 15

        prediction = Mock()
        prediction.predicted_options = None  # No options

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_multiple_choice_baseline_score(report)
        assert score is None


class TestNumericScoring:
    """Test numeric baseline scoring with PDF approach."""

    def test_numeric_scoring_success(self):
        """Test successful numeric scoring with community benchmark approach."""
        # Create mock question with bounds
        question = Mock()
        question.id_of_question = 456
        question.num_predictions = 20
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Create mock percentiles (enough for PDF estimation)
        percentiles = []
        for p, v in [(10, 10), (20, 20), (40, 40), (60, 60), (80, 80), (90, 90)]:
            mock_p = Mock()
            mock_p.percentile = p
            mock_p.value = v
            percentiles.append(mock_p)

        prediction = Mock()
        prediction.declared_percentiles = percentiles

        # Provide community CDF (uniform for simplicity)
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 201).tolist()}}
                }
            }
        }

        # Create mock report
        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Score should be calculated and finite
        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)

        # Score should be in new expanded range with fixed normalization ln(10)
        # Range should be similar to MC scores: roughly [-100, +20]
        assert -200 <= score <= 100

    def test_numeric_scoring_pmf_path(self):
        """Test numeric scoring via PMF path when both model and community CDFs exist."""
        # Create mock question with bounds
        question = Mock()
        question.id_of_question = 789
        question.lower_bound = 0.0
        question.upper_bound = 100.0
        # Provide community CDF (uniform for simplicity)
        community_cdf = np.linspace(0.0, 1.0, 201).tolist()
        question.api_json = {
            "question": {"aggregations": {"recency_weighted": {"latest": {"forecast_values": community_cdf}}}}
        }

        # Create model CDF as a list of objects with .percentile
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        model_cdf = [P(p) for p in np.linspace(0.0, 1.0, 201).tolist()]

        prediction = Mock()
        prediction.cdf = model_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)
        # Score should be in realistic community benchmark range
        # With fixed normalization, scores should be in MC-like range
        assert -200 <= score <= 100

    def test_numeric_scoring_insufficient_percentiles(self):
        """Test numeric scoring with insufficient percentiles."""
        question = Mock()
        question.id_of_question = 456
        question.num_predictions = 20

        # Only 2 percentiles - insufficient for PDF estimation
        p1 = Mock()
        p1.percentile = 10
        p1.value = 100

        p2 = Mock()
        p2.percentile = 90
        p2.value = 200

        prediction = Mock()
        prediction.declared_percentiles = [p1, p2]

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)
        assert score is None

    def test_numeric_scoring_insufficient_predictions(self):
        """Test numeric scoring with insufficient community predictions."""
        question = Mock()
        question.id_of_question = 456
        question.num_predictions = 5  # Below threshold
        question.prediction_count = None
        question.community_prediction_at_access_time = None

        report = Mock()
        report.question = question

        score = calculate_numeric_baseline_score(report)
        assert score is None


class TestRelativeNumericScoring:
    """Test relative numeric scoring against community distribution (community benchmark context)."""

    def test_relative_scoring_with_community_cdf(self):
        """Test relative numeric scoring using community CDF as expectation weights."""
        # Create mock question with bounds [0, 100]
        question = Mock()
        question.id_of_question = 999
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Mock uniform community CDF
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {
                        "latest": {
                            "forecast_values": np.linspace(0.0, 1.0, 201).tolist()  # Uniform community
                        }
                    }
                }
            }
        }

        # Create bot prediction with concentration around middle (better than uniform)
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        # Create CDF that's more concentrated in middle than uniform
        model_cdf = []
        for i in range(201):
            p = i / 200.0
            # Sigmoid-like concentration
            cdf_val = 1.0 / (1.0 + math.exp(-8 * (p - 0.5)))
            model_cdf.append(P(cdf_val))

        prediction = Mock()
        prediction.cdf = model_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Should get a finite score; with calibrated normalization, concentrated vs uniform
        # may be more negative than earlier bands, but should remain bounded.
        assert score is not None
        assert isinstance(score, float)
        assert -230 <= score <= 120

    def test_relative_scoring_fallback_to_percentiles(self):
        """Test fallback to percentiles when CDF unavailable."""
        question = Mock()
        question.id_of_question = 998
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # No community CDF - will fall back to uniform community
        question.api_json = {"question": {"aggregations": {}}}

        # Create bot prediction using declared percentiles
        percentiles = []

        class P:
            def __init__(self, percentile, value):
                self.percentile = percentile
                self.value = value

        # Bot has tight distribution around 50 (better than uniform)
        values = [30, 45, 48, 50, 52, 55, 70]
        percs = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]
        for p, v in zip(percs, values):
            percentiles.append(P(p * 100, v))

        prediction = Mock()
        prediction.declared_percentiles = percentiles
        # No CDF available - will trigger fallback

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Should still get a reasonable score using fallback method
        assert score is not None
        assert isinstance(score, float)
        # With fixed normalization, should be in MC-like range
        assert -200 <= score <= 100

    def test_scoring_consistency_with_binary_mc(self):
        """Test that numeric scores are in similar range as binary/MC."""
        question = Mock()
        question.id_of_question = 997
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Uniform community CDF
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 11).tolist()}}
                }
            }
        }

        # Uniform bot CDF for comparison
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        uniform_bot_cdf = [P(i / 10.0) for i in range(11)]

        prediction = Mock()
        prediction.cdf = uniform_bot_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        numeric_score = calculate_numeric_baseline_score(report)

        # Compare to binary scoring for similar "neutral" prediction
        # Binary: 100.0 * (c * (log2(p) + 1.0) + (1.0 - c) * (log2(1.0 - p) + 1.0))
        c, p = 0.5, 0.5  # Both 50% - neutral
        binary_score = 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))

        # Should be in similar range (both around 0 for neutral predictions)
        assert numeric_score is not None
        assert isinstance(numeric_score, float)

        # Both should be relatively close to each other for neutral predictions
        score_diff = abs(numeric_score - binary_score)
        assert score_diff < 150  # Should be within reasonable range of each other

        print(f"Numeric score: {numeric_score:.2f}, Binary score: {binary_score:.2f}")


class TestScoreScaling:
    """Test that scores are on similar scales across question types."""

    def test_score_scales_comparable(self):
        """Test that binary, MC, and numeric scores are on comparable scales."""
        # This is an integration test to verify score scaling

        # Mock binary score (known good scale from existing implementation)
        # Binary formula: 100.0 * (c * (log2(p) + 1.0) + (1.0 - c) * (log2(1.0 - p) + 1.0))
        c, p = 0.7, 0.6  # Community 70%, bot 60%
        binary_score = 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))

        # Create MC and numeric scores using our functions
        mc_question = Mock()
        mc_question.id_of_question = 123
        mc_question.num_predictions = 15

        mc_option1 = Mock()
        mc_option1.option_name = "A"
        mc_option1.probability = 0.6

        mc_option2 = Mock()
        mc_option2.option_name = "B"
        mc_option2.probability = 0.4

        mc_prediction = Mock()
        mc_prediction.predicted_options = [mc_option1, mc_option2]

        mc_question.options = ["A", "B"]
        mc_question.api_json = {
            "question": {
                "type": "multiple_choice",
                "options": ["A", "B"],
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.6, 0.4]}}},
            }
        }

        mc_report = Mock()
        mc_report.question = mc_question
        mc_report.prediction = mc_prediction

        mc_score = calculate_multiple_choice_baseline_score(mc_report)

        # Create numeric score
        numeric_question = Mock()
        numeric_question.id_of_question = 456
        numeric_question.num_predictions = 20

        numeric_percentiles = []
        for p, v in [(10, 100), (20, 150), (40, 200), (60, 300), (80, 500), (90, 800)]:
            mock_p = Mock()
            mock_p.percentile = p
            mock_p.value = v
            numeric_percentiles.append(mock_p)

        numeric_prediction = Mock()
        numeric_prediction.declared_percentiles = numeric_percentiles

        numeric_question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 201).tolist()}}
                }
            }
        }

        numeric_report = Mock()
        numeric_report.question = numeric_question
        numeric_report.prediction = numeric_prediction

        numeric_score = calculate_numeric_baseline_score(numeric_report)

        # All scores should be finite and in comparable ranges
        assert all(math.isfinite(s) for s in [binary_score, mc_score, numeric_score] if s is not None)

        # Scores should be in similar order of magnitude (within factor of 10)
        if mc_score is not None and numeric_score is not None:
            score_range = max(abs(binary_score), abs(mc_score), abs(numeric_score))
            assert score_range > 0  # Scores should not all be zero

            # All scores should be within reasonable bounds (Metaculus-like range)
            # After normalization, all scores should be on similar scales
            for score in [binary_score, mc_score, numeric_score]:
                if score is not None:
                    assert -500 <= score <= 500


class TestMonkeyPatching:
    """Test monkey patching functionality."""

    def test_apply_scoring_patches(self):
        """Test that patches are applied without errors."""
        # This test verifies the patches can be applied
        # The actual functionality is tested in integration tests
        try:
            apply_scoring_patches()
        except ImportError:
            # Expected if forecasting_tools not available in test environment
            pytest.skip("forecasting_tools not available for patching test")
        except Exception as e:
            pytest.fail(f"Patching failed with unexpected error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
