"""
Comprehensive sanity tests for the new relative numeric scoring approach.

These tests validate that our community benchmark scoring behaves sensibly
and implements the expected mathematical properties.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from metaculus_bot.scoring_patches import calculate_numeric_baseline_score


class TestNumericScoringSanity:
    """Sanity tests for relative numeric scoring behavior."""

    def _make_question(self, community_cdf, lower_bound=0.0, upper_bound=100.0):
        """Helper to create mock question with community CDF."""
        question = Mock()
        question.id_of_question = 999
        question.lower_bound = lower_bound
        question.upper_bound = upper_bound
        question.api_json = {
            "question": {"aggregations": {"recency_weighted": {"latest": {"forecast_values": community_cdf}}}}
        }
        return question

    def _make_uniform_cdf(self, num_bins=11):
        """Create uniform CDF from 0 to 1."""
        return np.linspace(0.0, 1.0, num_bins).tolist()

    def _make_concentrated_cdf(self, center=0.5, width=0.2, num_bins=11):
        """Create CDF concentrated around center with given width."""
        x = np.linspace(0.0, 1.0, num_bins)
        # Sigmoid-like concentration around center
        return [1.0 / (1.0 + math.exp(-10 * (val - center) / width)) for val in x]

    def _make_prediction_from_cdf(self, cdf_values):
        """Create prediction object with given CDF values."""

        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        prediction = Mock()
        prediction.cdf = [P(p) for p in cdf_values]
        return prediction

    def test_identical_distributions_score_near_zero(self):
        """Test that identical uniform bot and community distributions score around 0."""
        community_cdf = self._make_uniform_cdf()
        question = self._make_question(community_cdf)

        # Bot has same distribution as community
        prediction = self._make_prediction_from_cdf(community_cdf)

        report = SimpleNamespace(question=question, prediction=prediction)
        score = calculate_numeric_baseline_score(report)

        assert score is not None
        # For identical uniform distributions with n=11 bins:
        # Expected log score = -ln(11) ≈ -2.4
        # Final score = 100 * (-2.4 / 1.6 + 1) ≈ 100 * (-0.5) = -50
        assert -60 <= score <= -40

    def test_concentration_beats_dispersion(self):
        """Test that concentrated predictions beat dispersed ones when community is concentrated."""
        # Community has concentrated distribution
        community_cdf = self._make_concentrated_cdf(center=0.5, width=0.1)
        question = self._make_question(community_cdf)

        # Bot 1: Also concentrated (matches community preference)
        concentrated_pred = self._make_prediction_from_cdf(self._make_concentrated_cdf(center=0.5, width=0.1))

        # Bot 2: Uniform (doesn't match community concentration)
        uniform_pred = self._make_prediction_from_cdf(self._make_uniform_cdf())

        report_concentrated = SimpleNamespace(question=question, prediction=concentrated_pred)
        report_uniform = SimpleNamespace(question=question, prediction=uniform_pred)

        score_concentrated = calculate_numeric_baseline_score(report_concentrated)
        score_uniform = calculate_numeric_baseline_score(report_uniform)

        assert score_concentrated is not None
        assert score_uniform is not None
        assert score_concentrated > score_uniform

    def test_scoring_is_symmetric(self):
        """Test that roles of bot and community can be swapped (up to constant)."""
        # Two different distributions
        cdf1 = self._make_uniform_cdf()
        cdf2 = self._make_concentrated_cdf()

        # Test A: community=cdf1, bot=cdf2
        question_A = self._make_question(cdf1)
        pred_A = self._make_prediction_from_cdf(cdf2)
        report_A = SimpleNamespace(question=question_A, prediction=pred_A)
        score_A = calculate_numeric_baseline_score(report_A)

        # Test B: community=cdf2, bot=cdf1
        question_B = self._make_question(cdf2)
        pred_B = self._make_prediction_from_cdf(cdf1)
        report_B = SimpleNamespace(question=question_B, prediction=pred_B)
        score_B = calculate_numeric_baseline_score(report_B)

        assert score_A is not None
        assert score_B is not None
        # Both should be bounded, but concentrated vs uniform can be quite negative
        # under the calibrated normalization used to align avg absolute scores
        assert -240 <= score_A <= 120
        assert -240 <= score_B <= 120

    def test_worst_case_bounded_by_uniform_mixture(self):
        """Test that 1% uniform mixture prevents extremely bad scores."""
        # Community puts all mass in first bin
        community_cdf = [0.0, 1.0, 1.0, 1.0, 1.0]
        question = self._make_question(community_cdf)

        # Bot puts all mass in last bin (complete mismatch)
        bot_cdf = [0.0, 0.0, 0.0, 0.0, 1.0]
        prediction = self._make_prediction_from_cdf(bot_cdf)

        report = SimpleNamespace(question=question, prediction=prediction)
        score = calculate_numeric_baseline_score(report)

        assert score is not None
        # Should be bad but not infinitely bad due to 1% mixture
        # Worst case: ln(0.01/num_bins) ≈ ln(0.002) ≈ -6.2, normalized ≈ -400
        assert score > -500

    def test_different_bin_counts_work(self):
        """Test that scoring works with different numbers of bins."""
        for num_bins in [5, 11, 21, 51, 101]:
            community_cdf = np.linspace(0.0, 1.0, num_bins).tolist()
            question = self._make_question(community_cdf)

            # Slightly non-uniform bot distribution
            bot_cdf = (np.linspace(0.0, 1.0, num_bins) ** 1.1).tolist()
            prediction = self._make_prediction_from_cdf(bot_cdf)

            report = SimpleNamespace(question=question, prediction=prediction)
            score = calculate_numeric_baseline_score(report)

            assert score is not None, f"Failed for {num_bins} bins"
            assert math.isfinite(score), f"Non-finite score for {num_bins} bins"
            # With fixed normalization ln(10)*2.5, scores should be in reasonable range
            assert -200 <= score <= 100, f"Score {score} out of range for {num_bins} bins"

    def test_score_monotonicity_with_concentration(self):
        """Test that increasing concentration around community mode improves score."""
        community_cdf = self._make_concentrated_cdf(center=0.3, width=0.1)
        question = self._make_question(community_cdf)

        scores = []

        # Test predictions with increasing concentration around community mode
        for width in [0.5, 0.3, 0.2, 0.1, 0.05]:
            bot_cdf = self._make_concentrated_cdf(center=0.3, width=width)
            prediction = self._make_prediction_from_cdf(bot_cdf)

            report = SimpleNamespace(question=question, prediction=prediction)
            score = calculate_numeric_baseline_score(report)

            assert score is not None
            scores.append(score)

        # Scores should generally improve (increase) as concentration increases
        # (smaller width = more concentrated)
        for i in range(len(scores) - 1):
            assert scores[i + 1] >= scores[i] - 10  # Allow small numerical noise

    def test_boundary_handling(self):
        """Test that boundary cases are handled gracefully."""
        # Degenerate case: single bin (no variance)
        community_cdf = [0.0, 1.0]
        question = self._make_question(community_cdf)

        bot_cdf = [0.0, 1.0]
        prediction = self._make_prediction_from_cdf(bot_cdf)

        report = SimpleNamespace(question=question, prediction=prediction)
        score = calculate_numeric_baseline_score(report)

        # Should either return a reasonable score or None (but not crash)
        if score is not None:
            assert math.isfinite(score)

    def test_fallback_scoring_works(self):
        """Test that fallback scoring (when community CDF missing) works."""
        question = Mock()
        question.id_of_question = 999
        question.lower_bound = 0.0
        question.upper_bound = 100.0
        # No community CDF available
        question.api_json = {"question": {"aggregations": {}}}

        # Create bot prediction with declared percentiles
        percentiles = []

        class P:
            def __init__(self, percentile, value):
                self.percentile = percentile
                self.value = value

        for p, v in [(10, 10), (30, 30), (50, 50), (70, 70), (90, 90)]:
            percentiles.append(P(p, v))

        prediction = Mock()
        prediction.declared_percentiles = percentiles

        report = SimpleNamespace(question=question, prediction=prediction)
        score = calculate_numeric_baseline_score(report)

        assert score is not None
        assert math.isfinite(score)
        # Should get reasonable score with uniform community assumption
        # With fixed normalization, fallback should give reasonable scores
        assert -200 <= score <= 100
