"""
Unit tests for bounds clamping utilities.

Tests for bounds clamping and diagnostic functions extracted from main.py.
"""

from types import SimpleNamespace

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.bounds_clamping import (
    calculate_bounds_buffer,
    clamp_values_to_bounds,
    log_cluster_spreading_summary,
    log_corrections_summary,
    log_heavy_clamping_diagnostics,
)


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0):
    return SimpleNamespace(
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        upper_bound=upper,
        lower_bound=lower,
        id_of_question=999,
        page_url="https://example.com/q/999",
    )


class TestBoundsClamping:
    """Test bounds clamping and diagnostic functions."""

    def test_calculate_bounds_buffer_large_range(self):
        """Test buffer calculation for large range (> 100)."""
        question = _make_question(lower=0.0, upper=200.0)  # Range = 200
        buffer = calculate_bounds_buffer(question)
        assert buffer == 1.0  # Should be 1.0 for large ranges

    def test_calculate_bounds_buffer_small_range(self):
        """Test buffer calculation for small range (<= 100)."""
        question = _make_question(lower=0.0, upper=50.0)  # Range = 50
        buffer = calculate_bounds_buffer(question)
        expected = 50.0 * 0.01  # Should be 1% of range
        assert buffer == expected

    def test_clamp_values_to_bounds_no_violations(self):
        """Test clamping when no values violate bounds."""
        values = [10.0, 20.0, 30.0]
        percentiles = [
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.30, value=30.0),
        ]
        question = _make_question(open_lower=False, open_upper=False, lower=0.0, upper=100.0)
        buffer = 1.0

        result, corrections_made = clamp_values_to_bounds(values.copy(), percentiles, question, buffer)

        assert not corrections_made
        assert result == values  # Should be unchanged

    def test_clamp_values_to_bounds_lower_violation_within_tolerance(self):
        """Test clamping for lower bound violation within tolerance."""
        values = [-0.5, 20.0, 30.0]  # First value below lower bound
        percentiles = [
            Percentile(percentile=0.10, value=-0.5),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.30, value=30.0),
        ]
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)
        buffer = 1.0  # Violation is within tolerance

        result, corrections_made = clamp_values_to_bounds(values.copy(), percentiles, question, buffer)

        assert corrections_made
        assert result[0] == question.lower_bound + buffer  # Should be clamped
        assert result[1:] == values[1:]  # Others unchanged

    def test_clamp_values_to_bounds_upper_violation_within_tolerance(self):
        """Test clamping for upper bound violation within tolerance."""
        values = [10.0, 20.0, 100.5]  # Last value above upper bound
        percentiles = [
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.30, value=100.5),
        ]
        question = _make_question(open_upper=False, upper=100.0)
        buffer = 1.0  # Violation is within tolerance

        result, corrections_made = clamp_values_to_bounds(values.copy(), percentiles, question, buffer)

        assert corrections_made
        assert result[0:2] == values[0:2]  # Others unchanged
        assert result[2] == question.upper_bound - buffer  # Should be clamped

    def test_clamp_values_to_bounds_violation_exceeds_tolerance(self):
        """Test that violations exceeding tolerance raise ValueError."""
        values = [-5.0, 20.0, 30.0]  # Large violation of lower bound
        percentiles = [
            Percentile(percentile=0.10, value=-5.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.30, value=30.0),
        ]
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)
        buffer = 1.0  # Violation exceeds tolerance

        with pytest.raises(ValueError) as exc_info:
            clamp_values_to_bounds(values.copy(), percentiles, question, buffer)

        assert "too far below lower bound" in str(exc_info.value)

    def test_clamp_values_to_bounds_open_bounds_no_clamping(self):
        """Test that open bounds don't get clamped."""
        values = [-1.0, 20.0, 101.0]  # Values outside bounds
        percentiles = [
            Percentile(percentile=0.10, value=-1.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.30, value=101.0),
        ]
        question = _make_question(open_lower=True, open_upper=True, lower=0.0, upper=100.0)
        buffer = 1.0

        result, corrections_made = clamp_values_to_bounds(values.copy(), percentiles, question, buffer)

        assert not corrections_made
        assert result == values  # Should be unchanged for open bounds

    def test_log_heavy_clamping_diagnostics_no_heavy_clamping(self, caplog):
        """Test that no warning is logged when clamping is light."""
        modified_values = [1.0, 20.0, 30.0]  # Only one value near bound
        original_values = [0.0, 20.0, 30.0]
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)
        buffer = 1.0

        caplog.clear()
        log_heavy_clamping_diagnostics(modified_values, original_values, question, buffer)

        # Should not log warning (only 1/3 = 33% clamped, below 50% threshold)
        assert not any("Heavy bound clamping" in record.message for record in caplog.records)

    def test_log_heavy_clamping_diagnostics_heavy_lower(self, caplog):
        """Test warning for heavy lower bound clamping."""
        # More than 50% clamped to lower bound
        modified_values = [1.0, 1.0, 30.0]  # 2/3 near lower bound
        original_values = [0.0, 0.0, 30.0]
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)
        buffer = 1.0

        caplog.clear()
        caplog.set_level("WARNING")
        log_heavy_clamping_diagnostics(modified_values, original_values, question, buffer)

        # Should log heavy clamping warning
        assert any("Heavy bound clamping" in record.message for record in caplog.records)

    def test_log_corrections_summary_with_corrections(self, caplog):
        """Test logging when corrections were made."""
        modified_values = [1.0, 20.0, 30.0]
        original_values = [0.0, 20.0, 30.0]  # First value changed
        question = _make_question()

        caplog.clear()
        caplog.set_level("WARNING")
        log_corrections_summary(modified_values, original_values, question, corrections_made=True)

        assert any("Corrected numeric distribution" in record.message for record in caplog.records)

    def test_log_corrections_summary_no_corrections(self, caplog):
        """Test no logging when no corrections were made."""
        values = [10.0, 20.0, 30.0]
        question = _make_question()

        caplog.clear()
        log_corrections_summary(values, values, question, corrections_made=False)

        assert not any("Corrected numeric distribution" in record.message for record in caplog.records)

    def test_log_cluster_spreading_summary_with_clusters(self, caplog):
        """Test logging of cluster spreading summary."""
        modified_values = [10.0, 20.1, 20.2, 30.0]  # Spread cluster
        original_values = [10.0, 20.0, 20.0, 30.0]  # Original cluster
        question = _make_question()

        caplog.clear()
        caplog.set_level("WARNING")
        log_cluster_spreading_summary(
            modified_values, original_values, question, clusters_applied=1, spread_delta=0.1, count_like=False
        )

        assert any("Cluster spread applied" in record.message for record in caplog.records)

    def test_log_cluster_spreading_summary_no_clusters(self, caplog):
        """Test no logging when no clusters were spread."""
        values = [10.0, 20.0, 30.0]
        question = _make_question()

        caplog.clear()
        log_cluster_spreading_summary(values, values, question, clusters_applied=0, spread_delta=0.1, count_like=False)

        assert not any("Cluster spread applied" in record.message for record in caplog.records)
