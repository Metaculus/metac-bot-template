"""
Tests for PCHIP-based CDF construction.

Based on the panchul implementation with comprehensive validation.
"""

from typing import Dict, List

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.pchip_cdf import (
    _safe_cdf_bounds,
    enforce_strict_increasing,
    generate_pchip_cdf,
    percentiles_to_pchip_format,
)


class TestPercentilesToPchipFormat:
    """Test conversion from forecasting-tools format to PCHIP input format."""

    def test_basic_conversion(self):
        """Test basic percentile conversion."""
        percentiles = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=90.0),
        ]

        result = percentiles_to_pchip_format(percentiles)
        expected = {5.0: 10.0, 50.0: 50.0, 95.0: 90.0}

        assert result == expected

    def test_eight_percentile_conversion(self):
        """Test conversion of our standard 8-percentile set."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        result = percentiles_to_pchip_format(percentiles)
        expected = {5.0: 5.0, 10.0: 10.0, 20.0: 20.0, 40.0: 40.0, 60.0: 60.0, 80.0: 80.0, 90.0: 90.0, 95.0: 95.0}

        assert result == expected


class TestEnforceStrictIncreasing:
    """Test enforcement of strictly increasing values."""

    def test_already_increasing(self):
        """Test with already strictly increasing values."""
        data = {10.0: 1.0, 20.0: 2.0, 30.0: 3.0}
        result = enforce_strict_increasing(data)
        assert result == data

    def test_duplicate_values(self):
        """Test with duplicate values that need jittering."""
        data = {10.0: 5.0, 20.0: 5.0, 30.0: 5.0}  # All same value
        result = enforce_strict_increasing(data)

        # Values should be strictly increasing
        values = [result[k] for k in sorted(result.keys())]
        assert all(a < b for a, b in zip(values[:-1], values[1:]))

    def test_mixed_duplicates(self):
        """Test with some duplicates and some valid values."""
        data = {10.0: 1.0, 20.0: 3.0, 30.0: 3.0, 40.0: 5.0}
        result = enforce_strict_increasing(data)

        values = [result[k] for k in sorted(result.keys())]
        assert all(a < b for a, b in zip(values[:-1], values[1:]))
        assert result[10.0] == 1.0  # First value unchanged
        assert result[40.0] == 5.0  # Last value unchanged


class TestSafeCdfBounds:
    """Test CDF boundary constraint enforcement."""

    def test_open_bounds_clamping(self):
        """Test open bounds are clamped to [0.001, 0.999]."""
        cdf = np.array([0.0, 0.1, 0.9, 1.0])
        result = _safe_cdf_bounds(cdf, open_lower=True, open_upper=True, min_step=5e-5)

        assert result[0] >= 0.001
        assert result[-1] <= 0.999

    def test_closed_bounds_preserved(self):
        """Test closed bounds preserve [0, 1] values."""
        cdf = np.array([0.0, 0.1, 0.9, 1.0])
        result = _safe_cdf_bounds(cdf, open_lower=False, open_upper=False, min_step=5e-5)

        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_max_jump_enforcement(self):
        """Test that steps > 0.59 are redistributed."""
        cdf = np.array([0.0, 0.1, 0.8, 1.0])  # Step of 0.7 > 0.59
        result = _safe_cdf_bounds(cdf, open_lower=False, open_upper=False, min_step=5e-5)

        steps = np.diff(result)
        assert all(step <= 0.59 + 1e-6 for step in steps), f"Steps: {steps}"


class TestGeneratePchipCdf:
    """Test the main PCHIP CDF generation function."""

    def test_basic_generation(self):
        """Test basic CDF generation with simple percentiles."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0  # Closed lower bound
        assert cdf[-1] == 1.0  # Closed upper bound
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))  # Monotonic

    def test_open_bounds_generation(self):
        """Test CDF generation with open bounds."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] >= 0.001  # Open lower bound
        assert cdf[-1] <= 0.999  # Open upper bound
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))  # Monotonic

    def test_minimum_step_enforcement(self):
        """Test that minimum step size is enforced."""
        percentiles = {10.0: 1.0, 50.0: 5.0, 90.0: 9.0}
        min_step = 5e-5

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=min_step,
        )

        steps = np.diff(cdf)
        assert all(step >= min_step - 1e-10 for step in steps), f"Min step violated: {np.min(steps)}"

    def test_eight_percentile_realistic(self):
        """Test with realistic 8-percentile data."""
        percentiles = {
            5.0: 100.0,
            10.0: 110.0,
            20.0: 120.0,
            40.0: 140.0,
            60.0: 160.0,
            80.0: 180.0,
            90.0: 190.0,
            95.0: 195.0,
        }

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=200.0,
            lower_bound=50.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0

        # Verify monotonicity
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))

        # Verify minimum step
        steps = np.diff(cdf)
        assert all(step >= 5e-5 - 1e-10 for step in steps)

    def test_log_space_transform(self):
        """Test that log-space transform is applied when appropriate."""
        percentiles = {10.0: 1.0, 50.0: 10.0, 90.0: 100.0}  # All positive, good for log

        # Should work without error and produce smooth results
        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=1000.0,
            lower_bound=0.1,
            zero_point=None,  # No zero_point should enable log transform
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))

    def test_zero_point_handling(self):
        """Test that zero_point prevents log transform."""
        percentiles = {10.0: 1.0, 50.0: 10.0, 90.0: 100.0}

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=1000.0,
            lower_bound=0.1,
            zero_point=0.0,  # Should prevent log transform
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))

    def test_duplicate_values_handling(self):
        """Test handling of duplicate percentile values."""
        percentiles = {10.0: 5.0, 50.0: 5.0, 90.0: 5.0}  # All same value

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))

    def test_error_handling_empty_percentiles(self):
        """Test error handling for empty percentiles."""
        with pytest.raises(ValueError, match="Empty percentile values dictionary"):
            generate_pchip_cdf(
                percentile_values={}, open_upper_bound=False, open_lower_bound=False, upper_bound=10.0, lower_bound=0.0
            )

    def test_error_handling_invalid_bounds(self):
        """Test error handling for invalid bounds."""
        percentiles = {50.0: 5.0}

        with pytest.raises(ValueError, match="Upper bound.*must be greater than lower bound"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=0.0,
                lower_bound=10.0,  # Invalid: lower > upper
            )

    def test_error_handling_insufficient_percentiles(self):
        """Test error handling for too few percentiles."""
        percentiles = {50.0: 5.0}  # Only 1 percentile

        with pytest.raises(ValueError, match="Need at least 2 valid percentile points"):
            generate_pchip_cdf(
                percentile_values=percentiles,
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=10.0,
                lower_bound=0.0,
            )

    def test_realistic_numeric_distribution(self):
        """Test with realistic numeric distribution (right-skewed)."""
        # Simulating a right-skewed distribution like "days until event"
        percentiles = {
            5.0: 1.0,  # P5: 1 day
            10.0: 3.0,  # P10: 3 days
            20.0: 7.0,  # P20: 1 week
            40.0: 30.0,  # P40: 1 month
            60.0: 90.0,  # P60: 3 months
            80.0: 365.0,  # P80: 1 year
            90.0: 730.0,  # P90: 2 years
            95.0: 1095.0,  # P95: 3 years
        }

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=True,
            open_lower_bound=True,
            upper_bound=3650.0,  # 10 years
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        # Open bounds should be >= 0.001 and <= 0.999, but aggressive step enforcement
        # might result in slightly different values
        assert cdf[0] >= 0.001
        assert cdf[-1] <= 0.999

        # Should be smooth and monotonic despite the skewed nature
        steps = np.diff(cdf)
        assert all(step >= 5e-5 - 1e-10 for step in steps)
        assert all(step <= 0.59 + 1e-6 for step in steps)

    def test_discrete_style_distribution(self):
        """Test with discrete-style distribution (integer values)."""
        percentiles = {5.0: 1.0, 10.0: 2.0, 20.0: 3.0, 40.0: 5.0, 60.0: 7.0, 80.0: 10.0, 90.0: 12.0, 95.0: 15.0}

        cdf = generate_pchip_cdf(
            percentile_values=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=20.0,
            lower_bound=0.0,
            zero_point=None,
        )

        assert len(cdf) == 201
        assert cdf[0] == 0.0
        assert cdf[-1] == 1.0
        assert all(a <= b for a, b in zip(cdf[:-1], cdf[1:]))


if __name__ == "__main__":
    pytest.main([__file__])
