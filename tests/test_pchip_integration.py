"""
Integration tests for PCHIP CDF with forecasting-tools NumericDistribution.

Tests that our CDF override approach works correctly with the framework.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from metaculus_bot.pchip_cdf import generate_pchip_cdf, percentiles_to_pchip_format


class TestPchipIntegration:
    """Test integration between PCHIP CDF and NumericDistribution."""

    def test_cdf_override_format(self):
        """Test that our CDF override produces the expected format."""
        # Create test percentiles (our 8-percentile standard)
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

        # Create a mock question
        question = SimpleNamespace(
            open_upper_bound=False, open_lower_bound=False, upper_bound=100.0, lower_bound=0.0, zero_point=None
        )

        # Generate PCHIP CDF
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Create override exactly as in main.py
        x_vals = np.linspace(question.lower_bound, question.upper_bound, len(pchip_cdf))
        pchip_percentile_objects = [
            Percentile(percentile=prob_val, value=question_val) for question_val, prob_val in zip(x_vals, pchip_cdf)
        ]

        # Validate the format
        assert len(pchip_percentile_objects) == 201
        assert all(isinstance(p, Percentile) for p in pchip_percentile_objects)

        # Validate probability values are in [0,1]
        prob_values = [p.percentile for p in pchip_percentile_objects]
        assert all(0.0 <= p <= 1.0 for p in prob_values)

        # Validate question values are in bounds
        question_values = [p.value for p in pchip_percentile_objects]
        assert all(question.lower_bound <= v <= question.upper_bound for v in question_values)

        # Validate monotonicity (CDF requirements)
        assert all(a <= b for a, b in zip(prob_values[:-1], prob_values[1:]))
        assert all(a <= b for a, b in zip(question_values[:-1], question_values[1:]))

    def test_spacing_assertion_compliance(self):
        """Test that our PCHIP CDF satisfies the 5e-5 spacing requirement."""
        percentiles = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=90.0),
        ]

        question = SimpleNamespace(
            open_upper_bound=False, open_lower_bound=False, upper_bound=100.0, lower_bound=0.0, zero_point=None
        )

        # Generate PCHIP CDF
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Check that spacing assertion would pass
        for i in range(len(pchip_cdf) - 1):
            spacing = abs(pchip_cdf[i + 1] - pchip_cdf[i])
            assert spacing >= 5e-05, f"Spacing violation at index {i}: {spacing}"

    def test_pchip_subclass_approach_works(self):
        """Test that our PchipNumericDistribution subclass approach works."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        # Generate PCHIP CDF (simulating our main.py logic)
        question = SimpleNamespace(
            open_upper_bound=False, open_lower_bound=False, upper_bound=100.0, lower_bound=0.0, zero_point=None
        )

        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Create the subclass exactly as in main.py
        class PchipNumericDistribution(NumericDistribution):
            def __init__(self, pchip_cdf_values, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._pchip_cdf_values = pchip_cdf_values

            @property
            def cdf(self) -> list[Percentile]:
                """Return PCHIP-generated CDF as Percentile objects."""
                # Create the value axis (201 points from lower to upper bound)
                x_vals = np.linspace(self.lower_bound, self.upper_bound, len(self._pchip_cdf_values))

                # Create Percentile objects with correct mapping:
                # _pchip_cdf_values contains the probability values (0-1)
                # x_vals contains the corresponding question values
                return [
                    Percentile(percentile=prob_val, value=question_val)
                    for question_val, prob_val in zip(x_vals, self._pchip_cdf_values)
                ]

        prediction = PchipNumericDistribution(
            pchip_cdf_values=pchip_cdf,
            declared_percentiles=percentiles,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            cdf_size=201,
        )

        # Test that the subclass works
        result_cdf = prediction.cdf
        assert len(result_cdf) == 201
        assert all(isinstance(p, Percentile) for p in result_cdf)

        # Test that spacing assertion would pass
        for i in range(len(result_cdf) - 1):
            spacing = abs(result_cdf[i + 1].percentile - result_cdf[i].percentile)
            assert spacing >= 5e-05, f"Spacing violation at index {i}: {spacing}"

    def test_problematic_distribution_case(self):
        """Test with a distribution similar to the one that was failing."""
        # Create a distribution that would likely fail the original spacing check
        percentiles = [
            Percentile(percentile=0.05, value=70.0),
            Percentile(percentile=0.10, value=70.5),  # Very close values
            Percentile(percentile=0.20, value=71.0),
            Percentile(percentile=0.40, value=72.0),
            Percentile(percentile=0.60, value=73.0),
            Percentile(percentile=0.80, value=74.0),
            Percentile(percentile=0.90, value=74.5),  # Close values again
            Percentile(percentile=0.95, value=75.0),
        ]

        question = SimpleNamespace(
            open_upper_bound=False, open_lower_bound=False, upper_bound=100.0, lower_bound=0.0, zero_point=None
        )

        # This should work with PCHIP even though values are close
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Verify no spacing violations
        for i in range(len(pchip_cdf) - 1):
            spacing = abs(pchip_cdf[i + 1] - pchip_cdf[i])
            assert spacing >= 5e-05, f"PCHIP failed to fix spacing at index {i}: {spacing}"


if __name__ == "__main__":
    pytest.main([__file__])
