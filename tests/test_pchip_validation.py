"""
Tests for PCHIP CDF validation (QA checks that replace forecasting-tools validation).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile


class TestPchipValidation:
    """Test our custom PCHIP CDF validation logic."""

    def create_mock_template_forecaster(self):
        """Create a mock TemplateForecaster for testing validation."""
        # Import here to avoid circular imports in tests
        from main import TemplateForecaster

        mock_llms = {
            "default": MagicMock(),
            "parser": MagicMock(),
            "researcher": MagicMock(),
            "summarizer": MagicMock(),
        }
        return TemplateForecaster(llms=mock_llms, publish_reports_to_metaculus=False)

    def create_dummy_llm(self):
        """Create a dummy LLM for testing."""

        class DummyLLM:
            def __init__(self, reasoning: str):
                self._reasoning = reasoning
                self.model = "dummy-test-model"

            async def invoke(self, prompt: str):
                return self._reasoning

        return DummyLLM(
            "Test reasoning with percentiles:\nPercentile 5: 5.0\nPercentile 10: 10.0\nPercentile 20: 20.0\nPercentile 40: 40.0\nPercentile 60: 60.0\nPercentile 80: 80.0\nPercentile 90: 90.0\nPercentile 95: 95.0"
        )

    def create_mock_question(self, open_upper=False, open_lower=False, upper=100.0, lower=0.0):
        """Create a mock question for testing."""
        return SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            zero_point=None,
            id_of_question=123,
            question_text="Test numeric question",
            background_info="Test background",
            resolution_criteria="Test resolution criteria",
            fine_print="Test fine print",
            unit_of_measure="units",
            page_url="https://example.com/question/123",
        )

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_valid_pchip_cdf_passes_validation(self, mock_format, mock_generate):
        """Test that a valid PCHIP CDF passes all validation checks."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question(open_upper=False, open_lower=False)

        # Create valid CDF (201 points, monotonic, proper spacing)
        valid_cdf = np.linspace(0.0, 1.0, 201).tolist()
        mock_generate.return_value = valid_cdf
        mock_format.return_value = {}

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

        # Should not raise any exceptions
        llm = self.create_dummy_llm()
        with patch("main.structure_output", return_value=percentiles):
            result = await forecaster._run_forecast_on_numeric(question, "test research", llm)
            assert result is not None

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_wrong_length_cdf_fails_validation(self, mock_format, mock_generate):
        """Test that CDF with wrong length fails validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question()

        # Wrong length (200 instead of 201)
        invalid_cdf = np.linspace(0.0, 1.0, 200).tolist()
        mock_generate.return_value = invalid_cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to PCHIP validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_invalid_probabilities_fail_validation(self, mock_format, mock_generate):
        """Test that CDF with probabilities outside [0,1] fails validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question()

        # Invalid probabilities (some > 1.0)
        invalid_cdf = np.linspace(0.0, 1.2, 201).tolist()  # Goes up to 1.2
        mock_generate.return_value = invalid_cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_non_monotonic_cdf_fails_validation(self, mock_format, mock_generate):
        """Test that non-monotonic CDF fails validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question()

        # Non-monotonic CDF
        invalid_cdf = [0.0] * 100 + [0.5] + [0.4] * 100  # Goes down at position 101
        mock_generate.return_value = invalid_cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_step_size_violation_fails_validation(self, mock_format, mock_generate):
        """Test that CDF violating minimum step size fails validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question()

        # Create CDF with step size violation
        cdf = np.linspace(0.0, 0.5, 200).tolist()
        cdf.append(0.5 + 1e-6)  # Very small step < 5e-5
        mock_generate.return_value = cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_max_step_violation_fails_validation(self, mock_format, mock_generate):
        """Test that CDF violating maximum step size fails validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question()

        # Create CDF with max step violation
        cdf = [0.0] * 100 + [0.8] + [1.0] * 100  # Jump of 0.8 > 0.59
        mock_generate.return_value = cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_closed_bound_violations_fail_validation(self, mock_format, mock_generate):
        """Test that closed bound violations fail validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question(open_upper=False, open_lower=False)  # Closed bounds

        # CDF that doesn't start at 0.0 (closed lower bound violation)
        invalid_cdf = np.linspace(0.01, 1.0, 201).tolist()
        mock_generate.return_value = invalid_cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_open_bound_violations_fail_validation(self, mock_format, mock_generate):
        """Test that open bound violations fail validation."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question(open_upper=True, open_lower=True)  # Open bounds

        # CDF that starts too low for open bounds (< 0.001)
        invalid_cdf = np.linspace(0.0005, 0.9995, 201).tolist()  # Starts at 0.0005 < 0.001
        mock_generate.return_value = invalid_cdf
        mock_format.return_value = {}

        percentiles = [Percentile(percentile=0.50, value=50.0)]

        with patch.object(forecaster, "_apply_jitter_and_clamp", return_value=percentiles):
            with patch("main.structure_output", return_value=percentiles):
                with pytest.raises(Exception):  # Should fall back due to validation failure
                    await forecaster._run_forecast_on_numeric(question, "test", forecaster.get_llm("default"))


if __name__ == "__main__":
    pytest.main([__file__])
