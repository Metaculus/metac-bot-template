"""
Tests for numeric CDF smoothing:
- Cluster spreading in declared percentiles
- Probability-side ramp smoothing for PCHIP CDF
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile


def _make_forecaster():
    from main import TemplateForecaster

    mock_llms = {
        "default": MagicMock(),
        "parser": MagicMock(),
        "researcher": MagicMock(),
        "summarizer": MagicMock(),
    }
    return TemplateForecaster(llms=mock_llms, publish_reports_to_metaculus=False)


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0):
    return SimpleNamespace(
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        upper_bound=upper,
        lower_bound=lower,
        zero_point=None,
        id_of_question=999,
        question_text="Test numeric question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure="units",
        page_url="https://example.com/q/999",
    )


class DummyLLM:
    def __init__(self, reasoning: str = "reasoning"):
        self._reasoning = reasoning
        self.model = "dummy"

    async def invoke(self, prompt: str):
        return self._reasoning


class TestNumericCDFSmoothing:
    @pytest.mark.asyncio
    def test_cluster_spread_basic(self, caplog):
        f = _make_forecaster()
        q = _make_question()

        # 8 declared percentiles with a middle cluster (identical values)
        raw = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.10, value=20.0),
            Percentile(percentile=0.20, value=50.0),
            Percentile(percentile=0.40, value=50.0),
            Percentile(percentile=0.60, value=50.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        caplog.clear()
        caplog.set_level("WARNING")
        adjusted = f._apply_jitter_and_clamp(raw, q)

        vals = [p.value for p in adjusted]
        # Strictly increasing
        assert all(b > a for a, b in zip(vals, vals[1:])), vals
        # Warn about cluster spread
        assert any("Cluster spread applied" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    def test_count_like_spread_uses_larger_delta(self, caplog):
        f = _make_forecaster()
        q = _make_question(lower=0.0, upper=100.0)

        # All values very close to integers (count-like)
        raw = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=10.0),
            Percentile(percentile=0.40, value=11.0),
            Percentile(percentile=0.60, value=11.0),
            Percentile(percentile=0.80, value=12.0),
            Percentile(percentile=0.90, value=12.0),
            Percentile(percentile=0.95, value=12.0),
        ]

        caplog.clear()
        caplog.set_level("WARNING")
        adjusted = f._apply_jitter_and_clamp(raw, q)

        vals = [p.value for p in adjusted]
        # Strictly increasing after adjustment
        assert all(b > a for a, b in zip(vals, vals[1:])), vals
        # Differences within clusters should be at least ~1.0 in count-like case (allow some tolerance)
        diffs = [b - a for a, b in zip(vals, vals[1:])]
        assert any(d >= 0.5 for d in diffs), diffs
        assert any("Cluster spread applied" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    def test_bound_adjacent_cluster(self, caplog):
        f = _make_forecaster()
        # Closed upper bound; cluster near upper
        q = _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0)
        raw = [
            Percentile(percentile=0.05, value=80.0),
            Percentile(percentile=0.10, value=90.0),
            Percentile(percentile=0.20, value=99.5),
            Percentile(percentile=0.40, value=100.0),
            Percentile(percentile=0.60, value=100.0),
            Percentile(percentile=0.80, value=100.0),
            Percentile(percentile=0.90, value=100.0),
            Percentile(percentile=0.95, value=100.0),
        ]

        caplog.clear()
        caplog.set_level("WARNING")
        adjusted = f._apply_jitter_and_clamp(raw, q)
        vals = [p.value for p in adjusted]
        assert all(q.lower_bound <= v <= q.upper_bound for v in vals)
        assert all(b > a for a, b in zip(vals, vals[1:])), vals
        # Should log both correction and cluster spread
        msgs = [rec.message for rec in caplog.records]
        assert any("Corrected numeric distribution" in m for m in msgs)
        assert any("Cluster spread applied" in m for m in msgs)

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_ramp_smoothing_triggered(self, mock_format, mock_generate, caplog):
        f = _make_forecaster()
        q = _make_question(open_upper=False, open_lower=False)

        # Construct a CDF with very tiny min delta to trigger smoothing
        base = np.linspace(0.0, 1.0, 201)
        base[100:105] = base[100] + np.linspace(0, 1e-8, 5)  # tiny steps
        mock_generate.return_value = base.tolist()
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

        caplog.clear()
        caplog.set_level("WARNING")
        with patch("main.structure_output", return_value=percentiles):
            result = await f._run_forecast_on_numeric(q, "test research", DummyLLM())
            assert result is not None

        # Verify smoothing was logged
        assert any("CDF ramp smoothing" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    @patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format")
    async def test_noop_when_cdf_is_good(self, mock_format, mock_generate, caplog):
        f = _make_forecaster()
        q = _make_question(open_upper=False, open_lower=False)

        good = np.linspace(0.0, 1.0, 201).tolist()
        mock_generate.return_value = good
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

        caplog.clear()
        caplog.set_level("WARNING")
        with patch("main.structure_output", return_value=percentiles):
            result = await f._run_forecast_on_numeric(q, "test research", DummyLLM())
            assert result is not None

        # No cluster spread or ramp smoothing messages expected
        msgs = [rec.message for rec in caplog.records]
        assert not any("Cluster spread applied" in m for m in msgs)
        assert not any("CDF ramp smoothing" in m for m in msgs)
