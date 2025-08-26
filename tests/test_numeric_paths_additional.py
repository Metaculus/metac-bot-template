"""
Additional tests to cover PCHIP fallback/smoothing paths and input validation.
Concise scenarios to increase confidence in complex numeric forecast flow.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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


def _make_question(**overrides):
    opts = dict(
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        id_of_question=4242,
        question_text="num?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure="units",
        page_url="https://example/q/4242",
        cdf_size=201,
    )
    opts.update(overrides)
    return SimpleNamespace(**opts)


class DummyLLM:
    def __init__(self, reasoning: str = "r"):
        self._reasoning = reasoning
        self.model = "dummy"

    async def invoke(self, prompt: str):
        return self._reasoning


@pytest.mark.asyncio
@patch("metaculus_bot.pchip_cdf.generate_pchip_cdf", side_effect=RuntimeError("boom"))
@patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_pchip_fallback_success(mock_format, mock_generate, caplog):
    f = _make_forecaster()
    q = _make_question()

    # Valid 8-percentile set
    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95], [5, 10, 20, 40, 60, 80, 90, 95])
    ]

    with patch("main.structure_output", return_value=plist):
        caplog.clear()
        caplog.set_level("WARNING")
        result = await f._run_forecast_on_numeric(q, "", DummyLLM())

    # Fallback warning emitted
    assert any("PCHIP CDF construction failed" in rec.message for rec in caplog.records)

    # Fallback NumericDistribution returns a cdf that is monotone
    c = result.prediction_value.cdf  # type: ignore[attr-defined]
    probs = [p.percentile for p in c]
    assert all(a <= b for a, b in zip(probs[:-1], probs[1:]))


@pytest.mark.asyncio
@patch("metaculus_bot.pchip_cdf.generate_pchip_cdf", side_effect=RuntimeError("boom"))
@patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_pchip_fallback_failure_diagnostics(mock_format, mock_generate, caplog):
    f = _make_forecaster()
    q = _make_question()

    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95], [1, 1, 1, 1, 1, 1, 1, 1])
    ]

    # Force fallback NumericDistribution.cdf to raise via a fake class
    class FakeND:
        def __init__(self, *args, **kwargs):
            self.declared_percentiles = plist

        @property
        def cdf(self):  # noqa: D401
            raise AssertionError("Percentiles at indices are too close")

    with patch("main.structure_output", return_value=plist), patch("main.NumericDistribution", FakeND):
        caplog.clear()
        caplog.set_level("ERROR")
        with pytest.raises(AssertionError):
            await f._run_forecast_on_numeric(q, "", DummyLLM())

    # Rich diagnostics logged
    msgs = [r.message for r in caplog.records]
    assert any("Numeric CDF spacing assertion" in m for m in msgs)
    assert any("Bounds=" in m and "Declared percentiles" in m for m in msgs)


@pytest.mark.asyncio
@patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_smoothing_respects_open_bounds(mock_format, caplog):
    f = _make_forecaster()
    # Open bounds question
    q = _make_question(open_upper_bound=True, open_lower_bound=True)

    # Tiny deltas to trigger smoothing
    base = np.linspace(0.0, 1.0, 201)
    base[50:55] = base[50] + np.linspace(0, 1e-8, 5)

    with patch("metaculus_bot.pchip_cdf.generate_pchip_cdf", return_value=base.tolist()):
        plist = [
            Percentile(percentile=p, value=v)
            for p, v in zip([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95], [5, 10, 20, 40, 60, 80, 90, 95])
        ]
        with patch("main.structure_output", return_value=plist):
            caplog.clear()
            caplog.set_level("WARNING")
            result = await f._run_forecast_on_numeric(q, "", DummyLLM())

    # Smoothing log
    assert any("CDF ramp smoothing" in rec.message for rec in caplog.records)

    # Endpoints pinned for open bounds
    c = result.prediction_value.cdf  # type: ignore[attr-defined]
    probs = [p.percentile for p in c]
    assert probs[0] >= 0.001
    assert probs[-1] <= 0.999


@pytest.mark.asyncio
async def test_numeric_percentile_set_validation():
    f = _make_forecaster()
    q = _make_question()

    # 8 items but wrong set (0.50 instead of 0.60)
    bad = [
        Percentile(percentile=p, value=v)
        for p, v in zip([0.05, 0.10, 0.20, 0.40, 0.50, 0.80, 0.90, 0.95], [5, 10, 20, 40, 50, 80, 90, 95])
    ]

    with patch("main.structure_output", return_value=bad):
        with pytest.raises(Exception):  # pydantic ValidationError via from_exception_data
            await f._run_forecast_on_numeric(q, "", DummyLLM())


@pytest.mark.asyncio
@patch("metaculus_bot.pchip_cdf.generate_pchip_cdf")
@patch("metaculus_bot.pchip_cdf.percentiles_to_pchip_format", return_value={})
async def test_discrete_zero_point_override(mock_format, mock_generate):
    f = _make_forecaster()
    # Discrete (non-201) and zero_point provided → should pass zero_point=None into pchip
    q = _make_question(cdf_size=101, zero_point=0.0)

    # Return a valid CDF to avoid fallback
    mock_generate.return_value = np.linspace(0.0, 1.0, 201).tolist()

    plist = [
        Percentile(percentile=p, value=v)
        for p, v in zip([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95], [5, 10, 20, 40, 60, 80, 90, 95])
    ]

    with patch("main.structure_output", return_value=plist):
        await f._run_forecast_on_numeric(q, "", DummyLLM())

    # Capture the call arguments to ensure zero_point=None was used
    args, kwargs = mock_generate.call_args
    assert kwargs.get("zero_point", "sentinel") is None


def test_lower_bound_adjacent_cluster(caplog):
    from main import TemplateForecaster

    f = _make_forecaster()
    # Closed lower bound; cluster near lower
    q = _make_question(open_upper_bound=False, open_lower_bound=False, lower_bound=0.0, upper_bound=100.0)
    raw = [
        Percentile(percentile=0.05, value=0.0),
        Percentile(percentile=0.10, value=0.0),
        Percentile(percentile=0.20, value=0.0),
        Percentile(percentile=0.40, value=0.1),
        Percentile(percentile=0.60, value=10.0),
        Percentile(percentile=0.80, value=20.0),
        Percentile(percentile=0.90, value=30.0),
        Percentile(percentile=0.95, value=40.0),
    ]

    caplog.clear()
    caplog.set_level("WARNING")
    adjusted = f._apply_jitter_and_clamp(raw, q)

    vals = [p.value for p in adjusted]
    assert all(q.lower_bound <= v <= q.upper_bound for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:])), vals
    msgs = [rec.message for rec in caplog.records]
    assert any("Cluster spread applied" in m for m in msgs)
    assert any("Corrected numeric distribution" in m for m in msgs)


@pytest.mark.asyncio
async def test_binary_parse_additional_instructions_capture():
    from forecasting_tools import BinaryQuestion, GeneralLlm

    from main import TemplateForecaster

    bot = TemplateForecaster(llms={"default": "m", "parser": "p", "researcher": "r", "summarizer": "s"})
    q = MagicMock(spec=BinaryQuestion)
    q.page_url = "http://ex"
    q.question_text = "?"
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 7

    # Dummy forecaster LLM
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="rationale")

    seen = {}

    class _Bin:
        def __init__(self, val):
            self.prediction_in_decimal = val

    async def _fake_structure_output(*args, **kwargs):
        seen["additional_instructions"] = kwargs.get("additional_instructions", "")
        return _Bin(0.5)

    with patch("main.structure_output", _fake_structure_output):
        await bot._run_forecast_on_binary(q, "", llm)

    ai = seen.get("additional_instructions", "")
    assert "decimal in [0,1]" in ai
    assert "NN%" in ai and "NN/100" in ai
