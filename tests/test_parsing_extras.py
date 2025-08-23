from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_binary_parsing_clamps_extremes():
    bot = TemplateForecaster(llms={"default": "mock", "parser": "mock", "researcher": "mock", "summarizer": "mock"})

    # Minimal binary question
    q = MagicMock(spec=BinaryQuestion)
    q.page_url = "http://example.com"
    q.question_text = "?"
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 1

    llm = MagicMock(spec=GeneralLlm)
    llm.model = "parser-test"
    llm.invoke = AsyncMock(return_value="reasoning")

    class _Bin:
        def __init__(self, val: float) -> None:
            self.prediction_in_decimal = val

    # 0.0 gets clamped to 0.01
    with patch("main.structure_output", return_value=_Bin(0.0)):
        res = await bot._run_forecast_on_binary(q, "", llm)
        assert res.prediction_value == 0.01

    # 1.0 gets clamped to 0.99
    with patch("main.structure_output", return_value=_Bin(1.0)):
        res = await bot._run_forecast_on_binary(q, "", llm)
        assert res.prediction_value == 0.99


@pytest.mark.asyncio
async def test_numeric_parsing_raises_on_wrong_count():
    bot = TemplateForecaster(llms={"default": "mock", "parser": "mock", "researcher": "mock", "summarizer": "mock"})

    q = SimpleNamespace(
        question_text="num?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=False,
        open_lower_bound=False,
        lower_bound=0,
        upper_bound=100,
        page_url="http://ex/q",
        zero_point=None,
        id_of_question=2,
        cdf_size=201,
    )

    # Only 5 percentiles returned -> should raise ValidationError
    bad = [Percentile(value=v, percentile=p) for v, p in zip([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.6, 0.8])]

    async def _fake_structure_output(*args, **kwargs):  # noqa: D401
        return bad

    llm = SimpleNamespace(model="dummy")
    llm.invoke = AsyncMock(return_value="rationale")

    with patch("main.structure_output", _fake_structure_output):
        with pytest.raises(ValidationError):
            await bot._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_parser_llm_used_for_structured_output():
    bot = TemplateForecaster(llms={"default": "mock", "parser": "mock", "researcher": "mock", "summarizer": "mock"})

    sentinel_parser_model = object()
    original_get_llm = bot.get_llm
    bot.get_llm = MagicMock(side_effect=lambda purpose, *_: sentinel_parser_model if purpose == "parser" else original_get_llm(purpose))  # type: ignore[method-assign]

    captured = {}

    async def _fake_structure_output(*args, **kwargs):  # noqa: D401
        captured["model"] = kwargs.get("model")
        # Return a minimally valid object for each call site
        out_type = kwargs.get("output_type") if "output_type" in kwargs else (args[1] if len(args) > 1 else None)
        if out_type.__name__ == "BinaryPrediction":

            class _Bin:  # noqa: N801
                prediction_in_decimal = 0.5

            return _Bin()
        if out_type.__name__ == "PredictedOptionList":
            return MagicMock()
        if out_type.__name__ == "list":  # list[Percentile]
            return [
                Percentile(value=v, percentile=p)
                for v, p in zip([1, 2, 3, 4, 5, 6, 7, 8], [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
            ]
        return MagicMock()

    # Minimal binary question
    bq = MagicMock(spec=BinaryQuestion)
    bq.page_url = "url"
    bq.question_text = "?"
    bq.background_info = ""
    bq.resolution_criteria = ""
    bq.fine_print = ""
    bq.id_of_question = 10
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="r")

    with patch("main.structure_output", _fake_structure_output):
        await bot._run_forecast_on_binary(bq, "", llm)
        assert captured["model"] is sentinel_parser_model

    # Minimal multiple-choice question
    mcq = MagicMock(spec=MultipleChoiceQuestion)
    mcq.page_url = "url"
    mcq.question_text = "?"
    mcq.options = ["A", "B"]
    mcq.background_info = ""
    mcq.resolution_criteria = ""
    mcq.fine_print = ""
    mcq.id_of_question = 20
    with patch("main.structure_output", _fake_structure_output):
        await bot._run_forecast_on_multiple_choice(mcq, "", llm)
        assert captured["model"] is sentinel_parser_model

    # Minimal numeric question
    nq = SimpleNamespace(
        question_text="num?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=False,
        open_lower_bound=False,
        lower_bound=0,
        upper_bound=100,
        page_url="url",
        zero_point=None,
        id_of_question=11,
        cdf_size=201,
    )
    with patch("main.structure_output", _fake_structure_output):
        await bot._run_forecast_on_numeric(nq, "", llm)  # type: ignore[arg-type]
        assert captured["model"] is sentinel_parser_model


@pytest.mark.asyncio
async def test_mc_additional_instructions_include_options():
    bot = TemplateForecaster(llms={"default": "mock", "parser": "mock", "researcher": "mock", "summarizer": "mock"})
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.page_url = "url"
    q.question_text = "who?"
    q.options = ["Alpha", "Beta"]
    q.background_info = ""
    q.resolution_criteria = ""
    q.fine_print = ""
    q.id_of_question = 21

    llm = MagicMock(spec=GeneralLlm)
    llm.model = "m"
    llm.invoke = AsyncMock(return_value="r")

    seen = {}

    async def _fake_structure_output(*args, **kwargs):  # noqa: D401
        seen["additional_instructions"] = kwargs.get("additional_instructions", "")
        return MagicMock()

    with patch("main.structure_output", _fake_structure_output):
        await bot._run_forecast_on_multiple_choice(q, "", llm)

    ai = seen["additional_instructions"] or ""
    assert "Alpha" in ai and "Beta" in ai
