import asyncio
from typing import Any, cast
# type: ignore
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import PredictionExtractor
from forecasting_tools.data_models.numeric_report import NumericDistribution
from pydantic import ValidationError

# from forecasting_tools.data_models.questions import NumericQuestion
from main import TemplateForecaster


class DummyLLM:  # minimal async LLM for tests
    def __init__(self, reasoning: str):
        self._reasoning = reasoning
        self.model = "dummy-test-model"

    async def invoke(self, prompt: str):  # noqa: D401
        return self._reasoning


# Lightweight dummy object with the attrs _run_forecast_on_numeric needs.
from types import SimpleNamespace


def make_dummy_numeric_question():
    return SimpleNamespace(
        question_text="dummy numeric",
        background_info="",
        resolution_criteria="",
        fine_print="",
        unit_of_measure=None,
        open_upper_bound=True,
        open_lower_bound=True,
        lower_bound=0,
        upper_bound=9999,
        page_url="https://example.com/q",
        zero_point=0,
        id_of_question=123,  # Added for testing purposes
    )


@pytest.fixture
def dummy_forecaster():
    # Provide the bare minimum llm config so TemplateForecaster initialises.
    dummy_llm = MagicMock()
    dummy_llm.model = "dummy"
    return TemplateForecaster(
        llms={"default": dummy_llm},
        publish_reports_to_metaculus=False,
    )


@pytest.mark.asyncio
async def test_fallback_repairs_extra_numbers(dummy_forecaster):
    rationale = (
        "Some intro line with numbers 1 2 3\n"
        "Percentile 10: 110\n"
        "Percentile 20: 120\n"
        "Percentile 40: 130\n"
        "Percentile 60: 140\n"
        "Percentile 80: 150\n"
        "Percentile 90: 160\n"
        "Another distracting 99% line 200\n"
    )

    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)
    result = await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]

    values = [p.value for p in result.prediction_value.declared_percentiles]  # type: ignore
    assert values == [110, 120, 130, 140, 150, 160]


@pytest.mark.asyncio
async def test_fallback_reraises_when_insufficient_numbers(dummy_forecaster):
    rationale = "Percentile 10: 5\nPercentile 20: 6\n"

    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)
    with pytest.raises(ValidationError):
        await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]
