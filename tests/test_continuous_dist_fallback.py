# type: ignore
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools.data_models.numeric_report import Percentile as FTPercentile
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
        cdf_size=201,
    )


@pytest.fixture
def dummy_forecaster():
    # Provide the bare minimum llm config so TemplateForecaster initialises.
    dummy_llm = MagicMock()
    dummy_llm.model = "dummy"
    return TemplateForecaster(
        llms={
            "default": dummy_llm,
            "parser": "mock",
            "researcher": "mock",
            "summarizer": "mock",
        },
        publish_reports_to_metaculus=False,
    )


@pytest.mark.asyncio
async def test_numeric_parsing_success_without_fallback(dummy_forecaster):
    # We expect to use structured-output only; provide a valid structured parse.
    rationale = "irrelevant reasoning; parser output is mocked"
    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)

    fake_percentiles = [
        FTPercentile(value=v, percentile=p)
        for v, p in zip(
            [100, 110, 120, 130, 140, 150, 160, 170],
            [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95],
        )
    ]

    with patch("main.structure_output", return_value=fake_percentiles):
        result = await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]

    values = [p.value for p in result.prediction_value.declared_percentiles]  # type: ignore
    assert values == [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0]


@pytest.mark.asyncio
async def test_fallback_reraises_when_insufficient_numbers(dummy_forecaster):
    rationale = "Percentile 10: 5\nPercentile 20: 6\n"

    q = make_dummy_numeric_question()
    llm = DummyLLM(rationale)
    with patch(
        "main.structure_output",
        side_effect=ValidationError.from_exception_data("NumericDistribution", []),
    ):
        with pytest.raises(ValidationError):
            await dummy_forecaster._run_forecast_on_numeric(q, "", llm)  # type: ignore[arg-type]
