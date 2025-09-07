from abc import ABC
from typing import List

import pytest
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from pydantic import Field

from metaculus_bot.numeric_utils import (
    aggregate_binary_mean,
    aggregate_numeric,
    bound_messages,
)
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from metaculus_bot.utils.logging_utils import compact_log_report_summary

# ---------- Prompt builders -------------------------------------------------


def test_binary_prompt_contains_inputs():
    question = BinaryQuestion(
        id_of_question=2,
        id_of_post=2,
        page_url="example",
        question_text="Will it rain tomorrow?",
        background_info="Some background",
        resolution_criteria="Criteria",
        fine_print="Fine print",
        published_time=None,
        close_time=None,
    )
    prompt = binary_prompt(question, "research snippet")
    assert "Will it rain tomorrow?" in prompt
    assert "research snippet" in prompt


def test_multiple_choice_prompt_contains_options():
    question = MultipleChoiceQuestion(
        id_of_question=3,
        id_of_post=3,
        page_url="example",
        question_text="Who will win?",
        options=["A", "B"],
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
    )
    prompt = multiple_choice_prompt(question, "mc research")
    assert "Who will win?" in prompt
    assert "Option_A" in prompt  # output format marker


def test_numeric_prompt_bounds_and_research():
    question = NumericQuestion(
        id_of_question=4,
        id_of_post=4,
        page_url="example",
        question_text="How many widgets?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="widgets",
        zero_point=None,
    )
    prompt = numeric_prompt(question, "num research", "lower", "upper")
    assert "widgets" in prompt and "num research" in prompt


def test_numeric_prompt_includes_p5_and_p95():
    """Test that numeric prompt includes P5 and P95 in the output format."""
    question = NumericQuestion(
        id_of_question=5,
        id_of_post=5,
        page_url="example",
        question_text="Test numeric question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    prompt = numeric_prompt(question, "research", "", "")
    assert "Percentile 5:" in prompt
    assert "Percentile 95:" in prompt
    # Ensure all 8 percentiles are present in order
    lines = prompt.split("\n")
    example_section = False
    percentile_lines = []
    for line in lines:
        if "__Example:__" in line:
            example_section = True
            continue
        if example_section and "Percentile" in line and ":" in line:
            percentile_lines.append(line.strip())

    expected = [
        "Percentile 5: 10.1",
        "Percentile 10: 12.3",
        "Percentile 20: 23.4",
        "Percentile 40: 34.5",
        "Percentile 60: 56.7",
        "Percentile 80: 67.8",
        "Percentile 90: 78.9",
        "Percentile 95: 89.0",
    ]
    assert percentile_lines == expected


# ---------- Numeric utils ---------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_numeric_mean_and_median():
    question = NumericQuestion(
        id_of_question=1,
        id_of_post=1,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    # Note: numeric distribution will add 0% and 100% percentiles if they are not present,
    # so the values being tested are not at the boundaries.
    percentiles = [Percentile(value=v, percentile=p) for v, p in zip([10, 50, 90], [0.1, 0.5, 0.9])]
    dist_a = NumericDistribution(declared_percentiles=percentiles, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=percentiles, **question.model_dump())

    mean_result = await aggregate_numeric([dist_a, dist_b], question, "mean")
    median_result = await aggregate_numeric([dist_a, dist_b], question, "median")

    # Both mean and median aggregations now return a full 201-point distribution.
    # Since we are aggregating two identical distributions, the result should be
    # the same as the original interpolated CDF. We can check the 50th percentile.
    mean_p50 = next(p for p in mean_result.declared_percentiles if p.value == 50)
    median_p50 = next(p for p in median_result.declared_percentiles if p.value == 50)

    assert mean_p50.percentile == pytest.approx(0.5)
    assert median_p50.percentile == pytest.approx(0.5)


def test_aggregate_binary_mean():
    assert aggregate_binary_mean([0.4, 0.6]) == 0.5


def test_bound_messages():
    q = NumericQuestion(
        id_of_question=5,
        id_of_post=5,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=10,
        open_lower_bound=True,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    upper, lower = bound_messages(q)
    assert "higher" in upper
    # With open lower bound, we now include a practical/display lower bound hint
    assert "0.0" in lower or lower == ""


def test_bound_messages_uses_nominal_bounds():
    q = NumericQuestion(
        id_of_question=6,
        id_of_post=6,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        nominal_lower_bound=5,
        nominal_upper_bound=42,
    )

    upper, lower = bound_messages(q)
    assert "42" in upper and "5" in lower


def test_bound_messages_discrete_fallback():
    """Test that bound_messages derives nominal bounds for discrete questions when missing."""
    # Create a discrete question (cdf_size != 201) without nominal bounds
    q = NumericQuestion(
        id_of_question=7,
        id_of_post=7,
        page_url="example",
        question_text="Discrete question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,  # API bounds are typically off by 0.5 for discrete
        upper_bound=9.5,  # Representing 0-9 discrete values
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=11,  # 10 discrete values + 1 = 11
    )

    upper, lower = bound_messages(q)
    # Should derive nominal bounds: step = (9.5 - (-0.5)) / (11 - 1) = 1.0
    # nominal_lower = -0.5 + 1.0/2 = 0.0, nominal_upper = 9.5 - 1.0/2 = 9.0
    assert "9.0" in upper and "0.0" in lower


# ---------- Compact logger --------------------------------------------------


class DummyQuestion(MetaculusQuestion, ABC):
    pass


class DummyReport(ForecastReport):
    # This is a dummy report for testing the compact logger.
    # It needs to be a valid ForecastReport, so we provide minimal implementations
    # for abstract methods and required fields.
    question: MetaculusQuestion = DummyQuestion(
        id_of_question=99,
        id_of_post=99,
        page_url="dummy_url",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
    )
    explanation: str = "# Dummy"
    prediction: List[str] = Field(default_factory=list)

    @classmethod
    def make_readable_prediction(cls, prediction: "list[str]") -> str:
        return "N/A"

    @classmethod
    async def aggregate_predictions(cls: type, predictions: list, question: MetaculusQuestion) -> "DummyReport":
        raise NotImplementedError()

    async def publish_report_to_metaculus(self) -> None:
        raise NotImplementedError()


def test_compact_logger_no_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test that the compact logger runs without exceptions on a dummy report."""
    compact_log_report_summary([DummyReport()])  # should not raise
