from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (BinaryQuestion, GeneralLlm, MetaculusQuestion,
                               ReasonedPrediction)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import \
    ResearchWithPredictions

from main import TemplateForecaster


@pytest.fixture
def mock_general_llm():
    mock_llm = MagicMock(spec=GeneralLlm)
    mock_llm.model = "mock_model"
    mock_llm.invoke = AsyncMock(return_value="mock reasoning")
    return mock_llm


@pytest.fixture
def mock_metaculus_question():
    question = MagicMock(spec=MetaculusQuestion)
    question.page_url = "http://example.com/question"
    question.question_text = "Test Question"
    question.background_info = "Background info"
    question.resolution_criteria = "Resolution criteria"
    question.fine_print = "Fine print"
    question.unit_of_measure = "units"
    question.id_of_question = 123  # Add a mock ID for testing
    return question


@pytest.fixture
def mock_binary_question():
    question = MagicMock(spec=BinaryQuestion)
    question.page_url = "http://example.com/binary_question"
    question.question_text = "Binary Test Question"
    question.background_info = "Binary background info"
    question.resolution_criteria = "Binary resolution criteria"
    question.fine_print = "Binary fine print"
    question.unit_of_measure = "binary units"
    question.id_of_question = 456
    return question


@pytest.mark.asyncio
async def test_template_forecaster_init_with_forecasters(mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    assert bot._forecaster_llms == llms_config["forecasters"]
    assert bot.predictions_per_research_report == 2
    assert bot.get_llm("default") == mock_general_llm  # Should be the first forecaster


@pytest.mark.asyncio
async def test_template_forecaster_init_without_forecasters():
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=3)

    assert not bot._forecaster_llms
    assert bot.predictions_per_research_report == 3
    assert bot.get_llm("default").model == "test_default"


@pytest.mark.asyncio
async def test_template_forecaster_init_no_llms_provided():
    with pytest.raises(ValueError, match="Either 'forecasters' or a 'default' LLM must be provided."):
        TemplateForecaster(llms=None)


@pytest.mark.asyncio
async def test_template_forecaster_init_forecasters_not_list():
    llms_config = {
        "forecasters": "not_a_list",
        "default": GeneralLlm(model="test_default"),
    }
    with patch("main.logger.warning") as mock_warning:
        bot = TemplateForecaster(llms=llms_config)
        mock_warning.assert_called_once_with("'forecasters' key in llms must be a list of GeneralLlm objects.")
        assert not bot._forecaster_llms
        assert bot.predictions_per_research_report == 1  # Default value from parent class


@pytest.mark.asyncio
async def test_research_and_make_predictions_with_forecasters(mock_binary_question, mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock internal methods
    bot._get_notepad = AsyncMock(return_value=MagicMock(num_research_reports_attempted=0, num_predictions_attempted=0))
    bot.run_research = AsyncMock(return_value="mock research")
    bot.summarize_research = AsyncMock(return_value="mock summary")
    bot._make_prediction = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test"))
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="test"),
                ReasonedPrediction(prediction_value=0.6, reasoning="test2"),
            ],
            [],
            None,
        )
    )

    result = await bot._research_and_make_predictions(mock_binary_question)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.run_research.assert_called_once_with(mock_binary_question)
    bot.summarize_research.assert_called_once_with(mock_binary_question, "mock research")
    assert bot._make_prediction.call_count == 2  # Called once for each forecaster
    bot._make_prediction.assert_any_call(mock_binary_question, "mock research", mock_general_llm)
    assert isinstance(result, ResearchWithPredictions)
    assert (
        len(result.predictions) == 2
    )  # The mocked _gather_results_and_exceptions returns two ReasonedPrediction objects


@pytest.mark.asyncio
async def test_research_and_make_predictions_without_forecasters(mock_binary_question):
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=1)

    # Mock the super method call
    with patch(
        "forecasting_tools.forecast_bots.forecast_bot.ForecastBot._research_and_make_predictions",
        new_callable=AsyncMock,
    ) as mock_super_method:
        mock_super_method.return_value = ResearchWithPredictions(
            research_report="super research",
            summary_report="super summary",
            predictions=[ReasonedPrediction(prediction_value=0.6, reasoning="super test")],
        )
        result = await bot._research_and_make_predictions(mock_binary_question)
        mock_super_method.assert_called_once_with(mock_binary_question)
        assert isinstance(result, ResearchWithPredictions)
        assert result.research_report == "super research"


@pytest.mark.asyncio
async def test_make_prediction_with_provided_llm(mock_binary_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(num_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.7, reasoning="binary forecast")
    )

    result = await bot._make_prediction(mock_binary_question, "some research", mock_general_llm)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_general_llm)
    assert result.prediction_value == 0.7
    assert "Model: mock_model" in result.reasoning
    assert "binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_make_prediction_without_provided_llm(mock_binary_question):
    mock_default_llm = MagicMock(spec=GeneralLlm)
    mock_default_llm.model = "default_mock_model"
    mock_default_llm.invoke = AsyncMock(return_value="default reasoning")

    llms_config = {"default": mock_default_llm, "summarizer": "mock_summarizer_model"}
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(num_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.8, reasoning="default binary forecast")
    )
    bot.get_llm = MagicMock(return_value=mock_default_llm)

    result = await bot._make_prediction(mock_binary_question, "some research")

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.get_llm.assert_called_once_with("default", "llm")
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_default_llm)
    assert result.prediction_value == 0.8
    assert "Model: default_mock_model" in result.reasoning
    assert "default binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_binary_uses_provided_llm(mock_binary_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock structured_output to avoid external parsing LLM calls
    with patch("main.structure_output", return_value=type("_Bin", (), {"prediction_in_decimal": 0.65})()) as mock_struct:
        result = await bot._run_forecast_on_binary(mock_binary_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_struct.assert_called_once()
        assert result.prediction_value == 0.65
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_multiple_choice_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    mock_metaculus_question.options = ["A", "B"]

    # Mock structured_output for multiple-choice
    with patch("main.structure_output", return_value=MagicMock()) as mock_struct:
        result = await bot._run_forecast_on_multiple_choice(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_struct.assert_called_once()
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_numeric_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock _create_upper_and_lower_bound_messages and structured_output to return a valid percentile list
    from forecasting_tools.data_models.numeric_report import Percentile as FTPercentile
    fake_percentiles = [
        FTPercentile(value=v, percentile=p)
        for v, p in zip([1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    ]
    # Provide minimal numeric bounds attributes expected by NumericDistribution.from_question
    mock_metaculus_question.open_upper_bound = False
    mock_metaculus_question.open_lower_bound = False
    mock_metaculus_question.upper_bound = 100
    mock_metaculus_question.lower_bound = 0
    mock_metaculus_question.zero_point = None
    mock_metaculus_question.cdf_size = 201

    with patch.object(bot, "_create_upper_and_lower_bound_messages", return_value=("", "")) as mock_bounds, \
         patch("main.structure_output", return_value=fake_percentiles) as mock_struct:
        result = await bot._run_forecast_on_numeric(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_bounds.assert_called_once()
        mock_struct.assert_called_once()
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning
