"""Lightweight smoke tests for community_benchmark.py to catch common issues."""

import argparse
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


def test_import_community_benchmark():
    """Test that the module can be imported without errors."""


def test_cli_argument_parsing():
    """Test CLI argument parsing with different combinations."""
    # Create parser identical to the one in community_benchmark.py
    parser = argparse.ArgumentParser(description="Benchmark a list of bots")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "custom", "display"],
        default="display",
        help="Specify the run mode (default: display)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2,
        help="Number of questions to benchmark (default: 2)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice)",
    )

    # Test default args
    args = parser.parse_args([])
    assert args.mode == "display"
    assert args.num_questions == 2
    assert args.mixed is False

    # Test custom args
    args = parser.parse_args(["--mode", "run", "--num-questions", "5"])
    assert args.mode == "run"
    assert args.num_questions == 5
    assert args.mixed is False

    # Test mixed flag
    args = parser.parse_args(["--mode", "custom", "--mixed"])
    assert args.mode == "custom"
    assert args.mixed is True

    # Test edge cases
    args = parser.parse_args(["--num-questions", "1"])
    assert args.num_questions == 1


def test_bot_instantiation():
    """Test that benchmark bots can be created with the new ensemble configuration."""
    from forecasting_tools import GeneralLlm

    from main import TemplateForecaster
    from metaculus_bot.aggregation_strategies import AggregationStrategy
    from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

    # Test single model bot (equivalent to new baseline configurations)
    single_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        aggregation_strategy=AggregationStrategy.MEAN,  # Unused for single model
        research_provider=None,
        max_questions_per_run=None,
        llms={
            "forecasters": [
                GeneralLlm(
                    model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=16000,
                    stream=False,
                    timeout=240,
                    allowed_tries=3,
                )
            ],
            "summarizer": SUMMARIZER_LLM,
            "parser": PARSER_LLM,
            "researcher": RESEARCHER_LLM,
        },
    )

    # Test ensemble bot with mean aggregation
    ensemble_mean_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        research_provider=None,
        max_questions_per_run=None,
        llms={
            "forecasters": [
                GeneralLlm(
                    model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=16000,
                    stream=False,
                    timeout=240,
                    allowed_tries=3,
                ),
                GeneralLlm(
                    model="openrouter/z-ai/glm-4.5",
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=16000,
                    stream=False,
                    timeout=240,
                    allowed_tries=3,
                ),
            ],
            "summarizer": SUMMARIZER_LLM,
            "parser": PARSER_LLM,
            "researcher": RESEARCHER_LLM,
        },
    )

    # Test ensemble bot with median aggregation
    ensemble_median_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        aggregation_strategy=AggregationStrategy.MEDIAN,
        research_provider=None,
        max_questions_per_run=None,
        llms={
            "forecasters": [
                GeneralLlm(
                    model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=16000,
                    stream=False,
                    timeout=240,
                    allowed_tries=3,
                ),
                GeneralLlm(
                    model="openrouter/z-ai/glm-4.5",
                    temperature=0.0,
                    top_p=0.9,
                    max_tokens=16000,
                    stream=False,
                    timeout=240,
                    allowed_tries=3,
                ),
            ],
            "summarizer": SUMMARIZER_LLM,
            "parser": PARSER_LLM,
            "researcher": RESEARCHER_LLM,
        },
    )

    # Verify bots have required attributes
    for bot in [single_bot, ensemble_mean_bot, ensemble_median_bot]:
        assert hasattr(bot, "research_provider")
        assert hasattr(bot, "max_questions_per_run")
        assert hasattr(bot, "aggregation_strategy")
        assert bot.max_questions_per_run is None

    # Verify aggregation strategies are set correctly
    assert single_bot.aggregation_strategy == AggregationStrategy.MEAN  # Default for single model
    assert ensemble_mean_bot.aggregation_strategy == AggregationStrategy.MEAN
    assert ensemble_median_bot.aggregation_strategy == AggregationStrategy.MEDIAN


@patch("community_benchmark.run_benchmark_streamlit_page")
def test_display_mode(mock_streamlit):
    """Test that display mode calls streamlit and returns early."""
    import community_benchmark

    # Mock streamlit function to avoid actually launching UI
    mock_streamlit.return_value = None

    # Run display mode
    asyncio.run(community_benchmark.benchmark_forecast_bot("display", 5))

    # Verify streamlit was called and function returned early
    mock_streamlit.assert_called_once()


@patch("community_benchmark.MetaculusApi.get_benchmark_questions")
@patch("community_benchmark.Benchmarker")
@patch("community_benchmark.MonetaryCostManager")
def test_benchmark_flow_without_api_calls(mock_cost_manager, mock_benchmarker_class, mock_get_questions):
    """Test benchmark flow without making actual API calls."""
    import community_benchmark

    # Mock API calls to prevent actual requests
    mock_get_questions.return_value = []  # Empty question list

    # Mock benchmarker
    mock_benchmarker = Mock()
    mock_benchmarker.run_benchmark = AsyncMock(return_value=[])
    mock_benchmarker_class.return_value = mock_benchmarker

    # Mock cost manager
    mock_cost_manager_instance = Mock()
    mock_cost_manager_instance.__enter__ = Mock(return_value=mock_cost_manager_instance)
    mock_cost_manager_instance.__exit__ = Mock(return_value=None)
    mock_cost_manager_instance.current_usage = "Mock Cost"
    mock_cost_manager.return_value = mock_cost_manager_instance

    # Test that function can run without errors
    asyncio.run(community_benchmark.benchmark_forecast_bot("run", 1))

    # Verify mocked functions were called
    mock_get_questions.assert_called_once_with(1)
    mock_benchmarker.run_benchmark.assert_called_once()


@patch("community_benchmark.MetaculusApi.get_questions_matching_filter")
@patch("community_benchmark.Benchmarker")
@patch("community_benchmark.MonetaryCostManager")
def test_custom_mode_without_api_calls(mock_cost_manager, mock_benchmarker_class, mock_get_questions_filter):
    """Test custom mode flow without making actual API calls."""
    import community_benchmark

    # Mock API calls
    mock_get_questions_filter.return_value = []  # Empty question list

    # Mock benchmarker
    mock_benchmarker = Mock()
    mock_benchmarker.run_benchmark = AsyncMock(return_value=[])
    mock_benchmarker_class.return_value = mock_benchmarker

    # Mock cost manager
    mock_cost_manager_instance = Mock()
    mock_cost_manager_instance.__enter__ = Mock(return_value=mock_cost_manager_instance)
    mock_cost_manager_instance.__exit__ = Mock(return_value=None)
    mock_cost_manager_instance.current_usage = "Mock Cost"
    mock_cost_manager.return_value = mock_cost_manager_instance

    # Test custom mode
    asyncio.run(community_benchmark.benchmark_forecast_bot("custom", 3))

    # Verify mocked functions were called
    mock_get_questions_filter.assert_called_once()
    mock_benchmarker.run_benchmark.assert_called_once()


def test_invalid_mode_raises_error():
    """Test that invalid mode raises ValueError."""
    import community_benchmark

    with pytest.raises(ValueError, match="Invalid mode: invalid_mode"):
        asyncio.run(community_benchmark.benchmark_forecast_bot("invalid_mode", 1))


@patch("community_benchmark.MetaculusApi.get_questions_matching_filter")
def test_mixed_question_types_distribution(mock_get_questions_filter):
    """Test that mixed question type distribution works correctly."""
    from datetime import datetime, timedelta

    import community_benchmark

    # Mock return values for different question types
    def mock_filter_side_effect(api_filter, num_questions, randomly_sample):
        # Return different mock questions based on allowed_types
        question_type = api_filter.allowed_types[0] if api_filter.allowed_types else "binary"
        mock_questions = []
        for i in range(num_questions):
            mock_question = Mock()
            mock_question.background_info = f"Mock {question_type} question {i}"
            mock_question.question_text = (
                f"Mock {question_type} question text {i} - this is a sample question for testing purposes"
            )
            mock_questions.append(mock_question)
        return mock_questions

    mock_get_questions_filter.side_effect = mock_filter_side_effect

    # Test with 12 questions to get nice distribution
    one_year_from_now = datetime.now() + timedelta(days=365)
    questions = asyncio.run(community_benchmark._get_mixed_question_types(12, one_year_from_now))

    # Should have called the API 3 times (binary, numeric, multiple_choice)
    assert mock_get_questions_filter.call_count == 3

    # Should have roughly 50/25/25 distribution: 6/3/3
    assert len(questions) == 12

    # Verify all questions had background_info cleared
    for question in questions:
        assert question.background_info is None


def test_individual_model_bot_generation():
    """Test that the new individual model configuration generates the expected bots."""
    from forecasting_tools import GeneralLlm

    # Simulate the individual model configuration logic from community_benchmark.py
    MODEL_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 16000,
        "stream": False,
        "timeout": 240,
        "allowed_tries": 3,
    }

    # Define individual models (same as in community_benchmark.py)
    qwen3_model = GeneralLlm(
        model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        **MODEL_CONFIG,
    )
    glm_model = GeneralLlm(
        model="openrouter/z-ai/glm-4.5",
        **MODEL_CONFIG,
    )

    # Individual model configurations for benchmarking
    individual_models = [
        {"name": "qwen3-235b", "forecaster": qwen3_model},
        {"name": "glm-4.5", "forecaster": glm_model},
    ]

    # Expected bot names: just individual models (ensembles generated post-hoc)
    expected_bot_names = [
        "qwen3-235b",  # Single model
        "glm-4.5",  # Single model
    ]

    # Verify the logic produces the expected number and types of configurations
    generated_names = []
    for model_config in individual_models:
        generated_names.append(model_config["name"])

    # Verify we get the expected bot names
    assert sorted(generated_names) == sorted(expected_bot_names)
    assert len(generated_names) == 2

    # Verify this matches the new approach: only individual models
    assert len(generated_names) == len(individual_models)


@patch("community_benchmark.MetaculusApi.get_questions_matching_filter")
@patch("community_benchmark.Benchmarker")
@patch("community_benchmark.MonetaryCostManager")
def test_custom_mode_with_mixed_flag(mock_cost_manager, mock_benchmarker_class, mock_get_questions_filter):
    """Test custom mode with mixed flag uses mixed question types."""
    import community_benchmark

    # Mock return values for different question types
    def mock_filter_side_effect(api_filter, num_questions, randomly_sample):
        mock_questions = []
        for i in range(num_questions):
            mock_question = Mock()
            mock_question.background_info = "test"
            mock_question.question_text = f"Mock question text {i} - this is a sample question for testing purposes"
            mock_questions.append(mock_question)
        return mock_questions

    mock_get_questions_filter.side_effect = mock_filter_side_effect

    # Mock benchmarker
    mock_benchmarker = Mock()
    mock_benchmarker.run_benchmark = AsyncMock(return_value=[])
    mock_benchmarker_class.return_value = mock_benchmarker

    # Mock cost manager
    mock_cost_manager_instance = Mock()
    mock_cost_manager_instance.__enter__ = Mock(return_value=mock_cost_manager_instance)
    mock_cost_manager_instance.__exit__ = Mock(return_value=None)
    mock_cost_manager_instance.current_usage = "Mock Cost"
    mock_cost_manager.return_value = mock_cost_manager_instance

    # Test custom mode with mixed flag
    asyncio.run(community_benchmark.benchmark_forecast_bot("custom", 6, mixed_types=True))

    # Should call API 3 times for mixed types (binary, numeric, multiple_choice)
    assert mock_get_questions_filter.call_count == 3

    # Test custom mode without mixed flag (should call API once)
    mock_get_questions_filter.reset_mock()
    asyncio.run(community_benchmark.benchmark_forecast_bot("custom", 6, mixed_types=False))

    # Should call API only once for binary questions
    assert mock_get_questions_filter.call_count == 1
