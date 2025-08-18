"""
Benchmark treating the community prediction as approximate ground truth.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import random
import sys
import weakref
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import typeguard
from dotenv import load_dotenv
from forecasting_tools import (
    ApiFilter,
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MonetaryCostManager,
    run_benchmark_streamlit_page,
)

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.constants import BENCHMARK_BATCH_SIZE
from metaculus_bot.llm_configs import FORECASTER_LLMS, PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM
from metaculus_bot.scoring_patches import apply_scoring_patches, log_score_scale_validation

logger = logging.getLogger(__name__)

load_dotenv()
load_dotenv(".env.local", override=True)


# Quick mitigation for occasional "Unclosed client session" warnings from aiohttp when
# using EXA search under high concurrency. Tracks sessions and closes them at process exit.
def _enable_aiohttp_session_autoclose() -> None:
    open_sessions: "weakref.WeakSet[aiohttp.ClientSession]" = weakref.WeakSet()
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self: aiohttp.ClientSession, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        open_sessions.add(self)

    aiohttp.ClientSession.__init__ = tracking_init  # type: ignore[assignment]

    def _close_open_sessions() -> None:
        to_close = [s for s in list(open_sessions) if not s.closed]
        if not to_close:
            return
        logger.debug(f"Closing {len(to_close)} lingering aiohttp sessions at exit")

        async def _close_all() -> None:
            for s in to_close:
                try:
                    await s.close()
                except Exception as e:  # pragma: no cover - best-effort cleanup
                    logger.debug(f"Error closing aiohttp session at exit: {e}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_close_all())
        else:
            try:
                asyncio.run(_close_all())
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_close_all())
                finally:
                    new_loop.close()

    atexit.register(_close_open_sessions)


_enable_aiohttp_session_autoclose()


async def _get_mixed_question_types(total_questions: int, one_year_from_now: datetime) -> list:
    """Get mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice)."""

    # Calculate counts for each type (50/25/25 distribution)
    binary_count = int(total_questions * 0.5)
    numeric_count = int(total_questions * 0.25)
    mc_count = total_questions - binary_count - numeric_count  # Remainder goes to MC

    logger.info(f"Fetching mixed questions: {binary_count} binary, {numeric_count} numeric, {mc_count} multiple-choice")

    # Base filter settings for all question types
    base_filter_kwargs = {
        "allowed_statuses": ["open"],
        "num_forecasters_gte": 40,
        "scheduled_resolve_time_lt": one_year_from_now,
        "includes_bots_in_aggregates": False,
        "community_prediction_exists": True,
    }

    all_questions = []

    # Fetch each question type separately
    for i, (question_type, count) in enumerate(
        [("binary", binary_count), ("numeric", numeric_count), ("multiple_choice", mc_count)], 1
    ):
        if count > 0:
            try:
                logger.info(f"[{i}/3] Fetching {count} {question_type} questions...")
                sys.stdout.flush()  # Force immediate output

                # Customize filter for each question type
                filter_kwargs = base_filter_kwargs.copy()

                # Community prediction filter only works for binary questions
                if question_type != "binary":
                    filter_kwargs.pop("community_prediction_exists", None)
                    logger.info(f"⚠️  Removed community_prediction_exists filter for {question_type} questions")
                    sys.stdout.flush()

                logger.info(f"🔍 Starting API call for {question_type} questions...")
                sys.stdout.flush()

                api_filter = ApiFilter(
                    allowed_types=[question_type],
                    **filter_kwargs,
                )
                questions = await MetaculusApi.get_questions_matching_filter(
                    api_filter,
                    num_questions=count,
                    randomly_sample=True,
                )
                logger.info(f"✅ Successfully fetched {len(questions)} {question_type} questions")
                if questions:
                    logger.info(f"📋 Sample {question_type} question: {questions[0].question_text[:100]}...")
                all_questions.extend(questions)
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"❌ Failed to fetch {question_type} questions: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                sys.stdout.flush()

    # Shuffle to avoid clustering by type
    random.shuffle(all_questions)

    # Clear background_info for all questions (to test ability to find new information)
    for question in all_questions:
        question.background_info = None

    # Log final distribution
    type_counts = {}
    for q in all_questions:
        q_type = type(q).__name__
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

    logger.info(f"Final mixed question distribution: {type_counts}")
    return all_questions


async def benchmark_forecast_bot(mode: str, number_of_questions: int = 2, mixed_types: bool = False) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction.
    Ideally 100+ questions for meaningful error bars, but can use e.g. just a few for smoke testing or 30 for a quick run.
    """
    # TODO: make sure this is ok w/ the max predictions at once cost safety controls we have in place

    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        questions = MetaculusApi.get_benchmark_questions(number_of_questions)
    elif mode == "custom":
        # Below is an example of getting custom questions
        one_year_from_now = datetime.now() + timedelta(days=365)

        if mixed_types:
            # Get mixed question types with 50/25/25 distribution
            questions = await _get_mixed_question_types(number_of_questions, one_year_from_now)
        else:
            # Original binary-only approach
            api_filter = ApiFilter(
                allowed_statuses=["open"],
                allowed_types=["binary"],
                num_forecasters_gte=40,
                scheduled_resolve_time_lt=one_year_from_now,
                includes_bots_in_aggregates=False,
                community_prediction_exists=True,
            )
            questions = await MetaculusApi.get_questions_matching_filter(
                api_filter,
                num_questions=number_of_questions,
                randomly_sample=True,
            )

        for question in questions:
            question.background_info = None  # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Shared configuration for all benchmark bots
    BENCHMARK_BOT_CONFIG = {
        "research_reports_per_question": 1,
        "predictions_per_research_report": 1,  # Ignored when forecasters present
        "use_research_summary_to_forecast": False,
        "publish_reports_to_metaculus": False,  # Don't publish during benchmarking
        "folder_to_save_reports_to": None,
        "skip_previously_forecasted_questions": False,
        "research_provider": None,  # Use default provider selection
        "max_questions_per_run": None,  # No limit for benchmarking
        "is_benchmarking": True,  # Exclude prediction markets to avoid data leakage
        "allow_research_fallback": False,  # Ensure AskNews runs; do not fallback in benchmark
    }
    MODEL_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 16000,  # Prevent truncation issues with reasoning models
        "stream": False,
        "timeout": 240,
        "allowed_tries": 3,
    }
    DEFAULT_HELPER_LLMS = {
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }

    # Apply scoring patches for mixed question types
    apply_scoring_patches()

    with MonetaryCostManager() as cost_manager:
        # Keep benchmark and bot research concurrency aligned
        batch_size = BENCHMARK_BATCH_SIZE

        # Shared research cache for all bots to avoid duplicate API calls
        research_cache: dict[int, str] = {}

        # Define individual model configurations -- for sanity checking, can use these free models

        free_r1_0528_model = GeneralLlm(
            model="openrouter/deepseek/deepseek-r1-0528:free",
            **MODEL_CONFIG,
        )
        free_qwen3_coder_model = GeneralLlm(
            model="openrouter/qwen/qwen3-coder:free",
            **MODEL_CONFIG,
        )
        free_glm_4p5_air_model = GeneralLlm(
            model="openrouter/z-ai/glm-4.5-air:free",
            **MODEL_CONFIG,
        )

        # optional cheap models, commented out for now for dev:
        # qwen3_model = GeneralLlm(
        #     model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        #     **MODEL_CONFIG,
        # )
        # glm_model = GeneralLlm(
        #     model="openrouter/z-ai/glm-4.5",
        #     **MODEL_CONFIG,
        # )

        # Keep these commented for cost control during development:
        # deepseek_model = GeneralLlm(
        #     model="openrouter/deepseek/deepseek-r1-0528",
        #     **MODEL_CONFIG,
        # )
        # claude_model = GeneralLlm(
        #     model="openrouter/anthropic/claude-sonnet-4",
        #     reasoning={"max_tokens": 4000},
        #     api_key=get_openrouter_api_key("openrouter/anthropic/claude-sonnet-4"),
        #     **MODEL_CONFIG,
        # )
        # gpt5_model = GeneralLlm(
        #     model="openrouter/openai/gpt-5",
        #     reasoning_effort="high",
        #     api_key=get_openrouter_api_key("openrouter/openai/gpt-5"),
        #     **MODEL_CONFIG,
        # )
        # gemini_model = GeneralLlm(
        #     model="openrouter/google/gemini-2.5-pro",
        #     reasoning={"max_tokens": 8000},
        #     **MODEL_CONFIG,
        # )
        # o3_model = GeneralLlm(
        #     model="openrouter/openai/o3",
        #     reasoning_effort="high",
        #     api_key=get_openrouter_api_key("openrouter/openai/o3"),
        #     **MODEL_CONFIG,
        # )
        # grok_model = GeneralLlm(
        #     model="openrouter/x-ai/grok-4",
        #     reasoning={"effort": "high"},
        #     **MODEL_CONFIG,
        # )

        # Individual model configurations for benchmarking
        # Test each model separately - ensembles will be generated post-hoc by analyze_correlations.py
        individual_models = [
            {"name": "r1-0528", "forecaster": free_r1_0528_model},
            {"name": "qwen3-coder", "forecaster": free_qwen3_coder_model},
            {"name": "glm-4.5-air", "forecaster": free_glm_4p5_air_model},
            # {"name": "qwen3-235b", "forecaster": qwen3_model},
            # {"name": "glm-4.5", "forecaster": glm_model},
            # Additional models - commented for cost control during development:
            # {"name": "deepseek-r1", "forecaster": deepseek_model},
            # {"name": "claude-sonnet-4", "forecaster": claude_model},
            # {"name": "gpt-5", "forecaster": gpt5_model},
            # {"name": "gemini-2.5-pro", "forecaster": gemini_model},
            # {"name": "o3", "forecaster": o3_model},
            # {"name": "grok-4", "forecaster": grok_model},
        ]

        # Generate individual model bots - ensembles generated by CorrelationAnalyzer.find_optimal_ensembles()
        bots = []
        for model_config in individual_models:
            bot = TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                aggregation_strategy=AggregationStrategy.MEAN,  # Default, not used for single models
                llms={
                    "forecasters": [model_config["forecaster"]],
                    **DEFAULT_HELPER_LLMS,
                },
                max_concurrent_research=batch_size,
                research_cache=research_cache,
            )
            bot.name = model_config["name"]
            bots.append(bot)

        logger.info(
            f"Created {len(bots)} individual model bots for benchmarking. Ensembles will be generated post-hoc by correlation analysis."
        )
        bots = typeguard.check_type(bots, list[ForecastBot])

        # Log progress info
        total_predictions = len(bots) * len(questions)
        logger.info(
            f"🚀 Starting benchmark: {len(bots)} bots × {len(questions)} questions = {total_predictions} total predictions"
        )

        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=batch_size,
        ).run_benchmark()
        try:
            for i, benchmark in enumerate(benchmarks):
                logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
                logger.info(
                    f"- Final Metaculus Baseline Score: {benchmark.average_expected_baseline_score:.4f} (based on log score, 0=always predict same, https://www.metaculus.com/help/scores-faq/#baseline-score )"
                )
                logger.info(f"- Total Cost: {benchmark.total_cost:.2f}")
                logger.info(f"- Time taken: {benchmark.time_taken_in_minutes:.4f}")
        except ValueError as ve:
            # Provide clearer guidance when no reports exist (likely research provider failures)
            raise RuntimeError(
                "Benchmark produced no forecast reports." "Fallback is disabled for benchmarks by design."
            ) from ve
        logger.info(f"Total Cost: {cost_manager.current_usage}")

        # Log score scale validation for mixed question types
        log_score_scale_validation(benchmarks)

        # TODO: refactor out this logic, jank to have here.
        # Perform correlation analysis if we have multiple models
        if len(benchmarks) > 1:
            from metaculus_bot.correlation_analysis import CorrelationAnalyzer

            analyzer = CorrelationAnalyzer()
            analyzer.add_benchmark_results(benchmarks)

            # Generate and log correlation report
            report = analyzer.generate_correlation_report("benchmarks/correlation_analysis.md")
            logger.info("\n" + "=" * 50)
            logger.info("CORRELATION ANALYSIS")
            logger.info("=" * 50)
            logger.info(report)

            # Generate all possible ensemble combinations with different aggregation strategies
            logger.info("\n" + "=" * 50)
            logger.info("ENSEMBLE GENERATION (Post-hoc)")
            logger.info("=" * 50)
            optimal_ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=6, max_cost_per_question=1.0)
            if optimal_ensembles:
                logger.info(
                    f"Generated {len(optimal_ensembles)} ensemble combinations from {len(benchmarks)} individual models"
                )
                logger.info(f"\nTop 10 Recommended Ensembles (Both Aggregation Strategies, Cost ≤ $1.0/question):")
                for i, ensemble in enumerate(optimal_ensembles[:10], 1):
                    models = " + ".join(ensemble.model_names)
                    logger.info(f"{i}. {models} ({ensemble.aggregation_strategy.upper()})")
                    logger.info(
                        f"   Score: {ensemble.avg_performance:.2f} | "
                        f"Cost: ${ensemble.avg_cost:.3f} | "
                        f"Diversity: {ensemble.diversity_score:.3f} | "
                        f"Overall: {ensemble.ensemble_score:.3f}"
                    )

                logger.info(
                    f"\n💡 Use 'python analyze_correlations.py benchmarks/' to explore all {len(optimal_ensembles)} ensemble combinations"
                )
            else:
                logger.info("No viable ensemble combinations found within cost constraints")
        else:
            logger.info("Skipping correlation analysis (need multiple models)")


if __name__ == "__main__":
    # Create handlers with explicit flushing for real-time output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.flush = lambda: sys.stdout.flush()

    file_handler = logging.FileHandler(f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Enable forecasting-tools logging for progress visibility
    forecasting_logger = logging.getLogger("forecasting_tools")
    forecasting_logger.setLevel(logging.INFO)
    forecasting_logger.propagate = True

    # Enable our main modules
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)

    # Suppress noisy third-party loggers but keep errors visible
    for noisy_logger in ["LiteLLM", "httpx", "httpcore", "urllib3", "aiohttp"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
        logging.getLogger(noisy_logger).propagate = False

    original_info = logging.Logger.info

    def flushing_info(self, message, *args, **kwargs):
        result = original_info(self, message, *args, **kwargs)
        sys.stdout.flush()
        return result

    logging.Logger.info = flushing_info

    # Parse command line arguments
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
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode, args.num_questions, args.mixed))
