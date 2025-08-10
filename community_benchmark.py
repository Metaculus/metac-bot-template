from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Literal

import typeguard
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
from metaculus_bot.llm_configs import FORECASTER_LLMS, PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

logger = logging.getLogger(__name__)


async def benchmark_forecast_bot(mode: str, number_of_questions: int = 2) -> None:
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
        "numeric_aggregation_method": "mean",
        "research_provider": None,  # Use default provider selection
        "max_questions_per_run": None,  # No limit for benchmarking
    }
    MODEL_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 8000,  # Prevent truncation issues with reasoning models
        "stream": False,
        "timeout": 180,
        "allowed_tries": 3,
    }
    DEFAULT_HELPER_LLMS = {
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }

    with MonetaryCostManager() as cost_manager:
        bots = [
            # FIXME: temp disable to do model-by-model bench
            # Full ensemble bot using production configuration
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": FORECASTER_LLMS,  # Our current ensemble approach
            #         "summarizer": SUMMARIZER_LLM,
            #         "parser": PARSER_LLM,
            #         "researcher": RESEARCHER_LLM,
            #     },
            # ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/anthropic/claude-sonnet-4",
                            reasoning={"max_tokens": 4000},
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            # Best single forecaster -- GPT-5 only bot for comparison
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/openai/gpt-5",
                            reasoning_effort="medium",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            # TODO: single model benchmark vs g2.5pro, o3, grok4, sonnet4 (thinking?), r1-0528, qwen3-235b, glm2.5
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/z-ai/glm-4.5",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/deepseek/deepseek-r1-0528",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/google/gemini-2.5-pro",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/openai/o3",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/x-ai/grok-4",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
            ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=30,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
            logger.info(
                f"- Final Metaculus Baseline Score: {benchmark.average_expected_baseline_score:.4f} (based on log score, 0=always predict same, https://www.metaculus.com/help/scores-faq/#baseline-score )"
            )
            logger.info(f"- Total Cost: {benchmark.total_cost:.2f}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes:.4f}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")

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

            # Log optimal ensembles for easy reference
            optimal_ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=6, max_cost_per_question=1.0)
            if optimal_ensembles:
                logger.info(f"\nTop 3 Recommended Ensembles (Cost ≤ $0.50/question):")
                for i, ensemble in enumerate(optimal_ensembles[:3], 1):
                    models = " + ".join(ensemble.model_names)
                    logger.info(
                        f"{i}. {models} | Score: {ensemble.avg_performance:.2f} | "
                        f"Cost: ${ensemble.avg_cost:.3f} | Diversity: {ensemble.diversity_score:.3f}"
                    )
        else:
            logger.info("Skipping correlation analysis (need multiple models)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
        ],
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

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
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode, args.num_questions))
