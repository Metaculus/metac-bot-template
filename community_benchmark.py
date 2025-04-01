from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Literal

import typeguard
from forecasting_tools import (
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MonetaryCostManager,
    MetaculusApi,
    ApiFilter,
    run_benchmark_streamlit_page,
)

from main import TemplateForecaster

logger = logging.getLogger(__name__)



async def benchmark_forecast_bot(mode: str) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction
    """

    number_of_questions = 30 # Recommend 100+ for meaningful error bars, but 30 is faster/cheaper
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
            question.background_info = None # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    with MonetaryCostManager() as cost_manager:
        bots = [
            TemplateForecaster(
                predictions_per_research_report=5,
                llms={
                    "default": GeneralLlm(
                        model="gpt-4o-mini",
                        temperature=0.3,
                    ),
                },
            ),
            TemplateForecaster(
                predictions_per_research_report=1,
                llms={
                    "default": GeneralLlm(
                        model="gpt-4o-mini",
                        temperature=0.3,
                    ),
                },
            ),
            # Add other ForecastBots here (or same bot with different parameters)
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=10,
        ).run_benchmark()
        for i, benchmark in enumerate(benchmarks):
            logger.info(
                f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}"
            )
            logger.info(
                f"- Final Score: {benchmark.average_expected_baseline_score}"
            )
            logger.info(f"- Total Cost: {benchmark.total_cost}")
            logger.info(f"- Time taken: {benchmark.time_taken_in_minutes}")
        logger.info(f"Total Cost: {cost_manager.current_usage}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        ]
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Benchmark a list of bots"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "custom", "display"],
        default="display",
        help="Specify the run mode (default: display)",
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = (
        args.mode
    )
    asyncio.run(benchmark_forecast_bot(mode))


