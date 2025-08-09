import argparse
import asyncio
# ruff: noqa: F401
import logging
from typing import Literal

from forecasting_tools import MetaculusApi
from metaculus_bot.llm_configs import FORECASTER_LLMS, SUMMARIZER_LLM

# NOTE: TemplateForecaster is still defined in main.py during the first refactor phase.
from main import TemplateForecaster


def main() -> None:
    """Command-line entry-point for running the TemplateForecaster.

    This code was moved verbatim from the bottom of main.py so external behaviour
    (e.g. GitHub Actions invoking `python main.py`) remains identical.  The only
    difference is that main.py now delegates to this function.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the Q1TemplateBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Ignored when 'forecasters' present
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        numeric_aggregation_method="mean",
        llms={
            "forecasters": FORECASTER_LLMS,
            "summarizer": SUMMARIZER_LLM,
        },
    )

    if run_mode == "tournament":
        template_bot.skip_previously_forecasted_questions = True  # to not risk explosive spend, we won't update preds
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = True  # to not risk explosive spend, we won't update preds
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True)
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/questions/20683/which-ai-world/",  # Scott Aaronson's five AI worlds
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False  # obviously, we need to rerun test q predictions to test them :)
        questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))
    else:
        raise ValueError(f"Invalid run mode: {run_mode}")

    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore


if __name__ == "__main__":
    main() 