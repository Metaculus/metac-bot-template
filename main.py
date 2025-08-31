import argparse
import asyncio
import logging
from datetime import date
from typing import Literal, List, Sequence

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    ForecastReport,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class WobblyBot2025Q3(ForecastBot):
    _max_concurrent_questions = (1)
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        return "test research"
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return ReasonedPrediction(prediction_value=self.make_default_binary_prediction(), reasoning="test binary reason") #TODO
    
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        #FIXME The default method doesn't return the type needed here
        return ReasonedPrediction(prediction_value=self.make_default_multiple_choice_prediction(question), reasoning="test multiple choice reason")

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return ReasonedPrediction(prediction_value=self.make_default_numeric_prediction(question), reasoning="test numeric reason")
    
    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        #TODO Make the bot skip questions already predicted today
        reports: list[ForecastReport | BaseException] = []
        reports = await asyncio.gather(
            *[
                self._run_individual_question_with_error_propagation(question)
                for question in questions
            ],
            return_exceptions=return_exceptions,
        )
        return reports

    def make_default_binary_prediction(self):
        return 0.5
    
    def make_default_numeric_prediction(self, question: NumericQuestion):
        if (question.open_lower_bound and question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0.05, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=0.95, value=question.upper_bound),
            ]
        elif (question.open_lower_bound and not question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0.05, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=1, value=question.upper_bound),
            ]
        elif (not question.open_lower_bound and question.open_upper_bound):
            percentile_list: List[Percentile] = [
                Percentile(percentile=0, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=0.95, value=question.upper_bound),
            ]
        else:
            percentile_list: List[Percentile] = [
                Percentile(percentile=0, value=question.lower_bound),
                Percentile(percentile=0.50, value=(question.lower_bound + question.upper_bound) / 2),
                Percentile(percentile=1, value=question.upper_bound),
            ]
        return NumericDistribution.from_question(percentile_list, question)
    
    def make_default_multiple_choice_prediction(self, question: MultipleChoiceQuestion):
        num_options = len(question.options)
        probability_per_option = 1.0 / num_options
        probabilities: dict[str, float] = dict.fromkeys(question.options, probability_per_option)
        return probabilities

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the WobblyBot2025Q3"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["aib_tournament", "metaculus_cup", "mini_bench", "test_questions"],
        default="test_questions",
        help="Specify the run mode (default: test_questions)",
    )
    args = parser.parse_args()
    run_mode: Literal["aib_tournament", "metaculus_cup", "mini_bench", "test_questions"] = args.mode
    assert run_mode in [
        "aib_tournament",
        "metaculus_cup",
        "mini_bench",
        "test_questions",
    ], "Invalid run mode"

    bot = WobblyBot2025Q3(
        research_reports_per_question=1,
        predictions_per_research_report=2,
        enable_summarize_research=False,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={  
            "default": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "forecaster": GeneralLlm(
                model="openrouter/openai/gpt-5",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "researcher": GeneralLlm(
                model="openrouter/openai/gpt-5:online",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            )
        }
    )

if run_mode == "aib_tournament":
    logger.info("Running Wobbly Bot in AIB tournament mode")
    questions = MetaculusApi.get_all_open_questions_from_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID)
    asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

elif run_mode == "metaculus_cup":
    logger.info("Metaculus Cup mode not implemented yet") #TODO
elif run_mode == "mini_bench":
    logger.info("Mini Bench mode not implemented yet") #TODO
elif run_mode == "test_questions":
    logger.info("Running Wobbly Bot in test mode")

    EXAMPLE_QUESTIONS = [
        #578: Human Extinction - Binary
        "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  
        #8632: Total Yield of Nuc Det 1000MT by 2050 - Binary
        "https://www.metaculus.com/questions/8632/total-yield-of-nuc-det-1000mt-by-2050/",
        #14333: Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        #22427: Number of New Leading AI Labs - Multiple Choice  
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        #38880: Number of US Labor Strikes Due to AI in 2029 - Discrete  
        "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
    ]

    for question_url in EXAMPLE_QUESTIONS:
        question = MetaculusApi.get_question_by_url(question_url)
        if question.question_type == "binary":
            MetaculusApi.post_binary_question_prediction(question.id_of_question,bot.make_default_binary_prediction())           
        elif question.question_type == "numeric":
            prediction = bot.make_default_numeric_prediction(question)
            cdf = [percentile.percentile for percentile in prediction.cdf]
            MetaculusApi.post_numeric_question_prediction(question.id_of_question, cdf)
        elif question.question_type == "multiple_choice":
            MetaculusApi.post_multiple_choice_question_prediction(question.id_of_question, bot.make_default_multiple_choice_prediction(question))
        elif question.question_type == "date":
            continue # As of August 2025, this question type is still not supported by Metaculus for bots
        elif question.question_type == "discrete":
            #TODO make the code for disctrete questions more specialized
            prediction = bot.make_default_numeric_prediction(question)
            cdf = [percentile.percentile for percentile in prediction.cdf]
            MetaculusApi.post_numeric_question_prediction(question.id_of_question, cdf)