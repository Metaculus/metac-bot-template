import argparse
import asyncio
import logging
from datetime import date
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
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

    if run_mode == "aib_tournament":
        logger.info("Tournament mode not implemented yet") #TODO
    elif run_mode == "metaculus_cup":
        logger.info("Metaculus cup mode not implemented yet") #TODO
    elif run_mode == "mini_bench":
        logger.info("Mini bench mode not implemented yet") #TODO
    elif run_mode == "test_questions":
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
                MetaculusApi.post_binary_question_prediction(question.id_of_question,0.5)                
            elif question.question_type == "numeric":
                continue #TODO jvodlz
            elif question.question_type == "multiple_choice":
                num_options = len(question.options)
                probability_per_option = 1.0 / num_options
                probabilities: dict[str, float] = dict.fromkeys(question.options, probability_per_option)   
                MetaculusApi.post_multiple_choice_question_prediction(question.id_of_question, probabilities)
            elif question.question_type == "date":
                continue #As of August 2025, this question type is still not supported by Metaculus for bots
            elif question.question_type == "discrete":
                continue #TODO jvodlz