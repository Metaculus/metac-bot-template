import argparse
import asyncio
import logging
from datetime import date, datetime
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
    ApiFilter,
)

import json
import utils

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class WobblyBot2025Q3(ForecastBot):
    _max_concurrent_questions = (5)
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}

                Your most important job is to search polymarket.com, then kalshi.com, then metaculus.com, then manifold.markets for their predictions and report their predictions as percentages. If a question exists on those websites, the percentages there will always be available and you should report their value.
                
                You'll always be able to find the same question on metaculus.com because you're being given a question from there, so keep trying until you find it.
                
                If it's a question about a sporting event, also search betting markets such as betfair.com and oddschecker.com and calculate the implied probabilities form the odds.
                
                Complement by searching and reporting on other websites that could have predictions by superforecasters or other type of professional forecasts and give the values of their predictions.
                                
                Then, if applicable for this question, search and report the base rates. Your should inform how many times the event happened in the last (up to) 40 years and what the period considered was. 
                
                After that, report on the current status of the situation as of the current date.

                Finally, generate a concise but detailed rundown of the most relevant recent news, including if the question would resolve Yes or No based on current information. Report as many news as possible.
                
                For each task and for each one of the 6 websites listed before, report their predictions separately or report that you couldn't find anything there or that they're not applicable for the question.

                Explain how you're following each of those steps.

                Finish by delivering the full final report. There won't be another follow-up prompt, so do as much as asked now.

                """
            )

            if isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.  Think very hard before answering.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            Your research assistant says:
            {research}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            If the research was able to find probabilities from prediction markets, your prediction should mostly be based on that, with few adjustements to account for recent news.
            If data from prediction markets was not available, make your prediction based on the base rates, if available, and your independent rationale.
            Less importantly, also take into account the all the recent news from the report.
            If base rates and prediction markets data are unavailable, make your prediction based on the recent news.
            Make sure that, if the event being forecasted can happen any time, then assume that the probability of the event happening decays linearly over time, so that if not a lot of time if left until resolution, you would give a very low probability. For this, first estimate a prediction of the event happening in the next 12 months and then convert that probability considering how many days until the resolution are in fact left. If not many days are left, make sure the result is a low probability. This is not applicable for events that only happen once at a specified date, such as an election.
            Good forecasters also leave room for unknown unknowns, so make sure to never predict anything below 4% ou above 96%
            Explain how you're following each of those steps.

            """
        )
        if bot.verify_community_prediction_exists(question):
            logger.info(f"Question {question.id_of_question} has community prediction")
            lower_bound, upper_bound = self.community_prediction_divergence(question)
            cp_prompt = f"""
            Make sure your prediction is not below {lower_bound} and not above {upper_bound}.

            """
            prompt += cp_prompt

        final_instructions = f"""
            You write your rationale remembering that good forecasters put large weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
        """
        prompt += final_instructions
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. Think very hard before answering.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in the most probable choices
            
            If the research was able to find probabilities from prediction markets, your prediction should mostly be based on that, with few adjustements to account for recent news.
            If data from prediction markets was not available, make your prediction based on the base rates, if available, and your independent rationale.
            Less importantly, also take into account the all the recent news from the report.
            If base rates and prediction markets data are unavailable, make your prediction based on the recent news.
            Good forecasters also leave room for unknown unknowns, so make sure to never predict anything below 2% ou above 98% for each of the options.
            Explain how you're following each of those steps.
            
            You write your rationale remembering that good forecasters put large weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. Think very hard before answering

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) A brief description of an scenario that results in a low outcome.
            (e) A brief description of an scenario that results in a high outcome.

            If the research was able to find probabilities from prediction markets, your prediction should mostly be based on that, with few adjustements to account for recent news.
            If data from prediction markets was not available, make your prediction based on the base rates, if available, and your independent rationale.
            Less importantly, also take into account the all the recent news from the report.
            If base rates and prediction markets data are unavailable, make your prediction based on the recent news.
            Good forecasters also leave room for unknown unknowns, so follow the instructions below:
            {lower_bound_message}
            {upper_bound_message}               
            Also make sure that your range is not too narrow to account for many different possible values.
            
            Explain how you're following each of those steps.

            You write your rationale remembering that good forecasters put large weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as:
            "
            Percentile 5: XX
            Percentile 10: XX
            Percentile 15: XX
            Percentile 20: XX
            Percentile 30: XX
            Percentile 40: XX
            Percentile 50: XX
            Percentile 60: XX
            Percentile 70: XX
            Percentile 80: XX
            Percentile 85: XX
            Percentile 90: XX
            Percentile 95: XX
            "
            """
        )
        reasoning = await self.get_llm("forecaster", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
    
    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}, so make your Percentile 40 not higher than {upper_bound_number}, but make sure your Percentiles 85, 90 and 95 are higher than or equal to {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}, so make your Percentile 70 lower than{upper_bound_number}, but make your Percentiles 90 and 95 equal to {upper_bound_number}."
            )
            if question.question_type == "discrete":
                upper_bound_message = (
                    f"The outcome can not be higher than {upper_bound_number}, so make your Percentile 60 lower than {upper_bound_number}, but make your Percentiles 85, 90 and 95 equal to {upper_bound_number}."
                )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}, so make your Percentile 70 higher than {lower_bound_number}, but make sure your Percentiles 15, 10 and 5 are lower than or equal to {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}, so make your Percentile 60 higher than {lower_bound_number}, but make your Percentiles 10 and 5 equal to {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        prediction_date_dict: dict,
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:

        #TODO move the commented tests below to a new test mode
        # qturl = "https://www.metaculus.com/questions/39056/" # binary ishiba
        # qt = MetaculusApi.get_question_by_url(qturl)
        # questions_to_forecast = []
        # questions_to_forecast.append(qt)

        # qturl = "https://www.metaculus.com/questions/37322/" #multiple choice 2028 democrats
        # qt = MetaculusApi.get_question_by_url(qturl)
        # questions_to_forecast = []
        # questions_to_forecast.append(qt)

        # qturl = "https://www.metaculus.com/questions/26718/" #numeric 5 years after agi happiness
        # qt = MetaculusApi.get_question_by_url(qturl)
        # questions_to_forecast = []
        # questions_to_forecast.append(qt)

        # qturl = "https://www.metaculus.com/c/diffusion-community/38880" # discrete ai protests
        # qt = MetaculusApi.get_question_by_url(qturl)
        # questions_to_forecast = []
        # questions_to_forecast.append(qt)

        today = date.today().isoformat()

        questions_to_forecast = []
        for q in questions:
            # if q.question_text.startswith("[PRACTICE]"):
            #     logger.info(f"Skipping practice question {q.id_of_question}: {q.question_text}")
            #     continue
            if q.already_forecasted and prediction_date_dict.get(str(q.id_of_question)) == today:
                logger.info(f"Already made a prediction today on question {q.id_of_question}: {q.question_text}")
                continue        
            questions_to_forecast.append(q)

        if not questions_to_forecast:
            logger.info("No new tournament questions to forecast at this time")
            return []

        logger.info(f"Found {len(questions_to_forecast)} new or outdated questions to forecast")

        reports: list[ForecastReport | BaseException] = []
        reports = await asyncio.gather(
            *[
                self._run_individual_question_with_error_propagation(question)
                for question in questions_to_forecast
            ],
            return_exceptions=return_exceptions,
        )

        return reports
    
    def verify_community_prediction_exists(
        self,
        question: MetaculusQuestion,
    ) -> bool:
        res = json.loads(question.model_dump_json())
        try:
            reveal_time = datetime.fromisoformat(res["cp_reveal_time"])
            return reveal_time < datetime.now()
        except (KeyError, TypeError, ValueError):
            return False

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
    
    def community_prediction_divergence(self, question: MetaculusQuestion) -> tuple[float, float]:
        if question.question_type in ["binary"]:
            prediction = utils.get_binary_community_prediction(question)
            if prediction is not None:
                return prediction * 0.7, prediction * 1.3
                # return prediction - 0.25, (prediction / (1 - prediction)) / (0.25 / (1 - 0.25))

        return 0.0, 0.0

    
    @staticmethod
    def load_data_from_file(filepath) -> dict:
        data = {}
        try:
            with open(filepath, "r") as f:
                for line in f:
                    # Skip empty lines
                    if line.strip():
                        # Split only on the first colon to handle values that might contain colons
                        key, value = line.strip().split(":", 1)
                        data[key] = value
        except FileNotFoundError:
            # If the file doesn't exist, just return an empty dictionary
            logger.error(f"'{filepath}' not found. Starting with an empty dataset")
        return data
    
    @staticmethod
    def save_data_to_file(data: dict, filepath):
        with open(filepath, "w") as f:
            for key, value in data.items():
                f.write(f"{key}:{value}\n")
        print(f"Successfully saved data to '{filepath}'")

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
        research_reports_per_question=2,
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
                # model="openrouter/openai/o3",
                # model="openrouter/openai/gpt-5-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "researcher": GeneralLlm(
                # model="openrouter/openai/gpt-5:online",
                model="openrouter/openai/gpt-4o-search-preview",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                # model="openrouter/openai/gpt-5-mini",
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": GeneralLlm(
                # model="openrouter/openai/gpt-5-mini",
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            )
        }
    )

if run_mode == "aib_tournament":
    logger.info("Running Wobbly Bot in AIB tournament mode")

    prediction_file = "latest_prediction_dates_aib_tournament.txt"
    prediction_date_dict = bot.load_data_from_file(prediction_file)
    today = date.today().isoformat()

    questions = MetaculusApi.get_all_open_questions_from_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID)
    reports = asyncio.run(bot.forecast_questions(questions, prediction_date_dict, return_exceptions=True))

    # Only updates the status of successful predictions
    for report in reports:
        if isinstance(report, ForecastReport):
            question_id = str(report.question.id_of_question)
            prediction_date_dict[question_id] = today
            logger.info(f"Successfully processed and logged today's date for question ID: {question_id}")

    bot.save_data_to_file(prediction_date_dict, prediction_file)

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
        #38667: US Undergrad Enrollment Decline from 2024 to 2030 - Binary
        "https://www.metaculus.com/questions/39314/us-undergraduate-enrollment-decline-by-10-from-2024-to-2030",
        #26268: 5Y After AGI - AI Philosophical Competence - Binary
        "https://www.metaculus.com/questions/26268/5y-after-agi-ai-philosophical-competence/",
        #14333: Age of Oldest Human - Numeric
        "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
        #22427: Number of New Leading AI Labs - Multiple Choice
        "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        #38880: Number of US Labor Strikes Due to AI in 2029 - Discrete  
        "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
    ]

    prediction_date_dict = bot.load_data_from_file("latest_prediction_dates_test_questions.txt")
    today = date.today().isoformat()

    for question_url in EXAMPLE_QUESTIONS:
        question = MetaculusApi.get_question_by_url(question_url)
        has_community_prediction = bot.verify_community_prediction_exists(question)
        logger.info(f"QID: {question.id_of_question} of type: <{question.question_type}> has community prediction: {has_community_prediction}")
        
        if question.question_type == "binary":
            ## TESTING - Log default values (includes forecasted questions)
            # if has_community_prediction:
            #     dist_1, dist_2 = bot.community_prediction_divergence(question)
            #     print(f">>> divergence_1 = {dist_1}, divergence_2 = {dist_2}")

            if question.already_forecasted:
                if (today == prediction_date_dict.get(str(question.id_of_question))):
                    logger.info("Already made a prediction today on question " + str(question.id_of_question) + ": " + question.question_text)
                    continue
                logger.info("Updating the prediction on question " + str(question.id_of_question) + ": " + question.question_text)
                MetaculusApi.post_binary_question_prediction(question.id_of_question,bot.make_default_binary_prediction())
                prediction_date_dict[str(question.id_of_question)] = today
            else:
                if has_community_prediction:
                    dist_1, dist_2 = bot.community_prediction_divergence(question)
                    print(f">>> divergence_1 = {dist_1}, divergence_2 = {dist_2}")

                logger.info("Making the first prediction on question " + str(question.id_of_question) + ": " + question.question_text)
                MetaculusApi.post_binary_question_prediction(question.id_of_question,bot.make_default_binary_prediction())
                prediction_date_dict[str(question.id_of_question)] = today
         
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
    bot.save_data_to_file(prediction_date_dict, "latest_prediction_dates_test_questions.txt")