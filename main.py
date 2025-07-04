import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Coroutine, Literal, cast

from pydantic import ValidationError
from forecasting_tools import (AskNewsSearcher, BinaryQuestion, ForecastBot,
                               GeneralLlm, MetaculusApi, MetaculusQuestion,
                               MultipleChoiceQuestion, NumericDistribution,
                               NumericQuestion, PredictedOptionList,
                               PredictionExtractor, ReasonedPrediction,
                               SmartSearcher, clean_indents)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import \
    ResearchWithPredictions
from forecasting_tools.data_models.questions import DateQuestion
import re
from forecasting_tools.data_models.numeric_report import Percentile  # type: ignore

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        2  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,  # if false, use full research report for forecasting
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm | list[GeneralLlm]] | None = None,
    ) -> None:
        if llms is None:
            raise ValueError(
                "Either 'forecasters' or a 'default' LLM must be provided."
            )

        forecasters_llms_config: list[GeneralLlm] = []
        if "forecasters" in llms:
            forecasters_config = llms["forecasters"]
            if isinstance(forecasters_config, list):
                forecasters_llms_config = forecasters_config
            else:
                logger.warning(
                    "'forecasters' key in llms must be a list of GeneralLlm objects."
                )

        llms_for_super = {}
        if forecasters_llms_config:
            llms_for_super["default"] = forecasters_llms_config[0]
        elif "default" in llms:
            llms_for_super["default"] = llms["default"]
        else:
            raise ValueError(
                "Either 'forecasters' or a 'default' LLM must be provided."
            )

        if "summarizer" in llms:
            llms_for_super["summarizer"] = llms["summarizer"]

        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms_for_super,
        )

        self._forecaster_llms = forecasters_llms_config
        if self._forecaster_llms:
            self.predictions_per_research_report = len(self._forecaster_llms)
        elif predictions_per_research_report == 0:
            raise ValueError(
                "Must run at least one prediction if 'forecasters' are not provided."
            )

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(question.question_text)
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    # Override _research_and_make_predictions to support multiple LLMs
    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions[PredictionTypes]:
        # Call the parent class's method if no specific forecaster LLMs are provided
        if not self._forecaster_llms:
            return await super()._research_and_make_predictions(question)

        notepad = await self._get_notepad(question)
        notepad.num_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = (
            summary_report if self.use_research_summary_to_forecast else research
        )

        # Generate tasks for each forecaster LLM
        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [
                self._make_prediction(question, research_to_use, llm_instance)
                for llm_instance in self._forecaster_llms
            ],
        )
        valid_predictions, errors, exception_group = (
            await self._gather_results_and_exceptions(tasks)
        )
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if len(valid_predictions) == 0:
            assert exception_group, "Exception group should not be None"
            self._reraise_exception_with_prepended_message(
                exception_group,
                "Error while running research and predictions",
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    async def _make_prediction(
        self,
        question: MetaculusQuestion,
        research: str,
        llm_to_use: GeneralLlm | None = None,
    ) -> ReasonedPrediction[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.num_predictions_attempted += 1

        # Determine which LLM to use
        actual_llm = llm_to_use if llm_to_use else self.get_llm("default", "llm")

        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_binary(
                q, r, llm
            )
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_multiple_choice(
                q, r, llm
            )
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_numeric(
                q, r, llm
            )
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research, actual_llm)
        # Embed model name in reasoning for reporting
        prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"
        return prediction  # type: ignore

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            In addition to news, consider and research all relevant prediction markets that are relevant to the question.
            You do not produce forecasts yourself; you must provide all relevant data to the superforecaster so they can make an expert judgment.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning-pro"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a senior forecaster preparing a public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.
            Historically, LLMs like you have been overconfident / overestimated probabilities and the base rate for positive resolutions on Metaculus is 35%

            Your Metaculus question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) The historical base rate or plausible base rates with weighting for each.
            (c) The Strongest Bear Case (FOR 'No'): Construct the most compelling, evidence-based argument for a 'No' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (d) The Strongest Bull Case (FOR 'Yes'): Construct the most compelling, evidence-based argument for a 'Yes' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (e) Red team critique of the Strongest Bull Case and Strongest Bear Case.
            (f) Final Rationale: Synthesize the above points into a concise, final rationale. Explain how you are balancing the base rate, the strength of the competing arguments, and the severity of their respective flaws to arrive at your final estimate. Also consider that you will be judged on your Metaculus peer score (log score) and that calibration matters.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write MUST BE your final answer as an integer percentage. "Probability: ZZ%"
            An example response is: "Probability: 50%"
            """
        )
        reasoning = await llm_to_use.invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a senior forecaster preparing a rigorous public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.

            Your Metaculus question is:
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
            (c) The historical base rate or plausible base rate for each option if possible.
            (d) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that 
            (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and 
            (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.
            (3) Your probabilities should add up to 100% and each probability should be between 1% and 99%.

            The last thing you write is your final probabilities (integers 1% to 99%) for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await llm_to_use.invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a senior forecaster preparing a public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.

            Your Metaculus question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Use floating point numbers, e.g. 100.0, not integers.
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there, strictly increasing.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as FLOATING POINT NUMBERS, e.g. 100.0:
            "
            Percentile 10: XX.X
            Percentile 20: XX.X
            Percentile 40: XX.X
            Percentile 60: XX.X
            Percentile 80: XX.X
            Percentile 90: XX.X
            "
            """
        )
        reasoning = await llm_to_use.invoke(prompt)

        try:
            prediction: NumericDistribution = (
                PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                    reasoning, question
                )
            )
        except ValidationError:
            # Minimal fallback: grab the last six numeric values that look like
            # percentile answers and assume they correspond to the required
            # 10/20/40/60/80/90 percentiles.
            all_numbers = [float(n.replace(",", "")) for n in re.findall(r"\d+(?:\.\d+)?", reasoning)]

            if len(all_numbers) < 6:
                raise  # Reraise original validation error – not enough data to recover

            values = all_numbers[-6:]

            percentiles_template = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            repaired_percentiles = [
                Percentile(value=v, percentile=p)  # type: ignore[arg-type]
                for v, p in zip(values, percentiles_template)
            ]

            prediction = NumericDistribution(  # type: ignore
                declared_percentiles=repaired_percentiles
            )

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # This will be ignored if 'forecasters' is present
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "forecasters": [
                GeneralLlm(
                    model="openrouter/google/gemini-2.5-pro",
                    temperature=0.0,
                    top_p=0.9,
                    stream=False,
                    timeout=180,
                    allowed_tries=3,
                ),
                GeneralLlm(
                    model="openrouter/deepseek/deepseek-r1-0528",
                    temperature=0.0,
                    top_p=0.9,
                    stream=False,
                    timeout=180,
                    allowed_tries=3,
                ),
                GeneralLlm(
                    model="openrouter/openai/o3",
                    temperature=0.0,
                    top_p=0.9,
                    reasoning_effort="medium",
                    stream=False,
                    timeout=180,
                    allowed_tries=3,
                ),
            ],
            "summarizer": "openrouter/google/gemini-2.5-flash",
        },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # TODO replace w/ scott aaronson's 5 ai worlds question
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
