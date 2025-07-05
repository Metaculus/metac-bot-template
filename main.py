import argparse
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Coroutine, Literal, Sequence, cast

import numpy as np
from forecasting_tools import (AskNewsSearcher, BinaryQuestion, ForecastBot,
                               GeneralLlm, MetaculusApi, MetaculusQuestion,
                               MultipleChoiceQuestion, NumericDistribution,
                               NumericQuestion, PredictedOptionList,
                               PredictionExtractor, ReasonedPrediction,
                               SmartSearcher, clean_indents)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import (
    ForecastReport, ResearchWithPredictions)
from forecasting_tools.data_models.numeric_report import (  # type: ignore
    NumericReport, Percentile)
from forecasting_tools.data_models.questions import DateQuestion
from pydantic import ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
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
        numeric_aggregation_method: Literal["mean", "median"] = "mean",
        llms: dict[str, str | GeneralLlm | list[GeneralLlm]] | None = None,
    ) -> None:
        if llms is None:
            raise ValueError("Either 'forecasters' or a 'default' LLM must be provided.")

        forecasters_llms_config: list[GeneralLlm] = []
        if "forecasters" in llms:
            forecasters_config = llms["forecasters"]
            if isinstance(forecasters_config, list):
                forecasters_llms_config = forecasters_config
            else:
                logger.warning("'forecasters' key in llms must be a list of GeneralLlm objects.")

        llms_for_super = {}
        if forecasters_llms_config:
            llms_for_super["default"] = forecasters_llms_config[0]
        elif "default" in llms:
            llms_for_super["default"] = llms["default"]
        else:
            raise ValueError("Either 'forecasters' or a 'default' LLM must be provided.")

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
            raise ValueError("Must run at least one prediction if 'forecasters' are not provided.")
        self.numeric_aggregation_method = numeric_aggregation_method

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(question.question_text)
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(question.question_text)
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(question.question_text, use_open_router=True)
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
        research_to_use = summary_report if self.use_research_summary_to_forecast else research

        # Generate tasks for each forecaster LLM
        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [self._make_prediction(question, research_to_use, llm_instance) for llm_instance in self._forecaster_llms],
        )
        valid_predictions, errors, exception_group = await self._gather_results_and_exceptions(tasks)
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
            forecast_function = lambda q, r, llm: self._run_forecast_on_binary(q, r, llm)
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_multiple_choice(q, r, llm)
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_numeric(q, r, llm)
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research, actual_llm)
        # Embed model name in reasoning for reporting
        prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"
        return prediction  # type: ignore

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
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
            Historically, LLMs like you have overestimated probabilities, and the base rate for positive resolutions on Metaculus is 35%.

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
            (d) The Strongest Bear Case (FOR 'No'): Construct the most compelling, evidence-based argument for a 'No' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (e) The Strongest Bull Case (FOR 'Yes'): Construct the most compelling, evidence-based argument for a 'Yes' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (f) Red team critique of the Strongest Bull Case and Strongest Bear Case.
            (g) Final Rationale: Synthesize the above points into a concise, final rationale. Explain how you are balancing the base rate, the strength of the competing arguments, and the severity of their respective flaws to arrive at your final estimate. Also consider that you will be judged on your Metaculus peer score (log score) and that calibration matters.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"
            An example response is: "Probability: 50%"
            """
        )
        reasoning = await llm_to_use.invoke(prompt)
        self._log_raw_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )

        logger.info(f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
        You are a **senior forecaster** preparing a rigorous public report for expert peers.
        Your accuracy and *calibration* will be scored with Metaculus' log-score, so avoid
        over-confidence and make sure your probabilities sum to **100 %**.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}

        • Options (in resolution order): {question.options}

        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today's date: {datetime.now().strftime("%Y-%m-%d")}

        ── Write your analysis in the following numbered sections ────────────
        (1) **Time to resolution**: how long until the panel can decide.

        (2) **Status-quo outcome**: if present trends simply continue, which
            option is most plausible and why?

        (3) **Base-rate & expert priors**: assemble a table like:
            Option | Historical / analogous base-rate | Expert / market signal
            -------|-----------------------------------|-----------------------
            A      | …                                 | …
            …      | …                                 | …

        (4) **Strongest pro case** for the *currently most-likely* option
            (use evidence & causal chains from the briefing).

        (5) **Red-team critique**: attack the argument in (4); highlight
            hidden assumptions or data that could flip the conclusion.

        (6) **Unexpected scenario**: outline a plausible but overlooked
            pathway that would make a different option win.

        (7) **Final rationale**: reconcile everything above into calibrated
            probabilities.  Remember:
            • Good forecasters leave a little probability on most options.
            • Use integers 1-99 (no 0 % or 100 %).
            • They must sum to 100 %.

        ── OUTPUT FORMAT (must be last lines, nothing after) ────────────────
        Option_A: NN%
        Option_B: NN%
        …
        Option_N: NN%
        """
        )
        reasoning = await llm_to_use.invoke(prompt)
        self._log_raw_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        prediction: PredictedOptionList = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options
        )
        logger.info(f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
        You are a **senior forecaster** writing a public report for expert peers.
        You will be scored with Metaculus’ log-score, so accuracy **and** calibration
        (especially the width of your 90 / 10 interval) are critical.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}

        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        Units: {question.unit_of_measure or "Not stated: infer if possible"}

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today’s date: {datetime.now().strftime("%Y-%m-%d")}

        {lower_bound_message}
        {upper_bound_message}

        ── Write your analysis in the following numbered sections ────────────
        (1) **Time to resolution**: how long until we know the answer.

        (2) **Status-quo outcome**: what value is implied if current
            conditions simply persist?

        (3) **Trend continuation**: extrapolate historical data to 
            the closing date.

        (4) **Expert & market priors**: cite ranges or point forecasts from
            specialists, prediction markets, or peer forecasts.

        (5) **Unexpected low scenario**: describe a coherent pathway that
            would push the result into an unusually *low* tail.

        (6) **Unexpected high scenario**: analogous pathway for an unusually
            *high* tail.

        (7) **Red-team critique & final rationale**: challenge your own
            assumptions, then state how you weight everything to set each
            percentile.  Good forecasters:
            • keep 10 / 90 far apart (unknown unknowns)  
            • ensure strictly increasing values  
            • avoid scientific notation  
            • respect the explicit bounds above.

        ── OUTPUT FORMAT, floating point numbers (must be last lines, nothing after) ────────────────
        Percentile 10: XX.X
        Percentile 20: XX.X
        Percentile 40: XX.X
        Percentile 60: XX.X
        Percentile 80: XX.X
        Percentile 90: XX.X
        """
        )
        # TODO: ideally would use JSON above ^^
        reasoning = await llm_to_use.invoke(prompt)

        logger.info(
            f"""
            >>>>>>>>>>>>>>>>>> Raw LLM Output Start >>>>>>>>>>>>>>>>>
            LLM: {llm_to_use.model}
            Question ID: {question.id_of_question}

            {reasoning}
            <<<<<<<<<<<<<<<<<< Raw LLM Output End <<<<<<<<<<<<<<<<<
            """
        )

        try:
            prediction: NumericDistribution = (
                PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                    reasoning, question
                )
            )
            # Ensure we extracted all 6 required percentiles (10,20,40,60,80,90).
            if (
                hasattr(prediction, "declared_percentiles")
                and isinstance(prediction.declared_percentiles, list)
                and len(prediction.declared_percentiles) != 6
            ):
                raise ValidationError.from_exception_data(
                    "NumericDistribution",  # title
                    [
                        {
                            "type": "value_error",
                            "loc": ("declared_percentiles",),
                            "input": prediction.declared_percentiles,
                            "ctx": {
                                "error": "Expected 6 declared percentiles (10,20,40,60,80,90).",
                            },
                        }
                    ],
                )
        except ValidationError as err:
            # Fallback: find lines with "percentile", extract the last number,
            # and assume they correspond to the required 10/20/40/60/80/90.
            # Use strict filtering for percentile lines.
            logger.warning("Attempting to repair numeric distribution from LLM output.")
            percentile_lines = []
            for line in reasoning.split("\n"):
                match = re.match(
                    r"^[Pp]ercentile\s+\d+:\s+[-]?\d+(?:,\d{3})*(?:\.\d+)?$",
                    line.strip(),
                )
                if match:
                    percentile_lines.append(line)

            if len(percentile_lines) != 6:
                logger.warning("Did not receive exactly 6 valid percentile lines after strict filtering.")
                raise err  # Re-raise original error if strict filtering fails to find 6 lines

            values = []
            for line in percentile_lines:
                # Extract the last number from the strictly filtered line
                numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
                if numbers:
                    values.append(float(numbers[-1].replace(",", "")))

            if len(values) != 6:
                raise ValueError("Could not extract 6 numeric values from strictly filtered percentile lines.")

            percentiles_template = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            repaired_percentiles = [Percentile(value=v, percentile=p) for v, p in zip(values, percentiles_template)]

            prediction = NumericDistribution(
                declared_percentiles=repaired_percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=question.zero_point,
            )

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # Agg with MEAN not median so we can use fewer LLM calls.
    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if isinstance(question, BinaryQuestion):
            # Custom aggregation for BinaryQuestions: mean rounded to one decimal place
            if not predictions:
                raise ValueError("Cannot aggregate empty list of predictions")

            # Ensure all predictions are floats (as expected for binary)
            float_predictions = cast(list[float], predictions)

            mean_prediction = sum(float_predictions) / len(float_predictions)
            rounded_mean = round(mean_prediction, 3)
            return rounded_mean
        elif isinstance(question, NumericQuestion):
            # Configurable aggregation for NumericQuestions using full 201-point CDFs.
            if not predictions:
                raise ValueError("Cannot aggregate empty list of predictions")

            numeric_predictions = cast(list[NumericDistribution], predictions)

            # Median aggregation can delegate directly to the existing NumericReport helper.
            if self.numeric_aggregation_method == "median":
                return await NumericReport.aggregate_predictions(numeric_predictions, question)

            if self.numeric_aggregation_method == "mean":
                # Replicate NumericReport.aggregate_predictions but with mean instead of median.
                cdfs = [prediction.cdf for prediction in numeric_predictions]

                # Ensure all CDFs share the same x-axis.
                x_axis = [percentile.value for percentile in cdfs[0]]
                for cdf in cdfs:
                    if any(p.value != x_axis[i] for i, p in enumerate(cdf)):
                        raise ValueError("X axis between CDFs is not the same")

                all_percentiles_of_cdf: list[list[float]] = [[p.percentile for p in cdf] for cdf in cdfs]

                mean_percentile_list: list[float] = np.mean(np.array(all_percentiles_of_cdf), axis=0).tolist()

                mean_cdf = [
                    Percentile(value=value, percentile=percentile)
                    for value, percentile in zip(x_axis, mean_percentile_list)
                ]

                return NumericDistribution(
                    declared_percentiles=mean_cdf,
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                    # DO NOT INFER UPPER AND LOWER BOUNDS FROM THE CDF! that will break b/c each model's CDF will not be aligned.
                    upper_bound=question.upper_bound,
                    lower_bound=question.lower_bound,
                    zero_point=question.zero_point,
                )

            raise ValueError(f"Invalid numeric aggregation method: {self.numeric_aggregation_method}")
        # Fallback to the parent class's aggregation logic for all other question types
        return await super()._aggregate_predictions(predictions, question)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = f"The outcome can not be higher than {question.upper_bound}."
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = f"The outcome can not be lower than {question.lower_bound}."
        return upper_bound_message, lower_bound_message

    def _log_raw_llm_output(self, llm_to_use: GeneralLlm, question_id: int, reasoning: str):
        logger.info(
            f"""
>>>>>>>>>>>>>>>>>> Raw LLM Output Start >>>>>>>>>>>>>>>>>
LLM: {llm_to_use.model}
Question ID: {question_id}

{reasoning}
<<<<<<<<<<<<<<<<<< Raw LLM Output End <<<<<<<<<<<<<<<<<
"""
        )


# ──────────────────────────────────────────────────────────────────────────
# Compact summary monkey-patch
# Replaces the verbose log_report_summary from forecasting_tools so we
# avoid printing duplicated research / rationale blobs in the console.
# The new version prints exactly one line per successful forecast plus any
# exceptions.


def _compact_log_report_summary(
    forecast_reports: Sequence[ForecastReport | BaseException],
) -> None:
    """Lightweight replacement for ForecastBot.log_report_summary."""
    valid_reports = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]

    def _line(r: ForecastReport) -> str:
        readable = type(r).make_readable_prediction(r.prediction).strip()
        return f"✅ {r.question.page_url} | Prediction: {readable} | " f"Minor Errors: {len(r.errors)}"

    summary_lines = "\n".join(_line(r) for r in valid_reports)

    for exc in exceptions:
        msg = str(exc)
        if len(msg) > 300:
            msg = msg[:297] + "…"
        summary_lines += f"\n❌ Exception: {exc.__class__.__name__} | {msg}"

    logger = logging.getLogger(__name__)
    logger.info(summary_lines + "\n")

    # replicate original behaviour: log aggregated minor errors, then raise on major
    minor_lists = [r.errors for r in valid_reports if r.errors]
    if minor_lists:
        logger.error(f"{len(minor_lists)} minor error groups occurred while forecasting: {minor_lists}")

    if exceptions:
        raise RuntimeError(f"{len(exceptions)} errors occurred while forecasting: {exceptions}")


# Apply the patch
ForecastBot.log_report_summary = staticmethod(_compact_log_report_summary)
# ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
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
        numeric_aggregation_method="mean",
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
                    provider={"quantizations": ["fp16", "bf16", "fp8"]},  # think this is working
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
            template_bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True)
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
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
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(question_url) for question_url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
