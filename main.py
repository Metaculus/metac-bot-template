import asyncio
import logging
from typing import Any, Coroutine, Literal, Sequence, cast

from forecasting_tools import (  # AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import ForecastReport, ResearchWithPredictions
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion
from pydantic import ValidationError

from metaculus_bot.numeric_utils import aggregate_binary_mean, aggregate_numeric, bound_messages
from metaculus_bot.research_providers import ResearchCallable, choose_provider
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TemplateForecaster(CompactLoggingForecastBot):

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        numeric_aggregation_method: Literal["mean", "median"] = "mean",
        research_provider: ResearchCallable | None = None,
        max_questions_per_run: int | None = 10,
        is_benchmarking: bool = False,
        max_concurrent_research: int = 3,
    ) -> None:
        # Validate and normalize llm configs BEFORE calling super().__init__ to avoid defaulting/warnings.
        if llms is None:
            raise ValueError("Either 'forecasters' or a 'default' LLM must be provided.")

        normalized_llms: dict[str, str | GeneralLlm | list[GeneralLlm]] = dict(llms)

        # Setup optional forecasters list; if valid, override default and count.
        self._forecaster_llms: list[GeneralLlm] = []
        if "forecasters" in normalized_llms:
            value = normalized_llms["forecasters"]
            if isinstance(value, list) and all(isinstance(x, GeneralLlm) for x in value):
                if value:
                    self._forecaster_llms = list(value)
                    # Ensure default points at first forecaster
                    normalized_llms["default"] = self._forecaster_llms[0]
                    predictions_per_research_report = len(self._forecaster_llms)
            else:
                logger.warning("'forecasters' key in llms must be a list of GeneralLlm objects.")
            # Remove 'forecasters' before delegating to base to avoid spurious warnings.
            normalized_llms.pop("forecasters", None)

        # Fail fast if critical LLM purposes are missing. We require parser and researcher explicitly.
        required_keys = {"default", "parser", "researcher", "summarizer"}
        missing = sorted(k for k in required_keys if k not in normalized_llms)
        if missing:
            raise ValueError(
                f"Missing required LLM purposes: {', '.join(missing)}. Provide these in the 'llms' config."
            )

        if numeric_aggregation_method not in ("mean", "median"):
            raise ValueError("numeric_aggregation_method must be 'mean' or 'median'")
        self.numeric_aggregation_method: Literal["mean", "median"] = numeric_aggregation_method
        self._custom_research_provider: ResearchCallable | None = research_provider
        self.research_provider: ResearchCallable | None = research_provider  # For framework config access
        if max_questions_per_run is not None and max_questions_per_run <= 0:
            raise ValueError("max_questions_per_run must be a positive integer if provided")
        self.max_questions_per_run: int | None = max_questions_per_run
        self.is_benchmarking: bool = is_benchmarking

        if max_concurrent_research <= 0:
            raise ValueError("max_concurrent_research must be a positive integer")
        # Instance-level semaphore to avoid cross-instance throttling
        self._concurrency_limiter: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_research)

        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=normalized_llms,  # type: ignore[arg-type]
        )

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        # Apply skip filter first (mirrors base class behavior) so we cap unforecasted items
        if self.skip_previously_forecasted_questions:
            unforecasted_questions = [q for q in questions if not q.already_forecasted]
            if len(questions) != len(unforecasted_questions):
                logger.info(f"Skipping {len(questions) - len(unforecasted_questions)} previously forecasted questions")
            questions = unforecasted_questions

        # Enforce max questions per run safety cap
        if self.max_questions_per_run is not None and len(questions) > self.max_questions_per_run:
            logger.info(f"Limiting to first {self.max_questions_per_run} questions out of {len(questions)}")
            questions = list(questions)[: self.max_questions_per_run]

        # Log question processing info with progress
        if questions:
            bot_name = getattr(self, "name", "Bot")
            logger.info(f"📊 {bot_name}: Processing {len(questions)} questions...")

        return await super().forecast_questions(questions, return_exceptions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Determine provider each call unless a custom one was supplied.
            if self._custom_research_provider is not None:
                provider = self._custom_research_provider
            else:
                default_llm = self.get_llm("default", "llm") if hasattr(self, "get_llm") else None  # type: ignore[attr-defined]
                provider = choose_provider(
                    default_llm,
                    exa_callback=self._call_exa_smart_searcher,
                    perplexity_callback=self._call_perplexity,
                    openrouter_callback=lambda q: self._call_perplexity(q, use_open_router=True),
                    is_benchmarking=self.is_benchmarking,
                )

            research = await provider(question.question_text)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    # Override _research_and_make_predictions to support multiple LLMs
    async def _research_and_make_predictions(
        self,
        question: MetaculusQuestion,
    ) -> ResearchWithPredictions[PredictionTypes]:
        # Call the parent class's method if no specific forecaster LLMs are provided
        if not self._forecaster_llms:
            return await super()._research_and_make_predictions(question)

        notepad = await self._get_notepad(question)
        if not hasattr(notepad, "total_research_reports_attempted"):
            raise AttributeError("Notepad is missing expected attribute 'total_research_reports_attempted'")
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)

        # Only call summarizer if we plan to use the summary for forecasting
        if self.use_research_summary_to_forecast:
            summary_report = await self.summarize_research(question, research)
            research_to_use = summary_report
        else:
            summary_report = research  # Use raw research for reporting compatibility
            research_to_use = research

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
        if not hasattr(notepad, "total_predictions_attempted"):
            raise AttributeError("Notepad is missing expected attribute 'total_predictions_attempted'")
        notepad.total_predictions_attempted += 1

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

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")

        # Binary aggregation (floats)
        if isinstance(predictions[0], (int, float)):
            float_preds = [float(p) for p in predictions]  # type: ignore[list-item]
            return aggregate_binary_mean(float_preds)  # type: ignore[return-value]

        # Numeric aggregation (configurable)
        if isinstance(predictions[0], NumericDistribution) and isinstance(question, NumericQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            return await aggregate_numeric(numeric_preds, question, self.numeric_aggregation_method)  # type: ignore[return-value]

        # Delegate remaining types (e.g., multiple-choice) to default
        return await super()._aggregate_predictions(predictions, question)

    async def _call_perplexity(self, question: str, use_open_router: bool = True) -> str:
        # Exclude prediction markets research when benchmarking to avoid data leakage
        prediction_markets_instruction = (
            ""
            if self.is_benchmarking
            else "In addition to news, briefly research prediction markets that are relevant to the question. (If there are no relevant prediction markets, simply skip reporting on this and DO NOT speculate what they would say.)"
        )

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            {prediction_markets_instruction}
            You DO NOT produce forecasts yourself; you must provide ALL relevant data to the superforecaster so they can make an expert judgment.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = (
                "openrouter/perplexity/sonar-reasoning"  # sonar-reasoning-pro would be slightly better but pricier
            )
        else:
            model_name = "perplexity/sonar-reasoning"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
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
        from metaculus_bot.prompts import binary_prompt

        prompt = binary_prompt(question, research)
        reasoning = await llm_to_use.invoke(prompt)
        self._log_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        from metaculus_bot.prompts import multiple_choice_prompt

        prompt = multiple_choice_prompt(question, research)
        reasoning = await llm_to_use.invoke(prompt)
        self._log_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        from metaculus_bot.prompts import numeric_prompt

        prompt = numeric_prompt(question, research, lower_bound_message, upper_bound_message)
        reasoning = await llm_to_use.invoke(prompt)

        self._log_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore

        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
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

        logger.info(f"Forecasted URL {question.page_url} as {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        return bound_messages(question)

    def _log_llm_output(self, llm_to_use: GeneralLlm, question_id: int | None, reasoning: str) -> None:
        try:
            model_name = getattr(llm_to_use, "model", "<unknown-model>")
        except Exception:
            model_name = "<unknown-model>"

        # Log formatted raw output at info level
        logger.info(
            f"""
\n\n
========================================
LLM OUTPUT | Model: {model_name} | Question: {question_id} | Length: {len(reasoning)} chars
========================================

{reasoning}

========================================
END LLM OUTPUT | {model_name}
========================================
\n\n
"""
        )


if __name__ == "__main__":
    from metaculus_bot.cli import main as cli_main

    cli_main()
