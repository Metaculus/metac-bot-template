import asyncio
import logging
from typing import Any, Coroutine, Literal, Sequence, cast

import numpy as np
from dotenv import load_dotenv
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

from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    aggregate_binary_median,
    aggregate_multiple_choice_mean,
    aggregate_multiple_choice_median,
)
from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.constants import DEFAULT_MAX_CONCURRENT_RESEARCH
from metaculus_bot.numeric_utils import aggregate_binary_mean, aggregate_numeric, bound_messages
from metaculus_bot.research_providers import ResearchCallable, choose_provider_with_name
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()
load_dotenv(".env.local", override=True)

# --- Numeric CDF smoothing constants (tunable) ---
# Minimal threshold to detect nearly-equal declared values (relative to range)
VALUE_EPSILON_MULT = 1e-9
# Base minimal spread applied within a detected cluster (relative to range)
SPREAD_DELTA_MULT = 1e-6
# Target minimum adjacent probability step for numeric CDFs
MIN_PROB_STEP = 5.0e-5
# Factor to scale the injected ramp when smoothing probabilities
RAMP_K_FACTOR = 3.0


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
        aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN,
        research_provider: ResearchCallable | None = None,
        max_questions_per_run: int | None = 10,
        is_benchmarking: bool = False,
        max_concurrent_research: int = DEFAULT_MAX_CONCURRENT_RESEARCH,
        allow_research_fallback: bool = True,
        research_cache: dict[int, str] | None = None,
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

        if not isinstance(aggregation_strategy, AggregationStrategy):
            raise ValueError(f"aggregation_strategy must be an AggregationStrategy enum, got {aggregation_strategy}")
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy
        self._custom_research_provider: ResearchCallable | None = research_provider
        self.research_provider: ResearchCallable | None = research_provider  # For framework config access
        if max_questions_per_run is not None and max_questions_per_run <= 0:
            raise ValueError("max_questions_per_run must be a positive integer if provided")
        self.max_questions_per_run: int | None = max_questions_per_run
        self.is_benchmarking: bool = is_benchmarking
        self.allow_research_fallback: bool = allow_research_fallback
        self.research_cache: dict[int, str] | None = research_cache

        if max_concurrent_research <= 0:
            raise ValueError("max_concurrent_research must be a positive integer")
        # Persist for framework config introspection and logging
        self.max_concurrent_research: int = max_concurrent_research
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

        # Log ensemble + aggregation configuration once on init
        num_models = len(self._forecaster_llms) if self._forecaster_llms else 1
        logger.info(
            "Ensemble configured: %s model(s) | Aggregation: %s",
            num_models,
            self.aggregation_strategy.value,
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
        # Check cache first (only during benchmarking)
        if self.is_benchmarking and self.research_cache and question.id_of_question in self.research_cache:
            cache_key = question.id_of_question
            cached_research = self.research_cache[cache_key]
            logger.info(f"Using cached research for question {cache_key}")
            return cached_research

        async with self._concurrency_limiter:
            # Double-check cache in case another instance cached it while we were waiting
            if self.is_benchmarking and self.research_cache and question.id_of_question in self.research_cache:
                cache_key = question.id_of_question
                cached_research = self.research_cache[cache_key]
                logger.info(f"Using cached research for question {cache_key} (double-check)")
                return cached_research

            # Determine provider each call unless a custom one was supplied.
            if self._custom_research_provider is not None:
                provider = self._custom_research_provider
                provider_name = "custom"
            else:
                default_llm = self.get_llm("default", "llm") if hasattr(self, "get_llm") else None  # type: ignore[attr-defined]
                provider, provider_name = choose_provider_with_name(
                    default_llm,
                    exa_callback=self._call_exa_smart_searcher,
                    perplexity_callback=self._call_perplexity,
                    openrouter_callback=lambda q: self._call_perplexity(q, use_open_router=True),
                    is_benchmarking=self.is_benchmarking,
                )

            logger.info(f"Using research provider: {provider_name}")
            try:
                research = await provider(question.question_text)
            except Exception as e:
                # Optional fallback when primary provider (often AskNews) fails (e.g., 429s)
                if self.allow_research_fallback and provider_name == "asknews":
                    logger.warning(f"Primary research provider '{provider_name}' failed with {type(e).__name__}: {e}")
                    fallback_research: str | None = None
                    try:
                        import os

                        if os.getenv("OPENROUTER_API_KEY"):
                            logger.info("Falling back to openrouter/perplexity for research")
                            fallback_research = await self._call_perplexity(
                                question.question_text, use_open_router=True
                            )
                        elif os.getenv("PERPLEXITY_API_KEY"):
                            logger.info("Falling back to Perplexity for research")
                            fallback_research = await self._call_perplexity(
                                question.question_text, use_open_router=False
                            )
                        elif os.getenv("EXA_API_KEY") and hasattr(self, "_call_exa_smart_searcher"):
                            logger.info("Falling back to Exa search for research")
                            fallback_research = await self._call_exa_smart_searcher(question.question_text)
                    except Exception as fe:
                        logger.warning(f"Fallback research provider also failed: {type(fe).__name__}: {fe}")
                    if fallback_research is None:
                        raise
                    research = fallback_research
                else:
                    raise

            # Cache the result if we're in benchmarking mode
            if self.is_benchmarking and self.research_cache is not None:
                cache_key = question.id_of_question
                self.research_cache[cache_key] = research
                logger.info(f"Cached research for question {cache_key}")

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

        # High-level aggregation log for clarity
        qtype = (
            "binary"
            if isinstance(predictions[0], (int, float))
            else (
                "numeric"
                if isinstance(predictions[0], NumericDistribution)
                else (
                    "multiple-choice"
                    if isinstance(predictions[0], PredictedOptionList)
                    else type(predictions[0]).__name__
                )
            )
        )
        logger.info("Aggregating %s predictions with %s", qtype, self.aggregation_strategy.value)

        # Binary aggregation - strategy-based dispatch
        if isinstance(predictions[0], (int, float)):
            float_preds = [float(p) for p in predictions]  # type: ignore[list-item]

            if self.aggregation_strategy == AggregationStrategy.MEAN:
                result = aggregate_binary_mean(float_preds)
                logger.info("Binary question ensembling: mean of %s = %.3f (rounded)", float_preds, result)
                return result  # type: ignore[return-value]
            elif self.aggregation_strategy == AggregationStrategy.MEDIAN:
                result = aggregate_binary_median(float_preds)
                logger.info("Binary question ensembling: median of %s = %.3f", float_preds, result)
                return result  # type: ignore[return-value]
            else:
                raise ValueError(f"Unsupported aggregation strategy for binary questions: {self.aggregation_strategy}")

        # Numeric aggregation - convert strategy to string for existing function
        if isinstance(predictions[0], NumericDistribution) and isinstance(question, NumericQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            strategy_str = self.aggregation_strategy.value  # Convert enum to string
            aggregated = await aggregate_numeric(numeric_preds, question, strategy_str)
            lb = getattr(question, "lower_bound", None)
            ub = getattr(question, "upper_bound", None)
            logger.info(
                "Numeric aggregation=%s | preserved bounds [%s, %s] | CDF points=%d",
                strategy_str,
                lb,
                ub,
                len(getattr(aggregated, "cdf", [])),
            )
            return aggregated  # type: ignore[return-value]

        # Multiple choice aggregation - strategy-based dispatch (NO MORE super() delegation)
        if isinstance(predictions[0], PredictedOptionList):
            mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]

            if self.aggregation_strategy == AggregationStrategy.MEAN:
                aggregated = aggregate_multiple_choice_mean(mc_preds)
                summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
                logger.info("MC mean aggregation; renormalized to 1.0 | %s", summary)
                return aggregated  # type: ignore[return-value]
            elif self.aggregation_strategy == AggregationStrategy.MEDIAN:
                aggregated = aggregate_multiple_choice_median(mc_preds)
                summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
                logger.info("MC median aggregation; renormalized to 1.0 | %s", summary)
                return aggregated  # type: ignore[return-value]
            else:
                raise ValueError(
                    f"Unsupported aggregation strategy for multiple choice questions: {self.aggregation_strategy}"
                )

        # Fallback for unexpected prediction types
        raise ValueError(f"Unknown prediction type for aggregation: {type(predictions[0])}")

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
                "openrouter/perplexity/sonar-reasoning-pro"  # sonar-reasoning-pro would be slightly better but pricier
            )
        else:
            model_name = "perplexity/sonar-reasoning-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
            api_key=get_openrouter_api_key(model_name) if model_name.startswith("openrouter/") else None,
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
        # Provide strict parsing guidance so the parser returns a decimal in [0,1]
        binary_parse_instructions = (
            "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
            "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
            "Do not return percentages, strings, or any extra fields."
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            additional_instructions=binary_parse_instructions,
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

        # Apply custom clamping to 0.5%/99.5% for multiple choice questions
        for option in predicted_option_list.predicted_options:
            option.probability = max(0.005, min(0.995, option.probability))

        # Renormalize to ensure probabilities sum to 1 after clamping
        total_prob = sum(option.probability for option in predicted_option_list.predicted_options)
        if total_prob > 0:
            for option in predicted_option_list.predicted_options:
                option.probability /= total_prob

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

        # Validate we have exactly 8 percentiles with the expected set {5,10,20,40,60,80,90,95}
        expected_percentiles = {0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95}
        if len(percentile_list) != 8:
            raise ValidationError.from_exception_data(
                "NumericDistribution",
                [
                    {
                        "type": "value_error",
                        "loc": ("declared_percentiles",),
                        "input": percentile_list,
                        "ctx": {
                            "error": f"Expected 8 declared percentiles (5,10,20,40,60,80,90,95), got {len(percentile_list)}.",
                        },
                    }
                ],
            )

        # Validate percentile values are exactly the expected set (with tolerance for rounding)
        actual_percentiles = {round(p.percentile, 6) for p in percentile_list}
        expected_rounded = {round(p, 6) for p in expected_percentiles}
        if actual_percentiles != expected_rounded:
            raise ValidationError.from_exception_data(
                "NumericDistribution",
                [
                    {
                        "type": "value_error",
                        "loc": ("declared_percentiles",),
                        "input": percentile_list,
                        "ctx": {
                            "error": f"Expected percentile set {{5,10,20,40,60,80,90,95}}, got {sorted(p.percentile * 100 for p in percentile_list)}.",
                        },
                    }
                ],
            )

        # Sort percentiles by percentile value to ensure proper order
        percentile_list.sort(key=lambda p: p.percentile)

        # Apply jitter and clamp logic
        percentile_list = self._apply_jitter_and_clamp(percentile_list, question)

        # For discrete questions, force zero_point to None to avoid log-scaling
        cdf_size = getattr(question, "cdf_size", None)
        is_discrete = cdf_size is not None and cdf_size != 201
        zero_point = getattr(question, "zero_point", None)

        if is_discrete:
            if zero_point is not None:
                logger.debug(
                    f"Question {getattr(question, 'id_of_question', 'N/A')}: Forcing zero_point=None for discrete question"
                )
            zero_point = None
        elif zero_point is not None and zero_point == question.lower_bound:
            logger.warning(
                f"Question {getattr(question, 'id_of_question', 'N/A')}: zero_point ({zero_point}) is equal to lower_bound "
                f"({question.lower_bound}). Forcing linear scale for CDF generation."
            )
            zero_point = None

        # Phase 2: Use local PCHIP-based CDF construction for robust interpolation
        # - PCHIP (monotone cubic) interpolation for smoother distributions
        # - Applied min step increase (5e-5) and max jump cap (0.59) constraints
        # - Handles log-space transforms when all values are positive (continuous only)
        # - Enforces closed/open bound constraints directly in CDF construction

        from metaculus_bot.pchip_cdf import generate_pchip_cdf, percentiles_to_pchip_format

        try:
            # Convert percentiles to PCHIP input format
            pchip_percentiles = percentiles_to_pchip_format(percentile_list)

            # Generate robust CDF using PCHIP interpolation
            pchip_cdf = generate_pchip_cdf(
                percentile_values=pchip_percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=zero_point,
                min_step=5.0e-5,
                num_points=201,
            )

            # Optional probability-side ramp smoothing to enforce minimum step
            smoothing_applied = False
            try:
                diffs_before = np.diff(pchip_cdf)
                min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0
                if min_delta_before < MIN_PROB_STEP:
                    ramp = np.linspace(0.0, MIN_PROB_STEP * RAMP_K_FACTOR, len(pchip_cdf))
                    pchip_cdf = np.maximum.accumulate(np.array(pchip_cdf) + ramp).tolist()
                    smoothing_applied = True

                    # Re-pin endpoints to respect open/closed bounds semantics
                    if not question.open_lower_bound:
                        pchip_cdf[0] = 0.0
                    else:
                        pchip_cdf[0] = max(pchip_cdf[0], 0.001)
                    if not question.open_upper_bound:
                        pchip_cdf[-1] = 1.0
                    else:
                        pchip_cdf[-1] = min(pchip_cdf[-1], 0.999)

                    diffs_after = np.diff(pchip_cdf)
                    min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
                    logger.warning(
                        "CDF ramp smoothing for Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f | k_factor=%.1f",
                        getattr(question, "id_of_question", None),
                        getattr(question, "page_url", None),
                        min_delta_before,
                        min_delta_after,
                        RAMP_K_FACTOR,
                    )
            except Exception as smooth_e:
                logger.error("Ramp smoothing skipped due to error: %s", smooth_e)

            # Validate PCHIP CDF meets Metaculus requirements
            if len(pchip_cdf) != 201:
                raise ValueError(f"PCHIP CDF has {len(pchip_cdf)} points, expected 201")

            if not all(0.0 <= p <= 1.0 for p in pchip_cdf):
                raise ValueError(
                    f"PCHIP CDF contains invalid probabilities outside [0,1]: {[p for p in pchip_cdf if not (0.0 <= p <= 1.0)]}"
                )

            if not all(a <= b for a, b in zip(pchip_cdf[:-1], pchip_cdf[1:])):
                raise ValueError("PCHIP CDF is not monotonic")

            # Check minimum step requirement (same as forecasting-tools)
            min_step = np.min(np.diff(pchip_cdf))
            if min_step < 5e-5 - 1e-10:
                raise ValueError(f"PCHIP CDF violates minimum step requirement: {min_step:.8f} < 5e-5")

            # Check maximum step requirement
            max_step = np.max(np.diff(pchip_cdf))
            if max_step > 0.59 + 1e-6:
                raise ValueError(f"PCHIP CDF violates maximum step requirement: {max_step:.8f} > 0.59")

            # Check boundary conditions
            if not question.open_lower_bound and abs(pchip_cdf[0]) > 1e-6:
                raise ValueError(f"PCHIP CDF closed lower bound violation: {pchip_cdf[0]} != 0.0")

            if not question.open_upper_bound and abs(pchip_cdf[-1] - 1.0) > 1e-6:
                raise ValueError(f"PCHIP CDF closed upper bound violation: {pchip_cdf[-1]} != 1.0")

            if question.open_lower_bound and pchip_cdf[0] < 0.001:
                raise ValueError(f"PCHIP CDF open lower bound violation: {pchip_cdf[0]} < 0.001")

            if question.open_upper_bound and pchip_cdf[-1] > 0.999:
                raise ValueError(f"PCHIP CDF open upper bound violation: {pchip_cdf[-1]} > 0.999")

            # Emit INFO to make success clearly visible in smoke logs
            logger.info(
                "PCHIP OK for Q %s | points=%d | min_step=%.8f | max_step=%.8f | smoothing=%s | open_bounds=(%s,%s)",
                getattr(question, "id_of_question", "N/A"),
                len(pchip_cdf),
                min_step,
                max_step,
                smoothing_applied,
                question.open_lower_bound,
                question.open_upper_bound,
            )

            # Create custom NumericDistribution subclass that uses our PCHIP CDF
            class PchipNumericDistribution(NumericDistribution):
                def __init__(self, pchip_cdf_values, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._pchip_cdf_values = pchip_cdf_values

                @property
                def cdf(self) -> list[Percentile]:
                    """Return PCHIP-generated CDF as Percentile objects."""
                    # Create the value axis (201 points from lower to upper bound)
                    x_vals = np.linspace(self.lower_bound, self.upper_bound, len(self._pchip_cdf_values))

                    # Create Percentile objects with correct mapping:
                    # _pchip_cdf_values contains the probability values (0-1)
                    # x_vals contains the corresponding question values
                    return [
                        Percentile(percentile=prob_val, value=question_val)
                        for question_val, prob_val in zip(x_vals, self._pchip_cdf_values)
                    ]

            prediction = PchipNumericDistribution(
                pchip_cdf_values=pchip_cdf,
                declared_percentiles=percentile_list,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=zero_point,
                cdf_size=getattr(question, "cdf_size", None),
            )

        except Exception as e:
            logger.warning(
                f"Question {getattr(question, 'id_of_question', 'N/A')}: PCHIP CDF construction failed ({str(e)}), "
                "falling back to forecasting-tools default"
            )
            # Fallback to original forecasting-tools approach
            prediction = NumericDistribution(
                declared_percentiles=percentile_list,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=zero_point,
                cdf_size=getattr(question, "cdf_size", None),
            )

        # Proactively compute CDF to surface spacing issues and log rich diagnostics
        # Skip CDF validation for PCHIP distributions since they enforce constraints internally
        try:
            if hasattr(prediction, "_pchip_cdf_values"):
                logger.debug(
                    f"Question {getattr(question, 'id_of_question', 'N/A')}: Skipping CDF validation for PCHIP distribution"
                )
            else:
                _ = prediction.cdf  # force CDF construction
        except (AssertionError, ZeroDivisionError) as e:
            try:
                declared = getattr(prediction, "declared_percentiles", [])
                bounds = {
                    "lower_bound": getattr(question, "lower_bound", None),
                    "upper_bound": getattr(question, "upper_bound", None),
                    "open_lower_bound": getattr(question, "open_lower_bound", None),
                    "open_upper_bound": getattr(question, "open_upper_bound", None),
                    "zero_point": getattr(question, "zero_point", None),
                    "cdf_size": getattr(question, "cdf_size", None),
                }
                vals = [float(p.value) for p in declared]
                prcs = [float(p.percentile) for p in declared]
                deltas_val = [b - a for a, b in zip(vals, vals[1:])]
                deltas_pct = [b - a for a, b in zip(prcs, prcs[1:])]
                logger.error(
                    "Numeric CDF spacing assertion for Q %s | URL %s | error=%s\n"
                    "Bounds=%s\n"
                    "Declared percentiles (p%% -> v): %s\n"
                    "Value deltas: %s | Percentile deltas: %s",
                    getattr(question, "id_of_question", None),
                    getattr(question, "page_url", None),
                    e,
                    bounds,
                    [(p, v) for p, v in zip(prcs, vals)],
                    deltas_val,
                    deltas_pct,
                )
            except Exception as log_e:
                logger.error("Failed logging numeric CDF diagnostics: %s", log_e)
            raise
        # Validation of 8 percentiles is now done earlier, before creating NumericDistribution

        logger.info(f"Forecasted URL {question.page_url} as {prediction.declared_percentiles}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _apply_jitter_and_clamp(self, percentile_list: list[Percentile], question: NumericQuestion) -> list[Percentile]:
        """Apply jitter to ensure strictly increasing percentiles and clamp values to bounds.

        Enhancements:
        - Detect clusters of near-equal declared values and spread them minimally and symmetrically.
        - Preserve cluster intent while ensuring strictly increasing sequence before CDF construction.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Calculate buffer for bounds clamping
        range_size = question.upper_bound - question.lower_bound
        buffer = 1.0 if range_size > 100 else range_size * 0.01

        # Extract values
        values = [p.value for p in percentile_list]
        modified_values = list(values)  # Copy for modifications

        # Detect count-like patterns (all near integers)
        count_like = False
        try:
            if values:
                count_like = all(abs(v - round(v)) <= 1e-6 for v in values)
        except Exception:
            count_like = False

        # Detect and spread clusters of near-equal values
        value_eps = max(range_size * VALUE_EPSILON_MULT, 1e-12)
        base_delta = max(range_size * SPREAD_DELTA_MULT, 1e-12)
        spread_delta = max(base_delta, 1.0 if count_like else base_delta)

        # Compute pre-spread min delta for logging
        pre_deltas = [b - a for a, b in zip(values, values[1:])]
        min_value_delta_before = min(pre_deltas) if pre_deltas else float("inf")

        clusters_applied = 0
        i = 0
        while i < len(modified_values) - 1:
            j = i
            # Grow cluster while adjacent values within epsilon
            while j + 1 < len(modified_values) and abs(modified_values[j + 1] - modified_values[j]) <= value_eps:
                j += 1
            if j > i:
                # We have a cluster from i..j inclusive
                clusters_applied += 1
                k = j - i + 1
                # Base center value: mean of the cluster
                center = float(np.mean(modified_values[i : j + 1]))
                # Offsets: symmetric around center
                # Example for k=3: -d, 0, +d; for k=4: -1.5d, -0.5d, +0.5d, +1.5d
                offsets = [((idx - (k - 1) / 2.0) * spread_delta) for idx in range(k)]
                new_vals = [center + off for off in offsets]
                # Enforce bounds softly during spread to avoid later large clamps
                tiny = max(1e-9 * range_size, 1e-12)
                if not question.open_lower_bound:
                    new_vals = [max(v, question.lower_bound + tiny) for v in new_vals]
                if not question.open_upper_bound:
                    new_vals = [min(v, question.upper_bound - tiny) for v in new_vals]
                # Apply while preserving non-decreasing relation to neighbors
                # If previous value exists and is >= first new, shift all up minimally
                if i - 1 >= 0 and new_vals[0] <= modified_values[i - 1]:
                    shift = (modified_values[i - 1] + max(1e-12, value_eps)) - new_vals[0]
                    new_vals = [v + shift for v in new_vals]
                # If next value exists and last new exceeds it, compress offsets
                if j + 1 < len(modified_values) and new_vals[-1] >= modified_values[j + 1]:
                    # Compress spread to fit in available gap
                    available = max(modified_values[j + 1] - (new_vals[0]), max(value_eps, 1e-12))
                    if k > 1:
                        step = available / k
                        new_vals = [new_vals[0] + step * idx for idx in range(k)]
                # Assign
                for t in range(k):
                    modified_values[i + t] = new_vals[t]
                i = j + 1
            else:
                i += 1

        # Apply jitter for duplicates and ensure increasing without breaching bounds
        for i in range(1, len(modified_values)):
            if modified_values[i] <= modified_values[i - 1]:
                epsilon = max(1e-9 * range_size, 1e-12)
                target = modified_values[i - 1] + epsilon
                if not question.open_upper_bound:
                    target = min(target, question.upper_bound - epsilon)
                # Increase if possible; otherwise allow equality (PCHIP will handle de-dup)
                new_val = max(modified_values[i], target)
                # Also respect lower bound on closed lower
                if not question.open_lower_bound:
                    new_val = max(new_val, question.lower_bound + epsilon)
                modified_values[i] = new_val
                logger.debug(
                    f"Applied jitter: percentile {percentile_list[i].percentile} value {values[i]} -> {modified_values[i]}"
                )

        # Clamp values to bounds if they violate by small amounts
        corrections_made = False
        for i in range(len(modified_values)):
            original_value = modified_values[i]

            # Check lower bound violation
            if not question.open_lower_bound and modified_values[i] < question.lower_bound:
                if question.lower_bound - modified_values[i] <= buffer:
                    modified_values[i] = question.lower_bound + buffer
                    corrections_made = True
                    logger.debug(
                        f"Clamped lower: percentile {percentile_list[i].percentile} value {original_value} -> {modified_values[i]}"
                    )
                else:
                    raise ValueError(
                        f"Value {original_value} too far below lower bound {question.lower_bound} (tolerance: {buffer})"
                    )

            # Check upper bound violation
            if not question.open_upper_bound and modified_values[i] > question.upper_bound:
                if modified_values[i] - question.upper_bound <= buffer:
                    modified_values[i] = question.upper_bound - buffer
                    corrections_made = True
                    logger.debug(
                        f"Clamped upper: percentile {percentile_list[i].percentile} value {original_value} -> {modified_values[i]}"
                    )
                else:
                    raise ValueError(
                        f"Value {original_value} too far above upper bound {question.upper_bound} (tolerance: {buffer})"
                    )

        # Log warnings/diagnostics
        post_deltas = [b - a for a, b in zip(modified_values, modified_values[1:])]
        min_value_delta_after = min(post_deltas) if post_deltas else float("inf")

        if clusters_applied > 0:
            logger.warning(
                "Cluster spread applied for Q %s | URL %s | clusters=%d | delta_used=%.6g | min_value_delta_before=%.6g | min_value_delta_after=%.6g | count_like=%s",
                getattr(question, "id_of_question", None),
                getattr(question, "page_url", None),
                clusters_applied,
                spread_delta,
                min_value_delta_before,
                min_value_delta_after,
                count_like,
            )

        if corrections_made or any(v != orig for v, orig in zip(modified_values, values)):
            logger.warning(f"Corrected numeric distribution for question {getattr(question, 'id_of_question', 'N/A')}")

        # Optional heavy bound clamping diagnostic
        if len(values) > 0:
            tol = 1e-9
            clamped_lower = sum(
                1 for v in modified_values if not question.open_lower_bound and v <= question.lower_bound + buffer + tol
            )
            clamped_upper = sum(
                1 for v in modified_values if not question.open_upper_bound and v >= question.upper_bound - buffer - tol
            )
            if clamped_lower / len(values) > 0.5 or clamped_upper / len(values) > 0.5:
                logger.warning(
                    "Heavy bound clamping for Q %s | URL %s | clamped_to_lower=%d%% | clamped_to_upper=%d%% | bounds=[%s, %s]",
                    getattr(question, "id_of_question", None),
                    getattr(question, "page_url", None),
                    int(100 * clamped_lower / len(values)),
                    int(100 * clamped_upper / len(values)),
                    question.lower_bound,
                    question.upper_bound,
                )

        # Re-ensure increasing after clamping, bounded (left-to-right)
        for i in range(1, len(modified_values)):
            if modified_values[i] <= modified_values[i - 1]:
                epsilon = max(1e-9 * range_size, 1e-12)
                target = modified_values[i - 1] + epsilon
                if not question.open_upper_bound:
                    target = min(target, question.upper_bound - epsilon)
                if not question.open_lower_bound:
                    target = max(target, question.lower_bound + epsilon)
                modified_values[i] = max(modified_values[i], target)

        # Additional pass (right-to-left) to make room near closed upper bound
        # If upper bound is closed and strict increase is capped, slide earlier values down by epsilon
        epsilon = max(1e-9 * range_size, 1e-12)
        for i in range(len(modified_values) - 2, -1, -1):
            if modified_values[i] >= modified_values[i + 1]:
                target = modified_values[i + 1] - epsilon
                if not question.open_lower_bound:
                    target = max(target, question.lower_bound + epsilon)
                modified_values[i] = min(modified_values[i], target)

        # Create new percentile list with corrected values
        return [Percentile(value=v, percentile=p.percentile) for v, p in zip(modified_values, percentile_list)]

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
