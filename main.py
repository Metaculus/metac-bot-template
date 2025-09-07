import asyncio
import logging
import random
from typing import Any, Coroutine, Sequence, cast

from dotenv import load_dotenv
from forecasting_tools import (  # AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
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

from metaculus_bot import stacking as stacking
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    aggregate_binary_median,
    aggregate_multiple_choice_mean,
    aggregate_multiple_choice_median,
)
from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.bounds_clamping import (
    calculate_bounds_buffer,
    clamp_values_to_bounds,
    log_cluster_spreading_summary,
    log_corrections_summary,
    log_heavy_clamping_diagnostics,
)
from metaculus_bot.cluster_processing import (
    apply_cluster_spreading,
    apply_jitter_for_duplicates,
    compute_cluster_parameters,
    detect_count_like_pattern,
    ensure_strictly_increasing_bounded,
)
from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    DEFAULT_MAX_CONCURRENT_RESEARCH,
)
from metaculus_bot.mc_processing import build_mc_prediction
from metaculus_bot.numeric_config import (
    PCHIP_CDF_POINTS,
)
from metaculus_bot.numeric_diagnostics import log_final_prediction, log_pchip_fallback, validate_cdf_construction
from metaculus_bot.numeric_utils import (
    aggregate_binary_mean,
    aggregate_numeric,
    bound_messages,
)
from metaculus_bot.numeric_validation import (
    check_discrete_question_properties,
    detect_unit_mismatch,
    filter_to_standard_percentiles,
    sort_percentiles_by_value,
    validate_percentile_count_and_values,
)
from metaculus_bot.pchip_processing import (
    create_fallback_numeric_distribution,
    create_pchip_numeric_distribution,
    generate_pchip_cdf_with_smoothing,
    log_pchip_summary,
    reset_pchip_stats,
)
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from metaculus_bot.research_providers import ResearchCallable, choose_provider_with_name
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()
load_dotenv(".env.local", override=True)

# --- Numeric CDF smoothing constants centralized in metaculus_bot.constants ---


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
        stacking_fallback_on_failure: bool = True,
        stacking_randomize_order: bool = True,
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

        # Setup optional stacker LLM for stacking aggregation
        self._stacker_llm: GeneralLlm | None = None
        if "stacker" in normalized_llms:
            value = normalized_llms["stacker"]
            if isinstance(value, GeneralLlm):
                self._stacker_llm = value
            else:
                logger.warning("'stacker' key in llms must be a GeneralLlm object.")
            # Remove 'stacker' before delegating to base to avoid spurious warnings.
            normalized_llms.pop("stacker", None)

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
        self.stacking_fallback_on_failure: bool = stacking_fallback_on_failure
        self.stacking_randomize_order: bool = stacking_randomize_order
        # Per-question storage for stacker meta-analysis reasoning text
        self._stack_meta_reasoning: dict[int, str] = {}
        # Diagnostics counters for STACKING behavior
        self._stacking_guard_trigger_count: int = 0
        self._stacking_fallback_count: int = 0

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
        if self.aggregation_strategy == AggregationStrategy.STACKING:
            stacker_name = getattr(self._stacker_llm, "model", "<missing>") if self._stacker_llm else "<missing>"
            base_models = [getattr(m, "model", "<unknown>") for m in self._forecaster_llms]
            short_list = base_models if len(base_models) <= 6 else base_models[:6] + ["..."]
            logger.info(
                "STACKING config | stacker=%s | base_forecasters(%d)=%s | final_outputs_per_question=1",
                stacker_name,
                len(base_models),
                short_list,
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

        # Reset PCHIP statistics for this run
        reset_pchip_stats()

        results = await super().forecast_questions(questions, return_exceptions)

        # Log PCHIP summary at end of run
        log_pchip_summary()

        return results

    async def _run_stacking(
        self,
        question: MetaculusQuestion,
        research: str,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]],
    ) -> PredictionTypes:
        """Run stacking to aggregate multiple model predictions using a meta-model."""
        if self._stacker_llm is None:
            raise ValueError("No stacker LLM configured")

        # Strip model names from reasoning and prepare base predictions
        base_predictions = [stacking.strip_model_tag(pred.reasoning) for pred in reasoned_predictions]

        # Optionally randomize order to avoid position bias
        if self.stacking_randomize_order:
            combined = list(zip(base_predictions, reasoned_predictions))
            random.shuffle(combined)
            base_predictions, reasoned_predictions = zip(*combined)
            base_predictions = list(base_predictions)
            reasoned_predictions = list(reasoned_predictions)

        # Generate appropriate stacking call based on question type
        if isinstance(question, BinaryQuestion):
            value, meta_text = await stacking.run_stacking_binary(
                self._stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
            )
            self._log_llm_output(self._stacker_llm, question.id_of_question, meta_text)  # type: ignore
            self._stack_meta_reasoning[question.id_of_question] = meta_text
            logger.info(f"Stacked binary prediction for {getattr(question, 'page_url', '<unknown>')}: {value}")
            return value
        elif isinstance(question, MultipleChoiceQuestion):
            pol, meta_text = await stacking.run_stacking_mc(
                self._stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
            )
            self._log_llm_output(self._stacker_llm, question.id_of_question, meta_text)  # type: ignore
            self._stack_meta_reasoning[question.id_of_question] = meta_text
            logger.info(f"Stacked multiple choice prediction for {getattr(question, 'page_url', '<unknown>')}: {pol}")
            return pol
        elif isinstance(question, NumericQuestion):
            lower_msg, upper_msg = self._create_upper_and_lower_bound_messages(question)
            perc_list, meta_text = await stacking.run_stacking_numeric(
                self._stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
                lower_msg,
                upper_msg,
            )
            self._log_llm_output(self._stacker_llm, question.id_of_question, meta_text)  # type: ignore
            self._stack_meta_reasoning[question.id_of_question] = meta_text

            # Use same validation and processing logic as base numeric forecasting
            from metaculus_bot.numeric_validation import sort_percentiles_by_value, validate_percentile_count_and_values

            validate_percentile_count_and_values(perc_list)
            percentile_list = sort_percentiles_by_value(perc_list)

            # Unit mismatch guard (bail without posting if triggered)
            mismatch, reason = detect_unit_mismatch(percentile_list, question)  # type: ignore[arg-type]
            if mismatch:
                from metaculus_bot.exceptions import UnitMismatchError

                logger.error(
                    f"Unit mismatch likely for Q {getattr(question, 'id_of_question', 'N/A')} | "
                    f"URL {getattr(question, 'page_url', '<unknown>')} | reason={reason}. Withholding prediction."
                )
                raise UnitMismatchError(
                    f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
                )

            percentile_list = self._apply_jitter_and_clamp(percentile_list, question)

            from metaculus_bot.numeric_validation import check_discrete_question_properties

            is_discrete, should_force_zero_point_none = check_discrete_question_properties(question, PCHIP_CDF_POINTS)
            zero_point = getattr(question, "zero_point", None)
            if should_force_zero_point_none:
                zero_point = None

            try:
                pchip_cdf, smoothing_applied, aggressive_enforcement_used = generate_pchip_cdf_with_smoothing(
                    percentile_list, question, zero_point
                )
                prediction = create_pchip_numeric_distribution(pchip_cdf, percentile_list, question, zero_point)
            except Exception as e:
                log_pchip_fallback(question, e)
                prediction = create_fallback_numeric_distribution(percentile_list, question, zero_point)

            validate_cdf_construction(prediction, question)
            log_final_prediction(prediction, question)
            logger.info(f"Stacked numeric prediction for {getattr(question, 'page_url', '<unknown>')}")
            return prediction
        else:
            raise ValueError(f"Unsupported question type for stacking: {type(question)}")

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
        (
            valid_predictions,
            errors,
            exception_group,
        ) = await self._gather_results_and_exceptions(tasks)
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if len(valid_predictions) == 0:
            assert exception_group, "Exception group should not be None"
            self._reraise_exception_with_prepended_message(
                exception_group,
                "Error while running research and predictions",
            )
        # If using stacking, aggregate the predictions here
        if self.aggregation_strategy == AggregationStrategy.STACKING:
            try:
                if getattr(self, "research_reports_per_question", 1) != 1:
                    logger.warning(
                        "STACKING configured with research_reports_per_question=%s; final results will average per-report stacked outputs by mean.",
                        getattr(self, "research_reports_per_question", 1),
                    )
            except Exception:
                pass
            prediction_values = [pred.prediction_value for pred in valid_predictions]
            aggregated_value = await self._aggregate_predictions(
                prediction_values,
                question,
                research=research_to_use,
                reasoned_predictions=valid_predictions,
            )
            # Create a single aggregated prediction, preserving the stacker meta-analysis when available
            meta_text = self._stack_meta_reasoning.pop(
                question.id_of_question,
                "Stacked prediction aggregated from multiple models",
            )
            aggregated_prediction = ReasonedPrediction(prediction_value=aggregated_value, reasoning=meta_text)
            return ResearchWithPredictions(
                research_report=research,
                summary_report=summary_report,
                errors=errors,
                predictions=[aggregated_prediction],
            )
        else:
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

            def forecast_function(q, r, llm):
                return self._run_forecast_on_binary(q, r, llm)
        elif isinstance(question, MultipleChoiceQuestion):

            def forecast_function(q, r, llm):
                return self._run_forecast_on_multiple_choice(q, r, llm)
        elif isinstance(question, NumericQuestion):

            def forecast_function(q, r, llm):
                return self._run_forecast_on_numeric(q, r, llm)
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
        research: str | None = None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None = None,
    ) -> PredictionTypes:
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")

        # Guard for base aggregator calls when using STACKING.
        # If the base class calls aggregation after we've already stacked per research-report,
        # there will be no reasoned_predictions/research context provided here.
        # Only trigger guard when both reasoned_predictions and research are absent (base aggregator call)
        if (
            self.aggregation_strategy == AggregationStrategy.STACKING
            and reasoned_predictions is None
            and research is None
        ):
            try:
                self._stacking_guard_trigger_count += 1
            except Exception:
                pass
            # Single pre-stacked prediction – return as-is
            if len(predictions) == 1:
                logger.warning("STACKING guard: pre-stacked single prediction detected; returning as-is")
                return predictions[0]
            # Multiple research reports produced multiple stacked predictions – average by MEAN
            logger.warning(
                "STACKING guard: %d pre-stacked predictions; averaging by mean for final output",
                len(predictions),
            )
            first = predictions[0]
            if isinstance(first, (int, float)):
                values = [float(p) for p in predictions]  # type: ignore[list-item]
                result = aggregate_binary_mean(values)
                logger.warning("STACKING guard: binary mean of %s = %.3f", values, result)
                return result  # type: ignore[return-value]
            if isinstance(first, PredictedOptionList):
                mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
                aggregated = aggregate_multiple_choice_mean(mc_preds)
                summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
                logger.warning("STACKING guard: MC mean aggregation | %s", summary)
                return aggregated  # type: ignore[return-value]
            if isinstance(first, NumericDistribution) and isinstance(question, NumericQuestion):
                numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
                aggregated = await aggregate_numeric(numeric_preds, question, "mean")
                logger.warning(
                    "STACKING guard: numeric mean aggregation | CDF points=%d",
                    len(getattr(aggregated, "cdf", [])),
                )
                return aggregated  # type: ignore[return-value]
            raise ValueError(f"Unsupported prediction type for STACKING guard: {type(first)}")

        # Handle stacking strategy
        if self.aggregation_strategy == AggregationStrategy.STACKING:
            if self._stacker_llm is None:
                raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
            if reasoned_predictions is None:
                raise ValueError("STACKING aggregation strategy requires reasoned predictions")
            if research is None:
                raise ValueError("STACKING aggregation strategy requires research context")

            try:
                return await self._run_stacking(question, research, reasoned_predictions)
            except Exception as e:
                if self.stacking_fallback_on_failure:
                    # Increment diagnostics counter
                    try:
                        self._stacking_fallback_count += 1  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    logger.warning(f"Stacking failed ({type(e).__name__}: {e}), falling back to MEAN aggregation")
                    # Temporarily switch to MEAN for fallback
                    original_strategy = self.aggregation_strategy
                    self.aggregation_strategy = AggregationStrategy.MEAN
                    try:
                        result = await self._aggregate_predictions(predictions, question)
                        return result
                    finally:
                        self.aggregation_strategy = original_strategy
                else:
                    raise

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
                logger.info(
                    "Binary question ensembling: mean of %s = %.3f (rounded)",
                    float_preds,
                    result,
                )
                return result  # type: ignore[return-value]
            elif self.aggregation_strategy == AggregationStrategy.MEDIAN:
                result = aggregate_binary_median(float_preds)
                logger.info(
                    "Binary question ensembling: median of %s = %.3f",
                    float_preds,
                    result,
                )
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
        decimal_pred = max(
            BINARY_PROB_MIN,
            min(BINARY_PROB_MAX, binary_prediction.prediction_in_decimal),
        )

        logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = multiple_choice_prompt(question, research)
        reasoning = await llm_to_use.invoke(prompt)
        self._log_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore

        # Build parsing instructions (used in strict path and fallback)
        parsing_instructions = clean_indents(
            f"""
            Output a JSON array of objects with exactly these two keys per item: `option_name` (string) and `probability` (decimal in [0,1]).
            Use option names exactly from this list (case-insensitive match is OK, but prefer canonical spelling):
            {question.options}
            Do not include any options beyond this list. If the source text prefixes with words like 'Option A:' remove the prefix.
            Ensure the probabilities approximately sum to 1.0; slight floating-point drift is OK.
            """
        )

        # Try strict PredictedOptionList first for compatibility with existing tests
        try:
            predicted_option_list: PredictedOptionList = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions,
            )
            # Clamp and renormalize to avoid edge cases
            from metaculus_bot.numeric_utils import clamp_and_renormalize_mc

            try:
                predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
            except Exception:
                # Be tolerant in tests/mocks that don't return full shape
                pass
        except Exception:
            # Fallback tolerant parse: simple options then build final list
            raw_options: list[OptionProbability] = await structure_output(
                text_to_structure=reasoning,
                output_type=list[OptionProbability],
                model=self.get_llm("parser", "llm"),
                additional_instructions=parsing_instructions,
            )
            predicted_option_list = build_mc_prediction(raw_options, list(question.options))

        logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    # TODO: current monolithic numeric logic is disgusting and needs to be refactored
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = numeric_prompt(question, research, lower_bound_message, upper_bound_message)
        reasoning = await llm_to_use.invoke(prompt)

        self._log_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore

        unit_str = getattr(question, "unit_of_measure", None) or "base unit"
        parse_notes = (
            (
                "Return exactly these 8 percentiles and no others: 5,10,20,40,60,80,90,95. "
                "Do not include 0, 50, or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (float). "
                f"Values must be in the base unit '{unit_str}' and within [{{lower}}, {{upper}}]. "
                "If your text uses B/M/k, convert numerically to base unit (e.g., 350B → 350000000000). No suffixes."
            )
            .replace("{lower}", str(getattr(question, "lower_bound", 0)))
            .replace("{upper}", str(getattr(question, "upper_bound", 0)))
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parse_notes,
        )

        # Filter to the standard set in case the parser emitted extras

        percentile_list = filter_to_standard_percentiles(percentile_list)

        # Validate we have exactly 8 percentiles with the expected set {5,10,20,40,60,80,90,95}
        validate_percentile_count_and_values(percentile_list)

        # Sort percentiles by percentile value to ensure proper order
        percentile_list = sort_percentiles_by_value(percentile_list)

        # Apply jitter and clamp logic
        percentile_list = self._apply_jitter_and_clamp(percentile_list, question)

        # For discrete questions, force zero_point to None to avoid log-scaling
        is_discrete, should_force_zero_point_none = check_discrete_question_properties(question, PCHIP_CDF_POINTS)
        zero_point = getattr(question, "zero_point", None)

        if should_force_zero_point_none:
            zero_point = None

        # Generate PCHIP CDF with validation and smoothing
        try:
            pchip_cdf, smoothing_applied, aggressive_enforcement_used = generate_pchip_cdf_with_smoothing(
                percentile_list, question, zero_point
            )
            prediction = create_pchip_numeric_distribution(pchip_cdf, percentile_list, question, zero_point)
        except Exception as e:
            log_pchip_fallback(question, e)
            prediction = create_fallback_numeric_distribution(percentile_list, question, zero_point)

        # Validate CDF construction for non-PCHIP distributions
        validate_cdf_construction(prediction, question)

        # Unit mismatch guard (bail without posting if triggered) — run late to avoid disrupting
        # diagnostic tests that exercise fallback and CDF validation paths.
        mismatch, reason = detect_unit_mismatch(percentile_list, question)
        if mismatch:
            from metaculus_bot.exceptions import UnitMismatchError

            logger.error(
                f"Unit mismatch likely for Q {getattr(question, 'id_of_question', 'N/A')} | "
                f"URL {getattr(question, 'page_url', '<unknown>')} | reason={reason}. Withholding prediction."
            )
            raise UnitMismatchError(
                f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
            )

        # Log final prediction
        log_final_prediction(prediction, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _apply_jitter_and_clamp(self, percentile_list: list[Percentile], question: NumericQuestion) -> list[Percentile]:
        """Apply jitter to ensure strictly increasing percentiles and clamp values to bounds.

        Enhancements:
        - Detect clusters of near-equal declared values and spread them minimally and symmetrically.
        - Preserve cluster intent while ensuring strictly increasing sequence before CDF construction.
        """
        import logging

        logging.getLogger(__name__)

        # Calculate buffer for bounds clamping
        range_size = question.upper_bound - question.lower_bound
        buffer = calculate_bounds_buffer(question)

        # Extract values
        values = [p.value for p in percentile_list]
        modified_values = list(values)  # Copy for modifications

        # Detect count-like patterns (all near integers)
        count_like = detect_count_like_pattern(values)

        # Compute cluster detection and spreading parameters
        span = (max(values) - min(values)) if values else 0.0
        value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like, span)

        # Compute pre-spread min delta for logging
        pre_deltas = [b - a for a, b in zip(values, values[1:])]
        min(pre_deltas) if pre_deltas else float("inf")

        # Apply cluster spreading to ensure strictly increasing values
        modified_values, clusters_applied = apply_cluster_spreading(
            modified_values, question, value_eps, spread_delta, range_size
        )

        # Apply jitter for any remaining duplicates
        modified_values = apply_jitter_for_duplicates(modified_values, question, range_size, percentile_list)

        # Clamp values to bounds if they violate by small amounts
        modified_values, corrections_made = clamp_values_to_bounds(modified_values, percentile_list, question, buffer)

        # Log diagnostics and warnings
        log_cluster_spreading_summary(
            modified_values,
            values,
            question,
            clusters_applied,
            spread_delta,
            count_like,
        )
        log_corrections_summary(modified_values, values, question, corrections_made)
        log_heavy_clamping_diagnostics(modified_values, values, question, buffer)

        # Final pass to ensure strictly increasing values within bounds
        modified_values = ensure_strictly_increasing_bounded(modified_values, question, range_size)

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
