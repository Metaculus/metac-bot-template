"""
Benchmark treating the community prediction as approximate ground truth.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import random
import sys
import time
import weakref
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import typeguard
from dotenv import load_dotenv
from forecasting_tools import (
    ApiFilter,
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MonetaryCostManager,
    run_benchmark_streamlit_page,
)
from tqdm import tqdm

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    BENCHMARK_BATCH_SIZE,
    FETCH_PACING_SECONDS,
    FETCH_RETRY_BACKOFFS,
    HEARTBEAT_INTERVAL,
    TYPE_MIX,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import FORECASTER_LLMS, PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM
from metaculus_bot.scoring_patches import (
    apply_scoring_patches,
    log_score_scale_validation,
    log_scoring_path_stats,
    reset_scoring_path_stats,
)

logger = logging.getLogger(__name__)

load_dotenv()
load_dotenv(".env.local", override=True)


# Quick mitigation for occasional "Unclosed client session" warnings from aiohttp when
# using EXA search under high concurrency. Tracks sessions and closes them at process exit.
def _enable_aiohttp_session_autoclose() -> None:
    open_sessions: "weakref.WeakSet[aiohttp.ClientSession]" = weakref.WeakSet()
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self: aiohttp.ClientSession, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        open_sessions.add(self)

    aiohttp.ClientSession.__init__ = tracking_init  # type: ignore[assignment]

    def _close_open_sessions() -> None:
        to_close = [s for s in list(open_sessions) if not s.closed]
        if not to_close:
            return
        logger.debug(f"Closing {len(to_close)} lingering aiohttp sessions at exit")

        async def _close_all() -> None:
            for s in to_close:
                try:
                    await s.close()
                except Exception as e:  # pragma: no cover - best-effort cleanup
                    logger.debug(f"Error closing aiohttp session at exit: {e}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_close_all())
        else:
            try:
                asyncio.run(_close_all())
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_close_all())
                finally:
                    new_loop.close()

    atexit.register(_close_open_sessions)


_enable_aiohttp_session_autoclose()


# Global progress tracking state
_progress_state = {"total_predictions": 0, "start_time": 0, "completed_batches": 0, "total_batches": 0, "pbar": None}


def _install_benchmarker_heartbeat(interval_seconds: int = HEARTBEAT_INTERVAL) -> None:
    """Add a lightweight heartbeat to Benchmarker batch execution.

    Logs a progress line every ``interval_seconds`` while each batch is running,
    without changing the forecasting-tools package or internal flow.
    """
    try:
        original_run = Benchmarker._run_a_batch  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive: attribute should exist
        return

    # Avoid double-wrapping if re-imported
    if getattr(Benchmarker._run_a_batch, "_has_heartbeat", False):  # type: ignore[attr-defined]
        return

    async def _run_with_heartbeat(self, batch, _orig=original_run):  # type: ignore[no-untyped-def]
        start_time = datetime.now()
        task = asyncio.create_task(_orig(self, batch))
        try:
            while not task.done():
                await asyncio.sleep(interval_seconds)
                elapsed_min = (datetime.now() - start_time).total_seconds() / 60.0
                try:
                    # Update progress tracking
                    if _progress_state["pbar"] is not None:
                        _update_progress_estimate(batch)

                    logger.info(
                        f"[HB] {batch.benchmark.name} | {len(batch.questions)} questions | elapsed {elapsed_min:.1f}m"
                    )
                except Exception:
                    # Best-effort logging; do not interfere with main task
                    pass

            # Mark batch as completed
            _progress_state["completed_batches"] += 1
            if _progress_state["pbar"] is not None:
                _update_progress_final()

            return await task
        except Exception:
            # Bubble up original exceptions unchanged
            raise

    # Mark wrapper to prevent stacking
    setattr(_run_with_heartbeat, "_has_heartbeat", True)
    Benchmarker._run_a_batch = _run_with_heartbeat  # type: ignore[assignment]


def _update_progress_estimate(batch) -> None:
    """Update progress estimate during batch execution."""
    if _progress_state["total_predictions"] == 0:
        return

    # Estimate: each completed batch = batch_size questions per bot
    completed_predictions = _progress_state["completed_batches"] * len(batch.questions) * len(batch.forecast_bots)

    # Add partial progress for current batch (rough estimate)
    elapsed_total = time.time() - _progress_state["start_time"]
    if _progress_state["completed_batches"] > 0:
        avg_batch_time = elapsed_total / (_progress_state["completed_batches"] + 0.5)  # +0.5 for current partial
        progress_in_current = min(0.8, elapsed_total % avg_batch_time / avg_batch_time)  # Cap at 80% for current batch
        completed_predictions += int(progress_in_current * len(batch.questions) * len(batch.forecast_bots))

    completed_predictions = min(completed_predictions, _progress_state["total_predictions"])

    # Update progress bar
    pbar = _progress_state["pbar"]
    if pbar is not None:
        pbar.n = completed_predictions
        pbar.refresh()


def _update_progress_final() -> None:
    """Update progress when a batch completes."""
    if _progress_state["pbar"] is None or _progress_state["total_predictions"] == 0:
        return

    # Calculate exact completed predictions
    completed_predictions = min(
        _progress_state["completed_batches"]
        * (_progress_state["total_predictions"] // _progress_state["total_batches"]),
        _progress_state["total_predictions"],
    )

    pbar = _progress_state["pbar"]
    pbar.n = completed_predictions
    pbar.refresh()


_install_benchmarker_heartbeat(HEARTBEAT_INTERVAL)


async def _get_mixed_question_types(total_questions: int, one_year_from_now: datetime) -> list:
    """Get mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice).

    Reliability enhancements:
    - Add 2 retries with 5s then 15s backoff on transient fetch errors
    - Sleep 2s between type fetches to reduce burstiness
    - Fail fast with a clear error if an expected type cannot be fetched
    """

    # Calculate counts for each type (50/25/25 distribution)
    binary_count = int(total_questions * TYPE_MIX[0])
    numeric_count = int(total_questions * TYPE_MIX[1])
    mc_count = total_questions - binary_count - numeric_count  # Remainder goes to MC

    logger.info(f"Fetching mixed questions: {binary_count} binary, {numeric_count} numeric, {mc_count} multiple-choice")

    # Base filter settings for all question types
    base_filter_kwargs = {
        "allowed_statuses": ["open"],
        "num_forecasters_gte": 40,
        "scheduled_resolve_time_lt": one_year_from_now,
        "includes_bots_in_aggregates": False,
        "community_prediction_exists": True,
    }

    all_questions = []

    # Helper: fetch with retries and backoff
    async def _fetch_type_with_retries(question_type: str, count: int) -> list:
        import http.client

        from requests import exceptions as req_exc  # type: ignore
        from urllib3 import exceptions as ul3_exc  # type: ignore

        # Build filter per type
        filter_kwargs = base_filter_kwargs.copy()
        if question_type != "binary":
            filter_kwargs.pop("community_prediction_exists", None)
            logger.info(f"⚠️  Removed community_prediction_exists filter for {question_type} questions")
            sys.stdout.flush()

        # For numeric questions, include discrete types as well
        if question_type == "numeric":
            allowed_types = ["numeric", "discrete"]
        else:
            allowed_types = [question_type]

        api_filter = ApiFilter(allowed_types=allowed_types, **filter_kwargs)

        def _is_retryable_error(err: Exception) -> bool:
            retryables = (
                req_exc.ConnectionError,
                req_exc.Timeout,
                ul3_exc.ProtocolError,
                http.client.RemoteDisconnected,
            )
            if isinstance(err, retryables):
                return True
            # Best-effort string check for common transient statuses when wrapped
            msg = str(err).lower()
            return any(tok in msg for tok in ["429", "too many requests", "502", "503", "504", "timeout"])  # type: ignore[return-value]

        attempts = 0
        backoffs = list(FETCH_RETRY_BACKOFFS)  # seconds
        while True:
            try:
                logger.info(f"🔍 Attempt {attempts + 1}: fetching {count} {question_type} questions...")
                sys.stdout.flush()
                questions = await MetaculusApi.get_questions_matching_filter(
                    api_filter,
                    num_questions=count,
                    randomly_sample=True,
                )
                if not questions:
                    raise RuntimeError("API returned 0 questions")
                return questions
            except Exception as e:  # Retry on transient errors, otherwise raise
                if attempts < 2 and _is_retryable_error(e):
                    sleep_s = backoffs[attempts] if attempts < len(backoffs) else backoffs[-1]
                    logger.warning(
                        f"Retryable error fetching {question_type} questions (attempt {attempts + 1}/3): {e}. "
                        f"Backing off {sleep_s}s before retry."
                    )
                    sys.stdout.flush()
                    await asyncio.sleep(sleep_s)
                    attempts += 1
                    continue
                # Final failure or non-retryable
                logger.error(f"❌ Failed to fetch {question_type} questions: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                sys.stdout.flush()
                raise RuntimeError(
                    f"Aborting benchmark: unable to fetch {question_type} questions after {attempts + 1} attempts"
                ) from e

    # Fetch each question type separately with pacing and validation
    types_and_counts = [("binary", binary_count), ("numeric", numeric_count), ("multiple_choice", mc_count)]
    for i, (question_type, count) in enumerate(types_and_counts, 1):
        if count <= 0:
            continue
        logger.info(f"[{i}/3] Fetching {count} {question_type} questions...")
        sys.stdout.flush()
        questions = await _fetch_type_with_retries(question_type, count)
        logger.info(f"✅ Successfully fetched {len(questions)} {question_type} questions")
        if questions:
            logger.info(f"📋 Sample {question_type} question: {questions[0].question_text[:100]}...")
        all_questions.extend(questions)
        sys.stdout.flush()

        # Intentional pacing between types
        if i < len(types_and_counts):
            await asyncio.sleep(FETCH_PACING_SECONDS)

    # Shuffle to avoid clustering by type
    random.shuffle(all_questions)

    # Clear background_info for all questions (to test ability to find new information)
    for question in all_questions:
        question.background_info = None

    # Log final distribution
    type_counts = {}
    for q in all_questions:
        q_type = type(q).__name__
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

    logger.info(f"Final mixed question distribution: {type_counts}")
    return all_questions


async def benchmark_forecast_bot(mode: str, number_of_questions: int = 2, mixed_types: bool = False) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction.
    Ideally 100+ questions for meaningful error bars, but can use e.g. just a few for smoke testing or 30 for a quick run.
    """
    # TODO: make sure this is ok w/ the max predictions at once cost safety controls we have in place

    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        questions = MetaculusApi.get_benchmark_questions(number_of_questions)
    elif mode == "custom":
        # Below is an example of getting custom questions
        one_year_from_now = datetime.now() + timedelta(days=365)

        if mixed_types:
            # Get mixed question types with 50/25/25 distribution
            questions = await _get_mixed_question_types(number_of_questions, one_year_from_now)
        else:
            # Original binary-only approach
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
            question.background_info = None  # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Shared configuration for all benchmark bots
    BENCHMARK_BOT_CONFIG = {
        "research_reports_per_question": 1,
        "predictions_per_research_report": 1,  # Ignored when forecasters present
        "use_research_summary_to_forecast": False,
        "publish_reports_to_metaculus": False,  # Don't publish during benchmarking
        "folder_to_save_reports_to": None,
        "skip_previously_forecasted_questions": False,
        "research_provider": None,  # Use default provider selection
        "max_questions_per_run": None,  # No limit for benchmarking
        "is_benchmarking": True,  # Exclude prediction markets to avoid data leakage
        "allow_research_fallback": False,  # Ensure AskNews runs; do not fallback in benchmark
    }
    MODEL_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 16_000,  # Prevent truncation issues with reasoning models
        "stream": False,
        "timeout": 240,
        "allowed_tries": 3,
    }
    DEFAULT_HELPER_LLMS = {
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }

    # Apply scoring patches for mixed question types and reset counters
    apply_scoring_patches()
    reset_scoring_path_stats()

    with MonetaryCostManager() as cost_manager:
        # Keep benchmark and bot research concurrency aligned
        batch_size = BENCHMARK_BATCH_SIZE

        # Shared research cache for all bots to avoid duplicate API calls
        research_cache: dict[int, str] = {}

        # Define individual model configurations -- for sanity checking, can use these free models

        # Cheapies; avoid free models due to rate limits (very slow)
        r1_0528_model = GeneralLlm(
            model="openrouter/deepseek/deepseek-r1-0528",
            **MODEL_CONFIG,
        )
        ds_v3p1_model = GeneralLlm(
            model="openrouter/deepseek/deepseek-chat-v3.1",
            **MODEL_CONFIG,
        )
        kimi_k2_model = GeneralLlm(
            model="openrouter/moonshotai/kimi-k2-0905",
            **MODEL_CONFIG,
        )

        qwen3_model = GeneralLlm(
            model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
            **MODEL_CONFIG,
        )
        glm_model = GeneralLlm(
            model="openrouter/z-ai/glm-4.5",
            **MODEL_CONFIG,
        )
        claude_model = build_llm_with_openrouter_fallback(
            model="openrouter/anthropic/claude-sonnet-4",
            reasoning={"max_tokens": 4000},
            **MODEL_CONFIG,
        )
        gpt5_model = build_llm_with_openrouter_fallback(
            model="openrouter/openai/gpt-5",
            reasoning_effort="high",
            **MODEL_CONFIG,
        )
        gemini_model = GeneralLlm(
            model="openrouter/google/gemini-2.5-pro",
            reasoning={"max_tokens": 8000},
            **MODEL_CONFIG,
        )
        o3_model = build_llm_with_openrouter_fallback(
            model="openrouter/openai/o3",
            reasoning_effort="high",
            **MODEL_CONFIG,
        )
        grok_model = GeneralLlm(
            model="openrouter/x-ai/grok-4",
            reasoning={"effort": "high"},
            **MODEL_CONFIG,
        )

        # Individual model configurations for benchmarking
        # Test each model separately - ensembles will be generated post-hoc by analyze_correlations.py
        individual_models = [
            {"name": "qwen3-235b", "forecaster": qwen3_model},
            {"name": "deepseek-3.1", "forecaster": ds_v3p1_model},
            {"name": "kimi-k2", "forecaster": kimi_k2_model},
            # Additional models - comment for cost control during development:
            # {"name": "glm-4.5", "forecaster": glm_model},
            # {"name": "r1-0528", "forecaster": r1_0528_model},
            # {"name": "claude-sonnet-4", "forecaster": claude_model},
            # {"name": "gpt-5", "forecaster": gpt5_model},
            # {"name": "gemini-2.5-pro", "forecaster": gemini_model},
            # {"name": "o3", "forecaster": o3_model},
            # {"name": "grok-4", "forecaster": grok_model},
        ]

        # Stacking model configurations - these will aggregate predictions from ALL base models
        stacking_models = [
            {"name": "stack-qwen3", "stacker": qwen3_model},
            # Additional stackers - comment for cost control during development:
            # {"name": "stack-o3", "stacker": o3_model},
            # {"name": "stack-claude4", "stacker": claude_model},
            # {"name": "stack-gpt5", "stacker": gpt5_model},
        ]

        # Generate individual model bots - ensembles generated by CorrelationAnalyzer.find_optimal_ensembles()
        bots = []
        for model_config in individual_models:
            bot = TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                aggregation_strategy=AggregationStrategy.MEAN,  # Default, not used for single models
                llms={
                    "forecasters": [model_config["forecaster"]],
                    **DEFAULT_HELPER_LLMS,
                },
                max_concurrent_research=batch_size,
                research_cache=research_cache,
            )
            bot.name = model_config["name"]
            bots.append(bot)

        # Generate stacking bots - each gets ALL base model forecasters as input
        base_forecasters = [config["forecaster"] for config in individual_models]
        if len(base_forecasters) < 2:
            logger.warning(
                "STACKING configuration: fewer than 2 base forecasters (%d). Stacking quality may suffer.",
                len(base_forecasters),
            )
        stacking_bots: list[TemplateForecaster] = []
        for stacker_config in stacking_models:
            stacking_bot = TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                aggregation_strategy=AggregationStrategy.STACKING,
                llms={
                    "forecasters": base_forecasters,  # All base models
                    "stacker": stacker_config["stacker"],  # The stacking model
                    **DEFAULT_HELPER_LLMS,
                },
                max_concurrent_research=batch_size,
                research_cache=research_cache,
                stacking_fallback_on_failure=False,  # Fail in benchmarking
                stacking_randomize_order=True,  # Avoid position bias
            )
            stacking_bot.name = stacker_config["name"]
            # Benchmark-time validations and warnings
            try:
                if getattr(stacking_bot, "research_reports_per_question", 1) != 1:
                    logger.warning(
                        "STACKING benchmark: research_reports_per_question=%s; final results will average per-report stacked outputs by mean.",
                        getattr(stacking_bot, "research_reports_per_question", 1),
                    )
            except Exception:
                pass
            bots.append(stacking_bot)
            stacking_bots.append(stacking_bot)

        logger.info(
            f"Created {len(bots)} total bots for benchmarking: {len(individual_models)} individual models + {len(stacking_models)} stacking models. "
            f"Traditional ensembles will be generated post-hoc by correlation analysis."
        )
        bots = typeguard.check_type(bots, list[ForecastBot])

        # Log progress info
        total_predictions = len(bots) * len(questions)
        logger.info(
            f"🚀 Starting benchmark: {len(bots)} bots × {len(questions)} questions = {total_predictions} total predictions"
        )
        sys.stdout.flush()  # Ensure this critical message appears immediately

        # Initialize progress tracking
        _progress_state.update(
            {
                "total_predictions": total_predictions,
                "start_time": time.time(),
                "completed_batches": 0,
                "total_batches": len(bots),  # Each bot runs as a separate "batch"
                "pbar": tqdm(total=total_predictions, desc="Forecasting", unit="predictions"),
            }
        )

        logger.info("📊 Entering Benchmarker.run_benchmark() - this may take a while...")
        sys.stdout.flush()

        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=batch_size,
        ).run_benchmark()

        # Close progress bar
        if _progress_state["pbar"] is not None:
            _progress_state["pbar"].close()
            _progress_state["pbar"] = None

        logger.info("✅ Benchmarker.run_benchmark() completed, processing results...")
        sys.stdout.flush()
        try:
            for i, benchmark in enumerate(benchmarks):
                logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
                logger.info(
                    f"- Final Metaculus Baseline Score: {benchmark.average_expected_baseline_score:.4f} (based on log score, 0=always predict same, https://www.metaculus.com/help/scores-faq/#baseline-score )"
                )
                logger.info(f"- Total Cost: {benchmark.total_cost:.2f}")
                logger.info(f"- Time taken: {benchmark.time_taken_in_minutes:.4f}")
        except ValueError as ve:
            # Provide clearer guidance when no reports exist (likely research provider failures)
            raise RuntimeError(
                "Benchmark produced no forecast reports." "Fallback is disabled for benchmarks by design."
            ) from ve
        logger.info(f"Total Cost: {cost_manager.current_usage}")

        # Log score scale validation for mixed question types
        log_score_scale_validation(benchmarks)

        # Summarize scoring path usage and flag if fallbacks dominate
        log_scoring_path_stats()

        # TODO: refactor out this logic, jank to have here.
        # Perform correlation analysis if we have multiple models
        if len(benchmarks) > 1:
            from metaculus_bot.correlation_analysis import CorrelationAnalyzer

            analyzer = CorrelationAnalyzer()
            analyzer.add_benchmark_results(benchmarks)

            # Generate and log correlation report
            report = analyzer.generate_correlation_report("benchmarks/correlation_analysis.md")
            logger.info("\n" + "=" * 50)
            logger.info("CORRELATION ANALYSIS")
            logger.info("=" * 50)
            logger.info(report)

            # Generate all possible ensemble combinations with different aggregation strategies
            logger.info("\n" + "=" * 50)
            logger.info("ENSEMBLE GENERATION (Post-hoc)")
            logger.info("=" * 50)
            optimal_ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=6, max_cost_per_question=1.0)
            if optimal_ensembles:
                logger.info(
                    f"Generated {len(optimal_ensembles)} ensemble combinations from {len(benchmarks)} individual models"
                )
                logger.info(f"\nTop 10 Recommended Ensembles (Both Aggregation Strategies, Cost ≤ $1.0/question):")
                for i, ensemble in enumerate(optimal_ensembles[:10], 1):
                    models = " + ".join(ensemble.model_names)
                    logger.info(f"{i}. {models} ({ensemble.aggregation_strategy.upper()})")
                    logger.info(
                        f"   Score: {ensemble.avg_performance:.2f} | "
                        f"Cost: ${ensemble.avg_cost:.3f} | "
                        f"Diversity: {ensemble.diversity_score:.3f} | "
                        f"Overall: {ensemble.ensemble_score:.3f}"
                    )

                logger.info(
                    f"\n💡 Use 'python analyze_correlations.py benchmarks/' to explore all {len(optimal_ensembles)} ensemble combinations"
                )
            else:
                logger.info("No viable ensemble combinations found within cost constraints")
        else:
            logger.info("Skipping correlation analysis (need multiple models)")

        # Summarize any STACKING fallbacks encountered
        try:
            for sb in stacking_bots:
                count = getattr(sb, "_stacking_fallback_count", 0)
                if count:
                    logger.warning(
                        "STACKING fallback summary | bot=%s | fallbacks=%d (fell back to MEAN due to errors)",
                        getattr(sb, "name", "<unnamed>"),
                        count,
                    )
        except Exception:
            pass


if __name__ == "__main__":
    # Force unbuffered output for real-time logging in long-running processes
    import os

    os.environ["PYTHONUNBUFFERED"] = "1"

    # Create custom handler that properly flushes after each log
    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    console_handler = FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Enable forecasting-tools logging for progress visibility
    forecasting_logger = logging.getLogger("forecasting_tools")
    forecasting_logger.setLevel(logging.INFO)
    forecasting_logger.propagate = True

    # Enable our main modules
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)

    # Suppress noisy third-party loggers but keep errors visible
    for noisy_logger in ["LiteLLM", "httpx", "httpcore", "urllib3", "aiohttp"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
        logging.getLogger(noisy_logger).propagate = False

    # Enhanced logging monkey patch for all levels to ensure flushing
    for level_name in ["debug", "info", "warning", "error", "critical"]:
        original_method = getattr(logging.Logger, level_name)

        def make_flushing_method(orig_method):
            def flushing_method(self, message, *args, **kwargs):
                result = orig_method(self, message, *args, **kwargs)
                sys.stdout.flush()
                sys.stderr.flush()  # Also flush stderr for warnings/errors
                return result

            return flushing_method

        setattr(logging.Logger, level_name, make_flushing_method(original_method))

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark a list of bots")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "custom", "display"],
        default="display",
        help="Specify the run mode (default: display)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2,
        help="Number of questions to benchmark (default: 2)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice)",
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode, args.num_questions, args.mixed))
