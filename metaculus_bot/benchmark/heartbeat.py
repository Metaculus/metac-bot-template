import asyncio
import logging
from datetime import datetime

from forecasting_tools import Benchmarker

logger = logging.getLogger(__name__)


def install_benchmarker_heartbeat(interval_seconds: int, progress_state: dict) -> None:
    """Install a lightweight heartbeat on Benchmarker._run_a_batch.

    Updates the provided progress_state via local helpers during batch execution.
    Safe to call multiple times; only wraps once.
    """
    try:
        original_run = Benchmarker._run_a_batch  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return

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
                    if progress_state.get("pbar") is not None:
                        update_progress_estimate(batch, progress_state)
                    logger.info(
                        f"[HB] {batch.benchmark.name} | {len(batch.questions)} questions | elapsed {elapsed_min:.1f}m"
                    )
                except Exception:
                    pass

            # Mark batch as completed
            progress_state["completed_batches"] = progress_state.get("completed_batches", 0) + 1
            if progress_state.get("pbar") is not None:
                update_progress_final(progress_state)
            return await task
        except Exception:
            raise

    setattr(_run_with_heartbeat, "_has_heartbeat", True)
    Benchmarker._run_a_batch = _run_with_heartbeat  # type: ignore[assignment]


def update_progress_estimate(batch, state: dict) -> None:
    if not state.get("total_predictions"):
        return
    completed_batches = state.get("completed_batches", 0)
    completed_predictions = completed_batches * len(batch.questions) * len(batch.forecast_bots)
    elapsed_total = (state.get("_time_fn", __import__("time").time))() - state.get("start_time", 0)
    if completed_batches > 0:
        avg_batch_time = elapsed_total / (completed_batches + 0.5)
        progress_in_current = min(0.8, (elapsed_total % avg_batch_time) / avg_batch_time)
        completed_predictions += int(progress_in_current * len(batch.questions) * len(batch.forecast_bots))
    completed_predictions = min(completed_predictions, state.get("total_predictions", 0))
    pbar = state.get("pbar")
    if pbar is not None:
        pbar.n = completed_predictions
        pbar.refresh()


def update_progress_final(state: dict) -> None:
    pbar = state.get("pbar")
    total_preds = state.get("total_predictions", 0)
    total_batches = state.get("total_batches", 1) or 1
    if pbar is None or total_preds == 0:
        return
    completed_batches = state.get("completed_batches", 0)
    completed_predictions = min(completed_batches * (total_preds // total_batches), total_preds)
    pbar.n = completed_predictions
    pbar.refresh()
