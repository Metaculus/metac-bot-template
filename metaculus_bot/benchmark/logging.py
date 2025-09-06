from __future__ import annotations

import logging
from typing import Sequence

from forecasting_tools import ForecastBot

logger = logging.getLogger(__name__)


def log_bot_lineup(bots: Sequence[ForecastBot]) -> None:
    total_bots = len(bots)
    logger.info("📋 Bot lineup (%d total):", total_bots)
    for idx, b in enumerate(bots, 1):
        try:
            strat = getattr(b, "aggregation_strategy", None)
            strat_val = strat.value if strat else "(framework default)"
            r = getattr(b, "research_reports_per_question", "?")
            p = getattr(b, "predictions_per_research_report", "?")
            if strat_val == "stacking":
                stacker = getattr(getattr(b, "_stacker_llm", None), "model", "<missing>")
                base_f = getattr(b, "_forecaster_llms", [])
                base_names = [getattr(m, "model", "<unknown>") for m in base_f]
                short = base_names if len(base_names) <= 6 else base_names[:6] + ["..."]
                logger.info(
                    "- Bot %d/%d | name=%s | strategy=STACKING | R×P=%s×%s | stacker=%s | base_forecasters(%d)=%s | final_outputs_per_q=1",
                    idx,
                    total_bots,
                    getattr(b, "name", f"bot-{idx}"),
                    r,
                    p,
                    stacker,
                    len(base_names),
                    short,
                )
            else:
                logger.info(
                    "- Bot %d/%d | name=%s | strategy=%s | R×P=%s×%s",
                    idx,
                    total_bots,
                    getattr(b, "name", f"bot-{idx}"),
                    strat_val,
                    r,
                    p,
                )
        except Exception as be:
            logger.warning("Failed to log bot %d overview: %s", idx, be)


def log_benchmarker_headline_note() -> None:
    logger.info(
        "Note: 'Class | R × P | Model' means research_reports_per_question × predictions_per_research_report; Model is the default (stacker for STACKING)."
    )


def log_stacking_summaries(stacking_bots: Sequence[object]) -> None:
    for sb in stacking_bots:
        count = getattr(sb, "_stacking_fallback_count", 0)
        guard_count = getattr(sb, "_stacking_guard_trigger_count", 0)
        if count:
            logger.warning(
                "STACKING fallback summary | bot=%s | fallbacks=%d (fell back to MEAN due to errors)",
                getattr(sb, "name", "<unnamed>"),
                count,
            )
        if guard_count:
            logger.warning(
                "STACKING guard summary | bot=%s | guard_triggers=%d (base-aggregator combine across research reports)",
                getattr(sb, "name", "<unnamed>"),
                guard_count,
            )
