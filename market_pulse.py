"""Which Market Pulse questions to forecast on this run.

Market Pulse is spot scored: "only your predictions standing at the close
time of each question matter" (tournament page). Forecasting each question
once, when first seen -- what every other tournament here does -- would leave
a forecast that can be weeks old standing at the one instant that is scored.
Re-forecasting every open question on every run would fix that and multiply
LLM spend by the ~6 runs a day GitHub actually delivers.

So a question is forecast when either:

* we have never forecast it, or
* it closes within ``REFRESH_WINDOW`` and our latest forecast is older than
  ``MIN_AGE`` -- which, at the observed run cadence, refreshes each question
  about once or twice in its last day and a half.

Standard library only, so the selection is testable without the SDK; the
questions only need ``already_forecasted``, ``close_time`` and
``timestamp_of_my_last_forecast``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

logger = logging.getLogger(__name__)

REFRESH_WINDOW = timedelta(hours=36)
MIN_AGE = timedelta(hours=18)


def _utc(moment: datetime | None) -> datetime | None:
    """The SDK parses API dates with pendulum (aware, UTC); a naive value is
    taken as UTC rather than letting the subtraction below raise."""
    if moment is not None and moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _last_forecast(question: Any) -> datetime | None:
    try:
        return _utc(question.timestamp_of_my_last_forecast)
    except Exception:  # noqa: BLE001 - an unreadable timestamp means "unknown"
        return None


def select(questions: Sequence[Any], now: datetime) -> list[Any]:
    """The questions worth an LLM call now, new ones first."""
    now = _utc(now)
    new: list[Any] = []
    refresh: list[Any] = []
    for question in questions:
        if not question.already_forecasted:
            new.append(question)
            continue
        close = _utc(question.close_time)
        last = _last_forecast(question)
        if close is None or last is None:
            continue
        if close - now <= REFRESH_WINDOW and now - last >= MIN_AGE:
            refresh.append(question)
    logger.info(
        "market_pulse_selection open=%d new=%d refresh=%d",
        len(questions), len(new), len(refresh),
    )
    return new + refresh


#: The Metaculus Cup is scored against the human crowd over each question's
#: life, so a forecast left untouched for weeks loses ground as news arrives
#: (Metaculus flagged 9 of the bot's Cup forecasts "might be stale" on
#: 2026-10-16). A week is the refresh age: the Cup opens a batch weekly and
#: pays bots nothing, so more often is spend without a prize behind it.
CUP_MAX_AGE = timedelta(days=7)


def select_stale(
    questions: Sequence[Any], now: datetime, max_age: timedelta = CUP_MAX_AGE
) -> list[Any]:
    """New questions, plus those whose latest forecast is ``max_age`` old.

    Unlike :func:`select` this ignores close time: every open question is
    refreshed on the same cadence. Unknown forecast times are left alone.
    """
    now = _utc(now)
    new: list[Any] = []
    stale: list[Any] = []
    for question in questions:
        if not question.already_forecasted:
            new.append(question)
            continue
        last = _last_forecast(question)
        if last is not None and now - last >= max_age:
            stale.append(question)
    logger.info(
        "stale_selection open=%d new=%d stale=%d max_age_days=%.1f",
        len(questions), len(new), len(stale), max_age.total_seconds() / 86400,
    )
    return new + stale
