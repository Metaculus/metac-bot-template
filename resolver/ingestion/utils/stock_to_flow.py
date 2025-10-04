"""Convert stock series into non-negative monthly flows."""

from __future__ import annotations

from datetime import date
from typing import Mapping

from .month_bucket import month_start

Number = float | int


def flow_from_stock(series_by_month: Mapping[date | str, Number]) -> dict[date, Number]:
    """Return month-over-month non-negative differences for a stock series."""

    normalised: dict[date, float] = {}
    for key, value in series_by_month.items():
        bucket = month_start(key)
        if bucket is None:
            continue
        try:
            normalised[bucket] = float(value)
        except (TypeError, ValueError):
            continue
    flows: dict[date, float] = {}
    previous_value = 0.0
    for month in sorted(normalised):
        current = normalised[month]
        diff = current - previous_value
        flows[month] = max(0.0, diff)
        previous_value = current
    return flows
