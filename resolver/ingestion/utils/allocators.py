"""Allocation helpers used by monthly connectors."""

from __future__ import annotations

from datetime import date
from typing import Iterable, List, Tuple

from .month_bucket import days_in_month_segment, month_start


Number = float | int


def linear_split(total: Number | None, start: date, end: date | None) -> List[Tuple[date, Number]]:
    """Split ``total`` across months between ``start`` and ``end``."""

    if total is None:
        return []
    try:
        total_value = float(total)
    except (TypeError, ValueError):
        return []
    if total_value <= 0:
        return []
    if not isinstance(start, date):
        start = month_start(start)
    if start is None:
        return []
    if end is None:
        end = start
    if not isinstance(end, date):
        end = month_start(end)
    if end is None:
        end = start
    if end < start:
        end = start
    allocations = days_in_month_segment(start, end)
    if not allocations:
        return [(start.replace(day=1), total)]
    total_days = sum(allocations.values())
    if total_days <= 0:
        return [(start.replace(day=1), total)]
    is_integer = float(total_value).is_integer()
    results: list[tuple[date, Number]] = []
    if is_integer:
        integer_total = int(round(total_value))
        base_values: dict[date, int] = {}
        remainders: list[tuple[float, date]] = []
        assigned = 0
        for bucket, days in sorted(allocations.items()):
            share = (total_value * days) / total_days
            base = int(share)
            base_values[bucket] = base
            assigned += base
            remainders.append((share - base, bucket))
        remaining = integer_total - assigned
        for _, bucket in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            base_values[bucket] += 1
            remaining -= 1
        if remaining != 0:
            last_bucket = max(base_values)
            base_values[last_bucket] += remaining
        results = sorted((bucket, value) for bucket, value in base_values.items() if value)
    else:
        for bucket, days in sorted(allocations.items()):
            share = (total_value * days) / total_days
            results.append((bucket, share))
    return results
