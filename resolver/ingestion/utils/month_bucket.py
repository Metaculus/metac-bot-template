"""Date helpers for month bucket calculations."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Optional

_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y-%m",
    "%Y%m%d",
    "%Y%m",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_date(value: object) -> Optional[date]:
    """Parse ``value`` into a :class:`date` where possible."""

    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.date()


def month_start(value: object) -> Optional[date]:
    """Return the first day of the month for ``value``."""

    parsed = parse_date(value)
    if not parsed:
        return None
    return parsed.replace(day=1)


def ym_first(text: str | date | datetime) -> Optional[date]:
    """Parse ``text`` as a ``YYYY-MM`` or ``YYYY-MM-01`` style string."""

    if isinstance(text, (date, datetime)):
        return month_start(text)
    stripped = str(text or "").strip()
    if not stripped:
        return None
    if len(stripped) == 7 and stripped[4] == "-":
        stripped = f"{stripped}-01"
    return month_start(stripped)


def month_range(start: date, end: date) -> Iterable[date]:
    """Yield the month starts between ``start`` and ``end`` inclusive."""

    current = start.replace(day=1)
    end_month = end.replace(day=1)
    while current <= end_month:
        yield current
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = date(year, month, 1)


def days_in_month_segment(start: date, end: date) -> dict[date, int]:
    """Return inclusive day counts per month between ``start`` and ``end``."""

    if end < start:
        end = start
    allocations: dict[date, int] = {}
    cursor = start
    while cursor <= end:
        bucket = cursor.replace(day=1)
        next_month = (bucket.replace(day=28) + timedelta(days=4)).replace(day=1)
        segment_end = min(end, next_month - timedelta(days=1))
        allocations[bucket] = allocations.get(bucket, 0) + (segment_end - cursor).days + 1
        cursor = segment_end + timedelta(days=1)
    return allocations
