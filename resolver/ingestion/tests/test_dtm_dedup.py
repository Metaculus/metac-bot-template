from __future__ import annotations

from typing import Iterable

from resolver.ingestion import dtm_client


def _apply_admin_dedup(records: Iterable[dict[str, str | int]]) -> dict[str, str | int]:
    """Replicate the admin-level dedup logic for a single key."""

    key = ("NGA", "Borno", "2025-09-01", "dtm_source")
    dedup: dict[tuple[str, str, str, str], dict[str, str | int]] = {}
    for record in records:
        existing = dedup.get(key)
        if existing and not dtm_client._is_candidate_newer(existing["as_of"], record["as_of"]):
            continue
        dedup[key] = record
    return dedup[key]


def test_dtm_dedup_prefers_newer_and_skips_equal() -> None:
    """Newer `as_of` wins; equal timestamps keep the first record."""

    def make_record(as_of: str, value: int) -> dict[str, str | int]:
        return {
            "source": "dtm",
            "country_iso3": "NGA",
            "admin1": "Borno",
            "event_id": f"NGA-displacement-202509-{as_of}",
            "as_of": as_of,
            "month_start": "2025-09-01",
            "value_type": "new_displaced",
            "value": value,
            "unit": "people",
            "method": "dtm_flow",
            "confidence": "unknown",
            "raw_event_id": "dtm_source::Borno::202509",
            "raw_fields_json": "{}",
        }

    # Candidate with a newer as_of should replace the older record.
    newer_result = _apply_admin_dedup(
        [make_record("2025-10-01", 100), make_record("2025-10-03", 500)]
    )
    assert newer_result["as_of"] == "2025-10-03"
    assert newer_result["value"] == 500

    # A stale candidate arriving after a newer record must be ignored.
    stale_result = _apply_admin_dedup(
        [make_record("2025-10-03", 500), make_record("2025-10-01", 100)]
    )
    assert stale_result["as_of"] == "2025-10-03"
    assert stale_result["value"] == 500

    # Equal timestamps should keep the first record (stable ordering, no flapping).
    equal_result = _apply_admin_dedup(
        [make_record("2025-10-01", 100), make_record("2025-10-01", 999)]
    )
    assert equal_result["as_of"] == "2025-10-01"
    assert equal_result["value"] == 100
