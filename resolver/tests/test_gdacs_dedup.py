from datetime import date

from resolver.ingestion.gdacs_client import (
    GDACSEvent,
    allocate_event,
    dedupe_monthly_rows,
    hazard_from_key,
    _aggregate_final_rows,
)


def _event(event_id: str, episode: str, total: int) -> GDACSEvent:
    hazard = hazard_from_key("tropical_cyclone")
    return GDACSEvent(
        event_id=event_id,
        episode_id=episode,
        iso3="PHL",
        hazard=hazard,
        start_date=date(2025, 1, 5),
        end_date=date(2025, 1, 5),
        impact_value=total,
        source_url=f"https://example.com/{event_id}",
        doc_title=f"Alert {event_id}",
        publication_date=date(2025, 1, 6),
        impact_field="population",
    )


def _build_rows(events):
    rows = []
    for event in events:
        allocations = allocate_event(event, "start")
        for month, value in allocations.items():
            rows.append(
                {
                    "event_ref": event.event_id,
                    "episode_id": event.episode_id,
                    "iso3": event.iso3,
                    "hazard_code": event.hazard.code,
                    "hazard_label": event.hazard.label,
                    "hazard_class": event.hazard.hazard_class,
                    "as_of_date": month,
                    "value": value,
                    "source_url": event.source_url,
                    "doc_title": event.doc_title,
                    "publication_date": event.publication_date,
                    "impact_field": event.impact_field,
                }
            )
    return rows


def test_gdacs_dedup_strategy_max():
    events = [_event("E1", "EP1", 10000), _event("E1", "EP1", 12000)]
    rows = _build_rows(events)
    deduped = dedupe_monthly_rows(rows, "max")
    assert list(deduped.values())[0]["value"] == 12000


def test_gdacs_dedup_strategy_sum_and_rollup():
    events = [
        _event("E1", "EP1", 10000),
        _event("E1", "EP1", 12000),
        _event("E2", "", 5000),
    ]
    rows = _build_rows(events)
    deduped_max = dedupe_monthly_rows(rows, "max")
    final_max = _aggregate_final_rows(
        deduped_max.values(),
        {"PHL": "Philippines"},
        "start",
        "max",
        "GDACS",
        "other",
    )
    assert len(final_max) == 1
    assert final_max[0]["value"] == 17000

    deduped_sum = dedupe_monthly_rows(rows, "sum")
    final_sum = _aggregate_final_rows(
        deduped_sum.values(),
        {"PHL": "Philippines"},
        "start",
        "sum",
        "GDACS",
        "other",
    )
    assert final_sum[0]["value"] == 27000
