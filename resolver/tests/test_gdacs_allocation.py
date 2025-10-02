from datetime import date

from resolver.ingestion.gdacs_client import allocate_event, allocate_value, GDACSEvent, hazard_from_key


def _make_event(total: int) -> GDACSEvent:
    hazard = hazard_from_key("tropical_cyclone")
    return GDACSEvent(
        event_id="E1",
        episode_id="EP1",
        iso3="PHL",
        hazard=hazard,
        start_date=date(2025, 1, 20),
        end_date=date(2025, 2, 5),
        impact_value=total,
        source_url="https://example.com/e1",
        doc_title="Cyclone Alert",
        publication_date=date(2025, 1, 21),
        impact_field="population",
    )


def test_gdacs_allocate_prorata():
    allocations = allocate_value(12000, date(2025, 1, 20), date(2025, 2, 5), "prorata")
    assert allocations == {"2025-01": 8471, "2025-02": 3529}
    assert sum(allocations.values()) == 12000


def test_gdacs_allocate_start_month():
    event = _make_event(12000)
    allocations = allocate_event(event, "start")
    assert allocations == {"2025-01": 12000}
