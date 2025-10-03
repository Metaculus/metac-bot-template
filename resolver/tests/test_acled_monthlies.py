import datetime as dt

from resolver.ingestion import acled_client


def _rows_to_map(rows):
    out = {}
    for row in rows:
        key = (row["iso3"], row["hazard_code"], row["metric"], row["as_of_date"])
        out[key] = row
    return out


def test_acled_monthly_aggregation(monkeypatch):
    monkeypatch.delenv("ACLED_PARSE_PARTICIPANTS", raising=False)
    records = [
        {
            "event_date": "2023-01-15",
            "event_type": "Battles",
            "country": "Kenya",
            "iso3": "KEN",
            "fatalities": "3",
            "notes": "",
        },
        {
            "event_date": "2023-01-20",
            "event_type": "Riots",
            "country": "Kenya",
            "iso3": "",
            "fatalities": "1",
            "notes": "Reports of protesters 200 strong in the streets",
        },
        {
            "event_date": "2023-01-22",
            "event_type": "Protests",
            "country": "Kenya",
            "iso3": "KEN",
            "fatalities": "0",
            "notes": "Approximately protesters totaling 1,500 joined",
        },
        {
            "event_date": "2023-02-10",
            "event_type": "Battles",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "30",
            "notes": "",
        },
        {
            "event_date": "2023-02-15",
            "event_type": "Battles",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "5",
            "notes": "",
        },
        {
            "event_date": "2023-02-18",
            "event_type": "Protests",
            "country": "Uganda",
            "iso3": "UGA",
            "fatalities": "0",
            "notes": "Participants numbered 200 downtown",
        },
    ]

    config = acled_client.load_config()
    config.setdefault("participants", {})["enabled"] = True
    countries, shocks = acled_client.load_registries()
    publication_date = "2023-03-01"
    ingested_at = "2023-03-02T00:00:00Z"

    rows = acled_client._build_rows(
        records,
        config,
        countries,
        shocks,
        "https://example.com/acled",
        publication_date,
        ingested_at,
    )
    assert rows, "expected rows from aggregation"

    rows_map = _rows_to_map(rows)

    kenya_conflict = rows_map[("KEN", "ACE", "fatalities_battle_month", "2023-01")]
    assert kenya_conflict["value"] == 3
    assert "Prev12m" in kenya_conflict["definition_text"]

    uganda_escalation = rows_map[("UGA", "ACE", "fatalities_battle_month", "2023-02")]
    assert uganda_escalation["value"] == 35
    assert "battle_fatalities=35" in uganda_escalation["method"]

    uganda_onset = rows_map[("UGA", "ACO", "fatalities_battle_month", "2023-02")]
    assert "onset_rule_v1" in uganda_onset["method"]
    assert uganda_onset["value"] == 35

    kenya_unrest = rows_map[("KEN", "CU", "events", "2023-01")]
    assert kenya_unrest["value"] == 2
    assert kenya_unrest["unit"] == "events"

    uganda_unrest = rows_map[("UGA", "CU", "events", "2023-02")]
    assert uganda_unrest["value"] == 1

    kenya_participants = rows_map[("KEN", "CU", "participants", "2023-01")]
    assert kenya_participants["value"] == 1700
    assert kenya_participants["unit"] == "persons"

    uganda_participants = rows_map[("UGA", "CU", "participants", "2023-02")]
    assert uganda_participants["value"] == 200
    assert uganda_participants["unit"] == "persons"
