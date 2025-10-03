from resolver.ingestion import acled_client


def _rows_to_map(rows):
    mapping = {}
    for row in rows:
        mapping[(row["iso3"], row["hazard_code"], row["metric"], row["as_of_date"])] = row
    return mapping


def test_acled_onset_detection(monkeypatch):
    monkeypatch.delenv("ACLED_PARSE_PARTICIPANTS", raising=False)
    records = []
    for month in range(1, 13):
        records.append(
            {
                "event_date": f"2022-{month:02d}-15",
                "event_type": "Battles",
                "country": "Ethiopia",
                "iso3": "ETH",
                "fatalities": "2",
                "notes": "",
            }
        )
    records.extend(
        [
            {
                "event_date": "2023-01-10",
                "event_type": "Battles",
                "country": "Ethiopia",
                "iso3": "ETH",
                "fatalities": "30",
                "notes": "",
            },
            {
                "event_date": "2023-02-15",
                "event_type": "Battles",
                "country": "Ethiopia",
                "iso3": "ETH",
                "fatalities": "40",
                "notes": "",
            },
        ]
    )

    config = acled_client.load_config()
    countries, shocks = acled_client.load_registries()
    rows = acled_client._build_rows(
        records,
        config,
        countries,
        shocks,
        "https://example.com/acled",
        "2023-03-01",
        "2023-03-02T00:00:00Z",
    )

    rows_map = _rows_to_map(rows)

    jan_onset = rows_map[("ETH", "ACO", "fatalities_battle_month", "2023-01")]
    assert "onset_rule_v1" in jan_onset["method"]
    assert "Prev12m" in jan_onset["definition_text"]
    assert jan_onset["value"] == 30

    jan_escalation = rows_map[("ETH", "ACE", "fatalities_battle_month", "2023-01")]
    assert "battle_fatalities=30" in jan_escalation["method"]
    assert "threshold=25" in jan_escalation["method"]

    feb_escalation = rows_map[("ETH", "ACE", "fatalities_battle_month", "2023-02")]
    assert "prev12m_battle_fatalities=52" in feb_escalation["method"]
    assert feb_escalation["value"] == 40
    assert "onset_rule_v1" not in feb_escalation["method"]
