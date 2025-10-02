from resolver.ingestion import acled_client


def _rows_to_map(rows):
    mapping = {}
    for row in rows:
        mapping[(row["iso3"], row["metric"], row["as_of_date"])] = row
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
    jan_row = rows_map[("ETH", "fatalities", "2023-01")]
    assert jan_row["hazard_code"] == "ACO"
    assert "onset_prev12m=24" in jan_row["method"]
    assert "prev12m=24" in jan_row["definition_text"]

    feb_row = rows_map[("ETH", "fatalities", "2023-02")]
    assert feb_row["hazard_code"] == "ACE"
    assert "onset_prev12m=52" in feb_row["method"]
    assert "current_month=40" in feb_row["definition_text"]
