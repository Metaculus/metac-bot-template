from resolver.ingestion import dtm_client


def _make_record(iso3: str, hazard: str, metric: str, month: str, value, *, series_type="incident", source_id="src", **extra):
    record = {
        "iso3": iso3,
        "hazard_code": hazard,
        "metric": metric,
        "as_of_date": month,
        "value": value,
        "series_type": series_type,
        "source_id": source_id,
    }
    record.update(extra)
    return record


def test_incident_series_passthrough():
    rows = [
        _make_record("AAA", "DI", "in_need", "2025-01", 100, series_type=dtm_client.SERIES_INCIDENT),
        _make_record("AAA", "DI", "in_need", "2025-02", 120, series_type=dtm_client.SERIES_INCIDENT),
        _make_record("AAA", "DI", "in_need", "2025-03", 90, series_type=dtm_client.SERIES_INCIDENT),
    ]
    out = dtm_client.compute_monthly_deltas(rows, allow_first_month=False)
    assert [int(r["value"]) for r in out] == [100, 120, 90]
    assert all(r["series_type"] == dtm_client.SERIES_INCIDENT for r in out)


def test_cumulative_deltas_drop_first_month():
    rows = [
        _make_record("BBB", "DI", "in_need", "2025-01", 100, series_type=dtm_client.SERIES_CUMULATIVE),
        _make_record("BBB", "DI", "in_need", "2025-02", 150, series_type=dtm_client.SERIES_CUMULATIVE),
        _make_record("BBB", "DI", "in_need", "2025-03", 200, series_type=dtm_client.SERIES_CUMULATIVE),
    ]
    out = dtm_client.compute_monthly_deltas(rows, allow_first_month=False)
    assert [int(r["value"]) for r in out] == [50, 50]


def test_cumulative_plateaus_clip_to_zero():
    rows = [
        _make_record("CCC", "DI", "in_need", "2025-01", 100, series_type=dtm_client.SERIES_CUMULATIVE),
        _make_record("CCC", "DI", "in_need", "2025-02", 105, series_type=dtm_client.SERIES_CUMULATIVE),
        _make_record("CCC", "DI", "in_need", "2025-03", 90, series_type=dtm_client.SERIES_CUMULATIVE),
        _make_record("CCC", "DI", "in_need", "2025-04", 120, series_type=dtm_client.SERIES_CUMULATIVE),
    ]
    out = dtm_client.compute_monthly_deltas(rows, allow_first_month=False)
    assert [int(r["value"]) for r in out] == [5, 0, 30]


def test_subnational_rollup_before_delta():
    rows = [
        _make_record("DDD", "DI", "in_need", "2025-01", 100, series_type=dtm_client.SERIES_CUMULATIVE, admin1="State A"),
        _make_record("DDD", "DI", "in_need", "2025-01", 50, series_type=dtm_client.SERIES_CUMULATIVE, admin1="State B"),
        _make_record("DDD", "DI", "in_need", "2025-02", 150, series_type=dtm_client.SERIES_CUMULATIVE, admin1="State A"),
        _make_record("DDD", "DI", "in_need", "2025-02", 90, series_type=dtm_client.SERIES_CUMULATIVE, admin1="State B"),
    ]
    rolled = dtm_client.rollup_subnational(rows)

    jan = [r for r in rolled if r["as_of_date"] == "2025-01"]
    feb = [r for r in rolled if r["as_of_date"] == "2025-02"]
    assert len(jan) == 1 and int(jan[0]["value"]) == 150
    assert len(feb) == 1 and int(feb[0]["value"]) == 240

    out = dtm_client.compute_monthly_deltas(rolled, allow_first_month=False)
    assert [int(r["value"]) for r in out] == [90]
    assert all("admin1" not in r for r in out)
