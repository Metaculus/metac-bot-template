import pandas as pd

from resolver.ingestion import hdx_client


def test_extract_series_hxl_monthly_pin():
    df = pd.DataFrame(
        [
            {"Month": "#month", "Year": "#year", "PIN": "#inneed"},
            {"Month": "January", "Year": "2024", "PIN": "1,000"},
            {"Month": "February", "Year": "2024", "PIN": "2000"},
        ]
    )

    parsed = hdx_client.extract_metric_timeseries(df, allow_annual=False)
    assert parsed is not None
    metric, series = parsed
    assert metric == "in_need"
    assert [row.as_of_date for row in series] == ["2024-01", "2024-02"]
    assert [row.value for row in series] == [1000, 2000]


def test_extract_series_people_affected_fallback():
    df = pd.DataFrame(
        [
            {"Report Date": "2023-04-01", "People Affected": "500"},
            {"Report Date": "2023-05-01", "People Affected": "600"},
        ]
    )

    parsed = hdx_client.extract_metric_timeseries(df, allow_annual=False)
    assert parsed is not None
    metric, series = parsed
    assert metric == "affected"
    assert [row.as_of_date for row in series] == ["2023-04", "2023-05"]
    assert [row.value for row in series] == [500, 600]


def test_extract_series_prefers_total_row():
    df = pd.DataFrame(
        [
            {"Location": "Region A", "Period": "2024-03-01", "People in Need": "1,000"},
            {"Location": "Region B", "Period": "2024-03-01", "People in Need": "1,400"},
            {"Location": "National Total", "Period": "2024-03-01", "People in Need": "2,600"},
        ]
    )

    parsed = hdx_client.extract_metric_timeseries(df, allow_annual=False)
    assert parsed is not None
    metric, series = parsed
    assert metric == "in_need"
    assert len(series) == 1
    assert series[0].as_of_date == "2024-03"
    assert series[0].value == 2600
    assert series[0].is_total is True


def test_extract_series_annual_fallback():
    df = pd.DataFrame(
        [
            {"Year": "#year", "PIN": "#inneed"},
            {"Year": "2021", "PIN": "1000"},
        ]
    )

    assert hdx_client.extract_metric_timeseries(df, allow_annual=False) is None

    parsed = hdx_client.extract_metric_timeseries(df, allow_annual=True)
    assert parsed is not None
    metric, series = parsed
    assert metric == "in_need"
    assert [row.as_of_date for row in series] == ["2021"]
    assert [row.value for row in series] == [1000]


def test_infer_hazard_keywords_multi_default():
    _, shocks = hdx_client.load_registries()
    cfg = hdx_client.load_cfg()["shock_keywords"]

    hazard = hdx_client.infer_hazard(["Seasonal flood situation update"], shocks, cfg)
    assert hazard.code == "FL"

    hazard_multi = hdx_client.infer_hazard(["Flood and drought impacts"], shocks, cfg)
    assert hazard_multi.code == "multi"
    assert hazard_multi.label == "Multi-shock Needs"
    assert hazard_multi.hclass == "all"
