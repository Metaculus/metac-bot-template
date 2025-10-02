import importlib
from pathlib import Path

import pandas as pd


def _write_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_who_phe_monthly_resample(tmp_path, monkeypatch):
    monkeypatch.delenv("RESOLVER_SKIP_WHO", raising=False)
    monkeypatch.delenv("WHO_PHE_PIN_RATIO", raising=False)
    monkeypatch.delenv("WHO_PHE_ALLOW_FIRST_MONTH", raising=False)

    weekly_path = tmp_path / "weekly_incident.csv"
    _write_csv(
        weekly_path,
        [
            {"iso3": "KEN", "year": 2024, "week": 1, "cases": 5},
            {"iso3": "KEN", "year": 2024, "week": 2, "cases": 4},
            {"iso3": "KEN", "year": 2024, "week": 6, "cases": 10},
        ],
    )

    daily_path = tmp_path / "daily_incident.csv"
    _write_csv(
        daily_path,
        [
            {"iso3": "UGA", "date": "2024-01-01", "cases": 3},
            {"iso3": "UGA", "date": "2024-01-12", "cases": 2},
            {"iso3": "UGA", "date": "2024-02-15", "cases": 7},
        ],
    )

    cumulative_path = tmp_path / "cumulative_weekly.csv"
    _write_csv(
        cumulative_path,
        [
            {"iso3": "ETH", "year": 2024, "week": 1, "cases": 10},
            {"iso3": "ETH", "year": 2024, "week": 4, "cases": 25},
            {"iso3": "ETH", "year": 2024, "week": 6, "cases": 40},
            {"iso3": "ETH", "year": 2024, "week": 8, "cases": 45},
            {"iso3": "ETH", "year": 2024, "week": 10, "cases": 30},
        ],
    )

    subnational_path = tmp_path / "subnational.csv"
    _write_csv(
        subnational_path,
        [
            {"iso3": "TCD", "admin1": "Region A", "date": "2024-01-02", "cases": 2},
            {"iso3": "TCD", "admin1": "Region B", "date": "2024-01-05", "cases": 3},
            {"iso3": "TCD", "admin1": "Region A", "date": "2024-02-10", "cases": 4},
            {"iso3": "TCD", "admin1": "Region B", "date": "2024-02-12", "cases": 6},
        ],
    )

    config = tmp_path / "who_phe.yml"
    config.write_text(
        "\n".join(
            [
                "sources:",
                f"  - name: weekly_incident",
                f"    kind: csv",
                f"    url: {weekly_path}",
                f"    time_keys: ['week', 'year']",
                f"    country_keys: ['iso3']",
                f"    case_keys: ['cases']",
                f"    disease: cholera",
                f"    series_hint: incident",
                f"  - name: daily_incident",
                f"    kind: csv",
                f"    url: {daily_path}",
                f"    time_keys: ['date']",
                f"    country_keys: ['iso3']",
                f"    case_keys: ['cases']",
                f"    disease: measles",
                f"    series_hint: incident",
                f"  - name: cumulative_weekly",
                f"    kind: csv",
                f"    url: {cumulative_path}",
                f"    time_keys: ['week', 'year']",
                f"    country_keys: ['iso3']",
                f"    case_keys: ['cases']",
                f"    disease: cholera",
                f"    series_hint: cumulative",
                f"  - name: subnational_daily",
                f"    kind: csv",
                f"    url: {subnational_path}",
                f"    time_keys: ['date']",
                f"    country_keys: ['iso3']",
                f"    case_keys: ['cases']",
                f"    disease: cholera",
                f"    series_hint: incident",
                "prefer_hxl: false",
                "allow_first_month_delta: false",
            ]
        ),
        encoding="utf-8",
    )

    mod = importlib.import_module("resolver.ingestion.who_phe_client")
    monkeypatch.setattr(mod, "CONFIG", config)
    output = tmp_path / "who_phe.csv"
    monkeypatch.setattr(mod, "OUT_PATH", output)

    mod.main()

    df = pd.read_csv(output)

    ken = df[df["iso3"] == "KEN"].set_index("as_of_date")
    assert int(ken.loc["2024-01", "value"]) == 9
    assert int(ken.loc["2024-02", "value"]) == 10

    uga = df[df["iso3"] == "UGA"].set_index("as_of_date")
    assert int(uga.loc["2024-01", "value"]) == 5
    assert int(uga.loc["2024-02", "value"]) == 7

    eth = df[df["iso3"] == "ETH"].set_index("as_of_date")
    assert "2024-01" not in eth.index
    assert int(eth.loc["2024-02", "value"]) == 20
    assert int(eth.loc["2024-03", "value"]) == 0

    tcd = df[df["iso3"] == "TCD"].set_index("as_of_date")
    assert int(tcd.loc["2024-01", "value"]) == 5
    assert int(tcd.loc["2024-02", "value"]) == 10

    assert set(df["hazard_code"]) == {"PHE"}
    assert set(df["series_semantics"]) == {"incident"}
