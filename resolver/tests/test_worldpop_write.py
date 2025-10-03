import pandas as pd
import yaml

from resolver.ingestion import worldpop_client
from resolver.tools.denominators import clear_population_cache


def test_worldpop_skip_writes_headers(tmp_path, monkeypatch):
    data_path = tmp_path / "population.csv"
    staging_path = tmp_path / "worldpop.csv"

    monkeypatch.setenv("RESOLVER_SKIP_WORLDPOP", "1")
    monkeypatch.setattr(worldpop_client, "OUT_DATA", data_path)
    monkeypatch.setattr(worldpop_client, "OUT_STAGING", staging_path)

    assert worldpop_client.main() is False

    data_df = pd.read_csv(data_path)
    staging_df = pd.read_csv(staging_path)

    assert list(data_df.columns) == worldpop_client.CANONICAL_COLUMNS
    assert data_df.empty
    assert list(staging_df.columns) == worldpop_client.CANONICAL_COLUMNS
    assert staging_df.empty


def test_worldpop_upserts_population(tmp_path, monkeypatch):
    latest_csv = tmp_path / "latest.csv"
    prev_csv = tmp_path / "2023.csv"

    latest_df = pd.DataFrame(
        [
            {"iso3": "KEN", "year": 2024, "population": 1000000, "notes": "initial"},
            {"iso3": "UGA", "year": 2023, "population": 45000000},
        ]
    )
    latest_df.to_csv(latest_csv, index=False)

    prev_df = pd.DataFrame(
        [
            {"iso3": "KEN", "year": 2023, "population": 950000},
            {"iso3": "UGA", "year": 2022, "population": 44000000},
        ]
    )
    prev_df.to_csv(prev_csv, index=False)

    cfg = {
        "product": "un_adj_unconstrained",
        "years_back": 1,
        "prefer_hxl": False,
        "keys": {
            "iso3": ["iso3"],
            "year": ["year"],
            "population": ["population"],
            "notes": ["notes"],
        },
        "source": {
            "publisher": "WorldPop",
            "source_type": "official",
            "url_template": str(tmp_path / "{year}.csv"),
        },
    }
    cfg_path = tmp_path / "worldpop.yml"
    with open(cfg_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)

    data_path = tmp_path / "population.csv"
    staging_path = tmp_path / "worldpop.csv"

    monkeypatch.delenv("RESOLVER_SKIP_WORLDPOP", raising=False)
    monkeypatch.setattr(worldpop_client, "CONFIG", cfg_path)
    monkeypatch.setattr(worldpop_client, "OUT_DATA", data_path)
    monkeypatch.setattr(worldpop_client, "OUT_STAGING", staging_path)

    clear_population_cache()
    assert worldpop_client.main() is True

    written = pd.read_csv(data_path)
    assert set(written["iso3"]) == {"KEN", "UGA"}
    ken_rows = written[written["iso3"] == "KEN"].sort_values("year")
    assert list(ken_rows["year"]) == [2023, 2024]
    assert ken_rows.loc[ken_rows["year"] == 2024, "population"].iloc[0] == 1000000
    uga_rows = written[written["iso3"] == "UGA"].sort_values("year")
    assert list(uga_rows["year"]) == [2022, 2023]

    # Update latest population and rerun to confirm upsert (no duplicate rows)
    latest_df.loc[0, "population"] = 1100000
    latest_df.to_csv(latest_csv, index=False)

    clear_population_cache()
    assert worldpop_client.main() is True

    updated = pd.read_csv(data_path)
    ken_rows = updated[updated["iso3"] == "KEN"].sort_values("year")
    assert list(ken_rows["year"]) == [2023, 2024]
    assert ken_rows.loc[ken_rows["year"] == 2024, "population"].iloc[0] == 1100000
    uga_rows = updated[updated["iso3"] == "UGA"].sort_values("year")
    assert list(uga_rows["year"]) == [2022, 2023]

    staging = pd.read_csv(staging_path)
    assert not staging.empty
    assert set(staging["year"]) == {2022, 2023, 2024}


def test_worldpop_keeps_latest_year_per_iso(tmp_path, monkeypatch):
    latest_csv = tmp_path / "latest.csv"
    prev_csv = tmp_path / "2023.csv"

    latest_df = pd.DataFrame(
        [
            {"iso3": "AAA", "year": 2021, "population": 100},
            {"iso3": "BBB", "year": 2024, "population": 200},
        ]
    )
    latest_df.to_csv(latest_csv, index=False)

    prev_df = pd.DataFrame(
        [
            {"iso3": "BBB", "year": 2023, "population": 190},
        ]
    )
    prev_df.to_csv(prev_csv, index=False)

    cfg = {
        "product": "un_adj_unconstrained",
        "years_back": 1,
        "prefer_hxl": False,
        "keys": {
            "iso3": ["iso3"],
            "year": ["year"],
            "population": ["population"],
        },
        "source": {
            "publisher": "WorldPop",
            "source_type": "official",
            "url_template": str(tmp_path / "{year}.csv"),
        },
    }
    cfg_path = tmp_path / "worldpop.yml"
    with open(cfg_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)

    data_path = tmp_path / "population.csv"
    staging_path = tmp_path / "worldpop.csv"

    monkeypatch.delenv("RESOLVER_SKIP_WORLDPOP", raising=False)
    monkeypatch.setattr(worldpop_client, "CONFIG", cfg_path)
    monkeypatch.setattr(worldpop_client, "OUT_DATA", data_path)
    monkeypatch.setattr(worldpop_client, "OUT_STAGING", staging_path)

    clear_population_cache()
    assert worldpop_client.main() is True

    written = pd.read_csv(data_path)
    assert set(written["iso3"]) == {"AAA", "BBB"}
    aaa_rows = written[written["iso3"] == "AAA"].sort_values("year")
    assert list(aaa_rows["year"]) == [2021]
    bbb_rows = written[written["iso3"] == "BBB"].sort_values("year")
    assert list(bbb_rows["year"]) == [2023, 2024]
