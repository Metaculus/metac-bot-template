from pathlib import Path

import pandas as pd
import yaml

from resolver.ingestion import wfp_mvam_client


KEYWORDS = {
    "economic_crisis": ["inflation", "price"],
    "drought": ["drought"],
}


def _run_connector(tmp_path: Path, monkeypatch, data: pd.DataFrame) -> pd.DataFrame:
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    cfg = {
        "sources": [
            {
                "name": "test-shocks",
                "kind": "csv",
                "url": str(data_path),
                "time_keys": ["date"],
                "country_keys": ["iso3"],
                "admin_keys": ["adm1"],
                "people_keys": ["ifc_people"],
                "driver_keys": ["driver"],
                "series_hint": "stock",
                "publisher": "WFP",
                "source_type": "official",
            }
        ],
        "allow_percent": False,
        "prefer_hxl": False,
        "indicator_priority": ["people_ifc"],
        "shock_keywords": KEYWORDS,
        "default_hazard": "multi",
        "emit_stock": True,
        "emit_incident": False,
        "include_first_month_delta": False,
    }

    cfg_path = tmp_path / "config.yml"
    with open(cfg_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)

    overrides_path = tmp_path / "wfp_mvam_sources.yml"
    overrides = {
        "enabled": True,
        "sources": [
            {
                "name": "test-shocks",
                "url": str(data_path),
            }
        ],
    }
    with open(overrides_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(overrides, fp)

    out_path = tmp_path / "wfp_mvam.csv"
    monkeypatch.setenv("RESOLVER_SKIP_WFP_MVAM", "0")
    monkeypatch.setenv("WFP_MVAM_ALLOW_PERCENT", "0")
    monkeypatch.setenv("WFP_MVAM_INCIDENT", "0")
    monkeypatch.setenv("WFP_MVAM_STOCK", "1")
    monkeypatch.delenv("WFP_MVAM_DENOMINATOR_FILE", raising=False)

    monkeypatch.setattr(wfp_mvam_client, "CONFIG", cfg_path)
    monkeypatch.setattr(wfp_mvam_client, "SOURCES_CONFIG", overrides_path)
    monkeypatch.setattr(wfp_mvam_client, "OUT_PATH", out_path)

    wfp_mvam_client.main()

    return pd.read_csv(out_path)


def test_shock_keyword_mapping(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-01-01", "iso3": "KEN", "adm1": "National", "ifc_people": 1_000, "driver": "Inflation pressure"},
            {"date": "2024-02-01", "iso3": "KEN", "adm1": "National", "ifc_people": 1_200, "driver": "Drought update"},
            {"date": "2024-03-01", "iso3": "KEN", "adm1": "National", "ifc_people": 1_400, "driver": "Inflation and drought impacts"},
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data)
    stock = out[out["series_semantics"] == "stock"].sort_values("as_of_date")

    hazard_by_month = {row.as_of_date: row.hazard_code for row in stock.itertuples()}
    assert hazard_by_month == {
        "2024-01": "EC",
        "2024-02": "DR",
        "2024-03": "MULTI",
    }

    multi_row = stock[stock["as_of_date"] == "2024-03"].iloc[0]
    assert multi_row["hazard_label"] == "Multi-driver Food Insecurity"
