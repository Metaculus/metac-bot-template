from pathlib import Path

import pandas as pd
import yaml

from resolver.ingestion import wfp_mvam_client


def _run_connector(tmp_path: Path, monkeypatch, data: pd.DataFrame) -> pd.DataFrame:
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    cfg = {
        "sources": [
            {
                "name": "test-delta",
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
        "shock_keywords": {"drought": ["drought"]},
        "default_hazard": "multi",
        "emit_stock": True,
        "emit_incident": True,
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
                "name": "test-delta",
                "url": str(data_path),
            }
        ],
    }
    with open(overrides_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(overrides, fp)

    out_path = tmp_path / "wfp_mvam.csv"
    monkeypatch.setenv("RESOLVER_SKIP_WFP_MVAM", "0")
    monkeypatch.setenv("WFP_MVAM_ALLOW_PERCENT", "0")
    monkeypatch.setenv("WFP_MVAM_INCIDENT", "1")
    monkeypatch.setenv("WFP_MVAM_STOCK", "1")
    monkeypatch.delenv("WFP_MVAM_DENOMINATOR_FILE", raising=False)

    monkeypatch.setattr(wfp_mvam_client, "CONFIG", cfg_path)
    monkeypatch.setattr(wfp_mvam_client, "SOURCES_CONFIG", overrides_path)
    monkeypatch.setattr(wfp_mvam_client, "OUT_PATH", out_path)

    wfp_mvam_client.main()

    return pd.read_csv(out_path)


def test_weekly_to_monthly_delta(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-01-07", "iso3": "KEN", "adm1": "National", "ifc_people": 100_000, "driver": "drought"},
            {"date": "2024-02-07", "iso3": "KEN", "adm1": "National", "ifc_people": 110_000, "driver": "drought"},
            {"date": "2024-02-21", "iso3": "KEN", "adm1": "National", "ifc_people": 130_000, "driver": "drought"},
            {"date": "2024-03-05", "iso3": "KEN", "adm1": "National", "ifc_people": 100_000, "driver": "drought"},
            {"date": "2024-03-20", "iso3": "KEN", "adm1": "National", "ifc_people": 90_000, "driver": "drought"},
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data)
    stock = out[out["series_semantics"] == "stock"].sort_values("as_of_date").reset_index(drop=True)
    assert list(stock["as_of_date"]) == ["2024-01", "2024-02", "2024-03"]
    assert [int(v) for v in stock["value"]] == [100000, 120000, 95000]

    incident = out[out["series_semantics"] == "incident"]
    incident_map = {row.as_of_date: int(row.value) for row in incident.itertuples()}
    assert incident_map == {"2024-02": 20000}
