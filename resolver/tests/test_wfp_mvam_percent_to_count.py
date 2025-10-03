from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from resolver.ingestion import wfp_mvam_client


SHOCK_KEYWORDS = {
    "economic_crisis": ["inflation", "price"],
    "drought": ["drought"],
}


def _write_config(tmp_path: Path, data_path: Path, allow_percent: bool, denom_path: Optional[Path] = None) -> Path:
    cfg = {
        "sources": [
            {
                "name": "test",
                "kind": "csv",
                "url": str(data_path),
                "time_keys": ["date"],
                "country_keys": ["iso3"],
                "admin_keys": ["adm1"],
                "pct_keys": ["ifc_pct"],
                "people_keys": ["ifc_people"],
                "population_keys": ["population"],
                "driver_keys": ["driver"],
                "series_hint": "stock",
                "publisher": "WFP",
                "source_type": "official",
            }
        ],
        "allow_percent": allow_percent,
        "prefer_hxl": False,
        "indicator_priority": ["people_ifc", "ifc_pct"],
        "shock_keywords": SHOCK_KEYWORDS,
        "default_hazard": "multi",
        "emit_stock": True,
        "emit_incident": False,
        "include_first_month_delta": False,
        "suppress_admin_when_no_subpop": False,
    }
    if denom_path is not None:
        cfg["denominator_file"] = str(denom_path)
    cfg_path = tmp_path / "config.yml"
    with open(cfg_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)
    return cfg_path


def _run_connector(tmp_path: Path, monkeypatch, data: pd.DataFrame, *, allow_percent: bool, denom: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)

    denom_path = None
    if denom is not None:
        denom_path = tmp_path / "denom.csv"
        denom.to_csv(denom_path, index=False)

    cfg_path = _write_config(tmp_path, data_path, allow_percent, denom_path)

    overrides_path = tmp_path / "wfp_mvam_sources.yml"
    overrides = {
        "enabled": True,
        "sources": [
            {
                "name": "test",
                "url": str(data_path),
            }
        ],
    }
    with open(overrides_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(overrides, fp)

    out_path = tmp_path / "wfp_mvam.csv"
    monkeypatch.setenv("RESOLVER_SKIP_WFP_MVAM", "0")
    monkeypatch.setenv("WFP_MVAM_ALLOW_PERCENT", "1" if allow_percent else "0")
    monkeypatch.setenv("WFP_MVAM_INCIDENT", "0")
    monkeypatch.setenv("WFP_MVAM_STOCK", "1")
    if denom_path is not None:
        monkeypatch.setenv("WFP_MVAM_DENOMINATOR_FILE", str(denom_path))
    else:
        monkeypatch.delenv("WFP_MVAM_DENOMINATOR_FILE", raising=False)

    monkeypatch.setattr(wfp_mvam_client, "CONFIG", cfg_path)
    monkeypatch.setattr(wfp_mvam_client, "SOURCES_CONFIG", overrides_path)
    monkeypatch.setattr(wfp_mvam_client, "OUT_PATH", out_path)

    wfp_mvam_client.main()

    return pd.read_csv(out_path)


def test_percent_with_dataset_population(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-01-05", "iso3": "KEN", "adm1": "North", "ifc_pct": 10, "population": 1000, "driver": "drought"},
            {"date": "2024-01-19", "iso3": "KEN", "adm1": "North", "ifc_pct": 12, "population": 1000, "driver": "drought"},
            {"date": "2024-01-10", "iso3": "KEN", "adm1": "South", "ifc_pct": 15, "population": 2000, "driver": "drought"},
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data, allow_percent=True)
    stock = out[out["series_semantics"] == "stock"].reset_index(drop=True)
    assert not stock.empty
    assert stock.loc[0, "hazard_code"] == "DR"
    assert int(stock.loc[0, "value"]) == 410


def test_percent_with_external_denominator(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-02-07", "iso3": "KEN", "adm1": "National", "ifc_pct": 15, "driver": "price surge"},
        ]
    )
    denom = pd.DataFrame([
        {"iso3": "KEN", "year": 2024, "population": 5000},
    ])

    out = _run_connector(tmp_path, monkeypatch, data, allow_percent=True, denom=denom)
    stock = out[out["series_semantics"] == "stock"].reset_index(drop=True)
    assert not stock.empty
    assert int(stock.loc[0, "value"]) == 750


def test_percent_rows_skipped_when_disallowed(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-03-01", "iso3": "KEN", "adm1": "North", "ifc_pct": 20, "driver": "drought"},
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data, allow_percent=False)
    stock = out[out["series_semantics"] == "stock"]
    assert stock.empty


def test_people_column_preferred_over_percent(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {
                "date": "2024-04-01",
                "iso3": "KEN",
                "adm1": "National",
                "ifc_people": 2000,
                "ifc_pct": 50,
                "driver": "inflation"
            },
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data, allow_percent=True)
    stock = out[out["series_semantics"] == "stock"].reset_index(drop=True)
    assert not stock.empty
    assert int(stock.loc[0, "value"]) == 2000
    assert stock.loc[0, "hazard_code"] == "EC"
