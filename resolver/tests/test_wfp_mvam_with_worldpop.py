import pandas as pd
import yaml

from resolver.ingestion import wfp_mvam_client
from resolver.tools.denominators import clear_population_cache


def _write_config(tmp_path, data_path):
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
                "people_keys": [],
                "driver_keys": ["driver"],
                "series_hint": "stock",
                "publisher": "WFP",
                "source_type": "official",
            }
        ],
        "allow_percent": True,
        "prefer_hxl": False,
        "indicator_priority": ["ifc_pct"],
        "shock_keywords": {"drought": ["drought"], "economic_crisis": ["price"]},
        "default_hazard": "multi",
        "emit_stock": True,
        "emit_incident": True,
        "include_first_month_delta": False,
    }
    cfg_path = tmp_path / "config.yml"
    with open(cfg_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp)
    return cfg_path


def _run_connector(tmp_path, monkeypatch, data_df, denom_df):
    data_path = tmp_path / "wfp.csv"
    denom_path = tmp_path / "population.csv"
    data_df.to_csv(data_path, index=False)
    denom_df.to_csv(denom_path, index=False)

    cfg_path = _write_config(tmp_path, data_path)
    out_path = tmp_path / "wfp_mvam.csv"

    monkeypatch.delenv("RESOLVER_SKIP_WFP_MVAM", raising=False)
    monkeypatch.setenv("WFP_MVAM_ALLOW_PERCENT", "1")
    monkeypatch.setenv("WFP_MVAM_STOCK", "1")
    monkeypatch.setenv("WFP_MVAM_INCIDENT", "1")
    monkeypatch.setenv("WFP_MVAM_DENOMINATOR_FILE", str(denom_path))
    monkeypatch.setenv("WORLDPOP_PRODUCT", "test_product")

    monkeypatch.setattr(wfp_mvam_client, "CONFIG", cfg_path)
    monkeypatch.setattr(wfp_mvam_client, "OUT_PATH", out_path)

    clear_population_cache()
    wfp_mvam_client.main()

    return pd.read_csv(out_path)


def test_percent_rows_converted_with_worldpop(tmp_path, monkeypatch):
    data = pd.DataFrame(
        [
            {"date": "2024-01-05", "iso3": "KEN", "adm1": "National", "ifc_pct": 10, "driver": "drought"},
            {"date": "2024-01-19", "iso3": "KEN", "adm1": "National", "ifc_pct": 20, "driver": "drought"},
            {"date": "2024-02-02", "iso3": "KEN", "adm1": "National", "ifc_pct": 25, "driver": "drought"},
            {"date": "2024-02-16", "iso3": "KEN", "adm1": "National", "ifc_pct": 30, "driver": "drought"},
        ]
    )

    denom = pd.DataFrame(
        [
            {"iso3": "KEN", "year": 2023, "population": 1000000, "product": "test_product"},
        ]
    )

    out = _run_connector(tmp_path, monkeypatch, data, denom)
    stock = out[out["series_semantics"] == "stock"].sort_values("as_of_date").reset_index(drop=True)
    assert list(stock["as_of_date"]) == ["2024-01", "2024-02"]

    # 2024-01 mean = 15%, 2024-02 mean = 27.5% (fallback to 2023 denominator)
    assert int(stock.loc[0, "value"]) == 150000
    assert int(stock.loc[1, "value"]) == 275000

    method = stock.loc[0, "method"]
    assert "denominator=WorldPop test_product year=2023 (fallback for 2024)" in method
    definition = stock.loc[0, "definition_text"]
    assert "fallback for 2024" in definition

    incident = out[out["series_semantics"] == "incident"].reset_index(drop=True)
    assert len(incident) == 1
    assert incident.loc[0, "as_of_date"] == "2024-02"
    assert int(incident.loc[0, "value"]) == 125000
    assert "fallback for 2024" in incident.loc[0, "method"]
