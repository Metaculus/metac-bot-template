from pathlib import Path

import pandas as pd

from resolver.ingestion import ipc_client


def _base_config(rows):
    return {
        "sources": [
            {
                "name": "ipc_test",
                "kind": "inline",
                "data": rows,
                "country_keys": ["iso3"],
                "period_start_keys": ["period_start"],
                "period_end_keys": ["period_end"],
                "phase3p_keys": ["phase3plus"],
                "drivers_keys": ["drivers"],
                "publisher": "IPC",
                "source_type": "official",
            }
        ],
        "shock_keywords": {
            "drought": ["drought"],
        },
        "emit_stock": True,
        "emit_incident": True,
        "include_first_month_delta": False,
        "default_hazard": "multi",
    }


def test_ipc_period_to_month_expansion(tmp_path, monkeypatch):
    config = _base_config(
        [
            {
                "iso3": "SDN",
                "period_start": "2025-01-01",
                "period_end": "2025-03-31",
                "phase3plus": "100000",
                "drivers": "drought",
            }
        ]
    )

    out_path = Path(tmp_path) / "ipc.csv"
    monkeypatch.delenv("RESOLVER_SKIP_IPC", raising=False)
    monkeypatch.setattr(ipc_client, "OUT_PATH", out_path)
    monkeypatch.setattr(ipc_client, "load_config", lambda: config)

    assert ipc_client.main() is True

    df = pd.read_csv(out_path)
    stock = df[df["series_semantics"] == "stock"].copy()
    assert set(stock["as_of_date"]) == {"2025-01", "2025-02", "2025-03"}
    assert all(stock["value"].astype(float) == 100000.0)
