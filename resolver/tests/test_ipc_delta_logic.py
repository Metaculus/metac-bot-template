from pathlib import Path

import pandas as pd

from resolver.ingestion import ipc_client


def _base_config(rows):
    return {
        "sources": [
            {
                "name": "ipc_delta_test",
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


def test_ipc_delta_clips_negatives(tmp_path, monkeypatch):
    config = _base_config(
        [
            {
                "iso3": "ETH",
                "period_start": "2025-01-01",
                "period_end": "2025-01-31",
                "phase3plus": "100000",
                "drivers": "drought",
            },
            {
                "iso3": "ETH",
                "period_start": "2025-02-01",
                "period_end": "2025-02-28",
                "phase3plus": "120000",
                "drivers": "drought",
            },
            {
                "iso3": "ETH",
                "period_start": "2025-03-01",
                "period_end": "2025-03-31",
                "phase3plus": "110000",
                "drivers": "drought",
            },
        ]
    )

    out_path = Path(tmp_path) / "ipc.csv"
    monkeypatch.delenv("RESOLVER_SKIP_IPC", raising=False)
    monkeypatch.setattr(ipc_client, "OUT_PATH", out_path)
    monkeypatch.setattr(ipc_client, "load_config", lambda: config)

    ipc_client.main()

    df = pd.read_csv(out_path)
    incident = df[df["series_semantics"] == "incident"].copy()

    assert list(incident["as_of_date"]) == ["2025-02"]
    assert incident.iloc[0]["value"] == 20000.0
