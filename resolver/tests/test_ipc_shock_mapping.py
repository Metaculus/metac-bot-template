from pathlib import Path

import pandas as pd

from resolver.ingestion import ipc_client


def _config(rows):
    return {
        "sources": [
            {
                "name": "ipc_shock_test",
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
            "armed_conflict_escalation": ["conflict", "violence"],
            "economic_crisis": ["economic", "inflation"],
        },
        "emit_stock": True,
        "emit_incident": False,
        "include_first_month_delta": False,
        "default_hazard": "multi",
    }


def test_ipc_shock_keyword_mapping(tmp_path, monkeypatch):
    config = _config(
        [
            {
                "iso3": "KEN",
                "period_start": "2025-01-01",
                "period_end": "2025-01-31",
                "phase3plus": "50000",
                "drivers": "Drought; conflict",
            },
            {
                "iso3": "KEN",
                "period_start": "2025-02-01",
                "period_end": "2025-02-28",
                "phase3plus": "60000",
                "drivers": "Economic shock",
            },
            {
                "iso3": "KEN",
                "period_start": "2025-03-01",
                "period_end": "2025-03-31",
                "phase3plus": "70000",
                "drivers": "",
            },
        ]
    )

    out_path = Path(tmp_path) / "ipc.csv"
    monkeypatch.delenv("RESOLVER_SKIP_IPC", raising=False)
    monkeypatch.setattr(ipc_client, "OUT_PATH", out_path)
    monkeypatch.setattr(ipc_client, "load_config", lambda: config)

    ipc_client.main()

    df = pd.read_csv(out_path)
    stock = df[df["series_semantics"] == "stock"].sort_values("as_of_date").reset_index(drop=True)

    assert list(stock["hazard_code"]) == ["DR", "EC", "multi"]
    assert stock.iloc[2]["hazard_label"] == "Multi-driver Food Insecurity"
