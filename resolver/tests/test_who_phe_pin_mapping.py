import importlib
from pathlib import Path

import pandas as pd


def test_who_phe_pin_ratio(tmp_path, monkeypatch):
    monkeypatch.setenv("WHO_PHE_PIN_RATIO", "0.2")
    monkeypatch.delenv("RESOLVER_SKIP_WHO", raising=False)
    monkeypatch.delenv("WHO_PHE_ALLOW_FIRST_MONTH", raising=False)

    data_path = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {"iso3": "MLI", "date": "2024-01-10", "cases": 50},
            {"iso3": "MLI", "date": "2024-01-20", "cases": 30},
            {"iso3": "MLI", "date": "2024-02-12", "cases": 20},
        ]
    ).to_csv(data_path, index=False)

    config = tmp_path / "who_phe.yml"
    config.write_text(
        "\n".join(
            [
                "sources:",
                f"  - name: malaria_daily",
                f"    kind: csv",
                f"    url: {data_path}",
                f"    time_keys: ['date']",
                f"    country_keys: ['iso3']",
                f"    case_keys: ['cases']",
                f"    disease: malaria",
                f"    series_hint: incident",
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
    assert set(df["metric"]) == {"affected", "in_need"}
    grouped = df.groupby("metric").get_group

    affected = grouped("affected").iloc[0]
    pin = grouped("in_need").iloc[0]

    assert affected["iso3"] == "MLI"
    assert affected["as_of_date"] == "2024-01"
    assert int(affected["value"]) == 80
    assert int(pin["value"]) == 16

    aff_parts = affected["event_id"].split("-")
    pin_parts = pin["event_id"].split("-")
    assert aff_parts[:3] == ["MLI", "WHO", "phe"]
    assert pin_parts[:3] == ["MLI", "WHO", "phe"]
    assert aff_parts[4:6] == pin_parts[4:6]
    assert aff_parts[3] == "affected"
    assert pin_parts[3] == "in_need"

    assert pin["doc_title"] == affected["doc_title"]
