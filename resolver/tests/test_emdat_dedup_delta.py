from pathlib import Path

import pandas as pd

from resolver.ingestion import emdat_client


def _config(csv_path: Path) -> dict:
    return {
        "sources": [
            {
                "name": "emdat_dedup",
                "kind": "csv",
                "url": str(csv_path),
                "country_keys": ["ISO"],
                "start_date_keys": ["Start Date"],
                "end_date_keys": ["End Date"],
                "type_keys": ["Disaster Type"],
                "subtype_keys": ["Disaster Subtype"],
                "total_affected_keys": ["Total Affected"],
                "affected_keys": ["Affected"],
                "injured_keys": ["Injured"],
                "homeless_keys": ["Homeless"],
                "id_keys": ["Dis No"],
                "title_keys": ["Event Name"],
                "publisher": "CRED/EM-DAT",
                "source_type": "other",
            }
        ],
        "prefer_hxl": False,
        "allocation_policy": "prorata",
        "shock_map": {"flood": ["Flood"]},
        "default_hazard": "flood",
    }


def test_emdat_dedup_and_monthly_aggregation(tmp_path, monkeypatch):
    csv_path = Path(tmp_path) / "dedup.csv"
    df = pd.DataFrame(
        [
            {
                "ISO": "KEN",
                "Start Date": "2024-02-01",
                "End Date": "2024-02-05",
                "Total Affected": "100",
                "Affected": "",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2024-0001",
                "Event Name": "Flood A",
                "Disaster Type": "Flood",
                "Disaster Subtype": "",
            },
            {
                "ISO": "KEN",
                "Start Date": "2024-02-10",
                "End Date": "2024-02-12",
                "Total Affected": "",
                "Affected": "50",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2024-0002",
                "Event Name": "Flood B",
                "Disaster Type": "Flood",
                "Disaster Subtype": "",
            },
            {
                "ISO": "KEN",
                "Start Date": "2024-02-10",
                "End Date": "2024-02-12",
                "Total Affected": "",
                "Affected": "50",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2024-0002",
                "Event Name": "Flood B",
                "Disaster Type": "Flood",
                "Disaster Subtype": "",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    out_path = Path(tmp_path) / "dedup_out.csv"
    monkeypatch.setenv("RESOLVER_SKIP_EMDAT", "0")
    monkeypatch.delenv("EMDAT_ALLOC_POLICY", raising=False)
    monkeypatch.setattr(emdat_client, "OUT_DIR", Path(tmp_path))
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "load_config", lambda: _config(csv_path))

    assert emdat_client.main() is True
    out_df = pd.read_csv(out_path)
    kenya = out_df[out_df["iso3"] == "KEN"].reset_index(drop=True)
    assert len(kenya) == 1
    assert kenya.iloc[0]["as_of_date"] == "2024-02"
    assert int(kenya.iloc[0]["value"]) == 150
