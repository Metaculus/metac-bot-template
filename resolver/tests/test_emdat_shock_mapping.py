from pathlib import Path

import pandas as pd

from resolver.ingestion import emdat_client


def _config(csv_path: Path) -> dict:
    return {
        "sources": [
            {
                "name": "emdat_mapping",
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
        "shock_map": {
            "tropical_cyclone": ["Storm", "Tropical cyclone"],
            "phe": ["Epidemic"],
            "other": ["Industrial"],
        },
        "default_hazard": "other",
    }


def test_emdat_shock_mapping(tmp_path, monkeypatch):
    csv_path = Path(tmp_path) / "mapping.csv"
    df = pd.DataFrame(
        [
            {
                "ISO": "PHL",
                "Start Date": "2025-01-05",
                "End Date": "2025-01-06",
                "Total Affected": "100",
                "Affected": "",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2025-1001",
                "Event Name": "Storm Event",
                "Disaster Type": "Storm",
                "Disaster Subtype": "Tropical cyclone",
            },
            {
                "ISO": "GIN",
                "Start Date": "2025-02-01",
                "End Date": "2025-02-01",
                "Total Affected": "200",
                "Affected": "",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2025-2002",
                "Event Name": "Epidemic Event",
                "Disaster Type": "Epidemic",
                "Disaster Subtype": "",
            },
            {
                "ISO": "USA",
                "Start Date": "2025-03-10",
                "End Date": "2025-03-12",
                "Total Affected": "300",
                "Affected": "",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2025-3003",
                "Event Name": "Mystery Event",
                "Disaster Type": "Mystery",
                "Disaster Subtype": "",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    out_path = Path(tmp_path) / "mapping_out.csv"
    monkeypatch.setenv("RESOLVER_SKIP_EMDAT", "0")
    monkeypatch.delenv("EMDAT_ALLOC_POLICY", raising=False)
    monkeypatch.setattr(emdat_client, "OUT_DIR", Path(tmp_path))
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "load_config", lambda: _config(csv_path))

    assert emdat_client.main() is True
    out_df = pd.read_csv(out_path)
    mapping = {row.iso3: row.hazard_code for row in out_df.itertuples(index=False)}
    assert mapping["PHL"] == "TC"
    assert mapping["GIN"] == "PHE"
    assert mapping["USA"] == "OT"
