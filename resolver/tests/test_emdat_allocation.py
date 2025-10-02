from pathlib import Path

import pandas as pd

from resolver.ingestion import emdat_client


def _make_config(csv_path: Path) -> dict:
    return {
        "sources": [
            {
                "name": "test_emdat",
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
        "prefer_hxl": True,
        "allocation_policy": "prorata",
        "shock_map": {"flood": ["Flood"]},
        "default_hazard": "other",
    }


def test_emdat_allocation_prorata_vs_start(tmp_path, monkeypatch):
    csv_path = Path(tmp_path) / "emdat_events.csv"
    columns = [
        "ISO",
        "Start Date",
        "End Date",
        "Total Affected",
        "Affected",
        "Injured",
        "Homeless",
        "Dis No",
        "Event Name",
        "Disaster Type",
        "Disaster Subtype",
    ]
    df = pd.DataFrame(
        [
            {
                "ISO": "SDN",
                "Start Date": "2025-01-15",
                "End Date": "2025-03-10",
                "Total Affected": "9000",
                "Affected": "",
                "Injured": "",
                "Homeless": "",
                "Dis No": "2025-0001",
                "Event Name": "Flood Event",
                "Disaster Type": "Flood",
                "Disaster Subtype": "Riverine Flood",
            }
        ],
        columns=columns,
    )
    df.to_csv(csv_path, index=False)
    hxl_row = (
        "#country+code,#date+start,#date+end,#affected+total,#affected,#affected+injured," \
        "#affected+homeless,#event+id,#event+name,#disaster+type,#disaster+subtype"
    )
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, hxl_row)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path = Path(tmp_path) / "emdat_out.csv"

    monkeypatch.setenv("RESOLVER_SKIP_EMDAT", "0")
    monkeypatch.delenv("EMDAT_ALLOC_POLICY", raising=False)
    monkeypatch.setattr(emdat_client, "OUT_DIR", Path(tmp_path))
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_path)
    monkeypatch.setattr(emdat_client, "load_config", lambda: _make_config(csv_path))

    assert emdat_client.main() is True
    out_df = pd.read_csv(out_path)
    assert set(out_df["as_of_date"]) == {"2025-01", "2025-02", "2025-03"}
    allocations = {row.as_of_date: int(row.value) for row in out_df.itertuples(index=False)}
    assert allocations == {"2025-01": 2782, "2025-02": 4582, "2025-03": 1636}
    assert sum(allocations.values()) == 9000

    out_start = Path(tmp_path) / "emdat_start.csv"
    monkeypatch.setenv("EMDAT_ALLOC_POLICY", "start")
    monkeypatch.setattr(emdat_client, "OUT_PATH", out_start)
    assert emdat_client.main() is True
    start_df = pd.read_csv(out_start)
    assert list(start_df["as_of_date"]) == ["2025-01"]
    assert int(start_df.iloc[0]["value"]) == 9000
