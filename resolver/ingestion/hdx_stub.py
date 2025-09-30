#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "hdx.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    rows = []
    for _, c in countries.head(1).iterrows():
        event_id = f"{c.iso3}-MIX-hdx-stub-r1"
        rows.append([
            event_id, c.country_name, c.iso3,
            "DR", "Drought", "natural",
            "affected", "500000", "persons",
            as_of, pub,
            "HDX (CKAN)", "agency", "https://example.org/hdx", "Dataset Snapshot",
            "Aggregated affected estimate from country dataset discovery (proxy).",
            "api", "low", 1, ing
        ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT))
