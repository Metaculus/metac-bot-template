#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "gov_ndma.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # Government NDMA: official sitreps; good for PA when PIN not yet out
    hz = shocks[shocks["hazard_code"].isin(["FL", "TC", "HW", "CU"])]
    rows = []
    for _, c in countries.head(2).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-ndma-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "42000", "persons",
                as_of, pub,
                "NDMA", "gov", "https://example.org/ndma", f"{h.hazard_label} Gov Sitrep",
                "Official government 'people affected' total (subject to revision).",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT))
