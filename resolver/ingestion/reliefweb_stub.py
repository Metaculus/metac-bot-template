#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "reliefweb.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    ctry = countries.head(2)
    hz = shocks[shocks["hazard_code"].isin(["FL","TC","HW"])]

    rows = []
    for _, c in ctry.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-reliefweb-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "100000", "persons",
                as_of, pub,
                "OCHA", "sitrep", "https://example.org/reliefweb", f"{h.hazard_label} Sitrep",
                f"People affected per OCHA sitrep for {h.hazard_label}.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    p = write_staging(make_rows(), OUT)
    print(f"wrote {p}")
