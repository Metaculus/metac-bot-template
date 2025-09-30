#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "ifrc_go.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    ctry = countries.tail(2)
    hz = shocks[shocks["hazard_code"].isin(["FL","TC"])]

    rows = []
    for _, c in ctry.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-ifrc-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "75000", "persons",
                as_of, pub,
                "IFRC", "sitrep", "https://example.org/ifrc", f"{h.hazard_label} DREF",
                f"People affected per IFRC GO DREF for {h.hazard_label}.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    p = write_staging(make_rows(), OUT)
    print(f"wrote {p}")
