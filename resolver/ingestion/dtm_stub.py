#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "dtm.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    hz = shocks[shocks["hazard_code"].isin(["ACE","CU"])]

    rows = []
    sample = countries.sample(min(2, len(countries)), random_state=11)
    for _, c in sample.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-dtm-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "displaced", "18000", "persons",
                as_of, pub,
                "IOM-DTM", "cluster", "https://example.org/dtm", f"{h.hazard_label} Displacement Round",
                f"New internal displacement attributed to {h.hazard_label}.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    p = write_staging(make_rows(), OUT, series_semantics="new")
    print(f"wrote {p}")
