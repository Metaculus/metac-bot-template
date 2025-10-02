#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "unhcr.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    hz_codes = ["DI","ACE"]
    hz = shocks[shocks["hazard_code"].isin(hz_codes)]

    rows = []
    sample = countries.sample(min(2, len(countries)), random_state=7)
    for _, c in sample.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-unhcr-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "displaced", "25000", "persons",
                as_of, pub,
                "UNHCR", "cluster", "https://example.org/unhcr", f"{h.hazard_label} Update",
                f"Population displacement associated with {h.hazard_label}.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    p = write_staging(make_rows(), OUT, series_semantics="stock")
    print(f"wrote {p}")
