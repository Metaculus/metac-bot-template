#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "who.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()

    hz = shocks[shocks["hazard_code"].isin(["PHE"])]

    rows = []
    sample = countries.sample(min(2, len(countries)), random_state=13)
    for _, c in sample.iterrows():
        for _, h in hz.iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-who-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "cases", "3200", "persons_cases",
                as_of, pub,
                "WHO", "sitrep", "https://example.org/who", f"{h.hazard_label} DON",
                f"Case counts for {h.hazard_label}. Not equivalent to PIN.",
                "api", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    p = write_staging(make_rows(), OUT, series_semantics="stock")
    print(f"wrote {p}")
