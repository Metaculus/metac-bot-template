#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "ucdp.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    hz = shocks[shocks["hazard_code"].isin(["ACE", "ACO", "ACC"])]
    rows = []
    for _, c in countries.sample(min(2, len(countries)), random_state=31).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-ucdp-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "8000", "persons",
                as_of, pub,
                "UCDP", "agency", "https://example.org/ucdp", f"{h.hazard_label} Events (proxy)",
                "Conflict dataset context; affected number is a placeholder proxy.",
                "api", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT))
