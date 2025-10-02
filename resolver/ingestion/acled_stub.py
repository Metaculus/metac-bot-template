#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "acled.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # ACLED: conflict events — in real connector, we'd pull event counts; here we emit a proxy affected number for demo
    hz = shocks[shocks["hazard_code"].isin(["ACE", "CU"])]
    rows = []
    for _, c in countries.sample(min(2, len(countries)), random_state=27).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-acled-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "12000", "persons",
                as_of, pub,
                "ACLED", "agency", "https://example.org/acled", f"{h.hazard_label} Events (proxy)",
                "Conflict event volumes; affected number is a placeholder proxy (not for final resolution).",
                "api", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
