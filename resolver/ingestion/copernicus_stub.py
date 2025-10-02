#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "copernicus.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # Copernicus EMS: mapping of activation footprint; we record affected as exposure proxy
    hz = shocks[shocks["hazard_code"].isin(["FL", "TC", "HW"])]
    rows = []
    for _, c in countries.sample(min(2, len(countries)), random_state=21).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-copernicus-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "14000", "persons",
                as_of, pub,
                "Copernicus EMS", "agency", "https://example.org/copernicus", f"{h.hazard_label} Activation",
                "Population exposure estimated from activation footprint (proxy).",
                "api", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
