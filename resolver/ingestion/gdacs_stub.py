#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "gdacs.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # GDACS: near-real-time alerts; modeled impact for hydro-meteo hazards
    hz = shocks[shocks["hazard_code"].isin(["FL", "TC", "HW"])]
    rows = []
    for _, c in countries.tail(2).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-gdacs-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "32000", "persons",
                as_of, pub,
                "GDACS", "agency", "https://example.org/gdacs", f"{h.hazard_label} Alert",
                "Modeled affected population estimate for alert footprint (indicative).",
                "api", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
