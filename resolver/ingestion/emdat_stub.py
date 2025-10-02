#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "emdat.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # EM-DAT: standardized disaster impacts (lagged). Use natural hazards (no earthquakes in scope).
    hz = shocks[shocks["hazard_code"].isin(["FL", "TC", "HW", "DR"])]
    rows = []
    for _, c in countries.head(2).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-emdat-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "56000", "persons",
                as_of, pub,
                "EM-DAT", "agency", "https://example.org/emdat", f"{h.hazard_label} Event Record",
                "People affected per standardized EM-DAT disaster record.",
                "api", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
