#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "fews.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # FEWS NET: early warning; we align to DR/EC hazards when PIN missing
    hz = shocks[shocks["hazard_code"].isin(["DR", "EC"])]
    rows = []
    for _, c in countries.sample(min(2, len(countries)), random_state=33).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-fews-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "1800000", "persons",
                as_of, pub,
                "FEWS NET", "agency", "https://example.org/fews", f"{h.hazard_label} Outlook",
                "Projected food insecurity proxy aligned to IPC where available.",
                "manual", "med", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
