#!/usr/bin/env python3
from pathlib import Path
from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "wfp_mvam.csv"

def make_rows():
    countries, shocks = load_registries()
    as_of, pub, ing = now_dates()
    # WFP mVAM/price bulletins: context for EC/DR
    hz = shocks[shocks["hazard_code"].isin(["EC", "DR"])]
    rows = []
    for _, c in countries.sample(min(2, len(countries)), random_state=35).iterrows():
        for _, h in hz.head(1).iterrows():
            event_id = f"{c.iso3}-{h.hazard_code}-wfp-mvam-stub-r1"
            rows.append([
                event_id, c.country_name, c.iso3,
                h.hazard_code, h.hazard_label, h.hazard_class,
                "affected", "300000", "persons",
                as_of, pub,
                "WFP mVAM", "agency", "https://example.org/wfp", f"{h.hazard_label} Bulletin",
                "Market/food security indicator as affected proxy; not final PIN.",
                "manual", "low", 1, ing
            ])
    return rows

if __name__ == "__main__":
    print(write_staging(make_rows(), OUT, series_semantics="stock"))
