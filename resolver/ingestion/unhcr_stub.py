#!/usr/bin/env python3
import os
from pathlib import Path

from _stub_utils import load_registries, now_dates, write_staging

OUT = Path(__file__).resolve().parents[1] / "staging" / "unhcr.csv"
RESOLVER_DEBUG = bool(int(os.getenv("RESOLVER_DEBUG", "0") or 0))

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
    rows = make_rows()
    p = write_staging(rows, OUT, series_semantics="stock")
    print(f"wrote {p}")
    if RESOLVER_DEBUG:
        final = len(rows)
        summary = (
            "summary | "
            f"raw_count={final} final_rows={final} "
            "dropped_value_cast=0 dropped_country_unmatched=0 page_cap_hit=0"
        )
        print(summary)
