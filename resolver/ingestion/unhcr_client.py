#!/usr/bin/env python3
"""
UNHCR Population API → staging/unhcr.csv

Pulls recent cross-border displacement arrivals and emits canonical rows for
DI (Displacement Influx) with metric=affected, unit=persons.

ENV:
  RESOLVER_SKIP_UNHCR=1   → skip network, write header-only CSV
  RESOLVER_DEBUG=1        → verbose logs (throttled)
  RESOLVER_MAX_PAGES=     → override caps
  RESOLVER_MAX_RESULTS=
  RESOLVER_DEBUG_EVERY=

Config: resolver/ingestion/config/unhcr.yml
Registries: resolver/data/countries.csv, resolver/data/shocks.csv
"""

from __future__ import annotations
import os, time, datetime as dt
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "unhcr.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

COLUMNS = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

def _debug() -> bool:
    return os.getenv("RESOLVER_DEBUG","") == "1"

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name,"").strip() or default)
    except Exception:
        return default

def dbg(msg: str):
    if _debug():
        print(f"[UNHCR] {msg}")

def load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_registries():
    c = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    s = pd.read_csv(SHOCKS, dtype=str).fillna("")
    c["country_norm"] = c["country_name"].str.strip().str.lower()
    return c, s

def iso3_to_name(df_c: pd.DataFrame, iso3: str) -> Optional[str]:
    if not iso3:
        return None
    row = df_c[df_c["iso3"] == iso3]
    if row.empty:
        return None
    return row.iloc[0]["country_name"]

def make_rows() -> List[List[str]]:
    if os.getenv("RESOLVER_SKIP_UNHCR","") == "1":
        return []

    cfg = load_cfg()
    base = cfg["base_url"]
    arrivals_path = cfg["endpoints"]["arrivals"]
    headers = {
        "User-Agent": cfg["user_agent"],
        "Accept": "application/json",
    }
    df_countries, df_shocks = load_registries()

    # DI hazard registry row
    di = df_shocks[df_shocks["hazard_code"] == "DI"]
    if di.empty:
        return []
    hz_code = "DI"
    hz_label = di.iloc[0]["hazard_label"]
    hz_class = di.iloc[0]["hazard_class"]

    # window bounds
    since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(cfg["window_days"]))
    since = since_dt.date().isoformat()

    # caps
    MAX_PAGES = _int_env("RESOLVER_MAX_PAGES", int(cfg["defaults"]["max_pages"]))
    MAX_RESULTS = _int_env("RESOLVER_MAX_RESULTS", int(cfg["defaults"]["max_results"]))
    DEBUG_EVERY = _int_env("RESOLVER_DEBUG_EVERY", int(cfg["defaults"]["debug_every"]))

    rows: List[List[str]] = []
    total_made = 0
    page = 0
    offset = 0
    page_size = int(cfg["page_size"])

    # UNHCR population API typically supports limit/offset and filters by year/month or date.
    # We request recent data; connector tolerates variants by filtering client-side as well.
    while True:
        params = {
            "limit": page_size,
            "offset": offset,
            # Prefer updated records too; some deployments expose 'updated_at__gte'
            # If unsupported, server ignores it; we still filter client-side below.
            "updated_at__gte": since,
        }
        url = base.rstrip("/") + "/" + arrivals_path.lstrip("/")
        r = requests.get(url, params=params, headers=headers, timeout=30)
        page += 1
        if _debug() and (page % DEBUG_EVERY == 1):
            dbg(f"GET {r.url} -> {r.status_code}")
        if r.status_code != 200:
            # Fail-soft
            break

        data = r.json()
        # Shape can be {count, next, results} or plain list
        results = data.get("results") if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not results:
            break

        # Parse each item; tolerate schema variation via .get().
        emitted_this_page = 0
        for it in results:
            # Pull iso3 for asylum country
            asylum_iso = (it.get("asylum_country_iso3")
                          or it.get("country_of_asylum_iso3")
                          or "").strip().upper()
            country_name = iso3_to_name(df_countries, asylum_iso)
            if not asylum_iso or not country_name:
                continue

            # Individuals / arrivals count
            val = it.get("individuals")
            try:
                value = int(val) if val is not None else None
            except Exception:
                value = None
            if value is None or value < 0:
                continue

            # Dates: prefer record-level updated_at or explicit year/month
            updated = str(it.get("updated_at") or "")[:10]
            year = str(it.get("year") or "")
            month = str(it.get("month") or "")
            # Build a YYYY-MM-15 "as_of" if year+month present; else fallback to updated or today
            as_of = ""
            if year and month and month.isdigit():
                y = year
                m = f"{int(month):02d}"
                as_of = f"{y}-{m}-15"
            elif updated:
                as_of = updated
            else:
                as_of = dt.date.today().isoformat()

            # Publication date = same as_of by default; if explicit record_date provided, use it
            pub = str(it.get("record_date") or it.get("date") or as_of)[:10]

            # Client-side window filter (union of created/updated logic):
            if (as_of < since) and (pub < since) and (not updated or updated < since):
                continue

            src_url = url  # UNHCR API endpoint (could be enhanced with per-record link if available)
            title = f"UNHCR arrivals — {country_name}"
            definition = "Individuals newly arrived (cross-border) reported by UNHCR Population API; treated as Displacement Influx (DI)."

            event_id = f"{asylum_iso}-DI-unhcr-{year or ''}{month or ''}-{abs(hash((asylum_iso, as_of, value)))%10_000_000}"

            rows.append([
                event_id, country_name, asylum_iso,
                hz_code, hz_label, hz_class,
                "affected", str(value), "persons",
                as_of, pub,
                "UNHCR", "stat", src_url, title,
                definition,
                "api", "med", 1, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ])
            total_made += 1
            emitted_this_page += 1
            if total_made >= MAX_RESULTS:
                dbg(f"hit MAX_RESULTS={MAX_RESULTS}, stopping")
                break

        if total_made >= MAX_RESULTS:
            break

        # Early exit if page produced nothing and appears stale
        if emitted_this_page == 0:
            # crude: if first item has a date and it's older than window, count older-only page
            pass  # we rely on MAX_PAGES + natural empty/next logic

        # Pagination
        if isinstance(data, dict) and data.get("next"):
            offset += page_size
            time.sleep(0.15)
            if page >= MAX_PAGES:
                dbg(f"hit MAX_PAGES={MAX_PAGES}, stopping")
                break
            continue
        else:
            break

    return rows

def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    out = STAGING / "unhcr.csv"
    try:
        rows = make_rows()
    except Exception as e:
        dbg(f"ERROR: {e}")
        rows = []

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
        print(f"wrote empty {out}")
        return

    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
