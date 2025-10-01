#!/usr/bin/env python3
"""
UNHCR Population API → staging/unhcr.csv

Pulls recent cross-border asylum application counts and emits canonical rows for
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
import hashlib
import os, datetime as dt
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


def _stable_digest(parts: list[str], length: int = 12) -> str:
    """
    Deterministic, short hex digest for IDs.
    Join parts with a separator to avoid accidental collisions.
    """
    key = "|".join(parts).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:length]

def make_rows() -> List[List[str]]:
    if os.getenv("RESOLVER_SKIP_UNHCR","") == "1":
        return []

    cfg = load_cfg()
    base = cfg["base_url"]
    path = cfg["endpoints"]["asylum_applications"]
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/json"}

    since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(cfg["window_days"]))
    since = since_dt.date()

    today = dt.date.today()
    years = {today.year}
    if since.year < today.year:
        years.add(since.year)

    params_cfg = cfg.get("params", {}) or {}
    common_params = {
        "cf_type": params_cfg.get("cf_type", "ISO"),
        "coo_all": params_cfg.get("coo_all", "true"),
        "coa_all": params_cfg.get("coa_all", "true"),
    }

    rows: List[List[str]] = []
    MAX_PAGES = _int_env("RESOLVER_MAX_PAGES", int(cfg["defaults"]["max_pages"]))
    MAX_RESULTS = _int_env("RESOLVER_MAX_RESULTS", int(cfg["defaults"]["max_results"]))
    DEBUG_EVERY = _int_env("RESOLVER_DEBUG_EVERY", int(cfg["defaults"]["debug_every"]))
    page_size = int(cfg["page_size"])

    df_countries, df_shocks = load_registries()
    di = df_shocks[df_shocks["hazard_code"] == "DI"]
    if di.empty:
        return []
    hz_code, hz_label, hz_class = "DI", di.iloc[0]["hazard_label"], di.iloc[0]["hazard_class"]

    total = 0
    for yr in sorted(years, reverse=True):
        page = 0
        offset = 0
        while True:
            params = {
                "limit": page_size,
                "page": (offset // page_size) + 1,
                "yearFrom": yr,
                "yearTo": yr,
                **common_params,
            }
            url = base.rstrip("/") + "/" + path.lstrip("/")
            r = requests.get(url, params=params, headers=headers, timeout=30)
            page += 1
            if _debug() and (page % DEBUG_EVERY == 1):
                dbg(f"GET {r.url} -> {r.status_code}")
            if r.status_code != 200:
                break

            data = r.json()
            results = data.get("results") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if not results:
                break

            for it in results:
                asylum_iso = (it.get("coa") or it.get("country_of_asylum") or "").strip().upper()
                country_name = iso3_to_name(df_countries, asylum_iso)
                if not asylum_iso or not country_name:
                    continue

                val = it.get("value") or it.get("applications") or it.get("individuals")
                try:
                    value = int(val) if val is not None else None
                except Exception:
                    value = None
                if value is None or value < 0:
                    continue

                as_of = f"{yr}-12-31"
                pub = dt.date.today().isoformat()

                if dt.date.fromisoformat(as_of) < since and dt.date.fromisoformat(pub) < since:
                    continue

                title = f"UNHCR asylum applications — {country_name} ({yr})"
                definition = (
                    "Applications for international protection filed in the year; used here as a proxy for cross-border "
                    "Displacement Influx (DI)."
                )
                src_url = url

                origin_iso = (it.get("coo") or it.get("country_of_origin") or "").strip().upper() or "-"
                rid = _stable_digest([asylum_iso, origin_iso, str(yr), "apps", as_of, str(value)], length=12)
                event_id = f"{asylum_iso}-DI-unhcr-apps-{yr}-{rid}"

                rows.append([
                    event_id,
                    country_name,
                    asylum_iso,
                    hz_code,
                    hz_label,
                    hz_class,
                    "affected",
                    str(value),
                    "persons",
                    as_of,
                    pub,
                    "UNHCR",
                    "stat",
                    src_url,
                    title,
                    definition,
                    "api",
                    "med",
                    1,
                    dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ])
                total += 1
                if total >= MAX_RESULTS:
                    dbg(f"hit MAX_RESULTS={MAX_RESULTS}, stopping")
                    break

            if total >= MAX_RESULTS:
                break

            if len(results) < page_size or page >= MAX_PAGES:
                break
            offset += page_size

        if total >= MAX_RESULTS:
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
