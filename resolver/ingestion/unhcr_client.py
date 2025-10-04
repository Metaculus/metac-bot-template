#!/usr/bin/env python3
"""
UNHCR Population API → staging/unhcr.csv

Pulls recent cross-border asylum application counts and emits canonical rows for
DI (Displacement Influx) with metric=affected, unit=persons.

ENV:
  RESOLVER_SKIP_UNHCR=1   → skip network, write header-only CSV
  RESOLVER_DEBUG=1        → verbose logs (throttled)
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

from resolver.ingestion._manifest import ensure_manifest_for_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "unhcr.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

COLUMNS = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","series_semantics","value","unit",
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
    if os.getenv("RESOLVER_SKIP_UNHCR", "") == "1":
        return []

    cfg = load_cfg()
    base = cfg["base_url"]
    path = cfg["endpoints"]["asylum_applications"]
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/json"}

    since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(cfg["window_days"]))
    since = since_dt.date()

    params_cfg = cfg.get("params", {}) or {}
    defaults = cfg.get("defaults", {}) or {}

    today = dt.date.today()
    include_years_cfg = cfg.get("include_years") or []
    years: List[int] = []
    for value in include_years_cfg:
        try:
            years.append(int(str(value).strip()))
        except Exception:
            continue
    if not years:
        try:
            years_back = int(cfg.get("years_back", 3) or 0)
        except Exception:
            years_back = 3
        years = [today.year - offset for offset in range(years_back + 1)]
    years = sorted({year for year in years if isinstance(year, int) and year > 0}, reverse=True)
    if not years:
        years = [today.year]

    rows: List[List[str]] = []
    MAX_RESULTS = _int_env("RESOLVER_MAX_RESULTS", int(defaults.get("max_results", 20000) or 20000))
    DEBUG_EVERY = _int_env("RESOLVER_DEBUG_EVERY", int(defaults.get("debug_every", 10) or 10))
    gran = (params_cfg.get("granularity") or "year").lower()

    df_countries, df_shocks = load_registries()
    di = df_shocks[df_shocks["hazard_code"] == "DI"]
    if di.empty:
        return []
    hz_code, hz_label, hz_class = "DI", di.iloc[0]["hazard_label"], di.iloc[0]["hazard_class"]

    try:
        limit_default = int(cfg.get("page_limit") or cfg.get("page_size") or 500)
    except Exception:
        limit_default = 500
    LIMIT = _int_env("UNHCR_LIMIT", limit_default)

    try:
        max_pages_default = int(defaults.get("max_pages", 10) or 10)
    except Exception:
        max_pages_default = 10
    MAX_PAGES = _int_env("RESOLVER_MAX_PAGES", max_pages_default)

    base_params: Dict[str, Any] = {
        "cf_type": params_cfg.get("cf_type", "ISO"),
        "coo_all": params_cfg.get("coo_all", "true"),
        "coa_all": params_cfg.get("coa_all", "true"),
        "limit": str(LIMIT),
        "year[]": [str(year) for year in years],
    }
    if gran == "month":
        base_params["month[]"] = [f"{m:02d}" for m in range(1, 13)]

    url = base.rstrip("/") + "/" + path.lstrip("/")
    total = 0
    request_idx = 0
    page = 1
    more = True

    while more and page <= MAX_PAGES:
        params = dict(base_params)
        params["page"] = str(page)

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            dbg(f"request for page {page} raised {exc}")
            break

        request_idx += 1
        if _debug() and (request_idx % DEBUG_EVERY == 1):
            dbg(f"GET {response.url} -> {response.status_code}")
        if response.status_code != 200:
            dbg(f"UNHCR request failed with status {response.status_code}")
            break

        try:
            payload = response.json()
        except ValueError as exc:
            dbg(f"response JSON decode failed for page {page}: {exc}")
            break

        results: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            results = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key in ("results", "data", "items"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    results = [item for item in candidate if isinstance(item, dict)]
                    if results:
                        break
        if not results:
            break

        for item in results:
            asylum_iso = (
                item.get("coa_iso")
                or item.get("coa")
                or item.get("country_of_asylum")
                or ""
            ).strip().upper()
            country_name = iso3_to_name(df_countries, asylum_iso)
            if not asylum_iso or not country_name:
                continue

            year_value = None
            for key in ("year", "yr", "year_data", "yearvalue"):
                if key in item and item[key] is not None:
                    year_value = item[key]
                    break
            year_int: Optional[int] = None
            if year_value is not None:
                try:
                    year_int = int(str(year_value).strip())
                except Exception:
                    year_int = None
            if year_int is None:
                date_candidate = item.get("date") or item.get("month")
                if date_candidate:
                    text = str(date_candidate)
                    if len(text) >= 4 and text[:4].isdigit():
                        year_int = int(text[:4])
            if year_int is None and years:
                year_int = years[0]
            if year_int not in years:
                continue

            raw_value = item.get("value") or item.get("applications") or item.get("individuals")
            try:
                value = int(raw_value) if raw_value is not None else None
            except Exception:
                value = None
            if value is None or value < 0:
                continue

            month_raw = item.get("month") or item.get("mnth") or item.get("date") or ""
            as_of = ""
            if month_raw:
                text = str(month_raw)
                if len(text) == 2 and text.isdigit():
                    as_of = f"{year_int}-{text}-15"
                elif text.isdigit():
                    as_of = f"{year_int}-{int(text):02d}-15"
                elif len(text) >= 7 and text[4] == "-":
                    as_of = f"{text[:7]}-15"
            if not as_of:
                as_of = f"{year_int}-12-31"

            publication_date = dt.date.today().isoformat()
            if dt.date.fromisoformat(as_of) < since and dt.date.fromisoformat(publication_date) < since:
                continue

            title = f"UNHCR asylum applications — {country_name} ({year_int})"
            definition = (
                "Applications for international protection in the requested period; used here as a proxy for cross-border "
                "Displacement Influx (DI)."
            )
            src_url = response.url

            origin_iso = (
                item.get("coo_iso")
                or item.get("coo")
                or item.get("country_of_origin")
                or ""
            ).strip().upper() or "-"
            rid = _stable_digest([asylum_iso, origin_iso, as_of, "apps", str(value)], length=12)
            event_id = f"{asylum_iso}-DI-unhcr-apps-{as_of}-{rid}"

            rows.append([
                event_id,
                country_name,
                asylum_iso,
                hz_code,
                hz_label,
                hz_class,
                "affected",
                "stock",
                str(value),
                "persons",
                as_of,
                publication_date,
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
                more = False
                break

        if total >= MAX_RESULTS:
            break

        next_link = None
        if isinstance(payload, dict):
            links = payload.get("links")
            if isinstance(links, dict):
                next_link = links.get("next")
            if not next_link:
                metadata = payload.get("metadata")
                if isinstance(metadata, dict):
                    next_link = metadata.get("next") or metadata.get("links", {}).get("next")
            if not next_link:
                next_link = payload.get("next")

        if next_link or len(results) == LIMIT:
            page += 1
            more = True
        else:
            more = False

    if not rows:
        years_text = ",".join(str(year) for year in years)
        print(f"UNHCR returned 0 rows for requested years={years_text}; writing header only.")
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
        ensure_manifest_for_csv(out)
        print(f"wrote empty {out}")
        return

    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    ensure_manifest_for_csv(out)
    print(f"wrote {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
