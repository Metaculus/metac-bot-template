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

    rows: List[List[str]] = []
    MAX_RESULTS = _int_env("RESOLVER_MAX_RESULTS", int(cfg["defaults"]["max_results"]))
    DEBUG_EVERY = _int_env("RESOLVER_DEBUG_EVERY", int(cfg["defaults"]["debug_every"]))
    gran = (params_cfg.get("granularity") or "year").lower()

    df_countries, df_shocks = load_registries()
    di = df_shocks[df_shocks["hazard_code"] == "DI"]
    if di.empty:
        return []
    hz_code, hz_label, hz_class = "DI", di.iloc[0]["hazard_label"], di.iloc[0]["hazard_class"]

    try:
        limit_default = int(cfg.get("page_size", 1000) or 1000)
    except Exception:
        limit_default = 1000
    LIMIT = _int_env("UNHCR_LIMIT", limit_default)
    defaults = cfg.get("defaults", {}) or {}
    try:
        max_pages_default = int(defaults.get("max_pages", 10) or 10)
    except Exception:
        max_pages_default = 10
    MAX_PAGES = _int_env("RESOLVER_MAX_PAGES", max_pages_default)

    total = 0
    request_idx = 0
    stop = False
    for yr in sorted(years, reverse=True):
        base_params = {
            "cf_type": params_cfg.get("cf_type", "ISO"),
            "coo_all": params_cfg.get("coo_all", "true"),
            "coa_all": params_cfg.get("coa_all", "true"),
            "year[]": str(yr),
            "limit": str(LIMIT),
        }
        if gran == "month":
            base_params["month[]"] = [f"{m:02d}" for m in range(1, 13)]

        url = base.rstrip("/") + "/" + path.lstrip("/")
        page = 1
        while page <= MAX_PAGES:
            params = dict(base_params)
            params["page"] = str(page)
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as exc:
                dbg(f"request for {yr} page {page} raised {exc}")
                break
            request_idx += 1
            if _debug() and (request_idx % DEBUG_EVERY == 1):
                dbg(f"GET {r.url} -> {r.status_code}")
            if r.status_code != 200:
                break

            try:
                data = r.json()
            except ValueError:
                dbg("response JSON decode failed; skipping year %s page %s" % (yr, page))
                break

            results: List[Dict[str, Any]] = []
            if isinstance(data, list):
                results = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                for key in ("results", "data", "items"):
                    candidate = data.get(key)
                    if isinstance(candidate, list):
                        results = [item for item in candidate if isinstance(item, dict)]
                        if results:
                            break
            if not results:
                break

            for it in results:
                asylum_iso = (
                    it.get("coa_iso")
                    or it.get("coa")
                    or it.get("country_of_asylum")
                    or ""
                ).strip().upper()
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

            m_raw = it.get("month") or it.get("mnth") or it.get("date") or ""
            as_of = ""
            if m_raw:
                s = str(m_raw)
                if len(s) == 2 and s.isdigit():
                    as_of = f"{yr}-{s}-15"
                elif s.isdigit():
                    as_of = f"{yr}-{int(s):02d}-15"
                elif len(s) >= 7 and s[4] == "-":
                    as_of = f"{s[:7]}-15"
            if not as_of:
                as_of = f"{yr}-12-31"

            pub = dt.date.today().isoformat()

            if dt.date.fromisoformat(as_of) < since and dt.date.fromisoformat(pub) < since:
                continue

            title = f"UNHCR asylum applications — {country_name} ({yr})"
            definition = (
                "Applications for international protection in the year; used here as a proxy for cross-border "
                "Displacement Influx (DI)."
            )
            src_url = r.url

            origin_iso = (
                it.get("coo_iso")
                or it.get("coo")
                or it.get("country_of_origin")
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
                stop = True
                break
        if stop:
            break
        if len(results) < LIMIT:
            break
        page += 1
    if stop:
        return rows

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
