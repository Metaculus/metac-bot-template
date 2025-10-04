#!/usr/bin/env python3
"""UNHCR Population API → staging/unhcr.csv.

Adds debug counters, optional narrow test mode, and resilient parsing so we can
inspect intermediate drops when upstream payloads change.

ENV:
  RESOLVER_SKIP_UNHCR=1   → skip network, write header-only CSV
  RESOLVER_DEBUG=1        → verbose logs (throttled)
  RESOLVER_MAX_RESULTS=
  RESOLVER_MAX_PAGES=
  RESOLVER_DEBUG_EVERY=
  UNHCR_TEST_ISO3=

Config: resolver/ingestion/config/unhcr.yml
Registries: resolver/data/countries.csv, resolver/data/shocks.csv
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "unhcr.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "as_of_date",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "revision",
    "ingested_at",
]

logger = logging.getLogger(__name__)
RESOLVER_DEBUG = bool(int(os.getenv("RESOLVER_DEBUG", "0") or 0))
if RESOLVER_DEBUG:
    logger.setLevel(logging.DEBUG)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def _dbg(msg: str, **kv: object) -> None:
    if not RESOLVER_DEBUG:
        return
    if kv:
        suffix = " ".join(f"{key}={value}" for key, value in kv.items())
        logger.debug("%s | %s", msg, suffix)
    else:
        logger.debug("%s", msg)


def _yaml_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_cfg() -> Dict[str, Any]:
    return _yaml_load(CONFIG)


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    countries["country_norm"] = (
        countries["country_name"].astype(str).str.strip().str.lower()
    )
    name_index = {
        row.country_norm: row.iso3
        for row in countries.itertuples(index=False)
        if getattr(row, "country_norm", "") and getattr(row, "iso3", "")
    }
    return countries, shocks, name_index


def _iso3_to_name(df_countries: pd.DataFrame, iso3: str) -> Optional[str]:
    if not iso3:
        return None
    row = df_countries[df_countries["iso3"] == iso3]
    if row.empty:
        return None
    return str(row.iloc[0]["country_name"]) or None


def _normalise_name(value: str) -> str:
    return (value or "").strip().lower()


def _stable_digest(parts: Iterable[str], length: int = 12) -> str:
    key = "|".join(parts).encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:length]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except Exception:  # noqa: BLE001 - defensive
        return default


def _effective_years(params_cfg: Dict[str, Any]) -> List[int]:
    include = params_cfg.get("include_years") or []
    years: List[int] = []
    if isinstance(include, list):
        for value in include:
            try:
                years.append(int(str(value).strip()))
            except Exception:  # noqa: BLE001
                continue
    today = dt.date.today()
    if years:
        return sorted({year for year in years if year > 0}, reverse=True)
    try:
        years_back = int(params_cfg.get("years_back", 3) or 0)
    except Exception:  # noqa: BLE001
        years_back = 3
    window = [today.year - offset for offset in range(years_back + 1)]
    return sorted({year for year in window if year > 0}, reverse=True)


def _test_iso3(cfg: Dict[str, Any]) -> str:
    env_value = os.getenv("UNHCR_TEST_ISO3", "").strip().upper()
    if env_value:
        return env_value
    config_value = str(cfg.get("test_iso3", ""))
    return config_value.strip().upper()


def _match_test_iso3(item: Dict[str, Any], iso3: str) -> bool:
    if not iso3:
        return True
    iso_candidates = [
        str(item.get("coa_iso", "")),
        str(item.get("coa", "")),
        str(item.get("country_of_asylum", "")),
    ]
    name_candidates = [
        str(item.get("country_name", "")),
        str(item.get("country_of_asylum_name", "")),
    ]
    for candidate in iso_candidates:
        if candidate.strip().upper() == iso3:
            return True
    for candidate in name_candidates:
        if _normalise_name(candidate) == _normalise_name(iso3):
            return True
    return False


def _coerce_value(item: Dict[str, Any], counters: Counter) -> Optional[int]:
    for key in ("value", "applications", "individuals"):
        if key in item and item[key] not in (None, ""):
            try:
                value = int(float(str(item[key]).strip()))
            except Exception:  # noqa: BLE001
                counters["dropped_value_cast"] += 1
                return None
            if value < 0:
                counters["dropped_value_cast"] += 1
                return None
            return value
    counters["dropped_value_cast"] += 1
    return None


def _parse_date(
    item: Dict[str, Any],
    value_years: Iterable[int],
) -> Tuple[Optional[int], Optional[int]]:
    years_set = list(value_years)
    year_value = None
    for key in ("year", "yr", "year_data", "yearvalue"):
        if key in item and item[key] not in (None, ""):
            year_value = item[key]
            break
    if year_value is not None:
        try:
            year_int = int(str(year_value).strip())
        except Exception:  # noqa: BLE001
            year_int = None
    else:
        year_int = None
    if year_int is None:
        date_candidate = item.get("date") or item.get("month")
        if date_candidate:
            text = str(date_candidate)
            if len(text) >= 4 and text[:4].isdigit():
                year_int = int(text[:4])
    if year_int is None and years_set:
        year_int = years_set[0]

    month_int: Optional[int] = None
    month_candidate = item.get("month") or item.get("mnth") or item.get("date")
    if month_candidate is not None:
        text = str(month_candidate).strip()
        if text.isdigit():
            try:
                month_int = int(text)
            except Exception:  # noqa: BLE001
                month_int = None
        elif len(text) >= 7 and text[4] in {"-", "/"} and text[5:7].isdigit():
            try:
                month_int = int(text[5:7])
            except Exception:  # noqa: BLE001
                month_int = None
    return year_int, month_int


def _resolve_country(
    item: Dict[str, Any],
    df_countries: pd.DataFrame,
    country_index: Dict[str, str],
    counters: Counter,
) -> Tuple[str, Optional[str], bool]:
    iso = (
        str(item.get("coa_iso", ""))
        or str(item.get("coa", ""))
        or str(item.get("country_of_asylum", ""))
    ).strip().upper()
    raw_name = (
        str(item.get("country_name", ""))
        or str(item.get("country_of_asylum_name", ""))
        or str(item.get("country_of_asylum", ""))
    ).strip()

    if iso:
        name = _iso3_to_name(df_countries, iso) or raw_name or None
        return iso, name, False

    norm = _normalise_name(raw_name)
    if norm and norm in country_index:
        iso_match = country_index[norm]
        name = _iso3_to_name(df_countries, iso_match) or raw_name or None
        return iso_match, name, False

    if raw_name:
        counters["country_unmatched"] += 1
        if RESOLVER_DEBUG:
            return "", raw_name, True
    counters["dropped_country_unmatched"] += 1
    return "", None, False


def _format_as_of(year: int, month: Optional[int], granularity: str) -> str:
    if granularity == "year" or month is None or month <= 0:
        return f"{year:04d}-01-01"
    return f"{year:04d}-{month:02d}-15"


def _format_summary(counters: Counter) -> str:
    keys = (
        "raw_count",
        "after_keymap",
        "after_date_parse",
        "after_country_match",
        "after_window",
        "final_rows",
        "dropped_value_cast",
        "dropped_country_unmatched",
        "country_unmatched",
        "page_cap_hit",
    )
    parts = [f"{key}={int(counters.get(key, 0))}" for key in keys]
    return "summary | " + " ".join(parts)


def make_rows() -> Tuple[List[List[str]], Counter]:
    if os.getenv("RESOLVER_SKIP_UNHCR", "") == "1":
        return [], Counter()

    cfg = load_cfg()
    if not cfg:
        raise RuntimeError("UNHCR config missing or empty")

    params_cfg = cfg.get("params", {}) or {}
    paging_cfg = cfg.get("paging", {}) or {}
    debug_cfg = cfg.get("debug", {}) or {}
    defaults_cfg = cfg.get("defaults", {}) or {}

    base_url = str(cfg.get("base_url", "")).rstrip("/")
    endpoint = str(cfg.get("endpoints", {}).get("asylum_applications", "")).lstrip("/")
    if not base_url or not endpoint:
        raise RuntimeError("UNHCR config missing base_url or endpoint")

    window_days = int(cfg.get("window_days", 60) or 60)
    since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)
    since = since_dt.date()

    years = _effective_years(params_cfg)
    granularity = str(params_cfg.get("granularity", "year")).strip().lower() or "year"

    headers = {
        "User-Agent": str(cfg.get("user_agent", "spagbot-resolver/1.0")),
        "Accept": "application/json",
    }

    limit_default = int(cfg.get("page_limit") or cfg.get("page_size") or 500)
    limit = _int_env("UNHCR_LIMIT", limit_default)

    max_pages_default = int(paging_cfg.get("max_pages") or defaults_cfg.get("max_pages", 10) or 10)
    max_pages = _int_env("RESOLVER_MAX_PAGES", max_pages_default)

    debug_every_default = int(debug_cfg.get("debug_every") or defaults_cfg.get("debug_every", 10) or 10)
    debug_every = _int_env("RESOLVER_DEBUG_EVERY", debug_every_default)

    max_results_default = int(defaults_cfg.get("max_results", 20000) or 20000)
    max_results = _int_env("RESOLVER_MAX_RESULTS", max_results_default)

    base_params: Dict[str, Any] = {
        "cf_type": params_cfg.get("cf_type", "ISO"),
        "coo_all": params_cfg.get("coo_all", "true"),
        "coa_all": params_cfg.get("coa_all", "true"),
        "limit": str(limit),
        "year[]": [str(year) for year in years],
    }
    if granularity == "month":
        base_params["month[]"] = [f"{m:02d}" for m in range(1, 13)]

    url = f"{base_url}/{endpoint}".rstrip("/")

    df_countries, df_shocks, country_index = load_registries()
    di = df_shocks[df_shocks["hazard_code"] == "DI"]
    if di.empty:
        raise RuntimeError("DI hazard not found in shocks registry")
    hz_label = str(di.iloc[0]["hazard_label"])
    hz_class = str(di.iloc[0]["hazard_class"])

    counters: Counter = Counter()
    rows: List[List[str]] = []
    samples: List[List[str]] = []

    request_idx = 0
    page = 1
    total_rows = 0
    test_iso3 = _test_iso3(cfg)
    more = True
    page_cap_hit = False

    while more and page <= max_pages:
        params = dict(base_params)
        params["page"] = str(page)

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:  # noqa: BLE001
            _dbg("request_error", page=page, error=str(exc))
            break

        request_idx += 1
        if RESOLVER_DEBUG and (request_idx % max(1, debug_every) == 1):
            _dbg("request", url=response.url, status=response.status_code)

        if response.status_code != 200:
            _dbg("bad_status", status=response.status_code, page=page)
            break

        try:
            payload = response.json()
        except ValueError as exc:  # noqa: BLE001
            _dbg("json_decode_error", page=page, error=str(exc))
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
        page_raw = len(results)
        counters["raw_count"] += page_raw
        _dbg("raw_count", page=page, count=page_raw)
        if not results:
            break

        if test_iso3:
            filtered = [item for item in results if _match_test_iso3(item, test_iso3)]
            _dbg("narrow_test_iso3", page=page, iso3=test_iso3, kept=len(filtered))
            results = filtered
            if not results:
                page += 1
                continue

        for item in results:
            counters["after_keymap"] += 1
            value = _coerce_value(item, counters)
            if value is None:
                continue

            year_int, month_int = _parse_date(item, years)
            if year_int is None:
                _dbg("drop_missing_year", item=item)
                continue
            counters["after_date_parse"] += 1

            iso3, country_name, unmatched_debug = _resolve_country(
                item, df_countries, country_index, counters
            )
            if not iso3 and not unmatched_debug:
                continue
            counters["after_country_match"] += 1

            as_of = _format_as_of(year_int, month_int, granularity)
            try:
                as_of_date = dt.date.fromisoformat(as_of)
            except ValueError:
                _dbg("drop_bad_date", as_of=as_of)
                continue
            if as_of_date < since:
                continue
            counters["after_window"] += 1

            publication_date = dt.date.today().isoformat()
            src_url = response.url
            origin_iso = (
                str(item.get("coo_iso", ""))
                or str(item.get("coo", ""))
                or str(item.get("country_of_origin", ""))
            ).strip().upper() or "-"
            rid = _stable_digest([iso3 or country_name or "UNK", origin_iso, as_of, str(value)])
            event_id = f"{(iso3 or 'UNK')}-DI-unhcr-apps-{as_of}-{rid}"
            final_country_name = country_name or item.get("country_name") or ""

            row = [
                event_id,
                final_country_name,
                iso3,
                "DI",
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
                f"UNHCR asylum applications — {final_country_name} ({year_int})",
                (
                    "Applications for international protection in the requested period; "
                    "used here as a proxy for cross-border Displacement Influx (DI)."
                ),
                "api",
                "med",
                1,
                dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ]
            rows.append(row)
            counters["final_rows"] += 1
            total_rows += 1

            if RESOLVER_DEBUG and len(samples) < 2:
                samples.append(row)

            if total_rows >= max_results:
                _dbg("max_results_hit", max_results=max_results)
                more = False
                break

        if total_rows >= max_results:
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

        if next_link or (len(results) and len(results) >= limit):
            page += 1
            more = True
        else:
            more = False

        if more and page > max_pages:
            page_cap_hit = True
            counters["page_cap_hit"] += 1
            break

    if RESOLVER_DEBUG and samples:
        for idx, sample in enumerate(samples, start=1):
            _dbg(
                "sample_row",
                idx=idx,
                iso3=sample[2],
                as_of=sample[10],
                value=sample[8],
            )

    if page_cap_hit:
        _dbg("page_cap_reached", max_pages=max_pages)

    if RESOLVER_DEBUG:
        _dbg("after_keymap", count=counters.get("after_keymap", 0))
        _dbg("after_date_parse", count=counters.get("after_date_parse", 0))
        _dbg("after_country_match", count=counters.get("after_country_match", 0))
        _dbg("after_window", count=counters.get("after_window", 0))
        _dbg("final_rows", count=counters.get("final_rows", 0))

    if not rows:
        years_text = ",".join(str(year) for year in years)
        logger.info(
            "UNHCR returned no rows for years=%s (granularity=%s)",
            years_text,
            granularity,
        )
    return rows, counters


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    out = STAGING / "unhcr.csv"
    try:
        rows, counters = make_rows()
    except Exception as exc:  # noqa: BLE001
        logger.exception("unhandled error during UNHCR ingestion: %s", exc)
        rows, counters = [], Counter()

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
        ensure_manifest_for_csv(out)
        print(f"wrote empty {out}")
    else:
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
        print(f"wrote {out} rows={len(rows)}")

    summary = _format_summary(counters)
    print(summary)
    _dbg("summary_emitted", summary=summary)

    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    ensure_manifest_for_csv(out)
    print(f"wrote {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
