#!/usr/bin/env python3
"""WFP mVAM connector producing monthly national food insecurity counts."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.tools.denominators import (
    get_population_record,
    safe_pct_to_people,
)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
DEFAULT_CONFIG = ROOT / "ingestion" / "config" / "wfp_mvam.yml"
CONFIG = DEFAULT_CONFIG
SOURCES_CONFIG = ROOT / "ingestion" / "config" / "wfp_mvam_sources.yml"
COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"
OUT_PATH = STAGING / "wfp_mvam.csv"
DEFAULT_DENOMINATOR = DATA / "population.csv"

CANONICAL_HEADERS = [
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

DEFAULT_PUBLISHER = "WFP"
DEFAULT_SOURCE_TYPE = "official"
DEFAULT_UNIT = "persons"
DEFAULT_METRIC = "in_need"
DEFAULT_MULTI = ("MULTI", "Multi-driver Food Insecurity", "multi")
DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"

HAZARD_KEY_TO_CODE = {
    "economic_crisis": "EC",
    "drought": "DR",
    "flood": "FL",
    "armed_conflict_escalation": "ACE",
    "phe": "PHE",
}

DEFAULT_SOURCE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "mvam_ifc_global": {
        "name": "mvam_ifc_global",
        "kind": "csv",
        "time_keys": ["date", "week", "month", "#date"],
        "country_keys": ["iso3", "#country+code", "country_iso3", "country"],
        "admin_keys": ["adm1", "adm2", "#adm1+name", "#adm2+name"],
        "pct_keys": [
            "ifc_pct",
            "insufficient_food_consumption_pct",
            "#food+consumption:insufficient+%",
        ],
        "people_keys": ["ifc_people", "people_ifc", "#inneed"],
        "population_keys": ["pop", "population", "#population"],
        "driver_keys": ["driver", "drivers", "tags", "notes"],
        "series_hint": "stock",
        "publisher": "WFP",
        "source_type": "official",
        "metric": "in_need",
    }
}


def get_source_templates(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    templates: Dict[str, Dict[str, Any]] = {}
    for source in cfg.get("sources", []) or []:
        name = str(source.get("name") or "").strip()
        if not name:
            continue
        templates[name] = dict(source)
    for name, template in DEFAULT_SOURCE_TEMPLATES.items():
        templates.setdefault(name, dict(template))
    return templates


@dataclass
class Hazard:
    code: str
    label: str
    hazard_class: str


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[wfp_mvam] {message}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _select_first_text(series: pd.Series, default: str = "") -> str:
    """Return the first non-empty string from ``series``."""

    if series is None:
        return default
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return text
    return default


def _load_admin_population_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load an optional admin-level population lookup table."""

    if not path:
        return None
    table_path = Path(path)
    if not table_path.exists():
        dbg(f"admin population table {path} missing")
        return None
    try:
        df = pd.read_csv(table_path)
    except Exception as exc:  # pragma: no cover - defensive
        dbg(f"failed to read admin population table {path}: {exc}")
        return None
    if df.empty:
        return None
    mapping = {str(col).strip().lower(): col for col in df.columns}
    iso_col = mapping.get("iso3") or mapping.get("#country+code")
    admin_col = (
        mapping.get("admin_name")
        or mapping.get("admin")
        or mapping.get("admin_id")
        or mapping.get("admin_code")
        or mapping.get("adm1_name")
        or mapping.get("adm2_name")
    )
    pop_col = mapping.get("population") or mapping.get("pop")
    year_col = mapping.get("year")
    source_col = mapping.get("source")
    if not iso_col or not admin_col or not pop_col:
        dbg("admin population table missing required columns; ignoring")
        return None
    normalised = pd.DataFrame()
    normalised["iso3"] = df[iso_col].astype(str).str.strip().str.upper()
    normalised["admin_name"] = df[admin_col].astype(str).str.strip()
    normalised["population"] = df[pop_col].apply(_parse_float)
    if year_col:
        normalised["year"] = df[year_col].apply(_parse_float).apply(
            lambda v: int(v) if v is not None else None
        )
    else:
        normalised["year"] = None
    if source_col:
        normalised["source"] = df[source_col].astype(str).str.strip()
    else:
        normalised["source"] = ""
    normalised = normalised.dropna(subset=["iso3", "admin_name", "population"], how="any")
    normalised = normalised[normalised["population"] > 0]
    if normalised.empty:
        return None
    return normalised


def _lookup_admin_population(
    table: Optional[pd.DataFrame], iso3: str, admin_name: str, year: int
) -> Tuple[Optional[float], Optional[str]]:
    """Return an optional admin population from ``table``."""

    if table is None or not admin_name:
        return None, None
    subset = table[table["iso3"] == str(iso3).upper()]
    if subset.empty:
        return None, None
    subset_admin = subset[subset["admin_name"].str.lower() == admin_name.lower()]
    if subset_admin.empty:
        return None, None
    if "year" in subset_admin.columns and subset_admin["year"].notna().any():
        subset_year = subset_admin[subset_admin["year"].notna()].copy()
        subset_year["year"] = subset_year["year"].astype(int)
        candidates = subset_year[subset_year["year"] <= year]
        if candidates.empty:
            row = subset_year.sort_values("year").iloc[-1]
        else:
            row = candidates.sort_values("year").iloc[-1]
    else:
        row = subset_admin.iloc[-1]
    population = row.get("population")
    if population is None or pd.isna(population):
        return None, None
    source = str(row.get("source", "")).strip()
    label = "lookup population"
    if source:
        label = f"lookup population ({source})"
    return float(population), label


def _ingested_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(parts: Sequence[Any]) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        parsed = float(text)
    except Exception:
        return None
    if math.isnan(parsed):
        return None
    return float(parsed)


def _clamp_pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return min(100.0, max(0.0, numeric))


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _month_key(value: pd.Timestamp) -> Optional[str]:
    if value is None:
        return None
    return f"{value.year:04d}-{value.month:02d}"


def load_config() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_source_overrides() -> Dict[str, Any]:
    if not SOURCES_CONFIG.exists():
        return {"enabled": False, "sources": [], "auth": {}}
    with open(SOURCES_CONFIG, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        return {"enabled": False, "sources": [], "auth": {}}
    data.setdefault("enabled", False)
    sources = data.get("sources") or []
    if isinstance(sources, dict):
        sources = [dict(name=name, **(value or {})) for name, value in sources.items()]
    elif not isinstance(sources, list):
        sources = []
    data["sources"] = sources
    if not isinstance(data.get("auth"), dict):
        data["auth"] = {}
    return data


def load_countries() -> pd.DataFrame:
    df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    return df


def load_shocks() -> pd.DataFrame:
    df = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return df


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_HEADERS).to_csv(path, index=False)
    ensure_manifest_for_csv(path)


def _write_rows(rows: List[List[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CANONICAL_HEADERS)
    df.to_csv(path, index=False)
    ensure_manifest_for_csv(path)


def _maybe_apply_hxl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_row = df.iloc[0]
    if all(isinstance(v, str) and v.startswith("#") for v in first_row):
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = [str(v).strip() for v in first_row]
        return df
    if any(col.startswith("#") for col in df.columns):
        df.columns = [str(c).strip() for c in df.columns]
        return df
    if all(isinstance(v, str) and v.startswith("#") for v in df.columns):
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalise_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        mapping[lower] = col
    return mapping


def _find_column(mapping: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if not key:
            continue
        key_lower = key.strip().lower()
        if key_lower in mapping:
            return mapping[key_lower]
    return None


def _extract_text(row: pd.Series, columns: List[str]) -> str:
    parts: List[str] = []
    for col in columns:
        if col not in row:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _load_dataframe(source: Dict[str, Any]) -> Optional[pd.DataFrame]:
    url = source.get("url")
    if not url:
        return None
    kind = (source.get("kind") or "csv").lower()
    appname = os.getenv("RELIEFWEB_APPNAME")
    if url.startswith("http://") or url.startswith("https://"):
        headers = {}
        if appname:
            headers["User-Agent"] = appname
        if isinstance(source.get("headers"), dict):
            headers.update({str(k): str(v) for k, v in source["headers"].items()})
        try:
            resp = requests.get(url, headers=headers or None, timeout=60)
            if resp.status_code == 404:
                print(f"[wfp_mvam] {url} returned 404; skipping source")
                return None
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "HTTPError"
            print(f"[wfp_mvam] {url} returned {status}; skipping source")
            return None
        except requests.RequestException as exc:
            print(f"[wfp_mvam] request error for {url}: {exc}")
            return None
        data = io.BytesIO(resp.content)
        if kind == "csv":
            return pd.read_csv(data)
        if kind in {"xlsx", "xls", "excel"}:
            return pd.read_excel(data)
        if kind == "json":
            return pd.read_json(io.BytesIO(resp.content))
        return pd.read_csv(data)
    path = Path(url)
    if not path.exists():
        dbg(f"source path {url} missing")
        return None
    if kind == "csv":
        return pd.read_csv(path)
    if kind in {"xlsx", "xls", "excel"}:
        return pd.read_excel(path)
    if kind == "json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _resolve_hazard(code: str, shocks: pd.DataFrame) -> Hazard:
    match = shocks[shocks["hazard_code"].str.upper() == code.upper()]
    if match.empty:
        return Hazard(*DEFAULT_MULTI)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def _infer_hazard(texts: Iterable[str], shocks: pd.DataFrame, cfg: Dict[str, Any]) -> Hazard:
    keywords = cfg.get("shock_keywords", {})
    default_code = str(cfg.get("default_hazard", "multi"))
    sample = " ".join(str(t).lower() for t in texts if t)
    matches: List[str] = []
    for key, words in keywords.items():
        for word in words:
            if word.lower() in sample:
                matches.append(key)
                break
    unique = sorted(set(matches))
    if not unique:
        if default_code.lower() == "multi":
            return Hazard(*DEFAULT_MULTI)
        return _resolve_hazard(default_code, shocks)
    if len(unique) > 1:
        return Hazard(*DEFAULT_MULTI)
    hazard_key = unique[0]
    code = HAZARD_KEY_TO_CODE.get(hazard_key)
    if not code:
        return Hazard(*DEFAULT_MULTI)
    return _resolve_hazard(code, shocks)


def _aggregate_people(group: pd.DataFrame) -> Optional[float]:
    values = group["value"].apply(_parse_float).dropna()
    if values.empty:
        return None
    hint = group["series_hint"].dropna().iloc[0] if not group["series_hint"].dropna().empty else "stock"
    if hint == "incident":
        return float(values.sum())
    return float(values.mean())


def _aggregate_percent(group: pd.DataFrame) -> Optional[float]:
    parsed = group["value"].apply(_parse_float).dropna()
    if parsed.empty:
        return None
    clamped_values = [
        value for value in (_clamp_pct(v) for v in parsed) if value is not None
    ]
    if not clamped_values:
        return None
    mean_value = float(sum(clamped_values) / len(clamped_values))
    return _clamp_pct(mean_value)


def collect_rows() -> List[List[Any]]:
    cfg = load_config()
    templates = get_source_templates(cfg)
    dbg(
        "loaded source templates: "
        + (", ".join(sorted(templates)) if templates else "<none>")
    )
    override_cfg = load_source_overrides()

    if not override_cfg.get("enabled", False):
        print("WFP mVAM disabled via config; writing header-only CSV")
        return []

    auth_headers = override_cfg.get("auth") if isinstance(override_cfg.get("auth"), dict) else {}
    prepared_sources: List[Dict[str, Any]] = []
    overrides = override_cfg.get("sources", []) or []
    for entry in overrides:
        if isinstance(entry, str):
            if not templates:
                continue
            name = next(iter(templates))
            override = {"name": name, "url": entry}
        elif isinstance(entry, dict):
            override = dict(entry)
            name = str(override.get("name") or "").strip()
            if not name and templates:
                name = next(iter(templates))
                override["name"] = name
        else:
            continue
        name = str(override.get("name") or "").strip()
        url = str(override.get("url", "")).strip()
        if not name or not url:
            continue
        template = dict(templates.get(name) or {})
        if not template:
            template = dict(DEFAULT_SOURCE_TEMPLATES.get(name, {}))
        template.setdefault("name", name)
        template["url"] = url
        for key, value in override.items():
            if key == "url":
                continue
            template[key] = value
        if auth_headers and not template.get("headers"):
            template["headers"] = dict(auth_headers)
        prepared_sources.append(template)

    if not prepared_sources:
        print("WFP mVAM enabled but no sources configured; writing header-only CSV")
        return []

    cfg = dict(cfg)
    cfg["sources"] = prepared_sources
    countries = load_countries()
    shocks = load_shocks()

    country_lookup = {row.iso3.upper(): row.country_name for row in countries.itertuples()}

    allow_percent = _env_bool("WFP_MVAM_ALLOW_PERCENT", bool(cfg.get("allow_percent", True)))
    emit_stock = _env_bool("WFP_MVAM_STOCK", bool(cfg.get("emit_stock", True)))
    emit_incident = _env_bool("WFP_MVAM_INCIDENT", bool(cfg.get("emit_incident", True)))
    include_first_month = _env_bool(
        "WFP_MVAM_INCLUDE_FIRST_MONTH_DELTA",
        bool(cfg.get("include_first_month_delta", False)),
    )
    suppress_admin_when_no_subpop = _env_bool(
        "WFP_MVAM_SUPPRESS_ADMIN_IF_NO_SUBPOP",
        bool(cfg.get("suppress_admin_when_no_subpop", True)),
    )

    priority_env = os.getenv("WFP_MVAM_INDICATOR_PRIORITY")
    if priority_env:
        indicator_priority = [item.strip() for item in priority_env.split(",") if item.strip()]
    else:
        indicator_priority = cfg.get("indicator_priority", [])

    prefer_hxl = bool(cfg.get("prefer_hxl", False))
    denominator_override = os.getenv("WFP_MVAM_DENOMINATOR_FILE")
    denominator_cfg = cfg.get("denominator_file")
    if denominator_override:
        denominator_path = denominator_override
    elif denominator_cfg:
        denominator_path = str(denominator_cfg)
    else:
        denominator_path = str(DEFAULT_DENOMINATOR)

    worldpop_product_hint = os.getenv("WORLDPOP_PRODUCT", "").strip()
    admin_population_table = _load_admin_population_table(
        cfg.get("admin_population_table")
    )

    records: List[Dict[str, Any]] = []

    sources = cfg.get("sources", []) or []
    for source in sources:
        df = _load_dataframe(source)
        if df is None or df.empty:
            continue
        if prefer_hxl:
            df = _maybe_apply_hxl(df)
        else:
            df.columns = [str(c).strip() for c in df.columns]

        column_map = _normalise_columns(df)
        time_col = _find_column(column_map, source.get("time_keys", []))
        country_col = _find_column(column_map, source.get("country_keys", []))
        admin_cols: List[str] = []
        for key in source.get("admin_keys", []):
            col = _find_column(column_map, [key])
            if col and col not in admin_cols:
                admin_cols.append(col)
        pct_col = _find_column(column_map, source.get("pct_keys", []))
        people_col = _find_column(column_map, source.get("people_keys", []))
        population_col = _find_column(column_map, source.get("population_keys", []))
        driver_cols: List[str] = []
        for key in source.get("driver_keys", []):
            col = _find_column(column_map, [key])
            if col and col not in driver_cols:
                driver_cols.append(col)

        metric = source.get("metric", DEFAULT_METRIC)
        series_hint = source.get("series_hint", "stock")
        publisher = source.get("publisher", DEFAULT_PUBLISHER)
        source_type = source.get("source_type", DEFAULT_SOURCE_TYPE)
        source_url = source.get("url", "")
        doc_title = source.get("name") or "WFP mVAM dataset"
        source_id = source.get("name") or source_url or "wfp_mvam"

        if not time_col or not country_col:
            continue

        for _, row in df.iterrows():
            iso_raw = row.get(country_col)
            if pd.isna(iso_raw):
                continue
            iso3 = str(iso_raw).strip().upper()
            if len(iso3) != 3 or iso3 not in country_lookup:
                continue

            date_raw = row.get(time_col)
            parsed_date = _parse_date(date_raw)
            if parsed_date is None:
                continue
            month = _month_key(parsed_date)
            if not month:
                continue

            indicator_col = None
            indicator_kind = None
            indicator_name = None

            if indicator_priority:
                for candidate in indicator_priority:
                    lower = candidate.lower()
                    if "people" in lower and people_col:
                        indicator_col = people_col
                        indicator_kind = "people"
                        indicator_name = candidate
                        break
                    if "pct" in lower or "%" in lower:
                        if pct_col:
                            indicator_col = pct_col
                            indicator_kind = "percent"
                            indicator_name = candidate
                            break
                    if "rcsi" in lower:
                        rcsi_col = column_map.get("rcsi")
                        if rcsi_col:
                            indicator_col = rcsi_col
                            indicator_kind = "rcsi"
                            indicator_name = candidate
                            break
            if not indicator_col:
                if people_col is not None:
                    indicator_col = people_col
                    indicator_kind = "people"
                    indicator_name = people_col
                elif pct_col is not None:
                    indicator_col = pct_col
                    indicator_kind = "percent"
                    indicator_name = pct_col
                else:
                    continue

            raw_value = row.get(indicator_col)
            parsed_value = _parse_float(raw_value)
            if parsed_value is None:
                continue

            population_value = None
            if population_col:
                population_value = _parse_float(row.get(population_col))

            admin_parts: List[str] = []
            for col in admin_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                text_val = str(val).strip()
                if text_val:
                    admin_parts.append(text_val)
            admin_name = "|".join([p for p in admin_parts if p])

            driver_text = _extract_text(row, driver_cols)
            hazard = _infer_hazard([doc_title, driver_text], shocks, cfg)

            record = {
                "iso3": iso3,
                "country_name": country_lookup.get(iso3, ""),
                "month": month,
                "value": parsed_value,
                "indicator_kind": indicator_kind,
                "indicator_name": indicator_name,
                "series_hint": series_hint,
                "population": population_value,
                "metric": metric,
                "hazard_code": hazard.code,
                "hazard_label": hazard.label,
                "hazard_class": hazard.hazard_class,
                "publisher": publisher,
                "source_type": source_type,
                "source_url": source_url,
                "doc_title": doc_title,
                "driver_text": driver_text,
                "admin_name": admin_name,
                "source_id": source_id,
            }
            records.append(record)

    if not records:
        return []

    df_records = pd.DataFrame(records)

    key_columns = [
        "iso3",
        "hazard_code",
        "hazard_label",
        "hazard_class",
        "metric",
        "source_id",
        "month",
    ]
    grouped_keys = df_records.groupby(key_columns, dropna=False)

    national_records: List[Dict[str, Any]] = []
    has_admin_people_sum_keys: set[Tuple[str, ...]] = set()

    for key, group in grouped_keys:
        iso3, hz_code, hz_label, hz_class, metric, source_id, month = key
        month = str(month)
        if not month or "-" not in month:
            continue
        try:
            year = int(month.split("-")[0])
        except Exception:
            continue

        country_name = _select_first_text(group["country_name"], "")
        doc_title = _select_first_text(group["doc_title"], "WFP mVAM")
        source_url = _select_first_text(group["source_url"], "")
        publisher = _select_first_text(group["publisher"], DEFAULT_PUBLISHER)
        source_type = _select_first_text(group["source_type"], DEFAULT_SOURCE_TYPE)
        driver_texts = sorted({text for text in group["driver_text"].dropna() if str(text).strip()})
        driver_text = "; ".join(driver_texts)

        percent_rows = group[group["indicator_kind"] == "percent"]

        conversion_methods: set[str] = set()
        definition_notes: set[str] = set()
        denominator_sources: set[str] = set()
        people_sum = 0.0
        any_admin_convertible = False
        percent_entries: List[Dict[str, Any]] = []
        unconvertible_admins: List[str] = []

        for admin_name, admin_group in group.groupby("admin_name", dropna=False):
            admin_label = str(admin_name or "")
            people_rows = admin_group[admin_group["indicator_kind"] == "people"]
            if not people_rows.empty:
                agg_value = _aggregate_people(people_rows)
                if agg_value is not None:
                    any_admin_convertible = True
                    if agg_value > 0:
                        people_sum += float(agg_value)
                        conversion_methods.add("people direct")
                        definition_notes.add("Direct people counts")
                        dbg(
                            "admin people direct for "
                            f"{iso3} {month} {metric} {source_id} admin={admin_label or '<national>'}"
                        )
                continue

            percent_subset = admin_group[admin_group["indicator_kind"] == "percent"]
            if percent_subset.empty:
                continue
            if not allow_percent:
                continue

            pct_value = _clamp_pct(_aggregate_percent(percent_subset))
            if pct_value is None:
                continue
            if math.isnan(pct_value) or pct_value < 0:
                continue

            dataset_pops = percent_subset["population"].apply(_parse_float).dropna()
            dataset_pops = dataset_pops[dataset_pops > 0]
            population = None
            population_source = None
            if not dataset_pops.empty:
                population = float(dataset_pops.iloc[0])
                population_source = "dataset population"
            else:
                lookup_population, lookup_source = _lookup_admin_population(
                    admin_population_table,
                    iso3,
                    admin_label,
                    year,
                )
                if lookup_population is not None and lookup_population > 0:
                    population = float(lookup_population)
                    population_source = lookup_source or "lookup population"

            percent_entries.append(
                {
                    "admin_name": admin_label,
                    "pct_value": float(pct_value),
                    "population": float(population) if population else None,
                    "population_source": population_source,
                }
            )

            if population is not None and population > 0:
                any_admin_convertible = True
                pct_for_conversion = _clamp_pct(pct_value)
                if pct_for_conversion is None or pct_for_conversion < 0:
                    continue
                converted_people = pct_for_conversion / 100.0 * population
                if converted_people > 0:
                    people_sum += converted_people
                conversion_methods.add("pct→people (admin pop)")
                definition_notes.add("Percent converted with admin population")
                if population_source:
                    denominator_sources.add(population_source)
                dbg(
                    "admin percent×pop conversion for "
                    f"{iso3} {month} {metric} {source_id} admin={admin_label or '<national>'}"
                )
            else:
                name_for_log = admin_label or "<no admin>"
                unconvertible_admins.append(name_for_log)
                dbg(
                    "admin percent-only without population for "
                    f"{iso3} {month} {metric} {source_id} admin={name_for_log}"
                )

        if people_sum > 0:
            dbg(
                "national admin people sum for "
                f"{iso3} {month} {metric} {source_id} = {people_sum:.2f}"
            )
            has_admin_people_sum_keys.add(key)
            definition_note_parts = [
                "Sum of admin direct counts and admin percent×pop conversions",
            ]
            if definition_notes:
                definition_note_parts.append("; ".join(sorted(definition_notes)))
            if unconvertible_admins:
                definition_note_parts.append(
                    "Percent-only admins without population excluded from sum"
                )
            denominator_source = "; ".join(sorted(denominator_sources))
            national_records.append(
                {
                    "iso3": iso3,
                    "country_name": country_name,
                    "hazard_code": hz_code,
                    "hazard_label": hz_label,
                    "hazard_class": hz_class,
                    "metric": metric,
                    "source_id": source_id,
                    "month": month,
                    "value": people_sum,
                    "publisher": publisher,
                    "source_type": source_type,
                    "source_url": source_url,
                    "doc_title": doc_title,
                    "driver_text": driver_text,
                    "conversion_method": "national=sum(admin people)",
                    "definition_note": " | ".join(definition_note_parts),
                    "denominator_source": denominator_source,
                    "aggregation_note": "national=Σadmin",
                    "unit": DEFAULT_UNIT,
                    "series_semantics": "stock",
                }
            )
            continue

        if any_admin_convertible:
            # Convertible data existed but did not produce a positive sum (likely zeros).
            continue

        if not percent_entries:
            continue

        if not allow_percent:
            continue

        if not suppress_admin_when_no_subpop:
            dbg(
                "legacy national denominator path (suppress flag off) for "
                f"{iso3} {month} {metric} {source_id}"
            )
            record = get_population_record(iso3, year, denominator_path)
            if record is None:
                dbg(
                    f"no denominator available for {iso3} in {year}; skipping {month} legacy conversion"
                )
                continue
            product_label = record.product or worldpop_product_hint or "population"
            denominator_label = f"WorldPop {product_label}"
            if record.year < year:
                denominator_detail = f"{denominator_label} year={record.year} (fallback for {year})"
            else:
                denominator_detail = f"{denominator_label} year={record.year}"
            total_value = 0.0
            for entry in percent_entries:
                pct_value = _clamp_pct(entry["pct_value"])
                if pct_value is None or pct_value < 0:
                    continue
                converted = safe_pct_to_people(
                    pct_value,
                    iso3,
                    year,
                    denom_path=denominator_path,
                )
                if converted is None or converted <= 0:
                    continue
                total_value += float(converted)
            if total_value <= 0:
                continue
            has_admin_people_sum_keys.add(key)
            national_records.append(
                {
                    "iso3": iso3,
                    "country_name": country_name,
                    "hazard_code": hz_code,
                    "hazard_label": hz_label,
                    "hazard_class": hz_class,
                    "metric": metric,
                    "source_id": source_id,
                    "month": month,
                    "value": total_value,
                    "publisher": publisher,
                    "source_type": source_type,
                    "source_url": source_url,
                    "doc_title": doc_title,
                    "driver_text": driver_text,
                    "conversion_method": f"pct→people (denominator={denominator_detail})",
                    "definition_note": (
                        "Legacy conversion: monthly prevalence converted per admin using national denominator"
                    ),
                    "denominator_source": denominator_detail,
                    "aggregation_note": "national=sum(admin people) (legacy)",
                    "unit": DEFAULT_UNIT,
                    "series_semantics": "stock",
                }
            )
            continue

        weights: List[Tuple[float, float, Optional[str]]] = []
        for entry in percent_entries:
            weight = entry["population"]
            source_label = entry["population_source"]
            if weight is None or weight <= 0:
                weight, source_label = _lookup_admin_population(
                    admin_population_table,
                    iso3,
                    entry["admin_name"],
                    year,
                )
            if weight is None or weight <= 0:
                weights = []
                break
            pct_value = _clamp_pct(entry["pct_value"])
            if pct_value is None or pct_value < 0:
                weights = []
                break
            weights.append((pct_value, float(weight), source_label))

        if not weights:
            pct_mean = _clamp_pct(_aggregate_percent(percent_rows))
            if pct_mean is None:
                continue
            if math.isnan(pct_mean) or pct_mean < 0:
                continue
            dbg(
                "percent-only fallback (missing admin population) for "
                f"{iso3} {month} {metric} {source_id}"
            )
            national_records.append(
                {
                    "iso3": iso3,
                    "country_name": country_name,
                    "hazard_code": hz_code,
                    "hazard_label": hz_label,
                    "hazard_class": hz_class,
                    "metric": metric,
                    "source_id": source_id,
                    "month": month,
                    "value": float(pct_mean),
                    "publisher": publisher,
                    "source_type": source_type,
                    "source_url": source_url,
                    "doc_title": doc_title,
                    "driver_text": driver_text,
                    "conversion_method": "percent only (no admin population; not converted)",
                    "definition_note": (
                        f"Monthly mean prevalence {pct_mean:.2f}% reported without people conversion "
                        "due to missing admin population weights"
                    ),
                    "denominator_source": "",
                    "aggregation_note": "percent-only output",
                    "unit": "percent",
                    "series_semantics": "ratio",
                }
            )
            continue

        total_weight = sum(weight for _, weight, _ in weights)
        if total_weight <= 0:
            pct_mean = _clamp_pct(_aggregate_percent(percent_rows))
            if pct_mean is None:
                continue
            if math.isnan(pct_mean) or pct_mean < 0:
                continue
            dbg(
                "percent-only fallback (zero total weight) for "
                f"{iso3} {month} {metric} {source_id}"
            )
            national_records.append(
                {
                    "iso3": iso3,
                    "country_name": country_name,
                    "hazard_code": hz_code,
                    "hazard_label": hz_label,
                    "hazard_class": hz_class,
                    "metric": metric,
                    "source_id": source_id,
                    "month": month,
                    "value": float(pct_mean),
                    "publisher": publisher,
                    "source_type": source_type,
                    "source_url": source_url,
                    "doc_title": doc_title,
                    "driver_text": driver_text,
                    "conversion_method": "percent only (no admin population; not converted)",
                    "definition_note": (
                        f"Monthly mean prevalence {pct_mean:.2f}% reported without people conversion "
                        "due to zero admin population weight"
                    ),
                    "denominator_source": "",
                    "aggregation_note": "percent-only output",
                    "unit": "percent",
                    "series_semantics": "ratio",
                }
            )
            continue

        weighted_fraction = sum((pct / 100.0) * weight for pct, weight, _ in weights) / total_weight
        pct_weighted = weighted_fraction * 100.0
        record = get_population_record(iso3, year, denominator_path)
        if record is None:
            dbg(
                f"no national denominator for {iso3} in {year}; cannot convert weighted percent"
            )
            continue
        national_population = record.population
        converted_people = weighted_fraction * national_population
        if converted_people <= 0:
            dbg(
                f"weighted conversion produced {converted_people} for {iso3} {month}; skipping"
            )
            continue
        product_label = record.product or worldpop_product_hint or "population"
        denominator_label = f"WorldPop {product_label}"
        if record.year < year:
            denominator_detail = f"{denominator_label} year={record.year} (fallback for {year})"
        else:
            denominator_detail = f"{denominator_label} year={record.year}"
        dbg(
            "national single-conversion path (population-weighted) for "
            f"{iso3} {month} {metric} {source_id}"
        )
        national_records.append(
            {
                "iso3": iso3,
                "country_name": country_name,
                "hazard_code": hz_code,
                "hazard_label": hz_label,
                "hazard_class": hz_class,
                "metric": metric,
                "source_id": source_id,
                "month": month,
                "value": converted_people,
                "publisher": publisher,
                "source_type": source_type,
                "source_url": source_url,
                "doc_title": doc_title,
                "driver_text": driver_text,
                "conversion_method": "pct→people (national single-conversion; population-weighted)",
                "definition_note": (
                    f"Population-weighted prevalence {pct_weighted:.2f}% converted once at national level "
                    "due to missing admin population data"
                ),
                "denominator_source": denominator_detail,
                "aggregation_note": "national=weighted(pct)×pop",
                "unit": DEFAULT_UNIT,
                "series_semantics": "stock",
            }
        )

    if not national_records:
        return []

    national_records.sort(
        key=lambda r: (r["iso3"], r["hazard_code"], r["metric"], r["source_id"], r["month"])
    )

    rows: List[List[Any]] = []
    ingested_at = _ingested_timestamp()
    incident_flag = "on" if emit_incident else "off"

    method_template = "mVAM; monthly-first; {conversion}; national roll-up; incident delta=" + incident_flag
    definition_template = (
        "WFP mVAM food insecurity indicators aggregated to monthly national counts. "
        "Daily/weekly values are averaged to month prior to conversion. {note}"
    )

    if emit_stock:
        for rec in national_records:
            as_of = rec["month"]
            value_raw = rec.get("value")
            if isinstance(value_raw, (int, float)) and not isinstance(value_raw, bool):
                numeric_value = float(value_raw)
            else:
                numeric_value = _parse_float(value_raw)
            if numeric_value is None:
                continue
            unit = rec.get("unit", DEFAULT_UNIT)
            series_semantics = rec.get("series_semantics", "stock")
            if unit == DEFAULT_UNIT:
                if numeric_value <= 0:
                    continue
                value_str = str(int(round(numeric_value)))
            else:
                if numeric_value < 0:
                    continue
                value_str = f"{numeric_value:.2f}".rstrip("0").rstrip(".")
            digest = _digest(
                [rec["iso3"], rec["hazard_code"], rec["metric"], as_of, value_str, rec["source_url"]]
            )
            year, month = as_of.split("-")
            event_id = f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-{digest}"
            publication_date = f"{as_of}-01"
            conversion_method = rec.get("conversion_method") or "people direct"
            definition_parts: List[str] = []
            if rec.get("definition_note"):
                definition_parts.append(str(rec["definition_note"]).strip())
            denominator_source = rec.get("denominator_source")
            if denominator_source:
                definition_parts.append(f"Denominator source: {denominator_source}.")
            aggregation_note = rec.get("aggregation_note")
            if aggregation_note:
                definition_parts.append(f"Aggregation: {aggregation_note}.")
            definition_text = " ".join(part for part in definition_parts if part)
            rows.append(
                [
                    event_id,
                    rec.get("country_name", ""),
                    rec["iso3"],
                    rec["hazard_code"],
                    rec["hazard_label"],
                    rec["hazard_class"],
                    rec["metric"],
                    series_semantics,
                    value_str,
                    unit,
                    as_of,
                    publication_date,
                    rec["publisher"],
                    rec["source_type"],
                    rec["source_url"],
                    rec["doc_title"],
                    definition_template.format(note=definition_text),
                    method_template.format(conversion=conversion_method),
                    "",
                    0,
                    ingested_at,
                ]
            )

    if emit_incident:
        grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        for rec in national_records:
            if rec.get("unit", DEFAULT_UNIT) != DEFAULT_UNIT:
                continue
            key = (rec["iso3"], rec["hazard_code"], rec["metric"], rec["source_id"])
            grouped.setdefault(key, []).append(rec)
        for key, group in grouped.items():
            group.sort(key=lambda r: r["month"])
            prev_value: Optional[float] = None
            for idx, rec in enumerate(group):
                raw_value = rec.get("value")
                if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                    current_value = float(raw_value)
                else:
                    parsed = _parse_float(raw_value)
                    if parsed is None:
                        continue
                    current_value = float(parsed)
                if current_value <= 0:
                    continue
                current = current_value
                as_of = rec["month"]
                incident_conversion_method = rec.get("conversion_method") or "people direct"
                incident_definition_parts: List[str] = []
                if rec.get("definition_note"):
                    incident_definition_parts.append(str(rec["definition_note"]).strip())
                denominator_source = rec.get("denominator_source")
                if denominator_source:
                    incident_definition_parts.append(f"Denominator source: {denominator_source}.")
                aggregation_note = rec.get("aggregation_note")
                if aggregation_note:
                    incident_definition_parts.append(f"Aggregation: {aggregation_note}.")
                incident_definition_text = " ".join(
                    part for part in incident_definition_parts if part
                )
                if prev_value is None:
                    prev_value = current
                    if include_first_month and current > 0:
                        digest = _digest([rec["iso3"], rec["hazard_code"], rec["metric"], as_of, 0, rec["source_url"], "incident"])
                        year, month = as_of.split("-")
                        event_id = (
                            f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-incident-{digest}"
                        )
                        publication_date = f"{as_of}-01"
                        rows.append(
                            [
                                event_id,
                                rec.get("country_name", ""),
                                rec["iso3"],
                                rec["hazard_code"],
                                rec["hazard_label"],
                                rec["hazard_class"],
                                rec["metric"],
                                "incident",
                                "0",
                                DEFAULT_UNIT,
                                as_of,
                                publication_date,
                                rec["publisher"],
                                rec["source_type"],
                                rec["source_url"],
                                rec["doc_title"],
                                definition_template.format(note=incident_definition_text),
                                method_template.format(conversion=incident_conversion_method),
                                "",
                                0,
                                ingested_at,
                            ]
                        )
                    continue
                delta = current - prev_value
                prev_value = current
                if delta <= 0:
                    continue
                digest = _digest([rec["iso3"], rec["hazard_code"], rec["metric"], as_of, delta, rec["source_url"], "incident"])
                year, month = as_of.split("-")
                event_id = f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-incident-{digest}"
                publication_date = f"{as_of}-01"
                rows.append(
                    [
                        event_id,
                        rec.get("country_name", ""),
                        rec["iso3"],
                        rec["hazard_code"],
                        rec["hazard_label"],
                        rec["hazard_class"],
                        rec["metric"],
                        "incident",
                        str(int(delta)),
                        DEFAULT_UNIT,
                        as_of,
                        publication_date,
                        rec["publisher"],
                        rec["source_type"],
                        rec["source_url"],
                        rec["doc_title"],
                        definition_template.format(note=incident_definition_text),
                        method_template.format(conversion=incident_conversion_method),
                        "",
                        0,
                        ingested_at,
                    ]
                )

    return rows


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_WFP_MVAM") == "1":
        dbg("RESOLVER_SKIP_WFP_MVAM=1 — skipping WFP mVAM connector")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no WFP mVAM rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, OUT_PATH)
    print(f"wrote {OUT_PATH}")
    return True


if __name__ == "__main__":
    main()
