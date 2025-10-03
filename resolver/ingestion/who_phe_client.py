#!/usr/bin/env python3
"""WHO Public Health Emergency (PHE) connector."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "who_phe.yml"
COUNTRIES = DATA / "countries.csv"

OUT_PATH = STAGING / "who_phe.csv"

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

SERIES_INCIDENT = "incident"
SERIES_CUMULATIVE = "cumulative"

HAZARD_CODE = "PHE"
HAZARD_LABEL = "Public Health Emergency"
HAZARD_CLASS = "health"

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"

DEFAULT_SOURCE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "cholera_global": {
        "name": "cholera_global",
        "kind": "csv",
        "time_keys": ["week", "year"],
        "country_keys": ["iso3", "country", "country_code", "#country+code"],
        "case_keys": ["cases", "new_cases", "#affected+cases"],
        "disease": "cholera",
        "series_hint": SERIES_INCIDENT,
    },
    "measles_global": {
        "name": "measles_global",
        "kind": "csv",
        "time_keys": ["week", "year"],
        "country_keys": ["iso3", "country", "country_code", "#country+code"],
        "case_keys": ["cases", "new_cases", "#affected+cases"],
        "disease": "measles",
        "series_hint": SERIES_INCIDENT,
    },
}


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[who_phe] {message}")


@dataclass
class SourceFrame:
    df: pd.DataFrame
    column_map: Dict[str, str]
    hxl_map: Dict[str, str]
    hxl_present: bool


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Dict[str, Any]:
    raw: Dict[str, Any] = {"enabled": False, "sources": {}, "auth": {}}
    if CONFIG.exists():
        with open(CONFIG, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp) or {}
        if isinstance(loaded, dict):
            raw.update(loaded)

    enabled = bool(raw.get("enabled", False))
    auth_headers = raw.get("auth") if isinstance(raw.get("auth"), dict) else {}

    configured_sources = raw.get("sources", {})
    if isinstance(configured_sources, list):
        merged: Dict[str, Any] = {}
        for item in configured_sources:
            if isinstance(item, dict) and item.get("name"):
                merged[str(item["name"])]=item
        configured_sources = merged
    elif not isinstance(configured_sources, dict):
        configured_sources = {}

    sources: List[Dict[str, Any]] = []
    for key, template in DEFAULT_SOURCE_TEMPLATES.items():
        override = configured_sources.get(key)
        if not override:
            continue
        url = ""
        if isinstance(override, str):
            url = override
            override = {}
        elif isinstance(override, dict):
            url = str(override.get("url", ""))
        if not url:
            continue
        source_cfg = dict(template)
        source_cfg["url"] = url
        if isinstance(override, dict):
            for ok, ov in override.items():
                if ok == "url":
                    continue
                source_cfg[ok] = ov
        if auth_headers:
            source_cfg.setdefault("headers", dict(auth_headers))
        sources.append(source_cfg)

    result: Dict[str, Any] = {
        "enabled": enabled,
        "sources": sources if enabled else [],
        "auth": auth_headers,
        "prefer_hxl": raw.get("prefer_hxl", True),
        "monthly_first": raw.get("monthly_first", True),
        "allow_first_month_delta": raw.get("allow_first_month_delta", False),
    }
    return result


def load_countries() -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    iso3_to_name = {row.iso3: row.country_name for row in countries.itertuples(index=False)}
    name_to_iso3: Dict[str, str] = {}
    for row in countries.itertuples(index=False):
        key = _normalise_country(row.country_name)
        if key and key not in name_to_iso3:
            name_to_iso3[key] = row.iso3
    return countries, iso3_to_name, name_to_iso3


def _normalise_country(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").strip().lower())


def _normalise_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").strip().lower())


def _is_hxl_row(values: Iterable[Any]) -> bool:
    seen = False
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        text = str(value).strip()
        if not text:
            return False
        if not text.startswith("#"):
            return False
        seen = True
    return seen


def _prepare_frame(df: pd.DataFrame) -> SourceFrame:
    hxl_map: Dict[str, str] = {}
    hxl_present = False
    if not df.empty:
        for idx in list(df.index[:2]):
            row = df.loc[idx]
            if _is_hxl_row(row.values):
                hxl_present = True
                tags = [str(v).strip() for v in row.values]
                for col, tag in zip(df.columns, tags):
                    if tag:
                        hxl_map[col] = tag
                df = df.drop(index=idx)
    df = df.reset_index(drop=True)
    column_map: Dict[str, str] = {}
    for col in df.columns:
        norm = _normalise_key(col)
        if norm and norm not in column_map:
            column_map[norm] = col
        tag = hxl_map.get(col)
        if tag:
            norm_tag = _normalise_key(tag)
            if norm_tag and norm_tag not in column_map:
                column_map[norm_tag] = col
    return SourceFrame(df=df, column_map=column_map, hxl_map=hxl_map, hxl_present=hxl_present)


def _find_column(frame: SourceFrame, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        norm = _normalise_key(cand)
        if not norm:
            continue
        if norm in frame.column_map:
            return frame.column_map[norm]
    return None


def _parse_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"na", "nan", "none"}:
        return None
    try:
        parsed = float(text.replace(",", ""))
    except Exception:
        try:
            parsed = float(value)
        except Exception:
            return None
    if pd.isna(parsed) or not math.isfinite(parsed):
        return None
    return parsed


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed
    try:
        return pd.Timestamp(parsed)
    except Exception:
        return None


def _parse_week(year_val: Any, week_val: Any) -> Optional[pd.Timestamp]:
    try:
        year = int(float(year_val))
        week = int(float(week_val))
    except Exception:
        return None
    if year <= 0 or week <= 0:
        return None
    try:
        return pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u")
    except Exception:
        return None


def _parse_month_value(value: Any, year_val: Any = None) -> Optional[pd.Timestamp]:
    text = str(value or "").strip()
    if not text and year_val is None:
        return None
    if year_val is not None and text:
        try:
            year = int(float(year_val))
        except Exception:
            year = None
        try:
            month = int(float(text))
        except Exception:
            month = None
        if year and month and 1 <= month <= 12:
            return pd.Timestamp(dt.date(year, month, 1))
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _month_key(ts: pd.Timestamp) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _detect_series_type(source: Dict[str, Any], case_col: Optional[str], frame: SourceFrame) -> str:
    hint = str(source.get("series_hint", "infer") or "").strip().lower()
    if hint in {SERIES_INCIDENT, SERIES_CUMULATIVE}:
        return hint
    texts: List[str] = []
    if case_col:
        texts.append(case_col)
        tag = frame.hxl_map.get(case_col)
        if tag:
            texts.append(tag)
    for key in (source.get("case_keys") or []):
        texts.append(str(key))
    blob = " ".join(t.lower() for t in texts if t)
    if any(token in blob for token in ["new", "incident", "weekly", "wk"]):
        return SERIES_INCIDENT
    if any(token in blob for token in ["cumulative", "total", "overall", "cum"]):
        return SERIES_CUMULATIVE
    return SERIES_INCIDENT


def _detect_time_shape(time_keys: Sequence[str]) -> str:
    keys = " ".join(str(k).lower() for k in time_keys)
    if "week" in keys:
        return "weekly"
    if "date" in keys or "day" in keys:
        return "daily"
    if "month" in keys:
        return "monthly"
    return "unknown"


def _load_remote(url: str, *, headers: Optional[Dict[str, str]] = None) -> io.BytesIO:
    merged_headers: Dict[str, str] = {}
    if headers:
        merged_headers.update({k: str(v) for k, v in headers.items()})
    ua = os.getenv("RELIEFWEB_APPNAME")
    if ua and "User-Agent" not in merged_headers:
        merged_headers["User-Agent"] = ua
    resp = requests.get(url, headers=merged_headers or None, timeout=60)
    resp.raise_for_status()
    return io.BytesIO(resp.content)


def _load_frame(source: Dict[str, Any]) -> SourceFrame:
    url = str(source.get("url", "")).strip()
    kind = str(source.get("kind", "csv") or "csv").lower()
    if not url:
        raise ValueError("source missing url")
    buf: io.BytesIO | Path
    if url.startswith("http://") or url.startswith("https://"):
        dbg(f"fetching {url}")
        buf = _load_remote(url, headers=source.get("headers"))
    else:
        path = Path(url)
        if not path.is_absolute():
            path = ROOT / url
        if not path.exists():
            raise FileNotFoundError(f"source path not found: {path}")
        buf = path
    if kind == "xlsx" or kind == "xls":
        df = pd.read_excel(buf, dtype=str)
    else:
        df = pd.read_csv(buf, dtype=str)
    return _prepare_frame(df)


@dataclass
class ParsedRow:
    iso3: str
    timestamp: pd.Timestamp
    month: str
    value: float


def _standardise_iso3(
    row: MutableMapping[str, Any],
    *,
    iso_col: Optional[str],
    country_col: Optional[str],
    name_to_iso3: Dict[str, str],
) -> Optional[str]:
    if iso_col:
        raw = row.get(iso_col)
        text = str(raw or "").strip()
        if text:
            iso = text.upper()
            if len(iso) == 3:
                return iso
            key = _normalise_country(text)
            if key and key in name_to_iso3:
                return name_to_iso3[key]
    if country_col:
        raw = row.get(country_col)
        key = _normalise_country(raw)
        if key and key in name_to_iso3:
            return name_to_iso3[key]
    return None


def _parse_rows_for_source(
    source: Dict[str, Any],
    frame: SourceFrame,
    name_to_iso3: Dict[str, str],
) -> Tuple[List[ParsedRow], str]:
    time_keys: Sequence[str] = source.get("time_keys", []) or []
    country_keys: Sequence[str] = source.get("country_keys", []) or []
    case_keys: Sequence[str] = source.get("case_keys", []) or []

    iso_candidates = [k for k in country_keys if any(token in str(k).lower() for token in ("iso", "code"))]
    name_candidates = [k for k in country_keys if k not in iso_candidates]

    iso_col = _find_column(frame, iso_candidates or country_keys)
    country_col = _find_column(frame, name_candidates)
    if country_col is None and iso_col is not None:
        country_col = iso_col
    month_col = _find_column(frame, [k for k in time_keys if "month" in str(k).lower()])
    date_col = _find_column(frame, [k for k in time_keys if "date" in str(k).lower() or "day" in str(k).lower()])
    week_col = _find_column(frame, [k for k in time_keys if "week" in str(k).lower()])
    year_col = _find_column(frame, [k for k in time_keys if "year" in str(k).lower()])

    case_col = _find_column(frame, case_keys)
    if case_col is None:
        raise ValueError("no case column found")

    series_type = _detect_series_type(source, case_col, frame)
    time_shape = _detect_time_shape(time_keys)

    parsed_rows: List[ParsedRow] = []

    for record in frame.df.to_dict(orient="records"):
        iso3 = _standardise_iso3(record, iso_col=iso_col, country_col=country_col, name_to_iso3=name_to_iso3)
        if not iso3:
            continue
        if iso3 == "" or iso3.upper() == "UNK":
            continue

        timestamp: Optional[pd.Timestamp] = None
        if date_col:
            timestamp = _parse_date(record.get(date_col))
        if timestamp is None and month_col:
            timestamp = _parse_month_value(record.get(month_col), record.get(year_col))
        if timestamp is None and week_col and year_col:
            timestamp = _parse_week(record.get(year_col), record.get(week_col))
        if timestamp is None and date_col is None and week_col and "week" not in str(time_keys).lower():
            timestamp = _parse_week(record.get(year_col), record.get(week_col))
        if timestamp is None:
            continue

        month = _month_key(timestamp)
        value = _parse_value(record.get(case_col))
        if value is None:
            continue

        parsed_rows.append(ParsedRow(iso3=iso3, timestamp=timestamp, month=month, value=value))

    return parsed_rows, series_type


def _aggregate_monthly(
    rows: Sequence[ParsedRow],
    *,
    series_type: str,
    allow_first_month: bool,
) -> Dict[str, float]:
    if not rows:
        return {}
    df = pd.DataFrame([{"iso3": r.iso3, "timestamp": r.timestamp, "month": r.month, "value": r.value} for r in rows])
    df = df.sort_values("timestamp")
    totals: Dict[str, float] = {}
    if series_type == SERIES_CUMULATIVE:
        monthly_last = df.groupby("month")[["timestamp", "value"]].last()
        months_sorted = sorted(monthly_last.index, key=lambda m: pd.Period(m, freq="M"))
        prev: Optional[float] = None
        for month in months_sorted:
            value = float(monthly_last.loc[month, "value"])
            if prev is None:
                if allow_first_month:
                    totals[month] = max(value, 0.0)
                prev = value
                continue
            delta = value - prev
            if delta < 0:
                delta = 0.0
            totals[month] = delta
            prev = value
    else:
        monthly_sum = df.groupby("month")["value"].sum()
        for month, value in monthly_sum.items():
            totals[month] = max(float(value), 0.0)
    return totals


def _round_persons(value: float) -> int:
    if value is None:
        return 0
    try:
        numeric = float(value)
    except Exception:
        return 0
    if pd.isna(numeric) or not math.isfinite(numeric):
        return 0
    return int(round(numeric))


def build_event_id(
    iso3: str,
    metric: str,
    as_of_date: str,
    value: Any,
    source_url: str,
    phe_type: str,
) -> str:
    try:
        year, month = as_of_date.split("-")[:2]
    except ValueError:
        year, month = "0000", "00"
    digest = hashlib.sha1(
        "|".join([
            str(iso3 or "UNK"),
            str(metric or ""),
            str(as_of_date or ""),
            str(value or 0),
            str(source_url or ""),
            str(phe_type or ""),
        ]).encode("utf-8")
    ).hexdigest()[:12]
    return f"{iso3}-WHO-phe-{metric}-{year}-{month}-{digest}"


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_HEADERS).to_csv(path, index=False)


def _write_rows(rows: Sequence[Dict[str, Any]], *, path: Path) -> None:
    if not rows:
        _write_header_only(path)
        return
    df = pd.DataFrame(rows)
    for col in CANONICAL_HEADERS:
        if col not in df.columns:
            df[col] = ""
    df = df[CANONICAL_HEADERS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def collect_rows() -> List[Dict[str, Any]]:
    cfg = load_config()
    if not cfg.get("enabled", False):
        print("WHO-PHE disabled via config; writing header-only CSV")
        return []

    sources = cfg.get("sources", []) or []
    if not sources:
        print("WHO-PHE enabled but no sources configured; writing header-only CSV")
        return []

    _, iso3_to_name, name_to_iso3 = load_countries()

    allow_first_month = _env_bool(
        "WHO_PHE_ALLOW_FIRST_MONTH",
        bool(cfg.get("allow_first_month_delta", False)),
    )

    pin_ratio_env = os.getenv("WHO_PHE_PIN_RATIO")
    pin_ratio: Optional[float] = None
    if pin_ratio_env:
        try:
            pin_ratio = float(pin_ratio_env)
        except Exception:
            pin_ratio = None
        if pin_ratio is not None and pin_ratio <= 0:
            pin_ratio = None

    ingested_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows_out: List[Dict[str, Any]] = []

    max_results: Optional[int] = None
    max_env = os.getenv("RESOLVER_MAX_RESULTS")
    if max_env:
        try:
            max_results = max(0, int(max_env))
        except Exception:
            max_results = None

    for source in sources:
        name = str(source.get("name") or "who_phe").strip()
        disease = str(source.get("disease") or "PHE").strip()
        source_url = str(source.get("url", ""))
        publication_date = str(source.get("publication_date", ""))
        doc_title = source.get("doc_title") or f"WHO PHE {disease} surveillance"

        try:
            frame = _load_frame(source)
        except requests.HTTPError as exc:
            print(f"[who_phe] source {name} returned {exc.response.status_code if exc.response else 'HTTPError'}; skipping")
            continue
        except requests.RequestException as exc:
            print(f"[who_phe] source {name} request failed: {exc}")
            continue
        except Exception as exc:
            dbg(f"source {name} failed to load: {exc}")
            continue

        try:
            parsed_rows, series_type = _parse_rows_for_source(source, frame, name_to_iso3)
        except Exception as exc:
            dbg(f"source {name} parse error: {exc}")
            continue

        if not parsed_rows:
            dbg(f"source {name} produced no usable rows")
            continue

        monthly_totals_by_iso: Dict[str, Dict[str, float]] = {}
        for iso3, group in _group_rows_by_iso(parsed_rows).items():
            totals = _aggregate_monthly(group, series_type=series_type, allow_first_month=allow_first_month)
            if not totals:
                continue
            monthly_totals_by_iso[iso3] = totals

        if not monthly_totals_by_iso:
            continue

        time_shape = _detect_time_shape(source.get("time_keys", []) or [])
        method_parts = ["WHO PHE", "monthly-first"]
        if time_shape == "weekly":
            method_parts.append("weekly→monthly sum")
        elif time_shape == "daily":
            method_parts.append("daily→monthly sum")
        elif time_shape == "monthly":
            method_parts.append("monthly direct")
        if series_type == SERIES_CUMULATIVE:
            method_parts.append("delta-on-cumulative")
        else:
            method_parts.append("incident")
        method = "; ".join(method_parts)

        definition_bits = [f"{disease} cases"]
        if series_type == SERIES_CUMULATIVE:
            definition_bits.append("cumulative series converted to incident deltas")
        else:
            definition_bits.append("incident series")
        if frame.hxl_present:
            definition_bits.append("HXL tags detected")
        definition_text = "; ".join(definition_bits)

        for iso3, totals in monthly_totals_by_iso.items():
            country_name = iso3_to_name.get(iso3, "")
            if not country_name:
                continue
            for month, value in sorted(totals.items()):
                persons = _round_persons(value)
                if persons < 0:
                    persons = 0
                row_base = {
                    "event_id": build_event_id(iso3, "affected", month, persons, source_url, disease),
                    "country_name": country_name,
                    "iso3": iso3,
                    "hazard_code": HAZARD_CODE,
                    "hazard_label": HAZARD_LABEL,
                    "hazard_class": HAZARD_CLASS,
                    "metric": "affected",
                    "series_semantics": SERIES_INCIDENT,
                    "value": persons,
                    "unit": "persons",
                    "as_of_date": month,
                    "publication_date": publication_date,
                    "publisher": "WHO",
                    "source_type": "official",
                    "source_url": source_url,
                    "doc_title": doc_title,
                    "definition_text": definition_text,
                    "method": method,
                    "confidence": "",
                    "revision": 0,
                    "ingested_at": ingested_at,
                }
                rows_out.append(row_base)

                if pin_ratio is not None and pin_ratio > 0:
                    pin_value = _round_persons(persons * pin_ratio)
                    pin_row = dict(row_base)
                    pin_row["metric"] = "in_need"
                    pin_row["value"] = pin_value
                    pin_row["event_id"] = build_event_id(
                        iso3,
                        "in_need",
                        month,
                        pin_value,
                        source_url,
                        disease,
                    )
                    rows_out.append(pin_row)

        if max_results is not None and len(rows_out) >= max_results:
            rows_out = rows_out[:max_results]
            break

    rows_out.sort(key=lambda r: (r.get("iso3", ""), r.get("as_of_date", ""), r.get("metric", "")))
    return rows_out


def _group_rows_by_iso(rows: Sequence[ParsedRow]) -> Dict[str, List[ParsedRow]]:
    grouped: Dict[str, List[ParsedRow]] = {}
    for row in rows:
        grouped.setdefault(row.iso3, []).append(row)
    return grouped


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_WHO") == "1":
        dbg("RESOLVER_SKIP_WHO=1 — skipping WHO connector")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no WHO PHE rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, path=OUT_PATH)
    print(f"wrote {OUT_PATH}")
    return True


if __name__ == "__main__":
    main()
