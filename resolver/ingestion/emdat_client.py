"""Compatibility facade for the EM-DAT connector used in Resolver tests.

This module focuses on the subset of behaviour exercised by the unit tests:

* Reading per-event CSV inputs (optionally with HXL headers).
* Allocating totals across calendar months using either proportional or
  start-of-period policies.
* Mapping hazard keywords to Resolver hazard codes.
* Writing the canonical connector CSV even when the connector is skipped.

The implementation keeps logging and error handling lightweight while
remaining side-effect free on import so tests can monkeypatch configuration
paths easily.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import (
    ensure_headers,
    linear_split,
    map_hazard,
    month_start,
    parse_date,
    stable_digest,
    to_iso3,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "emdat.yml"

COUNTRIES = DATA / "countries.csv"

OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "emdat_pa.csv"

LOG = logging.getLogger("resolver.ingestion.emdat")

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

_HAZARD_INFO: Dict[str, Tuple[str, str, str]] = {
    "flood": ("FL", "Flood", "natural"),
    "drought": ("DR", "Drought", "natural"),
    "tropical_cyclone": ("TC", "Tropical Cyclone", "natural"),
    "storm": ("TC", "Tropical Cyclone", "natural"),
    "earthquake": ("EQ", "Earthquake", "natural"),
    "conflict": ("CF", "Conflict", "conflict"),
    "volcano": ("VO", "Volcanic Eruption", "natural"),
    "wildfire": ("WF", "Wildfire", "natural"),
    "landslide": ("LS", "Landslide", "natural"),
    "phe": ("PHE", "Public Health Emergency", "health"),
    "other": ("OT", "Other", "other"),
}


def _env_bool(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def ensure_header_only() -> None:
    ensure_headers(OUT_PATH, CANONICAL_HEADERS)
    ensure_manifest_for_csv(OUT_PATH)


def _normalise_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


@dataclass
class SourceFrame:
    df: pd.DataFrame
    column_map: Dict[str, str]
    hxl_map: Dict[str, str] = field(default_factory=dict)


def _prepare_frame(df: pd.DataFrame) -> SourceFrame:
    df = df.fillna("")
    hxl_map: Dict[str, str] = {}
    if not df.empty:
        for idx in list(df.index[:2]):
            row = df.loc[idx]
            if all(str(value).strip().startswith("#") for value in row.values if str(value).strip()):
                for column, tag in zip(df.columns, row.values):
                    tag_norm = _normalise_key(tag)
                    if tag_norm:
                        hxl_map[tag_norm] = column
                df = df.drop(index=idx)
        df = df.reset_index(drop=True)
    column_map: Dict[str, str] = {}
    for column in df.columns:
        norm = _normalise_key(column)
        if norm and norm not in column_map:
            column_map[norm] = column
    return SourceFrame(df=df, column_map=column_map, hxl_map=hxl_map)


def _find_column(frame: SourceFrame, keys: Sequence[str], prefer_hxl: bool) -> Optional[str]:
    for key in keys:
        norm = _normalise_key(key)
        if not norm:
            continue
        if prefer_hxl:
            column = frame.hxl_map.get(norm)
            if column:
                return column
        column = frame.column_map.get(norm)
        if column:
            return column
        if not prefer_hxl:
            column = frame.hxl_map.get(norm)
            if column:
                return column
    return None


def _best_of(
    record: MutableMapping[str, Any],
    frame: SourceFrame,
    keys: Sequence[str],
    *,
    prefer_hxl: bool,
) -> Any:
    if not keys:
        return None
    column = _find_column(frame, keys, prefer_hxl)
    if column:
        return record.get(column)
    return None


def _parse_people(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not pd.notna(number) or number <= 0:
        return None
    return float(number)


def _normalise_date_value(value: Any) -> str:
    parsed = parse_date(value)
    if parsed:
        return parsed.isoformat()
    return ""


def _allocation(
    total: float,
    start: date,
    end: Optional[date],
    *,
    policy: str,
) -> Dict[str, float]:
    if total is None or total <= 0 or start is None:
        return {}
    if end and end < start:
        end = start
    if policy == "start":
        bucket = month_start(start) or start.replace(day=1)
        return {f"{bucket.year:04d}-{bucket.month:02d}": float(total)}
    allocations = linear_split(total, start, end)
    buckets: Dict[str, float] = {}
    for bucket, amount in allocations:
        if amount is None:
            continue
        month = month_start(bucket) or bucket.replace(day=1)
        key = f"{month.year:04d}-{month.month:02d}"
        buckets[key] = buckets.get(key, 0.0) + float(amount)
    if not buckets:
        month = month_start(start) or start.replace(day=1)
        buckets[f"{month.year:04d}-{month.month:02d}"] = float(total)
    return buckets


def _hazard_tuple(key: str) -> Tuple[str, str, str]:
    normalised = str(key or "other").strip().lower()
    if normalised in _HAZARD_INFO:
        return _HAZARD_INFO[normalised]
    return _HAZARD_INFO["other"]


def _resolve_hazard(
    type_value: Any,
    subtype_value: Any,
    *,
    shock_map: Mapping[str, Sequence[str]],
    default_key: str,
) -> str:
    combined = " ".join(str(part or "") for part in (type_value, subtype_value)).strip()
    lowered = combined.lower()
    for hazard_key, keywords in shock_map.items():
        for keyword in keywords or []:
            if keyword and str(keyword).strip().lower() in lowered:
                return str(hazard_key).strip().lower()
    canonical = (
        map_hazard(type_value)
        or map_hazard(subtype_value)
        or map_hazard(combined)
    )
    if canonical:
        return canonical
    fallback = str(default_key or "other").strip().lower()
    if fallback:
        return fallback
    return "other"


def _country_names() -> Dict[str, str]:
    if not hasattr(_country_names, "_cache"):
        df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
        _country_names._cache = {row.iso3: row.country_name for row in df.itertuples(index=False)}  # type: ignore[attr-defined]
    return getattr(_country_names, "_cache")  # type: ignore[attr-defined]


def _load_source_frame(source: Mapping[str, Any], prefer_hxl: bool) -> SourceFrame:
    if "data" in source:
        df = pd.DataFrame(source.get("data", []))
    else:
        kind = str(source.get("kind") or "csv").lower()
        url = source.get("url")
        if not url:
            return SourceFrame(df=pd.DataFrame(), column_map={})
        if kind == "xlsx":
            df = pd.read_excel(url, dtype=str)
        else:
            df = pd.read_csv(url, dtype=str)
    return _prepare_frame(df)


def _extract_metric_values(
    record: MutableMapping[str, Any],
    frame: SourceFrame,
    source: Mapping[str, Any],
    *,
    prefer_hxl: bool,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    key_map = {
        "affected": ["total_affected_keys", "affected_keys"],
        "injured": ["injured_keys"],
        "homeless": ["homeless_keys"],
    }
    for metric, groups in key_map.items():
        for group in groups:
            keys = source.get(group) or []
            value = _parse_people(_best_of(record, frame, keys, prefer_hxl=prefer_hxl))
            if value is None:
                continue
            metrics[metric] = metrics.get(metric, 0.0) + float(value)
    return metrics


def collect_rows(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sources = cfg.get("sources") or []
    if not isinstance(sources, Iterable):
        return []
    sources_list = [source for source in sources if isinstance(source, Mapping)]
    if not sources_list:
        return []

    prefer_hxl = bool(cfg.get("prefer_hxl", True))
    policy = str(os.getenv("EMDAT_ALLOC_POLICY") or cfg.get("allocation_policy") or "prorata").strip().lower()
    if policy not in {"prorata", "start"}:
        policy = "prorata"

    shock_map_cfg = cfg.get("shock_map") or {}
    shock_map: Dict[str, Sequence[str]] = {}
    if isinstance(shock_map_cfg, Mapping):
        for key, values in shock_map_cfg.items():
            if not key:
                continue
            if isinstance(values, str):
                value_list = [values]
            else:
                value_list = [str(v) for v in values or []]
            shock_map[str(key).strip().lower()] = value_list

    default_hazard = str(cfg.get("default_hazard", "other")).strip().lower() or "other"
    country_aliases = cfg.get("country_aliases") or {}
    if not isinstance(country_aliases, Mapping):
        country_aliases = {}

    country_names = _country_names()
    ingested_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    aggregates: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    raw_ids_by_key: Dict[Tuple[str, str, str, str], set[str]] = defaultdict(set)
    seen_records: set[Tuple[Any, ...]] = set()

    for source in sources_list:
        frame = _load_source_frame(source, prefer_hxl)
        if frame.df.empty:
            continue

        country_keys = source.get("country_keys") or ["iso3", "iso", "country"]
        start_keys = source.get("start_date_keys") or ["start", "start_date", "startdate"]
        end_keys = source.get("end_date_keys") or ["end", "end_date", "enddate"]
        type_keys = source.get("type_keys") or ["disastertype", "type"]
        subtype_keys = source.get("subtype_keys") or ["disastersubtype", "subtype"]
        id_keys = source.get("id_keys") or ["disno", "disasterno", "eventid"]
        title_keys = source.get("title_keys") or ["event name", "title"]
        source_url_keys = source.get("source_url_keys") or ["source_url", "source", "url"]
        publication_keys = source.get("publication_keys") or ["entry date", "publication_date"]

        publisher = str(source.get("publisher") or "CRED/EM-DAT")
        source_type = str(source.get("source_type") or "other")

        for record in frame.df.to_dict(orient="records"):
            iso_value = _best_of(record, frame, country_keys, prefer_hxl=prefer_hxl)
            iso3 = to_iso3(iso_value, country_aliases)
            if not iso3:
                continue

            start_raw = _best_of(record, frame, start_keys, prefer_hxl=prefer_hxl)
            end_raw = _best_of(record, frame, end_keys, prefer_hxl=prefer_hxl)
            start_date = parse_date(start_raw)
            end_date = parse_date(end_raw) or start_date
            if start_date is None:
                continue
            if end_date and end_date < start_date:
                end_date = start_date

            hazard_type = _best_of(record, frame, type_keys, prefer_hxl=prefer_hxl)
            hazard_subtype = _best_of(record, frame, subtype_keys, prefer_hxl=prefer_hxl)
            hazard_key = _resolve_hazard(
                hazard_type,
                hazard_subtype,
                shock_map=shock_map,
                default_key=default_hazard,
            )
            hazard_code, hazard_label, hazard_class = _hazard_tuple(hazard_key)

            raw_id = str(_best_of(record, frame, id_keys, prefer_hxl=prefer_hxl) or "").strip()
            if not raw_id:
                raw_id = stable_digest([
                    iso3,
                    hazard_code,
                    start_date.isoformat(),
                    end_date.isoformat() if end_date else "",
                ])

            metric_values = _extract_metric_values(record, frame, source, prefer_hxl=prefer_hxl)
            if not metric_values:
                continue

            record_key = (
                raw_id,
                iso3,
                hazard_key,
                start_date.isoformat(),
                end_date.isoformat() if end_date else "",
                tuple(sorted((metric, round(value, 6)) for metric, value in metric_values.items())),
            )
            if record_key in seen_records:
                continue
            seen_records.add(record_key)

            doc_title = str(
                _best_of(record, frame, title_keys, prefer_hxl=prefer_hxl)
                or source.get("name")
                or "EM-DAT event"
            )
            source_url = str(
                _best_of(record, frame, source_url_keys, prefer_hxl=prefer_hxl)
                or source.get("url")
                or ""
            )
            publication_date = _normalise_date_value(
                _best_of(record, frame, publication_keys, prefer_hxl=prefer_hxl)
            )

            for metric, value in metric_values.items():
                allocations = _allocation(value, start_date, end_date, policy=policy)
                if not allocations:
                    continue
                definition_text = (
                    f"EM-DAT reported {metric.replace('_', ' ')} persons ({policy} allocation)."
                )
                method = f"EM-DAT {policy} allocation"
                for month, amount in allocations.items():
                    if amount is None or amount <= 0:
                        continue
                    key = (iso3, hazard_code, metric, month)
                    entry = aggregates.get(key)
                    if not entry:
                        entry = {
                            "country_name": country_names.get(iso3, iso3),
                            "hazard_code": hazard_code,
                            "hazard_label": hazard_label,
                            "hazard_class": hazard_class,
                            "metric": metric,
                            "series_semantics": "incident",
                            "unit": "persons",
                            "publisher": publisher,
                            "source_type": source_type,
                            "source_url": source_url,
                            "doc_title": doc_title,
                            "definition_text": definition_text,
                            "method": method,
                            "publication_date": publication_date,
                            "value": 0.0,
                        }
                        aggregates[key] = entry
                    else:
                        if not entry["source_url"] and source_url:
                            entry["source_url"] = source_url
                        if not entry["doc_title"] and doc_title:
                            entry["doc_title"] = doc_title
                        if not entry["publication_date"] and publication_date:
                            entry["publication_date"] = publication_date
                    entry["value"] += float(amount)
                    raw_ids_by_key[key].add(raw_id)

    rows: List[Dict[str, Any]] = []
    for key, entry in sorted(aggregates.items()):
        iso3, hazard_code, metric, month = key
        value = entry.get("value", 0.0)
        if value is None or value <= 0:
            continue
        value_int = int(round(float(value)))
        digest = stable_digest([
            iso3,
            hazard_code,
            metric,
            month,
            ",".join(sorted(raw_ids_by_key.get(key, {""}))),
        ])
        event_id = f"{iso3}-EMDAT-{hazard_code}-{metric}-{month.replace('-', '')}-{digest}"
        rows.append(
            {
                "event_id": event_id,
                "country_name": entry.get("country_name", ""),
                "iso3": iso3,
                "hazard_code": hazard_code,
                "hazard_label": entry.get("hazard_label", ""),
                "hazard_class": entry.get("hazard_class", ""),
                "metric": metric,
                "series_semantics": entry.get("series_semantics", "incident"),
                "value": value_int,
                "unit": entry.get("unit", "persons"),
                "as_of_date": month,
                "publication_date": entry.get("publication_date", ""),
                "publisher": entry.get("publisher", ""),
                "source_type": entry.get("source_type", ""),
                "source_url": entry.get("source_url", ""),
                "doc_title": entry.get("doc_title", ""),
                "definition_text": entry.get("definition_text", ""),
                "method": entry.get("method", ""),
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )

    rows.sort(key=lambda row: (row.get("iso3", ""), row.get("as_of_date", ""), row.get("metric", "")))
    return rows


def _write_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        ensure_header_only()
        return
    df = pd.DataFrame(rows)
    for column in CANONICAL_HEADERS:
        if column not in df.columns:
            df[column] = ""
    df = df[CANONICAL_HEADERS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    ensure_manifest_for_csv(OUT_PATH)


def main() -> bool:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if _env_bool("RESOLVER_SKIP_EMDAT"):
        LOG.info("emdat: skipped via RESOLVER_SKIP_EMDAT")
        ensure_header_only()
        return False

    cfg = load_config()
    enabled_flag = cfg.get("enabled")
    if enabled_flag is None:
        enabled = bool(cfg.get("sources"))
    else:
        enabled = bool(enabled_flag)
    if not enabled:
        LOG.info("emdat: disabled via config; writing header only")
        ensure_header_only()
        return False

    try:
        rows = collect_rows(cfg)
    except Exception as exc:  # pragma: no cover - defensive logging for tests
        LOG.info("emdat: failed to collect rows: %s", exc)
        ensure_header_only()
        return False

    if not rows:
        LOG.info("emdat: no rows collected; writing header only")
        ensure_header_only()
        return False

    _write_rows(rows)
    LOG.info("emdat: wrote %s rows", len(rows))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
