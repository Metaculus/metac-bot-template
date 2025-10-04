#!/usr/bin/env python3
"""DTM connector that converts stock or flow tables into monthly displacement flows."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import flow_from_stock, month_start, stable_digest, to_iso3

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "dtm.yml"
OUTPUT_PATH = STAGING / "dtm_displacement.csv"

LOG = logging.getLogger("resolver.ingestion.dtm")

COLUMNS = [
    "source",
    "country_iso3",
    "admin1",
    "event_id",
    "as_of",
    "month_start",
    "value_type",
    "value",
    "unit",
    "method",
    "confidence",
    "raw_event_id",
    "raw_fields_json",
]

DEFAULT_CAUSE = "unknown"


_CANDIDATE_DATE_FIELDS = [
    "as_of",
    "updated_at",
    "last_updated",
    "update_date",
    "report_date",
    "reporting_date",
    "date",
    "dtm_date",
]


def _parse_iso_date_or_none(s: str):
    # Fast path: YYYY-MM-DD or YYYY-MM
    if not s:
        return None
    s = str(s).strip()
    # Normalize common formats
    try:
        # Try YYYY-MM-DD
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
        # Try YYYY/MM/DD
        if len(s) >= 10 and s[4] in "/." and s[7] in "/.":
            return f"{s[:4]}-{s[5:7]}-{s[8:10]}"
        # Try YYYY-MM
        if len(s) >= 7 and s[4] == "-":
            return f"{s[:7]}-01"
    except Exception:
        return None
    return None


_FILENAME_DATE_REGEX = r"(20\d{2})[-_]?([01]\d)(?:[-_]?([0-3]\d))?"


def _asof_from_filename(fname: str):
    import re

    m = re.search(_FILENAME_DATE_REGEX, fname or "")
    if not m:
        return None
    yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
    if not dd:
        dd = "01"
    return f"{yyyy}-{mm}-{dd}"


def _file_mtime_iso(path: str):
    import os

    try:
        ts = os.path.getmtime(path)
        d = datetime.utcfromtimestamp(ts).date().isoformat()
        return d
    except Exception:
        return None


_AS_OF_FALLBACK_COUNTS: dict[str, int] = {"filename": 0, "mtime": 0, "run": 0}


def _extract_record_as_of(row: dict, file_ctx: dict) -> str:
    """
    Choose the best available per-record as_of, in order of preference:
      1) Any known per-row timestamp field (normalized to YYYY-MM-DD or YYYY-MM-01)
      2) File name embedded date (e.g., 2024-07-15 or 202407)
      3) File modified time (UTC date)
      4) Run date (UTC today) as last resort
    """

    # 1) row fields
    for k in _CANDIDATE_DATE_FIELDS:
        if k in row and row[k]:
            iso = _parse_iso_date_or_none(str(row[k]))
            if iso:
                return iso
    # 2) file name date
    fname_iso = _asof_from_filename(file_ctx.get("filename") or "")
    if fname_iso:
        _AS_OF_FALLBACK_COUNTS["filename"] += 1
        return fname_iso
    # 3) file mtime
    mtime_iso = _file_mtime_iso(file_ctx.get("path") or "")
    if mtime_iso:
        _AS_OF_FALLBACK_COUNTS["mtime"] += 1
        return mtime_iso
    # 4) run date fallback
    _AS_OF_FALLBACK_COUNTS["run"] += 1
    return datetime.utcnow().date().isoformat()


def _is_candidate_newer(existing_iso: str, candidate_iso: str) -> bool:
    """Return True when the candidate `as_of` timestamp is strictly newer."""

    if not candidate_iso:
        return False
    if not existing_iso:
        return True
    return candidate_iso > existing_iso


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_header_only() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def _load_csv(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _resolve_cause(row: Mapping[str, Any], cause_map: Mapping[str, str]) -> str:
    for key in ("cause", "cause_category", "reason"):
        value = row.get(key)
        if value:
            norm = str(value).strip().lower()
            mapped = cause_map.get(norm)
            if mapped:
                return mapped
            return norm
    return DEFAULT_CAUSE


def _source_label(entry: Mapping[str, Any]) -> str:
    return str(entry.get("id") or entry.get("name") or entry.get("id_or_path") or "dtm_source")


def _column(row: Mapping[str, Any], *candidates: str) -> Optional[str]:
    lowered = {col.lower(): col for col in row.keys()}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        for col in row.keys():
            if col.lower().replace(" ", "") == candidate.lower().replace(" ", ""):
                return col
    return None


def _read_source(entry: Mapping[str, Any], cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source_type = str(entry.get("type") or "file").strip().lower()
    if source_type != "file":
        raise ValueError("DTM connector currently supports file sources only")
    path = entry.get("id_or_path")
    if not path:
        raise FileNotFoundError("DTM source missing id_or_path")
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DTM source not found: {csv_path}")
    file_ctx = {"filename": csv_path.name, "path": str(csv_path)}
    aliases = cfg.get("country_aliases") or {}
    measure = str(entry.get("measure") or "stock").strip().lower()
    cause_map = {str(k).strip().lower(): str(v) for k, v in (cfg.get("cause_map") or {}).items()}
    country_column = entry.get("country_column")
    admin_column = entry.get("admin1_column")
    date_column = entry.get("date_column")
    value_column = entry.get("value_column")
    cause_column = entry.get("cause_column")
    rows = list(_load_csv(csv_path))
    if not rows:
        return []
    if not country_column:
        country_column = _column(rows[0], "country_iso3", "iso3", "country")
    if not admin_column:
        admin_column = _column(rows[0], "admin1", "adm1", "province", "state")
    if not date_column:
        date_column = _column(rows[0], "date", "month", "period")
    if not value_column:
        value_column = _column(rows[0], "value", "count", "population", "total")
    if not value_column or not date_column or not country_column:
        raise ValueError("DTM source missing required columns")
    per_admin: dict[tuple[str, str], dict[datetime, float]] = defaultdict(dict)
    per_admin_asof: dict[tuple[str, str], dict[datetime, str]] = defaultdict(dict)
    causes: dict[tuple[str, str], str] = {}
    for row in rows:
        iso = to_iso3(row.get(country_column), aliases)
        if not iso:
            continue
        bucket = month_start(row.get(date_column))
        if not bucket:
            continue
        admin1 = str(row.get(admin_column) or "").strip() if admin_column else ""
        value = _parse_float(row.get(value_column))
        if value is None or value < 0:
            continue
        per_admin[(iso, admin1)][bucket] = value
        as_of_value = _extract_record_as_of(row, file_ctx)
        existing_asof = per_admin_asof[(iso, admin1)].get(bucket)
        if not existing_asof or _is_candidate_newer(existing_asof, as_of_value):
            per_admin_asof[(iso, admin1)][bucket] = as_of_value
        if cause_column and row.get(cause_column):
            causes[(iso, admin1)] = _resolve_cause({cause_column: row.get(cause_column)}, cause_map)
        else:
            causes[(iso, admin1)] = _resolve_cause(row, cause_map)
    records: List[Dict[str, Any]] = []
    source_label = _source_label(entry)
    for key, series in per_admin.items():
        iso, admin1 = key
        if measure == "stock":
            flows = flow_from_stock(series)
        else:
            flows = {month_start(k): float(v) for k, v in series.items() if month_start(k)}
        cause = causes.get(key, DEFAULT_CAUSE)
        for bucket, value in flows.items():
            if not bucket or value is None:
                continue
            if value <= 0:
                continue
            record_as_of = per_admin_asof.get(key, {}).get(bucket) or datetime.utcnow().date().isoformat()
            records.append(
                {
                    "iso3": iso,
                    "admin1": admin1,
                    "month": bucket,
                    "value": float(value),
                    "cause": cause,
                    "measure": measure,
                    "source_id": source_label,
                    "as_of": record_as_of,
                }
            )
    return records


def build_rows(cfg: Mapping[str, Any]) -> List[List[Any]]:
    sources = cfg.get("sources") or []
    admin_mode = str(cfg.get("admin_agg") or "both").strip().lower()
    all_records: List[Dict[str, Any]] = []
    for entry in sources:
        if not isinstance(entry, Mapping):
            continue
        records = _read_source(entry, cfg)
        all_records.extend(records)
    if not all_records:
        _log_as_of_fallbacks()
        return []
    run_date = datetime.utcnow().date().isoformat()
    method = "dtm_stock_to_flow" if any(rec.get("measure") == "stock" for rec in all_records) else "dtm_flow"
    dedup: dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    country_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    country_as_of: dict[tuple[str, str, str], str] = {}
    for rec in all_records:
        iso3 = rec["iso3"]
        admin1 = rec.get("admin1") or ""
        month = rec["month"]
        month_iso = month.isoformat()
        value = float(rec.get("value", 0.0))
        record_as_of = rec.get("as_of") or run_date
        if admin_mode in {"admin1", "both"} and admin1:
            key = (iso3, admin1, month_iso, rec["source_id"])
            event_id = f"{iso3}-displacement-{month.strftime('%Y%m')}-{stable_digest(key)}"
            record = {
                "source": "dtm",
                "country_iso3": iso3,
                "admin1": admin1,
                "event_id": event_id,
                "as_of": record_as_of,
                "month_start": month_iso,
                "value_type": "new_displaced",
                "value": int(round(value)),
                "unit": "people",
                "method": method,
                "confidence": rec.get("cause", DEFAULT_CAUSE),
                "raw_event_id": f"{rec['source_id']}::{admin1 or 'national'}::{month.strftime('%Y%m')}",
                "raw_fields_json": json.dumps(
                    {
                        "source_id": rec["source_id"],
                        "admin1": admin1,
                        "cause": rec.get("cause", DEFAULT_CAUSE),
                        "measure": rec.get("measure"),
                    },
                    ensure_ascii=False,
                ),
            }
            existing = dedup.get(key)
            if existing and not _is_candidate_newer(existing["as_of"], record["as_of"]):
                continue
            dedup[key] = record
        if admin_mode in {"country", "both"}:
            country_key = (iso3, month_iso, rec["source_id"])
            country_totals[country_key] += value
            existing_country_asof = country_as_of.get(country_key)
            if not existing_country_asof or _is_candidate_newer(existing_country_asof, record_as_of):
                country_as_of[country_key] = record_as_of
    rows = list(dedup.values())
    if admin_mode in {"country", "both"}:
        for (iso3, month_iso, source_id), total in country_totals.items():
            if total <= 0:
                continue
            month = datetime.strptime(month_iso, "%Y-%m-%d").date()
            event_id = f"{iso3}-displacement-{month.strftime('%Y%m')}-{stable_digest([iso3, month_iso, source_id])}"
            rows.append(
                {
                    "source": "dtm",
                    "country_iso3": iso3,
                    "admin1": "",
                    "event_id": event_id,
                    "as_of": country_as_of.get((iso3, month_iso, source_id), run_date),
                    "month_start": month_iso,
                    "value_type": "new_displaced",
                    "value": int(round(total)),
                    "unit": "people",
                    "method": method,
                    "confidence": DEFAULT_CAUSE,
                    "raw_event_id": f"{source_id}::country::{month.strftime('%Y%m')}",
                    "raw_fields_json": json.dumps(
                        {
                            "source_id": source_id,
                            "aggregation": "country",
                            "total_value": total,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
    formatted = [
        [
            rec["source"],
            rec["country_iso3"],
            rec.get("admin1", ""),
            rec["event_id"],
            rec["as_of"],
            rec["month_start"],
            rec["value_type"],
            rec["value"],
            rec["unit"],
            rec["method"],
            rec["confidence"],
            rec["raw_event_id"],
            rec["raw_fields_json"],
        ]
        for rec in rows
    ]
    formatted.sort(key=lambda row: (row[1], row[2], row[5], row[3]))
    _log_as_of_fallbacks()
    return formatted


def _log_as_of_fallbacks() -> None:
    filename_count = _AS_OF_FALLBACK_COUNTS.get("filename", 0)
    if filename_count:
        LOG.info("dtm: as_of from filename for %s records", filename_count)
    mtime_count = _AS_OF_FALLBACK_COUNTS.get("mtime", 0)
    if mtime_count:
        LOG.info("dtm: as_of from file mtime for %s records", mtime_count)
    run_count = _AS_OF_FALLBACK_COUNTS.get("run", 0)
    if run_count:
        LOG.warning("dtm: as_of from run date fallback for %s records", run_count)
    for key in _AS_OF_FALLBACK_COUNTS:
        _AS_OF_FALLBACK_COUNTS[key] = 0


def write_rows(rows: List[List[Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config()
    if not cfg.get("enabled"):
        LOG.info("dtm: disabled via config; writing header only")
        ensure_header_only()
        return 0
    rows = build_rows(cfg)
    write_rows(rows)
    LOG.info("dtm: wrote %s rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
