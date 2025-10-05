#!/usr/bin/env python3
"""EM-DAT connector emitting monthly people-affected style records."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "emdat.yml"
OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "emdat_pa.csv"
OUTPUT_PATH = OUT_PATH  # backwards compatibility alias

LOG = logging.getLogger("resolver.ingestion.emdat")

COLUMNS = [
    "source",
    "hazard_type",
    "country_iso3",
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

VALUE_COLUMN_HINTS = {
    "affected": ["total affected", "affected", "no affected"],
    "deaths": ["total deaths", "deaths", "no deaths"],
    "injured": ["total injured", "injured"],
    "homeless": ["total homeless", "homeless"],
}


CANONICAL_HEADERS = COLUMNS

def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_header_only() -> None:
    ensure_headers(OUT_PATH, COLUMNS)
    ensure_manifest_for_csv(OUT_PATH, schema_version="emdat_pa.v1", source_id="emdat")


def _normalise_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _lookup_column(row: Mapping[str, Any], hints: Iterable[str]) -> Optional[str]:
    lowered = {col.lower(): col for col in row.keys()}
    for hint in hints:
        key = hint.lower()
        if key in lowered:
            return lowered[key]
    for column in row.keys():
        if _normalise_key(column) in {_normalise_key(hint) for hint in hints}:
            return column
    return None


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _build_date(row: Mapping[str, Any], prefix: str) -> Optional[date]:
    year = _parse_number(row.get(f"{prefix} Year")) or _parse_number(row.get(f"{prefix}_Year"))
    month = _parse_number(row.get(f"{prefix} Month")) or _parse_number(row.get(f"{prefix}_Month"))
    day = _parse_number(row.get(f"{prefix} Day")) or _parse_number(row.get(f"{prefix}_Day"))
    direct = parse_date(row.get(prefix))
    if direct:
        return direct
    if not year:
        return None
    try:
        y = int(year)
        m = int(month or 1)
        d = int(day or 1)
        m = max(1, min(12, m))
        d = max(1, min(28, d))
        return date(y, m, d)
    except ValueError:
        return None


def _as_iso_date(value: Any, default: str) -> str:
    parsed = parse_date(value)
    if parsed:
        return parsed.isoformat()
    return default


def _raw_event_id(row: Mapping[str, Any]) -> str:
    for key in ("Dis No", "Disaster No", "Dis No.", "DIS_NO"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _hazard_label(row: Mapping[str, Any], overrides: Mapping[str, str]) -> Optional[str]:
    for key in ("Disaster Type", "Disaster Subtype", "Disaster Group"):
        value = row.get(key)
        hazard = map_hazard(value, overrides)
        if hazard:
            return hazard
    combined = " ".join(str(row.get(key) or "") for key in ("Disaster Type", "Disaster Subtype"))
    return map_hazard(combined, overrides)


def read_rows(cfg: Mapping[str, Any]) -> List[List[Any]]:
    source_cfg = cfg.get("source") or {}
    source_type = str(source_cfg.get("type") or "file").strip().lower()
    if source_type != "file":
        raise ValueError("Only file sources are supported for EM-DAT connector")
    path = source_cfg.get("path")
    if not path:
        raise FileNotFoundError("emdat source path is empty")
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"emdat source path not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        aliases = cfg.get("country_aliases") or {}
        hazard_overrides = cfg.get("hazard_map") or {}
        value_types = cfg.get("value_types") or ["affected"]
        as_of_default = datetime.utcnow().date().isoformat()
        method = "emdat_linear_allocation"
        dedup: dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
        for row in reader:
            iso3 = to_iso3(row.get("ISO"), aliases) or to_iso3(row.get("Country"), aliases)
            if not iso3:
                continue
            hazard = _hazard_label(row, hazard_overrides)
            if not hazard:
                continue
            start = _build_date(row, "Start")
            end = _build_date(row, "End") or start
            if not start:
                start = end
            if not start:
                continue
            if end and end < start:
                end = start
            duration = (end - start).days if end else 0
            raw_id = _raw_event_id(row)
            as_of = _as_iso_date(row.get("Entry Date"), as_of_default)
            raw_json = json.dumps({
                "dis_no": raw_id,
                "country": row.get("Country"),
                "hazard_type": row.get("Disaster Type"),
            }, ensure_ascii=False)
            for value_type in value_types:
                column = _lookup_column(row, VALUE_COLUMN_HINTS.get(value_type, []))
                if not column:
                    continue
                total = _parse_number(row.get(column))
                if not total or total <= 0:
                    continue
                if duration > 14 and end:
                    allocations = linear_split(total, start, end)
                else:
                    bucket = month_start(start) or start.replace(day=1)
                    allocations = [(bucket, total)]
                for bucket, amount in allocations:
                    if amount is None:
                        continue
                    value = float(amount)
                    if value <= 0:
                        continue
                    month_iso = bucket.isoformat()
                    digest = stable_digest([iso3, hazard, month_iso, raw_id, value_type])
                    event_id = f"{iso3}-{hazard}-{bucket.strftime('%Y%m')}-{digest}"
                    record = {
                        "source": "emdat",
                        "hazard_type": hazard,
                        "country_iso3": iso3,
                        "event_id": event_id,
                        "as_of": as_of,
                        "month_start": month_iso,
                        "value_type": value_type,
                        "value": int(round(value)),
                        "unit": "people",
                        "method": method,
                        "confidence": "",
                        "raw_event_id": raw_id,
                        "raw_fields_json": raw_json,
                    }
                    key = (iso3, hazard, month_iso, raw_id, value_type)
                    existing = dedup.get(key)
                    if existing and existing["as_of"] >= record["as_of"]:
                        continue
                    dedup[key] = record
        rows = [
            [
                rec["source"],
                rec["hazard_type"],
                rec["country_iso3"],
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
            for rec in dedup.values()
        ]
        rows.sort(key=lambda row: (row[2], row[1], row[5], row[6]))
        return rows


def write_rows(rows: List[List[Any]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUT_PATH, schema_version="emdat_pa.v1", source_id="emdat")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if os.getenv("RESOLVER_SKIP_EMDAT"):
        LOG.info("emdat: skipped via RESOLVER_SKIP_EMDAT")
        ensure_header_only()
        return False
    cfg = load_config()
    if not cfg.get("enabled"):
        LOG.info("emdat: disabled via config; writing header only")
        ensure_header_only()
        return False
    rows = read_rows(cfg)
    write_rows(rows)
    LOG.info("emdat: wrote %s rows", len(rows))
    return True


if __name__ == "__main__":
    raise SystemExit(main())
