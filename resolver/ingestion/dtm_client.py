#!/usr/bin/env python3
"""DTM connector that converts stock or flow tables into monthly displacement flows."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import (
    ensure_headers,
    flow_from_stock,
    month_start,
    stable_digest,
    to_iso3,
)

if TYPE_CHECKING:  # pragma: no cover - import guard for typing only
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "dtm.yml"
OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "dtm_displacement.csv"
OUTPUT_PATH = OUT_PATH  # backwards compatibility alias

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

CANONICAL_HEADERS = COLUMNS


DATA_PATH = ROOT / "data"
COUNTRIES_PATH = DATA_PATH / "countries.csv"
SHOCKS_PATH = DATA_PATH / "shocks.csv"

SERIES_INCIDENT = "incident"
SERIES_CUMULATIVE = "cumulative"

HAZARD_KEY_TO_CODE = {
    "flood": "FL",
    "drought": "DR",
    "tropical_cyclone": "TC",
    "heat_wave": "HW",
    "armed_conflict_onset": "ACO",
    "armed_conflict_escalation": "ACE",
    "armed_conflict_cessation": "ACC",
    "civil_unrest": "CU",
    "displacement_influx": "DI",
    "economic_crisis": "EC",
    "phe": "PHE",
}

MULTI_HAZARD = ("multi", "Multi-shock Displacement/Needs", "all")


@dataclass(frozen=True)
class Hazard:
    """Lightweight hazard tuple for legacy helpers."""

    code: str
    label: str
    hclass: str


__all__ = [
    "SERIES_INCIDENT",
    "SERIES_CUMULATIVE",
    "Hazard",
    "load_registries",
    "infer_hazard",
    "rollup_subnational",
    "compute_monthly_deltas",
    "load_config",
    "ensure_header_only",
    "build_rows",
    "write_rows",
    "main",
]


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean feature flags from the environment."""

    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "y", "yes", "on"}


def _normalize_month(value: object) -> Optional[str]:
    """Return ``YYYY-MM`` strings when ``value`` parses as a date."""

    bucket = month_start(value)
    if not bucket:
        return None
    return bucket.strftime("%Y-%m")


def _is_subnational(record: Mapping[str, Any]) -> bool:
    for key in ("admin1", "admin2", "admin_pcode", "admin_name"):
        if str(record.get(key) or "").strip():
            return True
    return False


def load_registries() -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return the canonical country and shock registries used by legacy tests."""

    import pandas as pd  # local import to keep module import-side-effect free

    countries = pd.read_csv(COUNTRIES_PATH, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS_PATH, dtype=str).fillna("")
    return countries, shocks


def _hazard_from_code(code: str, shocks: "pd.DataFrame") -> Hazard:
    if not code:
        return Hazard(*MULTI_HAZARD)
    if str(code).strip().lower() == "multi":
        return Hazard(*MULTI_HAZARD)
    match = shocks[shocks["hazard_code"].str.upper() == str(code).strip().upper()]
    if match.empty:
        return Hazard(*MULTI_HAZARD)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def infer_hazard(
    texts: Iterable[str],
    shocks: Optional["pd.DataFrame"] = None,
    keywords_cfg: Optional[Mapping[str, Iterable[str]]] = None,
    *,
    default_key: Optional[str] = None,
) -> Hazard:
    """Map dataset text snippets to a hazard triple.

    This mirrors the legacy helper relied on by resolver tests and keeps the
    behaviour stable for CI while the production connector evolves.
    """

    if shocks is None:
        _, shocks = load_registries()
    if keywords_cfg is None:
        keywords_cfg = load_config().get("shock_keywords", {})
    if default_key is None:
        default_key = os.getenv(
            "DTM_DEFAULT_HAZARD",
            load_config().get("default_hazard", "displacement_influx"),
        )

    text_blob = " ".join(str(t).lower() for t in texts if t)
    matches: list[str] = []
    for key, keywords in (keywords_cfg or {}).items():
        for kw in keywords:
            if str(kw).lower() in text_blob:
                matches.append(str(key).strip().lower())
                break

    if not matches and shocks is not None:
        for _, row in shocks.iterrows():
            label = str(row.get("hazard_label", "")).strip().lower()
            if label and label in text_blob:
                matches.append(label)

    unique = sorted({m for m in matches if m})
    if not unique:
        mapped = HAZARD_KEY_TO_CODE.get(str(default_key or "").strip().lower())
        if not mapped:
            return Hazard(*MULTI_HAZARD)
        return _hazard_from_code(mapped, shocks)
    if len(unique) > 1:
        return Hazard(*MULTI_HAZARD)

    mapped = HAZARD_KEY_TO_CODE.get(unique[0], unique[0])
    return _hazard_from_code(mapped, shocks)


def rollup_subnational(
    records: Sequence[MutableMapping[str, Any]]
) -> List[MutableMapping[str, Any]]:
    """Aggregate subnational rows into national totals per month and source."""

    grouped: Dict[
        Tuple[str, str, str, str, str, str], List[MutableMapping[str, Any]]
    ] = defaultdict(list)
    for rec in records:
        as_of = _normalize_month(rec.get("as_of_date")) or ""
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            as_of,
            str(rec.get("source_id", "")),
            str(rec.get("series_type", SERIES_INCIDENT)),
        )
        rec_copy = dict(rec)
        rec_copy["as_of_date"] = as_of
        grouped[key].append(rec_copy)

    rolled: List[MutableMapping[str, Any]] = []
    for key, rows in grouped.items():
        nationals = [r for r in rows if not _is_subnational(r)]
        if nationals:
            nationals.sort(key=lambda r: r.get("as_of_date", ""))
            rolled.extend(nationals)
            continue
        total = 0.0
        template = dict(rows[0])
        for row in rows:
            try:
                total += float(row.get("value", 0) or 0)
            except Exception:
                continue
        template["value"] = max(total, 0.0)
        for drop_key in ("admin1", "admin2", "admin_pcode", "admin_name"):
            template.pop(drop_key, None)
        rolled.append(template)

    rolled.sort(
        key=lambda r: (
            str(r.get("iso3", "")),
            str(r.get("hazard_code", "")),
            str(r.get("metric", "")),
            str(r.get("as_of_date", "")),
        )
    )
    return rolled


def compute_monthly_deltas(
    records: Sequence[MutableMapping[str, Any]],
    *,
    allow_first_month: Optional[bool] = None,
) -> List[MutableMapping[str, Any]]:
    """Convert cumulative series to month-over-month deltas.

    Incident series are passed through, cumulative series become non-negative
    monthly flows. This mirrors the legacy helper that powers resolver tests.
    """

    if allow_first_month is None:
        cfg = load_config()
        allow_first_month = _env_bool(
            "DTM_ALLOW_FIRST_MONTH",
            bool(cfg.get("allow_first_month_delta", False)),
        )

    grouped: Dict[Tuple[str, str, str, str], List[MutableMapping[str, Any]]] = defaultdict(list)
    for rec in records:
        as_of = _normalize_month(rec.get("as_of_date")) or ""
        rec_copy = dict(rec)
        rec_copy["as_of_date"] = as_of
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            str(rec.get("source_id", "")),
        )
        grouped[key].append(rec_copy)

    output: List[MutableMapping[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r.get("as_of_date", ""))
        series_type = str(rows[0].get("series_type", SERIES_INCIDENT)).strip().lower()
        prev_value: Optional[float] = None
        for row in rows:
            value = row.get("value", 0)
            try:
                value_num = float(value)
            except Exception:
                value_num = 0.0
            if series_type != SERIES_CUMULATIVE:
                new_val = max(value_num, 0.0)
            else:
                if prev_value is None:
                    if allow_first_month:
                        new_val = max(value_num, 0.0)
                        prev_value = value_num
                    else:
                        prev_value = value_num
                        continue
                else:
                    delta = value_num - prev_value
                    if delta < 0:
                        delta = 0.0
                    new_val = delta
                    prev_value = value_num
            out_row = dict(row)
            out_row["value"] = new_val
            out_row["series_type"] = SERIES_INCIDENT
            output.append(out_row)

    output.sort(
        key=lambda r: (
            str(r.get("iso3", "")),
            str(r.get("hazard_code", "")),
            str(r.get("metric", "")),
            str(r.get("as_of_date", "")),
        )
    )
    return output


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
    ensure_headers(OUT_PATH, COLUMNS)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if os.getenv("RESOLVER_SKIP_DTM"):
        LOG.info("dtm: skipped via RESOLVER_SKIP_DTM")
        ensure_header_only()
        return False
    cfg = load_config()
    if not cfg.get("enabled"):
        LOG.info("dtm: disabled via config; writing header only")
        ensure_header_only()
        return False
    rows = build_rows(cfg)
    write_rows(rows)
    LOG.info("dtm: wrote %s rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
