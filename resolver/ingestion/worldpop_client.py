#!/usr/bin/env python3
"""WorldPop denominators connector."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import ensure_headers, to_iso3

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "worldpop.yml"
CONFIG: Path | str = CONFIG_PATH
OUT_DATA = ROOT / "data" / "population.csv"
OUT_STAGING = STAGING / "worldpop_denominators.csv"
OUTPUT_PATH = OUT_STAGING  # backwards compatibility alias
DATA_DIR = ROOT / "data"

LOG = logging.getLogger("resolver.ingestion.worldpop")

COLUMNS = ["iso3", "year", "population", "as_of", "source", "method"]

CANONICAL_COLUMNS = COLUMNS


def load_config() -> dict[str, Any]:
    path = Path(CONFIG)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_header_only() -> None:
    ensure_headers(OUT_DATA, COLUMNS)
    ensure_headers(OUT_STAGING, COLUMNS)
    ensure_manifest_for_csv(OUT_STAGING, schema_version="worldpop_denominators.v1", source_id="worldpop")


def _dataset_path(cfg: Mapping[str, Any]) -> Path:
    product = str(cfg.get("product") or "").strip()
    candidate = cfg.get("dataset_path")
    if candidate:
        return Path(candidate)
    if product:
        preferred = DATA_DIR / f"{product}.csv"
        if preferred.exists():
            return preferred
    fallback = DATA_DIR / "worldpop.csv"
    if fallback.exists():
        return fallback
    return DATA_DIR / "population.csv"


def _load_dataset(path: Path) -> List[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"WorldPop dataset not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _coerce_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "latest":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _normalise_year_list(values: Any) -> Optional[List[int]]:
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    elif not isinstance(values, Iterable):
        values = [values]
    years: List[int] = []
    for value in values:
        year = _coerce_year(value)
        if year is not None:
            years.append(year)
    return years or None


def _infer_latest_year(rows: Iterable[Mapping[str, Any]]) -> Optional[int]:
    candidates = [_coerce_year(row.get("year")) for row in rows]
    numbers = [year for year in candidates if year is not None]
    if not numbers:
        return None
    return max(numbers)


def _years_to_fetch(cfg: Mapping[str, Any], year_filter: Optional[List[int]], latest_rows: List[Mapping[str, Any]]) -> List[int]:
    if year_filter:
        return year_filter
    years_back_raw = cfg.get("years_back")
    if years_back_raw is None:
        return []
    try:
        years_back = int(years_back_raw)
    except (TypeError, ValueError):
        years_back = 0
    latest_hint = _coerce_year(cfg.get("latest_year")) or _coerce_year(cfg.get("year"))
    if latest_hint is None:
        latest_hint = _infer_latest_year(latest_rows)
    if latest_hint is None:
        latest_hint = datetime.utcnow().year
    return [latest_hint - offset for offset in range(max(years_back, 0) + 1)]


def _collect_source_rows(cfg: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], Optional[set[int]]]:
    year_filter_list = _normalise_year_list(cfg.get("years"))
    source_cfg = cfg.get("source") or {}
    template = str(source_cfg.get("url_template") or "").strip()
    rows: List[Mapping[str, Any]] = []
    if template:
        latest_rows: List[Mapping[str, Any]] = []
        latest_path = Path(template.format(year="latest"))
        if latest_path.exists():
            latest_rows = _load_dataset(latest_path)
        years_to_fetch = _years_to_fetch(cfg, year_filter_list, latest_rows)
        for year in sorted({year for year in years_to_fetch if isinstance(year, int)}):
            try:
                path = Path(template.format(year=year))
            except KeyError:
                continue
            if not path.exists():
                LOG.debug("worldpop: missing configured year file %s", path)
                continue
            rows.extend(_load_dataset(path))
        rows.extend(latest_rows)
        if rows:
            return rows, set(year_filter_list) if year_filter_list else None
    dataset = _load_dataset(_dataset_path(cfg))
    return dataset, set(year_filter_list) if year_filter_list else None


def _read_existing() -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not OUT_DATA.exists():
        return {}
    existing: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with OUT_DATA.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso = str(row.get("iso3") or row.get("country_iso3") or "").strip().upper()
            try:
                year = int(row.get("year"))
            except (TypeError, ValueError):
                continue
            existing[(iso, year)] = dict(row)
    return existing


def _parse_population(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text.replace(",", "")))
    except ValueError:
        return None


def build_rows(cfg: Mapping[str, Any]) -> List[List[Any]]:
    dataset, year_filter = _collect_source_rows(cfg)
    aliases = cfg.get("country_aliases") or {}
    as_of = datetime.utcnow().date().isoformat()
    source = "worldpop"
    method = str(cfg.get("product") or "worldpop_national")
    updates: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in dataset:
        raw_iso = row.get("iso3") or row.get("country")
        iso = to_iso3(raw_iso, aliases) if raw_iso else None
        if not iso and raw_iso:
            iso = str(raw_iso).strip().upper()
        if not iso:
            continue
        try:
            year_val = _coerce_year(row.get("year"))
        except Exception:
            year_val = None
        if year_val is None:
            continue
        year = year_val
        if year_filter is not None and year not in year_filter:
            continue
        population = _parse_population(row.get("population"))
        if population is None:
            continue
        updates[(iso, year)] = {
            "iso3": iso,
            "year": year,
            "population": population,
            "as_of": as_of,
            "source": source,
            "method": method,
        }
    existing = _read_existing()
    existing.update(updates)
    rows = [
        [
            data["iso3"],
            data["year"],
            data["population"],
            data["as_of"],
            data["source"],
            data["method"],
        ]
        for data in existing.values()
    ]
    rows.sort(key=lambda row: (row[0], int(row[1])))
    return rows


def _write_csv(path: Path, rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def write_rows(rows: List[List[Any]]) -> None:
    _write_csv(OUT_DATA, rows)
    _write_csv(OUT_STAGING, rows)
    ensure_manifest_for_csv(OUT_STAGING, schema_version="worldpop_denominators.v1", source_id="worldpop")


def collect_rows() -> List[List[Any]]:
    """Return WorldPop rows without writing side effects.

    This helper restores the historical public API used by stubs/tests while
    keeping the heavy lifting in :func:`build_rows`.
    """

    if os.getenv("RESOLVER_SKIP_WORLDPOP"):
        return []

    cfg = load_config()
    if not cfg or not cfg.get("enabled", True):
        return []

    return build_rows(cfg)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if os.getenv("RESOLVER_SKIP_WORLDPOP"):
        LOG.info("worldpop: skipped via RESOLVER_SKIP_WORLDPOP")
        ensure_header_only()
        return False
    rows = collect_rows()
    if not rows:
        LOG.info("worldpop: no rows collected; writing header only")
        ensure_header_only()
        return False
    write_rows(rows)
    LOG.info("worldpop: wrote %s rows", len(rows))
    return True


__all__ = [
    "COLUMNS",
    "CANONICAL_COLUMNS",
    "CONFIG",
    "CONFIG_PATH",
    "OUT_DATA",
    "OUT_STAGING",
    "OUTPUT_PATH",
    "build_rows",
    "collect_rows",
    "ensure_header_only",
    "load_config",
    "main",
    "write_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
