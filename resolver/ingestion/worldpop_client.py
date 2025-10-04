#!/usr/bin/env python3
"""WorldPop denominators connector."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import to_iso3

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "worldpop.yml"
OUTPUT_PATH = STAGING / "worldpop_denominators.csv"
DATA_DIR = ROOT / "data"

LOG = logging.getLogger("resolver.ingestion.worldpop")

COLUMNS = ["country_iso3", "year", "population", "as_of", "source", "method"]


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
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="worldpop_denominators.v1", source_id="worldpop")


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


def _read_existing() -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return {}
    existing: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso = str(row.get("country_iso3") or "").strip().upper()
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
    dataset = _load_dataset(_dataset_path(cfg))
    aliases = cfg.get("country_aliases") or {}
    years = [int(y) for y in cfg.get("years", []) if str(y).isdigit()]
    if not years:
        raise ValueError("worldpop config must include years list")
    as_of = datetime.utcnow().date().isoformat()
    source = "worldpop"
    method = str(cfg.get("product") or "worldpop_national")
    updates: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in dataset:
        iso = to_iso3(row.get("iso3"), aliases) or to_iso3(row.get("country"), aliases)
        if not iso:
            continue
        try:
            year = int(row.get("year"))
        except (TypeError, ValueError):
            continue
        if year not in years:
            continue
        population = _parse_population(row.get("population"))
        if population is None:
            continue
        updates[(iso, year)] = {
            "country_iso3": iso,
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
            data["country_iso3"],
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


def write_rows(rows: List[List[Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="worldpop_denominators.v1", source_id="worldpop")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config()
    if not cfg.get("enabled"):
        LOG.info("worldpop: disabled via config; writing header only")
        ensure_header_only()
        return 0
    rows = build_rows(cfg)
    write_rows(rows)
    LOG.info("worldpop: wrote %s rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
