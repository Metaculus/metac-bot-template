#!/usr/bin/env python3
"""IOM DTM connector scaffolding with monthly-first logic and helpers."""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "dtm.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "dtm.csv"

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

HAZARD_KEY_TO_CODE = {
    "flood": "FL",
    "drought": "DR",
    "tropical_cyclone": "TC",
    "heat_wave": "HW",
    "armed_conflict_onset": "ACO",
    "armed_conflict_escalation": "ACE",
    "civil_unrest": "CU",
    "displacement_influx": "DI",
    "economic_crisis": "EC",
    "phe": "PHE",
}

MULTI_HAZARD = ("multi", "Multi-shock Displacement/Needs", "all")

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


@dataclass
class Hazard:
    code: str
    label: str
    hclass: str


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[dtm] {message}")


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "y", "yes", "on"}


def load_config() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return countries, shocks


def _parse_month(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return None
    return int(parsed.year), int(parsed.month)


def _normalise_month(value: Any) -> Optional[str]:
    parsed = _parse_month(value)
    if not parsed:
        return None
    year, month = parsed
    return f"{year:04d}-{month:02d}"


def _is_subnational(record: MutableMapping[str, Any]) -> bool:
    for key in ("admin1", "admin2", "admin_pcode", "admin_name"):
        if str(record.get(key, "")).strip():
            return True
    return False


def rollup_subnational(records: Sequence[MutableMapping[str, Any]]) -> List[MutableMapping[str, Any]]:
    """Aggregate subnational rows into national totals per month and source."""

    grouped: Dict[Tuple[str, str, str, str, str, str], List[MutableMapping[str, Any]]] = defaultdict(list)
    for rec in records:
        as_of = _normalise_month(rec.get("as_of_date")) or ""
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            as_of,
            str(rec.get("source_id", "")),
            str(rec.get("series_type", SERIES_INCIDENT)),
        )
        rec = dict(rec)
        rec["as_of_date"] = as_of
        grouped[key].append(rec)

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

    rolled.sort(key=lambda r: (
        str(r.get("iso3", "")),
        str(r.get("hazard_code", "")),
        str(r.get("metric", "")),
        str(r.get("as_of_date", "")),
    ))
    return rolled


def compute_monthly_deltas(
    records: Sequence[MutableMapping[str, Any]],
    *,
    allow_first_month: Optional[bool] = None,
) -> List[MutableMapping[str, Any]]:
    """Convert cumulative series to month-over-month deltas.

    Incident series are passed through. Values are clipped to zero to avoid
    negative deltas from noisy cumulative plateaus.
    """

    cfg = load_config()
    if allow_first_month is None:
        allow_first_month = _env_bool(
            "DTM_ALLOW_FIRST_MONTH",
            bool(cfg.get("allow_first_month_delta", False)),
        )

    grouped: Dict[Tuple[str, str, str, str], List[MutableMapping[str, Any]]] = defaultdict(list)
    for rec in records:
        as_of = _normalise_month(rec.get("as_of_date")) or ""
        rec = dict(rec)
        rec["as_of_date"] = as_of
        key = (
            str(rec.get("iso3", "")),
            str(rec.get("hazard_code", "")),
            str(rec.get("metric", "")),
            str(rec.get("source_id", "")),
        )
        grouped[key].append(rec)

    output: List[MutableMapping[str, Any]] = []
    for key, rows in grouped.items():
        rows.sort(key=lambda r: _parse_month(r.get("as_of_date")) or (0, 0))
        series_type = str(rows[0].get("series_type", SERIES_INCIDENT)).strip().lower()
        prev_value: Optional[float] = None
        for idx, row in enumerate(rows):
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

    output.sort(key=lambda r: (
        str(r.get("iso3", "")),
        str(r.get("hazard_code", "")),
        str(r.get("metric", "")),
        str(r.get("as_of_date", "")),
    ))
    return output


def _hazard_from_code(code: str, shocks: pd.DataFrame) -> Hazard:
    if not code:
        return Hazard(*MULTI_HAZARD)
    if code.lower() == "multi":
        return Hazard(*MULTI_HAZARD)
    match = shocks[shocks["hazard_code"].str.upper() == code.upper()]
    if match.empty:
        return Hazard(*MULTI_HAZARD)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def infer_hazard(
    texts: Iterable[str],
    shocks: Optional[pd.DataFrame] = None,
    keywords_cfg: Optional[Dict[str, List[str]]] = None,
    *,
    default_key: Optional[str] = None,
) -> Hazard:
    """Map dataset metadata to a forecastable hazard."""

    if shocks is None:
        _, shocks = load_registries()
    if keywords_cfg is None:
        keywords_cfg = load_config().get("shock_keywords", {})
    if default_key is None:
        default_key = os.getenv(
            "DTM_DEFAULT_HAZARD",
            load_config().get("default_hazard", "displacement_influx"),
        )

    text_blob = " ".join([str(t).lower() for t in texts if t])
    matches: List[str] = []
    for key, keywords in keywords_cfg.items():
        for kw in keywords:
            if kw.lower() in text_blob:
                matches.append(key)
                break

    if not matches:
        mapped = HAZARD_KEY_TO_CODE.get(str(default_key).strip().lower())
        if not mapped:
            return Hazard(*MULTI_HAZARD)
        return _hazard_from_code(mapped, shocks)

    unique = sorted(set(matches))
    if len(unique) > 1:
        return Hazard(*MULTI_HAZARD)

    mapped = HAZARD_KEY_TO_CODE.get(unique[0])
    if not mapped:
        return Hazard(*MULTI_HAZARD)
    return _hazard_from_code(mapped, shocks)


def build_event_id(
    iso3: str,
    hazard_code: str,
    metric: str,
    as_of_date: str,
    value: Any,
    source_url: str,
) -> str:
    """Construct deterministic event IDs for downstream dedupe."""

    year, month = _parse_month(as_of_date) or (0, 0)
    digest = hashlib.sha1(
        "|".join([
            str(iso3 or "UNK"),
            str(hazard_code or ""),
            str(metric or ""),
            f"{year:04d}-{month:02d}",
            str(value or 0),
            str(source_url or ""),
        ]).encode("utf-8")
    ).hexdigest()[:12]
    iso = iso3 or "UNK"
    hz = hazard_code or "UNK"
    metric = metric or "metric"
    return f"{iso}-DTM-{hz}-{metric}-{year:04d}-{month:02d}-{digest}"


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_HEADERS).to_csv(path, index=False)


def _write_rows(rows: Sequence[Dict[str, Any]], *, path: Path) -> None:
    if not rows:
        _write_header_only(path)
        return
    df = pd.DataFrame(rows)
    if "series_semantics" not in df.columns:
        df["series_semantics"] = SERIES_INCIDENT
    else:
        df["series_semantics"] = df["series_semantics"].replace("", SERIES_INCIDENT).fillna(SERIES_INCIDENT)
    for col in CANONICAL_HEADERS:
        if col not in df.columns:
            df[col] = ""
    df = df[CANONICAL_HEADERS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def collect_rows() -> List[Dict[str, Any]]:
    """Placeholder for phased DTM ingestion (HDX mirrors in phase 1).

    The network discovery and HXL parsing logic will be expanded in follow-up
    cards. For now we fail-soft by returning no rows so the connector writes the
    canonical header only (satisfying CI expectations).
    """

    return []


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_DTM") == "1":
        dbg("RESOLVER_SKIP_DTM=1 — skipping network access")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:  # fail-soft
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, path=OUT_PATH)
    dbg(f"wrote {len(rows)} DTM rows")
    return True


if __name__ == "__main__":
    main()
