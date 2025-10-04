#!/usr/bin/env python3
"""GDACS connector that emits monthly hazard signals."""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion.utils import map_hazard, month_start, parse_date, stable_digest, to_iso3

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "gdacs.yml"
OUTPUT_PATH = STAGING / "gdacs_signals.csv"

LOG = logging.getLogger("resolver.ingestion.gdacs")

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

DEFAULT_ENDPOINTS = {
    "event_list": "https://www.gdacs.org/gdacsapi/api/events/geteventlist",
    "event_details": "https://www.gdacs.org/gdacsapi/api/events/getevent",
}
DEFAULT_SEVERITY = {"green": 0, "orange": 1, "red": 2}

USER_AGENT = "UNICEF-Resolver-P1L1T6"


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
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="gdacs_signals.v1", source_id="gdacs")


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _as_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {} for item in payload]
    if isinstance(payload, dict):
        for key in ("events", "data", "results", "features", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {} for item in value]
        return [payload]
    return []


def _ci_get(mapping: Mapping[str, Any], key: str) -> Any:
    lower = key.lower()
    for actual_key, value in mapping.items():
        if actual_key.lower() == lower:
            return value
    return None


def _pluck(record: Mapping[str, Any], *candidates: str) -> Any:
    for key in candidates:
        if key in record:
            value = record.get(key)
            if value not in (None, ""):
                return value
        if "properties" in record and isinstance(record["properties"], Mapping):
            nested = _pluck(record["properties"], key)
            if nested not in (None, ""):
                return nested
        lowered = key.lower()
        for actual, value in record.items():
            if actual.lower() == lowered and value not in (None, ""):
                return value
    return None


def _extract_countries(event: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    countries: List[Mapping[str, Any]] = []
    candidate = _pluck(event, "countries", "country", "countrylist")
    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, Mapping):
                countries.append(item)
            elif isinstance(item, str):
                countries.append({"name": item})
    elif isinstance(candidate, Mapping):
        countries.append(candidate)
    elif isinstance(candidate, str):
        for part in candidate.replace(";", ",").split(","):
            part = part.strip()
            if part:
                countries.append({"name": part})
    else:
        iso = _pluck(event, "countryiso", "iso3", "iso")
        name = _pluck(event, "countryname")
        if iso or name:
            countries.append({"iso3": iso, "name": name})
    return countries


def _country_iso(entry: Mapping[str, Any], aliases: Mapping[str, str]) -> Optional[str]:
    iso = None
    if isinstance(entry, Mapping):
        if "iso3" in entry:
            iso = str(entry.get("iso3") or "").strip()
        elif "iso" in entry:
            iso = str(entry.get("iso") or "").strip()
        if iso:
            resolved = to_iso3(iso, aliases)
            if resolved:
                return resolved
        for key in ("name", "country", "countryname"):
            if key in entry:
                resolved = to_iso3(entry.get(key), aliases)
                if resolved:
                    return resolved
    return None


def _severity_value(event: Mapping[str, Any], severity_map: Mapping[str, int]) -> int:
    raw = _pluck(event, "alertlevel", "severity", "alert")
    if not raw:
        return 0
    text = str(raw).strip()
    if not text:
        return 0
    value = severity_map.get(text)
    if value is not None:
        return int(value)
    lowered = text.lower()
    if lowered in severity_map:
        return int(severity_map[lowered])
    return int(DEFAULT_SEVERITY.get(lowered, 0))


def _as_iso_date(value: Any) -> Optional[str]:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def _event_month(event: Mapping[str, Any]) -> Optional[date]:
    for key in ("eventdate", "fromdate", "begindate", "startdate"):
        candidate = _pluck(event, key)
        parsed = month_start(candidate)
        if parsed:
            return parsed
    updated = parse_date(_pluck(event, "updatedate", "lastupdate"))
    if updated:
        return updated.replace(day=1)
    return None


def fetch_events(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    endpoints = dict(DEFAULT_ENDPOINTS)
    endpoints.update(cfg.get("endpoints") or {})
    window_days = int(cfg.get("window_days", 365) or 365)
    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)
    session = _create_session()
    tries = int(cfg.get("retry", {}).get("tries", 3) or 3)
    backoff = float(cfg.get("retry", {}).get("backoff", 2.0) or 2.0)
    params = {"from": start_date.isoformat(), "to": end_date.isoformat()}
    url = endpoints.get("event_list") or DEFAULT_ENDPOINTS["event_list"]
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(url, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            return _as_list(payload)
        except requests.RequestException as exc:  # pragma: no cover - network
            if attempt >= tries:
                raise RuntimeError(f"GDACS event list request failed: {exc}") from exc
            sleep_for = backoff * attempt
            LOG.warning("gdacs event list failed; retrying", extra={"error": str(exc), "attempt": attempt})
            time.sleep(sleep_for)


def build_rows(cfg: Mapping[str, Any], events: Iterable[Mapping[str, Any]]) -> List[List[Any]]:
    aliases = cfg.get("country_aliases") or {}
    hazard_overrides = cfg.get("hazard_map") or {}
    severity_map: dict[str, int] = {}
    for key, value in DEFAULT_SEVERITY.items():
        severity_map[key] = value
    for key, value in (cfg.get("severity_map") or {}).items():
        severity_map[str(key).strip().lower()] = int(value)
    as_of_default = date.today().isoformat()
    dedup: dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for event in events:
        hazard_token = _pluck(event, "eventtype", "hazardtype", "type")
        hazard = map_hazard(hazard_token, hazard_overrides)
        if not hazard:
            continue
        countries = _extract_countries(event)
        if not countries:
            continue
        month = _event_month(event)
        if not month:
            continue
        raw_event_id = _pluck(event, "eventid", "eventid", "id", "glide")
        if not raw_event_id:
            raw_event_id = _pluck(event, "name", "title")
        month_iso = month.isoformat()
        as_of = _as_iso_date(_pluck(event, "updatedate", "lastupdate", "pubdate")) or as_of_default
        value = _severity_value(event, severity_map)
        confidence = _pluck(event, "alertscore", "confidence")
        raw_json = json.dumps({
            "alertlevel": _pluck(event, "alertlevel"),
            "eventid": raw_event_id,
            "eventtype": hazard_token,
            "countries": countries,
        }, ensure_ascii=False)
        for entry in countries:
            iso3 = _country_iso(entry, aliases)
            if not iso3:
                continue
            digest = stable_digest([iso3, hazard, month_iso, raw_event_id])
            event_id = f"{iso3}-{hazard}-{month.strftime('%Y%m')}-{digest}"
            key = (iso3, hazard, month_iso, str(raw_event_id))
            record = {
                "source": "gdacs",
                "hazard_type": hazard,
                "country_iso3": iso3,
                "event_id": event_id,
                "as_of": as_of,
                "month_start": month_iso,
                "value_type": "signal_level",
                "value": value,
                "unit": "",
                "method": "gdacs_alert",
                "confidence": str(confidence or "").strip(),
                "raw_event_id": str(raw_event_id or ""),
                "raw_fields_json": raw_json,
            }
            existing = dedup.get(key)
            if existing:
                if existing["as_of"] >= record["as_of"]:
                    continue
            dedup[key] = record
    rows = [
        [
            record["source"],
            record["hazard_type"],
            record["country_iso3"],
            record["event_id"],
            record["as_of"],
            record["month_start"],
            record["value_type"],
            record["value"],
            record["unit"],
            record["method"],
            record["confidence"],
            record["raw_event_id"],
            record["raw_fields_json"],
        ]
        for record in dedup.values()
    ]
    rows.sort(key=lambda row: (row[2], row[1], row[5], row[11]))
    return rows


def write_rows(rows: List[List[Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUTPUT_PATH, schema_version="gdacs_signals.v1", source_id="gdacs")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config()
    if not cfg.get("enabled"):
        LOG.info("gdacs: disabled via config; writing header only")
        ensure_header_only()
        return 0
    try:
        events = fetch_events(cfg)
    except Exception as exc:  # noqa: BLE001
        LOG.error("gdacs fetch failed", exc_info=exc)
        raise
    rows = build_rows(cfg, events)
    write_rows(rows)
    LOG.info("gdacs: wrote %s rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
