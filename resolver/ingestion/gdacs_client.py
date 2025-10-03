#!/usr/bin/env python3
"""GDACS connector that converts alerts into monthly incident PA rows."""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "gdacs.yml"

COUNTRIES = DATA / "countries.csv"

OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "gdacs.csv"

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

SERIES_SEMANTICS = "incident"
UNIT = "persons"
METRIC = "affected"

DEFAULT_METHOD = (
    "GDACS alerts; monthly-first; event→month allocation; "
    "dedup=episode-aware; policy={policy}"
)

DEFAULT_DEFINITION = (
    "Population from GDACS alerts using fields {fields}; monthly allocation policy={policy}; "
    "per-event dedup strategy={strategy}."
)

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


@dataclass(frozen=True)
class Hazard:
    code: str
    label: str
    hazard_class: str


HAZARD_METADATA = {
    "earthquake": Hazard("earthquake", "Earthquake", "geophysical"),
    "tropical_cyclone": Hazard("tropical_cyclone", "Tropical Cyclone", "meteorological"),
    "flood": Hazard("flood", "Flood", "hydrological"),
    "volcano": Hazard("volcano", "Volcanic Activity", "volcanic"),
    "other": Hazard("other", "Other", "other"),
}


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[gdacs] {message}")


def _is_placeholder_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if "<" in text or ">" in text:
        return True
    if "event-list endpoint" in lowered or "event_list endpoint" in lowered:
        return True
    if "event-details endpoint" in lowered or "event_details endpoint" in lowered:
        return True
    if "placeholder" in lowered:
        return True
    parsed = urlparse(text)
    if not parsed.scheme or parsed.scheme not in {"http", "https"}:
        return True
    if not parsed.netloc:
        return True
    return False


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _config_invalid_reason(cfg: Dict[str, Any]) -> Optional[str]:
    base_urls = cfg.get("base_urls") or {}
    event_list_url = base_urls.get("event_list")
    if _is_placeholder_url(event_list_url):
        return "event_list endpoint"
    details_url = base_urls.get("event_details")
    if details_url and _is_placeholder_url(details_url):
        return "event_details endpoint"
    return None


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        return {}
    with open(CONFIG, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    return cfg


def load_countries() -> Tuple[Dict[str, str], Dict[str, str]]:
    df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    iso_to_name: Dict[str, str] = {}
    name_to_iso: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        iso = str(row.iso3).strip().upper()
        name = str(row.country_name).strip()
        if not iso:
            continue
        iso_to_name[iso] = name
        key = _normalise_text(name)
        if key:
            name_to_iso[key] = iso
    return iso_to_name, name_to_iso


def _normalise_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _pick(mapping: MutableMapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping:
            val = mapping.get(key)
            if val not in (None, ""):
                return val
    return None


def _coerce_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def hazard_from_key(key: str) -> Hazard:
    key_norm = str(key or "").strip().lower()
    return HAZARD_METADATA.get(key_norm, HAZARD_METADATA["other"])


def map_hazard(raw: Any, hazard_map: Dict[str, Sequence[str]], default_key: str) -> Hazard:
    text = str(raw or "").strip()
    if not text:
        return hazard_from_key(default_key)
    norm = _normalise_text(text)
    for canonical, candidates in hazard_map.items():
        for cand in candidates or []:
            if norm == _normalise_text(cand):
                return hazard_from_key(canonical)
    return hazard_from_key(default_key)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            if fmt.endswith("%z"):
                return datetime.strptime(text, fmt)
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _ensure_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = _parse_datetime(value)
    if parsed:
        return parsed.date()
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _month_key(day: date) -> str:
    return day.strftime("%Y-%m")


def _month_end(day: date) -> date:
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


def _inclusive_days(start: date, end: date) -> int:
    return (end - start).days + 1


def month_segments(start: date, end: date) -> List[Tuple[str, int]]:
    if end < start:
        end = start
    segments: List[Tuple[str, int]] = []
    cursor = start
    while cursor <= end:
        segment_end = min(_month_end(cursor), end)
        month = _month_key(cursor)
        days = _inclusive_days(cursor, segment_end)
        segments.append((month, days))
        cursor = segment_end + timedelta(days=1)
    return segments


def allocate_value(total: int, start: date, end: date, policy: str) -> Dict[str, int]:
    policy_norm = (policy or "prorata").strip().lower()
    if total <= 0:
        return {}
    if end < start:
        end = start
    if policy_norm == "start":
        return {_month_key(start): int(round(total))}

    segments = month_segments(start, end)
    total_days = sum(days for _, days in segments)
    if total_days <= 0:
        return {_month_key(start): int(round(total))}

    allocations: Dict[str, int] = {month: 0 for month, _ in segments}
    remainders: List[Tuple[int, str]] = []
    assigned = 0
    for month, days in segments:
        raw = total * days
        base = raw // total_days
        remainder = raw % total_days
        allocations[month] += int(base)
        assigned += int(base)
        remainders.append((int(remainder), month))

    remaining = int(round(total)) - assigned
    for remainder, month in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        allocations[month] += 1
        remaining -= 1

    if remaining != 0:
        # Adjust final bucket to ensure totals match (guard against rounding drift)
        last_month = segments[-1][0]
        allocations[last_month] += remaining

    return {month: max(0, value) for month, value in allocations.items() if value > 0}


def allocate_event(event: "GDACSEvent", policy: str) -> Dict[str, int]:
    return allocate_value(event.impact_value, event.start_date, event.end_date, policy)


@dataclass
class GDACSEvent:
    event_id: str
    episode_id: str
    iso3: str
    hazard: Hazard
    start_date: date
    end_date: date
    impact_value: int
    source_url: str
    doc_title: str
    publication_date: Optional[date]
    impact_field: str


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CANONICAL_HEADERS)


def _digest(parts: Iterable[Any]) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _prepare_session() -> requests.Session:
    session = requests.Session()
    appname = os.getenv("RELIEFWEB_APPNAME", "Resolver-GDACS-Client/1.0")
    session.headers.update({"User-Agent": appname})
    return session


def _read_response_json(data: Any) -> List[MutableMapping[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [dict(item) if isinstance(item, MutableMapping) else {} for item in data]
    if isinstance(data, MutableMapping):
        for key in ("events", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                return [dict(item) if isinstance(item, MutableMapping) else {} for item in data[key]]
        return [dict(data)]
    return []


def _fetch_event_list(session: requests.Session, cfg: Dict[str, Any], start: date, end: date) -> List[MutableMapping[str, Any]]:
    base = (cfg.get("base_urls") or {}).get("event_list")
    if not base:
        raise RuntimeError("GDACS event_list base URL missing in config")
    params = {"from": start.isoformat(), "to": end.isoformat()}
    try:
        response = session.get(base, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"GDACS event list request failed: {exc}") from exc
    except ValueError as exc:  # pragma: no cover - JSON decode errors
        raise RuntimeError("GDACS event list did not return JSON") from exc
    return _read_response_json(data)


def _fetch_event_details(
    session: requests.Session,
    cfg: Dict[str, Any],
    event: MutableMapping[str, Any],
    keys: Dict[str, Sequence[str]],
) -> MutableMapping[str, Any]:
    details_url = (cfg.get("base_urls") or {}).get("event_details")
    if not details_url:
        return event
    event_id = _pick(event, keys.get("id", []))
    if not event_id:
        return event
    params: Dict[str, Any] = {"eventid": event_id}
    episode_id = _pick(event, keys.get("episode", []))
    if episode_id:
        params["episodeid"] = episode_id
    try:
        response = session.get(details_url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return event
    except ValueError:  # pragma: no cover - JSON decode errors
        return event
    details_list = _read_response_json(payload)
    if not details_list:
        return event
    merged = dict(event)
    merged.update(details_list[0])
    return merged


def _resolve_iso3(
    event: MutableMapping[str, Any],
    keys: Dict[str, Sequence[str]],
    name_to_iso: Dict[str, str],
) -> List[str]:
    iso_candidates: List[str] = []
    seen: set[str] = set()

    iso_raw = _pick(event, keys.get("iso3", []))
    for value in _coerce_to_list(iso_raw):
        iso = value.upper()
        if len(iso) == 3 and iso not in seen:
            iso_candidates.append(iso)
            seen.add(iso)

    if iso_candidates:
        return iso_candidates

    country_val = _pick(event, keys.get("country", []))
    for name in _coerce_to_list(country_val):
        iso = name_to_iso.get(_normalise_text(name))
        if iso and iso not in seen:
            iso_candidates.append(iso)
            seen.add(iso)

    return iso_candidates


def _extract_publication_date(event: MutableMapping[str, Any], keys: Dict[str, Sequence[str]]) -> Optional[date]:
    value = _pick(event, keys.get("updated", []))
    if not value:
        return None
    parsed = _ensure_date(value)
    return parsed


def _prepare_events(
    raw_events: List[MutableMapping[str, Any]],
    cfg: Dict[str, Any],
    hazard_map: Dict[str, Sequence[str]],
    default_hazard: str,
    keys: Dict[str, Sequence[str]],
    name_to_iso: Dict[str, str],
) -> List[GDACSEvent]:
    events: List[GDACSEvent] = []
    for raw in raw_events:
        iso_codes = _resolve_iso3(raw, keys, name_to_iso)
        if not iso_codes:
            dbg(f"skip event missing ISO3: {raw}")
            continue
        hazard = map_hazard(_pick(raw, keys.get("type", [])), hazard_map, default_hazard)
        start_date = _ensure_date(_pick(raw, keys.get("start", [])))
        end_date = _ensure_date(_pick(raw, keys.get("end", []))) or start_date
        if not start_date:
            dbg("skip event without start date")
            continue
        if end_date is None:
            end_date = start_date
        impact_field = ""
        impact_value_raw: Optional[Any] = None
        for candidate in keys.get("impact", []):
            if candidate in raw and raw.get(candidate) not in (None, ""):
                impact_field = candidate
                impact_value_raw = raw.get(candidate)
                break
        impact_value = _parse_float(impact_value_raw)
        if impact_value is None:
            dbg("skip event without impact value")
            continue
        impact_int = int(round(impact_value))
        if impact_int <= 0:
            dbg("skip event with non-positive impact")
            continue
        event_id = str(_pick(raw, keys.get("id", [])) or "").strip()
        if not event_id:
            dbg("skip event missing id")
            continue
        episode_id = str(_pick(raw, keys.get("episode", [])) or "").strip()
        source_url = str(_pick(raw, keys.get("url", [])) or "").strip()
        doc_title = str(_pick(raw, keys.get("title", [])) or "").strip()
        publication_date = _extract_publication_date(raw, keys)

        for iso3 in iso_codes:
            events.append(
                GDACSEvent(
                    event_id=event_id,
                    episode_id=episode_id,
                    iso3=iso3,
                    hazard=hazard,
                    start_date=start_date,
                    end_date=end_date or start_date,
                    impact_value=impact_int,
                    source_url=source_url,
                    doc_title=doc_title,
                    publication_date=publication_date,
                    impact_field=impact_field or "impact",
                )
            )
    return events


def dedupe_monthly_rows(
    rows: Iterable[Dict[str, Any]],
    strategy: str,
) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    strategy_norm = (strategy or "max").strip().lower()
    per_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("event_ref", ""),
            row.get("episode_id", ""),
            row.get("iso3", ""),
            row.get("as_of_date", ""),
        )
        value = int(row.get("value", 0))
        if key not in per_key:
            per_key[key] = dict(row)
            per_key[key]["value"] = value
            continue
        if strategy_norm == "sum":
            per_key[key]["value"] += value
        else:
            if value > int(per_key[key].get("value", 0)):
                per_key[key] = dict(row)
                per_key[key]["value"] = value
    return per_key


def _aggregate_final_rows(
    rows: Iterable[Dict[str, Any]],
    iso_to_name: Dict[str, str],
    policy: str,
    strategy: str,
    publisher: str,
    source_type: str,
) -> List[Dict[str, Any]]:
    aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["iso3"], row["hazard_code"], row["as_of_date"])
        bucket = aggregated.setdefault(
            key,
            {
                "iso3": row["iso3"],
                "hazard_code": row["hazard_code"],
                "hazard_label": row["hazard_label"],
                "hazard_class": row["hazard_class"],
                "as_of_date": row["as_of_date"],
                "value": 0,
                "source_urls": set(),
                "doc_titles": set(),
                "event_refs": set(),
                "episode_ids": set(),
                "publication_dates": [],
                "impact_fields": set(),
            },
        )
        bucket["value"] += int(row.get("value", 0))
        if row.get("source_url"):
            bucket["source_urls"].add(str(row.get("source_url")))
        if row.get("doc_title"):
            bucket["doc_titles"].add(str(row.get("doc_title")))
        if row.get("event_ref"):
            bucket["event_refs"].add(str(row.get("event_ref")))
        if row.get("episode_id"):
            bucket["episode_ids"].add(str(row.get("episode_id")))
        if row.get("publication_date"):
            bucket["publication_dates"].append(row.get("publication_date"))
        if row.get("impact_field" ):
            bucket["impact_fields"].add(str(row.get("impact_field")))

    final_rows: List[Dict[str, Any]] = []
    ingested_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for key, bucket in aggregated.items():
        iso3, hazard_code, as_of_month = key
        country_name = iso_to_name.get(iso3, "")
        value = int(bucket["value"])
        if value <= 0:
            continue
        source_urls = sorted(bucket["source_urls"])
        doc_titles = sorted(bucket["doc_titles"])
        event_refs = sorted(bucket["event_refs"])
        episode_ids = sorted(bucket["episode_ids"])
        publication_dates = [d for d in bucket["publication_dates"] if d]
        publication_date = ""
        if publication_dates:
            latest = max(publication_dates)
            if isinstance(latest, date):
                publication_date = latest.isoformat()
            else:
                publication_date = str(latest)
        impact_fields = sorted(bucket["impact_fields"]) or ["impact"]
        method = DEFAULT_METHOD.format(policy=policy)
        definition_text = DEFAULT_DEFINITION.format(
            fields="/".join(impact_fields), policy=policy, strategy=strategy
        )
        digest = _digest(
            [
                iso3,
                hazard_code,
                as_of_month,
                value,
                ";".join(source_urls),
                ";".join(event_refs),
                ";".join(episode_ids),
            ]
        )
        year, month = as_of_month.split("-")
        event_id = f"{iso3}-GDACS-{hazard_code}-{METRIC}-{year}-{month}-{digest}"
        final_rows.append(
            {
                "event_id": event_id,
                "country_name": country_name,
                "iso3": iso3,
                "hazard_code": hazard_code,
                "hazard_label": bucket["hazard_label"],
                "hazard_class": bucket["hazard_class"],
                "metric": METRIC,
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": UNIT,
                "as_of_date": as_of_month,
                "publication_date": publication_date,
                "publisher": publisher,
                "source_type": source_type,
                "source_url": ";".join(source_urls),
                "doc_title": "; ".join(doc_titles),
                "definition_text": definition_text,
                "method": method,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    final_rows.sort(key=lambda r: (r["iso3"], r["hazard_code"], r["as_of_date"]))
    return final_rows


def _apply_max_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cap = _env_int("RESOLVER_MAX_RESULTS")
    if cap is None or cap <= 0:
        return rows
    return rows[:cap]


def run(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    if not cfg:
        dbg("missing GDACS config; returning no rows")
        return []
    reason = _config_invalid_reason(cfg)
    if reason:
        dbg(f"invalid config reason: {reason}")
        return []
    skip = _env_bool("RESOLVER_SKIP_GDACS", False)
    if skip:
        dbg("RESOLVER_SKIP_GDACS=1 → skip network fetch")
        return []

    window_days = cfg.get("window_days", 60)
    env_window = _env_int("GDACS_WINDOW_DAYS")
    if env_window is not None and env_window > 0:
        window_days = env_window
    allocation_policy = os.getenv("GDACS_ALLOC_POLICY", cfg.get("allocation_policy", "prorata"))
    dedup_strategy = os.getenv("GDACS_DEDUP_STRATEGY", cfg.get("dedup_strategy", "max"))
    publisher = cfg.get("publisher", "GDACS")
    source_type = cfg.get("source_type", "other")

    hazard_map = cfg.get("hazard_map", {}) or {}
    default_hazard = cfg.get("default_hazard", "other")
    keys = cfg.get("keys", {}) or {}

    iso_to_name, name_to_iso = load_countries()

    end_date = date.today()
    start_date = end_date - timedelta(days=int(window_days))

    session = _prepare_session()
    try:
        raw_events = _fetch_event_list(session, cfg, start_date, end_date)
    except Exception as exc:  # pragma: no cover - network failure
        dbg(str(exc))
        return []

    enriched_events: List[MutableMapping[str, Any]] = []
    for raw in raw_events:
        enriched_events.append(_fetch_event_details(session, cfg, raw, keys))

    gdacs_events = _prepare_events(
        enriched_events,
        cfg,
        hazard_map,
        default_hazard,
        keys,
        name_to_iso,
    )

    rows: List[Dict[str, Any]] = []
    per_event_rows: List[Dict[str, Any]] = []
    for event in gdacs_events:
        allocations = allocate_event(event, allocation_policy)
        for month, value in allocations.items():
            per_event_rows.append(
                {
                    "event_ref": event.event_id,
                    "episode_id": event.episode_id,
                    "iso3": event.iso3,
                    "hazard_code": event.hazard.code,
                    "hazard_label": event.hazard.label,
                    "hazard_class": event.hazard.hazard_class,
                    "as_of_date": month,
                    "value": value,
                    "source_url": event.source_url,
                    "doc_title": event.doc_title,
                    "publication_date": event.publication_date,
                    "impact_field": event.impact_field,
                }
            )

    deduped = dedupe_monthly_rows(per_event_rows, dedup_strategy)
    final_rows = _aggregate_final_rows(
        deduped.values(),
        iso_to_name,
        allocation_policy,
        dedup_strategy,
        publisher,
        source_type,
    )
    rows.extend(_apply_max_results(final_rows))
    return rows


def write_rows(rows: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CANONICAL_HEADERS)
    if df.empty:
        df = pd.DataFrame(columns=CANONICAL_HEADERS)
    df.to_csv(OUT_PATH, index=False)


def main() -> bool:
    if _env_bool("RESOLVER_SKIP_GDACS", False):
        _write_header_only(OUT_PATH)
        return False
    cfg = load_config()
    reason = _config_invalid_reason(cfg)
    if reason:
        print(f"[gdacs] disabled/invalid config; writing header-only ({reason})")
        dbg(f"invalid config reason: {reason}")
        _write_header_only(OUT_PATH)
        return False
    try:
        rows = run(cfg)
    except Exception as exc:  # pragma: no cover - defensive fail-soft
        dbg(f"gdacs main failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        _write_header_only(OUT_PATH)
        return False

    write_rows(rows)
    return True


if __name__ == "__main__":  # pragma: no cover
    main()

