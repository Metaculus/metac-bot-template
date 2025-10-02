#!/usr/bin/env python3
"""EM-DAT connector that allocates event impacts to monthly incident totals."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "emdat.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

OUT_DIR = STAGING
OUT_PATH = OUT_DIR / "emdat.csv"

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

HAZARD_KEY_TO_CODE = {
    "flood": "FL",
    "drought": "DR",
    "tropical_cyclone": "TC",
    "heat_wave": "HW",
    "cold_wave": "CW",
    "wildfire": "WF",
    "earthquake": "EQ",
    "landslide": "LS",
    "volcano": "VO",
    "phe": "PHE",
    "other": "OT",
    "multi": "MULTI",
}

DEFAULT_METHOD_PREFIX = "EM-DAT event→month allocation"
DEFAULT_DEFINITION_PRORATA = (
    "Total affected people from EM-DAT events allocated to each overlapping month "
    "proportionally by days (incident new affected)."
)
DEFAULT_DEFINITION_START = (
    "Total affected people from EM-DAT events allocated entirely to the event start month "
    "(incident new affected)."
)

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


@dataclass(frozen=True)
class Hazard:
    code: str
    label: str
    hazard_class: str


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[emdat] {message}")


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


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        return {"sources": []}
    with open(CONFIG, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    cfg.setdefault("sources", [])
    return cfg


def load_registries() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Hazard]]:
    countries_df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks_df = pd.read_csv(SHOCKS, dtype=str).fillna("")

    iso3_to_name = {}
    name_to_iso3 = {}
    for row in countries_df.itertuples(index=False):
        iso = str(row.iso3).strip().upper()
        name = str(row.country_name).strip()
        if not iso:
            continue
        iso3_to_name[iso] = name
        key = _normalise_text(name)
        if key:
            name_to_iso3[key] = iso

    hazard_lookup: Dict[str, Hazard] = {}
    for row in shocks_df.itertuples(index=False):
        code = str(row.hazard_code).strip().upper()
        hazard_lookup[code] = Hazard(
            code=code,
            label=str(row.hazard_label).strip(),
            hazard_class=str(row.hazard_class).strip(),
        )

    return iso3_to_name, name_to_iso3, hazard_lookup


def _normalise_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _read_source_frame(
    source: MutableMapping[str, Any],
    *,
    prefer_hxl: bool,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    kind = str(source.get("kind", "csv")).strip().lower() or "csv"
    url = source.get("url")
    if not url:
        raise ValueError("source url missing")

    hxl_tags: Dict[int, str] = {}
    if kind == "xlsx":
        frame = pd.read_excel(url, dtype=str, keep_default_na=False)
    else:
        frame = pd.read_csv(url, dtype=str, keep_default_na=False)

    frame = frame.fillna("")
    if frame.empty:
        return frame, hxl_tags

    first_row = frame.iloc[0]
    if prefer_hxl and any(str(val).strip().startswith("#") for val in first_row):
        hxl_tags = {idx: str(val).strip() for idx, val in enumerate(first_row)}
        frame = frame.iloc[1:].reset_index(drop=True)

    return frame, hxl_tags


def _match_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    hxl_tags: Dict[int, str],
    prefer_hxl: bool,
) -> Optional[str]:
    if not candidates:
        return None
    normalised_candidates = [_normalise_text(cand) for cand in candidates if cand]
    if not normalised_candidates:
        return None

    def _match_from_tags() -> Optional[str]:
        for idx, tag in hxl_tags.items():
            norm_tag = _normalise_text(tag)
            if norm_tag in normalised_candidates and idx < len(frame.columns):
                return frame.columns[idx]
        return None

    if prefer_hxl and hxl_tags:
        matched = _match_from_tags()
        if matched:
            return matched

    for column in frame.columns:
        norm = _normalise_text(column)
        if norm in normalised_candidates:
            return column

    if not prefer_hxl and hxl_tags:
        matched = _match_from_tags()
        if matched:
            return matched

    return None


def _parse_date(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_persons(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in {".", "-"})
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if number <= 0:
        return None
    return int(round(number))


def _normalise_iso3(
    value: Any,
    *,
    iso3_to_name: Dict[str, str],
    name_to_iso3: Dict[str, str],
) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 3 and upper.isalpha() and upper in iso3_to_name:
        return upper
    key = _normalise_text(text)
    return name_to_iso3.get(key)


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _month_end(value: date) -> date:
    return _next_month(value) - timedelta(days=1)


def _iter_month_segments(start: date, end: date) -> Iterable[Tuple[str, int]]:
    if end < start:
        end = start
    current = date(start.year, start.month, 1)
    segments: List[Tuple[str, int]] = []
    while current <= end:
        month_end = _month_end(current)
        seg_start = start if current < start else current
        seg_end = end if month_end > end else month_end
        if seg_end >= seg_start:
            month_key = f"{current.year:04d}-{current.month:02d}"
            days = (seg_end - seg_start).days + 1
            segments.append((month_key, days))
        current = _next_month(current)
    return segments


def _allocate_event(
    start: date,
    end: date,
    total: int,
    *,
    policy: str,
) -> List[Tuple[str, int]]:
    if total <= 0:
        return []
    if policy == "start":
        month_key = f"{start.year:04d}-{start.month:02d}"
        return [(month_key, int(total))]

    segments = list(_iter_month_segments(start, end))
    if not segments:
        month_key = f"{start.year:04d}-{start.month:02d}"
        return [(month_key, int(total))]

    total_days = sum(days for _, days in segments)
    if total_days <= 0:
        month_key = f"{start.year:04d}-{start.month:02d}"
        return [(month_key, int(total))]

    allocations: List[Tuple[str, int]] = []
    running = 0
    for idx, (month_key, days) in enumerate(segments):
        last = idx == len(segments) - 1
        if last:
            value = total - running
        else:
            raw = total * days / total_days
            value = int(round(raw))
            if running + value > total:
                value = max(total - running, 0)
            running += value
        if last:
            running += value
        value = int(value)
        if value > 0:
            allocations.append((month_key, value))
    return allocations


def _infer_hazard_key(
    type_value: Any,
    subtype_value: Any,
    *,
    shock_map: Dict[str, Sequence[str]],
    default_key: str,
) -> str:
    texts = [str(type_value or "").strip(), str(subtype_value or "").strip()]
    norm_texts = [_normalise_text(text) for text in texts if text]

    for hazard_key, keywords in shock_map.items():
        targets = [_normalise_text(keyword) for keyword in keywords if keyword]
        if not targets:
            continue
        for candidate in norm_texts:
            for target in targets:
                if not target:
                    continue
                if candidate == target or candidate.startswith(target) or target in candidate:
                    return hazard_key

    return default_key


def _resolve_hazard(
    hazard_key: str,
    *,
    hazard_lookup: Dict[str, Hazard],
) -> Hazard:
    code = HAZARD_KEY_TO_CODE.get(hazard_key, HAZARD_KEY_TO_CODE.get("other", "OT"))
    if code in hazard_lookup:
        return hazard_lookup[code]
    label = hazard_key.replace("_", " ").title() if hazard_key else "Other"
    return Hazard(code=code, label=label, hazard_class="other")


def _build_event_id(
    iso3: str,
    hazard_code: str,
    month: str,
    value: int,
    source_url: str,
    dis_refs: Sequence[str],
) -> str:
    digest_input = "|".join([
        iso3,
        hazard_code,
        month,
        str(value),
        source_url,
        ",".join(sorted({ref for ref in dis_refs if ref})),
    ])
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
    year, month_part = month.split("-")
    return f"{iso3}-EMDAT-{hazard_code}-affected-{year}-{month_part}-{digest}"


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        fp.write(",".join(CANONICAL_HEADERS) + "\n")


def _write_rows(rows: Sequence[Sequence[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=CANONICAL_HEADERS)
    frame.to_csv(path, index=False)


def main() -> bool:
    if _env_bool("RESOLVER_SKIP_EMDAT", False):
        dbg("RESOLVER_SKIP_EMDAT=1 — writing header only")
        _write_header_only(OUT_PATH)
        return False

    try:
        cfg = load_config()
    except Exception as exc:  # pragma: no cover - defensive
        dbg(f"failed to load config: {exc}")
        _write_header_only(OUT_PATH)
        return False

    sources = cfg.get("sources", [])
    if not sources:
        dbg("no EM-DAT sources configured")
        _write_header_only(OUT_PATH)
        return False

    try:
        iso3_to_name, name_to_iso3, hazard_lookup = load_registries()
    except Exception as exc:  # pragma: no cover - defensive
        dbg(f"failed to load registries: {exc}")
        _write_header_only(OUT_PATH)
        return False

    prefer_hxl = bool(cfg.get("prefer_hxl", False))
    alloc_policy = os.getenv("EMDAT_ALLOC_POLICY", str(cfg.get("allocation_policy", "prorata")))
    alloc_policy = alloc_policy.strip().lower() or "prorata"
    if alloc_policy not in {"prorata", "start"}:
        alloc_policy = "prorata"

    definition_text = (
        DEFAULT_DEFINITION_PRORATA if alloc_policy == "prorata" else DEFAULT_DEFINITION_START
    )
    method = f"{DEFAULT_METHOD_PREFIX}; {alloc_policy}"

    default_hazard_key = str(cfg.get("default_hazard", "other")).strip().lower() or "other"
    shock_map = cfg.get("shock_map", {}) or {}

    allocations: List[Dict[str, Any]] = []
    seen_events: set[Tuple[str, str, date, date]] = set()

    for source in sources:
        try:
            frame, hxl_tags = _read_source_frame(source, prefer_hxl=prefer_hxl)
        except Exception as exc:
            dbg(f"failed to load source {source.get('name')}: {exc}")
            continue

        if frame.empty:
            continue

        country_col = _match_column(frame, source.get("country_keys", []), hxl_tags, prefer_hxl)
        start_col = _match_column(frame, source.get("start_date_keys", []), hxl_tags, prefer_hxl)
        end_col = _match_column(frame, source.get("end_date_keys", []), hxl_tags, prefer_hxl)
        type_col = _match_column(frame, source.get("type_keys", []), hxl_tags, prefer_hxl)
        subtype_col = _match_column(frame, source.get("subtype_keys", []), hxl_tags, prefer_hxl)
        total_col = _match_column(frame, source.get("total_affected_keys", []), hxl_tags, prefer_hxl)
        affected_col = _match_column(frame, source.get("affected_keys", []), hxl_tags, prefer_hxl)
        injured_col = _match_column(frame, source.get("injured_keys", []), hxl_tags, prefer_hxl)
        homeless_col = _match_column(frame, source.get("homeless_keys", []), hxl_tags, prefer_hxl)
        id_col = _match_column(frame, source.get("id_keys", []), hxl_tags, prefer_hxl)
        title_col = _match_column(frame, source.get("title_keys", []), hxl_tags, prefer_hxl)

        dedup_cols = [
            col
            for col in [country_col, start_col, end_col, id_col, type_col, subtype_col, total_col]
            if col
        ]
        if dedup_cols:
            frame = frame.drop_duplicates(subset=dedup_cols)

        publisher = str(source.get("publisher", cfg.get("publisher", "CRED/EM-DAT")))
        source_type = str(source.get("source_type", cfg.get("source_type", "other")))
        source_url = str(source.get("url", ""))
        doc_title = source.get("doc_title") or f"{source.get('name', 'EM-DAT')} event table"

        for record in frame.to_dict("records"):
            iso_candidate = record.get(country_col) if country_col else None
            iso3 = _normalise_iso3(iso_candidate, iso3_to_name=iso3_to_name, name_to_iso3=name_to_iso3)
            if not iso3:
                continue

            start_date = _parse_date(record.get(start_col) if start_col else None)
            if not start_date:
                continue
            end_date = _parse_date(record.get(end_col) if end_col else None) or start_date

            dis_no = str(record.get(id_col, "")) if id_col else ""
            event_name = str(record.get(title_col, "")) if title_col else ""

            hazard_key = _infer_hazard_key(
                record.get(type_col) if type_col else "",
                record.get(subtype_col) if subtype_col else "",
                shock_map=shock_map,
                default_key=default_hazard_key,
            )
            hazard = _resolve_hazard(hazard_key, hazard_lookup=hazard_lookup)

            total_value = None
            if total_col:
                total_value = _parse_persons(record.get(total_col))
            if total_value is None and affected_col:
                total_value = _parse_persons(record.get(affected_col))
            if total_value is None:
                components: List[int] = []
                for col in (affected_col, injured_col, homeless_col):
                    if not col:
                        continue
                    parsed = _parse_persons(record.get(col))
                    if parsed is not None:
                        components.append(parsed)
                if components:
                    total_value = int(sum(components)) if sum(components) > 0 else None

            if total_value is None or total_value <= 0:
                continue

            event_key = (iso3, dis_no, start_date, end_date)
            if dis_no and event_key in seen_events:
                continue
            if dis_no:
                seen_events.add(event_key)

            month_allocations = _allocate_event(start_date, end_date, int(total_value), policy=alloc_policy)
            for month, value in month_allocations:
                if value <= 0:
                    continue
                allocations.append(
                    {
                        "iso3": iso3,
                        "hazard": hazard,
                        "month": month,
                        "value": int(value),
                        "publisher": publisher,
                        "source_type": source_type,
                        "source_url": source_url,
                        "doc_title": doc_title,
                        "dis_ref": dis_no or event_name or f"{iso3}-{month}",
                        "event_name": event_name,
                    }
                )

    if not allocations:
        dbg("no allocations generated")
        _write_header_only(OUT_PATH)
        return False

    ingested_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    max_results = _env_int("RESOLVER_MAX_RESULTS")

    aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for entry in allocations:
        key = (entry["iso3"], entry["hazard"].code, entry["month"])
        bucket = aggregated.setdefault(
            key,
            {
                "iso3": entry["iso3"],
                "hazard": entry["hazard"],
                "month": entry["month"],
                "value": 0,
                "publishers": set(),
                "source_types": set(),
                "source_urls": set(),
                "doc_titles": set(),
                "dis_refs": [],
            },
        )
        bucket["value"] += int(entry["value"])
        bucket["publishers"].add(entry["publisher"])
        bucket["source_types"].add(entry["source_type"])
        if entry["source_url"]:
            bucket["source_urls"].add(entry["source_url"])
        if entry["doc_title"]:
            bucket["doc_titles"].add(entry["doc_title"])
        if entry["dis_ref"]:
            bucket["dis_refs"].append(entry["dis_ref"])

    rows: List[List[Any]] = []
    for (iso3, hazard_code, month), bucket in sorted(aggregated.items()):
        hazard = bucket["hazard"]
        country_name = iso3_to_name.get(iso3, "")
        if not country_name:
            continue
        value = int(round(bucket["value"]))
        if value <= 0:
            continue

        dis_refs = bucket["dis_refs"] or [f"{iso3}-{month}"]
        source_url = " | ".join(sorted(bucket["source_urls"]))
        doc_title = " | ".join(sorted(bucket["doc_titles"])) or "EM-DAT events"
        publisher = " | ".join(sorted(bucket["publishers"]))
        source_type = " | ".join(sorted(bucket["source_types"]))
        event_id = _build_event_id(iso3, hazard.code, month, value, source_url, dis_refs)

        publication_date = f"{month}-01"

        rows.append(
            [
                event_id,
                country_name,
                iso3,
                hazard.code,
                hazard.label,
                hazard.hazard_class,
                "affected",
                SERIES_SEMANTICS,
                str(int(value)),
                "persons",
                month,
                publication_date,
                publisher,
                source_type,
                source_url,
                doc_title,
                definition_text,
                method,
                "low",
                "0",
                ingested_at,
            ]
        )

    if not rows:
        dbg("no valid rows after aggregation")
        _write_header_only(OUT_PATH)
        return False

    if max_results is not None and max_results >= 0:
        rows = rows[: max_results]

    _write_rows(rows, OUT_PATH)
    dbg(f"wrote {len(rows)} EM-DAT rows")
    return True


if __name__ == "__main__":
    main()
