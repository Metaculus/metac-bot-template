#!/usr/bin/env python3
"""IPC (Integrated Food Security Classification) connector."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "ipc.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

OUT_PATH = STAGING / "ipc.csv"

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

SERIES_STOCK = "stock"
SERIES_INCIDENT = "incident"

HAZARD_KEY_TO_CODE = {
    "drought": "DR",
    "flood": "FL",
    "armed_conflict_escalation": "ACE",
    "economic_crisis": "EC",
    "phe": "PHE",
}

MULTI_HAZARD = ("multi", "Multi-driver Food Insecurity", "complex")

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[ipc] {message}")


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except ValueError:
        return None


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        return {"sources": []}
    with open(CONFIG, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    cfg.setdefault("sources", [])
    return cfg


def load_registries() -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Tuple[str, str]]]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    iso3_to_name = {row.iso3: row.country_name for row in countries.itertuples(index=False)}
    hazard_lookup: Dict[str, Tuple[str, str]] = {}
    for row in shocks.itertuples(index=False):
        hazard_lookup[row.hazard_code] = (row.hazard_label, row.hazard_class)
    return countries, iso3_to_name, hazard_lookup


def _normalise_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").strip().lower())


def _is_hxl_row(values: Iterable[Any]) -> bool:
    seen = False
    for value in values:
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        text = str(value).strip()
        if not text or not text.startswith("#"):
            return False
        seen = True
    return seen


@dataclass
class SourceFrame:
    df: pd.DataFrame
    column_map: Dict[str, str]
    hxl_map: Dict[str, str] = field(default_factory=dict)


def _prepare_frame(df: pd.DataFrame) -> SourceFrame:
    df = df.copy()
    hxl_map: Dict[str, str] = {}
    # Drop leading HXL rows and register their tags
    for idx in list(df.index[:2]):
        row = df.loc[idx]
        if _is_hxl_row(row.values):
            tags = [str(v).strip() for v in row.values]
            for col, tag in zip(df.columns, tags):
                if tag:
                    hxl_map[col] = tag
            df = df.drop(index=idx)
    df = df.reset_index(drop=True)
    column_map: Dict[str, str] = {}
    for col in df.columns:
        norm = _normalise_key(col)
        if norm and norm not in column_map:
            column_map[norm] = col
        tag = hxl_map.get(col)
        if tag:
            norm_tag = _normalise_key(tag)
            if norm_tag and norm_tag not in column_map:
                column_map[norm_tag] = col
    return SourceFrame(df=df, column_map=column_map, hxl_map=hxl_map)


def _find_column(frame: SourceFrame, candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        norm = _normalise_key(cand)
        if not norm:
            continue
        if norm in frame.column_map:
            return frame.column_map[norm]
    return None


def _normalise_month(value: Any) -> Optional[str]:
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
    return f"{int(parsed.year):04d}-{int(parsed.month):02d}"


def _normalise_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _expand_period(start: Any, end: Any) -> List[str]:
    start_month = _normalise_month(start)
    end_month = _normalise_month(end)
    if not start_month or not end_month:
        return []
    start_ts = pd.to_datetime(start_month)
    end_ts = pd.to_datetime(end_month)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    months = pd.period_range(start_ts, end_ts, freq="M")
    return [f"{p.year:04d}-{p.month:02d}" for p in months]


def _parse_people(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "%" in text or "percent" in text.lower():
        return None
    cleaned = text.replace(",", "").replace(" ", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if math.isnan(number) or number < 0:
        return None
    return float(number)


def _digest(parts: Sequence[Any]) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _ingested_timestamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class MonthlyRecord:
    iso3: str
    country_name: str
    hazard_code: str
    hazard_label: str
    hazard_class: str
    month: str
    value: float
    source_key: str
    source_url: str
    doc_title: str
    publication_date: str
    definition_parts: List[str]
    drivers_text: str
    method_notes: str
    publisher: str
    source_type: str


def _resolve_hazard(
    drivers: str,
    *,
    cfg: Dict[str, Any],
    hazard_lookup: Dict[str, Tuple[str, str]],
    default_key: str,
) -> Tuple[str, str, str]:
    default_key = default_key or "multi"
    text = str(drivers or "").lower()
    if text:
        for hazard_key, keywords in cfg.get("shock_keywords", {}).items():
            if hazard_key not in HAZARD_KEY_TO_CODE:
                continue
            for keyword in keywords or []:
                if keyword and keyword.lower() in text:
                    code = HAZARD_KEY_TO_CODE[hazard_key]
                    label, hclass = hazard_lookup.get(code, (hazard_key.title(), ""))
                    return code, label, hclass
    # Fallback to configured default
    if default_key in HAZARD_KEY_TO_CODE:
        code = HAZARD_KEY_TO_CODE[default_key]
        label, hclass = hazard_lookup.get(code, (default_key.title(), ""))
        return code, label, hclass
    return MULTI_HAZARD


def _merge_definition(existing: List[str], addition: str) -> List[str]:
    if not addition:
        return existing
    if addition not in existing:
        existing.append(addition)
    return existing


def _load_source_frame(source: Dict[str, Any]) -> SourceFrame:
    if "data" in source:
        df = pd.DataFrame(source.get("data", []))
    else:
        kind = (source.get("kind") or "csv").lower()
        url = source.get("url")
        if not url:
            return SourceFrame(df=pd.DataFrame(), column_map={})
        if kind == "xlsx":
            df = pd.read_excel(url, dtype=str, engine="openpyxl" if os.getenv("IPC_FORCE_OPENPYXL") else None)
        else:
            df = pd.read_csv(url, dtype=str)
    df = df.fillna("")
    return _prepare_frame(df)


def _lookup_value(row: MutableMapping[str, Any], column: Optional[str]) -> Any:
    if column is None:
        return None
    return row.get(column)


def _best_of(row: MutableMapping[str, Any], frame: SourceFrame, keys: Sequence[str]) -> Any:
    column = _find_column(frame, keys)
    if column:
        return row.get(column)
    return None


def _series_source_key(source: Dict[str, Any]) -> str:
    key = source.get("name") or source.get("url") or "ipc"
    return _normalise_key(key) or "ipc"


def _definition_from_row(
    *,
    period_start: str,
    period_end: str,
    phase4: Optional[float],
    phase5: Optional[float],
    drivers: str,
) -> str:
    parts = ["IPC Acute Food Insecurity Phase 3+ population estimate."]
    if period_start and period_end:
        parts.append(f"Period: {period_start} to {period_end} (inclusive of all months).")
    elif period_start:
        parts.append(f"Period starting {period_start}.")
    if phase4 is not None:
        parts.append(f"Phase 4+: {int(phase4):,} (if reported).")
    if phase5 is not None:
        parts.append(f"Phase 5: {int(phase5):,} (if reported).")
    if drivers:
        parts.append(f"Reported drivers: {drivers}.")
    return " ".join(parts)


def _collect_rows(
    *,
    cfg: Dict[str, Any],
    emit_stock: bool,
    emit_incident: bool,
    include_first_delta: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _, iso3_to_name, hazard_lookup = load_registries()
    stock_rows: Dict[Tuple[str, str, str, str], MonthlyRecord] = {}

    default_hazard_key = os.getenv("IPC_DEFAULT_HAZARD", cfg.get("default_hazard", "multi"))

    for source in cfg.get("sources", []):
        frame = _load_source_frame(source)
        if frame.df.empty:
            continue
        dbg(f"processing source {source.get('name', 'ipc')} with {len(frame.df)} rows")

        country_keys = source.get("country_keys") or []
        period_start_keys = source.get("period_start_keys") or []
        period_end_keys = source.get("period_end_keys") or []
        phase3p_keys = source.get("phase3p_keys") or []
        phase4p_keys = source.get("phase4p_keys") or []
        phase5_keys = source.get("phase5_keys") or []
        drivers_keys = source.get("drivers_keys") or []
        title_keys = source.get("title_keys") or ["doc_title", "title", "dataset_title"]
        publication_keys = source.get("publication_keys") or [
            "publication_date",
            "published",
            "pub_date",
            "date",
            "#date+published",
        ]
        source_url_keys = source.get("source_url_keys") or ["source", "source_url", "url"]

        publisher = source.get("publisher", "IPC")
        source_type = source.get("source_type", "official")
        source_url_fallback = source.get("url", "")
        source_key = _series_source_key(source)

        for record in frame.df.to_dict(orient="records"):
            iso_raw = _best_of(record, frame, country_keys)
            iso = str(iso_raw or "").strip().upper()
            if len(iso) == 2:
                iso = ""  # skip alpha-2 until population join is defined
            if not iso or iso not in iso3_to_name:
                continue

            start_val = _best_of(record, frame, period_start_keys)
            end_val = _best_of(record, frame, period_end_keys)
            months = _expand_period(start_val, end_val)
            if not months:
                continue

            phase3_val = _best_of(record, frame, phase3p_keys)
            phase3 = _parse_people(phase3_val)
            if phase3 is None:
                continue

            phase4 = _parse_people(_best_of(record, frame, phase4p_keys))
            phase5 = _parse_people(_best_of(record, frame, phase5_keys))

            drivers_parts: List[str] = []
            for key in drivers_keys:
                val = record.get(frame.column_map.get(_normalise_key(key), key))
                if val:
                    drivers_parts.append(str(val))
            drivers_text = ", ".join(part for part in ("; ".join(drivers_parts)).split(";")) if drivers_parts else ""

            hazard_code, hazard_label, hazard_class = _resolve_hazard(
                drivers_text,
                cfg=cfg,
                hazard_lookup=hazard_lookup,
                default_key=default_hazard_key,
            )

            doc_title = str(_best_of(record, frame, title_keys) or source.get("name") or "IPC")
            publication_date = _normalise_date(_best_of(record, frame, publication_keys))
            source_url = str(_best_of(record, frame, source_url_keys) or source_url_fallback)

            definition = _definition_from_row(
                period_start=_normalise_date(start_val),
                period_end=_normalise_date(end_val),
                phase4=phase4,
                phase5=phase5,
                drivers=drivers_text,
            )

            for month in months:
                key = (iso, hazard_code, month, source_key)
                if key not in stock_rows:
                    stock_rows[key] = MonthlyRecord(
                        iso3=iso,
                        country_name=iso3_to_name.get(iso, iso),
                        hazard_code=hazard_code,
                        hazard_label=hazard_label,
                        hazard_class=hazard_class,
                        month=month,
                        value=0.0,
                        source_key=source_key,
                        source_url=source_url,
                        doc_title=doc_title,
                        publication_date=publication_date,
                        definition_parts=[],
                        drivers_text=drivers_text,
                        method_notes="",
                        publisher=publisher,
                        source_type=source_type,
                    )
                stock_rec = stock_rows[key]
                stock_rec.value += phase3
                stock_rec.source_url = stock_rec.source_url or source_url
                stock_rec.doc_title = stock_rec.doc_title or doc_title
                stock_rec.publication_date = stock_rec.publication_date or publication_date
                stock_rec.definition_parts = _merge_definition(stock_rec.definition_parts, definition)

    if not stock_rows:
        return [], []

    ingested_at = _ingested_timestamp()
    stock_output: List[Dict[str, Any]] = []

    for key in sorted(stock_rows.keys()):
        record = stock_rows[key]
        if not emit_stock:
            continue
        value = float(record.value)
        if value <= 0:
            continue
        event_id = _make_event_id(
            record.iso3,
            record.hazard_code,
            "in_need",
            record.month,
            value,
            record.source_url,
        )
        stock_output.append(
            {
                "event_id": event_id,
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": record.hazard_code,
                "hazard_label": record.hazard_label,
                "hazard_class": record.hazard_class,
                "metric": "in_need",
                "series_semantics": SERIES_STOCK,
                "value": round(value, 3),
                "unit": "persons",
                "as_of_date": record.month,
                "publication_date": record.publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": record.source_url,
                "doc_title": record.doc_title or "IPC Acute Food Insecurity",
                "definition_text": " ".join(record.definition_parts).strip(),
                "method": "IPC; Phase 3+ stock; period→month expansion; national sum",
                "confidence": "",
                "revision": "",
                "ingested_at": ingested_at,
                "_source_key": record.source_key,
            }
        )

    incident_output: List[Dict[str, Any]] = []
    if emit_incident:
        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in stock_output:
            key = (row["iso3"], row["hazard_code"], row.get("_source_key", ""))
            grouped.setdefault(key, []).append(row)
        for group_rows in grouped.values():
            group_rows.sort(key=lambda r: r["as_of_date"])
            prev_value: Optional[float] = None
            for idx, row in enumerate(group_rows):
                current_value = float(row["value"])
                if idx == 0:
                    if include_first_delta and current_value > 0:
                        delta = current_value
                    else:
                        prev_value = current_value
                        continue
                else:
                    delta = max(current_value - (prev_value or 0.0), 0.0)
                prev_value = current_value
                if delta <= 0:
                    continue
                event_id = _make_event_id(
                    row["iso3"],
                    row["hazard_code"],
                    "in_need",
                    row["as_of_date"],
                    delta,
                    row.get("source_url", ""),
                )
                incident_output.append(
                    {
                        **{k: v for k, v in row.items() if k not in {"value", "series_semantics", "event_id", "_source_key"}},
                        "event_id": event_id,
                        "series_semantics": SERIES_INCIDENT,
                        "value": round(delta, 3),
                        "method": "IPC; Phase 3+ stock; period→month expansion; national sum; incident=new PIN (MoM positive delta)",
                        "_source_key": row.get("_source_key", ""),
                    }
                )

    for row in stock_output + incident_output:
        row.pop("_source_key", None)

    return stock_output, incident_output


def _make_event_id(iso3: str, hazard_code: str, metric: str, month: str, value: float, source_url: str) -> str:
    digest = _digest([iso3, hazard_code, metric, month, f"{value:.3f}", source_url])
    year, month_part = month.split("-", 1)
    return f"{iso3}-IPC-{hazard_code}-{metric}-{year}-{month_part}-{digest}"


def _write_rows(rows: Sequence[Dict[str, Any]], *, path: Path) -> None:
    df = pd.DataFrame(rows, columns=CANONICAL_HEADERS)
    if not rows:
        df = pd.DataFrame(columns=CANONICAL_HEADERS)
    else:
        for col in CANONICAL_HEADERS:
            if col not in df.columns:
                df[col] = ""
        df = df[CANONICAL_HEADERS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_header_only(path: Path) -> None:
    _write_rows([], path=path)


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_IPC") == "1":
        dbg("RESOLVER_SKIP_IPC=1 → writing header only")
        _write_header_only(OUT_PATH)
        return False

    try:
        cfg = load_config()
    except Exception as exc:
        dbg(f"failed to load config: {exc}")
        _write_header_only(OUT_PATH)
        return False

    emit_stock = _env_bool("IPC_EMIT_STOCK", bool(cfg.get("emit_stock", True)))
    emit_incident = _env_bool("IPC_EMIT_INCIDENT", bool(cfg.get("emit_incident", True)))
    include_first_delta = _env_bool(
        "IPC_INCLUDE_FIRST_MONTH_DELTA", bool(cfg.get("include_first_month_delta", False))
    )

    try:
        stock_rows, incident_rows = _collect_rows(
            cfg=cfg,
            emit_stock=emit_stock,
            emit_incident=emit_incident,
            include_first_delta=include_first_delta,
        )
    except Exception as exc:
        dbg(f"IPC processing failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    rows: List[Dict[str, Any]] = []
    rows.extend(stock_rows)
    rows.extend(incident_rows)

    max_results = _env_int("RESOLVER_MAX_RESULTS")
    if max_results is not None:
        rows = rows[:max_results]

    _write_rows(rows, path=OUT_PATH)
    print(f"wrote {OUT_PATH}")
    return bool(rows)


if __name__ == "__main__":
    main()
