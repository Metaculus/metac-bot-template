#!/usr/bin/env python3
"""ACLED connector — monthly-first aggregation with conflict onset detection."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "acled.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

OUT_PATH = STAGING / "acled.csv"

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
DOC_TITLE = "ACLED monthly aggregation"

HAZARD_KEY_TO_CODE = {
    "armed_conflict_onset": "ACO",
    "armed_conflict_escalation": "ACE",
    "civil_unrest": "CU",
}

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[acled] {message}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "y", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_config() -> Dict[str, Any]:
    if not CONFIG.exists():
        return {}
    with open(CONFIG, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return countries, shocks


def _normalise_month(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        parsed = pd.NaT
    if pd.isna(parsed):
        return None
    return parsed.to_period("M").strftime("%Y-%m")


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return 0
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    text = text.replace(",", "").replace(" ", "")
    try:
        return int(float(text))
    except Exception:
        return 0


def _digest(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _build_source_url(base_url: str, params: Dict[str, Any], token_keys: Sequence[str]) -> str:
    safe_params = {}
    for key, value in params.items():
        if key in token_keys:
            continue
        safe_params[key] = value
    if not safe_params:
        return base_url
    return f"{base_url}?{urlencode(safe_params, doseq=True)}"


def fetch_events(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    base_url = os.getenv("ACLED_BASE", config.get("base_url", "https://api.acleddata.com"))
    token = os.getenv("ACLED_TOKEN")
    if not token:
        raise RuntimeError("ACLED_TOKEN is required for ACLED ingestion")

    window_days = int(os.getenv("ACLED_WINDOW_DAYS", config.get("window_days", 450)))
    limit = int(os.getenv("ACLED_MAX_LIMIT", config.get("limit", 1000)))
    max_pages = _env_int("RESOLVER_MAX_PAGES")
    max_results = _env_int("RESOLVER_MAX_RESULTS")

    end_date = date.today()
    start_date = end_date - timedelta(days=window_days)

    params: Dict[str, Any] = {
        "event_date": f"{start_date:%Y-%m-%d}|{end_date:%Y-%m-%d}",
        "limit": limit,
        "page": 1,
        "format": "json",
        "access_token": token,
    }

    token_keys = {"access_token", "key", "token"}
    source_url = _build_source_url(base_url, params, token_keys)

    records: List[Dict[str, Any]] = []
    session = requests.Session()

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            dbg(f"max pages reached at page {page}")
            break
        params["page"] = page
        dbg(f"fetching page {page}")
        resp = session.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        try:
            payload = resp.json() or {}
        except ValueError as exc:  # pragma: no cover - JSON decode errors
            dbg("ACLED payload was not valid JSON")
            raise RuntimeError("ACLED payload was not valid JSON") from exc
        if not isinstance(payload, dict):
            dbg(f"ACLED payload unexpected type: {type(payload)!r}")
            raise RuntimeError("ACLED payload missing expected fields (data/results/count)")
        if page == 1:
            expected = {"data", "results", "count"}
            present = sorted(key for key in expected if key in payload)
            if not present:
                dbg(f"ACLED payload missing expected keys; received: {sorted(payload.keys())}")
                raise RuntimeError("ACLED payload missing expected fields (data/results/count)")
            dbg(f"ACLED connectivity ok; payload keys include: {', '.join(present)}")
        data = payload.get("data") or payload.get("results") or []
        if not isinstance(data, list):
            raise RuntimeError("Unexpected ACLED payload structure")
        if not data:
            break
        records.extend(data)
        dbg(f"page {page} returned {len(data)} rows (total={len(records)})")
        if max_results is not None and len(records) >= max_results:
            dbg("max results reached; truncating")
            records = records[:max_results]
            break
        if len(data) < limit:
            break
        page += 1

    return records, source_url


def _extract_first(record: MutableMapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            value = record.get(key)
            if value is not None:
                return value
    return None


def _prepare_dataframe(
    records: Sequence[MutableMapping[str, Any]],
    config: Dict[str, Any],
    countries: pd.DataFrame,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["iso3", "country_name", "event_type", "month", "fatalities", "notes"])

    keys = config.get("keys", {})
    iso_keys = keys.get("iso3", [])
    country_keys = keys.get("country", [])
    date_keys = keys.get("date", [])
    event_type_keys = keys.get("event_type", [])
    fatalities_keys = keys.get("fatalities", [])
    notes_keys = keys.get("notes", [])

    iso_lookup = {str(row.iso3).strip().upper(): str(row.country_name)
                  for row in countries.itertuples(index=False)}
    name_lookup = {str(row.country_name).strip().lower(): str(row.iso3).strip().upper()
                   for row in countries.itertuples(index=False)}

    rows: List[Dict[str, Any]] = []
    for record in records:
        event_date = _extract_first(record, date_keys)
        month = _normalise_month(event_date)
        if not month:
            continue
        iso = str(_extract_first(record, iso_keys) or "").strip().upper()
        if not iso:
            country_name = str(_extract_first(record, country_keys) or "").strip()
            iso = name_lookup.get(country_name.lower(), "") if country_name else ""
        if not iso or iso not in iso_lookup:
            continue
        country_name = iso_lookup[iso]
        event_type = str(_extract_first(record, event_type_keys) or "").strip()
        fatalities = _to_int(_extract_first(record, fatalities_keys))
        notes = str(_extract_first(record, notes_keys) or "").strip()
        rows.append(
            {
                "iso3": iso,
                "country_name": country_name,
                "event_type": event_type,
                "month": month,
                "fatalities": fatalities,
                "notes": notes,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["iso3", "country_name", "event_type", "month", "fatalities", "notes"])

    df = pd.DataFrame(rows)
    df["event_type_lower"] = df["event_type"].str.lower()
    return df


def _parse_participants(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    participants_cfg = config.get("participants", {})
    enabled_cfg = bool(participants_cfg.get("enabled", False))
    enabled_env = _env_bool("ACLED_PARSE_PARTICIPANTS", enabled_cfg)
    if not enabled_env:
        return pd.Series([None] * len(df))

    regex_text = participants_cfg.get("regex", "")
    if not regex_text:
        return pd.Series([None] * len(df))
    pattern = re.compile(regex_text, flags=re.IGNORECASE)

    values: List[Optional[int]] = []
    for note in df["notes"].fillna(""):
        match = pattern.search(str(note))
        if not match:
            values.append(None)
            continue
        raw = match.group(1)
        if raw is None:
            values.append(None)
            continue
        raw = raw.replace(",", "").replace(".", "")
        try:
            values.append(int(raw))
        except Exception:
            values.append(None)
    return pd.Series(values)


def _compute_battle_windows(battle_df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[int, int]]:
    if battle_df.empty:
        return {}
    battle_df = battle_df.copy()
    battle_df["period"] = pd.PeriodIndex(battle_df["month"], freq="M")
    result: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for iso3, group in battle_df.groupby("iso3"):
        group = group.sort_values("period")
        idx = pd.period_range(group["period"].min(), group["period"].max(), freq="M")
        series = pd.Series(0, index=idx)
        for _, row in group.iterrows():
            series.loc[row["period"]] = int(row["fatalities"])
        prev12 = series.shift(1).rolling(window=12, min_periods=1).sum().fillna(0)
        for period in idx:
            prev_value = int(prev12.loc[period])
            current_value = int(series.loc[period])
            result[(iso3, period.strftime("%Y-%m"))] = (prev_value, current_value)
    return result


def _make_conflict_rows(
    totals: pd.DataFrame,
    battle_windows: Dict[Tuple[str, str], Tuple[int, int]],
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    shocks_index = shocks.set_index("hazard_code")
    for record in totals.sort_values(["iso3", "month"]).itertuples(index=False):
        key = (record.iso3, record.month)
        prev12, current = battle_windows.get(key, (0, 0))
        hazard_key = "armed_conflict_onset" if prev12 < 25 and current >= 25 else "armed_conflict_escalation"
        hazard_code = HAZARD_KEY_TO_CODE[hazard_key]
        hazard_row = shocks_index.loc[hazard_code]
        value = int(record.fatalities)
        digest = _digest([record.iso3, hazard_code, "fatalities", record.month, str(value), source_url])
        year, month = record.month.split("-")
        definition_text = (
            f"{definition_base} Onset rule inputs: prev12m={prev12}, current_month={current}."
        )
        method = (
            f"{method_base}; onset_prev12m={prev12}; onset_current_month={current}"
        )
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-fatalities-{year}-{month}-{digest}",
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                "metric": "fatalities",
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": "persons",
                "as_of_date": record.month,
                "publication_date": publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": source_url,
                "doc_title": DOC_TITLE,
                "definition_text": definition_text,
                "method": method,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _make_unrest_rows(
    unrest_counts: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
) -> List[Dict[str, Any]]:
    if unrest_counts.empty:
        return []
    shocks_index = shocks.set_index("hazard_code")
    hazard_code = HAZARD_KEY_TO_CODE["civil_unrest"]
    hazard_row = shocks_index.loc[hazard_code]
    rows: List[Dict[str, Any]] = []
    for record in unrest_counts.sort_values(["iso3", "month"]).itertuples(index=False):
        value = int(record.events)
        digest = _digest([record.iso3, hazard_code, "events", record.month, str(value), source_url])
        year, month = record.month.split("-")
        definition_text = f"{definition_base} Metric=events aggregated for unrest types."
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-events-{year}-{month}-{digest}",
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                "metric": "events",
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": "events",
                "as_of_date": record.month,
                "publication_date": publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": source_url,
                "doc_title": DOC_TITLE,
                "definition_text": definition_text,
                "method": method_base,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _make_participant_rows(
    participants: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
) -> List[Dict[str, Any]]:
    if participants.empty:
        return []
    shocks_index = shocks.set_index("hazard_code")
    hazard_code = HAZARD_KEY_TO_CODE["civil_unrest"]
    hazard_row = shocks_index.loc[hazard_code]
    rows: List[Dict[str, Any]] = []
    for record in participants.sort_values(["iso3", "month"]).itertuples(index=False):
        value = int(record.participants)
        digest = _digest([record.iso3, hazard_code, "participants", record.month, str(value), source_url])
        year, month = record.month.split("-")
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-participants-{year}-{month}-{digest}",
                "country_name": record.country_name,
                "iso3": record.iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                "metric": "participants",
                "series_semantics": SERIES_SEMANTICS,
                "value": value,
                "unit": "persons",
                "as_of_date": record.month,
                "publication_date": publication_date,
                "publisher": record.publisher,
                "source_type": record.source_type,
                "source_url": source_url,
                "doc_title": DOC_TITLE,
                "definition_text": f"{definition_base} Metric=participants from event notes heuristic.",
                "method": method_base,
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )
    return rows


def _aggregate_participants(df: pd.DataFrame, participants_values: pd.Series, aggregate: str) -> pd.DataFrame:
    df = df.copy()
    df["participants_value"] = participants_values
    df = df.dropna(subset=["participants_value"])
    if df.empty:
        return pd.DataFrame(columns=["iso3", "country_name", "month", "participants"])
    if aggregate == "median":
        grouped = df.groupby(["iso3", "country_name", "month"], as_index=False)["participants_value"].median()
    else:
        grouped = df.groupby(["iso3", "country_name", "month"], as_index=False)["participants_value"].sum()
    grouped.rename(columns={"participants_value": "participants"}, inplace=True)
    grouped["participants"] = grouped["participants"].round().astype(int)
    return grouped


def _build_rows(
    records: Sequence[MutableMapping[str, Any]],
    config: Dict[str, Any],
    countries: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
) -> List[Dict[str, Any]]:
    df = _prepare_dataframe(records, config, countries)
    if df.empty:
        return []

    publisher = config.get("publisher", "ACLED")
    source_type = config.get("source_type", "other")
    df["publisher"] = publisher
    df["source_type"] = source_type

    unrest_types = {str(v).strip().lower() for v in config.get("unrest_types", []) if v}
    df_unrest = df[df["event_type_lower"].isin(unrest_types)]
    unrest_counts = (
        df_unrest.groupby(["iso3", "country_name", "month", "publisher", "source_type"], as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )

    totals = (
        df.groupby(["iso3", "country_name", "month", "publisher", "source_type"], as_index=False)["fatalities"].sum()
    )
    battle_df = (
        df[df["event_type_lower"] == "battles"]
        .groupby(["iso3", "country_name", "month"], as_index=False)["fatalities"].sum()
    )
    battle_windows = _compute_battle_windows(battle_df)

    unrest_label = " + ".join(sorted(config.get("unrest_types", []))) or "Protests + Riots"
    definition_base = (
        "ACLED monthly-first aggregation; fatalities sum across all event types; "
        f"civil unrest events counted from {unrest_label}."
    )
    method_base = (
        "ACLED; monthly-first; fatality sum across all event types; "
        f"unrest events={unrest_label}; onset rule applied"
    )

    participants_values = _parse_participants(df, config)
    participants_rows: List[Dict[str, Any]] = []
    if participants_values.notna().any():
        aggregate = config.get("participants", {}).get("aggregate", "sum").lower()
        participants_totals = _aggregate_participants(df, participants_values, aggregate)
        if not participants_totals.empty:
            participants_totals["publisher"] = publisher
            participants_totals["source_type"] = source_type
            participants_rows = _make_participant_rows(
                participants_totals,
                shocks,
                source_url,
                publication_date,
                ingested_at,
                method_base,
                definition_base,
            )

    conflict_rows = _make_conflict_rows(
        totals,
        battle_windows,
        shocks,
        source_url,
        publication_date,
        ingested_at,
        method_base,
        definition_base,
    )
    unrest_rows = _make_unrest_rows(
        unrest_counts,
        shocks,
        source_url,
        publication_date,
        ingested_at,
        method_base,
        definition_base,
    )

    rows = conflict_rows + unrest_rows + participants_rows
    rows.sort(key=lambda r: (r["iso3"], r["as_of_date"], r["metric"]))
    return rows


def collect_rows() -> List[Dict[str, Any]]:
    config = load_config()
    countries, shocks = load_registries()
    records, source_url = fetch_events(config)
    if not records:
        return []
    publication_date = date.today().isoformat()
    ingested_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return _build_rows(records, config, countries, shocks, source_url, publication_date, ingested_at)


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CANONICAL_HEADERS)


def _write_rows(rows: Sequence[MutableMapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CANONICAL_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_ACLED") == "1":
        dbg("RESOLVER_SKIP_ACLED=1 — skipping ACLED pull")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:  # fail-soft
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no ACLED rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, OUT_PATH)
    dbg(f"wrote {len(rows)} ACLED rows")
    return True


if __name__ == "__main__":
    main()
