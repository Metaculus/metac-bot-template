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

from .acled_auth import get_auth_header

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
CONFLICT_METRIC = "fatalities_battle_month"

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


def compute_conflict_onset_flags(
    df: pd.DataFrame,
    *,
    iso_col: str = "iso3",
    date_col: str = "month",
    event_type_col: str = "event_type",
    fatalities_col: str = "fatalities",
    battle_event_types: Sequence[str] = ("Battles",),
    lookback_months: int = 12,
    threshold: int = 25,
) -> pd.DataFrame:
    """Return battle fatalities with rolling lookback totals and onset flags."""

    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    lookback = max(int(lookback_months or 0), 1)
    threshold_value = int(threshold or 0)

    work = df[[iso_col, date_col, event_type_col, fatalities_col]].copy()
    work[iso_col] = work[iso_col].astype(str).str.strip().str.upper()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[iso_col, date_col])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    battle_types = {str(v).strip().lower() for v in battle_event_types if str(v).strip()}
    work["event_type_lower"] = work[event_type_col].astype(str).str.strip().str.lower()
    if battle_types:
        work = work[work["event_type_lower"].isin(battle_types)]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    work[fatalities_col] = work[fatalities_col].map(_to_int)
    work["month_period"] = work[date_col].dt.to_period("M")
    work = work.dropna(subset=["month_period"])
    if work.empty:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    grouped = (
        work.groupby([iso_col, "month_period"], as_index=False)[fatalities_col].sum()
    )
    grouped.rename(
        columns={iso_col: "iso3", "month_period": "month", fatalities_col: "battle_fatalities"},
        inplace=True,
    )

    rows: List[pd.DataFrame] = []
    for iso3, group in grouped.groupby("iso3"):
        group = group.sort_values("month")
        start = group["month"].min()
        end = group["month"].max()
        idx = pd.period_range(start, end, freq="M")
        series = pd.Series(0, index=idx, dtype="int64")
        for record in group.itertuples(index=False):
            series.loc[record.month] = int(record.battle_fatalities)
        prev_window = (
            series.shift(1).rolling(window=lookback, min_periods=1).sum().fillna(0).astype(int)
        )
        frame = pd.DataFrame(
            {
                "iso3": iso3,
                "month": idx.strftime("%Y-%m"),
                "battle_fatalities": series.astype(int).to_list(),
                "prev12_battle_fatalities": prev_window.to_list(),
            }
        )
        frame["is_onset"] = (
            (frame["prev12_battle_fatalities"] < threshold_value)
            & (frame["battle_fatalities"] >= threshold_value)
        )
        rows.append(frame)

    if not rows:
        return pd.DataFrame(
            columns=[
                "iso3",
                "month",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    return result


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
    }

    token_keys = {"access_token", "key", "token"}
    source_url = _build_source_url(base_url, params, token_keys)

    records: List[Dict[str, Any]] = []
    session = requests.Session()
    headers = get_auth_header()

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            dbg(f"max pages reached at page {page}")
            break
        params["page"] = page
        dbg(f"fetching page {page}")
        resp = session.get(base_url, params=params, headers=headers, timeout=60)
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


def _make_conflict_rows(
    conflict_stats: pd.DataFrame,
    shocks: pd.DataFrame,
    source_url: str,
    publication_date: str,
    ingested_at: str,
    method_base: str,
    definition_base: str,
    threshold: int,
    lookback_months: int,
    onset_enabled: bool,
) -> List[Dict[str, Any]]:
    if conflict_stats.empty:
        return []

    shocks_index = shocks.set_index("hazard_code")
    lookback = max(int(lookback_months or 0), 1)
    threshold_value = int(threshold or 0)

    rows: List[Dict[str, Any]] = []
    sorted_stats = conflict_stats.sort_values(["iso3", "month"])
    for record in sorted_stats.itertuples(index=False):
        fatalities = int(getattr(record, "battle_fatalities", 0))
        if fatalities <= 0:
            continue
        prev_value = int(getattr(record, "prev12_battle_fatalities", 0))
        hazard_code = HAZARD_KEY_TO_CODE["armed_conflict_escalation"]
        hazard_row = shocks_index.loc[hazard_code]
        digest = _digest(
            [
                record.iso3,
                hazard_code,
                CONFLICT_METRIC,
                record.month,
                str(fatalities),
                source_url,
            ]
        )
        year, month = record.month.split("-")
        definition_text = (
            f"{definition_base} Prev{lookback}m battle fatalities={prev_value}; "
            f"current month battle fatalities={fatalities}; threshold={threshold_value}."
        )
        method_parts = [
            method_base,
            f"battle_fatalities={fatalities}",
            f"prev{lookback}m_battle_fatalities={prev_value}",
            f"threshold={threshold_value}",
        ]
        method = "; ".join(method_parts)
        common_row = {
            "country_name": record.country_name,
            "iso3": record.iso3,
            "metric": CONFLICT_METRIC,
            "series_semantics": SERIES_SEMANTICS,
            "value": fatalities,
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
        rows.append(
            {
                "event_id": f"{record.iso3}-ACLED-{hazard_code}-{CONFLICT_METRIC}-{year}-{month}-{digest}",
                "hazard_code": hazard_code,
                "hazard_label": hazard_row["hazard_label"],
                "hazard_class": hazard_row["hazard_class"],
                **common_row,
            }
        )

        if onset_enabled and bool(getattr(record, "is_onset", False)):
            onset_code = HAZARD_KEY_TO_CODE["armed_conflict_onset"]
            onset_row = shocks_index.loc[onset_code]
            onset_digest = _digest(
                [
                    record.iso3,
                    onset_code,
                    CONFLICT_METRIC,
                    record.month,
                    str(fatalities),
                    source_url,
                    "onset",
                ]
            )
            onset_method = method + "; onset_rule_v1"
            onset_definition = definition_text + " Onset rule triggered."
            onset_common = dict(common_row)
            onset_common.update({"definition_text": onset_definition, "method": onset_method})
            rows.append(
                {
                    "event_id": f"{record.iso3}-ACLED-{onset_code}-{CONFLICT_METRIC}-{year}-{month}-{onset_digest}",
                    "hazard_code": onset_code,
                    "hazard_label": onset_row["hazard_label"],
                    "hazard_class": onset_row["hazard_class"],
                    **onset_common,
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

    onset_cfg = config.get("onset", {})
    onset_enabled = bool(onset_cfg.get("enabled", True))
    lookback_months = int(onset_cfg.get("lookback_months", 12) or 12)
    threshold = int(onset_cfg.get("threshold_battle_deaths", 25) or 25)
    battle_event_types_cfg = onset_cfg.get("battle_event_types", ["Battles"])
    if not battle_event_types_cfg:
        battle_event_types_cfg = ["Battles"]
    battle_types_lower = {str(v).strip().lower() for v in battle_event_types_cfg if str(v).strip()}
    if not battle_types_lower:
        battle_types_lower = {"battles"}

    battle_events = df[df["event_type_lower"].isin(battle_types_lower)]
    battle_totals = (
        battle_events.groupby(
            ["iso3", "country_name", "month", "publisher", "source_type"], as_index=False
        )["fatalities"].sum()
    )
    battle_totals.rename(columns={"fatalities": "battle_fatalities"}, inplace=True)

    onset_flags = pd.DataFrame(
        columns=["iso3", "month", "battle_fatalities", "prev12_battle_fatalities", "is_onset"]
    )
    if onset_enabled and not df.empty:
        onset_input = df[["iso3", "month", "event_type", "fatalities"]].copy()
        onset_flags = compute_conflict_onset_flags(
            onset_input,
            iso_col="iso3",
            date_col="month",
            event_type_col="event_type",
            fatalities_col="fatalities",
            battle_event_types=tuple(battle_event_types_cfg),
            lookback_months=lookback_months,
            threshold=threshold,
        )

    conflict_stats = battle_totals.copy()
    if conflict_stats.empty:
        conflict_stats = pd.DataFrame(
            columns=[
                "iso3",
                "country_name",
                "month",
                "publisher",
                "source_type",
                "battle_fatalities",
                "prev12_battle_fatalities",
                "is_onset",
            ]
        )
    else:
        if not onset_flags.empty:
            conflict_stats = conflict_stats.merge(
                onset_flags[["iso3", "month", "prev12_battle_fatalities", "is_onset"]],
                on=["iso3", "month"],
                how="left",
            )
        if "prev12_battle_fatalities" not in conflict_stats.columns:
            conflict_stats["prev12_battle_fatalities"] = 0
        conflict_stats["prev12_battle_fatalities"] = (
            conflict_stats["prev12_battle_fatalities"].fillna(0).astype(int)
        )
        if "is_onset" not in conflict_stats.columns:
            conflict_stats["is_onset"] = False
        conflict_stats["is_onset"] = conflict_stats["is_onset"].fillna(False).astype(bool)
        if not onset_enabled:
            conflict_stats["is_onset"] = False

    unrest_label = " + ".join(sorted(config.get("unrest_types", []))) or "Protests + Riots"
    battle_label = ", ".join(str(v).strip() for v in battle_event_types_cfg if str(v).strip()) or "Battles"
    definition_base = (
        "ACLED monthly-first aggregation; battle fatalities aggregated from "
        f"{battle_label}; civil unrest events counted from {unrest_label}."
    )
    method_base = (
        "ACLED; monthly-first; battle fatalities aggregated; "
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
        conflict_stats,
        shocks,
        source_url,
        publication_date,
        ingested_at,
        method_base,
        definition_base,
        threshold,
        lookback_months,
        onset_enabled,
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
    ingestion_mode = (os.getenv("RESOLVER_INGESTION_MODE") or "").strip().lower()
    legacy_token = os.getenv("ACLED_TOKEN") or str(config.get("token", ""))
    if legacy_token:
        os.environ.setdefault("ACLED_ACCESS_TOKEN", legacy_token)

    try:
        records, source_url = fetch_events(config)
    except RuntimeError as exc:
        message = f"ACLED auth failed: {exc}"
        if ingestion_mode == "real":
            print(message)
            if os.getenv("RESOLVER_FAIL_ON_STUB_ERROR") == "1":
                raise
            return []
        dbg(message)
        return []
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
