#!/usr/bin/env python3
"""WFP mVAM connector producing monthly national food insecurity counts."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
DEFAULT_CONFIG = ROOT / "ingestion" / "config" / "wfp_mvam.yml"
CONFIG = DEFAULT_CONFIG
COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"
OUT_PATH = STAGING / "wfp_mvam.csv"

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

DEFAULT_PUBLISHER = "WFP"
DEFAULT_SOURCE_TYPE = "official"
DEFAULT_UNIT = "persons"
DEFAULT_METRIC = "in_need"
DEFAULT_MULTI = ("MULTI", "Multi-driver Food Insecurity", "multi")
DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"

HAZARD_KEY_TO_CODE = {
    "economic_crisis": "EC",
    "drought": "DR",
    "flood": "FL",
    "armed_conflict_escalation": "ACE",
    "phe": "PHE",
}

@dataclass
class Hazard:
    code: str
    label: str
    hazard_class: str


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[wfp_mvam] {message}")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ingested_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(parts: Sequence[Any]) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        parsed = float(text)
    except Exception:
        return None
    if math.isnan(parsed):
        return None
    return float(parsed)


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _month_key(value: pd.Timestamp) -> Optional[str]:
    if value is None:
        return None
    return f"{value.year:04d}-{value.month:02d}"


def load_config() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_countries() -> pd.DataFrame:
    df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    return df


def load_shocks() -> pd.DataFrame:
    df = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return df


def _write_header_only(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_HEADERS).to_csv(path, index=False)


def _write_rows(rows: List[List[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CANONICAL_HEADERS)
    df.to_csv(path, index=False)


def _maybe_apply_hxl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_row = df.iloc[0]
    if all(isinstance(v, str) and v.startswith("#") for v in first_row):
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = [str(v).strip() for v in first_row]
        return df
    if any(col.startswith("#") for col in df.columns):
        df.columns = [str(c).strip() for c in df.columns]
        return df
    if all(isinstance(v, str) and v.startswith("#") for v in df.columns):
        df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalise_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        mapping[lower] = col
    return mapping


def _find_column(mapping: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if not key:
            continue
        key_lower = key.strip().lower()
        if key_lower in mapping:
            return mapping[key_lower]
    return None


def _extract_text(row: pd.Series, columns: List[str]) -> str:
    parts: List[str] = []
    for col in columns:
        if col not in row:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _load_dataframe(source: Dict[str, Any]) -> Optional[pd.DataFrame]:
    url = source.get("url")
    if not url:
        return None
    kind = (source.get("kind") or "csv").lower()
    appname = os.getenv("RELIEFWEB_APPNAME")
    if url.startswith("http://") or url.startswith("https://"):
        headers = {"User-Agent": appname} if appname else None
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            dbg(f"download failed for {url}: {resp.status_code}")
            return None
        data = io.BytesIO(resp.content)
        if kind == "csv":
            return pd.read_csv(data)
        if kind in {"xlsx", "xls", "excel"}:
            return pd.read_excel(data)
        if kind == "json":
            return pd.read_json(io.BytesIO(resp.content))
        return pd.read_csv(data)
    path = Path(url)
    if not path.exists():
        dbg(f"source path {url} missing")
        return None
    if kind == "csv":
        return pd.read_csv(path)
    if kind in {"xlsx", "xls", "excel"}:
        return pd.read_excel(path)
    if kind == "json":
        return pd.read_json(path)
    return pd.read_csv(path)


def _load_denominator_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    denom_path = Path(path)
    if not denom_path.exists():
        dbg(f"denominator file {path} missing")
        return None
    df = pd.read_csv(denom_path, dtype=str)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    for col in ["population", "pop", "value"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_float)
    if "year" in df.columns:
        df["year"] = df["year"].apply(lambda x: int(float(x)) if _parse_float(x) is not None else None)
    return df


def _lookup_denominator(df: Optional[pd.DataFrame], iso3: str, year: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    data = df.copy()
    if "iso3" not in data.columns:
        return None
    sample = data[data["iso3"].str.upper() == iso3.upper()]
    if sample.empty:
        return None
    if "population" in sample.columns:
        pop_col = "population"
    elif "pop" in sample.columns:
        pop_col = "pop"
    else:
        pop_col = None
    if pop_col is None:
        return None
    if "year" in sample.columns and sample["year"].notna().any():
        exact = sample[sample["year"] == year]
        if exact.empty:
            exact = sample[sample["year"] < year]
        if exact.empty:
            exact = sample
        sample = exact
    population = sample[pop_col].dropna()
    if population.empty:
        return None
    return float(population.iloc[-1])


def _resolve_hazard(code: str, shocks: pd.DataFrame) -> Hazard:
    match = shocks[shocks["hazard_code"].str.upper() == code.upper()]
    if match.empty:
        return Hazard(*DEFAULT_MULTI)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def _infer_hazard(texts: Iterable[str], shocks: pd.DataFrame, cfg: Dict[str, Any]) -> Hazard:
    keywords = cfg.get("shock_keywords", {})
    default_code = str(cfg.get("default_hazard", "multi"))
    sample = " ".join(str(t).lower() for t in texts if t)
    matches: List[str] = []
    for key, words in keywords.items():
        for word in words:
            if word.lower() in sample:
                matches.append(key)
                break
    unique = sorted(set(matches))
    if not unique:
        if default_code.lower() == "multi":
            return Hazard(*DEFAULT_MULTI)
        return _resolve_hazard(default_code, shocks)
    if len(unique) > 1:
        return Hazard(*DEFAULT_MULTI)
    hazard_key = unique[0]
    code = HAZARD_KEY_TO_CODE.get(hazard_key)
    if not code:
        return Hazard(*DEFAULT_MULTI)
    return _resolve_hazard(code, shocks)


def _aggregate_people(group: pd.DataFrame) -> Optional[float]:
    values = group["value"].apply(_parse_float).dropna()
    if values.empty:
        return None
    hint = group["series_hint"].dropna().iloc[0] if not group["series_hint"].dropna().empty else "stock"
    if hint == "incident":
        return float(values.sum())
    return float(values.mean())


def _aggregate_percent(group: pd.DataFrame) -> Optional[float]:
    values = group["value"].apply(_parse_float).dropna()
    if values.empty:
        return None
    return float(values.mean())


def collect_rows() -> List[List[Any]]:
    cfg = load_config()
    countries = load_countries()
    shocks = load_shocks()

    country_lookup = {row.iso3.upper(): row.country_name for row in countries.itertuples()}

    allow_percent = _env_bool("WFP_MVAM_ALLOW_PERCENT", bool(cfg.get("allow_percent", False)))
    emit_stock = _env_bool("WFP_MVAM_STOCK", bool(cfg.get("emit_stock", True)))
    emit_incident = _env_bool("WFP_MVAM_INCIDENT", bool(cfg.get("emit_incident", True)))
    include_first_month = _env_bool(
        "WFP_MVAM_INCLUDE_FIRST_MONTH_DELTA",
        bool(cfg.get("include_first_month_delta", False)),
    )

    priority_env = os.getenv("WFP_MVAM_INDICATOR_PRIORITY")
    if priority_env:
        indicator_priority = [item.strip() for item in priority_env.split(",") if item.strip()]
    else:
        indicator_priority = cfg.get("indicator_priority", [])

    prefer_hxl = bool(cfg.get("prefer_hxl", False))
    denominator_override = os.getenv("WFP_MVAM_DENOMINATOR_FILE")
    denominator_path = denominator_override or cfg.get("denominator_file")
    denominator_table = _load_denominator_table(denominator_path) if allow_percent else None

    records: List[Dict[str, Any]] = []

    sources = cfg.get("sources", []) or []
    for source in sources:
        df = _load_dataframe(source)
        if df is None or df.empty:
            continue
        if prefer_hxl:
            df = _maybe_apply_hxl(df)
        else:
            df.columns = [str(c).strip() for c in df.columns]

        column_map = _normalise_columns(df)
        time_col = _find_column(column_map, source.get("time_keys", []))
        country_col = _find_column(column_map, source.get("country_keys", []))
        admin_cols: List[str] = []
        for key in source.get("admin_keys", []):
            col = _find_column(column_map, [key])
            if col and col not in admin_cols:
                admin_cols.append(col)
        pct_col = _find_column(column_map, source.get("pct_keys", []))
        people_col = _find_column(column_map, source.get("people_keys", []))
        population_col = _find_column(column_map, source.get("population_keys", []))
        driver_cols: List[str] = []
        for key in source.get("driver_keys", []):
            col = _find_column(column_map, [key])
            if col and col not in driver_cols:
                driver_cols.append(col)

        metric = source.get("metric", DEFAULT_METRIC)
        series_hint = source.get("series_hint", "stock")
        publisher = source.get("publisher", DEFAULT_PUBLISHER)
        source_type = source.get("source_type", DEFAULT_SOURCE_TYPE)
        source_url = source.get("url", "")
        doc_title = source.get("name") or "WFP mVAM dataset"
        source_id = source.get("name") or source_url or "wfp_mvam"

        if not time_col or not country_col:
            continue

        for _, row in df.iterrows():
            iso_raw = row.get(country_col)
            if pd.isna(iso_raw):
                continue
            iso3 = str(iso_raw).strip().upper()
            if len(iso3) != 3 or iso3 not in country_lookup:
                continue

            date_raw = row.get(time_col)
            parsed_date = _parse_date(date_raw)
            if parsed_date is None:
                continue
            month = _month_key(parsed_date)
            if not month:
                continue

            indicator_col = None
            indicator_kind = None
            indicator_name = None

            if indicator_priority:
                for candidate in indicator_priority:
                    lower = candidate.lower()
                    if "people" in lower and people_col:
                        indicator_col = people_col
                        indicator_kind = "people"
                        indicator_name = candidate
                        break
                    if "pct" in lower or "%" in lower:
                        if pct_col:
                            indicator_col = pct_col
                            indicator_kind = "percent"
                            indicator_name = candidate
                            break
                    if "rcsi" in lower:
                        rcsi_col = column_map.get("rcsi")
                        if rcsi_col:
                            indicator_col = rcsi_col
                            indicator_kind = "rcsi"
                            indicator_name = candidate
                            break
            if not indicator_col:
                if people_col is not None:
                    indicator_col = people_col
                    indicator_kind = "people"
                    indicator_name = people_col
                elif pct_col is not None:
                    indicator_col = pct_col
                    indicator_kind = "percent"
                    indicator_name = pct_col
                else:
                    continue

            raw_value = row.get(indicator_col)
            parsed_value = _parse_float(raw_value)
            if parsed_value is None:
                continue

            population_value = None
            if population_col:
                population_value = _parse_float(row.get(population_col))

            admin_parts: List[str] = []
            for col in admin_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                text_val = str(val).strip()
                if text_val:
                    admin_parts.append(text_val)
            admin_name = "|".join([p for p in admin_parts if p])

            driver_text = _extract_text(row, driver_cols)
            hazard = _infer_hazard([doc_title, driver_text], shocks, cfg)

            record = {
                "iso3": iso3,
                "country_name": country_lookup.get(iso3, ""),
                "month": month,
                "value": parsed_value,
                "indicator_kind": indicator_kind,
                "indicator_name": indicator_name,
                "series_hint": series_hint,
                "population": population_value,
                "metric": metric,
                "hazard_code": hazard.code,
                "hazard_label": hazard.label,
                "hazard_class": hazard.hazard_class,
                "publisher": publisher,
                "source_type": source_type,
                "source_url": source_url,
                "doc_title": doc_title,
                "driver_text": driver_text,
                "admin_name": admin_name,
                "source_id": source_id,
            }
            records.append(record)

    if not records:
        return []

    df_records = pd.DataFrame(records)

    admin_groups = (
        df_records.groupby(
            [
                "iso3",
                "hazard_code",
                "hazard_label",
                "hazard_class",
                "metric",
                "source_id",
                "month",
                "admin_name",
            ],
            dropna=False,
        )
    )

    admin_aggregated: List[Dict[str, Any]] = []

    for (iso3, hz_code, hz_label, hz_class, metric, source_id, month, admin_name), group in admin_groups:
        if group["indicator_kind"].eq("people").any():
            chosen = group[group["indicator_kind"] == "people"]
            agg_value = _aggregate_people(chosen)
            if agg_value is None:
                continue
            conversion_method = "people direct"
            names = chosen["indicator_name"].dropna()
            indicator_label = names.iloc[0] if not names.empty else "people"
            definition_note = f"Direct people counts ({indicator_label})"
        elif group["indicator_kind"].eq("percent").any() and allow_percent:
            chosen = group[group["indicator_kind"] == "percent"]
            pct_value = _aggregate_percent(chosen)
            if pct_value is None:
                continue
            population_value = chosen["population"].dropna()
            denominator = float(population_value.iloc[0]) if not population_value.empty else None
            denominator_source = "dataset population"
            if denominator is None:
                year = int(month.split("-")[0])
                denominator = _lookup_denominator(denominator_table, iso3, year)
                denominator_source = "denominator file"
            if denominator is None or denominator <= 0:
                continue
            agg_value = round(pct_value / 100.0 * denominator)
            conversion_method = f"pct→people ({denominator_source})"
            definition_note = (
                f"Monthly mean prevalence {pct_value:.2f}% converted using {denominator_source}"
            )
            chosen = chosen.iloc[[0]]
        else:
            continue

        if agg_value is None or agg_value <= 0:
            continue

        row_template = chosen.iloc[0]
        admin_aggregated.append(
            {
                "iso3": iso3,
                "country_name": row_template.get("country_name", ""),
                "hazard_code": hz_code,
                "hazard_label": hz_label,
                "hazard_class": hz_class,
                "metric": metric,
                "source_id": source_id,
                "month": month,
                "admin_name": admin_name,
                "value": float(agg_value),
                "publisher": row_template.get("publisher", DEFAULT_PUBLISHER),
                "source_type": row_template.get("source_type", DEFAULT_SOURCE_TYPE),
                "source_url": row_template.get("source_url", ""),
                "doc_title": row_template.get("doc_title", "WFP mVAM"),
                "driver_text": row_template.get("driver_text", ""),
                "conversion_method": conversion_method,
                "definition_note": definition_note,
            }
        )

    if not admin_aggregated:
        return []

    df_admin = pd.DataFrame(admin_aggregated)

    national_groups = (
        df_admin.groupby(
            ["iso3", "hazard_code", "hazard_label", "hazard_class", "metric", "source_id", "month"],
            dropna=False,
        )
    )

    national_records: List[Dict[str, Any]] = []

    for (iso3, hz_code, hz_label, hz_class, metric, source_id, month), group in national_groups:
        total_value = group["value"].sum()
        if total_value <= 0:
            continue
        conversion_methods = sorted(set(group["conversion_method"].dropna()))
        definition_notes = sorted(set(group["definition_note"].dropna()))
        doc_title = group["doc_title"].dropna().iloc[0] if not group["doc_title"].dropna().empty else "WFP mVAM"
        source_url = group["source_url"].dropna().iloc[0] if not group["source_url"].dropna().empty else ""
        publisher = group["publisher"].dropna().iloc[0] if not group["publisher"].dropna().empty else DEFAULT_PUBLISHER
        source_type = group["source_type"].dropna().iloc[0] if not group["source_type"].dropna().empty else DEFAULT_SOURCE_TYPE
        driver_texts = sorted({text for text in group["driver_text"].dropna() if text})

        national_records.append(
            {
                "iso3": iso3,
                "country_name": group["country_name"].dropna().iloc[0] if not group["country_name"].dropna().empty else "",
                "hazard_code": hz_code,
                "hazard_label": hz_label,
                "hazard_class": hz_class,
                "metric": metric,
                "source_id": source_id,
                "month": month,
                "value": int(round(total_value)),
                "publisher": publisher,
                "source_type": source_type,
                "source_url": source_url,
                "doc_title": doc_title,
                "driver_text": "; ".join(driver_texts),
                "conversion_method": " | ".join(conversion_methods),
                "definition_note": "; ".join(definition_notes),
            }
        )

    if not national_records:
        return []

    national_records.sort(key=lambda r: (r["iso3"], r["hazard_code"], r["metric"], r["source_id"], r["month"]))

    rows: List[List[Any]] = []
    ingested_at = _ingested_timestamp()
    incident_flag = "on" if emit_incident else "off"

    method_template = "mVAM; monthly-first; {conversion}; national roll-up; incident delta=" + incident_flag
    definition_template = (
        "WFP mVAM food insecurity indicators aggregated to monthly national counts. "
        "Daily/weekly values are averaged to month prior to conversion. {note}"
    )

    if emit_stock:
        for rec in national_records:
            as_of = rec["month"]
            value = rec["value"]
            if value <= 0:
                continue
            digest = _digest([rec["iso3"], rec["hazard_code"], rec["metric"], as_of, value, rec["source_url"]])
            year, month = as_of.split("-")
            event_id = f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-{digest}"
            publication_date = f"{as_of}-01"
            rows.append(
                [
                    event_id,
                    rec.get("country_name", ""),
                    rec["iso3"],
                    rec["hazard_code"],
                    rec["hazard_label"],
                    rec["hazard_class"],
                    rec["metric"],
                    "stock",
                    str(int(value)),
                    DEFAULT_UNIT,
                    as_of,
                    publication_date,
                    rec["publisher"],
                    rec["source_type"],
                    rec["source_url"],
                    rec["doc_title"],
                    definition_template.format(note=rec["definition_note"] or ""),
                    method_template.format(conversion=rec["conversion_method"] or "people direct"),
                    "",
                    0,
                    ingested_at,
                ]
            )

    if emit_incident:
        grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        for rec in national_records:
            key = (rec["iso3"], rec["hazard_code"], rec["metric"], rec["source_id"])
            grouped.setdefault(key, []).append(rec)
        for key, group in grouped.items():
            group.sort(key=lambda r: r["month"])
            prev_value: Optional[int] = None
            for idx, rec in enumerate(group):
                current = rec["value"]
                as_of = rec["month"]
                if prev_value is None:
                    prev_value = current
                    if include_first_month and current > 0:
                        digest = _digest([rec["iso3"], rec["hazard_code"], rec["metric"], as_of, 0, rec["source_url"], "incident"])
                        year, month = as_of.split("-")
                        event_id = (
                            f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-incident-{digest}"
                        )
                        publication_date = f"{as_of}-01"
                        rows.append(
                            [
                                event_id,
                                rec.get("country_name", ""),
                                rec["iso3"],
                                rec["hazard_code"],
                                rec["hazard_label"],
                                rec["hazard_class"],
                                rec["metric"],
                                "incident",
                                "0",
                                DEFAULT_UNIT,
                                as_of,
                                publication_date,
                                rec["publisher"],
                                rec["source_type"],
                                rec["source_url"],
                                rec["doc_title"],
                                definition_template.format(note=rec["definition_note"] or ""),
                                method_template.format(conversion=rec["conversion_method"] or "people direct"),
                                "",
                                0,
                                ingested_at,
                            ]
                        )
                    continue
                delta = current - prev_value
                prev_value = current
                if delta <= 0:
                    continue
                digest = _digest([rec["iso3"], rec["hazard_code"], rec["metric"], as_of, delta, rec["source_url"], "incident"])
                year, month = as_of.split("-")
                event_id = f"{rec['iso3']}-WFP-mVAM-{rec['hazard_code']}-{rec['metric']}-{year}-{month}-incident-{digest}"
                publication_date = f"{as_of}-01"
                rows.append(
                    [
                        event_id,
                        rec.get("country_name", ""),
                        rec["iso3"],
                        rec["hazard_code"],
                        rec["hazard_label"],
                        rec["hazard_class"],
                        rec["metric"],
                        "incident",
                        str(int(delta)),
                        DEFAULT_UNIT,
                        as_of,
                        publication_date,
                        rec["publisher"],
                        rec["source_type"],
                        rec["source_url"],
                        rec["doc_title"],
                        definition_template.format(note=rec["definition_note"] or ""),
                        method_template.format(conversion=rec["conversion_method"] or "people direct"),
                        "",
                        0,
                        ingested_at,
                    ]
                )

    return rows


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_WFP_MVAM") == "1":
        dbg("RESOLVER_SKIP_WFP_MVAM=1 — skipping WFP mVAM connector")
        _write_header_only(OUT_PATH)
        return False

    try:
        rows = collect_rows()
    except Exception as exc:
        dbg(f"collect_rows failed: {exc}")
        _write_header_only(OUT_PATH)
        return False

    if not rows:
        dbg("no WFP mVAM rows collected; writing header only")
        _write_header_only(OUT_PATH)
        return False

    _write_rows(rows, OUT_PATH)
    print(f"wrote {OUT_PATH}")
    return True


if __name__ == "__main__":
    main()
