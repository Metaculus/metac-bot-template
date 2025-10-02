#!/usr/bin/env python3
"""HDX (CKAN) connector → resolver/staging/hdx.csv."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml

from resolver.tools.denominators import get_population_record, safe_pct_to_people

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "hdx.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"
DEFAULT_DENOMINATOR = DATA / "population.csv"

COLUMNS = [
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

TOTAL_KEYWORDS = ["total", "overall", "national", "country total", "grand total"]
PIN_TAGS = {"#inneed", "#inneed+num"}
PA_TAGS = {"#affected", "#affected+num"}
MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MULTI = ("multi", "Multi-shock Needs", "all")

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


@dataclass
class Hazard:
    code: str
    label: str
    hclass: str


@dataclass
class SeriesRow:
    as_of_date: str
    value: float
    is_total: bool
    is_percent: bool = False


def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[hdx] {msg}")


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    return countries, shocks


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _month_from(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        try:
            iv = int(value)
            if 1 <= iv <= 12:
                return iv
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9]", "", text)
    if cleaned:
        try:
            iv = int(cleaned)
            if 1 <= iv <= 12:
                return iv
        except Exception:
            pass
    lowered = text.lower().replace(".", "").replace("-", " ").replace("_", " ").strip()
    if lowered in MONTH_NAMES:
        return MONTH_NAMES[lowered]
    for part in lowered.split():
        if part in MONTH_NAMES:
            return MONTH_NAMES[part]
    match = re.match(r"(\d{4})[-/](\d{1,2})", text)
    if match:
        try:
            iv = int(match.group(2))
            if 1 <= iv <= 12:
                return iv
        except Exception:
            return None
    return None


def _year_from(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        iv = int(value)
        if 1900 <= iv <= 2100:
            return f"{iv:04d}"
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return match.group(0)
    return None


def _parse_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = pd.to_datetime(text[:10], errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, dt.datetime):
        return parsed.date()
    if isinstance(parsed, dt.date):
        return parsed
    return None


def _detect_hxl(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    columns = [str(c) for c in df.columns]
    if columns and all(str(c).startswith("#") for c in columns):
        tags = [str(c).strip().lower() for c in columns]
        df = df.copy()
        df.columns = [f"col_{i}" for i in range(len(columns))]
        return df, tags
    hxl_tags: Optional[List[str]] = None
    if not df.empty:
        first_row = df.iloc[0]
        if all(str(v).startswith("#") for v in first_row):
            hxl_tags = [str(v).strip().lower() for v in first_row]
            df = df.iloc[1:].reset_index(drop=True)
    return df, hxl_tags


def _detect_metric(columns: Sequence[str], hxl_tags: Optional[Sequence[str]]) -> Tuple[Optional[str], Optional[str]]:
    if hxl_tags:
        for idx, tag in enumerate(hxl_tags):
            if tag in PIN_TAGS and idx < len(columns):
                return "in_need", columns[idx]
        for idx, tag in enumerate(hxl_tags):
            if tag in PA_TAGS and idx < len(columns):
                return "affected", columns[idx]
    for idx, col in enumerate(columns):
        lowered = str(col).strip().lower()
        if re.search(r"people\s+in\s+need|\bpin\b", lowered):
            return "in_need", columns[idx]
    for idx, col in enumerate(columns):
        lowered = str(col).strip().lower()
        if "affected" in lowered:
            return "affected", columns[idx]
    if hxl_tags:
        for idx, tag in enumerate(hxl_tags):
            if tag in PA_TAGS and idx < len(columns):
                return "affected", columns[idx]
    return None, None


def _detect_time_columns(columns: Sequence[str], hxl_tags: Optional[Sequence[str]]) -> Dict[str, Optional[str]]:
    date_col = month_col = year_col = None
    if hxl_tags:
        for idx, tag in enumerate(hxl_tags):
            if idx >= len(columns):
                continue
            tag_norm = tag.split("+")[0]
            if tag_norm == "#date" and date_col is None:
                date_col = columns[idx]
            elif tag_norm == "#month" and month_col is None:
                month_col = columns[idx]
            elif tag_norm == "#year" and year_col is None:
                year_col = columns[idx]
    for col in columns:
        lowered = str(col).strip().lower()
        if date_col is None and re.search(r"date|period", lowered):
            date_col = col
        if month_col is None and re.search(r"month", lowered):
            month_col = col
        if year_col is None and re.search(r"year", lowered):
            year_col = col
    return {"date": date_col, "month": month_col, "year": year_col}


def _is_total_row(row: pd.Series, columns: Sequence[str], metric_col: Optional[str]) -> bool:
    for col in columns:
        if col == metric_col:
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip().lower()
        if not text:
            continue
        for keyword in TOTAL_KEYWORDS:
            if keyword in text:
                return True
    return False


def _normalize_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        try:
            return int(round(float(value)))
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if "%" in text or "percent" in lowered:
        return None
    cleaned = re.sub(r"[^0-9.+-]", "", text).replace(",", "")
    if not cleaned:
        return None
    try:
        return int(round(float(cleaned)))
    except Exception:
        return None


def _parse_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if "%" not in text and "percent" not in lowered:
        return None
    cleaned = text.replace("%", "")
    cleaned = cleaned.replace("percent", "")
    cleaned = cleaned.replace("Percent", "")
    cleaned = cleaned.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return float(number)


def _year_from_as_of(as_of: str) -> Optional[int]:
    if not as_of:
        return None
    try:
        if len(as_of) == 4:
            return int(as_of)
        return int(as_of.split("-", 1)[0])
    except Exception:
        return None


def extract_metric_timeseries(df: pd.DataFrame, allow_annual: bool = False) -> Optional[Tuple[str, List[SeriesRow]]]:
    """Return (metric, aggregated rows) from an HDX-style table."""

    if df is None or df.empty:
        return None

    df_local = df.copy()
    df_local.columns = [str(c) for c in df_local.columns]

    df_local, hxl_tags = _detect_hxl(df_local)
    columns = list(df_local.columns)

    metric, metric_col = _detect_metric(columns, hxl_tags)
    if not metric or not metric_col:
        return None

    time_cols = _detect_time_columns(columns, hxl_tags)
    if not time_cols["date"] and not time_cols["month"] and not time_cols["year"]:
        return None

    rows: List[SeriesRow] = []

    for _, row in df_local.iterrows():
        date_candidate = None
        if time_cols["date"]:
            date_candidate = _parse_date(row.get(time_cols["date"]))
        month_val = None
        year_val = None

        if time_cols["month"]:
            month_val = _month_from(row.get(time_cols["month"]))
        if time_cols["year"]:
            year_val = _year_from(row.get(time_cols["year"]))

        if date_candidate and (not month_val or not year_val):
            month_val = date_candidate.month
            year_val = f"{date_candidate.year:04d}"
        elif month_val and not year_val and date_candidate:
            year_val = f"{date_candidate.year:04d}"
        elif year_val and not month_val and date_candidate:
            month_val = date_candidate.month

        if month_val and not year_val and time_cols["year"] is None and date_candidate:
            year_val = f"{date_candidate.year:04d}"

        if month_val and year_val:
            as_of = f"{int(year_val):04d}-{int(month_val):02d}"
        elif year_val and allow_annual:
            as_of = year_val
        else:
            continue

        raw_value = row.get(metric_col)
        value = _normalize_value(raw_value)
        if value is None:
            pct = _parse_percent(raw_value)
            if pct is None:
                continue
            rows.append(
                SeriesRow(as_of, float(pct), _is_total_row(row, columns, metric_col), True)
            )
            continue

        rows.append(SeriesRow(as_of, float(value), _is_total_row(row, columns, metric_col), False))

    if not rows:
        return None

    percent_rows = [row for row in rows if row.is_percent]
    count_rows = [row for row in rows if not row.is_percent]

    if percent_rows and not count_rows:
        latest: Dict[str, SeriesRow] = {}
        for item in percent_rows:
            current = latest.get(item.as_of_date)
            if current is None or (item.is_total and not current.is_total):
                latest[item.as_of_date] = item
        aggregated = sorted(latest.values(), key=lambda r: r.as_of_date)
        return metric, aggregated

    if not count_rows:
        return None

    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": None, "parts": []})
    for item in count_rows:
        entry = grouped[item.as_of_date]
        if item.is_total and entry["total"] is None:
            entry["total"] = item.value
        else:
            entry["parts"].append(item.value)

    aggregated: List[SeriesRow] = []
    for as_of_date, info in grouped.items():
        if info["total"] is not None:
            aggregated.append(SeriesRow(as_of_date, float(info["total"]), True, False))
        elif info["parts"]:
            aggregated.append(SeriesRow(as_of_date, float(sum(info["parts"])), False, False))

    aggregated.sort(key=lambda r: r.as_of_date)
    if not aggregated:
        return None
    return metric, aggregated


def infer_hazard(texts: Iterable[str], shocks: pd.DataFrame, keywords_cfg: Dict[str, List[str]]) -> Hazard:
    sample = " ".join([str(t).lower() for t in texts if t])
    matches: List[str] = []
    for key, keywords in keywords_cfg.items():
        for kw in keywords:
            if kw.lower() in sample:
                matches.append(key)
                break
    if not matches:
        return Hazard(*MULTI)
    unique = sorted(set(matches))
    if len(unique) > 1:
        return Hazard(*MULTI)
    key = unique[0]
    code_map = {
        "flood": "FL",
        "drought": "DR",
        "tropical_cyclone": "TC",
        "heat_wave": "HW",
        "armed_conflict_onset": "ACO",
        "armed_conflict_escalation": "ACE",
        "armed_conflict_cessation": "ACC",
        "civil_unrest": "CU",
        "displacement_influx": "DI",
        "economic_crisis": "EC",
        "phe": "PHE",
    }
    hazard_code = code_map.get(key)
    if not hazard_code:
        return Hazard(*MULTI)
    match = shocks[shocks["hazard_code"].str.upper() == hazard_code.upper()]
    if match.empty:
        return Hazard(*MULTI)
    row = match.iloc[0]
    return Hazard(row["hazard_code"], row["hazard_label"], row["hazard_class"])


def _digest(parts: Sequence[Any]) -> str:
    joined = "|".join([str(p) for p in parts])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def _request_json(session: requests.Session, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("success"):
        raise RuntimeError("HDX package_search failed")
    return data.get("result", {})


def _download_resource(session: requests.Session, url: str) -> Optional[pd.DataFrame]:
    if not url:
        return None
    resp = session.get(url, timeout=60)
    if resp.status_code != 200:
        return None
    content_type = resp.headers.get("content-type", "").lower()
    if "excel" in content_type or url.lower().endswith(".xlsx") or url.lower().endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(resp.content))
        except Exception:
            return None
    try:
        text = resp.content.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception:
            return None


def collect_rows() -> List[List[Any]]:
    cfg = load_cfg()
    countries, shocks = load_registries()

    allow_annual = _env_bool("ALLOW_ANNUAL_FALLBACK", bool(cfg.get("allow_annual_fallback", False)))
    max_results_env = os.getenv("RESOLVER_MAX_RESULTS")
    max_results = int(max_results_env) if max_results_env and max_results_env.isdigit() else None

    base_url = os.getenv("HDX_BASE", cfg.get("base_url", "https://data.humdata.org")).rstrip("/")
    query_text = cfg.get("query_text", "people in need")
    topic_filters = cfg.get("topic_filters", [])
    prefer_hxl = bool(cfg.get("prefer_hxl", True))
    max_datasets = int(cfg.get("max_datasets", 200))

    keywords_cfg = cfg.get("shock_keywords", {})
    allow_percent = _env_bool("HDX_ALLOW_PERCENT", False)
    denom_override = os.getenv("HDX_DENOMINATOR_FILE")
    denom_cfg = cfg.get("denominator_file")
    if denom_override:
        denominator_path = denom_override
    elif denom_cfg:
        denominator_path = str(denom_cfg)
    else:
        denominator_path = str(DEFAULT_DENOMINATOR)
    worldpop_hint = os.getenv("WORLDPOP_PRODUCT", "").strip()

    user_agent = os.getenv("RELIEFWEB_APPNAME", "resolver-ingestion")

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    results: List[List[Any]] = []
    seen_resources: set[str] = set()

    search_url = f"{base_url}/api/3/action/package_search"

    for _, crow in countries.iterrows():
        iso3 = crow.get("iso3", "").strip()
        country_name = crow.get("country_name", "").strip()
        if not iso3 or not country_name:
            continue

        params = {
            "q": f"{query_text} \"{country_name}\"",
            "rows": max_datasets,
        }
        if topic_filters:
            params["fq"] = " OR ".join([f"groups:\"{t}\"" for t in topic_filters])

        try:
            payload = _request_json(session, search_url, params)
        except Exception as exc:
            dbg(f"package_search failed for {iso3}: {exc}")
            continue

        datasets = payload.get("results", []) if isinstance(payload, dict) else []

        for dataset in datasets:
            title = dataset.get("title") or dataset.get("name") or "HDX Dataset"
            notes = dataset.get("notes", "")
            tags = [tag.get("name", "") for tag in dataset.get("tags", [])]

            hazard = infer_hazard([title, notes, " ".join(tags)], shocks, keywords_cfg)

            resources = dataset.get("resources", []) or []
            for resource in resources:
                rid = str(resource.get("id"))
                if rid and rid in seen_resources:
                    continue
                if rid:
                    seen_resources.add(rid)
                format_hint = (resource.get("format") or "").lower()
                if format_hint not in ("csv", "xlsx", "xls"):
                    url_hint = str(resource.get("download_url") or resource.get("url") or "").lower()
                    if not (url_hint.endswith(".csv") or url_hint.endswith(".xlsx") or url_hint.endswith(".xls")):
                        continue
                source_url = resource.get("download_url") or resource.get("url") or dataset.get("url")
                if not source_url:
                    continue

                try:
                    df = _download_resource(session, source_url)
                except Exception as exc:
                    dbg(f"download failed for {source_url}: {exc}")
                    continue
                if df is None or df.empty:
                    continue

                parsed = extract_metric_timeseries(df, allow_annual)
                if not parsed and prefer_hxl:
                    parsed = extract_metric_timeseries(df.copy(), allow_annual)
                if not parsed:
                    continue
                metric, series = parsed

                publication_date = (
                    resource.get("last_modified")
                    or resource.get("created")
                    or dataset.get("metadata_modified")
                    or dataset.get("metadata_created")
                    or ""
                )
                doc_title = f"{title} — {resource.get('name') or resource.get('description') or 'Resource'}"

                for item in series:
                    as_of = item.as_of_date
                    if len(as_of) == 4 and not allow_annual:
                        continue
                    value = item.value
                    conversion_note = ""
                    method_value = "api"
                    percent_value: Optional[float] = None
                    if getattr(item, "is_percent", False):
                        if not allow_percent:
                            continue
                        percent_value = float(value)
                        year = _year_from_as_of(as_of)
                        if year is None:
                            continue
                        converted = safe_pct_to_people(
                            percent_value,
                            iso3,
                            year,
                            denom_path=denominator_path,
                        )
                        if converted is None or converted <= 0:
                            dbg(
                                f"unable to convert HDX percent for {iso3} {as_of} ({percent_value})"
                            )
                            continue
                        value = int(converted)
                        record = get_population_record(iso3, year, denominator_path)
                        product_label = (
                            record.product
                            if record and record.product
                            else (worldpop_hint or "population")
                        )
                        denom_label = f"WorldPop {product_label}".strip()
                        if record and record.year < year:
                            detail = f"{denom_label} year={record.year} (fallback for {year})"
                        elif record:
                            detail = f"{denom_label} year={record.year}"
                        else:
                            detail = denom_label
                        conversion_note = (
                            f"Converted from prevalence {percent_value:.2f}% using {detail}."
                        )
                        method_value = f"api; pct→people ({detail})"
                    else:
                        value = int(round(float(value)))
                    if value <= 0:
                        continue
                    definition_base = (
                        f"Aggregated {metric.replace('_', ' ')} figures from HDX resource."
                    )
                    definition_text = (
                        f"{definition_base} {conversion_note}".strip()
                        if conversion_note
                        else definition_base
                    )
                    hazard_code = hazard.code
                    hazard_label = hazard.label
                    hazard_class = hazard.hclass

                    as_of_or_pub = as_of if as_of else publication_date[:7]
                    digest = _digest([iso3, hazard_code, metric, as_of_or_pub, value, source_url])
                    event_id = f"{iso3 or 'UNK'}-HDX-{hazard_code}-{metric}-{as_of_or_pub}-{digest}"

                    ingested_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                    results.append([
                        event_id,
                        country_name,
                        iso3,
                        hazard_code,
                        hazard_label,
                        hazard_class,
                        metric,
                        "stock",
                        str(int(value)),
                        "persons",
                        as_of,
                        publication_date,
                        "HDX (CKAN)",
                        "agency",
                        source_url,
                        doc_title,
                        definition_text,
                        method_value,
                        "med",
                        1,
                        ingested_at,
                    ])

                    if max_results and len(results) >= max_results:
                        return results

    return results


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    out = STAGING / "hdx.csv"

    if os.getenv("RESOLVER_SKIP_HDX") == "1":
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
        print(f"RESOLVER_SKIP_HDX=1 — wrote empty {out}")
        return

    try:
        rows = collect_rows()
    except Exception as exc:
        dbg(f"connector failed: {exc}")
        rows = []

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
        print(f"wrote empty {out}")
        return

    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
