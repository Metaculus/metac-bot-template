#!/usr/bin/env python3
"""
ReliefWeb API → staging/reliefweb.csv

- Queries recent reports (last N days) with basic filters
- Paginates with retries and backoff
- Maps to (iso3, hazard_code) via keyword heuristics
- Extracts PIN/PA (or cases) from title/summary using regex
- Writes canonical staging CSV expected by our exporter/validator

Usage:
  python resolver/ingestion/reliefweb_client.py
"""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "reliefweb.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

# Canonical output columns
COLUMNS = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
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

NUM_RE = re.compile(
    r"\b(?:about|approx\.?|around)?\s*([0-9][0-9., ]{0,15})(?:\s*(?:people|persons|individuals))?\b",
    re.I,
)


def load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    countries = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS, dtype=str).fillna("")
    countries["country_norm"] = countries["country_name"].str.strip().str.lower()
    return countries, shocks


def norm(text: str) -> str:
    return (text or "").strip().lower()


def detect_hazard(text: str, cfg: Dict[str, Any]) -> Optional[str]:
    sample = norm(text)
    for code, keywords in cfg["hazard_keywords"].items():
        for keyword in keywords:
            if keyword in sample:
                return code
    return None


def extract_metric_value(text: str, cfg: Dict[str, Any]) -> Optional[Tuple[str, int, str, str]]:
    """Return (metric, value, unit, matched_phrase)."""

    combined = text or ""
    lowered = combined.lower()

    for pattern_cfg in cfg["metric_patterns"]:
        metric = pattern_cfg["metric"]
        unit = pattern_cfg["unit"]
        for phrase in pattern_cfg["patterns"]:
            idx = lowered.find(phrase)
            if idx == -1:
                continue
            window_start = max(0, idx - 80)
            window_end = min(len(combined), idx + len(phrase) + 80)
            window = combined[window_start:window_end]
            match = NUM_RE.search(window)
            if not match:
                continue
            raw_value = match.group(1)
            cleaned = raw_value.replace(",", "").replace(" ", "")
            cleaned = re.sub(r"[^0-9]", "", cleaned)
            try:
                value = int(cleaned)
            except ValueError:
                continue
            if value < 0:
                continue
            return metric, value, unit, phrase
    return None


def iso3_from_reliefweb_countries(
    countries_df: pd.DataFrame, rw_countries: List[Dict[str, Any]]
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for country in rw_countries or []:
        name = country.get("name") or country.get("shortname") or ""
        if not name:
            continue
        match = countries_df[countries_df["country_name"].str.lower() == name.lower()]
        if match.empty:
            continue
        rows.append((match.iloc[0]["country_name"], match.iloc[0]["iso3"]))
    return rows


def rw_request(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    tries: int = 4,
    backoff: float = 1.2,
) -> Dict[str, Any]:
    for attempt in range(1, tries + 1):
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 502, 503):
            time.sleep((backoff ** attempt) + 0.3 * attempt)
            continue
        message = response.text.strip()
        snippet = message[:500]
        raise RuntimeError(
            f"ReliefWeb API error: HTTP {response.status_code}: {snippet}"
        )
    raise RuntimeError("ReliefWeb API failed after retries (no 200 after backoff)")


def build_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    since = (
        dt.datetime.now(dt.UTC) - dt.timedelta(days=int(cfg["window_days"]))
    ).strftime("%Y-%m-%dT00:00:00Z")
    fields = [
        "id",
        "title",
        "body",
        "date.created",
        "date.original",
        "date.changed",
        "url",
        "source",
        "format",
        "type",
        "disaster_type",
        "country",
    ]
    return {
        "appname": cfg["appname"],
        "filter": {
            "operator": "AND",
            "conditions": [
                {"field": "date.created", "value": {"from": since}},
                {"field": "language", "value": "en"},
                {"field": "format", "value": ["Report", "Appeal", "Update"]},
            ],
        },
        "fields": {"include": fields},
        "sort": ["date.created:desc"],
        "limit": int(cfg["page_size"]),
    }


def map_source_type(rw_type: str, cfg: Dict[str, Any]) -> str:
    return cfg["source_type_map"].get(str(rw_type).lower(), "sitrep")


def pick_dates(rec: Dict[str, Any]) -> Tuple[str, str]:
    dates = rec.get("date", {}) or {}
    created = (dates.get("created") or "").split("T")[0]
    original = (dates.get("original") or "").split("T")[0]
    as_of = original or created or ""
    publication = created or as_of
    return as_of, publication


def make_rows() -> List[List[str]]:
    cfg = load_cfg()
    countries, shocks = load_registries()
    iso_exclude = {code.upper() for code in cfg.get("iso3_exclude", [])}

    headers = {
        "User-Agent": cfg.get("user_agent", "spagbot-resolver"),
        "Content-Type": "application/json",
        "Accept": cfg.get("accept_header", "application/json"),
    }
    base_url = cfg["base_url"]
    appname = cfg.get("appname", "spagbot-resolver")
    url = f"{base_url}?appname={appname}"
    payload = build_payload(cfg)

    rows: List[List[str]] = []
    offset = 0
    total = None

    while True:
        payload["offset"] = offset
        data = rw_request(url, payload, headers)
        total = total or data.get("totalCount", 0)
        items = data.get("data", [])
        if not items:
            break

        for item in items:
            report_id = str(item.get("id"))
            fields = item.get("fields", {}) or {}
            title = fields.get("title", "")
            body = fields.get("body", "")
            report_type_entries = fields.get("type") or [{}]
            report_type = report_type_entries[0].get("name", "report")
            sources = fields.get("source") or []
            source_name = sources[0].get("shortname") if sources else "OCHA"

            iso_pairs = iso3_from_reliefweb_countries(
                countries, fields.get("country") or []
            )
            if not iso_pairs:
                continue

            hazard_code = detect_hazard(f"{title} {body}", cfg)
            if not hazard_code:
                continue

            shock_row = shocks[shocks["hazard_code"] == hazard_code]
            if shock_row.empty:
                continue
            hazard_label = shock_row.iloc[0]["hazard_label"]
            hazard_class = shock_row.iloc[0]["hazard_class"]

            text_for_metrics = " ".join([title or "", body or ""])
            metric_info = extract_metric_value(text_for_metrics, cfg)
            if not metric_info:
                continue
            metric, value, unit, phrase = metric_info

            as_of, publication = pick_dates(fields)
            source_url = fields.get("url", "")
            doc_title = title

            ingested_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            for country_name, iso3 in iso_pairs:
                if iso3 in iso_exclude:
                    continue
                event_id = f"{iso3}-{hazard_code}-rw-{report_id}"
                rows.append(
                    [
                        event_id,
                        country_name,
                        iso3,
                        hazard_code,
                        hazard_label,
                        hazard_class,
                        metric,
                        str(value),
                        unit,
                        as_of,
                        publication,
                        source_name or "OCHA",
                        map_source_type(report_type, cfg),
                        source_url,
                        doc_title,
                        f"Extracted {metric} via phrase '{phrase}' in ReliefWeb report.",
                        "api",
                        "med",
                        1,
                        ingested_at,
                    ]
                )

        offset += len(items)
        if offset >= total:
            break
        time.sleep(0.25)

    return rows


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    output = STAGING / "reliefweb.csv"
    rows = make_rows()

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
        print(f"wrote empty {output}")
        return

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output, index=False)
    print(f"wrote {output} rows={len(df)}")


if __name__ == "__main__":
    main()
