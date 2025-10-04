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
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "reliefweb.yml"

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"

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

NUM_RE = re.compile(
    r"\b(?:about|approx\.?|around)?\s*([0-9][0-9., ]{0,15})(?:\s*(?:people|persons|individuals))?\b",
    re.I,
)

# Future fallback consideration: https://reliefweb.int/updates/rss


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


def _dump(resp: requests.Response) -> str:
    try:
        body = resp.text[:500]
    except Exception:  # pragma: no cover - defensive
        body = "<no-body>"
    try:
        hdrs = dict(resp.headers)
    except Exception:  # pragma: no cover - defensive
        hdrs = {}
    return f"HTTP {resp.status_code}, headers={hdrs}, body[0:500]={body}"


def _is_waf_challenge(resp: requests.Response) -> bool:
    return (
        resp.status_code == 202
        and resp.headers.get("x-amzn-waf-action", "").lower() == "challenge"
    )


def rw_request(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    since: str,
    max_retries: int,
    retry_backoff: float,
    timeout: float,
    challenge_tracker: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    last_err: Optional[str] = None

    # 0) Connectivity probe (optional; ignore body)
    for attempt in range(1, max_retries + 1):
        try:
            probe = session.get(url, params={"limit": 1}, timeout=timeout)
            if DEBUG:
                print(f"[reliefweb] GET probe status={probe.status_code}")
            if _is_waf_challenge(probe):
                challenge_tracker["count"] += 1
                if DEBUG:
                    print("[reliefweb] GET WAF challenge:", _dump(probe))
                if attempt >= max_retries:
                    challenge_tracker["persisted"] = True
                    return None, "empty"
                time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
                continue
            if probe.status_code in (429, 502, 503):
                if DEBUG:
                    print(
                        f"[reliefweb] GET probe rate limit status={probe.status_code}"
                    )
                last_err = _dump(probe)
                if attempt >= max_retries:
                    break
                time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
                continue
            if probe.status_code == 200:
                try:
                    probe.json()
                except ValueError:
                    if DEBUG:
                        print("[reliefweb] GET probe invalid JSON")
                break
            break
        except Exception as exc:
            if DEBUG:
                print("[reliefweb] GET probe exception:", str(exc))
            break

    # 1) Real request with filters via POST
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json(), "post"
            if response.status_code in (429, 502, 503):
                if DEBUG:
                    print(
                        f"[reliefweb] POST attempt {attempt} backoff; status={response.status_code}"
                    )
                time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
                continue
            if _is_waf_challenge(response):
                challenge_tracker["count"] += 1
                if DEBUG:
                    print("[reliefweb] POST WAF challenge:", _dump(response))
                if attempt >= max_retries:
                    challenge_tracker["persisted"] = True
                    return None, "empty"
                time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
                continue
            last_err = _dump(response)
            break
        except Exception as exc:  # pragma: no cover - network failure paths
            last_err = str(exc)
            if DEBUG:
                print(f"[reliefweb] POST exception attempt {attempt}: {last_err}")
            if attempt >= max_retries:
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))

    if challenge_tracker.get("persisted"):
        return None, "empty"

    # 2) GET fallback (single flow with retry loop)
    offset = int(payload.get("offset", 0))
    get_params: List[Tuple[str, str]] = []
    for field in payload.get("fields", {}).get("include", []):
        get_params.append(("fields[include][]", field))

    # Date filter
    get_params.append(("filter[conditions][0][field]", "date.created"))
    get_params.append(("filter[conditions][0][value][from]", since))
    # Language filter
    get_params.append(("filter[conditions][1][field]", "language"))
    get_params.append(("filter[conditions][1][value]", "en"))
    # Format filter(s)
    formats = payload.get("filter", {}).get("conditions", [])
    format_values: List[str] = []
    if len(formats) >= 3:
        format_entry = formats[2]
        value = format_entry.get("value", []) if isinstance(format_entry, dict) else []
        if isinstance(value, list):
            format_values = [str(v) for v in value]
    get_params.append(("filter[conditions][2][field]", "format"))
    for fmt in format_values:
        get_params.append(("filter[conditions][2][value][]", fmt))

    get_params.append(("sort[]", "date.created:desc"))
    get_params.append(("limit", str(payload.get("limit", 100))))
    get_params.append(("offset", str(offset)))

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=get_params, timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive network handling
            last_err = str(exc)
            if attempt >= max_retries:
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
            continue

        if response.status_code == 200:
            try:
                return response.json(), "get"
            except ValueError as exc:  # pragma: no cover - malformed payload
                last_err = str(exc)
                break
        if _is_waf_challenge(response):
            challenge_tracker["count"] += 1
            if DEBUG:
                print("[reliefweb] GET fallback WAF challenge:", _dump(response))
            if attempt >= max_retries:
                challenge_tracker["persisted"] = True
                return None, "empty"
            time.sleep(retry_backoff * attempt + random.uniform(0, 0.5))
            continue
        if response.status_code in (429, 502, 503):
            if DEBUG:
                print(
                    f"[reliefweb] GET fallback rate limit attempt {attempt}; status={response.status_code}"
                )
            if attempt >= max_retries:
                last_err = _dump(response)
                break
            time.sleep((1.5 ** attempt) + random.uniform(0, 0.5))
            continue
        last_err = _dump(response)
        break

    if challenge_tracker.get("persisted"):
        return None, "empty"

    raise RuntimeError(f"ReliefWeb API error: {last_err or 'no 200 after retries'}")


def build_payload(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
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
    payload = {
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
    return payload, since


def map_source_type(rw_type: str, cfg: Dict[str, Any]) -> str:
    return cfg["source_type_map"].get(str(rw_type).lower(), "sitrep")


def pick_dates(rec: Dict[str, Any]) -> Tuple[str, str]:
    dates = rec.get("date", {}) or {}
    created = (dates.get("created") or "").split("T")[0]
    original = (dates.get("original") or "").split("T")[0]
    as_of = original or created or ""
    publication = created or as_of
    return as_of, publication


def make_rows() -> Tuple[List[List[str]], Dict[str, Any]]:
    if os.getenv("RESOLVER_SKIP_RELIEFWEB", "") == "1":
        return [], {"count": 0, "persisted": False, "mode": "empty"}

    cfg = load_cfg()
    countries, shocks = load_registries()
    iso_exclude = {code.upper() for code in cfg.get("iso3_exclude", [])}

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": cfg.get(
                "user_agent",
                "spagbot-resolver/1.0 (+https://github.com/kwyjad/Spagbot_metac-bot)",
            ),
            "Content-Type": "application/json",
            "Accept": cfg.get("accept_header", "application/json"),
        }
    )
    adapter = HTTPAdapter(
        max_retries=Retry(total=0, connect=4, read=4, backoff_factor=0)
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    base_url = cfg["base_url"]
    appname_cfg = cfg.get("appname", "spagbot-resolver")
    appname = os.getenv("RELIEFWEB_APPNAME", appname_cfg)
    url = f"{base_url}?appname={appname}"
    payload, since = build_payload(cfg)
    timeout = float(cfg.get("timeout_seconds", 30))
    max_retries = int(cfg.get("max_retries", 6))
    retry_backoff = float(cfg.get("retry_backoff_seconds", 2))
    challenge_tracker: Dict[str, Any] = {"count": 0, "persisted": False}
    page_pause = float(cfg.get("min_page_pause_seconds", 0.6))

    rows: List[List[str]] = []
    offset = 0
    total = None
    mode_used = "post"

    while True:
        payload["offset"] = offset
        data, mode = rw_request(
            session,
            url,
            payload,
            since,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            timeout=timeout,
            challenge_tracker=challenge_tracker,
        )
        if data is None:
            challenge_tracker["mode"] = mode
            return rows, challenge_tracker
        if mode == "get":
            mode_used = "get"
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
                        "stock",
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
        time.sleep(page_pause)

    challenge_tracker["mode"] = mode_used

    return rows, challenge_tracker


def main() -> None:
    if os.getenv("RESOLVER_SKIP_RELIEFWEB", "0") == "1":
        print("ReliefWeb connector skipped due to RESOLVER_SKIP_RELIEFWEB=1")
        STAGING.mkdir(parents=True, exist_ok=True)
        output = STAGING / "reliefweb.csv"
        pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
        print("[reliefweb] rows=0 challenged=0 mode=empty")
        return

    STAGING.mkdir(parents=True, exist_ok=True)
    output = STAGING / "reliefweb.csv"
    try:
        rows, challenge_tracker = make_rows()
    except RuntimeError as exc:
        message = str(exc)
        if "WAF_CHALLENGE" in message:
            print(
                "ReliefWeb blocked by AWS WAF challenge (202 + x-amzn-waf-action=challenge). "
                "Writing empty CSV and continuing."
            )
            pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
            print("[reliefweb] rows=0 challenged=0 mode=empty")
            return
        raise
    challenged = int(challenge_tracker.get("count", 0))
    mode = challenge_tracker.get("mode", "post")
    if challenge_tracker.get("persisted"):
        print("ReliefWeb WAF challenge persisted; writing empty CSV this run")
        pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
        print(f"[reliefweb] rows=0 challenged={challenged} mode=empty")
        return

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
        print(f"[reliefweb] rows=0 challenged={challenged} mode={mode}")
        return

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output, index=False)
    print(f"[reliefweb] rows={len(df)} challenged={challenged} mode={mode}")


if __name__ == "__main__":
    main()
