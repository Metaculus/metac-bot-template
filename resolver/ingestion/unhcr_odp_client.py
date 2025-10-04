#!/usr/bin/env python3
"""UNHCR Operational Data Portal (ODP) arrivals → staging/unhcr_odp.csv.

Adds debug counters, optional country whitelist for narrow test runs, and more
structured logging to troubleshoot payload changes.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests

from resolver.ingestion._manifest import ensure_manifest_for_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "unhcr_odp.yml"
COUNTRIES_CSV = DATA / "countries.csv"

OUT_PATH = STAGING / "unhcr_odp.csv"
BASE = "https://data.unhcr.org"
DEFAULT_SITUATION_PATH = "/en/situations/europe-sea-arrivals"
UA = "spagbot-resolver/1.0 (github.com/kwyjad/Spagbot_metac-bot)"

CANONICAL_HEADER = [
    "source",
    "source_event_id",
    "as_of_date",
    "country_iso3",
    "country_name",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric_name",
    "metric_unit",
    "series_semantics",
    "value",
    "evidence_url",
    "evidence_label",
]

HAZARD_CODE = "DI"
HAZARD_LABEL = "Displacement Influx (cross-border from neighbouring country)"
HAZARD_CLASS = "human-induced"

logger = logging.getLogger(__name__)
RESOLVER_DEBUG = bool(int(os.getenv("RESOLVER_DEBUG", "0") or 0))
if RESOLVER_DEBUG:
    logger.setLevel(logging.DEBUG)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def _dbg(msg: str, **kv: object) -> None:
    if not RESOLVER_DEBUG:
        return
    if kv:
        suffix = " ".join(f"{key}={value}" for key, value in kv.items())
        logger.debug("%s | %s", msg, suffix)
    else:
        logger.debug("%s", msg)


class _LocationLinkParser(HTMLParser):
    """Minimal HTML parser that extracts location links from the situation page."""

    def __init__(self) -> None:
        super().__init__()
        self._capture: bool = False
        self._current_href: Optional[str] = None
        self._chunks: List[str] = []
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = {k: (v or "") for k, v in attrs}
        href = attrs_dict.get("href", "")
        if not href:
            return
        if re.match(r"^/en/situations/.+/location/\d+$", href):
            self._capture = True
            self._current_href = href
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._capture and data.strip():
            self._chunks.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._capture:
            return
        name = " ".join(self._chunks).strip()
        if self._current_href:
            self.links.append((name, self._current_href))
        self._capture = False
        self._current_href = None
        self._chunks = []


@dataclass
class CountryIndex:
    name_to_iso: Dict[str, str]
    iso_to_name: Dict[str, str]

    @classmethod
    def load(cls) -> "CountryIndex":
        name_to_iso: Dict[str, str] = {}
        iso_to_name: Dict[str, str] = {}
        if not COUNTRIES_CSV.exists():
            return cls(name_to_iso, iso_to_name)
        with open(COUNTRIES_CSV, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iso = (row.get("iso3") or "").strip().upper()
                name = (row.get("country_name") or "").strip()
                if not iso or not name:
                    continue
                iso_to_name[iso] = name
                norm = _normalize_name(name)
                if norm:
                    name_to_iso.setdefault(norm, iso)
                    alt = norm.replace("and", "")
                    if alt and alt != norm:
                        name_to_iso.setdefault(alt, iso)
                    alt_no_the = norm.replace("the", "")
                    if alt_no_the and alt_no_the != norm:
                        name_to_iso.setdefault(alt_no_the, iso)
        return cls(name_to_iso, iso_to_name)

    def lookup(self, raw_name: str) -> Tuple[str, str]:
        """Return (iso3, country_name) for a given raw label."""
        if not raw_name:
            return "", ""
        norm = _normalize_name(raw_name)
        iso = self.name_to_iso.get(norm, "")
        if not iso and norm:
            trimmed = re.sub(r"\s*\(.*?\)\s*", "", raw_name).strip()
            norm2 = _normalize_name(trimmed)
            iso = self.name_to_iso.get(norm2, "")
        if not iso and norm:
            collapsed = norm.replace("and", "")
            iso = self.name_to_iso.get(collapsed, "")
        if not iso:
            return "", raw_name.strip()
        return iso, self.iso_to_name.get(iso, raw_name.strip())


def _normalize_name(name: str) -> str:
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*\(.*?\)\s*", "", cleaned)
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^a-z]", "", cleaned)
    return cleaned


def _yaml_safe_load(handle) -> Dict[str, object]:
    import yaml  # local import to avoid import-time dependency if unused

    data = yaml.safe_load(handle)
    return data or {}


def _load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        return {"situation_path": DEFAULT_SITUATION_PATH}
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        cfg = _yaml_safe_load(handle)
    if not isinstance(cfg, dict):
        cfg = {}
    if not cfg.get("situation_path"):
        cfg["situation_path"] = DEFAULT_SITUATION_PATH
    return cfg


def _get(
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 4,
    backoff: float = 1.5,
    request_tracker: Optional[Dict[str, int]] = None,
    debug_every: int = 5,
) -> requests.Response:
    merged_headers = {"User-Agent": UA, "Accept": "text/html,application/json"}
    if headers:
        merged_headers.update(headers)
    last_exc: Optional[Exception] = None
    if request_tracker is not None:
        request_tracker.setdefault("count", 0)
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=merged_headers, timeout=30)
        except requests.RequestException as exc:  # noqa: BLE001
            last_exc = exc
            _dbg("request_error", url=url, error=str(exc))
            time.sleep(backoff * (attempt + 1))
            continue
        if request_tracker is not None:
            request_tracker["count"] += 1
            if RESOLVER_DEBUG and request_tracker["count"] % max(1, debug_every) == 1:
                resolved = response.url
                _dbg("request", url=resolved, status=response.status_code)
        if response.status_code == 200:
            return response
        if response.status_code in {429, 502, 503, 504}:
            time.sleep(backoff * (attempt + 1))
            continue
        response.raise_for_status()
    if last_exc:
        raise last_exc
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts")


def _iter_location_pages(situation_path: str) -> List[Tuple[str, str]]:
    url = urljoin(BASE, situation_path)
    html_text = _get(url).text
    parser = _LocationLinkParser()
    parser.feed(html_text)
    seen: Dict[str, Tuple[str, str]] = {}
    for raw_name, href in parser.links:
        absolute = urljoin(BASE, href)
        if absolute not in seen:
            seen[absolute] = (raw_name.strip(), absolute)
    return list(seen.values())


def _extract_json_links(location_html: str) -> List[str]:
    matches = re.findall(r'href="(https?://data\.unhcr\.org/population/get/timeseries\?[^"#]+)"', location_html)
    rel_matches = re.findall(r'href="(/population/get/timeseries\?[^"#]+)"', location_html)
    links: List[str] = []
    for raw in matches + rel_matches:
        decoded = html.unescape(raw)
        full = decoded if decoded.startswith("http") else urljoin(BASE, decoded)
        if full not in links:
            links.append(full)
    return links


def _discover_widget_links(location_html: str) -> List[str]:
    combos: List[Dict[str, str]] = []
    pattern = re.compile(r"widget_id[\"']?\s*[:=]\s*[\"']?([0-9A-Za-z_-]+)", re.IGNORECASE)
    for match in pattern.finditer(location_html):
        widget_id = match.group(1)
        context_start = max(match.start() - 200, 0)
        context_end = match.end() + 200
        context = location_html[context_start:context_end]
        geo_match = re.search(r"geo_id[\"']?\s*[:=]\s*[\"']?([0-9A-Za-z_-]+)", context, re.IGNORECASE)
        sv_match = re.search(r"sv_id[\"']?\s*[:=]\s*[\"']?([0-9A-Za-z_-]+)", context, re.IGNORECASE)
        combos.append({"widget_id": widget_id})
        if geo_match:
            combos.append({"widget_id": widget_id, "geo_id": geo_match.group(1)})
        if sv_match:
            combos.append({"widget_id": widget_id, "sv_id": sv_match.group(1)})
        if geo_match and sv_match:
            combos.append(
                {
                    "widget_id": widget_id,
                    "geo_id": geo_match.group(1),
                    "sv_id": sv_match.group(1),
                }
            )

    data_attr_matches = re.findall(
        r'data-url="(/population/get/timeseries\?[^"#]+)"', location_html
    )
    for raw in data_attr_matches:
        if "?" not in raw:
            continue
        query = raw.split("?", 1)[1]
        parsed = parse_qs(query)
        params = {key: values[0] for key, values in parsed.items() if values}
        if params:
            combos.append(params)

    unique_params: List[Dict[str, str]] = []
    seen: set[Tuple[Tuple[str, str], ...]] = set()
    for combo in combos:
        if isinstance(combo, dict):
            flattened = tuple(sorted((str(k), str(v)) for k, v in combo.items()))
            if flattened in seen:
                continue
            seen.add(flattened)
            unique_params.append({str(k): str(v) for k, v in combo.items()})

    links: List[str] = []
    for params in unique_params:
        if not params:
            continue
        query = urlencode(params, doseq=True)
        full = urljoin(BASE, f"/population/get/timeseries?{query}")
        if full not in links:
            links.append(full)
    return links


def _select_series(
    json_links: Iterable[str],
    desired_frequency: str,
    population_group: Optional[str],
) -> Optional[str]:
    desired_frequency = desired_frequency.lower()
    for link in json_links:
        parsed = urlparse(link)
        params = parse_qs(parsed.query)
        freq_values = {value.lower() for values in params.get("frequency", []) for value in values.split(",")}
        pop_values = set(params.get("population_group", []))
        series_freq = next(iter(freq_values), "")
        if desired_frequency and desired_frequency not in freq_values:
            continue
        if population_group and population_group not in pop_values:
            continue
        _dbg("series_candidate", url=link, frequency=series_freq or "unknown")
        return link
    return next(iter(json_links), None)


def _deterministic_event_id(iso3: str, as_of: str, value: str) -> str:
    iso = iso3 or "UNK"
    digest = hashlib.sha256(f"{iso}|DI|{as_of}|{value}".encode("utf-8")).hexdigest()[:16]
    return f"{iso}-DI-odpm-{digest}"


def _parse_series_payload(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _format_summary(counters: Counter) -> str:
    keys = (
        "locations_processed",
        "raw_points",
        "after_date_parse",
        "after_country_map",
        "final_rows",
        "page_cap_hit",
    )
    return "summary | " + " ".join(f"{key}={int(counters.get(key, 0))}" for key in keys)


def make_rows() -> Tuple[List[Dict[str, str]], Counter]:
    if os.getenv("RESOLVER_SKIP_UNHCR_ODP", "") == "1":
        return [], Counter()

    cfg = _load_config()
    situation_path = os.getenv("ODP_SITUATION_PATH", cfg.get("situation_path", DEFAULT_SITUATION_PATH))
    series_cfg = cfg.get("series", {}) if isinstance(cfg, dict) else {}
    desired_frequency = str(series_cfg.get("frequency", "month")) if isinstance(series_cfg, dict) else "month"
    population_group = None
    if isinstance(series_cfg, dict):
        pg = series_cfg.get("population_group")
        population_group = str(pg) if pg is not None else None

    paging_cfg = cfg.get("paging", {}) if isinstance(cfg, dict) else {}
    max_pages = int(paging_cfg.get("max_pages", 100) or 100)
    debug_cfg = cfg.get("debug", {}) if isinstance(cfg, dict) else {}
    debug_every = int(debug_cfg.get("debug_every", 5) or 5)

    whitelist_raw = cfg.get("countries_whitelist", []) if isinstance(cfg, dict) else []
    whitelist = [str(value).strip().lower() for value in whitelist_raw if str(value).strip()]

    request_tracker = {"count": 0}

    locations = _iter_location_pages(situation_path)
    counters: Counter = Counter()
    rows: List[Dict[str, str]] = []

    if whitelist:
        filtered_locations = []
        for raw_name, loc_url in locations:
            norm = _normalize_name(raw_name)
            if norm in {w.replace(" ", "") for w in whitelist} or raw_name.lower() in whitelist:
                filtered_locations.append((raw_name, loc_url))
        locations = filtered_locations
        _dbg("countries_whitelist", count=len(locations))

    country_index = CountryIndex.load()
    page_cap = False

    for idx, (raw_name, loc_url) in enumerate(locations, start=1):
        if idx > max_pages:
            page_cap = True
            counters["page_cap_hit"] += 1
            break
        counters["locations_processed"] += 1
        try:
            resp = _get(loc_url, request_tracker=request_tracker, debug_every=debug_every)
        except Exception as exc:  # noqa: BLE001
            _dbg("location_fetch_failed", url=loc_url, error=str(exc))
            continue
        loc_html = resp.text

        country_name = raw_name.strip() or _discover_country_name(loc_html)
        json_links = _extract_json_links(loc_html)
        if not json_links:
            widget_links = _discover_widget_links(loc_html)
            if widget_links:
                json_links = widget_links
            else:
                logger.warning("[ODP] no ODP widgets discovered for URL=%s", loc_url)
                continue
        series_url = _select_series(json_links, desired_frequency, population_group)
        if not series_url:
            _dbg("series_missing", location=loc_url)
            continue
        try:
            r = _get(
                series_url,
                headers={"Accept": "application/json"},
                request_tracker=request_tracker,
                debug_every=debug_every,
            )
        except Exception as exc:  # noqa: BLE001
            _dbg("series_fetch_failed", url=series_url, error=str(exc))
            continue
        try:
            payload = r.json()
        except json.JSONDecodeError as exc:
            _dbg("series_json_error", url=series_url, error=str(exc))
            continue
        points = _parse_series_payload(payload)
        counters["raw_points"] += len(points)
        _dbg(
            "series_selected",
            series_url=series_url,
            series_freq=desired_frequency,
            raw_points=len(points),
        )
        if not points:
            continue

        iso3, canonical_name = country_index.lookup(country_name)
        if RESOLVER_DEBUG:
            _dbg("country_lookup", raw=country_name, iso3=iso3 or "", canonical=canonical_name)

        sample_logged = False
        for point in points:
            as_of = str(point.get("date") or point.get("as_of") or "").strip()
            if not as_of:
                continue
            try:
                dt.date.fromisoformat(as_of)
            except ValueError:
                try:
                    parsed = time.strptime(as_of[:10], "%Y-%m-%d")
                    as_of = time.strftime("%Y-%m-%d", parsed)
                except Exception:  # noqa: BLE001
                    continue
            counters["after_date_parse"] += 1

            value_raw = point.get("value")
            if value_raw is None:
                continue
            value = str(value_raw)

            row_iso3, row_name = iso3, canonical_name
            if not row_iso3:
                maybe_iso = str(point.get("iso3") or point.get("country_iso3") or "").strip().upper()
                if maybe_iso:
                    row_iso3 = maybe_iso
                    row_name = country_index.iso_to_name.get(maybe_iso, row_name or country_name)
            if not row_name:
                row_name = country_name
            if not row_name:
                continue
            counters["after_country_map"] += 1

            event_id = _deterministic_event_id(row_iso3, as_of, value)
            row = {
                "source": "UNHCR-ODP",
                "source_event_id": event_id,
                "as_of_date": as_of,
                "country_iso3": row_iso3,
                "country_name": row_name,
                "hazard_code": HAZARD_CODE,
                "hazard_label": HAZARD_LABEL,
                "hazard_class": HAZARD_CLASS,
                "metric_name": "persons_arrivals",
                "metric_unit": "persons",
                "series_semantics": "new",
                "value": value,
                "evidence_url": series_url,
                "evidence_label": "UNHCR ODP population timeseries (monthly sea arrivals)",
            }
            rows.append(row)
            counters["final_rows"] += 1
            if RESOLVER_DEBUG and not sample_logged:
                _dbg(
                    "sample_point",
                    country=row_name,
                    iso3=row_iso3 or "",
                    as_of=as_of,
                    value=value,
                )
                sample_logged = True

    if page_cap:
        _dbg("page_cap_reached", max_pages=max_pages)

    if not rows:
        logger.info("UNHCR-ODP produced no rows (locations processed=%s)", counters.get("locations_processed", 0))

    if RESOLVER_DEBUG:
        _dbg("raw_points_total", count=counters.get("raw_points", 0))
        _dbg("after_date_parse", count=counters.get("after_date_parse", 0))
        _dbg("after_country_map", count=counters.get("after_country_map", 0))
        _dbg("final_rows", count=counters.get("final_rows", 0))

    return rows, counters


def _discover_country_name(html_text: str) -> str:
    title_match = re.search(r"<h1[^>]*>\s*([^<]{2,100})\s*</h1>", html_text)
    if title_match:
        return title_match.group(1).strip()
    return ""


def write_rows(rows: Iterable[Dict[str, str]]) -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CANONICAL_HEADER})
    ensure_manifest_for_csv(OUT_PATH)


def main() -> int:
    counters: Counter
    try:
        rows, counters = make_rows()
    except Exception as exc:  # noqa: BLE001
        print(f"[ODP] ERROR: {exc}", file=sys.stderr)
        rows, counters = [], Counter()
    write_rows(rows)
    print(f"wrote {OUT_PATH.resolve()} rows={len(rows)}")
    summary = _format_summary(counters)
    print(summary)
    _dbg("summary_emitted", summary=summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
