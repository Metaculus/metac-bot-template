#!/usr/bin/env python3
"""
IFRC GO Admin v2 → staging/ifrc_go.csv

- Pulls recent Field Reports, Appeals, and Situation Reports
- Paginates with retries/backoff; optional token header
- Maps reports → (iso3, hazard_code) via keyword heuristics
- Extracts PIN/PA (and cases for PHE) from numeric fields first, else from text with regex
- Writes canonical staging CSV used by exporter/validator

ENV:
  RESOLVER_SKIP_IFRCGO=1  → skip connector, write header-only CSV
  RESOLVER_DEBUG=1        → verbose HTTP logging
  GO_API_TOKEN=<token>    → optional Authorization: Token <token>

Refs:
  GO Admin v2 overview & endpoints (Swagger & wiki)  # see README for links
"""

from __future__ import annotations
import os, time, re, json, datetime as dt
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import requests
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "ifrc_go.yml"

COUNTRIES = DATA / "countries.csv"
SHOCKS = DATA / "shocks.csv"

COLUMNS = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","series_semantics","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

NUM_WINDOW_RE = re.compile(r"([0-9][0-9., ]{0,15})")
PHRASE_PAD = 80

def _debug() -> bool:
    return os.getenv("RESOLVER_DEBUG", "") == "1"

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default

# Paging / debug controls (hard caps)
MAX_PAGES = _int_env("RESOLVER_MAX_PAGES", 50)         # hard stop per endpoint
MAX_RESULTS = _int_env("RESOLVER_MAX_RESULTS", 5000)   # hard stop total rows per endpoint
DEBUG_EVERY = _int_env("RESOLVER_DEBUG_EVERY", 10)     # log every N pages when DEBUG=1

def dbg(msg: str):
    if _debug():
        print(f"[IFRC-GO] {msg}")

def load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_registries():
    c = pd.read_csv(COUNTRIES, dtype=str).fillna("")
    s = pd.read_csv(SHOCKS, dtype=str).fillna("")
    c["country_norm"] = c["country_name"].str.strip().str.lower()
    return c, s

def norm(s: str) -> str:
    return (s or "").strip().lower()

def detect_hazard(text: str, cfg: Dict[str, Any]) -> Optional[str]:
    t = norm(text)
    for code, keys in cfg["hazard_keywords"].items():
        for k in keys:
            if k in t:
                return code
    return None

def extract_metric_record_first(rec: dict, cfg: Dict[str, Any]) -> Optional[Tuple[str,int,str,str]]:
    # Try numeric fields directly if present
    for fld in cfg.get("numeric_fields", []):
        if fld in rec and rec[fld] not in (None, "", 0, "0"):
            try:
                val = int(float(str(rec[fld]).replace(",", "")))
                if val >= 0:
                    # Heuristic: if field name suggests PIN, treat as in_need; else 'affected'
                    metric = "in_need" if "need" in fld else "affected"
                    return (metric, val, "persons", f"{fld}")
            except Exception:
                continue
    return None

def extract_metric_text(text: str, cfg: Dict[str, Any]) -> Optional[Tuple[str,int,str,str]]:
    tl = text.lower()
    for block in cfg.get("regex_patterns", []):
        metric = block["metric"]; unit = block["unit"]
        for phrase in block["phrases"]:
            idx = tl.find(phrase)
            if idx == -1: 
                continue
            w0 = max(0, idx - PHRASE_PAD); w1 = min(len(text), idx + len(phrase) + PHRASE_PAD)
            window = text[w0:w1]
            m = NUM_WINDOW_RE.search(window)
            if m:
                raw = m.group(1)
                cleaned = re.sub(r"[^0-9]", "", raw)
                try:
                    val = int(cleaned)
                    return (metric, val, unit, phrase)
                except Exception:
                    pass
    return None

def req_json(base: str, path: str, params: dict, headers: dict, tries=4, backoff=1.3) -> dict:
    url = base.rstrip("/") + "/" + path.lstrip("/")
    last = None
    for t in range(1, tries+1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if _debug():
                dbg(f"GET {r.url} -> {r.status_code}")
                if r.status_code != 200:
                    dbg(f"Headers: {dict(r.headers)}")
                    dbg(f"Body snippet: {r.text[:400]}")
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503):
                time.sleep((backoff ** t) + 0.2 * t); continue
            # Some endpoints may 202 for processing; fail-soft
            if r.status_code == 202:
                return {}
            r.raise_for_status()
        except Exception as e:
            last = e
            time.sleep((backoff ** t) + 0.2 * t)
    if last: raise last
    return {}

def map_source_type(endpoint_key: str, cfg: Dict[str, Any]) -> str:
    return cfg["source_type_map"].get(endpoint_key, "sitrep")

def iso3_pairs_from_go(countries_df: pd.DataFrame, rec: dict) -> List[Tuple[str,str]]:
    """
    Return [(country_name, ISO3), ...] using countries_details if present,
    else fallback to empty (we don't chase IDs here).
    """
    out: List[Tuple[str,str]] = []
    details = rec.get("countries_details") or []
    if isinstance(details, list) and details:
        for c in details:
            iso = str(c.get("iso3", "")).strip().upper()
            name = str(c.get("name", "") or c.get("name_en","")).strip()
            if iso:
                row = countries_df[countries_df["iso3"] == iso]
                if not row.empty:
                    out.append((row.iloc[0]["country_name"], iso))
                    continue
            if name:
                row = countries_df[countries_df["country_name"].str.lower() == name.lower()]
                if not row.empty:
                    out.append((row.iloc[0]["country_name"], row.iloc[0]["iso3"]))
    return out

def collect_rows() -> List[List[str]]:
    if os.getenv("RESOLVER_SKIP_IFRCGO", "") == "1":
        return []  # caller will still write header-only CSV

    cfg = load_cfg()
    countries, shocks = load_registries()

    base = cfg["base_url"]
    page_size = int(cfg["page_size"])
    since_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(cfg["window_days"]))
    since = since_dt.date().isoformat()

    token = os.getenv("GO_API_TOKEN", "").strip()
    headers = {
        "User-Agent": cfg["user_agent"],
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Token {token}"

    rows: List[List[str]] = []
    seen_ids: set[str] = set()

    # Each endpoint paginates with ?limit=&offset= ; Admin v2 documents pagination on Swagger/wiki.
    for key, path in cfg["endpoints"].items():
        seen_ids.clear()
        made_rows = 0

        def run_pass(pass_name: str, filter_key: str):
            nonlocal made_rows
            offset = 0
            page_count = 0
            older_pages_in_row = 0
            while True:
                params = {
                    "limit": page_size,
                    "offset": offset,
                    "ordering": "-created_at",
                    "fields": (
                        "id,title,name,summary,description,created_at,updated_at,report_date,"
                        "document_url,external_link,source,disaster_type,disaster_type_details,"
                        "countries,countries_details,num_affected,people_in_need,affected"
                    ),
                    filter_key: since,
                }
                data = req_json(base, path, params, headers)
                page_count += 1
                if _debug() and (page_count % DEBUG_EVERY == 1):
                    dbg(f"[{key}:{pass_name}] page={page_count} offset={offset} since={since}")

                results = data.get("results") if isinstance(data, dict) else (
                    data if isinstance(data, list) else []
                )
                if not results:
                    break

                # Early-exit: if ENTIRE page older than window on BOTH created/updated
                all_older = True
                for r in results:
                    created = str(r.get("created_at") or "")[:10]
                    updated = str(r.get("updated_at") or "")[:10]
                    if (created and created >= since) or (updated and updated >= since):
                        all_older = False
                        break
                if all_older:
                    older_pages_in_row += 1
                else:
                    older_pages_in_row = 0
                if older_pages_in_row >= 2:
                    dbg(f"[{key}:{pass_name}] early-exit: consecutive older-only pages")
                    break

                for r in results:
                    rid = str(r.get("id") or "")
                    if not rid or rid in seen_ids:
                        continue

                    created = str(r.get("created_at") or "")[:10]
                    updated = str(r.get("updated_at") or "")[:10]
                    if (not created or created < since) and (not updated or updated < since):
                        continue

                    iso_pairs = iso3_pairs_from_go(countries, r)
                    if not iso_pairs:
                        continue

                    title = str(r.get("title") or r.get("name") or "")
                    summary = str(r.get("summary") or r.get("description") or "")
                    dtype = ""
                    dt_obj = (
                        r.get("disaster_type_details")
                        or r.get("disaster_type")
                        or r.get("dtype")
                        or {}
                    )
                    if isinstance(dt_obj, dict):
                        dtype = dt_obj.get("name") or ""
                    elif isinstance(dt_obj, list) and dt_obj:
                        dtype = (dt_obj[0].get("name") or "") if isinstance(dt_obj[0], dict) else ""

                    hz_text = " ".join([title, summary, dtype])
                    hazard_code = detect_hazard(hz_text, cfg)
                    if not hazard_code:
                        continue

                    srow = shocks[shocks["hazard_code"] == hazard_code]
                    if srow.empty:
                        continue
                    hz_label = srow.iloc[0]["hazard_label"]
                    hz_class = srow.iloc[0]["hazard_class"]

                    metric_pack = extract_metric_record_first(r, cfg)
                    if not metric_pack:
                        metric_pack = extract_metric_text(" ".join([title, summary]), cfg)
                    if not metric_pack:
                        continue

                    metric, value, unit, why = metric_pack

                    as_of = (str(r.get("report_date") or r.get("updated_at") or created) or "")[:10]
                    pub = created or as_of

                    url = (
                        r.get("document_url")
                        or r.get("document")
                        or r.get("external_link")
                        or r.get("source")
                        or ""
                    )
                    doc_title = title

                    emitted = False
                    for country_name, iso3 in iso_pairs:
                        event_id = f"{iso3}-{hazard_code}-ifrcgo-{r.get('id','0')}"
                        rows.append([
                            event_id,
                            country_name,
                            iso3,
                            hazard_code,
                            hz_label,
                            hz_class,
                            metric,
                            "stock",
                            str(value),
                            unit,
                            as_of,
                            pub,
                            "IFRC",
                            map_source_type(key, cfg),
                            url,
                            doc_title,
                            f"Extracted {metric} via {why} from IFRC GO {key.replace('_',' ')}.",
                            "api",
                            "med",
                            1,
                            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        ])
                        emitted = True
                        made_rows += 1
                        if made_rows >= MAX_RESULTS:
                            dbg(f"[{key}:{pass_name}] hit MAX_RESULTS={MAX_RESULTS}, stopping")
                            break

                    if emitted:
                        seen_ids.add(rid)

                    if made_rows >= MAX_RESULTS:
                        break

                if made_rows >= MAX_RESULTS:
                    break

                has_next = isinstance(data, dict) and bool(data.get("next"))
                if has_next and page_count < MAX_PAGES and made_rows < MAX_RESULTS:
                    offset += page_size
                    time.sleep(0.25)
                    continue
                else:
                    break

        run_pass("created", "created_at__gte")
        if made_rows < MAX_RESULTS:
            run_pass("updated", "updated_at__gte")

    return rows

def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    out = STAGING / "ifrc_go.csv"

    try:
        rows = collect_rows()
    except Exception as e:
        dbg(f"ERROR: {e}")
        rows = []

    if not rows:
        pd.DataFrame(columns=COLUMNS).to_csv(out, index=False)
        print(f"wrote empty {out}")
        return

    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
