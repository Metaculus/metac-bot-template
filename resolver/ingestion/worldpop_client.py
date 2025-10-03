#!/usr/bin/env python3
"""WorldPop connector that writes population denominators for Resolver."""

from __future__ import annotations

import datetime as dt
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG = ROOT / "ingestion" / "config" / "worldpop.yml"

OUT_DATA = DATA / "population.csv"
OUT_STAGING = STAGING / "worldpop.csv"

CANONICAL_COLUMNS = [
    "iso3",
    "year",
    "population",
    "source",
    "product",
    "download_date",
    "source_url",
    "notes",
]

DEBUG = os.getenv("RESOLVER_DEBUG", "0") == "1"


@dataclass
class PopulationRow:
    iso3: str
    year: int
    population: int
    source: str
    product: str
    download_date: str
    source_url: str
    notes: str


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


def dbg(message: str) -> None:
    if DEBUG:
        print(f"[worldpop] {message}")


def _is_placeholder_url(url: str) -> bool:
    text = str(url or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if "<" in text or ">" in text:
        return True
    parsed = urlparse(text)
    if not parsed.scheme or parsed.scheme not in {"http", "https"}:
        return True
    host = parsed.netloc.lower()
    if not host:
        return True
    placeholder_hosts = (
        "example.com",
        "example.org",
        "example.net",
        "example.edu",
        "example.gov",
    )
    if any(host == candidate or host.endswith(f".{candidate}") for candidate in placeholder_hosts):
        return True
    if "placeholder" in host:
        return True
    if "placeholder" in lowered:
        return True
    return False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:  # pragma: no cover - defensive
        return default


def load_config() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data


def _ensure_population_header() -> None:
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    if not OUT_DATA.exists():
        pd.DataFrame(columns=CANONICAL_COLUMNS).to_csv(OUT_DATA, index=False)


def _write_header_only_staging() -> None:
    OUT_STAGING.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=CANONICAL_COLUMNS).to_csv(OUT_STAGING, index=False)


def _load_dataframe(url: str, *, kind: str = "csv") -> pd.DataFrame:
    dbg(f"downloading {url}")
    if url.startswith("http://") or url.startswith("https://"):
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"download failed: {url} status={resp.status_code}")
        data = io.BytesIO(resp.content)
    else:
        data = url
        if not Path(url).exists():
            raise FileNotFoundError(f"source path missing: {url}")
    if kind == "csv":
        return pd.read_csv(data)
    if kind in {"xlsx", "xls", "excel"}:
        return pd.read_excel(data)
    if kind == "json":
        return pd.read_json(data)
    return pd.read_csv(data)


def _maybe_apply_hxl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_row = df.iloc[0]
    if all(isinstance(v, str) and v.startswith("#") for v in first_row):
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = [str(v).strip() for v in first_row]
        return df
    if all(isinstance(c, str) and c.startswith("#") for c in df.columns):
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return df


def _normalise_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower()
        mapping[key] = column
    return mapping


def _find_column(mapping: Dict[str, str], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if not key:
            continue
        lowered = key.strip().lower()
        if lowered in mapping:
            return mapping[lowered]
    return None


def _parse_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        number = int(float(text))
    except Exception:
        return None
    return number


def _parse_population(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        number = float(cleaned)
    except Exception:
        return None
    if pd.isna(number):
        return None
    return int(round(number))


def _format_url(template: str, *, product: str, year: str | int) -> str:
    placeholders = _DefaultDict(product=product, year=year)
    return template.format_map(placeholders)


def _collect_from_frame(
    df: pd.DataFrame,
    *,
    keys_cfg: Dict[str, Sequence[str]],
    prefer_hxl: bool,
    source: str,
    product: str,
    download_date: str,
    source_url: str,
) -> List[PopulationRow]:
    if df is None or df.empty:
        return []
    if prefer_hxl:
        df = _maybe_apply_hxl(df)
    else:
        df.columns = [str(c).strip() for c in df.columns]

    mapping = _normalise_columns(df)
    iso_col = _find_column(mapping, keys_cfg.get("iso3", []))
    year_col = _find_column(mapping, keys_cfg.get("year", []))
    pop_col = _find_column(mapping, keys_cfg.get("population", []))
    notes_col = _find_column(mapping, keys_cfg.get("notes", []))

    if not iso_col or not pop_col:
        return []

    rows: List[PopulationRow] = []
    for _, row in df.iterrows():
        iso_raw = row.get(iso_col)
        if pd.isna(iso_raw):
            continue
        iso3 = str(iso_raw).strip().upper()
        if not iso3 or len(iso3) != 3:
            continue
        year_val = _parse_year(row.get(year_col)) if year_col else None
        if year_val is None:
            continue
        pop_val = _parse_population(row.get(pop_col))
        if pop_val is None or pop_val <= 0:
            continue
        notes_val = ""
        if notes_col:
            notes_raw = row.get(notes_col)
            if isinstance(notes_raw, str):
                notes_val = notes_raw.strip()
            elif notes_raw is not None and not pd.isna(notes_raw):
                notes_val = str(notes_raw).strip()
        rows.append(
            PopulationRow(
                iso3=iso3,
                year=year_val,
                population=pop_val,
                source=source,
                product=product,
                download_date=download_date,
                source_url=source_url,
                notes=notes_val,
            )
        )
    return rows


def collect_rows() -> List[PopulationRow]:
    cfg = load_config()
    product_env = os.getenv("WORLDPOP_PRODUCT")
    product = (product_env or cfg.get("product") or "un_adj_unconstrained").strip()
    years_back = _env_int("WORLDPOP_YEARS_BACK", int(cfg.get("years_back", 0)))
    prefer_hxl = bool(cfg.get("prefer_hxl", False))
    keys_cfg = cfg.get("keys", {})
    source_cfg = cfg.get("source", {})
    source_label = source_cfg.get("publisher", "WorldPop")
    url_template_env = os.getenv("WORLDPOP_URL_TEMPLATE")
    url_template = url_template_env or source_cfg.get("url_template")
    static_url = source_cfg.get("url")
    source_kind = (source_cfg.get("kind") or "csv").lower()
    download_date = dt.date.today().isoformat()

    for candidate in (url_template, static_url):
        if candidate and _is_placeholder_url(str(candidate)):
            print("[worldpop] disabled/placeholder config; header-only")
            dbg(f"placeholder url detected: {candidate}")
            return []

    if not url_template and not static_url:
        raise RuntimeError("worldpop config missing url or url_template")

    collected: Dict[tuple[str, int], PopulationRow] = {}
    seen_years: set[int] = set()

    def _register(rows: List[PopulationRow]) -> None:
        for row in rows:
            key = (row.iso3, row.year)
            collected[key] = row
            seen_years.add(row.year)

    targets: List[tuple[int, str]] = []

    if url_template:
        latest_url = _format_url(url_template, product=product, year="latest")
        targets.append((-1, latest_url))
    elif static_url:
        targets.append((-1, static_url))

    for hint, url in targets:
        try:
            df = _load_dataframe(url, kind=source_kind)
        except Exception as exc:
            dbg(f"failed to load {url}: {exc}")
            continue
        rows = _collect_from_frame(
            df,
            keys_cfg=keys_cfg,
            prefer_hxl=prefer_hxl,
            source=source_label,
            product=product,
            download_date=download_date,
            source_url=url,
        )
        _register(rows)

    if not collected:
        return []

    latest_year = max(seen_years)

    if years_back > 0 and url_template:
        for offset in range(1, years_back + 1):
            year = latest_year - offset
            if year < 0:
                continue
            url = _format_url(url_template, product=product, year=year)
            try:
                df = _load_dataframe(url, kind=source_kind)
            except Exception as exc:
                dbg(f"failed to load {url}: {exc}")
                continue
            rows = _collect_from_frame(
                df,
                keys_cfg=keys_cfg,
                prefer_hxl=prefer_hxl,
                source=source_label,
                product=product,
                download_date=download_date,
                source_url=url,
            )
            if not rows:
                continue
            _register(rows)

    filtered: List[PopulationRow] = []
    grouped: Dict[tuple[str, str], List[PopulationRow]] = {}
    for row in collected.values():
        grouped.setdefault((row.iso3, row.product), []).append(row)

    for (iso3, product_label), rows in grouped.items():
        years = {row.year for row in rows}
        if not years:
            continue
        iso_max_year = max(years)
        iso_min_year = (
            iso_max_year - years_back if years_back >= 0 else iso_max_year
        )
        kept_years = []
        for row in rows:
            if iso_min_year <= row.year <= iso_max_year:
                filtered.append(row)
                kept_years.append(row.year)
        if DEBUG:
            kept_years_sorted = sorted(set(kept_years))
            dbg(
                f"{iso3} product={product_label!r} kept years={kept_years_sorted} "
                f"(iso_max={iso_max_year}, years_back={years_back})"
            )

    filtered.sort(key=lambda r: (r.iso3, r.product, r.year))
    return filtered


def _load_existing() -> pd.DataFrame:
    if not OUT_DATA.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(OUT_DATA)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df["iso3"] = df["iso3"].astype(str).str.upper()
    df["year"] = df["year"].apply(lambda x: _parse_year(x) or 0)
    df["population"] = df["population"].apply(lambda x: _parse_population(x) or 0)
    df["notes"] = df.get("notes", "").fillna("")
    df = df[(df["iso3"].astype(bool)) & (df["population"] > 0)]
    if "year" in df:
        df = df[df["year"] > 0]
    return df.reset_index(drop=True)


def _merge_population(existing: pd.DataFrame, rows: List[PopulationRow]) -> tuple[pd.DataFrame, int, int]:
    if not rows:
        return existing, 0, 0
    inserted = 0
    updated = 0
    result = existing.copy()
    if result.empty:
        data = [row.__dict__ for row in rows]
        df = pd.DataFrame(data, columns=CANONICAL_COLUMNS)
        return df, len(rows), 0

    for row in rows:
        mask = (result["iso3"] == row.iso3) & (result["year"] == row.year)
        row_dict = {
            "iso3": row.iso3,
            "year": int(row.year),
            "population": int(row.population),
            "source": row.source,
            "product": row.product,
            "download_date": row.download_date,
            "source_url": row.source_url,
            "notes": row.notes or "",
        }
        if mask.any():
            for column, value in row_dict.items():
                result.loc[mask, column] = value
            updated += 1
        else:
            result = pd.concat([result, pd.DataFrame([row_dict])], ignore_index=True)
            inserted += 1
    result = result.drop_duplicates(subset=["iso3", "year"], keep="last")
    result.sort_values(["iso3", "year"], inplace=True, ignore_index=True)
    return result, inserted, updated


def _write_outputs(rows: List[PopulationRow]) -> tuple[int, int, int]:
    existing = _load_existing()
    merged, inserted, updated = _merge_population(existing, rows)
    merged.to_csv(OUT_DATA, index=False)

    stage_df = pd.DataFrame([row.__dict__ for row in rows], columns=CANONICAL_COLUMNS)
    OUT_STAGING.parent.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(OUT_STAGING, index=False)
    return len(rows), inserted, updated


def main() -> bool:
    if os.getenv("RESOLVER_SKIP_WORLDPOP") == "1":
        dbg("RESOLVER_SKIP_WORLDPOP=1 — writing headers only")
        _ensure_population_header()
        _write_header_only_staging()
        return False

    try:
        rows = collect_rows()
    except Exception as exc:
        dbg(f"collect_rows failed: {exc}")
        _ensure_population_header()
        _write_header_only_staging()
        return False

    if not rows:
        dbg("no WorldPop rows collected; writing headers")
        _ensure_population_header()
        _write_header_only_staging()
        return False

    count, inserted, updated = _write_outputs(rows)
    print(f"WorldPop wrote {count} rows (inserted={inserted}, updated={updated})")
    return True


if __name__ == "__main__":
    main()
