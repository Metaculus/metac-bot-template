"""Shared population denominator lookup utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "population.csv"


@dataclass(frozen=True)
class PopulationRecord:
    iso3: str
    year: int
    population: int
    source: Optional[str] = None
    product: Optional[str] = None
    download_date: Optional[str] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None


def _normalise_path(path: Optional[str | Path]) -> Path:
    if path is None:
        return DEFAULT_PATH
    if isinstance(path, Path):
        return path
    return Path(path)


@lru_cache(maxsize=8)
def _load_population_frame(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=["iso3", "year", "population", "source", "product", "download_date", "source_url", "notes"])
    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        return df
    df = df.rename(columns={str(col).strip().lower(): col for col in df.columns})
    # Standardise columns we care about
    col_map: dict[str, str] = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        col_map[lower] = col
    def _get(col: str) -> Optional[pd.Series]:
        lower = col.lower()
        if lower in col_map:
            return df[col_map[lower]]
        return None

    iso_series = _get("iso3")
    year_series = _get("year")
    pop_series = _get("population")
    source_series = _get("source")
    product_series = _get("product")
    download_series = _get("download_date")
    url_series = _get("source_url")
    notes_series = _get("notes")

    normalised = pd.DataFrame()
    if iso_series is not None:
        normalised["iso3"] = iso_series.astype(str).str.strip().str.upper()
    else:
        normalised["iso3"] = ""

    if year_series is not None:
        normalised["year"] = year_series.apply(_parse_year)
    else:
        normalised["year"] = pd.Series(dtype="Int64")

    if pop_series is not None:
        normalised["population"] = pop_series.apply(_parse_population)
    else:
        normalised["population"] = pd.Series(dtype="Int64")

    if source_series is not None:
        normalised["source"] = source_series.astype(str).str.strip()
    else:
        normalised["source"] = ""

    if product_series is not None:
        normalised["product"] = product_series.astype(str).str.strip()
    else:
        normalised["product"] = ""

    if download_series is not None:
        normalised["download_date"] = download_series.astype(str).str.strip()
    else:
        normalised["download_date"] = ""

    if url_series is not None:
        normalised["source_url"] = url_series.astype(str).str.strip()
    else:
        normalised["source_url"] = ""

    if notes_series is not None:
        normalised["notes"] = notes_series.astype(str).str.strip()
    else:
        normalised["notes"] = ""

    normalised = normalised.dropna(subset=["iso3", "population"], how="any")
    normalised = normalised[normalised["iso3"].astype(bool)]
    normalised = normalised[normalised["population"].notna()]
    normalised = normalised.reset_index(drop=True)
    return normalised


def _parse_year(value: str | float | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
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


def _parse_population(value: str | float | int | None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
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
    if math.isnan(number):
        return None
    return int(round(number))


def _select_row(df: pd.DataFrame, iso3: str, year: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    iso = str(iso3).strip().upper()
    if not iso:
        return None
    subset = df[df["iso3"] == iso]
    subset = subset.dropna(subset=["population"], how="any")
    if subset.empty:
        return None
    if "year" in subset.columns and subset["year"].notna().any():
        subset_year = subset[subset["year"].notna()].copy()
        subset_year["year"] = subset_year["year"].astype(int)
        candidates = subset_year[subset_year["year"] <= year]
        if candidates.empty:
            # fall back to the latest available population entry
            return subset_year.sort_values("year").iloc[-1]
        return candidates.sort_values("year").iloc[-1]
    # No year column; return the last row
    return subset.iloc[-1]


def get_population_record(iso3: str, year: int, path: str | Path | None = None) -> Optional[PopulationRecord]:
    """Return the best available population record for ``iso3`` and ``year``."""

    csv_path = _normalise_path(path)
    frame = _load_population_frame(str(csv_path))
    row = _select_row(frame, iso3, year)
    if row is None:
        return None
    population = row.get("population")
    if population is None:
        return None
    used_year = row.get("year")
    if pd.isna(used_year):
        used_year = None
    year_int = int(used_year) if used_year is not None else year
    return PopulationRecord(
        iso3=str(row.get("iso3", iso3)).upper(),
        year=year_int,
        population=int(population),
        source=_safe_str(row.get("source")),
        product=_safe_str(row.get("product")),
        download_date=_safe_str(row.get("download_date")),
        source_url=_safe_str(row.get("source_url")),
        notes=_safe_str(row.get("notes")),
    )


def _safe_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_population(iso3: str, year: int, path: str | Path | None = None) -> Optional[int]:
    """Return the population for ``iso3`` and ``year`` if available."""

    record = get_population_record(iso3, year, path)
    if record is None:
        return None
    return record.population


def safe_pct_to_people(
    pct: float | int | str | None,
    iso3: str,
    year: int,
    *,
    denom_path: str | Path | None = None,
    min_threshold: float = 0.0,
) -> Optional[int]:
    """Convert ``pct`` prevalence to people using the stored denominators."""

    if pct is None:
        return None
    try:
        pct_value = float(pct)
    except Exception:
        return None
    if math.isnan(pct_value):
        return None
    if pct_value < min_threshold:
        return None

    record = get_population_record(iso3, year, denom_path)
    if record is None:
        return None
    if record.population <= 0:
        return None
    people = record.population * pct_value / 100.0
    return int(round(people))


def clear_population_cache() -> None:
    """Clear the cached population table (useful for tests)."""

    _load_population_frame.cache_clear()
