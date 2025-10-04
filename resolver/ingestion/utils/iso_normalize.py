"""Helpers for normalising country identifiers to ISO3 codes."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
COUNTRY_CSV = ROOT / "data" / "countries.csv"


def _normalise_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _load_country_lookup() -> tuple[Mapping[str, str], Mapping[str, str]]:
    iso_to_name: dict[str, str] = {}
    token_to_iso: dict[str, str] = {}
    if not COUNTRY_CSV.exists():
        return iso_to_name, token_to_iso
    with COUNTRY_CSV.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iso = (row.get("iso3") or "").strip().upper()
            name = (row.get("country_name") or "").strip()
            if not iso:
                continue
            iso_to_name[iso] = name
            if name:
                token = _normalise_token(name)
                if token:
                    token_to_iso.setdefault(token, iso)
    return iso_to_name, token_to_iso


def _normalised_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not aliases:
        return mapping
    for raw_key, raw_value in aliases.items():
        key = _normalise_token(str(raw_key))
        if not key:
            continue
        iso = str(raw_value or "").strip().upper()
        if len(iso) == 3:
            mapping[key] = iso
    return mapping


def to_iso3(name: str | None, aliases: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Normalise ``name`` to an ISO3 code if possible."""

    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    iso_to_name, token_lookup = _load_country_lookup()
    candidate = text.upper()
    if len(candidate) == 3 and candidate.isalpha():
        if candidate in iso_to_name:
            return candidate
    alias_map = _normalised_aliases(aliases)
    if alias_map:
        token = _normalise_token(text)
        alias_iso = alias_map.get(token)
        if alias_iso:
            return alias_iso
    token = _normalise_token(text)
    if token and token in token_lookup:
        return token_lookup[token]
    for iso, country in iso_to_name.items():
        if text.lower() == country.lower():
            return iso
    return None
