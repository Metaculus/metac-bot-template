from __future__ import annotations
import json, os, glob, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOOLS = ROOT / "tools"
EXPORTS = ROOT / "exports"
REVIEW  = ROOT / "review"
SNAPS   = ROOT / "snapshots"
STATE   = ROOT / "state"

SCHEMA_YML = TOOLS / "schema.yml"
COUNTRIES_CSV = DATA / "countries.csv"
SHOCKS_CSV = DATA / "shocks.csv"

def today_iso() -> str:
    return dt.date.today().isoformat()

def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_YML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_countries() -> pd.DataFrame:
    return pd.read_csv(COUNTRIES_CSV, dtype=str).fillna("")

def load_shocks() -> pd.DataFrame:
    return pd.read_csv(SHOCKS_CSV, dtype=str).fillna("")

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def discover_export_files() -> List[Path]:
    files: List[Path] = []
    if EXPORTS.exists():
        files += [p for p in [EXPORTS / "facts.csv"] if p.exists()]
        files += list(EXPORTS.glob("resolved*.csv"))
    # remote-first state: include any committed exports csvs
    if STATE.exists():
        files += list(STATE.glob("pr/*/exports/*.csv"))
        files += list(STATE.glob("daily/*/exports/*.csv"))
    # de-duplicate
    uniq = []
    seen = set()
    for f in files:
        s = str(f)
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq

def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]
