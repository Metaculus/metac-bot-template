#!/usr/bin/env python3
"""
validate_facts.py — lightweight checks for resolver facts

Usage:
  python resolver/tools/validate_facts.py --facts resolver/samples/facts_sample.csv

You can point --facts to any CSV or Parquet file with the "facts" columns.
The script:
  - loads canonical registries from resolver/data/{countries.csv,shocks.csv}
  - validates required columns & enums per resolver/tools/schema.yml
  - checks iso3 and hazard_code against registries
  - confirms hazard_label/class match the code row
  - validates dates (YYYY-MM-DD), and as_of_date <= publication_date <= today
  - checks value >= 0 and unit/metric rules (incl. cases vs persons)
Exits non-zero if any errors are found; prints a concise summary.
"""

import argparse, sys, os, json, datetime as dt
from typing import List, Dict, Any

# Local imports safe even if pandas isn't available yet.
try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow' to run the validator.", file=sys.stderr)
    sys.exit(2)

try:
    import yaml
except ImportError:
    print("Please 'pip install pyyaml' to run the validator.", file=sys.stderr)
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
TOOLS_DIR = os.path.join(ROOT, "tools")

SCHEMA_PATH = os.path.join(TOOLS_DIR, "schema.yml")
COUNTRIES_CSV = os.path.join(DATA_DIR, "countries.csv")
SHOCKS_CSV = os.path.join(DATA_DIR, "shocks.csv")

ALLOWED_SOURCE_TYPES = {"appeal","sitrep","gov","cluster","agency","media"}
ALLOWED_CONFIDENCE   = {"high","med","low"}
ALLOWED_UNITS        = {"persons","persons_cases"}

def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")

def _load_table(path: str) -> pd.DataFrame:
    # Support CSV or Parquet
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return pd.read_csv(path, dtype=str).fillna("")
    elif ext in (".parquet",):
        return pd.read_parquet(path)
    else:
        raise SystemExit(f"Unsupported file extension: {ext}. Use .csv or .parquet")

def _is_date(s: str) -> bool:
    try:
        dt.date.fromisoformat(s)
        return True
    except Exception:
        return False

def _as_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)

def validate(df: pd.DataFrame, schema: Dict[str, Any], countries: pd.DataFrame, shocks: pd.DataFrame) -> List[str]:
    errors: List[str] = []

    # Required columns
    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors  # can't proceed

    # Enum checks
    metric_enum = set(schema["enums"]["metric"])
    unit_enum   = set(schema["enums"]["unit"])

    # Build registries
    countries.index = countries["iso3"]
    shocks.index = shocks["hazard_code"]

    today = dt.date.today()

    for i, row in df.iterrows():
        prefix = f"row {i+1}"

        # Registry: iso3
        iso3 = str(row.get("iso3",""))
        iso3 = iso3.strip()
        if not iso3 or iso3 not in countries.index:
            errors.append(f"{prefix}: iso3 '{iso3}' not found in countries registry")

        # Registry: hazard_code
        hcode = str(row.get("hazard_code",""))
        hcode = hcode.strip()
        hlabel = str(row.get("hazard_label",""))
        hlabel = hlabel.strip()
        hclass = str(row.get("hazard_class",""))
        hclass = hclass.strip()
        if not hcode or hcode not in shocks.index:
            errors.append(f"{prefix}: hazard_code '{hcode}' not found in shocks registry")
        else:
            reg = shocks.loc[hcode]
            # Label/class must match registry
            if hlabel and hlabel != str(reg["hazard_label"]):
                errors.append(f"{prefix}: hazard_label '{hlabel}' does not match registry '{reg['hazard_label']}' for code {hcode}")
            if hclass and hclass != str(reg["hazard_class"]):
                errors.append(f"{prefix}: hazard_class '{hclass}' does not match registry '{reg['hazard_class']}' for code {hcode}")

        # Metric & unit
        metric = str(row.get("metric",""))
        metric = metric.strip()
        unit   = str(row.get("unit","persons"))
        unit = unit.strip() or "persons"
        if metric not in metric_enum:
            errors.append(f"{prefix}: metric '{metric}' not in {sorted(metric_enum)}")
        if unit not in unit_enum:
            errors.append(f"{prefix}: unit '{unit}' not in {sorted(unit_enum)}")

        # value >= 0 and numeric
        val_raw = row.get("value","")
        try:
            val = float(val_raw)
            if val < 0:
                errors.append(f"{prefix}: value {val} < 0")
        except Exception:
            errors.append(f"{prefix}: value '{val_raw}' is not numeric")

        # Dates
        as_of = str(row.get("as_of_date",""))
        as_of = as_of.strip()
        pub   = str(row.get("publication_date",""))
        pub = pub.strip()
        if not _is_date(as_of):
            errors.append(f"{prefix}: as_of_date '{as_of}' not ISO YYYY-MM-DD")
        if not _is_date(pub):
            errors.append(f"{prefix}: publication_date '{pub}' not ISO YYYY-MM-DD")
        if _is_date(as_of) and _is_date(pub):
            if _as_date(as_of) > _as_date(pub):
                errors.append(f"{prefix}: as_of_date {as_of} > publication_date {pub}")
            if _as_date(pub) > today:
                errors.append(f"{prefix}: publication_date {pub} > today {today.isoformat()}")

        # Source & confidence enums
        stype = str(row.get("source_type",""))
        stype = stype.strip()
        conf  = str(row.get("confidence",""))
        conf = conf.strip()
        if stype and stype not in ALLOWED_SOURCE_TYPES:
            errors.append(f"{prefix}: source_type '{stype}' not in {sorted(ALLOWED_SOURCE_TYPES)}")
        if conf and conf not in ALLOWED_CONFIDENCE:
            errors.append(f"{prefix}: confidence '{conf}' not in {sorted(ALLOWED_CONFIDENCE)}")

        # Metric-specific rules
        if metric == "in_need" and stype == "media":
            errors.append(f"{prefix}: metric 'in_need' cannot be sourced from media (must be inter-agency/gov/cluster/agency)")
        if metric == "cases" and unit != "persons_cases":
            errors.append(f"{prefix}: metric 'cases' must use unit 'persons_cases'")

    return errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True, help="Path to CSV or Parquet containing facts")
    args = ap.parse_args()

    if not os.path.exists(args.facts):
        print(f"Not found: {args.facts}", file=sys.stderr)
        sys.exit(2)

    schema = _load_schema()
    df = _load_table(args.facts)
    # Normalize columns (string) to avoid pandas NA surprises
    for c in df.columns:
        if df[c].dtype.name not in ("float64","int64"):
            df[c] = df[c].astype(str)

    countries = _load_csv(COUNTRIES_CSV)
    shocks    = _load_csv(SHOCKS_CSV)

    errors = validate(df, schema, countries, shocks)
    if errors:
        print("❌ Validation failed:")
        for e in errors:
            print(" -", e)
        print(f"\nChecked {len(df)} rows; {len(errors)} issue(s) found.")
        sys.exit(1)
    else:
        print(f"✅ Validation passed for {len(df)} rows.")
        sys.exit(0)

if __name__ == "__main__":
    main()
