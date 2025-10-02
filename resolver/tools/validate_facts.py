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

import argparse, sys, os, json, datetime as dt, re
from typing import List, Dict, Any, Tuple

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

SERIES_NEW_CONTRADICTIONS = ["cumulative", "to date", "since", "total to date"]
SERIES_STOCK_HINTS = ["in the last 30 days", "this month"]
YM_REGEX = re.compile(r"^\d{4}-\d{2}$")

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

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == "nan"
    return False


def validate(df: pd.DataFrame, schema: Dict[str, Any], countries: pd.DataFrame, shocks: pd.DataFrame) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    # Required columns
    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors, warnings  # can't proceed

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

        # Series semantics + deltas metadata
        series_semantics = str(row.get("series_semantics", "")).strip().lower()
        definition_text = str(row.get("definition_text", ""))
        lower_def = definition_text.lower()
        if series_semantics == "new":
            contradictions = [phrase for phrase in SERIES_NEW_CONTRADICTIONS if phrase in lower_def]
            if contradictions:
                warnings.append(
                    f"{prefix}: series_semantics 'new' but definition_text contains {contradictions}"
                )
        elif series_semantics == "stock":
            hints = [phrase for phrase in SERIES_STOCK_HINTS if phrase in lower_def]
            if hints:
                warnings.append(
                    f"{prefix}: series_semantics 'stock' but definition_text contains {hints}"
                )

        value_new = row.get("value_new")
        value_stock = row.get("value_stock")
        if not _is_missing(value_new) and not _is_missing(value_stock):
            warnings.append(
                f"{prefix}: both value_new and value_stock provided; typically only one should be set"
            )

        ym_value = row.get("ym")
        if not _is_missing(ym_value):
            ym_str = str(ym_value).strip()
            if not YM_REGEX.match(ym_str):
                errors.append(f"{prefix}: ym '{ym_str}' must match YYYY-MM")

    return errors, warnings

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

    errors, warnings = validate(df, schema, countries, shocks)
    if warnings:
        print("⚠️ Warnings:")
        for w in warnings:
            print(" -", w)
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
