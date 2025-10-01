#!/usr/bin/env python3
"""
resolver_cli.py — answer:
  "By <DATE>, how many people <METRIC> due to <HAZARD> in <COUNTRY>?"

Examples:
  python resolver/cli/resolver_cli.py \
    --country "Philippines" \
    --hazard "Tropical Cyclone" \
    --cutoff 2025-09-30

  python resolver/cli/resolver_cli.py \
    --iso3 PHL --hazard_code TC --cutoff 2025-09-30

Behavior:
  - If cutoff month < current month: read snapshots/YYYY-MM/facts.parquet (preferred)
    - If snapshot not found, optionally fall back to exports/resolved(_reviewed).csv (warn)
  - If cutoff is current month: prefer exports/resolved_reviewed.csv, else exports/resolved.csv
  - Applies selection rules already enforced upstream (precedence engine & review)
  - Returns a single record (value + citation) or explains why none exists
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ImportError:  # pragma: no cover - guidance for operators
    print("Please 'pip install pandas pyarrow' to run resolver_cli.", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOTS = ROOT / "snapshots"
EXPORTS = ROOT / "exports"

COUNTRIES_CSV = DATA / "countries.csv"
SHOCKS_CSV = DATA / "shocks.csv"


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load registries used for lookup and add normalized helper columns."""
    countries = pd.read_csv(COUNTRIES_CSV, dtype=str).fillna("")
    shocks = pd.read_csv(SHOCKS_CSV, dtype=str).fillna("")

    countries["country_norm"] = countries["country_name"].str.strip().str.lower()
    shocks["hazard_norm"] = shocks["hazard_label"].str.strip().str.lower()
    return countries, shocks


def resolve_country(
    countries: pd.DataFrame, country: Optional[str], iso3: Optional[str]
) -> Tuple[str, str]:
    """Return canonical (name, iso3) pair from either user input."""
    if iso3:
        iso3_code = iso3.strip().upper()
        match = countries[countries["iso3"] == iso3_code]
        if not match.empty:
            row = match.iloc[0]
            return row["country_name"], iso3_code

    if country:
        query = country.strip().lower()
        match = countries[countries["country_norm"] == query]
        if not match.empty:
            row = match.iloc[0]
            return row["country_name"], row["iso3"]

    raise SystemExit(
        "Could not resolve country; provide --country or --iso3 matching the registry."
    )


def resolve_hazard(
    shocks: pd.DataFrame, hazard: Optional[str], hazard_code: Optional[str]
) -> Tuple[str, str, str]:
    """Return canonical (label, code, class) triplet from label or code."""
    if hazard_code:
        hz_code = hazard_code.strip().upper()
        match = shocks[shocks["hazard_code"] == hz_code]
        if not match.empty:
            row = match.iloc[0]
            return row["hazard_label"], row["hazard_code"], row["hazard_class"]

    if hazard:
        query = hazard.strip().lower()
        match = shocks[shocks["hazard_norm"] == query]
        if not match.empty:
            row = match.iloc[0]
            return row["hazard_label"], row["hazard_code"], row["hazard_class"]

    raise SystemExit(
        "Could not resolve hazard; provide --hazard or --hazard_code matching the registry."
    )


IST = ZoneInfo("Europe/Istanbul")


def current_ym_istanbul() -> str:
    now = dt.datetime.now(IST)
    return f"{now.year:04d}-{now.month:02d}"


def current_ym_utc() -> str:
    """Backwards-compatible alias; resolver now tracks Istanbul month boundary."""
    return current_ym_istanbul()


def ym_from_cutoff(cutoff: str) -> str:
    year, month, _ = cutoff.split("-")
    return f"{int(year):04d}-{int(month):02d}"


def load_resolved_for_month(ym: str, is_current_month: bool) -> Tuple[Optional[pd.DataFrame], str]:
    """Load the resolved dataset according to month selection rules."""
    snapshot_path = SNAPSHOTS / ym / "facts.parquet"

    if not is_current_month:
        if snapshot_path.exists():
            return pd.read_parquet(snapshot_path), "snapshot"

        print(
            f"Warning: snapshot {snapshot_path} not found, falling back to exports.",
            file=sys.stderr,
        )

    reviewed = EXPORTS / "resolved_reviewed.csv"
    if reviewed.exists():
        return pd.read_csv(reviewed, dtype=str).fillna(""), "resolved_reviewed"

    base = EXPORTS / "resolved.csv"
    if base.exists():
        return pd.read_csv(base, dtype=str).fillna(""), "resolved"

    if snapshot_path.exists():
        return pd.read_parquet(snapshot_path), "snapshot"

    return None, ""


def select_row(df: pd.DataFrame, iso3: str, hazard_code: str, cutoff_iso: str) -> Optional[dict]:
    """Select the single row that best answers the resolver question."""
    candidate = df[
        (df["iso3"].astype(str) == iso3) & (df["hazard_code"].astype(str) == hazard_code)
    ].copy()

    if candidate.empty:
        return None

    if "metric" in candidate.columns:
        candidate["metric"] = candidate["metric"].fillna("")
        candidate["metric_ord"] = pd.Categorical(
            candidate["metric"], categories=["in_need", "affected"], ordered=True
        )

        if "as_of_date" in candidate.columns:
            candidate = candidate[candidate["as_of_date"] <= cutoff_iso]
            if candidate.empty:
                return None

        sort_cols = ["metric_ord"]
        extra_cols = [col for col in ["as_of_date", "publication_date"] if col in candidate.columns]
        sort_cols.extend(extra_cols)
        candidate = candidate.sort_values(by=sort_cols, ascending=[True] + [False] * len(extra_cols))
        candidate = candidate.drop(columns=["metric_ord"], errors="ignore")
    else:
        sort_cols = [col for col in ["as_of_date", "publication_date"] if col in candidate.columns]
        if sort_cols:
            candidate = candidate.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    top = candidate.iloc[0].to_dict()

    raw_value = top.get("value", "")
    try:
        top["value"] = int(float(raw_value))
    except Exception:
        top["value"] = raw_value

    return top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", help="Country name (as in countries.csv)")
    parser.add_argument("--iso3", help="Country ISO3 code")
    parser.add_argument("--hazard", help="Hazard label (as in shocks.csv)")
    parser.add_argument("--hazard_code", help="Hazard code (as in shocks.csv)")
    parser.add_argument("--cutoff", required=True, help="Cut-off date YYYY-MM-DD (23:59 Europe/Istanbul)")
    parser.add_argument("--json_only", action="store_true", help="Print JSON only (no human summary)")
    args = parser.parse_args()

    countries, shocks = load_registries()
    country_name, iso3 = resolve_country(countries, args.country, args.iso3)
    hazard_label, hazard_code, hazard_class = resolve_hazard(shocks, args.hazard, args.hazard_code)

    ym = ym_from_cutoff(args.cutoff)
    current_month = ym == current_ym_istanbul()
    df, source_dataset = load_resolved_for_month(ym, current_month)

    if df is None:
        message = (
            "No data found. Expected snapshot at snapshots/"
            f"{ym}/facts.parquet or exports/resolved(_reviewed).csv."
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": message,
                    "iso3": iso3,
                    "hazard_code": hazard_code,
                    "cutoff": args.cutoff,
                }
            ),
            flush=True,
        )
        if not args.json_only:
            print("\n" + message, file=sys.stderr)
        sys.exit(1)

    row = select_row(df, iso3, hazard_code, args.cutoff)
    if not row:
        message = (
            f"No eligible record for iso3={iso3}, hazard={hazard_code} at cutoff {args.cutoff}."
        )
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": message,
                    "iso3": iso3,
                    "hazard_code": hazard_code,
                    "cutoff": args.cutoff,
                }
            ),
            flush=True,
        )
        if not args.json_only:
            print("\n" + message, file=sys.stderr)
        sys.exit(1)

    snapshot_used = source_dataset == "snapshot"
    source_bucket = "snapshot" if snapshot_used else "exports"

    output = {
        "ok": True,
        "iso3": iso3,
        "country_name": country_name,
        "hazard_code": hazard_code,
        "hazard_label": hazard_label,
        "hazard_class": hazard_class,
        "cutoff": args.cutoff,
        "metric": row.get("metric", ""),
        "unit": row.get("unit", "persons"),
        "value": row.get("value", ""),
        "as_of_date": row.get("as_of_date", ""),
        "publication_date": row.get("publication_date", ""),
        "publisher": row.get("publisher", ""),
        "source_type": row.get("source_type", ""),
        "source_url": row.get("source_url", ""),
        "doc_title": row.get("doc_title", ""),
        "definition_text": row.get("definition_text", ""),
        "precedence_tier": row.get("precedence_tier", ""),
        "event_id": row.get("event_id", ""),
        "confidence": row.get("confidence", ""),
        "proxy_for": row.get("proxy_for", ""),
        "source": source_bucket,
        "source_dataset": source_dataset,
    }

    print(json.dumps(output, ensure_ascii=False), flush=True)

    if args.json_only:
        return

    print("\n=== Resolver ===")
    print(f"{country_name} ({iso3}) — {hazard_label} [{hazard_code}]")
    value = output["value"]
    metric = output["metric"] or "value"
    unit = output["unit"]
    try:
        human_value = f"{int(value):,}"
    except Exception:
        human_value = f"{value}"
    print(f"By {args.cutoff}: {human_value} {metric.replace('_', ' ')} ({unit})")
    print("— source —")
    print(f"{output['publisher']} | as-of {output['as_of_date']} | pub {output['publication_date']}")
    if output["source_url"]:
        print(output["source_url"])
    if output["definition_text"]:
        definition = output["definition_text"]
        trimmed = definition[:200]
        print(f"def: {trimmed}{'...' if len(definition) > 200 else ''}")
    if output["proxy_for"]:
        print(f"(proxy for {output['proxy_for']})")
    if output["precedence_tier"]:
        print(f"tier: {output['precedence_tier']}")
    if output["confidence"]:
        print(f"confidence: {output['confidence']}")
    dataset_label = output.get("source_dataset")
    detail = f" ({dataset_label})" if dataset_label else ""
    print(f"[source bucket: {output['source']}{detail}]")


if __name__ == "__main__":
    main()
