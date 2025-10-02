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
  - Defaults to monthly NEW deltas when available (`--series new`); use `--series stock` for totals. Missing deltas print a note and fall back to stocks.
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
STATE = ROOT / "state"

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


def first_day_of_month_from_ym(ym: str) -> str:
    year_str, month_str = ym.split("-")
    return dt.date(int(year_str), int(month_str), 1).isoformat()


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


def load_deltas_for_month(ym: str, is_current_month: bool) -> Tuple[Optional[pd.DataFrame], str]:
    """Load monthly deltas for a given month if available."""
    candidates: list[Tuple[Path, str]] = []

    if not is_current_month:
        monthly_path = STATE / "monthly" / ym / "deltas.csv"
        candidates.append((monthly_path, "monthly_deltas"))
        snapshot_deltas = SNAPSHOTS / ym / "deltas.csv"
        candidates.append((snapshot_deltas, "snapshot_deltas"))

    exports_deltas = EXPORTS / "deltas.csv"
    candidates.append((exports_deltas, "deltas"))

    for path, label in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:  # pragma: no cover - defensive
            continue
        if "ym" in df.columns:
            df = df[df["ym"].astype(str) == ym]
        if df.empty:
            continue
        return df, label

    return None, ""


def prepare_deltas_frame(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    filtered = df.copy()
    if "ym" in filtered.columns:
        filtered = filtered[filtered["ym"].astype(str) == ym]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    default_as_of = first_day_of_month_from_ym(ym)

    if "as_of_date" in filtered.columns:
        filtered["as_of_date"] = filtered["as_of_date"].astype(str)
    elif "as_of" in filtered.columns:
        filtered["as_of_date"] = filtered["as_of"].astype(str)
    else:
        filtered["as_of_date"] = default_as_of
    filtered["as_of_date"] = filtered["as_of_date"].fillna("")
    filtered.loc[filtered["as_of_date"].str.strip() == "", "as_of_date"] = default_as_of

    if "publication_date" in filtered.columns:
        filtered["publication_date"] = filtered["publication_date"].astype(str).fillna("")
        pub_blank = filtered["publication_date"].str.strip() == ""
        filtered.loc[pub_blank, "publication_date"] = filtered.loc[pub_blank, "as_of_date"]
    else:
        filtered["publication_date"] = filtered["as_of_date"]

    if "value_new" in filtered.columns:
        filtered["value"] = filtered["value_new"]
    elif "value" not in filtered.columns:
        filtered["value"] = ""
    filtered["value"] = filtered["value"].astype(str).fillna("")

    if "metric" in filtered.columns:
        filtered["metric"] = filtered["metric"].astype(str).fillna("")
    else:
        filtered["metric"] = ""

    if "unit" in filtered.columns:
        filtered["unit"] = filtered["unit"].astype(str).fillna("")
        filtered.loc[filtered["unit"].str.strip() == "", "unit"] = "persons"
    else:
        filtered["unit"] = "persons"

    if "publisher" in filtered.columns:
        filtered["publisher"] = filtered["publisher"].astype(str).fillna("")
    else:
        filtered["publisher"] = ""
    if "source_name" in filtered.columns:
        source_series = filtered["source_name"].astype(str).fillna("")
        blank_pub = filtered["publisher"].str.strip() == ""
        filtered.loc[blank_pub, "publisher"] = source_series[blank_pub]

    for col in ["source_type", "source_url", "doc_title", "definition_text"]:
        if col in filtered.columns:
            filtered[col] = filtered[col].astype(str).fillna("")
        else:
            filtered[col] = ""

    filtered["series_semantics"] = "new"
    filtered["ym"] = ym

    # Normalize common placeholder strings
    for column in filtered.columns:
        filtered[column] = filtered[column].replace({"nan": "", "NaT": ""})

    return filtered.fillna("")


def load_series_for_month(
    ym: str, is_current_month: bool, requested_series: str
) -> Tuple[Optional[pd.DataFrame], str, str]:
    """Load data for the requested series ("new" or "stock")."""

    normalized_series = (requested_series or "stock").strip().lower()
    if normalized_series == "new":
        deltas_df, dataset_label = load_deltas_for_month(ym, is_current_month)
        if deltas_df is None or deltas_df.empty:
            return None, "", "new"
        prepared = prepare_deltas_frame(deltas_df, ym)
        if prepared.empty:
            return None, "", "new"
        return prepared, dataset_label, "new"

    resolved_df, dataset_label = load_resolved_for_month(ym, is_current_month)
    if resolved_df is not None:
        resolved_df = resolved_df.copy()
        if "series_semantics" not in resolved_df.columns:
            resolved_df["series_semantics"] = "stock"
        else:
            resolved_df["series_semantics"] = (
                resolved_df["series_semantics"].fillna("").replace("", "stock")
            )
    return resolved_df, dataset_label, "stock"
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
    parser.add_argument(
        "--series",
        choices=["new", "stock"],
        default="new",
        help="Return monthly NEW deltas (default) or STOCK totals.",
    )
    parser.add_argument("--json_only", action="store_true", help="Print JSON only (no human summary)")
    args = parser.parse_args()

    countries, shocks = load_registries()
    country_name, iso3 = resolve_country(countries, args.country, args.iso3)
    hazard_label, hazard_code, hazard_class = resolve_hazard(shocks, args.hazard, args.hazard_code)

    ym = ym_from_cutoff(args.cutoff)
    current_month = ym == current_ym_istanbul()
    series_requested = args.series
    df, source_dataset, series_used = load_series_for_month(ym, current_month, series_requested)

    def fallback_to_stock(reason: str) -> None:
        nonlocal df, source_dataset, series_used
        print(reason, file=sys.stderr)
        df_stock, stock_dataset, stock_series = load_series_for_month(
            ym, current_month, "stock"
        )
        df = df_stock
        source_dataset = stock_dataset
        series_used = stock_series

    def emit_no_data(message: str) -> None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": message,
                    "iso3": iso3,
                    "hazard_code": hazard_code,
                    "cutoff": args.cutoff,
                    "series_requested": series_requested,
                }
            ),
            flush=True,
        )
        if not args.json_only:
            print("\n" + message, file=sys.stderr)
        sys.exit(1)

    if df is None and series_requested == "new":
        fallback_to_stock(f"Note: No deltas for {ym}; returning stock totals.")

    if df is None:
        message = (
            "No data found. Expected snapshot at snapshots/"
            f"{ym}/facts.parquet, exports/resolved(_reviewed).csv, or exports/deltas.csv."
        )
        emit_no_data(message)

    row = select_row(df, iso3, hazard_code, args.cutoff)
    if not row and series_requested == "new" and series_used == "new":
        fallback_to_stock(
            f"Note: No delta record for iso3={iso3}, hazard={hazard_code} at {ym}; returning stock totals."
        )
        if df is None:
            message = (
                "No data found. Expected snapshot at snapshots/"
                f"{ym}/facts.parquet, exports/resolved(_reviewed).csv, or exports/deltas.csv."
            )
            emit_no_data(message)
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
                    "series_requested": series_requested,
                }
            ),
            flush=True,
        )
        if not args.json_only:
            print("\n" + message, file=sys.stderr)
        sys.exit(1)

    snapshot_used = source_dataset in {"snapshot", "snapshot_deltas"}
    if source_dataset == "monthly_deltas":
        source_bucket = "state"
    elif snapshot_used:
        source_bucket = "snapshot"
    else:
        source_bucket = "exports"

    row_series = str(row.get("series_semantics", "")).strip().lower() or series_used

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
        "series_semantics": row_series,
        "series_requested": series_requested,
        "series_returned": row_series,
        "ym": row.get("ym", ym),
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
    if output["series_returned"] != output["series_requested"]:
        print(
            f"Series returned: {output['series_returned']} (requested {output['series_requested']})"
        )
    else:
        print(f"Series: {output['series_returned']}")
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
