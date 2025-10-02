#!/usr/bin/env python3
"""
precedence_engine.py — select one authoritative total per country/hazard at a cut-off.

Example:
  python resolver/tools/precedence_engine.py \
    --facts resolver/exports/facts.csv \
    --cutoff 2025-09-30

Outputs:
  resolver/exports/resolved.csv
  resolver/exports/resolved.jsonl
  resolver/exports/resolved_diagnostics.csv   # conflicts & reasons

Rules implemented (from policy A2):
  - Metric preference: in_need → else affected (configurable)
  - Source precedence tiers with publisher/source_type mapping
  - 7-day publication lag allowed if as_of ≤ cutoff
  - One total only per (iso3, hazard_code)
  - Conflict rule: if >20% apart within same tier, pick latest as_of; record alternative
  - Always carry citation fields and definition_text
"""

import argparse, sys, json, datetime as dt
from pathlib import Path
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow pyyaml python-dateutil' to run the engine.", file=sys.stderr)
    sys.exit(2)

try:
    import yaml
except ImportError:
    print("Please 'pip install pyyaml' to run the engine.", file=sys.stderr)
    sys.exit(2)

try:
    from dateutil import parser as date_parser
except ImportError:
    print("Please 'pip install python-dateutil' to run the engine.", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPORTS = ROOT / "exports"
CONFIG = TOOLS / "precedence_config.yml"
ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")

def _load(f: Path) -> pd.DataFrame:
    ext = f.suffix.lower()
    if ext in (".csv", ".tsv"):
        return pd.read_csv(f, dtype=str).fillna("")
    elif ext == ".parquet":
        return pd.read_parquet(f)
    else:
        raise SystemExit(f"Unsupported facts format: {ext}")

def _load_cfg() -> Dict[str, Any]:
    with open(CONFIG, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

def _to_date(s: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _parse_as_of(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = date_parser.isoparse(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ISTANBUL_TZ)
    return parsed.astimezone(ISTANBUL_TZ)


def _as_of_local_date(value: str) -> dt.date | None:
    parsed = _parse_as_of(value)
    return parsed.date() if parsed else None

def _within_pub_lag(pub: str, cutoff: dt.date, lag_days: int) -> bool:
    pub_date = _to_date(pub)
    if pub_date is None:
        return False
    return pub_date <= cutoff + dt.timedelta(days=lag_days)

def _assign_tier(row, mapping: Dict[str, Any], tiers: List[str]) -> int:
    st = str(row.get("source_type",""))
    pub = str(row.get("publisher",""))
    pub_norm = pub.lower()
    for idx, tier in enumerate(tiers):
        spec = mapping.get(tier, {})
        stypes = [x.lower() for x in spec.get("source_type",[])]
        pubs   = [x.lower() for x in spec.get("publisher",[])]
        ok_st = (not stypes) or (st.lower() in stypes)
        ok_pub = (not pubs) or ("*" in pubs) or any(p in pub_norm for p in pubs)
        if ok_st and ok_pub:
            return idx
    return len(tiers)  # lowest (unknown)

def _pct_diff(a: float, b: float) -> float:
    if a == 0 and b == 0:
        return 0.0
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom * 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True, help="Path to canonical facts CSV/Parquet")
    ap.add_argument("--cutoff", required=True, help="Cut-off date YYYY-MM-DD (Europe/Istanbul at 23:59)")
    ap.add_argument("--outdir", default=str(EXPORTS), help="Output directory")
    args = ap.parse_args()

    facts_path = Path(args.facts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg()
    facts = _load(facts_path)

    # Basic type coercions
    for c in facts.columns:
        if c not in ("value",):
            facts[c] = facts[c].astype(str).fillna("")
    # numeric
    facts["value_num"] = pd.to_numeric(facts["value"], errors="coerce").fillna(-1)

    # Filter by cutoff & lag
    cutoff_date = dt.date.fromisoformat(args.cutoff)
    lag_days = int(cfg["cutoff"]["lag_days_allowed"])

    eligible = facts.copy()

    def _as_of_valid(value: str) -> bool:
        parsed = _as_of_local_date(value)
        return (parsed is not None) and (parsed <= cutoff_date)

    # as_of must be <= cutoff
    eligible = eligible[eligible["as_of_date"].apply(_as_of_valid)]
    # publication must be <= cutoff + lag
    eligible = eligible[eligible["publication_date"].apply(lambda x: _within_pub_lag(x, cutoff_date, lag_days))]

    # Assign source_tier index
    tiers = cfg["source_precedence"]
    eligible["source_tier"] = eligible.apply(lambda r: _assign_tier(r, cfg["source_mapping"], tiers), axis=1)

    # Resolve metric per group with preference list
    m_pref = cfg["metric_preference"]

    records = []
    diags = []

    group_cols = ["iso3","hazard_code"]
    for (iso3, hz), g in eligible.groupby(group_cols, dropna=False):

        # iterate metric preference
        chosen_metric = None
        chosen_row = None

        for metric in m_pref:
            mg = g[g["metric"] == metric].copy()
            if mg.empty:
                continue

            # Take best tier (lowest index)
            best_tier = mg["source_tier"].min()
            mg = mg[mg["source_tier"] == best_tier].copy()

            # Within tier: apply conflict rule
            mg["val"] = mg["value_num"]
            mg = mg[mg["val"] >= 0].copy()
            if mg.empty:
                continue

            # Sort by Istanbul-aware timestamps to respect localized cutoff semantics
            sentinel = dt.datetime(1900, 1, 1, tzinfo=ISTANBUL_TZ)
            mg["_as_of_sort"] = mg["as_of_date"].apply(lambda x: _parse_as_of(x) or sentinel)
            mg["_pub_sort"] = mg["publication_date"].apply(lambda x: _parse_as_of(x) or sentinel)
            mg = mg.sort_values(by=["_as_of_sort", "_pub_sort"], ascending=[False, False])
            mg = mg.drop(columns=["_as_of_sort", "_pub_sort"], errors="ignore")

            top = mg.iloc[0]
            chosen_metric = metric
            chosen_row = top

            # If there is a second candidate in same tier far apart, record diagnostics
            if len(mg) > 1:
                second = mg.iloc[1]
                diff = _pct_diff(float(top["val"]), float(second["val"]))
                if diff > float(cfg["conflict_rule"]["threshold_pct"]):
                    diags.append({
                        "iso3": iso3,
                        "hazard_code": hz,
                        "metric": metric,
                        "kept_value": top["val"],
                        "kept_publisher": top["publisher"],
                        "kept_source_url": top["source_url"],
                        "alt_value": second["val"],
                        "alt_publisher": second["publisher"],
                        "alt_source_url": second["source_url"],
                        "pct_diff": round(diff, 2),
                        "note": "conflict > threshold; kept latest as_of within best tier"
                    })
            break  # stop after first available metric in preference list

        if chosen_row is None:
            # Nothing for preferred metrics; skip this iso3/hazard
            continue

        as_of_dt = _parse_as_of(chosen_row["as_of_date"])
        ym_label = as_of_dt.strftime("%Y-%m") if as_of_dt else ""
        as_of_norm = as_of_dt.strftime("%Y-%m-%d") if as_of_dt else str(chosen_row.get("as_of_date", "") or "")

        rec = {
            # Keys for downstream
            "iso3": iso3,
            "hazard_code": hz,
            "hazard_label": chosen_row["hazard_label"],
            "hazard_class": chosen_row["hazard_class"],
            "metric": chosen_metric,
            "value": int(float(chosen_row["val"])),
            "unit": chosen_row.get("unit","persons"),
            "ym": ym_label,
            "as_of": as_of_norm,
            "as_of_date": as_of_norm,
            "publication_date": chosen_row["publication_date"],
            "publisher": chosen_row["publisher"],
            "source_type": chosen_row["source_type"],
            "source_url": chosen_row["source_url"],
            "source_name": chosen_row["publisher"],
            "doc_title": chosen_row["doc_title"],
            "definition_text": chosen_row["definition_text"],
            "precedence_tier": tiers[int(chosen_row["source_tier"])] if int(chosen_row["source_tier"]) < len(tiers) else "unknown",
            "event_id": chosen_row["event_id"],
            "proxy_for": chosen_row.get("proxy_for",""),
            "confidence": chosen_row.get("confidence",""),
        }
        records.append(rec)

    if not records:
        print("No eligible records found at this cutoff.", file=sys.stderr)

    # Write outputs
    resolved = pd.DataFrame(records)
    csv_out = outdir / "resolved.csv"
    jl_out  = outdir / "resolved.jsonl"
    diag_out = outdir / "resolved_diagnostics.csv"

    if len(resolved):
        resolved.to_csv(csv_out, index=False)
        with open(jl_out, "w", encoding="utf-8") as f:
            for _, r in resolved.iterrows():
                f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}, ensure_ascii=False) + "\n")
    pd.DataFrame(diags).to_csv(diag_out, index=False)

    print("✅ precedence engine complete")
    print(f" - {csv_out}")
    print(f" - {jl_out}")
    print(f" - {diag_out}")

if __name__ == "__main__":
    main()
