#!/usr/bin/env python3
import sys
import datetime as dt
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow' to run the review builder.", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
REVIEW = ROOT / "review"

RESOLVED_CSV = EXPORTS / "resolved.csv"
DIAG_CSV = EXPORTS / "resolved_diagnostics.csv"
QUEUE_CSV = REVIEW / "review_queue.csv"
DECISIONS_EX = REVIEW / "decisions_example.csv"

TOP_TIERS = {"inter_agency_plan", "ifrc_or_gov_sitrep", "un_cluster_snapshot"}


def _is_date(s: str) -> bool:
    try:
        dt.date.fromisoformat(s)
        return True
    except Exception:
        return False


def load_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def main():
    if not RESOLVED_CSV.exists():
        print(f"Missing {RESOLVED_CSV}. Run precedence engine first.", file=sys.stderr)
        sys.exit(2)

    resolved = pd.read_csv(RESOLVED_CSV, dtype=str).fillna("")
    diags = load_or_empty(DIAG_CSV)

    resolved["key"] = (
        resolved["iso3"].astype(str)
        + "|"
        + resolved["hazard_code"].astype(str)
        + "|"
        + resolved["metric"].astype(str)
    )
    if not diags.empty:
        diags["key"] = (
            diags["iso3"].astype(str)
            + "|"
            + diags["hazard_code"].astype(str)
            + "|"
            + diags["metric"].astype(str)
        )
        conflict_keys = set(diags["key"])
    else:
        conflict_keys = set()

    today = dt.date.today().isoformat()

    def flags(r):
        a = r.get("as_of_date", "")
        p = r.get("publication_date", "")
        date_anomaly = "0"
        if _is_date(a) and _is_date(p):
            if a > p or p > today:
                date_anomaly = "1"
        else:
            date_anomaly = "1"

        tier = r.get("precedence_tier", "")
        f = {
            "conflict": "1" if r["key"] in conflict_keys else "0",
            "low_confidence": "1" if r.get("confidence", "").lower() == "low" else "0",
            "media_in_need": "1"
            if (r.get("source_type", "") == "media" and r.get("metric", "") == "in_need")
            else "0",
            "proxy_used": "1" if str(r.get("proxy_for", "")).upper() == "PIN" else "0",
            "tier_risk": "1" if tier not in TOP_TIERS else "0",
            "date_anomaly": date_anomaly,
        }
        f["needs_review"] = "1" if any(v == "1" for v in f.values()) else "0"
        return pd.Series(f)

    flagged = resolved.copy()
    flagged = pd.concat([flagged, flagged.apply(flags, axis=1)], axis=1)

    for col in [
        "analyst_decision",
        "override_value",
        "override_source_url",
        "override_notes",
    ]:
        flagged[col] = ""

    REVIEW.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(QUEUE_CSV, index=False)
    flagged.head(2).to_csv(DECISIONS_EX, index=False)

    print("✅ review queue written:")
    print(f" - {QUEUE_CSV}")
    print(f" - {DECISIONS_EX} (example)")


if __name__ == "__main__":
    main()
