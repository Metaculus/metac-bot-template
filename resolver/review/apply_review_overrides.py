#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow' to run the override applier.", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    resolved = pd.read_csv(args.resolved, dtype=str).fillna("")
    decisions = pd.read_csv(args.decisions, dtype=str).fillna("")

    for df in (resolved, decisions):
        df["key"] = (
            df["iso3"].astype(str)
            + "|"
            + df["hazard_code"].astype(str)
            + "|"
            + df["metric"].astype(str)
        )

    dec = decisions[
        ["key", "analyst_decision", "override_value", "override_source_url", "override_notes"]
    ].copy()

    merged = resolved.merge(dec, on="key", how="left", suffixes=("", "_dec"))

    out_rows = []
    out_status = []

    for _, r in merged.iterrows():
        action = (r.get("analyst_decision", "") or "").strip().lower()
        row = r.copy()
        status_label = "no_decision"

        if action == "drop":
            continue

        elif action == "override":
            val = (r.get("override_value", "") or "").strip()
            try:
                vnum = int(float(val))
                row["value"] = str(vnum)
                url = (r.get("override_source_url", "") or "").strip()
                if url:
                    row["source_url"] = url
                notes = (r.get("override_notes", "") or "").strip()
                if notes:
                    row["definition_text"] = (
                        row.get("definition_text", "") + f" [OVERRIDE NOTE: {notes}]"
                    ).strip()
                status_label = "overridden"
            except Exception:
                status_label = "no_decision"

        elif action == "keep":
            status_label = "kept"

        else:
            status_label = "no_decision"

        out_rows.append(row)
        out_status.append(status_label)

    out = pd.DataFrame(out_rows).drop(columns=["key"], errors="ignore")
    out["review_status"] = out_status

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✅ overrides applied → {args.out} (rows: {len(out)})")


if __name__ == "__main__":
    main()
