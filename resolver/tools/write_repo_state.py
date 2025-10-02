#!/usr/bin/env python3
"""
write_repo_state.py — copy current run outputs into repo paths for committing.

Usage examples:
  # PR run:
  python resolver/tools/write_repo_state.py --mode pr --id 123

  # Nightly (non-PR):
  python resolver/tools/write_repo_state.py --mode daily --id 2025-09-30

This script copies:
- exports/facts.csv, exports/resolved.csv, exports/resolved_diagnostics.csv, exports/deltas.csv
- review/review_queue.csv
- snapshots/YYYY-MM/facts.parquet + manifest.json (if exist)
- monthly deltas split to resolver/state/monthly/YYYY-MM/deltas.csv

To:
- PR:     resolver/state/pr/<PR_NUMBER>/{exports/*,review/*}
- Daily:  resolver/state/daily/<YYYY-MM-DD>/{exports/*,review/*}
- Snap:   resolver/snapshots/<YYYY-MM>/*  (if present)
"""

import argparse, csv, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
REVIEW  = ROOT / "review"
SNAPS   = ROOT / "snapshots"
STATE   = ROOT / "state"

def safe_copy(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def copy_dir(src_dir: Path, dst_dir: Path, patterns: list[str]):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ptn in patterns:
        for p in src_dir.glob(ptn):
            if p.is_file():
                shutil.copy2(p, dst_dir / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pr","daily"])
    ap.add_argument("--id", required=True, help="PR number for mode=pr, or YYYY-MM-DD for mode=daily")
    args = ap.parse_args()

    if args.mode == "pr":
        base = STATE / "pr" / str(args.id)
    else:
        base = STATE / "daily" / str(args.id)

    # Copy exports files
    copy_dir(
        EXPORTS,
        base / "exports",
        ["facts.csv","resolved.csv","resolved.jsonl","resolved_diagnostics.csv","deltas.csv"],
    )

    # Copy review file
    safe_copy(REVIEW / "review_queue.csv", base / "review" / "review_queue.csv")

    # Copy any snapshot written in this run (pattern snapshots/*/*)
    if SNAPS.exists():
        for ym_dir in SNAPS.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]"):
            f = ym_dir / "facts.parquet"
            m = ym_dir / "manifest.json"
            if f.exists() and m.exists():
                # We copy snapshots in-place (they already live under resolver/snapshots/),
                # so no duplicate under state/. Leaving them where they are is enough.
                pass

    write_monthly_deltas(EXPORTS / "deltas.csv")


def write_monthly_deltas(src: Path) -> None:
    if not src.exists():
        return

    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "ym" not in fieldnames:
            return

        rows_by_month: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            ym = row.get("ym")
            if not ym:
                continue
            rows_by_month.setdefault(ym, []).append(row)

    monthly_base = STATE / "monthly"
    for ym, rows in rows_by_month.items():
        out_dir = monthly_base / ym
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "deltas.csv"
        with out_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    main()
