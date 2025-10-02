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
- monthly resolved + deltas split to resolver/state/monthly/YYYY-MM/{resolved.csv,deltas.csv}

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


def prune_old_daily(retain_days: int) -> None:
    """Keep only the newest ``retain_days`` daily state folders."""
    if retain_days < 0:
        return
    daily_root = STATE / "daily"
    if not daily_root.exists():
        return

    folders = sorted([d for d in daily_root.iterdir() if d.is_dir()])
    if len(folders) <= retain_days:
        return

    to_remove = folders[: max(0, len(folders) - retain_days)]
    for folder in to_remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"Warning: could not prune {folder}: {exc}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pr","daily"])
    ap.add_argument("--id", required=True, help="PR number for mode=pr, or YYYY-MM-DD for mode=daily")
    ap.add_argument(
        "--retain-days",
        type=int,
        default=10,
        help="When mode=daily, keep only this many most recent daily state folders",
    )
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

    write_monthly_outputs(EXPORTS / "resolved.csv", EXPORTS / "deltas.csv")

    if args.mode == "daily":
        prune_old_daily(args.retain_days)


def read_rows_by_month(src: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    if not src.exists():
        return {}, []

    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "ym" not in fieldnames:
            return {}, []

        rows_by_month: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            ym = (row.get("ym") or "").strip()
            if not ym:
                continue
            rows_by_month.setdefault(ym, []).append(row)

    return rows_by_month, fieldnames


def write_monthly_outputs(resolved: Path, deltas: Path) -> None:
    resolved_by_month, resolved_fields = read_rows_by_month(resolved)
    deltas_by_month, deltas_fields = read_rows_by_month(deltas)

    if not resolved_by_month and not deltas_by_month:
        return

    months = sorted(set(resolved_by_month) | set(deltas_by_month))
    monthly_base = STATE / "monthly"

    for ym in months:
        out_dir = monthly_base / ym
        out_dir.mkdir(parents=True, exist_ok=True)

        if resolved_fields:
            out_path = out_dir / "resolved.csv"
            with out_path.open("w", encoding="utf-8", newline="") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=resolved_fields)
                writer.writeheader()
                writer.writerows(resolved_by_month.get(ym, []))

        if deltas_fields:
            out_path = out_dir / "deltas.csv"
            with out_path.open("w", encoding="utf-8", newline="") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=deltas_fields)
                writer.writeheader()
                writer.writerows(deltas_by_month.get(ym, []))

if __name__ == "__main__":
    main()
