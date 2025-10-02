#!/usr/bin/env python3
"""Guardrail for resolver artifacts and repository size.

This script is intended to run in CI after exports/review steps. It warns when
fresh artifacts exceed recommended sizes and fails the job if the tracked repo
contents grow beyond an upper bound.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
SNAPSHOTS = ROOT / "snapshots"


def _to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _check_artifact(path: Path, limit_mb: float, label: str) -> Tuple[bool, str]:
    if not path.exists():
        return False, ""
    size_mb = _to_mb(path.stat().st_size)
    if size_mb > limit_mb:
        rel = path.relative_to(ROOT)
        return True, f"{label}: {rel} is {size_mb:.1f} MB (limit {limit_mb:.1f} MB)"
    return False, ""


def _iter_tracked_files() -> Iterable[Path]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk the working tree while skipping the Git metadata folder.
        for dirpath, dirnames, filenames in os.walk(ROOT):
            if ".git" in dirnames:
                dirnames.remove(".git")
            for name in filenames:
                yield Path(dirpath) / name
        return

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        yield ROOT / line


def _compute_repo_size_mb() -> float:
    total_bytes = 0
    for path in _iter_tracked_files():
        try:
            total_bytes += path.stat().st_size
        except FileNotFoundError:
            continue
    return _to_mb(total_bytes)


def main() -> None:
    limit_parquet_mb = float(os.getenv("RESOLVER_LIMIT_PARQUET_MB", "150"))
    limit_csv_mb = float(os.getenv("RESOLVER_LIMIT_CSV_MB", "25"))
    limit_repo_mb = float(os.getenv("RESOLVER_LIMIT_REPO_MB", "2000"))

    warnings: List[str] = []

    checks = [
        (EXPORTS / "facts.parquet", limit_parquet_mb, "exports"),
        (EXPORTS / "resolved.csv", limit_csv_mb, "resolved.csv"),
        (EXPORTS / "resolved.jsonl", limit_csv_mb * 2, "resolved.jsonl"),
    ]

    for path, limit, label in checks:
        is_over, message = _check_artifact(path, limit, label)
        if is_over:
            warnings.append(message)

    if SNAPSHOTS.exists():
        monthly_dirs = sorted([d for d in SNAPSHOTS.iterdir() if d.is_dir()])
        if monthly_dirs:
            latest = monthly_dirs[-1]
            snapshot_path = latest / "facts.parquet"
            is_over, message = _check_artifact(
                snapshot_path, limit_parquet_mb, f"snapshot:{latest.name}"
            )
            if is_over:
                warnings.append(message)

    repo_size_mb = _compute_repo_size_mb()

    if warnings:
        print("⚠️ Artifact size warnings:")
        for msg in warnings:
            print(f" - {msg}")
    else:
        print("✅ Artifact sizes within thresholds")

    print(f"ℹ️ Repo tracked size ≈ {repo_size_mb:.0f} MB (limit {limit_repo_mb:.0f} MB)")

    if repo_size_mb > limit_repo_mb:
        print("❌ Repo size limit exceeded", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
