# seen_guard.py  — DROP-IN REPLACEMENT
# ------------------------------------------------------------
# Purpose:
#   Stop duplicate forecasting on the same Metaculus post across runs.
#
# How it works (simple mental model):
#   1) Load the already-committed forecasts.csv from the repository.
#   2) Build a set of "seen post IDs".
#   3) Offer:
#       - filter_post_ids([...])  -> returns only fresh IDs
#       - assert_not_seen(id)     -> raises if id already seen
#   4) (Optional) lock files to guard against double-handling within the same job.
#
# Where to call:
#   A) Immediately after you fetch open posts, BEFORE research:
#        open_ids = seen_guard.filter_post_ids(open_ids)
#      If this empties the list, stop the run gracefully.
#
#   B) Right before submitting a forecast for a single PID:
#        seen_guard.assert_not_seen(pid)
#
# Notes:
#   - We read FORECASTS_CSV_PATH (default "forecasts.csv").
#   - We try to be flexible about column names:
#       ID columns checked in priority order: post_id, pid, qid, question_id, metaculus_post_id
#       Time columns checked: timestamp, created_at, ist_iso, time_iso
#   - If a time column exists and you set SEEN_COOLDOWN_HOURS (>0), we will
#     skip only if the same PID was logged within that window; otherwise we skip on any prior appearance.
# ------------------------------------------------------------

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
from datetime import datetime, timedelta

# ---------------------------
# Environment/config knobs
# ---------------------------
CSV_PATH = os.getenv("FORECASTS_CSV_PATH", "forecasts.csv")
# If > 0, only skip if the same PID appears within this many hours (based on a usable timestamp column).
SEEN_COOLDOWN_HOURS = float(os.getenv("SEEN_COOLDOWN_HOURS", "0"))
# Lock files to prevent re-entrancy within one job:
LOCK_DIR = Path(os.getenv("FORECAST_LOCK_DIR", "forecast_logs/locks"))

# Candidate column names (we accept any one that exists)
ID_COLS = ["post_id", "pid", "qid", "question_id", "metaculus_post_id"]
TIME_COLS = ["timestamp", "created_at", "ist_iso", "time_iso"]

# ---------------------------
# Utilities
# ---------------------------

def _parse_dt(s: str) -> Optional[datetime]:
    """Try a few ISO-ish datetime parses; return None if not parseable."""
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # If no timezone, parse naive
            dt = datetime.strptime(s, fmt)
            return dt
        except ValueError:
            continue
    return None

def _find_first_col(header: List[str], candidates: List[str]) -> Optional[str]:
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand.lower() in lowered:
            # return the original-cased header name
            return header[lowered.index(cand.lower())]
    return None

def _load_seen_and_latest_time(csv_path: Path) -> Tuple[Set[str], dict]:
    """
    Returns:
      - seen_ids: set of stringified IDs that appear at least once
      - latest_time_by_id: mapping id -> latest datetime (when a time column exists)
    """
    seen_ids: Set[str] = set()
    latest_time_by_id: dict = {}
    if not csv_path.exists():
        return seen_ids, latest_time_by_id

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return seen_ids, latest_time_by_id

        id_col = _find_first_col(header, ID_COLS)
        time_col = _find_first_col(header, TIME_COLS)

        if not id_col:
            # If we can't find an ID column, we cannot dedupe—return empty sets so nothing is skipped silently.
            return seen_ids, latest_time_by_id

        idx_id = header.index(id_col)
        idx_time = header.index(time_col) if time_col else None

        for row in reader:
            if not row or len(row) <= idx_id:
                continue
            pid = row[idx_id].strip()
            if not pid:
                continue
            seen_ids.add(pid)
            if idx_time is not None and len(row) > idx_time:
                dt = _parse_dt(row[idx_time].strip())
                if dt:
                    prev = latest_time_by_id.get(pid)
                    if prev is None or dt > prev:
                        latest_time_by_id[pid] = dt
    return seen_ids, latest_time_by_id

def _as_str_ids(ids: Iterable[int | str]) -> List[str]:
    return [str(x).strip() for x in ids if str(x).strip()]

def _lock_path_for(pid: str) -> Path:
    return LOCK_DIR / f"{pid}.lock"

def _already_locked(pid: str) -> bool:
    p = _lock_path_for(pid)
    return p.exists()

def _touch_lock(pid: str) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    _lock_path_for(pid).write_text("locked", encoding="utf-8")

# ---------------------------
# Public API
# ---------------------------

class AlreadySeenError(RuntimeError):
    pass

def filter_post_ids(open_ids: Iterable[int | str]) -> List[str]:
    """
    Use this immediately after fetching open posts to filter out duplicates.
    """
    csv_path = Path(CSV_PATH)
    seen_ids, latest_time_by_id = _load_seen_and_latest_time(csv_path)
    incoming = _as_str_ids(open_ids)

    if SEEN_COOLDOWN_HOURS > 0 and latest_time_by_id:
        cutoff = datetime.utcnow() - timedelta(hours=SEEN_COOLDOWN_HOURS)
        keep: List[str] = []
        for pid in incoming:
            if pid not in seen_ids:
                keep.append(pid)
                continue
            # It is seen; only keep if older than cooldown (no usable time -> treat as seen)
            last_dt = latest_time_by_id.get(pid)
            if last_dt and last_dt < cutoff:
                keep.append(pid)
        return keep

    # Default: skip anything seen at all
    return [pid for pid in incoming if pid not in seen_ids and not _already_locked(pid)]

def assert_not_seen(pid: int | str) -> None:
    """
    Call this RIGHT BEFORE submitting to Metaculus for a single post.
    Raises AlreadySeenError if the post appears in the CSV or has a lock file.
    Also creates a lock so that concurrent paths in the same job don't double-submit.
    """
    pid_s = str(pid).strip()
    if _already_locked(pid_s):
        raise AlreadySeenError(f"Post {pid_s} already locked in this run.")

    seen_ids, latest_time_by_id = _load_seen_and_latest_time(Path(CSV_PATH))
    if pid_s in seen_ids:
        if SEEN_COOLDOWN_HOURS > 0 and latest_time_by_id:
            cutoff = datetime.utcnow() - timedelta(hours=SEEN_COOLDOWN_HOURS)
            last_dt = latest_time_by_id.get(pid_s)
            if last_dt and last_dt >= cutoff:
                raise AlreadySeenError(f"Post {pid_s} seen within cooldown window.")
            # else: allow submit (older than window)
        else:
            raise AlreadySeenError(f"Post {pid_s} already seen (no cooldown).")

    # Mark lock so we don’t double-handle within the same workflow run.
    _touch_lock(pid_s)
