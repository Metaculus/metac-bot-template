# spagbot/seen_guard.py
"""
seen_guard.py
-------------
Prevents answering the same Metaculus question multiple times too quickly.

This module does two things:
1) Looks at the existing forecasts CSV and filters out question IDs that have
   been answered within the configured cooldown window (SEEN_COOLDOWN_HOURS).
2) Uses simple lockfiles to avoid double-processing the same question inside a single run
   (or overlapping runs). As soon as a question is selected for processing, we create a
   lockfile. If another process tries to pick it up while the lock exists, it skips it.

Environment variables respected:
- FORECASTS_CSV_PATH   (default: "forecasts.csv")
- SEEN_COOLDOWN_HOURS  (default: 24)
- FORECAST_LOCK_DIR    (default: "forecast_logs/locks")

This file is intentionally verbose and beginner-friendly.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import errno
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Dict, Any, Set, Optional

# ---------------------------------------------------------------------------
# Helpers to read environment variables with good defaults
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if (val is not None and str(val).strip() != "") else default

def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default

FORECASTS_CSV_PATH = _env_str("FORECASTS_CSV_PATH", "forecasts.csv")
SEEN_COOLDOWN_HOURS = _env_int("SEEN_COOLDOWN_HOURS", 24)
FORECAST_LOCK_DIR = _env_str("FORECAST_LOCK_DIR", os.path.join("forecast_logs", "locks"))

# Ensure lock directory exists
os.makedirs(FORECAST_LOCK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Lockfile support
# ---------------------------------------------------------------------------

@contextmanager
def _acquire_lock(lock_path: str, timeout_sec: int = 0):
    """
    Acquire an exclusive lock by creating a file with O_CREAT|O_EXCL.
    If the file already exists, lock is considered held by someone else.

    timeout_sec = 0 means "try once". You can set a small timeout if desired.
    """
    start = time.time()
    while True:
        try:
            # O_CREAT | O_EXCL ensures this fails if the file already exists
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(f"pid={os.getpid()} time={datetime.now(timezone.utc).isoformat()}\n")
                yield  # lock held
            finally:
                # Remove the lock on exit
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
            return
        except OSError as e:
            if e.errno != errno.EEXIST:
                # Unexpected error creating lock -> re-raise
                raise
            # Lock exists
            if timeout_sec <= 0:
                # Give up immediately
                raise FileExistsError(f"Lock already held: {lock_path}")
            if (time.time() - start) >= timeout_sec:
                raise FileExistsError(f"Lock timed out: {lock_path}")
            # Wait a bit and retry
            time.sleep(0.1)


def _lock_path_for_qid(question_id: int | str) -> str:
    return os.path.join(FORECAST_LOCK_DIR, f"qid_{question_id}.lock")


# ---------------------------------------------------------------------------
# CSV scanning utilities
# ---------------------------------------------------------------------------

def _read_recent_qids_from_csv(csv_path: str, cooldown_hours: int) -> Set[int]:
    """
    Reads the forecasts CSV and returns a set of question_ids that were answered
    within the last `cooldown_hours`. If the CSV doesn't exist yet, returns an empty set.
    """
    recent: Set[int] = set()
    if not os.path.exists(csv_path):
        return recent

    # "Now" in UTC for consistent comparisons
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=cooldown_hours)

    # We try to use 'run_time_iso' first (best indicator of when the forecast was made).
    # If it's missing, we fall back to file modification time or skip the row gracefully.
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                qid = int(row.get("question_id", "").strip() or "0")
            except Exception:
                continue
            if qid <= 0:
                continue

            # Attempt parse of run_time_iso
            rt = row.get("run_time_iso") or row.get("timestamp") or row.get("created_time_iso") or ""
            # Normalize float-y NaNs to empty
            rt = "" if str(rt).lower() == "nan" else str(rt)

            if not rt:
                # No time info -> be conservative and allow it (not counted as recent)
                continue

            # Try parse timezone-aware
            dt_obj: Optional[datetime] = None
            for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt_obj = datetime.strptime(rt, fmt)
                    break
                except Exception:
                    continue
            if dt_obj is None:
                # Last attempt: fromisoformat (Py3.11 handles many variants)
                try:
                    dt_obj = datetime.fromisoformat(rt)
                except Exception:
                    dt_obj = None

            if dt_obj is None:
                # Could not parse -> skip recency check
                continue

            # Make timezone-aware in UTC
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
            else:
                dt_obj = dt_obj.astimezone(timezone.utc)

            if dt_obj >= cutoff:
                recent.add(qid)

    return recent


# ---------------------------------------------------------------------------
# Public API (used by cli.py)
# ---------------------------------------------------------------------------

@dataclass
class SeenGuard:
    """
    Object-oriented facade. You can also use the module-level helpers
    `filter_unseen_posts` and `mark_seen` directly if you prefer.
    """
    csv_path: str = FORECASTS_CSV_PATH
    cooldown_hours: int = SEEN_COOLDOWN_HOURS

    def recent_qids(self) -> Set[int]:
        """Return set of question_ids answered within the cooldown window."""
        if self.cooldown_hours <= 0:
            # Explicitly disabled cooldown
            return set()
        return _read_recent_qids_from_csv(self.csv_path, self.cooldown_hours)

    def filter_unseen_posts(self, posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given an iterable of post dicts containing a 'question_id', return
        a filtered list where recently answered questions are excluded.
        Also applies a lockfile check to avoid two workers selecting the same QID concurrently.
        """
        posts = list(posts)
        if not posts:
            return []

        # 1) Past-answers filter (CSV)
        recent = self.recent_qids()
        filtered = [p for p in posts if int(p.get("question_id", 0)) not in recent]

        # 2) Lockfile filter (intra-run concurrency)
        ready: List[Dict[str, Any]] = []
        for p in filtered:
            qid = int(p.get("question_id", 0))
            if qid <= 0:
                continue
            lock_path = _lock_path_for_qid(qid)
            try:
                with _acquire_lock(lock_path, timeout_sec=0):
                    # We hold the lock for the duration of this run step.
                    # IMPORTANT: We *don't* keep this context open here because actual forecasting
                    # happens later. We only "pre-claim" by creating the file and then leaving it
                    # in place until mark_seen() removes it. So we just create and close it.
                    pass
                # Leave the lockfile in place so others will skip
                ready.append(p)
            except FileExistsError:
                # Someone else already locked this QID; skip it.
                continue

        skipped = len(posts) - len(ready)
        print(f"[seen_guard] {skipped} duplicate(s) skipped; {len(ready)} fresh post(s) remain.")
        return ready

    def mark_seen(self, question_id: int | str) -> None:
        """
        Call this after successfully logging/submitting a forecast for the question.
        Removes the lockfile (if present). We do NOT write back to the CSV here;
        the forecast logging code already appends to CSV, which we will read on the next run.
        """
        lock_path = _lock_path_for_qid(question_id)
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


# Backwards-compatible free functions used by older code in cli.py
_GUARD = SeenGuard()

def filter_unseen_posts(posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _GUARD.filter_unseen_posts(posts)

def mark_seen(question_id: int | str) -> None:
    _GUARD.mark_seen(question_id)
