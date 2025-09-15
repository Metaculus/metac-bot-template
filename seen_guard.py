# spagbot/seen_guard.py
# -----------------------------------------------------------------------------
# Purpose
#   Robust duplicate-prevention for Spagbot runs:
#     1) Reads your committed forecasts.csv to learn what has been handled.
#     2) Honors an optional cooldown window (skip only if seen within N hours).
#     3) Maintains a tiny JSONL registry for immediate persistence during a run.
#     4) Uses lock files to prevent double-processing within the same job.
#
# Public API (used by cli.py):
#   - filter_post_ids(ids: Iterable[int|str]) -> List[str or int]
#   - assert_not_seen(post_id: int|str, *, mark_on_pass: bool=True, meta: dict|None=None) -> None
#       raises AlreadySeenError when duplicate (respecting cooldown policy)
#   - mark_post_seen(post_id: int|str, meta: dict|None=None) -> None
#   - AlreadySeenError(Exception)
#
# How it works (mental model):
#   - CSV source of truth: we parse forecasts.csv and build:
#       * seen_ids: set[str] of post IDs
#       * latest_time_by_id: map[str, datetime] of the most recent timestamp per post
#   - JSONL sidecar: .spagbot/seen_forecasts.jsonl (append-only), with per-run marks.
#   - Lock files: forecast_logs/locks/{pid}.lock to stop duplicate handling within a single run.
#
# Configuration via env vars (safe defaults exist):
#   FORECASTS_CSV_PATH        default "forecasts.csv"
#   SEEN_COOLDOWN_HOURS       default "0" (0 = skip if ever seen; >0 = skip only if seen within window)
#   FORECAST_LOCK_DIR         default "forecast_logs/locks"
#   SEENGUARD_DIR             default ".spagbot"
#   SEENGUARD_FILE            default "seen_forecasts.jsonl"
#   SEENGUARD_NS              default "tournament"  (namespace tag; purely informational)
#
# Notes:
#   - We try to be forgiving with CSV header names for ID/time columns.
#   - If CSV is missing or empty, we fall back to JSONL (and then to nothing).
#   - If cooldown > 0 but we have no usable timestamp for a seen item, we treat it as "seen".
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Dict
from datetime import datetime, timedelta

# ---------------------------
# Configuration (env-driven)
# ---------------------------
CSV_PATH = os.getenv("FORECASTS_CSV_PATH", "forecasts.csv")
SEEN_COOLDOWN_HOURS = float(os.getenv("SEEN_COOLDOWN_HOURS", "0"))
LOCK_DIR = Path(os.getenv("FORECAST_LOCK_DIR", "forecast_logs/locks"))

SEEN_DIR = Path(os.getenv("SEENGUARD_DIR", ".spagbot"))
SEEN_FILE = os.getenv("SEENGUARD_FILE", "seen_forecasts.jsonl")
SEEN_NS = os.getenv("SEENGUARD_NS", "tournament")

# Flexible header names (we accept the first one we find)
ID_COLS = ["post_id", "pid", "qid", "question_id", "metaculus_post_id"]
TIME_COLS = ["timestamp", "created_at", "ist_iso", "time_iso"]

# ---------------------------
# Exceptions
# ---------------------------
class AlreadySeenError(Exception):
    """Raised when a duplicate post is detected (respecting cooldown policy)."""
    pass

# ---------------------------
# Date/time helpers
# ---------------------------
def _parse_dt(s: str) -> Optional[datetime]:
    """Try a few ISO-ish formats; return None if parsing fails."""
    if not s:
        return None
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",    "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",      "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def _utc_now() -> datetime:
    # naive UTC is fine for comparisons if we compare consistent naives
    return datetime.utcnow()

# ---------------------------
# Filesystem helpers
# ---------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _lock_path(pid: str) -> Path:
    return LOCK_DIR / f"{pid}.lock"

def _is_locked(pid: str) -> bool:
    return _lock_path(pid).exists()

def _touch_lock(pid: str) -> None:
    _ensure_dir(LOCK_DIR)
    _lock_path(pid).write_text("locked", encoding="utf-8")

# ---------------------------
# CSV loader (forecasts.csv)
# ---------------------------
def _find_first_col(header: List[str], candidates: List[str]) -> Optional[str]:
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand.lower() in lowered:
            return header[lowered.index(cand.lower())]  # return original-cased name
    return None

def _load_csv_seen(csv_path: Path) -> Tuple[Set[str], Dict[str, datetime]]:
    """
    Returns:
      - seen_ids: set of post IDs (as strings)
      - latest_time_by_id: map of post ID -> latest datetime we can parse
    """
    seen_ids: Set[str] = set()
    latest_time_by_id: Dict[str, datetime] = {}
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
            # Without an ID column we cannot dedupe; return empty to be safe.
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

# ---------------------------
# JSONL registry (sidecar)
# ---------------------------
class _JsonlRegistry:
    """
    Append-only JSONL file to mark "seen" items during/after a run
    so future runs can skip immediately (even if CSV write/commit lags).
    Format per line:
      {"ns": "...", "pid": "12345", "ts": 1699999999, "meta": {...}}
    """
    def __init__(self, dir_path: Path, filename: str, ns: str) -> None:
        self.ns = ns
        self.dir = dir_path
        _ensure_dir(self.dir)
        self.path = self.dir / filename
        self._seen: Set[str] = set()
        self._latest_time: Dict[str, datetime] = {}

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                        ns_v = rec.get("ns")
                        pid_v = str(rec.get("pid"))
                        if ns_v and pid_v and ns_v == self.ns:
                            self._seen.add(pid_v)
                            ts = rec.get("ts")
                            if isinstance(ts, (int, float)):
                                dt = datetime.utcfromtimestamp(ts)
                                prev = self._latest_time.get(pid_v)
                                if prev is None or dt > prev:
                                    self._latest_time[pid_v] = dt
                    except Exception:
                        # ignore malformed
                        continue

    def has(self, pid: str) -> bool:
        return pid in self._seen

    def latest(self, pid: str) -> Optional[datetime]:
        return self._latest_time.get(pid)

    def mark(self, pid: str, meta: Optional[dict] = None) -> None:
        rec = {"ns": self.ns, "pid": pid, "ts": int(time.time()), "meta": meta or {}}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._seen.add(pid)
        self._latest_time[pid] = datetime.utcfromtimestamp(rec["ts"])

_JSONREG = _JsonlRegistry(SEEN_DIR, SEEN_FILE, SEEN_NS)

# ---------------------------
# Merge CSV + JSONL "seen" info
# ---------------------------
def _merge_seen_sources() -> Tuple[Set[str], Dict[str, datetime]]:
    """
    Combines CSV and JSONL views:
      - seen_ids = union of both
      - latest_time_by_id = max timestamp across both when available
    """
    csv_seen, csv_latest = _load_csv_seen(Path(CSV_PATH))
    # start with CSV view
    seen_ids: Set[str] = set(csv_seen)
    latest_time_by_id: Dict[str, datetime] = dict(csv_latest)

    # fold in JSONL
    for pid in list(_JSONREG._seen):
        seen_ids.add(pid)
        jt = _JSONREG.latest(pid)
        if jt:
            prev = latest_time_by_id.get(pid)
            if prev is None or jt > prev:
                latest_time_by_id[pid] = jt

    return seen_ids, latest_time_by_id

# ---------------------------
# Public API
# ---------------------------
def filter_post_ids(ids: Iterable[int | str]) -> List[int]:
    """
    Batch-level filter: return ONLY the IDs we consider "fresh".
    Rules:
      - If SEEN_COOLDOWN_HOURS == 0: drop if ever seen (CSV or JSONL) or locked.
      - If SEEN_COOLDOWN_HOURS > 0: drop only if latest seen time >= now - cooldown.
      - Always drop IDs currently locked (protects against within-run duplicates).
    """
    incoming: List[str] = []
    for raw in ids or []:
        s = str(raw).strip()
        if s and s.isdigit():
            incoming.append(s)

    seen_ids, latest_time_by_id = _merge_seen_sources()
    now = _utc_now()
    fresh: List[int] = []

    for pid in incoming:
        # lock wins immediately
        if _is_locked(pid):
            continue

        if pid in seen_ids:
            if SEEN_COOLDOWN_HOURS <= 0:
                # seen at any time => drop
                continue
            # cooldown mode
            last = latest_time_by_id.get(pid)
            if last is None:
                # no timestamp info but it's seen -> safest to drop
                continue
            if last >= now - timedelta(hours=SEEN_COOLDOWN_HOURS):
                # within window -> drop
                continue

        # otherwise keep
        try:
            fresh.append(int(pid))
        except Exception:
            # ignore conversion anomalies
            pass

    return fresh

def assert_not_seen(post_id: int | str, *, mark_on_pass: bool = True, meta: Optional[dict] = None) -> None:
    """
    Call RIGHT BEFORE submitting a forecast for a single post.
    Raises AlreadySeenError if:
      - the post appears in CSV or JSONL, AND
      - either cooldown==0, or cooldown>0 and the latest time is within the window,
      - OR a lock file exists.
    If it passes and mark_on_pass=True, we create a lock and mark JSONL immediately
    to prevent later stages in the same job from re-processing the same post.
    """
    pid = str(post_id).strip()
    if not pid.isdigit():
        # If it's weird, don't block the run—just allow through.
        return

    if _is_locked(pid):
        raise AlreadySeenError(f"Post {pid} already locked in this run.")

    seen_ids, latest_time_by_id = _merge_seen_sources()
    if pid in seen_ids:
        if SEEN_COOLDOWN_HOURS <= 0:
            raise AlreadySeenError(f"Post {pid} already seen (no cooldown).")
        last = latest_time_by_id.get(pid)
        if last is None or last >= _utc_now() - timedelta(hours=SEEN_COOLDOWN_HOURS):
            raise AlreadySeenError(f"Post {pid} seen within cooldown window.")

    if mark_on_pass:
        # Preemptively lock + mark JSON to avoid races within this run
        _touch_lock(pid)
        try:
            _JSONREG.mark(pid, meta or {})
        except Exception:
            # Never fail the run for registry issues
            pass

def mark_post_seen(post_id: int | str, meta: Optional[dict] = None) -> None:
    """
    Call AFTER a successful CSV write (and/or successful submission) to persist
    that the post was handled. This makes the skip immediate on future runs
    even before a git commit finishes.
    """
    pid = str(post_id).strip()
    if not pid.isdigit():
        return
    try:
        _JSONREG.mark(pid, meta or {})
    except Exception:
        pass
    # Lock remains present from assert_not_seen(); if not, ensure it's there.
    try:
        if not _is_locked(pid):
            _touch_lock(pid)
    except Exception:
        pass
