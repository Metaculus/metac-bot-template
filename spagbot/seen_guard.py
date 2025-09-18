# spagbot/seen_guard.py
"""
seen_guard.py
-------------
Prevents answering the same Metaculus question multiple times too quickly.

This module does two things:
1) Looks at the existing forecasts CSV and a dedicated state file to filter out
   question IDs that have been answered within the configured cooldown window.
2) Uses simple lockfiles to avoid double-processing the same question inside a single run
   (or overlapping runs). As soon as a question is selected for processing, we create a
   lockfile. If another process tries to pick it up while the lock exists, it skips it.

Environment variables respected:
- FORECASTS_CSV_PATH   (default: "forecasts.csv")
- SEEN_GUARD_PATH      (default: "forecast_logs/state/seen_forecasts.jsonl")
- SEEN_COOLDOWN_HOURS  (default: 24)
- FORECAST_LOCK_DIR    (default: "forecast_logs/locks")
"""

from __future__ import annotations

import csv
import os
import sys
import json
import errno
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Dict, Any, Set, Generator, Optional

# ---------------------------------------------------------------------------
# Helpers to read environment variables with good defaults
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val is not None and val.strip() != "" else default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default

# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class AlreadySeenError(Exception):
    """Custom exception for questions seen within the cooldown period."""
    pass

# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

@dataclass
class SeenGuard:
    """Manages the state of seen questions and provides filtering logic."""
    csv_path: str = field(default_factory=lambda: _env_str("FORECASTS_CSV_PATH", "forecasts.csv"))
    state_file_path: str = field(default_factory=lambda: _env_str("SEEN_GUARD_PATH", "forecast_logs/state/seen_forecasts.jsonl"))
    lock_dir: str = field(default_factory=lambda: _env_str("FORECAST_LOCK_DIR", "forecast_logs/locks"))
    cooldown: timedelta = field(default_factory=lambda: timedelta(hours=_env_int("SEEN_COOLDOWN_HOURS", 24)))

    def _get_qid(self, post: Dict[str, Any]) -> Optional[int]:
        """Safely extracts the question ID from a post object."""
        q = post.get("question", {}) or {}
        qid = q.get("id")
        return int(qid) if qid is not None else None

    def _load_history_from_state_file(self) -> Set[int]:
        """Loads recently seen question IDs from the JSONL state file."""
        if not os.path.exists(self.state_file_path):
            return set()

        seen_qids = set()
        now_utc = datetime.now(timezone.utc)
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if (now_utc - timestamp) < self.cooldown:
                            seen_qids.add(int(data["question_id"]))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            print(f"[seen_guard] Error reading state file {self.state_file_path}: {e!r}", file=sys.stderr)
        return seen_qids

    def _load_history_from_csv(self) -> Set[int]:
        """Loads recently seen question IDs by reading the main forecasts CSV."""
        if not os.path.exists(self.csv_path):
            return set()

        seen_qids = set()
        now_utc = datetime.now(timezone.utc)
        try:
            with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Assuming 'run_time_iso' or a similar timestamp column exists
                        ts_str = row.get("run_time_iso") or row.get("RunTime")
                        qid_str = row.get("question_id") or row.get("QuestionID")
                        if not ts_str or not qid_str:
                            continue
                        
                        # Handle various possible ISO formats, assuming UTC if no timezone is specified
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)

                        if (now_utc - timestamp) < self.cooldown:
                            seen_qids.add(int(qid_str))
                    except (ValueError, KeyError, TypeError):
                        continue
        except Exception as e:
            print(f"[seen_guard] Error reading CSV {self.csv_path}: {e!r}", file=sys.stderr)
        return seen_qids
        
    def _get_recently_seen_qids(self) -> Set[int]:
        """Consolidates seen QIDs from both the state file and the main CSV."""
        from_state = self._load_history_from_state_file()
        from_csv = self._load_history_from_csv()
        return from_state.union(from_csv)

    def filter_unseen_posts(self, posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Given a list of post dicts, returns only those not seen within the cooldown period."""
        if not posts:
            return []
        
        seen_qids = self._get_recently_seen_qids()
        
        unseen_posts = []
        skipped_count = 0
        for post in posts:
            qid = self._get_qid(post)
            if qid and qid in seen_qids:
                skipped_count += 1
            else:
                unseen_posts.append(post)
        
        print(f"[seen_guard] {skipped_count} duplicate(s) skipped; {len(unseen_posts)} fresh post(s) remain.")
        return unseen_posts

    def mark_seen(self, question_id: int | str) -> None:
        """Appends a record to the state file to mark a question as seen."""
        state_dir = os.path.dirname(self.state_file_path)
        os.makedirs(state_dir, exist_ok=True)
        
        record = {
            "question_id": int(question_id),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self.state_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[seen_guard] Failed to write to state file for QID {question_id}: {e!r}", file=sys.stderr)

    def _lock_path_for_qid(self, qid: str) -> str:
        """Computes the file path for a given question ID's lock file."""
        os.makedirs(self.lock_dir, exist_ok=True)
        return os.path.join(self.lock_dir, f"qid_{qid}.lock")

    @contextmanager
    def lock(self, question_id: int | str) -> Generator[bool, None, None]:
        """
        A context manager to lock a question ID for the duration of processing.
        Yields True if the lock was acquired, False otherwise.
        """
        qid_str = str(question_id)
        lock_path = self._lock_path_for_qid(qid_str)

        try:
            # O_CREAT | O_EXCL is an atomic "create if not exists" operation.
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            yield True
        except OSError as e:
            if e.errno == errno.EEXIST:
                # The file already exists, meaning it's locked by another process.
                yield False
            else:
                print(f"[seen_guard] Unexpected error acquiring lock for QID {qid_str}: {e!r}", file=sys.stderr)
                yield False
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

# ---------------------------------------------------------------------------
# Singleton instance and public functions
# ---------------------------------------------------------------------------

_GUARD = SeenGuard()

def filter_unseen_posts(posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _GUARD.filter_unseen_posts(posts)

def mark_seen(question_id: int | str) -> None:
    _GUARD.mark_seen(question_id)

def lock(question_id: int | str):
    """Public-facing lock function that uses the singleton guard instance."""
    return _GUARD.lock(question_id)

# --- Defensive Patch for backward compatibility ---
# This block is here to prevent an ImportError from old code.
def filter_post_ids(post_ids: list[int]) -> list[int]:
    print("[warn] An obsolete function ('filter_post_ids') was called. Please find and remove the call.")
    return post_ids

def mark_post_seen(post_id: int, meta=None) -> None:
    print(f"[warn] An obsolete function ('mark_post_seen') was called for post {post_id}. Please find and remove the call.")