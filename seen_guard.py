# seen_guard.py
# -----------------------------------------------------------------------------
# Persistent "seen item" registry for Spagbot.
# Stores one JSONL record per "seen" question so we can skip duplicates on
# subsequent runs. This is intentionally simple and robust.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, re
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_STATE_DIR = ".spagbot"
DEFAULT_STATE_FILE = "seen_forecasts.jsonl"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

class SeenGuard:
    """
    Minimal persistence layer:
      - Each line in the JSONL is a dict with keys: {"ns", "key", "ts", "meta"}.
      - ('ns','key') is the uniqueness pair.
    """
    def __init__(self,
                 state_dir: str = DEFAULT_STATE_DIR,
                 state_file: str = DEFAULT_STATE_FILE,
                 namespace: str = "default") -> None:
        self.ns = namespace
        self.dir = Path(state_dir)
        _ensure_dir(self.dir)
        self.path = self.dir / state_file
        self._seen = set()  # in-memory index of (ns, key)

        # Load existing seen items into memory
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        k = (rec.get("ns"), rec.get("key"))
                        if k[0] is not None and k[1] is not None:
                            self._seen.add(k)
                    except Exception:
                        # Don't crash on malformed lines
                        continue

    def has_seen(self, key: str) -> bool:
        """Return True if (ns, key) has been recorded."""
        return (self.ns, key) in self._seen

    def mark_seen(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Append a JSONL row and update the in-memory index."""
        rec = {
            "ns": self.ns,
            "key": key,
            "ts": int(time.time()),
            "meta": metadata or {}
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._seen.add((self.ns, key))


# Utility: extract Metaculus integer ID from a URL like:
#   https://www.metaculus.com/questions/39722/
METACULUS_ID_RE = re.compile(r"/questions/(\d+)/?")

def extract_metaculus_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = METACULUS_ID_RE.search(url)
    return m.group(1) if m else None
