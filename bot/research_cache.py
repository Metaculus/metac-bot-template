from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bot.env import env_bool, env_float, env_int

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResearchCacheConfig:
    enabled: bool
    path: Path
    ttl_hours: float
    max_entries: int

    @classmethod
    def from_env(cls) -> "ResearchCacheConfig":
        default_path = ".cache/metaculus-bot/research_cache.json"
        return cls(
            enabled=env_bool("BOT_ENABLE_RESEARCH_CACHE", True),
            path=Path(os.getenv("BOT_RESEARCH_CACHE_PATH", default_path)),
            ttl_hours=max(0.0, env_float("BOT_RESEARCH_CACHE_TTL_HOURS", 24.0)),
            max_entries=max(1, env_int("BOT_RESEARCH_CACHE_MAX_ENTRIES", 500)),
        )


class ResearchCache:
    def __init__(self, config: ResearchCacheConfig | None = None) -> None:
        self._config = config or ResearchCacheConfig.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled)

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        now = time.time()
        with _LOCK:
            payload = self._read_payload()
            entry = payload.get("entries", {}).get(key)
            if not isinstance(entry, dict):
                return None
            created_at = _as_float(entry.get("created_at"))
            research = entry.get("research")
            if not isinstance(research, str) or not research.strip():
                return None
            if self._is_expired(created_at, now):
                return None
            entry["last_accessed_at"] = now
            entry["hits"] = int(entry.get("hits") or 0) + 1
            self._write_payload(payload)
            return research

    def set(self, key: str, research: str) -> None:
        if not self.enabled:
            return
        research = (research or "").strip()
        if not research:
            return
        now = time.time()
        with _LOCK:
            payload = self._read_payload()
            entries = payload.setdefault("entries", {})
            if not isinstance(entries, dict):
                entries = {}
                payload["entries"] = entries
            entries[key] = {
                "created_at": now,
                "last_accessed_at": now,
                "hits": 0,
                "research": research,
            }
            self._prune(payload, now)
            self._write_payload(payload)

    def _is_expired(self, created_at: float | None, now: float) -> bool:
        if created_at is None:
            return True
        ttl = self._config.ttl_hours
        if ttl <= 0:
            return True
        return (now - created_at) > ttl * 3600.0

    def _read_payload(self) -> dict[str, Any]:
        path = self._config.path
        if not path.exists():
            return {"version": _CACHE_VERSION, "entries": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Research cache read failed; ignoring cache. Error: %s", e)
            return {"version": _CACHE_VERSION, "entries": {}}
        if not isinstance(data, dict):
            return {"version": _CACHE_VERSION, "entries": {}}
        if data.get("version") != _CACHE_VERSION:
            return {"version": _CACHE_VERSION, "entries": {}}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            data["entries"] = {}
        return data

    def _write_payload(self, payload: dict[str, Any]) -> None:
        path = self._config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            Path(tmp_name).replace(path)
        except Exception:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _prune(self, payload: dict[str, Any], now: float) -> None:
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            payload["entries"] = {}
            return
        for key, entry in list(entries.items()):
            if not isinstance(entry, dict) or self._is_expired(
                _as_float(entry.get("created_at")), now
            ):
                entries.pop(key, None)
        if len(entries) <= self._config.max_entries:
            return
        ordered = sorted(
            entries.items(),
            key=lambda item: _as_float(item[1].get("last_accessed_at"))
            if isinstance(item[1], dict)
            else 0.0,
        )
        for key, _entry in ordered[: max(0, len(entries) - self._config.max_entries)]:
            entries.pop(key, None)


def build_research_cache_key(
    *,
    question: object,
    researcher_name: str,
    options: dict[str, Any] | None = None,
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "question": {
            "page_url": getattr(question, "page_url", None),
            "question_text": getattr(question, "question_text", None),
            "background_info": getattr(question, "background_info", None),
            "resolution_criteria": getattr(question, "resolution_criteria", None),
            "fine_print": getattr(question, "fine_print", None),
            "close_time": _stable_time(getattr(question, "close_time", None)),
            "scheduled_resolution_time": _stable_time(
                getattr(question, "scheduled_resolution_time", None)
            ),
        },
        "researcher_name": researcher_name,
        "options": options or {},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def research_cache_options_from_env() -> dict[str, Any]:
    names = [
        "BOT_ENABLE_LOCAL_QUESTION_CRAWL",
        "BOT_LOCAL_CRAWL_INCLUDE_QUESTION_PAGE",
        "BOT_ENABLE_TOOL_ROUTER",
        "SMART_SEARCHER_NUM_SEARCHES",
        "SMART_SEARCHER_NUM_SITES_PER_SEARCH",
        "SMART_SEARCHER_USE_ADVANCED_FILTERS",
        "TAVILY_SEARCH_DEPTH",
        "TAVILY_TOPIC",
        "TAVILY_TIME_RANGE",
        "TAVILY_INCLUDE_RAW_CONTENT",
        "TAVILY_EXTRACT_MISSING_CONTENT",
        "TAVILY_EXTRACT_MIN_CONTENT_CHARS",
        "TAVILY_EXTRACT_MAX_URLS",
        "TAVILY_EXTRACT_TIMEOUT_SECONDS",
        "BOT_OFFICIAL_TOTAL_CHAR_BUDGET",
    ]
    return {name: os.getenv(name, "") for name in names}


def _stable_time(value: object) -> str | None:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return None


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
