from __future__ import annotations

"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

import os

# Concurrency tuning for research providers (e.g., AskNews, Exa)
# Start conservatively for AskNews; adjust after observing rate limits.
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 1

# Benchmark driver settings
# Keep the benchmark question batch size aligned with the bot concurrency to
# avoid oversubscription and rate-limit spikes.
BENCHMARK_BATCH_SIZE: int = DEFAULT_MAX_CONCURRENT_RESEARCH

# Optional environment variable to force research provider selection.
# Accepted values (case-insensitive): "auto", "asknews", "exa", "perplexity", "openrouter"
RESEARCH_PROVIDER_ENV: str = "RESEARCH_PROVIDER"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


# AskNews provider safety limits (global, across all bots in-process)
# Defaults are conservative for pro plans (1 RPS sustained, 5 RPS burst, 5 concurrency)
ASKNEWS_MAX_CONCURRENCY: int = max(1, _int_env("ASKNEWS_MAX_CONCURRENCY", 1))
# Conservative sustained rate well below pro plan limits (1 RPS sustained)
ASKNEWS_MAX_RPS: float = max(0.1, _float_env("ASKNEWS_MAX_RPS", 0.8))

# Retry tuning for AskNews
ASKNEWS_MAX_TRIES: int = max(1, _int_env("ASKNEWS_MAX_TRIES", 3))
ASKNEWS_BACKOFF_SECS: float = max(0.0, _float_env("ASKNEWS_BACKOFF_SECS", 2.0))
