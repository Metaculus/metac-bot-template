"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

import os
from typing import Tuple

# Load .env early so ASKNEWS_* values are read correctly at import time in local runs
try:  # pragma: no cover - best-effort convenience for local dev
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(".env.local", override=True)
except Exception:
    pass

# Concurrency tuning for research providers (e.g., AskNews, Exa)
# Start conservatively for AskNews; adjust after observing rate limits.
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 1

# Benchmark driver settings
# Default batch size for benchmarking runs
# Keep this modest to balance concurrency and rate limits.
BENCHMARK_BATCH_SIZE: int = 4

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

# --- Forecasting clamps and numeric smoothing ---
# Binary prediction clamp
BINARY_PROB_MIN: float = 0.01
BINARY_PROB_MAX: float = 0.99

# Multiple-choice prediction clamp
MC_PROB_MIN: float = 0.005
MC_PROB_MAX: float = 0.995

# Numeric CDF smoothing and spacing
NUM_VALUE_EPSILON_MULT: float = 1e-9
NUM_SPREAD_DELTA_MULT: float = 1e-6
NUM_MIN_PROB_STEP: float = 5e-5
NUM_MAX_STEP: float = 0.59
NUM_RAMP_K_FACTOR: float = 3.0

# --- Benchmark driver tuning ---
HEARTBEAT_INTERVAL: int = 60
FETCH_RETRY_BACKOFFS: list[int] = [5, 15]
# Distribution mix: (binary, numeric, multiple_choice)
TYPE_MIX: Tuple[float, float, float] = (0.5, 0.25, 0.25)
FETCH_PACING_SECONDS: int = 2
