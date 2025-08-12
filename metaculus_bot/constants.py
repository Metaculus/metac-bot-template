from __future__ import annotations

"""Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

# Concurrency tuning for research providers (e.g., AskNews, Exa)
# Start conservatively for AskNews; adjust after observing rate limits.
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 8

# Benchmark driver settings
# Keep the benchmark question batch size aligned with the bot concurrency to
# avoid oversubscription and rate-limit spikes.
BENCHMARK_BATCH_SIZE: int = DEFAULT_MAX_CONCURRENT_RESEARCH

# Optional environment variable to force research provider selection.
# Accepted values (case-insensitive): "auto", "asknews", "exa", "perplexity", "openrouter"
RESEARCH_PROVIDER_ENV: str = "RESEARCH_PROVIDER"
