"""Per-provider rate limiting, shared across the whole process.

Why this exists
---------------
The E2E run (workflow 32295883751) fired ~185 LLM calls in 18 seconds, peaking
at 30 in a single second, and produced zero forecasts. The providers answered
with exactly what they meter on:

    Gemini free: quotaId "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                 quotaValue 15          -> 15 REQUESTS per minute, per model
    Groq free:   "tokens per minute (TPM): Limit 8000, Used 5951,
                  Requested 2396"       -> 8000 TOKENS per minute

Two different meters. A semaphore fixes neither: it caps how many calls are in
flight at once, not how many happen per minute. Ten calls that each take 200ms
pass a semaphore of 2 easily and still blow a 15/minute quota.

Where it must live
------------------
`ForecastBot.get_llm(purpose)` returns one instance per ROLE, and with the
fallback active there are four of them (default, summarizer, researcher,
parser). Each holds its OWN GeneralLlm pointing at Gemini. A limiter stored on
the instance would therefore become four independent limiters allowing 60
req/min against a 15 req/min quota - the control would look present and do
nothing.

So limiters live in a process-global registry keyed by MODEL string. All four
roles, every question and every prediction share one limiter per model, which
is also the granularity Gemini actually meters ("PerProjectPerModel").

What it deliberately does not do
--------------------------------
It delays calls. It never drops one, never shortens a prediction, and never
touches what a forecast is. A call that waits is still exactly one call; a
prediction that waits is still exactly one prediction.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


class RateLimitTimeout(RuntimeError):
    """The wait for a slot exceeded the configured ceiling.

    Raised so FallbackLlm treats a saturated provider like any other provider
    failure and moves to the next link. A jammed Gemini should yield to Groq
    rather than hold a prediction hostage.
    """


@dataclass
class ProviderLimits:
    """Known quota for one model.

    Defaults are transcribed from what the providers actually returned during
    the E2E run, not from documentation. The two quota values,
    requests_per_minute and tokens_per_minute, are overridable by env var (see
    _env_override) so a tier change does not need a code change;
    estimated_tokens_per_call and max_wait_seconds are not.
    """

    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    # Cannot be known before the call. Groq's own 429 reported "Requested 2396"
    # for a real forecasting prompt, so 2500 is a slightly conservative default.
    estimated_tokens_per_call: int = 2500
    max_wait_seconds: float = 300.0


# Keyed by the model string as it appears in the llms= block, which is exactly
# the granularity Gemini meters on.
DEFAULT_LIMITS: dict[str, ProviderLimits] = {
    "gemini/gemini-3.5-flash-lite": ProviderLimits(
        requests_per_minute=15.0,          # measured: quotaValue "15"
        tokens_per_minute=None,            # not the binding meter for Gemini
    ),
    "groq/openai/gpt-oss-120b": ProviderLimits(
        requests_per_minute=None,          # Groq metered us on tokens, not requests
        tokens_per_minute=8000.0,          # measured: "TPM: Limit 8000"
    ),
    # Parser fallback, added after a catalogue sweep found it is the one Groq
    # model that emits schema-constrained JSON (run 34409492553).
    #
    # The 8000 TPM here is INHERITED from gpt-oss-120b above, not measured for
    # this model: Groq meters per model, so the real ceiling may differ. It is
    # registered rather than left out because limits_for() hands an unknown key
    # an UNTHROTTLED limiter -- a missing entry would look controlled and
    # enforce nothing. Inheriting a measured-conservative number errs toward
    # waiting, which for the parser costs latency; the alternative errs toward
    # 429s, which costs the prediction. Re-measure if Groq starts 429ing here.
    "groq/qwen/qwen3.8-27b": ProviderLimits(
        requests_per_minute=None,
        tokens_per_minute=8000.0,
    ),
    # OpenRouter's observed cap is per DAY (50), never per minute: 63 logs and
    # not one per-minute error. Left unlimited so the current production path
    # is unchanged - the point of this module is to stop bursts against
    # per-minute meters, not to slow down what already works.
    "openrouter/nvidia/nemotron-3.5-lightning:free": ProviderLimits(),
    "openrouter/openai/gpt-4o-mini": ProviderLimits(),
    # Sponsored-key models. Left unthrottled like the other OpenRouter entries:
    # the observed cap there is per DAY and per credit balance, not per minute,
    # so a per-minute limiter would only add latency without preventing the
    # failure mode that actually occurs. Registered rather than omitted because
    # limits_for() hands an UNKNOWN key an unthrottled limiter too -- and an
    # unregistered key looks controlled while enforcing nothing.
    "openrouter/anthropic/claude-opus-4.6": ProviderLimits(),
    # The forecaster since 2026-10-02. Opus 4.6 stays registered above so a
    # models.txt override back to it keeps a registered bucket key.
    "openrouter/anthropic/claude-opus-5.5": ProviderLimits(),
    "openrouter/anthropic/claude-haiku-4.5": ProviderLimits(),
    # Second-vendor forecast voter (pin_models.FORECAST_VOTERS), 2026-10-18.
    "openrouter/openai/gpt-5.6-sol": ProviderLimits(),
    # The researcher's first link when an AskNews credential is present (see
    # pin_models.ACTIVE_ASKNEWS). Not an LLM: forecasting_tools' GeneralLlm
    # routes this model string to AskNewsSearcher, which paces its own two
    # news calls with a 12s sleep for AskNews' free tier, and main.py runs
    # research one question at a time. Registered unthrottled for the same
    # reason as the OpenRouter entries: an unknown key looks controlled and
    # enforces nothing, and the registry is pinned against
    # pin_models.RATE_LIMITED_MODELS.
    "asknews/news-summaries": ProviderLimits(),
}

_ENV_PREFIX = "LLM_RATE_"


def _env_override(model: str, field_name: str) -> float | None:
    """e.g. LLM_RATE_GEMINI_GEMINI_3_5_FLASH_LITE_RPM=10"""
    key = _ENV_PREFIX + "".join(
        c.upper() if c.isalnum() else "_" for c in model
    ) + "_" + field_name
    raw = os.environ.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", key, raw)
        return None


def limits_for(model: str) -> ProviderLimits:
    base = DEFAULT_LIMITS.get(model, ProviderLimits())
    rpm = _env_override(model, "RPM")
    tpm = _env_override(model, "TPM")
    return ProviderLimits(
        requests_per_minute=rpm if rpm is not None else base.requests_per_minute,
        tokens_per_minute=tpm if tpm is not None else base.tokens_per_minute,
        estimated_tokens_per_call=base.estimated_tokens_per_call,
        max_wait_seconds=base.max_wait_seconds,
    )


@dataclass
class InfraCounters:
    """INFRASTRUCTURE observability only.

    Deliberately separate from anything that counts forecasts. Nothing here is
    ever read by the metric: n_questions_with_own_forecast is derived from
    Metaculus' own my_forecasts records and knows nothing about this module.
    """

    llm_attempts_total: int = 0
    llm_success_total: int = 0
    llm_fallback_total: int = 0
    rate_limit_waits_total: int = 0
    rate_limit_wait_seconds_total: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_attempts_total": self.llm_attempts_total,
            "llm_success_total": self.llm_success_total,
            "llm_fallback_total": self.llm_fallback_total,
            "rate_limit_waits_total": self.rate_limit_waits_total,
            "rate_limit_wait_seconds_total": round(self.rate_limit_wait_seconds_total, 2),
        }

    def reset(self) -> None:
        self.llm_attempts_total = 0
        self.llm_success_total = 0
        self.llm_fallback_total = 0
        self.rate_limit_waits_total = 0
        self.rate_limit_wait_seconds_total = 0.0


COUNTERS = InfraCounters()


class ProviderRateLimiter:
    """Sliding-window limiter over both requests and tokens.

    Admission is serialised through one lock, which makes the queue FIFO: calls
    are admitted in the order they arrived, so a prediction cannot starve
    behind newer ones. Serialising the *decision* costs nothing, because the
    thing being rationed is slower than the decision by orders of magnitude.

    The clock and sleep are injectable so tests can prove the pacing
    deterministically instead of actually sleeping for minutes.
    """

    def __init__(
        self,
        model: str,
        limits: ProviderLimits | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.model = model
        self.limits = limits or limits_for(model)
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        # The lock is bound lazily, per running event loop. main.py calls
        # asyncio.run() twice in its `run_mode == "tournament"` branch - once
        # for the tournament, once for MiniBench - so a lock created under the
        # first loop would raise "got Future attached to a different loop" for
        # every call in the second, wiping out MiniBench entirely. The
        # sliding-window state below deliberately does NOT reset with the
        # loop: the provider's quota does not care that our process started a
        # new loop.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            self._tokens.popleft()

    def _tokens_in_window(self) -> int:
        return sum(count for _t, count in self._tokens)

    def _wait_needed(self, now: float, tokens: int) -> float:
        """Seconds until a slot frees. 0.0 means admit now."""
        waits = [0.0]
        rpm = self.limits.requests_per_minute
        if rpm is not None and len(self._requests) >= rpm:
            waits.append(self._requests[0] + WINDOW_SECONDS - now)
        tpm = self.limits.tokens_per_minute
        if tpm is not None and self._tokens_in_window() + tokens > tpm:
            # Release oldest token records until the request fits.
            freed = 0
            needed = self._tokens_in_window() + tokens - tpm
            for timestamp, count in self._tokens:
                freed += count
                if freed >= needed:
                    waits.append(timestamp + WINDOW_SECONDS - now)
                    break
            else:
                # Even an empty window cannot fit this request: the estimate
                # exceeds the whole per-minute budget. Admitting it is the only
                # way it can ever run; the provider will reject it if truly
                # oversized, and that is a provider failure, not a lost call.
                waits.append(0.0)
        return max(waits)

    async def acquire(self, tokens: int | None = None) -> float:
        """Wait until a slot is free. Returns the seconds waited.

        Never returns without a slot, and never silently drops the call: the
        only non-return path is RateLimitTimeout, which the caller turns into a
        provider failover.
        """
        tokens = tokens if tokens is not None else self.limits.estimated_tokens_per_call
        started = self._clock()
        slept = False
        async with self._get_lock():
            while True:
                now = self._clock()
                self._prune(now)
                wait = self._wait_needed(now, tokens)
                if wait <= 0:
                    self._requests.append(now)
                    self._tokens.append((now, tokens))
                    waited = now - started
                    # Counted only when we actually slept. With a real clock,
                    # an unimpeded call still shows a microsecond of elapsed
                    # time, and counting that would report phantom waits.
                    if slept:
                        COUNTERS.rate_limit_waits_total += 1
                        COUNTERS.rate_limit_wait_seconds_total += waited
                    return waited
                if (now - started) + wait > self.limits.max_wait_seconds:
                    raise RateLimitTimeout(
                        "{0}: slot would take {1:.1f}s, over the {2:.0f}s ceiling".format(
                            self.model, (now - started) + wait, self.limits.max_wait_seconds
                        )
                    )
                slept = True
                await self._sleep(wait)


_REGISTRY: dict[str, ProviderRateLimiter] = {}


def get_limiter(model: str) -> ProviderRateLimiter:
    """The one limiter for this model in this process.

    Keyed by model rather than by object, which is the whole point: the four
    role-level FallbackLlm instances each hold a separate GeneralLlm for the
    same Gemini model, and they must contend for one quota, not four.
    """
    limiter = _REGISTRY.get(model)
    if limiter is None:
        limiter = ProviderRateLimiter(model)
        _REGISTRY[model] = limiter
        logger.info(
            "rate_limiter_created model=%s rpm=%s tpm=%s",
            model, limiter.limits.requests_per_minute, limiter.limits.tokens_per_minute,
        )
    return limiter


def reset_registry() -> None:
    """Test-only: drop every limiter so cases cannot leak state into each other."""
    _REGISTRY.clear()
    COUNTERS.reset()


def registry_size() -> int:
    return len(_REGISTRY)
