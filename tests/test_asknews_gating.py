import types
from typing import Callable

import pytest


def _install_asknews_stub(monkeypatch: pytest.MonkeyPatch, on_search: Callable[[str], object]) -> None:
    """Install a minimal asknews_sdk stub that routes search calls to on_search.

    The stub emulates only what our provider needs: async context manager and
    a .news.search_news coroutine returning an object with .as_dicts.
    """

    module = types.ModuleType("asknews_sdk")

    class _News:
        async def search_news(self, *, query, n_articles, return_type, strategy):  # type: ignore[no-untyped-def]
            import inspect

            result = on_search(strategy)
            if inspect.isawaitable(result):
                result = await result  # type: ignore[assignment]
            return result

    class AsyncAskNewsSDK:  # noqa: N801 - match import name in provider
        def __init__(self, *_, **__):  # type: ignore[no-untyped-def]
            self.news = _News()

        async def __aenter__(self):  # pragma: no cover - trivial
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

    module.AsyncAskNewsSDK = AsyncAskNewsSDK
    monkeypatch.setitem(__import__("sys").modules, "asknews_sdk", module)


@pytest.mark.asyncio
async def test_asknews_rate_gate_runs_before_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure AskNews is selected and globals are reset
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setenv("RESEARCH_PROVIDER", "asknews")

    # Capture rate-gate invocations and internal sleeps
    gate_calls: list[None] = []
    sleep_durations: list[float] = []

    async def fake_gate() -> None:
        gate_calls.append(None)

    async def fake_sleep(secs: float) -> None:
        sleep_durations.append(secs)

    # Install stubs
    _install_asknews_stub(
        monkeypatch,
        on_search=lambda strategy: types.SimpleNamespace(as_dicts=[]),
    )
    import metaculus_bot.research_providers as rp

    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_LAST_CALL_TS", 0.0, raising=False)
    monkeypatch.setattr(rp, "_asknews_rate_gate", fake_gate, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", fake_sleep, raising=True)

    provider, name = rp.choose_provider_with_name()
    assert name == "asknews"
    await provider("Will X happen?")

    # Two calls -> two gates; and we keep an explicit inter-call sleep
    assert len(gate_calls) == 2
    assert any(d > 0 for d in sleep_durations)


@pytest.mark.asyncio
async def test_global_semaphore_serializes_concurrent_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")

    in_flight = 0
    max_in_flight = 0

    async def _record(_strategy: str) -> object:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield to allow other tasks to run
        import asyncio as _a

        await _a.sleep(0)  # cooperative yield only
        in_flight -= 1
        return types.SimpleNamespace(as_dicts=[])

    _install_asknews_stub(monkeypatch, on_search=_record)
    import metaculus_bot.research_providers as rp

    # Force concurrency 1 and neutralize gating/sleeps to keep the test fast
    monkeypatch.setattr(rp, "ASKNEWS_MAX_CONCURRENCY", 1, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)

    async def noop_gate() -> None:  # pragma: no cover - trivial
        return None

    async def noop_sleep(_secs: float) -> None:  # pragma: no cover - trivial
        return None

    monkeypatch.setattr(rp, "_asknews_rate_gate", noop_gate, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", noop_sleep, raising=True)

    provider, _ = rp.choose_provider_with_name(forced_provider := None)  # type: ignore[arg-type]
    # Launch multiple requests concurrently; each request does 2 sequential calls
    await __import__("asyncio").gather(*(provider(f"Q{i}") for i in range(4)))

    # Entire attempt is under the global semaphore -> at most 1 in-flight search
    assert max_in_flight == 1


@pytest.mark.asyncio
async def test_rps_gate_sleeps_before_historical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setenv("RESEARCH_PROVIDER", "asknews")

    # Return empty articles; only timing matters
    _install_asknews_stub(
        monkeypatch,
        on_search=lambda strategy: types.SimpleNamespace(as_dicts=[]),
    )
    import metaculus_bot.research_providers as rp

    # Use a low RPS so the second call must sleep
    monkeypatch.setattr(rp, "ASKNEWS_MAX_RPS", 0.5, raising=False)  # min interval = 2.0s
    monkeypatch.setattr(rp, "_ASKNEWS_GLOBAL_SEMAPHORE", None, raising=False)
    monkeypatch.setattr(rp, "_ASKNEWS_LAST_CALL_TS", 0.0, raising=False)

    # Controlled monotonic sequence: first gate no-sleep, second gate sleeps ~2s
    times = [100.0, 100.0, 102.0]

    def fake_monotonic() -> float:
        return times.pop(0) if times else 102.0

    sleep_calls: list[float] = []

    async def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    monkeypatch.setattr(rp.time, "monotonic", fake_monotonic, raising=True)
    monkeypatch.setattr(rp.asyncio, "sleep", fake_sleep, raising=True)

    provider, _ = rp.choose_provider_with_name()
    await provider("Test")

    # We expect both the explicit inter-call sleep and ~2.0s RPS sleep
    assert any(s > 0 for s in sleep_calls)
    assert any(abs(s - 2.0) < 1e-6 for s in sleep_calls)
