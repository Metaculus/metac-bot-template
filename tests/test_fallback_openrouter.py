import types

import pytest

from metaculus_bot.fallback_openrouter import (
    FallbackOpenRouterLlm,
    build_llm_with_openrouter_fallback,
    is_openrouter_openai_or_anthropic,
    should_retry_with_general_key,
)


class TestPredicates:
    def test_is_openrouter_openai_or_anthropic(self) -> None:
        assert is_openrouter_openai_or_anthropic("openrouter/openai/gpt-5") is True
        assert is_openrouter_openai_or_anthropic("openrouter/anthropic/claude-sonnet-4") is True
        assert is_openrouter_openai_or_anthropic("openrouter/google/gemini-2.5-pro") is False
        assert is_openrouter_openai_or_anthropic("perplexity/sonar") is False

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("HTTP 402 Payment Required", True),
            ("payment required", True),
            ("insufficient credit on key", True),
            ("401 Unauthorized", True),
            ("invalid API key", True),
            ("disabled api key", True),
            ("403 Forbidden moderation", False),
            ("429 Too Many Requests", False),
            ("Rate limit exceeded", False),
            ("502 Bad Gateway", False),
            ("503 Service Unavailable", False),
        ],
    )
    def test_should_retry_with_general_key(self, message: str, expected: bool) -> None:
        assert should_retry_with_general_key(Exception(message)) is expected


class TestFallbackOpenRouterLlm:
    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        async def fake_primary(self, prompt):  # type: ignore[no-untyped-def]
            return "answer"

        # Patch the internal primary call point to avoid network
        llm._invoke_once_using_primary = types.MethodType(fake_primary, llm)  # type: ignore[method-assign]

        out = await llm.invoke("hi")
        assert out == "answer"

    @pytest.mark.asyncio
    async def test_fallback_on_402(self) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-sonnet-4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        async def fail_primary(self, prompt):  # type: ignore[no-untyped-def]
            raise Exception("HTTP 402 Payment Required: insufficient credit")

        async def succeed_secondary(self, prompt):  # type: ignore[no-untyped-def]
            return "ok"

        llm._invoke_once_using_primary = types.MethodType(fail_primary, llm)  # type: ignore[method-assign]
        llm._invoke_once_using_secondary = types.MethodType(succeed_secondary, llm)  # type: ignore[method-assign]

        out = await llm.invoke("hi")
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_no_fallback_on_403(self) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        async def fail_primary(self, prompt):  # type: ignore[no-untyped-def]
            raise Exception("403 Forbidden moderation")

        llm._invoke_once_using_primary = types.MethodType(fail_primary, llm)  # type: ignore[method-assign]

        with pytest.raises(Exception):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_secondary_key_configured(self) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5",
            primary_api_key="special",
            secondary_api_key=None,
            temperature=0,
        )

        async def fail_primary(self, prompt):  # type: ignore[no-untyped-def]
            raise Exception("401 Unauthorized")

        llm._invoke_once_using_primary = types.MethodType(fail_primary, llm)  # type: ignore[method-assign]

        with pytest.raises(Exception):
            await llm.invoke("hi")


class TestBuilder:
    def test_builder_returns_wrapper_when_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_when_only_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_non_openai_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-2.5-pro")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)
