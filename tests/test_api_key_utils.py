import os

import pytest

from metaculus_bot.api_key_utils import get_openrouter_api_key


class TestApiKeyUtils:
    def test_openai_model_uses_special_key(self, monkeypatch):
        """OpenAI models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5")
        assert result == "special_key"

    def test_anthropic_model_uses_special_key(self, monkeypatch):
        """Anthropic models should use OAI_ANTH_OPENROUTER_KEY when available."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/anthropic/claude-sonnet-4")
        assert result == "special_key"

    def test_other_openrouter_model_uses_general_key(self, monkeypatch):
        """Non-OpenAI/Anthropic OpenRouter models should use general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/google/gemini-2.5-pro")
        assert result == "general_key"

    def test_fallback_to_general_key_when_special_missing(self, monkeypatch):
        """Should fall back to general key when special key not available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("openrouter/openai/gpt-5")
        assert result == "general_key"

    def test_non_openrouter_model_uses_general_key(self, monkeypatch):
        """Non-OpenRouter models should use general key."""
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special_key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general_key")

        result = get_openrouter_api_key("perplexity/sonar-reasoning-pro")
        assert result == "general_key"

    def test_returns_none_when_no_keys_available(self, monkeypatch):
        """Should return None when no API keys are available."""
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = get_openrouter_api_key("openrouter/openai/gpt-5")
        assert result is None
