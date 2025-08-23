from __future__ import annotations

"""Utility functions for API key management."""

import os


def get_openrouter_api_key(model: str) -> str | None:
    """
    Determine the correct OpenRouter API key based on the model provider.

    Uses special Metaculus-provided credits for Anthropic and OpenAI models
    via OAI_ANTH_OPENROUTER_KEY, falls back to general OPENROUTER_API_KEY.

    Args:
        model: The model name (e.g., "openrouter/anthropic/claude-sonnet-4")

    Returns:
        The appropriate API key or None if no key is available
    """
    # Check if this is an OpenRouter model for Anthropic or OpenAI
    if model.startswith("openrouter/"):
        provider = model.split("/")[1] if "/" in model else ""

        # Use special key for Anthropic and OpenAI models
        if provider in ("anthropic", "openai"):
            special_key = os.getenv("OAI_ANTH_OPENROUTER_KEY")
            if special_key:
                return special_key

    # Fall back to general OpenRouter key
    return os.getenv("OPENROUTER_API_KEY")
