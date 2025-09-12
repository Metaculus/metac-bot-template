"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = ["FORECASTER_LLMS", "SUMMARIZER_LLM", "PARSER_LLM", "RESEARCHER_LLM"]
MODEL_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.85,  # Does nothing given temp == 0
    "max_tokens": 16_000,  # Prevent truncation issues with reasoning models
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}

FORECASTER_LLMS = [
    # TODO: consider adding add'l LLMs to ensemble
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5",
        reasoning={"effort": "high"},
        **MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/o3",
        reasoning={"effort": "high"},
        **MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-sonnet-4",
        reasoning={"max_tokens": 8_000},
        **MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        provider={
            "quantizations": [
                "fp8",
                "fp16",
                "bf16",
                "fp32",
                "unknown",
            ]
        },
        **MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/moonshotai/kimi-k2-0905",
        provider={
            "quantizations": [
                "fp8",
                "fp16",
                "bf16",
                "fp32",
                "unknown",
            ]
        },
        **MODEL_CONFIG,
    ),
]

SUMMARIZER_LLM: str = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5-mini")
# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: str = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5-mini")
# Researcher is only used by the base bot when internal research is invoked.
# Our implementation uses providers, but we still set it explicitly to avoid silent defaults.
RESEARCHER_LLM = build_llm_with_openrouter_fallback(
    model="openrouter/openai/gpt-5",
)
