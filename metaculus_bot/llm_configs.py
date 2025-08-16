from __future__ import annotations

"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from forecasting_tools import GeneralLlm

from metaculus_bot.api_key_utils import get_openrouter_api_key

__all__ = ["FORECASTER_LLMS", "SUMMARIZER_LLM", "PARSER_LLM", "RESEARCHER_LLM"]
MODEL_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 16_000,  # Prevent truncation issues with reasoning models
    "stream": False,
    "timeout": 240,
    "allowed_tries": 3,
}

FORECASTER_LLMS = [
    # TODO: expand suite of LLMs. Likely add Grok4, maybe GLM-4.5, probably not Qwen3-225B
    GeneralLlm(
        model="openrouter/openai/gpt-5",
        api_key=get_openrouter_api_key("openrouter/openai/gpt-5"),
        **MODEL_CONFIG,
    ),
    GeneralLlm(
        model="openrouter/anthropic/claude-sonnet-4",
        api_key=get_openrouter_api_key("openrouter/anthropic/claude-sonnet-4"),
        **MODEL_CONFIG,
    ),
    # GeneralLlm(
    #     model="openrouter/google/gemini-2.5-pro",
    #     temperature=0.0,
    #     top_p=0.9,
    #     max_tokens=16000,  # Prevent truncation issues with reasoning models
    #     reasoning={"max_tokens": 8000},
    #     stream=False,
    #     timeout=180,
    #     allowed_tries=3,
    # ),
    GeneralLlm(
        model="openrouter/deepseek/deepseek-r1-0528",
        provider={"quantizations": ["fp16", "bf16", "fp8"]},
        **MODEL_CONFIG,
    ),
]

SUMMARIZER_LLM: str = "openrouter/google/gemini-2.5-flash"

# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: str = "openrouter/google/gemini-2.5-flash"

# Researcher is only used by the base bot when internal research is invoked.
# Our implementation uses providers, but we still set it explicitly to avoid silent defaults.
RESEARCHER_LLM = GeneralLlm(
    model="openrouter/openai/gpt-5",
    api_key=get_openrouter_api_key("openrouter/openai/gpt-5"),
)
