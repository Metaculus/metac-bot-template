from __future__ import annotations

"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from forecasting_tools import GeneralLlm

__all__ = ["FORECASTER_LLMS", "SUMMARIZER_LLM", "PARSER_LLM", "RESEARCHER_LLM"]

FORECASTER_LLMS = [
    # TODO: expand suite of LLMs. Likely add Grok4, maybe GLM-4.5, probably not Qwen3-225B
    GeneralLlm(
        model="openrouter/openai/gpt-5",
        temperature=0.0,
        top_p=0.9,
        max_tokens=16000,  # Prevent truncation issues with reasoning models
        reasoning_effort="high",
        stream=False,
        timeout=180,
        allowed_tries=3,
    ),
    GeneralLlm(
        model="openrouter/anthropic/claude-sonnet-4",
        reasoning={"max_tokens": 8000},
        temperature=0.0,
        top_p=0.9,
        max_tokens=16000,  # Prevent truncation issues with reasoning models
        stream=False,
        timeout=180,
        allowed_tries=3,
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
        temperature=0.0,
        top_p=0.9,
        max_tokens=16000,  # Prevent truncation issues with reasoning models
        stream=False,
        timeout=180,
        allowed_tries=3,
        provider={"quantizations": ["fp16", "bf16", "fp8"]},
    ),
]

SUMMARIZER_LLM: str = "openrouter/google/gemini-2.5-flash"

# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: str = "openrouter/google/gemini-2.5-flash"

# Researcher is only used by the base bot when internal research is invoked.
# Our implementation uses providers, but we still set it explicitly to avoid silent defaults.
RESEARCHER_LLM: str = "openrouter/openai/gpt-5"
