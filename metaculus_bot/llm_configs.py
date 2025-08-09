from __future__ import annotations

"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from forecasting_tools import GeneralLlm

__all__ = ["FORECASTER_LLMS", "SUMMARIZER_LLM"]

FORECASTER_LLMS = [
    # GeneralLlm(
    #     model="openrouter/google/gemini-2.5-pro",
    #     temperature=0.0,
    #     top_p=0.9,
    #     stream=False,
    #     timeout=180,
    #     allowed_tries=3,
    # ),
    GeneralLlm(
        model="openrouter/deepseek/deepseek-r1-0528",
        temperature=0.0,
        top_p=0.9,
        stream=False,
        timeout=180,
        allowed_tries=3,
        provider={"quantizations": ["fp16", "bf16", "fp8"]},
    ),
    GeneralLlm(
        model="openrouter/openai/gpt-5",
        temperature=0.0,
        top_p=0.9,
        reasoning_effort="medium",
        stream=False,
        timeout=180,
        allowed_tries=3,
    ),
]

SUMMARIZER_LLM: str = "openrouter/google/gemini-2.5-flash" 