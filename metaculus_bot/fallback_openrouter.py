import logging
import os
from typing import Any

from forecasting_tools import GeneralLlm

logger: logging.Logger = logging.getLogger(__name__)


def is_openrouter_openai_or_anthropic(model: str) -> bool:
    if not isinstance(model, str):
        return False
    if not model.startswith("openrouter/"):
        return False
    parts = model.split("/")
    if len(parts) < 2:
        return False
    provider = parts[1]
    return provider in {"openai", "anthropic"}


def should_retry_with_general_key(exc: Exception) -> bool:
    """
    Decide whether a failure likely indicates a credit/credential issue where falling back is appropriate.

    Triggers fallback on:
    - 401 Unauthorized (invalid/disabled key),
    - 402 Payment Required (insufficient credits),
    - Common text cues for these scenarios.

    Avoids fallback on:
    - 403 Forbidden (moderation/blocked),
    - 429 Too Many Requests (rate limit),
    - 502/503 upstream/provider outages.
    """
    msg = str(exc).lower()
    # Positive signals: credentials/credits
    if "401" in msg or "unauthorized" in msg or "invalid api key" in msg or "disabled api key" in msg:
        return True
    if (
        "402" in msg
        or "payment required" in msg
        or "insufficient credit" in msg
        or "out of credits" in msg
        or "insufficient funds" in msg
    ):
        return True

    # Negative signals: do not swap keys for these
    if "403" in msg or "forbidden" in msg or "moderation" in msg:
        return False
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return False
    if "502" in msg or "bad gateway" in msg:
        return False
    if "503" in msg or "service unavailable" in msg:
        return False

    # Default: be conservative and do not fallback when unsure
    return False


class FallbackOpenRouterLlm(GeneralLlm):
    """
    A GeneralLlm that attempts a call with a primary OpenRouter key and falls back to a secondary key
    on credential/credit failures. Only intended for OpenRouter OpenAI/Anthropic models.
    """

    def __init__(
        self,
        *,
        model: str,
        primary_api_key: str | None,
        secondary_api_key: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=primary_api_key, **kwargs)
        self._secondary_llm: GeneralLlm | None = (
            GeneralLlm(model=model, api_key=secondary_api_key, **kwargs) if secondary_api_key else None
        )
        self._has_warned_once: bool = False

    async def invoke(self, prompt: Any) -> str:  # type: ignore[override]
        try:
            return await self._invoke_once_using_primary(prompt)
        except Exception as e:
            if self._secondary_llm is not None and should_retry_with_general_key(e):
                if not self._has_warned_once:
                    logger.warning(
                        "Primary OpenRouter key failed with credential/credit error; falling back to generic key. "
                        f"model={self.model}; error={type(e).__name__}: {e}"
                    )
                    self._has_warned_once = True
                return await self._invoke_once_using_secondary(prompt)
            raise

    async def _invoke_once_using_primary(self, prompt: Any) -> str:
        return await super().invoke(prompt)

    async def _invoke_once_using_secondary(self, prompt: Any) -> str:
        if self._secondary_llm is None:
            raise RuntimeError("No secondary key configured for fallback")
        return await self._secondary_llm.invoke(prompt)


def build_llm_with_openrouter_fallback(model: str, **kwargs: Any) -> GeneralLlm:
    """
    Construct a GeneralLlm that automatically falls back from Metaculus OpenRouter key to generic key
    for OpenAI/Anthropic providers on OpenRouter. For other models, returns a plain GeneralLlm.
    """
    if is_openrouter_openai_or_anthropic(model):
        special_key = os.getenv("OAI_ANTH_OPENROUTER_KEY")
        general_key = os.getenv("OPENROUTER_API_KEY")

        # If both keys exist and are distinct, use the fallback wrapper
        if special_key and general_key and special_key != general_key:
            return FallbackOpenRouterLlm(
                model=model,
                primary_api_key=special_key,
                secondary_api_key=general_key,
                **kwargs,
            )

        # Else fall back to whichever key is available (no runtime fallback possible)
        api_key = special_key or general_key
        return GeneralLlm(model=model, api_key=api_key, **kwargs)

    # Non-OpenAI/Anthropic OpenRouter models: plain GeneralLlm
    return GeneralLlm(model=model, **kwargs)
