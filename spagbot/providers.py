# spagbot/providers.py
from __future__ import annotations
"""
providers.py — unified access to OpenRouter, direct Gemini, and direct Grok.

- HONORS .env flags:
    USE_OPENROUTER_DEFAULT, USE_ANTHROPIC, USE_GEMINI, USE_GROK
- KNOWN_MODELS gives stable names used by CSV columns & weighting.
- DEFAULT_ENSEMBLE lists only active models this run.
- call_chat_ms(ModelSpec, prompt, temperature) → (text, usage_dict, error)
- estimate_cost_usd(...) uses MODEL_COSTS_JSON ($ per 1k tokens).
"""

import os, asyncio, json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# ---------------- stable names for CSV schema ----------------
KNOWN_MODELS = [
    "OpenRouter-Default",
    "Claude-3.7-Sonnet (OR)",
    "Gemini",
    "Grok",
]

@dataclass
class ModelSpec:
    name: str         # stable display name (must match KNOWN_MODELS)
    provider: str     # "openrouter" | "gemini" | "grok"
    model_id: str     # provider-specific model identifier
    weight: float = 1.0
    active: bool = True

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1","true","yes","y","on")

# ----------- provider toggles -----------
USE_OPENROUTER_DEFAULT = _env_bool("USE_OPENROUTER_DEFAULT", True)
USE_ANTHROPIC         = _env_bool("USE_ANTHROPIC", True)
USE_GEMINI            = _env_bool("USE_GEMINI", True)    # DIRECT Gemini
USE_GROK              = _env_bool("USE_GROK", True)      # DIRECT Grok

# ----------- OpenRouter (AsyncOpenAI-compatible) -----------
OR_API_KEY  = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OR_API_KEY", "")
OR_BASE_URL = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
OPENROUTER_FALLBACK_ID = os.getenv("OPENROUTER_FALLBACK_ID", "openai/gpt-4o")
CLAUDE_ON_OR_ID        = os.getenv("OPENROUTER_CLAUDE37_ID", "anthropic/claude-3.7-sonnet")

try:
    from openai import AsyncOpenAI  # works for OpenRouter when base_url is set
except Exception:
    AsyncOpenAI = None  # type: ignore

llm_semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY","4")))
_async_client_singleton = None

def _get_or_client():
    """Return AsyncOpenAI client (OpenRouter) or None."""
    global _async_client_singleton
    if not OR_API_KEY or AsyncOpenAI is None:
        return None
    if _async_client_singleton is None:
        _async_client_singleton = AsyncOpenAI(api_key=OR_API_KEY, base_url=OR_BASE_URL)
    return _async_client_singleton

# ----------- Gemini (DIRECT) -----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","").strip()
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID","gemini-2.5-pro").strip()

# ----------- Grok (DIRECT) -----------
XAI_API_KEY = os.getenv("XAI_API_KEY","").strip()
GROK_MODEL_ID = os.getenv("GROK_MODEL_ID","grok-4").strip()
XAI_BASE_URL = os.getenv("XAI_BASE_URL","https://api.x.ai/v1/chat/completions").strip()

# ----------- Build the potential ensemble -----------
_POTENTIAL_ENSEMBLE: List[ModelSpec] = [
    ModelSpec("OpenRouter-Default",     "openrouter", OPENROUTER_FALLBACK_ID, active=USE_OPENROUTER_DEFAULT),
    ModelSpec("Claude-3.7-Sonnet (OR)", "openrouter", CLAUDE_ON_OR_ID,        active=USE_ANTHROPIC),
    ModelSpec("Gemini",                 "gemini",     GEMINI_MODEL_ID,        active=USE_GEMINI),
    ModelSpec("Grok",                   "grok",       GROK_MODEL_ID,          active=USE_GROK),
]
DEFAULT_ENSEMBLE: List[ModelSpec] = [m for m in _POTENTIAL_ENSEMBLE if m.active]

# ---------------- pricing helpers ----------------
_MODEL_PRICES: Optional[Dict[str, Dict[str, float]]] = None
def _load_model_prices() -> Dict[str, Dict[str, float]]:
    global _MODEL_PRICES
    if _MODEL_PRICES is not None:
        return _MODEL_PRICES
    s = os.getenv("MODEL_COSTS_JSON","").strip()
    if s:
        try: _MODEL_PRICES = json.loads(s)
        except Exception: _MODEL_PRICES = {}
    else:
        _MODEL_PRICES = {}
    return _MODEL_PRICES

def usage_to_dict(usage_obj: Any) -> Dict[str,int]:
    """
    Map provider-specific usage to a common dict.
    For OpenRouter/OpenAI v1: usage has .prompt_tokens/.completion_tokens.
    """
    d = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
    try:
        if usage_obj is None:
            return d
        pt = getattr(usage_obj, "prompt_tokens", None)
        ct = getattr(usage_obj, "completion_tokens", None)
        tt = getattr(usage_obj, "total_tokens", None)
        d["prompt_tokens"] = int(pt or 0)
        d["completion_tokens"] = int(ct or 0)
        d["total_tokens"] = int(tt or (d["prompt_tokens"] + d["completion_tokens"]))
        return d
    except Exception:
        return d

def estimate_cost_usd(model_id: str, usage: Dict[str,int]) -> float:
    """
    usage: {"prompt_tokens": int, "completion_tokens": int}
    MODEL_COSTS_JSON: {"model_id":{"prompt":$/1k,"completion":$/1k}, ...}
    """
    prices = _load_model_prices()
    if not usage or not isinstance(usage, dict):
        return 0.0
    p = prices.get(model_id) or prices.get(model_id.split("/",1)[-1]) or {}
    try:
        rp = float(p.get("prompt", 0.0))
        rc = float(p.get("completion", 0.0))
        return (usage.get("prompt_tokens",0)/1000.0)*rp + (usage.get("completion_tokens",0)/1000.0)*rc
    except Exception:
        return 0.0

# ---------------- provider-specific calls ----------------
import requests

async def _call_openrouter(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    client = _get_or_client()
    if client is None:
        return "", {}, "no OpenRouter client"
    try:
        async with llm_semaphore:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
            )
        text = (resp.choices[0].message.content or "").strip()
        usage = usage_to_dict(getattr(resp, "usage", None))
        return text, usage, ""
    except Exception as e:
        return "", {}, f"{type(e).__name__}: {str(e)[:200]}"

async def _call_gemini_direct(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    if not GEMINI_API_KEY:
        return "", {}, "no GEMINI_API_KEY"
    def _do():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
        body = {
            "contents":[{"role":"user","parts":[{"text": prompt}]}],
            "generationConfig":{"temperature":float(temperature)}
        }
        try:
            r = requests.post(url, json=body, timeout=300)
            j = r.json()
        except Exception as e:
            return "", {}, f"Gemini request error: {e!r}"
        if r.status_code != 200:
            msg = j.get("error", {}).get("message","")
            return "", {}, f"Gemini HTTP {r.status_code}: {msg[:160]}"
        # Extract text
        text = ""
        try:
            text = j["candidates"][0]["content"]["parts"][0].get("text","").strip()
        except Exception:
            text = j.get("text","").strip()
        # usageMetadata: promptTokenCount / candidatesTokenCount / totalTokenCount
        um = j.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": int(um.get("promptTokenCount", 0)),
            "completion_tokens": int(um.get("candidatesTokenCount", 0)),
            "total_tokens": int(um.get("totalTokenCount", 0)),
        }
        return text, usage, ""
    return await asyncio.to_thread(_do)

async def _call_grok_direct(model_id: str, prompt: str, temperature: float) -> tuple[str, Dict[str,int], str]:
    if not XAI_API_KEY:
        return "", {}, "no XAI_API_KEY"
    def _do():
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type":"application/json"}
        body = {"model": model_id, "messages":[{"role":"user","content":prompt}], "temperature": float(temperature)}
        try:
            r = requests.post(XAI_BASE_URL, json=body, headers=headers, timeout=300)
            j = r.json()
        except Exception as e:
            return "", {}, f"Grok request error: {e!r}"
        if r.status_code != 200:
            msg = j.get("error", {}).get("message","")
            return "", {}, f"Grok HTTP {r.status_code}: {msg[:160]}"
        text = ""
        try:
            text = j["choices"][0]["message"].get("content","").strip()
        except Exception:
            text = ""
        u = j.get("usage", {}) or {}
        usage = {
            "prompt_tokens": int(u.get("prompt_tokens", 0)),
            "completion_tokens": int(u.get("completion_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", u.get("totalTokens", 0)) or (u.get("prompt_tokens",0)+u.get("completion_tokens",0))),
        }
        return text, usage, ""
    return await asyncio.to_thread(_do)

# -------------- public: one call to rule them all --------------
async def call_chat_ms(ms: ModelSpec, prompt: str, temperature: float = 0.2) -> tuple[str, Dict[str,int], str]:
    """
    Return (text, usage_dict, error_message).
    error_message non-empty means provider call failed quickly or was unauthorized.
    """
    if ms.provider == "openrouter":
        return await _call_openrouter(ms.model_id, prompt, temperature)
    if ms.provider == "gemini":
        return await _call_gemini_direct(ms.model_id, prompt, temperature)
    if ms.provider == "grok":
        return await _call_grok_direct(ms.model_id, prompt, temperature)
    return "", {}, f"unsupported provider {ms.provider}"

# -------- Gemini helper used by research.py fallback ----------
async def _call_google(prompt_text: str, model: str = None, timeout: float = 120.0, temperature: float = 0.3) -> str:
    """
    Lightweight wrapper for research composition. Uses DIRECT Gemini when enabled,
    otherwise returns "".
    """
    if not USE_GEMINI:
        return ""
    model_id = (model or GEMINI_MODEL_ID).strip()
    text, _, err = await _call_gemini_direct(model_id, prompt_text, temperature)
    return text if not err else ""
