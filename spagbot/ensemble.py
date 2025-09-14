from __future__ import annotations
"""
ensemble.py — async per-model calls with robust parsing + usage/cost capture,
using providers.call_chat_ms(...) which supports OpenRouter, Gemini-direct, Grok-direct.
"""

import asyncio, time, re, json
from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

from .providers import ModelSpec, llm_semaphore, call_chat_ms, estimate_cost_usd

@dataclass
class MemberOutput:
    name: str
    ok: bool
    parsed: Any
    raw_text: str
    elapsed_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""  # non-empty if provider errored quickly

@dataclass
class EnsembleResult:
    members: List[MemberOutput]

# ---------- helpers ----------
def sanitize_mcq_vector(vec: List[float], n_options: Optional[int] = None) -> List[float]:
    try:
        v = np.array([float(x) for x in (vec or [])], dtype=float)
    except Exception:
        v = np.array([], dtype=float)
    if n_options is not None:
        if v.size < n_options:
            v = np.pad(v, (0, n_options - v.size), mode="constant", constant_values=0.0)
        elif v.size > n_options:
            v = v[:n_options]
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, 1.0)
    s = float(v.sum())
    if s <= 0.0:
        n = v.size if v.size > 0 else (n_options if n_options else 1)
        return (np.ones(n) / float(n)).tolist()
    return (v / s).tolist()

def _parse_binary_probability(text: str) -> Optional[float]:
    if not text:
        return None
    t = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.S)
    # percents
    percents = re.findall(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%", t)
    for p in reversed(percents):
        try:
            v = float(p)/100.0
            if 0.0 <= v <= 1.0: return v
        except: pass
    # probability terms
    prob_blocks = re.findall(
        r"(?:prob(?:ability)?|likelihood|chance|p\s*=?|pr\s*\()\s*[:=]?\s*(\d{1,2}(?:\.\d+)?\s*%|0?\.\d+|1(?:\.0+)?)",
        t, flags=re.I)
    for tok in reversed(prob_blocks):
        tok = tok.strip()
        if tok.endswith('%'):
            try:
                v = float(tok[:-1].strip())/100.0
                if 0.0 <= v <= 1.0: return v
            except: pass
        else:
            try:
                v = float(tok)
                if 0.0 <= v <= 1.0: return v
            except: pass
    # naked decimals
    decimals = re.findall(r"(?<!\d)(0?\.\d+|1(?:\.0+)?)\b", t)
    for d in reversed(decimals):
        try:
            v = float(d)
            if 0.0 <= v <= 1.0: return v
        except: pass
    return None

# ---------- BINARY ----------
async def run_ensemble_binary(prompt: str, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.1)
            p = _parse_binary_probability(text)
            ok = p is not None
            if not ok: p = 0.0
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=p, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))

# ---------- MCQ ----------
async def run_ensemble_mcq(prompt: str, n_options: int, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.2)
            ok = False
            try:
                jmatch = re.search(r"\[[^\]]+\]", text, flags=re.S)
                if jmatch:
                    vec = [float(x) for x in json.loads(jmatch.group(0))]
                else:
                    nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", text)]
                    vec = nums[:n_options]
                vec = sanitize_mcq_vector(vec, n_options)
                ok = len(vec) == n_options
            except Exception:
                vec = sanitize_mcq_vector([], n_options)
                ok = False
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=vec, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))

# ---------- NUMERIC ----------
async def run_ensemble_numeric(prompt: str, specs: List[ModelSpec]) -> EnsembleResult:
    tasks = []
    for ms in specs:
        async def _one(ms=ms):
            t0 = time.time()
            text, usage, err = await call_chat_ms(ms, prompt, temperature=0.2)
            out = {"P10": None, "P50": None, "P90": None}
            ok = False
            try:
                nums = [float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", text)]
                if len(nums) >= 3:
                    out["P10"], out["P50"], out["P90"] = nums[0], nums[1], nums[2]
                    ok = True
            except Exception:
                ok = False
            cost = estimate_cost_usd(ms.model_id, usage)
            return MemberOutput(
                name=ms.name, ok=ok, parsed=out, raw_text=text, error=err,
                elapsed_ms=int((time.time()-t0)*1000),
                prompt_tokens=usage.get("prompt_tokens",0),
                completion_tokens=usage.get("completion_tokens",0),
                total_tokens=usage.get("total_tokens",0),
                cost_usd=cost
            )
        tasks.append(_one())
    return EnsembleResult(members=await asyncio.gather(*tasks))
