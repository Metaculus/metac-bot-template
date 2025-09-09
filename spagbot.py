#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spagbot — Ensemble (OpenRouter GPT-5 → fallback 4o + Gemini + Grok) + GTMC1 + Bayes-MC
Build date: 2025-09-01 

This build updates Bayesian aggregation:
1) WEAK PRIORS: Use weak priors (alpha=0.1) for Binary & MCQ questions to let LLM evidence dominate.
2) NUMERIC MIXTURE MODEL: For numeric questions, pass each LLM's percentile forecast to bayes_mc
   to be modeled as a separate distribution in a robust Normal mixture model. This replaces
   the old, unstable method of averaging percentiles before fitting a single log-normal.
3) CSV LOGGING FIX: The `forecasts.csv` now correctly logs the final aggregated P10/P50/P90
   from the Bayesian process, not the pre-aggregation median inputs.
"""

from __future__ import annotations
import argparse, asyncio, csv, json, math, os, re, sys, logging
from seen_guard import SeenGuard, extract_metaculus_id  # DEDUPE GUARD
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

# --- Timezone handling (Europe/Istanbul) -------------------------------------
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    # Fallback if zoneinfo not available (older Python): UTC+03 fixed offset
    IST_TZ = timezone(timedelta(hours=3))

import dotenv; dotenv.load_dotenv()
import requests
import numpy as np
import difflib, urllib.parse  # for title similarity & URL param encoding

# Optional deps (OpenAI SDK for OpenRouter and xAI)
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
try:
    from asknews_sdk import AskNewsSDK
except Exception:
    AskNewsSDK = None

# Local modules
import GTMC1
import bayes_mc as BMC

# -------------------------------------------------------------------------------------
# Fallback: if bayes_mc in your repo doesn't have apply_calibration_weight yet,
# define a local shim and use it (prevents AttributeError at runtime).
# -------------------------------------------------------------------------------------
try:
    _APPLY_CAL_W = BMC.apply_calibration_weight  # type: ignore[attr-defined]
except Exception:
    def _APPLY_CAL_W(raw_weight: float, question_type: str, top_prob: float) -> float:
        """
        Calibrated weight for external signals when explicit calibration data is sparse.
        Parabolic shrink near extremes; floor at 0.2 for stability.
        """
        try:
            raw_weight = float(raw_weight)
            if question_type == "binary":
                penalty = 1.0 - 4.0 * (float(top_prob) - 0.5) ** 2  # 1 at 0.5; 0 at 0 or 1
                return float(max(0.2, min(1.0, raw_weight * max(0.0, penalty))))
            return float(max(0.2, min(1.0, raw_weight)))
        except Exception:
            return float(max(0.2, min(1.0, raw_weight)))
else:
    def _APPLY_CAL_W(raw_weight: float, question_type: str, top_prob: float) -> float:
        return float(_APPLY_CAL_W(raw_weight, question_type, top_prob))  # proxy

# =====================================================================================
# Configuration
# =====================================================================================

PRINT_PREFIX = "🚀"

# Toggles
SUBMIT_PREDICTION = (os.getenv("SUBMIT_PREDICTION", "0") == "1")
USE_OPENROUTER    = (os.getenv("USE_OPENROUTER", "1") == "1")
USE_GOOGLE        = (os.getenv("USE_GOOGLE", "1") == "1")
ENABLE_GROK       = (os.getenv("ENABLE_GROK", "1") == "1")

# Keys / endpoints
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")         # OpenRouter key (or OpenAI if not using OpenRouter)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")        # should be https://openrouter.ai/api/v1 when using OpenRouter
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")         # direct Gemini
XAI_API_KEY     = os.getenv("XAI_API_KEY")            # direct xAI
XAI_BASE_URL    = "https://api.x.ai/v1"               # xAI's OpenAI-compatible endpoint
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET    = os.getenv("ASKNEWS_SECRET")

# Model ids (updated defaults)
OPENROUTER_GPT5_ID        = os.getenv("OPENROUTER_GPT5_ID", "openai/gpt-4o")
OPENROUTER_GPT5_THINK_ID  = os.getenv("OPENROUTER_GPT5_THINK_ID", "openai/gpt-4o")
OPENROUTER_FALLBACK_ID    = os.getenv("OPENROUTER_FALLBACK_ID", "openai/gpt-4o")
GEMINI_MODEL              = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
XAI_GROK_ID               = os.getenv("XAI_GROK_ID", "grok-4")
OPENROUTER_CLAUDE37_ID    = os.getenv("OPENROUTER_CLAUDE37_ID", "anthropic/claude-3.7-sonnet")

# Timeouts
GPT5_CALL_TIMEOUT_SEC     = float(os.getenv("GPT5_CALL_TIMEOUT_SEC", 120))
GEMINI_CALL_TIMEOUT_SEC   = float(os.getenv("GEMINI_CALL_TIMEOUT_SEC", 120))
GROK_CALL_TIMEOUT_SEC     = float(os.getenv("GROK_CALL_TIMEOUT_SEC", 180))

# === Decoding controls (separate knobs for research vs. forecasting) ===
FORECAST_TEMP = float(os.getenv("FORECAST_TEMP", "0.00"))   # 0.00 is near-deterministic
RESEARCH_TEMP = float(os.getenv("RESEARCH_TEMP", "0.20"))   # research may stay a bit creative

# Optional nucleus clamp; if provider ignores it, no harm done
FORECAST_TOP_P = float(os.getenv("FORECAST_TOP_P", "0.20"))  # try 0.10–0.30
RESEARCH_TOP_P = float(os.getenv("RESEARCH_TOP_P", "1.00"))  # default no clamp for research

# Ensemble concurrency
NUM_RUNS_PER_QUESTION     = int(os.getenv("NUM_RUNS_PER_QUESTION", 3))
CONCURRENT_REQUESTS_LIMIT = 5

# Research cache control
DISABLE_RESEARCH_CACHE = os.getenv("SPAGBOT_DISABLE_RESEARCH_CACHE", "0").lower() in ("1","true","yes")

# Prediction-market lookups
ENABLE_MARKET_SNAPSHOT = os.getenv("ENABLE_MARKET_SNAPSHOT", "1").lower() in ("1","true","yes")
MARKET_SNAPSHOT_MAX_MATCHES = int(os.getenv("MARKET_SNAPSHOT_MAX_MATCHES", 3))
METACULUS_INCLUDE_RESOLVED  = os.getenv("METACULUS_INCLUDE_RESOLVED", "1").lower() in ("1","true","yes")

# Tournament & API
TOURNAMENT_ID = os.getenv("TOURNAMENT_ID", "fall-aib-2025")
API_BASE_URL  = "https://www.metaculus.com/api"
AUTH_HEADERS  = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}

# Files & logs
FORECASTS_CSV      = "forecasts.csv"
FORECASTS_BY_MODEL = "forecasts_by_model.csv"
FORECAST_LOG_DIR   = "forecast_logs"
RUN_LOG_DIR        = "logs"
CACHE_DIR          = "cache"

# ANCHOR: "MINI-BENCH one-shot helpers"
ONE_SHOT_PATH = os.path.join(CACHE_DIR, "mini_bench_submitted.json")

def _load_one_shot_ids() -> set[int]:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(ONE_SHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(int(x) for x in data)
    except Exception:
        return set()

def _save_one_shot_ids(ids: set[int]) -> None:
    try:
        with open(ONE_SHOT_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)
    except Exception:
        pass

def is_mini_bench() -> bool:
    return isinstance(TOURNAMENT_ID, str) and ("mini" in TOURNAMENT_ID.lower())

def mark_submitted_if_mini(qid: int) -> None:
    if not is_mini_bench():
        return
    ids = _load_one_shot_ids()
    if qid not in ids:
        ids.add(int(qid))
        _save_one_shot_ids(ids)

# === Calibration loader (auto-included in all prompts) ======================
# ANCHOR: "CALIBRATION: loader"
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "data/calibration_advice.txt")

def _load_calibration_note() -> str:
    """
    Pulls the latest calibration guidance written by update_calibration.py.
    Strategy:
      1) Try CALIBRATION_PATH (env or default 'data/calibration_advice.txt').
      2) If missing, auto-fallback to './calibration_advice.txt' at repo root.
    Returns "" if nothing readable is found, so prompts stay valid.
    """
    # First path: env/default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")
    except Exception:
        pass
    # Fallback path at repo root
    try:
        alt = "calibration_advice.txt"
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                return txt if len(txt) <= 4000 else (txt[:3800] + "\n…[truncated]")
    except Exception:
        pass
    return ""

_CAL_NOTE = _load_calibration_note()
_CAL_PREFIX = (
    "CALIBRATION GUIDANCE (auto-generated weekly):\n"
    + (_CAL_NOTE if _CAL_NOTE else "(none available yet)")
    + "\n— end calibration —\n\n"
)

# ---- MCQ wide CSV (fixed columns) ----
MAX_MCQ_OPTIONS = 20
MCQ_WIDE_CSV = "forecasts_mcq_wide.csv"

def _mcq_wide_headers() -> list[str]:
    base = ["RunID","RunTime","URL","QuestionID","Question","K"]
    for i in range(1, MAX_MCQ_OPTIONS + 1):
        base.append(f"OptionLabel_{i}")
        base.append(f"OptionProb_{i}")
    return base

# --- Logging to console + file
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spagbot")

# =====================================================================================
# Utilities (IST timestamps; CSV float formatters; safe cache helpers)
# =====================================================================================

def ist_stamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    return datetime.now(IST_TZ).strftime(fmt)

def ist_iso(fmt: str = "%Y-%m-%d %H:%M:%S %z") -> str:
    return datetime.now(IST_TZ).strftime(fmt)

def ist_date(fmt: str = "%Y-%m-%d") -> str:
    return datetime.now(IST_TZ).strftime(fmt)

def _clip01(x: float) -> float:
    return max(0.01, min(0.99, float(x)))

def _fmt_float_or_blank(x: Optional[float]) -> str:
    return "" if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else f"{float(x):.6f}"

def _slug_for_post(post_id: int) -> str:
    return f"q{post_id}"

def _cache_path(kind: str, slug: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{kind}__{slug}.json")

def _read_cache(kind: str, slug: str) -> Optional[dict]:
    p = _cache_path(kind, slug)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _write_cache(kind: str, slug: str, data: dict) -> None:
    try:
        with open(_cache_path(kind, slug), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =====================================================================================
# Metaculus API
# =====================================================================================

def list_posts_from_tournament(tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50) -> dict:
    url_qparams = {
        "limit": count, "offset": offset, "order_by": "-hotness",
        "forecast_type": ",".join(["binary", "multiple_choice", "numeric", "discrete"]),
        "tournaments": [tournament_id], "statuses": "open", "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    r = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not r.ok:
        raise RuntimeError(r.text)
    return json.loads(r.content)

def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()
    out: list[tuple[int, int]] = []
    for post in posts["results"]:
        if question := post.get("question"):
            if question.get("status") == "open":
                out.append((question["id"], post["id"]))
    return out

def get_post_details(post_id: int) -> dict:
    url = f"{API_BASE_URL}/posts/{post_id}/"
    r = requests.get(url, **AUTH_HEADERS)  # type: ignore
    if not r.ok:
        raise RuntimeError(r.text)
    return json.loads(r.content)

def create_forecast_payload(forecast: float | dict[str, float] | list[float], question_type: str) -> dict:
    if question_type == "binary":
        return {"probability_yes": forecast, "probability_yes_per_category": None, "continuous_cdf": None}
    if question_type == "multiple_choice":
        return {"probability_yes": None, "probability_yes_per_category": forecast, "continuous_cdf": None}
    return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": forecast}

def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    url = f"{API_BASE_URL}/questions/forecast/"
    r = requests.post(url, json=[{"question": question_id, **forecast_payload}], **AUTH_HEADERS)  # type: ignore
    logger.info(f"POST forecast status: {r.status_code}")
    if not r.ok:
        raise RuntimeError(r.text)

def post_question_comment(post_id: int, comment_text: str) -> None:
    r = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={"text": comment_text, "parent": None, "included_forecast": True, "is_private": True, "on_post": post_id},
        **AUTH_HEADERS,  # type: ignore
    )
    logger.info(f"POST comment status: {r.status_code}")
    if not r.ok:
        raise RuntimeError(r.text)

# =====================================================================================
# Market consensus snapshot (Metaculus + Manifold) — appended to research
# =====================================================================================

def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())

def _title_similarity(a: str, b: str) -> float:
    na, nb = _normalize_text(a), _normalize_text(b)
    return difflib.SequenceMatcher(None, na, nb).ratio()

def _metaculus_prob_yes_from_post(post_json: dict):
    q = post_json.get("question", {})
    aggs = (q.get("aggregations") or post_json.get("aggregations") or {})
    rw = (aggs.get("recency_weighted") or {})
    latest = (rw.get("latest") or {})
    yes = None

    centers = latest.get("centers")
    if isinstance(centers, list) and centers:
        try:
            cand = float(centers[0])
            if 0.0 <= cand <= 1.0:
                yes = cand
        except Exception:
            pass

    if yes is None:
        fv = latest.get("forecast_values")
        if isinstance(fv, list) and len(fv) == 2:
            try:
                yes = float(fv[1])
            except Exception:
                yes = None

    if yes is None and isinstance(centers, list) and q.get("type") == "multiple_choice":
        opts = q.get("options") or []
        return {str(opts[i]): float(centers[i]) for i in range(min(len(opts), len(centers)))}

    return yes

def _format_percent(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"

def _manifold_top_match(title: str, *, threshold: float = 0.60):
    try:
        tried = []
        def _attempt(url, params):
            tried.append((url, params))
            return requests.get(url, params=params, timeout=10)

        r = _attempt("https://api.manifold.markets/v0/search-markets",
                     {"term": title, "limit": 50})
        if not r.ok:
            r = _attempt("https://api.manifold.markets/v0/markets",
                         {"q": title, "limit": 50})
        if not r.ok:
            r = _attempt("https://api.manifold.markets/v0/markets",
                         {"query": title, "limit": 50})
        if not r.ok:
            print(f"[manifold] HTTP {r.status_code}: {r.text[:200]}")
            print(f"[manifold] Tried: {tried}")
            return None

        candidates = r.json()

    except Exception as e:
        print(f"[manifold] request failed: {e}")
        return None

    best = None
    best_score = threshold
    for m in candidates:
        m_title = m.get("question") or m.get("title") or ""
        outcome = (m.get("outcomeType") or m.get("outcome_type") or "").upper()
        if outcome not in ("BINARY", "PSEUDO_NUMERIC"):
            continue

        score = _title_similarity(title, m_title)
        if score >= best_score:
            prob = m.get("probability")
            if prob is None:
                prob = m.get("p") or m.get("lastProbability") or m.get("closeProb")
            try:
                prob = float(prob)
            except Exception:
                continue

            url = m.get("url")
            if not url:
                slug = m.get("slug") or ""
                creator = m.get("creatorUsername") or m.get("creator") or ""
                url = f"https://manifold.markets/{creator}/{slug}".rstrip("/")

            best = {"title": m_title.strip(), "prob_yes": prob, "url": url,
                    "similarity": score, "outcome": "binary"}
            best_score = score
    return best

def build_market_consensus_snapshot(*, title: str, post: dict,
                                    similarity_threshold: float = 0.60,
                                    include_manifold: bool = True) -> str:
    lines = []
    lines.append("### Market Consensus Snapshot")
    lines.append(f"_Captured {ist_iso()} (Similarity threshold ≥{similarity_threshold:.2f} on title match)._")

    # -- Metaculus
    lines.append("**Metaculus (community forecast):**")
    try:
        mid = post.get("id") or post.get("post_id") or post.get("question", {}).get("id")
        m_url = f"https://www.metaculus.com/questions/{mid}/" if mid else None
        q = post.get("question", {}) or {}
        qtype = q.get("type") or "unknown"
        val = _metaculus_prob_yes_from_post(post)

        if isinstance(val, dict):  # MCQ
            mcq_bits = [f"- {k} — {_format_percent(v)}" for k, v in val.items()]
            item = f"- {title} — mcq — " + "; ".join(mcq_bits)
            if m_url: item += f" — [link]({m_url})"
            lines.append(item)
        elif isinstance(val, (int, float)):
            item = f"- {title} — {qtype} — {_format_percent(val)} YES"
            if m_url: item += f" — [link]({m_url})"
            lines.append(item)
        else:
            item = f"- {title} — {qtype} — n/a"
            if m_url: item += f" — [link]({m_url})"
            lines.append(item)
    except Exception as e:
        lines.append(f"- {title} — error extracting forecast: {e!r}")

    # -- Manifold
    if include_manifold:
        lines.append("")
        lines.append("**Manifold (play-money odds):**")
        try:
            hit = _manifold_top_match(title, threshold=similarity_threshold)
            if hit:
                lines.append(f"- {hit['title']} — {hit['outcome']} — {_format_percent(hit['prob_yes'])} YES — [link]({hit['url']})")
            else:
                lines.append("- No sufficiently similar market found.")
        except Exception as e:
            lines.append(f"- lookup failed: {e!r}")

    lines.append("")
    lines.append("_Sources: **Metaculus** aggregates user forecasts into a community prediction; **Manifold** odds come from user betting with play-money. Treat these as noisy evidence, not ground truth; they update continuously._")
    return "\n".join(lines)

# =====================================================================================
# Research (ALWAYS ON, cached; injected into prompts)
# =====================================================================================

async def run_research_async(
    *,
    title: str,
    description: str,
    criteria: str,
    qtype: str,
    options: list[str] | None,
    units: str | None,
    slug: str,
) -> str:
    """
    LLM-first research pipeline with smart fallbacks + market snapshot:
      1) If AskNews creds are set, fetch 4–8 items and pass as SOURCES to the LLM.
      2) Try ChatGPT-4o (OpenRouter). If unavailable: Gemini 2.5 Pro.
      3) Cache the LLM brief.
    Always returns *something* so the forecaster prompt never goes empty.
    """
    # Try to load only the LLM text from cache; we still want live market data
    llm_text: Optional[str] = None
    if not DISABLE_RESEARCH_CACHE:
        cached = _read_cache("research_llm", slug)
        if cached and "text" in cached:
            llm_text = cached["text"]

    # If we HIT the research cache, wrap it with calibration guidance and return early
    if llm_text is not None:
        try:
            out_text = ""
            if _CAL_PREFIX:  # calibration note prefix loaded at import-time
                out_text += _CAL_PREFIX + "\n"
        except NameError:
            out_text = ""
        out_text += (llm_text or "")
        return out_text

    # 1) Collect source snippets (optional)
    sources_text = ""
    if llm_text is None and AskNewsSDK and ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
        def _pull_news():
            ask = AskNewsSDK(client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"]))
            res = ask.news.search_news(query=f"{title}\n\n{description}", n_articles=8, return_type="both", strategy="latest news")
            return res.as_string or ""
        try:
            sources_text = await asyncio.to_thread(_pull_news)
        except Exception as e:
            sources_text = f"[AskNews failed: {e}]"

    # 2) Build the LLM prompt and call a model (only if we don't have cached text)
    if llm_text is None:
        units_or_options = ""
        if qtype == "multiple_choice" and options:
            units_or_options = " | ".join(str(o) for o in options)
        elif units:
            units_or_options = str(units)

        today = ist_date()
        prompt = build_research_prompt(
            title=title,
            qtype=qtype,
            units_or_options=units_or_options,
            background=description,
            criteria=criteria,
            today=today,
            sources_text=sources_text,
        )

        # Preferred: OpenRouter (ChatGPT-4o), then Gemini
        spec_gpt4o = ModelSpec(name="ChatGPT-Research-4o", provider="openrouter", model_id=OPENROUTER_FALLBACK_ID, temperature=RESEARCH_TEMP, timeout_s=GPT5_CALL_TIMEOUT_SEC)
        text1 = await _call_openrouter(spec_gpt4o, prompt)
        if not text1.startswith("[error]") and text1.strip():
            llm_text = (text1 or "").strip()
            _write_cache("research_llm", slug, {"text": llm_text, "provider": spec_gpt4o.model_id})
            print(f"Research by: {spec_gpt4o.model_id}")
        else:
            spec_gem = ModelSpec(name="Gemini-Research", provider="google", model_id=GEMINI_MODEL, temperature=RESEARCH_TEMP, timeout_s=GEMINI_CALL_TIMEOUT_SEC)
            text2 = await _call_google(spec_gem, prompt)
            if not text2.startswith("[error]") and text2.strip():
                llm_text = (text2 or "").strip()
                _write_cache("research_llm", slug, {"text": llm_text, "provider": spec_gem.model_id})
                print(f"Research by: {spec_gem.model_id}")
    
            else:
                llm_text = "No external research provider succeeded. Proceed with background and general knowledge. (LLM research fallback failed.)"
                _write_cache("research_llm", slug, {"text": llm_text, "provider": "none"})
                print("Research by: Failed")

        # Inject calibration guidance before returning
        out_text = ""
        try:
            if _CAL_PREFIX:
                out_text += _CAL_PREFIX + "\n"
        except NameError:
            pass
        out_text += (llm_text or "")
        return out_text

# =====================================================================================
# LLM Ensemble (OpenRouter + Google direct + xAI direct)
# =====================================================================================

llm_semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)

# Lazy clients
_or_client: Optional[Any]  = None
_xai_client: Optional[Any] = None

def _get_or_client() -> Optional[Any]:
    """
    Build an OpenAI-compatible client. If using OpenRouter and no base_url is set,
    force the correct OpenRouter URL to avoid '404 / model not found' errors.
    """
    global _or_client
    if _or_client is not None:
        return _or_client
    if AsyncOpenAI is None or OPENAI_API_KEY is None:
        return None
    base = OPENAI_BASE_URL
    if USE_OPENROUTER and (not base or "openrouter" not in str(base).lower()):
        base = "https://openrouter.ai/api/v1"  # hard default to prevent silent misrouting
    _or_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=base) if base else AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _or_client

def _get_xai_client() -> Optional[Any]:
    global _xai_client
    if _xai_client is not None:
        return _xai_client
    if AsyncOpenAI is None or (not XAI_API_KEY):
        return None
    _xai_client = AsyncOpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
    return _xai_client

@dataclass
class ModelSpec:
    name: str
    provider: str      # 'openrouter' | 'google' | 'xai'
    model_id: str
    temperature: float
    timeout_s: float
    enabled: bool = True
    weight: float = 1.0

DEFAULT_ENSEMBLE: List[ModelSpec] = []
# OpenRouter default reasoning model (currently set to GPT-4o in .env)
if USE_OPENROUTER:
    DEFAULT_ENSEMBLE.append(
        ModelSpec(
            "OpenRouter-Default",
            "openrouter",
            OPENROUTER_GPT5_THINK_ID,   # maps to openai/gpt-4o in your .env
            FORECAST_TEMP,
            GPT5_CALL_TIMEOUT_SEC,
            True,
            1.0,
        )
    )

# Anthropic Claude 3.7 Sonnet via OpenRouter (guarded)
if USE_OPENROUTER:
    DEFAULT_ENSEMBLE.append(
        ModelSpec(
            "Claude-3.7-Sonnet (OR)",
            "openrouter",
            OPENROUTER_CLAUDE37_ID,
            FORECAST_TEMP,
            GPT5_CALL_TIMEOUT_SEC,
            True,
            1.0,
        )
    )

# Gemini via Google (direct)
if USE_GOOGLE:
    DEFAULT_ENSEMBLE.append(ModelSpec("Gemini", "google", GEMINI_MODEL, FORECAST_TEMP, GEMINI_CALL_TIMEOUT_SEC, True, 0.9))
# Grok via xAI (direct)
if ENABLE_GROK:
    DEFAULT_ENSEMBLE.append(ModelSpec("Grok", "xai", XAI_GROK_ID, FORECAST_TEMP, GROK_CALL_TIMEOUT_SEC, True, 0.8))

# ---- Provider-specific call helpers ----

async def _call_openrouter(spec: ModelSpec, prompt: str) -> str:
    client = _get_or_client()
    if client is None:
        return "[error] OpenRouter client not available"
    try:
        # Heuristic: choose top_p by whether this looks like a research call (same temp as RESEARCH_TEMP)
        top_p = RESEARCH_TOP_P if abs(spec.temperature - RESEARCH_TEMP) < 1e-6 else FORECAST_TOP_P
        async with llm_semaphore:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=spec.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=spec.temperature,
                    top_p=top_p,
                    stream=False,
                ),
                timeout=spec.timeout_s,
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e).lower()
        if ("404" in msg) or ("model_not_found" in msg) or ("no allowed providers" in msg) or ("not found" in msg):
            try:
                async with llm_semaphore:
                    resp2 = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=OPENROUTER_FALLBACK_ID,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=spec.temperature,
                            top_p=top_p,
                            stream=False,
                        ),
                        timeout=spec.timeout_s,
                    )
                return (resp2.choices[0].message.content or "").strip()
            except Exception as e2:
                return f"[error] OpenRouter fallback: {e2!r}"
        return f"[error] OpenRouter: {e!r}"

def _google_blocking_call(model_id: str, api_key: str, prompt: str, temperature: float, timeout_s: float) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    # Decide topP using the same heuristic (compare to RESEARCH_TEMP)
    top_p = RESEARCH_TOP_P if abs(temperature - RESEARCH_TEMP) < 1e-6 else FORECAST_TOP_P
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature), "topP": float(top_p)}
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if not r.ok:
            return f"[error] Google API {r.status_code}: {r.text[:200]}"
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return "[error] Google API: no candidates"
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = "\n".join([t for t in texts if t]).strip()
        return out or "[error] Google API: empty text"
    except Exception as e:
        return f"[error] Google API exception: {e!r}"

async def _call_google(spec: ModelSpec, prompt: str) -> str:
    if not GOOGLE_API_KEY:
        return "[error] GOOGLE_API_KEY missing"
    return await asyncio.to_thread(_google_blocking_call, spec.model_id, GOOGLE_API_KEY, prompt, spec.temperature, spec.timeout_s)

async def _call_xai(spec: ModelSpec, prompt: str) -> str:
    client = _get_xai_client()
    if client is None:
        return "[error] xAI client not available (missing XAI_API_KEY or SDK)"
    try:
        top_p = RESEARCH_TOP_P if abs(spec.temperature - RESEARCH_TEMP) < 1e-6 else FORECAST_TOP_P
        async with llm_semaphore:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=spec.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=spec.temperature,
                    top_p=top_p,
                    stream=False,
                ),
                timeout=spec.timeout_s,
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[error] xAI: {e!r}"

def _should_disable(txt: str) -> bool:
    t = txt.lower()
    return (
        "timeout" in t or
        "no allowed providers" in t or
        "404" in t or
        "model_not_found" in t or
        "bad request" in t or
        "not found" in t
    )

async def _call_one_model(spec: ModelSpec, prompt: str) -> Tuple[str, str]:
    if not spec.enabled:
        return (spec.name, "[error] Model disabled")
    if spec.provider == "openrouter":
        text = await _call_openrouter(spec, prompt)
    elif spec.provider == "google":
        text = await _call_google(spec, prompt)
    elif spec.provider == "xai":
        text = await _call_xai(spec, prompt)
    else:
        text = "[error] Unknown provider"
    return (spec.name, text)

# ---- Parsing helpers ----

@dataclass
class MemberOutput:
    name: str
    raw_text: str
    parsed: Any
    ok: bool
    note: str = ""

@dataclass
class EnsembleResult:
    members: List[MemberOutput] = field(default_factory=list)

def _extract_last_percent(text: str) -> Optional[float]:
    m = re.findall(r"(\d+(?:\.\d+)?)\s*%", text or "")
    if not m: return None
    return _clip01(float(m[-1]) / 100.0)

def _parse_mcq_vector(text: str, n: int) -> Optional[List[float]]:
    nums = [float(x)/100.0 for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text or "")]
    if len(nums) < n: return None
    arr = np.array(nums[-n:], dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    s = float(arr.sum())
    if s <= 0.0: return None
    return (arr / s).tolist()

def sanitize_mcq_vector(vec: List[float]) -> List[float]:
    v = np.clip(np.array(vec, dtype=float), 0.001, 0.999)
    s = float(v.sum())
    v = v / (s if s > 0 else 1.0)
    diff = 1.0 - float(v.sum())
    if abs(diff) > 1e-12:
        i = int(np.argmax(v))
        v[i] = float(v[i] + diff)
    return v.tolist()

_NUMERIC_PAT = re.compile(r"P(10|20|40|50|60|80|90)\s*:\s*(-?\d+(?:\.\d+)?)")

def _parse_numeric_pcts(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in re.findall(_NUMERIC_PAT, text or ""):
        out[f"P{k}"] = float(v)
    return out

def _coerce_p10_p50_p90(d: Dict[str, float]) -> Tuple[float, float, float]:
    p10 = float(d.get("P10", 10.0))
    p90 = float(d.get("P90", 90.0))
    if "P50" in d:
        p50 = float(d["P50"])
    else:
        if "P40" in d and "P60" in d:
            p50 = 0.5 * (float(d["P40"]) + float(d["P60"]))
        else:
            p50 = 0.5 * (p10 + p90)
    return p10, p50, p90

# =====================================================================================
# Prompts
# =====================================================================================

BINARY_PROMPT = _CAL_PREFIX + """
You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign a probability (0–100%) to whether the binary event will occur, using Bayesian reasoning.

Follow these steps in your reasoning before giving the final probability:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate (prior probability P(H)) for the event.
   - Clearly explain why you chose this base rate (e.g., historical frequencies, reference class data, general statistics).
   - State the initial prior in probability or odds form.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference base case.
   - Explain how it is different, and why those differences matter for adjusting the probability.

3. **Evidence Evaluation (Likelihoods)**
   - For each key piece of evidence, consider how likely it would be if the event happens (P(E | H)) versus if it does not happen (P(E | ~H)).
   - Compute or qualitatively describe the likelihood ratio (P(E | H) / P(E | ~H)).
   - State clearly whether each piece of evidence increases or decreases the probability.

4. **Bayesian Updating (Posterior Probability)**
   - Use Bayes’ Rule conceptually:
       Posterior odds = Prior odds × Likelihood ratio
       Posterior probability = (Posterior odds) / (1 + Posterior odds)
   - Walk through at least one explicit update step, showing how the prior probability is adjusted by evidence.
   - Summarize the resulting posterior probability and explain how confident or uncertain it remains.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.
    
5. **Final Forecast**
   - Provide the final forecast as a single calibrated probability.
   - Ensure it reflects both the base rate and the impact of the evidence.

6. **Output Format**
   - End with EXACTLY this line (no other commentary):
Final: ZZ%

Question: {title}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

NUMERIC_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to produce a full probabilistic forecast for a numeric quantity using Bayesian reasoning.

Follow these steps in your reasoning before giving the final percentiles:

1. **Base Rate (Prior) Selection**
   - Identify an appropriate base rate or reference distribution for the target variable.
   - Clearly explain why you chose this base rate (e.g., historical averages, statistical reference classes, domain-specific priors).
   - State the mean/median and variance (or spread) of this base rate.

2. **Comparison to Base Case**
   - Explain how the current situation is similar to the reference distribution.
   - Explain how it is different, and why those differences matter for shifting or stretching the distribution.

3. **Evidence Evaluation (Likelihoods)**
   - For each major piece of evidence in the background or research report, consider how consistent it is with higher vs. lower values.
   - Translate this into a likelihood ratio or qualitative directional adjustment (e.g., “this factor makes higher outcomes 2× as likely as lower outcomes”).
   - Make clear which evidence pushes the forecast up or down, and by how much.

4. **Bayesian Updating (Posterior Distribution)**
   - Use Bayes’ Rule conceptually:
       Posterior ∝ Prior × Likelihood
   - Walk through at least one explicit update step to show how evidence modifies your prior distribution.
   - Describe how the posterior mean, variance, or skew has shifted.

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Percentiles**
   - Provide calibrated percentiles that summarize your posterior distribution.
   - Ensure they are internally consistent (P10 < P20 < P40 < P60 < P80 < P90).
   - Think carefully about tail risks and avoid overconfidence.

7. **Output Format**
   - End with EXACTLY these 6 lines (no other commentary):
P10: X
P20: X
P40: X
P60: X
P80: X
P90: X

Question: {title}
Units: {units}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution:
{criteria}

Today (Istanbul time): {today}
"""

MCQ_PROMPT = _CAL_PREFIX + """You are a careful probabilistic forecaster. Use the background context AND the research report AND your general knowlodge as an LLM.
Your task is to assign probabilities to each of the multiple-choice options using Bayesian reasoning. 
Follow these steps clearly in your reasoning before giving your final answer:

1. **Base Rate (Prior) Selection** - Identify an appropriate base rate (prior probability P(H)) for each option.  
   - Clearly explain why you chose this base rate (e.g., historical frequencies, general statistics, or a reference class).  

2. **Comparison to Base Case** - Explain how the current case is similar to the base rate scenario.  
   - Explain how it is different, and why those differences matter.  

3. **Evidence Evaluation (Likelihoods)** - For each piece of evidence in the background or research report, consider how likely it would be if the option were true (P(E | H)) versus if it were not true (P(E | ~H)).  
   - State these likelihood assessments clearly, even if approximate or qualitative.  

4. **Bayesian Updating (Posterior)** - Use Bayes’ Rule conceptually:  
     Posterior odds = Prior odds × Likelihood ratio  
     Posterior probability = (Posterior odds) / (1 + Posterior odds)  
   - Walk through at least one explicit update step for key evidence, showing how the prior changes into a posterior.  
   - Explain qualitatively how other evidence shifts the probabilities up or down.  

5. **Red Team Thinking**
    - Critically evaluate your own forecast for overconfidence or blind spots.
    - Consider tail risks and alternative scenarios that might affect the distribution.
    - Think of the best alternative forecast and why it might be plausible, as well as rebuttals
    - Adjust your percentiles if necessary to account for these considerations.

6. **Final Normalization** - Ensure the probabilities across all options are consistent and sum to approximately 100%.  
   - Check calibration: if uncertain, distribute probability mass proportionally.  

7. **Output Format** - After reasoning, provide your final forecast as probabilities for each option.  
   - Use EXACTLY N lines, one per option, formatted as:  

Option_1: XX%  
Option_2: XX%  
Option_3: XX%  
...  
(sum ~100%)  

Question: {title}
Options: {options}

Background:
{background}

Research Report (recent/contextual):
{research}

Resolution criteria:
{criteria}

Today (Istanbul time): {today}
"""

# ---------- LLM Researcher Prompt ----------
RESEARCHER_PROMPT = """You are the RESEARCHER for a Bayesian forecasting panel.
Your job is to produce a concise, decision-useful research brief that helps a statistician
update a prior. The forecasters will combine your brief with a statistical aggregator that
expects: base rates (reference class), recency-weighted evidence (relative to horizon),
key mechanisms, differences vs. the base rate, and indicators to watch.

QUESTION
Title: {title}
Type: {qtype}
Units/Options: {units_or_options}

BACKGROUND
{background}

RESOLUTION CRITERIA (what counts as “true”/resolution)
{criteria}

HORIZON & RECENCY
Today (Istanbul): {today}
Guideline: define “recent” relative to time-to-resolution:
- if >12 months to resolution: emphasize last 24 months
- if 3–12 months: emphasize last 12 months
- if <3 months: emphasize last 6 months

SOURCES (optional; may be empty)
Use these snippets primarily if present; if not present, rely on general knowledge.
Do NOT fabricate precise citations; if unsure, say “uncertain”.
{sources}

=== REQUIRED OUTPUT FORMAT (use headings exactly as written) ===
### Reference class & base rates
- Identify 1–3 plausible reference classes; give ballpark base rates or ranges; note limitations.

### Recent developments (timeline bullets)
- [YYYY-MM-DD] item — direction (↑/↓ for event effect on YES) — why it matters (≤25 words)
- Focus on events within the recency guideline above.

### Mechanisms & drivers (causal levers)
- List 3–6 drivers that move probability up/down; note typical size (small/moderate/large).

### Differences vs. the base rate (what’s unusual now)
- 3–6 bullets contrasting this case with the reference class (structure, actors, constraints, policy).

### Bayesian update sketch (for the statistician)
- Prior: brief sentence suggesting a plausible prior and “equivalent n” (strength).
- Evidence mapping: 3–6 bullets with sign (↑/↓) and rough magnitude (small/moderate/large).
- Net effect: one line describing whether the posterior should move up/down and by how much qualitatively.

### Indicators to watch (leading signals; next weeks/months)
- UP indicators: 3–5 short bullets.
- DOWN indicators: 3–5 short bullets.

### Caveats & pitfalls
- 3–5 bullets on uncertainty, data gaps, deception risks, regime changes, definitional gotchas.

Final Research Summary: One or two sentences for the forecaster. Keep the entire brief under ~450 words.
"""

def build_research_prompt(
    title: str,
    qtype: str,
    units_or_options: str,
    background: str,
    criteria: str,
    today: str,
    sources_text: str,
) -> str:
    sources_text = sources_text.strip() if sources_text else "No external sources provided."
    return RESEARCHER_PROMPT.format(
        title=title,
        qtype=qtype,
        units_or_options=units_or_options or "N/A",
        background=(background or "N/A"),
        criteria=(criteria or "N/A"),
        today=today,
        sources=sources_text,
    )

# =====================================================================================
# GTMC1 (strategic heuristic + actor extraction)
# =====================================================================================

STRATEGIC_KEYWORDS = [
    "coalition","alliance","armed conflict","war","attack","airstrike","sanction","negotiation","bargain","actors","faction","bloc",
    "coup","cabinet","parliament","ceasefire","peace talks","trade deal","veto","ratify",
    "election"," vote","government formation","strike","union","boycott",
    "cartel","oligopoly","pricing strategy","hostage","mediation","broker","concession"
]

def looks_strategic(title: str) -> bool:
    text = title.lower()
    return any(k in text for k in STRATEGIC_KEYWORDS)

def build_actor_extraction_prompt(title: str, description: str, research_text: str) -> str:
    return f"""You are a research analyst preparing inputs for a Bruce Bueno de Mesquita-style
game-theoretic bargaining model (BDM/Scholz). Identify actors and quantitative inputs on four dimensions.

TITLE:
{title}

CONTEXT:
{description}

LATEST RESEARCH:
{research_text}

INSTRUCTIONS
1) Define a POLICY CONTINUUM 0–100 for this question:
   0 = outcome least favorable to YES resolution; 100 = most favorable to YES resolution.
2) Identify 3–8 ACTORS that materially influence the outcome (government, opposition, factions,
   mediators, veto players, firms, unions, external patrons).
3) For each actor, provide:
   - "position" (0–100)
   - "capability" (0–100)
   - "salience" (0–100)
   - "risk_threshold" (0.00–0.10)
4) OUTPUT STRICT JSON ONLY; NO commentary; schema:
{
  "policy_continuum": "Short one-sentence description of the 0–100 axis.",
  "actors": [
    {"name":"Government","position":62,"capability":70,"salience":80,"risk_threshold":0.04},
    {"name":"Opposition","position":35,"capability":60,"salience":85,"risk_threshold":0.05}
  ]
}
Constraints: All numbers within ranges; 3–8 total actors; valid JSON.
"""

async def extract_actors_via_llm(title: str, description: str, research_text: str, slug: str) -> tuple[Optional[List[Dict[str, float]]], Optional[str]]:
    cached = _read_cache("actors", slug)
    if cached and "actors" in cached:
        return cached["actors"], None

    if not USE_OPENROUTER:
        return None, "OpenRouter disabled; no actor-extraction model configured"
    client = _get_or_client()
    if client is None:
        return None, "No OpenRouter client configured"

    prompt = build_actor_extraction_prompt(title, description, research_text)
    try:
        async with llm_semaphore:
            resp = await client.chat.completions.create(
                model=OPENROUTER_FALLBACK_ID,
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, f"LLM error: {e!r}"

    if not text:
        return None, "LLM returned empty output"
    try:
        data = json.loads(text)
    except Exception as e:
        return None, f"Invalid JSON from LLM: {e}"

    if "actors" not in data or not isinstance(data["actors"], list):
        return None, "JSON missing 'actors' array"
    actors_raw = data["actors"]
    if not (3 <= len(actors_raw) <= 8):
        return None, f"Expected 3–8 actors, got {len(actors_raw)}"

    cleaned: List[Dict[str, float]] = []
    for i, a in enumerate(actors_raw, 1):
        try:
            name = str(a.get("name","")).strip()
            pos  = float(a.get("position"))
            cap  = float(a.get("capability"))
            sal  = float(a.get("salience"))
            thr  = float(a.get("risk_threshold"))
            if not name: return None, f"Actor {i} has empty name"
            if not (0.0 <= pos <= 100.0): return None, f"Actor {name}: position out of range"
            if not (0.0 <= cap <= 100.0): return None, f"Actor {name}: capability out of range"
            if not (0.0 <= sal <= 100.0): return None, f"Actor {name}: salience out of range"
            if not (0.0 <= thr <= 0.10):  return None, f"Actor {name}: risk_threshold out of [0.00, 0.10]"
            cleaned.append({"name":name, "position":pos, "capability":cap, "salience":sal, "risk_threshold":thr})
        except Exception as e:
            return None, f"Actor {i} parse error: {e}"
    _write_cache("actors", slug, {"actors": cleaned})
    return cleaned, None

# =====================================================================================
# Ensemble runners
# =====================================================================================

async def run_ensemble_binary(prompt: str, ensemble: List[ModelSpec]) -> EnsembleResult:
    outs = await asyncio.gather(*[_call_one_model(ms, prompt) for ms in ensemble])
    res = EnsembleResult()
    for spec, (name, txt) in zip(ensemble, outs):
        display_name = f"{name} [{spec.model_id}]"
        p = _extract_last_percent(txt) if not txt.startswith("[error]") else None
        res.members.append(
            MemberOutput(
                name=display_name,
                raw_text=txt,
                parsed=p,
                ok=(p is not None),
                note="" if p is not None else "parse_failed"
         )
        )
    return res

async def run_ensemble_mcq(prompt: str, n_options: int, ensemble: List[ModelSpec]) -> EnsembleResult:
    outs = await asyncio.gather(*[_call_one_model(ms, prompt) for ms in ensemble])
    res = EnsembleResult()
    for spec, (name, txt) in zip(ensemble, outs):
        display_name = f"{name} [{spec.model_id}]"
        v = _parse_mcq_vector(txt, n_options) if not txt.startswith("[error]") else None
        if v is None:
            v = [1.0 / n_options] * n_options
            res.members.append(
                MemberOutput(name=display_name, raw_text=txt, parsed=v, ok=False, note="fallback_equal")
            )
        else:
            res.members.append(
                MemberOutput(name=display_name, raw_text=txt, parsed=v, ok=True)
            )
    return res

async def run_ensemble_numeric(prompt: str, ensemble: List[ModelSpec]) -> EnsembleResult:
    outs = await asyncio.gather(*[_call_one_model(ms, prompt) for ms in ensemble])
    res = EnsembleResult()
    for spec, (name, txt) in zip(ensemble, outs):
        display_name = f"{name} [{spec.model_id}]"
        d = _parse_numeric_pcts(txt) if not txt.startswith("[error]") else {}
        if not d:
            d = {"P10": 10.0, "P20": 20.0, "P40": 40.0, "P50": 50.0, "P60": 60.0, "P80": 80.0, "P90": 90.0}
            res.members.append(
                MemberOutput(name=display_name, raw_text=txt, parsed=d, ok=False, note="fallback_default")
            )
        else:
            res.members.append(
                MemberOutput(name=display_name, raw_text=txt, parsed=d, ok=True)
            )
    return res

# =====================================================================================
# CSV writers
# =====================================================================================

AGG_HEADERS = [
    "RunID","RunTime","Type","Model","Pool","Collection","URL","QuestionID","Question",
    "Binary_Prob","MCQ_Probs","Numeric_P10","Numeric_P50","Numeric_P90",
    "GTMC1","GTMC1_Signal","BMC_Summary","Explanation_Short"
]

PER_HEADERS = [
    "RunID","RunTime","Type","QuestionID","Question","ModelName","Parsed","OK","Note"
]

def ensure_csvs():
    os.makedirs(FORECAST_LOG_DIR, exist_ok=True)
    if not os.path.exists(FORECASTS_CSV):
        with open(FORECASTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(AGG_HEADERS)
    if not os.path.exists(FORECASTS_BY_MODEL):
        with open(FORECASTS_BY_MODEL, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(PER_HEADERS)

def write_rows(agg_row: Dict[str, str], per_rows: List[Dict[str, str]]) -> None:
    ensure_csvs()
    with open(FORECASTS_CSV, "a", newline="", encoding="utf-8") as fa:
        csv.DictWriter(fa, fieldnames=AGG_HEADERS).writerow(agg_row)
    if per_rows:
        with open(FORECASTS_BY_MODEL, "a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=PER_HEADERS)
            for r in per_rows: w.writerow(r)

def ensure_mcq_wide_csv():
    if not os.path.exists(MCQ_WIDE_CSV):
        with open(MCQ_WIDE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_mcq_wide_headers())

def write_mcq_wide_row(
    run_id: str,
    run_time: str,
    url: str,
    question_id: int,
    question_title: str,
    option_labels: list[str],
    option_probs: list[float],
):
    ensure_mcq_wide_csv()
    K = len(option_labels)
    K_clip = min(K, MAX_MCQ_OPTIONS)

    row = [run_id, run_time, url, str(question_id), question_title, K]

    for i in range(K_clip):
        row.append(option_labels[i])
        row.append(f"{float(option_probs[i]):.6f}")

    for _ in range(K_clip + 1, MAX_MCQ_OPTIONS + 1):
        row.append("")  # OptionLabel_i
        row.append("")  # OptionProb_i

    with open(MCQ_WIDE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# =====================================================================================
# Detailed log generation
# =====================================================================================

def append_detailed_forecast_log(
    log_file_handle,
    run_id: str,
    now: str,
    post_id: int,
    question_id: int,
    title: str,
    url: str,
    qtype: str,
    slug: str,
    research_text: str,
    ensemble_result: EnsembleResult,
    used_gtmc1: bool,
    actors: Optional[List[Dict[str, Any]]],
    gtmc1_signal: dict,
    bmc_summary: dict,
    final_forecast: dict | float,
):
    content = []
    content.append("\n" + ("-" * 60) + "\n")
    content.append(f"❓ Question: {title}")
    content.append(f"🔗 URL: {url}\n")
    content.append("📝 Full Reasoning:")
    content.append("# SUMMARY")
    content.append(f"*Question*: {title}")

    if qtype == "binary":
        pred_str = f"{final_forecast:.2%}"
        content.append(f"*Final Prediction*: {pred_str}")
    elif qtype == "multiple_choice":
        content.append("*Final Prediction*:")
        for k, v in final_forecast.items():
            content.append(f"- {k}: {v:.1%}")
    else:
        content.append("*Final Prediction*: Probability distribution:")
        p10 = bmc_summary.get('p10', 0)
        p50 = bmc_summary.get('p50', 0)
        p90 = bmc_summary.get('p90', 0)
        content.append(f"- P10: {p10:.4g}")
        content.append(f"- P50: {p50:.4g}")
        content.append(f"- P90: {p90:.4g}")

    content.append("\n" + ("-" * 60) + "\n")

    content.append("# RESEARCH")
    content.append("## Research Report")
    content.append(research_text)
    content.append("\n" + ("-" * 60) + "\n")

    content.append("# GTMC1 DETAILS")
    if used_gtmc1 and actors:
        content.append("GTMC1 activated: Yes")
        content.append("\n**Actor Table Input:**")
        content.append("```json")
        content.append(json.dumps(actors, indent=2))
        content.append("```")
        content.append("\n**Output Signal:**")
        content.append("```json")
        content.append(json.dumps(gtmc1_signal, indent=2))
        content.append("```")
    else:
        content.append("GTMC1 activated: No (not a strategic question or actor extraction failed).")
    content
    content.append("\n" + ("-" * 60) + "\n")

    # ENSEMBLE DETAILS
    content.append("# ENSEMBLE OUTPUTS (Raw → Parsed)")
    for m in (ensemble_result.members or []):
        content.append(f"## {m.name}")
        content.append("```text")
        content.append(m.raw_text if isinstance(m.raw_text, str) else str(m.raw_text))
        content.append("```")
        content.append(f"- parsed: {m.parsed!r} | ok={m.ok} | note={m.note or '—'}")
        content.append("")

    # BMC SUMMARY (always present)
    content.append("\n" + ("-" * 60) + "\n")
    content.append("# BAYES-MC AGGREGATION (Summary)")
    try:
        content.append("```json")
        content.append(json.dumps(bmc_summary, indent=2))
        content.append("```")
    except Exception:
        content.append("(unavailable)")

    # Write to the open handle
    try:
        log_file_handle.write("\n".join(content) + "\n")
        log_file_handle.flush()
    except Exception:
        pass


# =====================================================================================
# Prompt builders for different question types
# =====================================================================================

def build_binary_prompt(title: str, background: str, research_text: str, criteria: str) -> str:
    return BINARY_PROMPT.format(
        title=title,
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_mcq_prompt(title: str, options: list[str], background: str, research_text: str, criteria: str) -> str:
    return MCQ_PROMPT.format(
        title=title,
        options="\n".join([str(o) for o in (options or [])]) or "N/A",
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )

def build_numeric_prompt(title: str, units: str, background: str, research_text: str, criteria: str) -> str:
    return NUMERIC_PROMPT.format(
        title=title,
        units=(units or "N/A"),
        background=(background or "N/A"),
        research=(research_text or "N/A"),
        criteria=(criteria or "N/A"),
        today=ist_date(),
    )


# =====================================================================================
# Aggregators (core pipeline, GTMC1 ignored in this pass)
# =====================================================================================

def _weighted_mean(values: list[float], weights: list[float]) -> float:
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    if float(w.sum()) <= 0.0:
        return float(np.mean(v)) if len(v) else 0.5
    return float((v * w).sum() / w.sum())

def aggregate_binary_core(ensemble_res: EnsembleResult, weights: list[float] | None = None) -> tuple[float, dict]:
    """
    Deterministic core aggregation for binary:
    1) Take parsed YES probabilities from members that parsed ok; fallback to 0.5 if none.
    2) Weighted mean (weights from ModelSpec if provided), clipped to [0.01, 0.99].
    3) Return (p_yes, summary_dict).
    """
    vals, oks, names = [], [], []
    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, (int, float)):
            vals.append(float(m.parsed))
        oks.append(m.ok)
        names.append(m.name)

    if not vals:
        p = 0.5
    else:
        if weights and len(weights) == len(ensemble_res.members):
            # build aligned weights (0 for bad parses)
            aligned = []
            j = 0
            for k, m in enumerate(ensemble_res.members):
                if m.ok and isinstance(m.parsed, (int, float)):
                    aligned.append(float(weights[k]))
                    j += 1
                else:
                    aligned.append(0.0)
            p = _weighted_mean(vals, [w for w, m in zip(aligned, ensemble_res.members) if m.ok and isinstance(m.parsed, (int, float))])
        else:
            p = float(np.mean(vals))

    p = _clip01(p)
    summary = {
        "method": "weighted_mean(core)",
        "n_ok": int(sum(1 for m in ensemble_res.members if m.ok)),
        "members": [{"name": m.name, "ok": m.ok, "parsed": m.parsed} for m in ensemble_res.members],
        "p_core": p,
    }
    return p, summary

def aggregate_mcq_core(ensemble_res: EnsembleResult, n_options: int, weights: list[float] | None = None) -> tuple[list[float], dict]:
    """
    For MCQ:
    - Take each member's probability vector (fallback_equal when parsing failed).
    - Average (weighted if weights supplied), renormalize.
    """
    mat = []
    for m in ensemble_res.members:
        v = m.parsed if (isinstance(m.parsed, dict) or isinstance(m.parsed, list)) else None
        if isinstance(v, dict):
            # If dict, assume {label: prob} — convert to list order by option index
            # But our upstream parser returns list; keep simple here
            v = list(v.values())
        if not isinstance(v, list):
            v = [1.0 / n_options] * n_options
        if len(v) != n_options:
            v = (v + [0.0] * n_options)[:n_options]
        mat.append(np.array(v, dtype=float))

    if not mat:
        vec = np.array([1.0 / n_options] * n_options, dtype=float)
    else:
        stack = np.stack(mat, axis=0)
        if weights and len(weights) == len(mat):
            w = np.clip(np.array(weights, dtype=float), 0.0, np.inf)
            if float(w.sum()) > 0:
                vec = (stack * w[:, None]).sum(axis=0) / float(w.sum())
            else:
                vec = stack.mean(axis=0)
        else:
            vec = stack.mean(axis=0)

    vec = sanitize_mcq_vector(vec.tolist())
    summary = {
        "method": "weighted_mean(core)",
        "n_members": len(ensemble_res.members),
        "vector": vec,
    }
    return vec, summary

def aggregate_numeric_core(ensemble_res: EnsembleResult) -> tuple[dict, dict]:
    """
    Numeric:
    - Each member provides some subset of {P10,P20,P40,P50,P60,P80,P90}.
    - We build robust aggregates per percentile using the median across members (trimmed mean alternative).
    - Return (percentiles_dict, summary_dict).
    """
    # Collect per-key values
    keys = ["P10","P20","P40","P50","P60","P80","P90"]
    bag: dict[str, list[float]] = {k: [] for k in keys}
    for m in ensemble_res.members:
        d = m.parsed if isinstance(m.parsed, dict) else {}
        for k in keys:
            if k in d:
                try:
                    bag[k].append(float(d[k]))
                except Exception:
                    pass

    agg: dict[str, float] = {}
    for k in keys:
        vals = sorted(bag[k])
        if not vals:
            continue
        # robust center: median (or trimmed mean if 4+)
        if len(vals) >= 4:
            lo = vals[int(0.1 * (len(vals)-1))]
            hi = vals[int(0.9 * (len(vals)-1))]
            trimmed = [x for x in vals if lo <= x <= hi]
            agg[k] = float(np.mean(trimmed)) if trimmed else float(np.median(vals))
        else:
            agg[k] = float(np.median(vals))

    # Coerce to P10,P50,P90 and sanity
    p10, p50, p90 = _coerce_p10_p50_p90(agg)
    if p10 > p50: p10, p50 = p50, p10
    if p50 > p90: p50, p90 = p90, p50
    if p10 > p50: p10, p50 = p50, p10

    agg_out = {"P10": p10, "P50": p50, "P90": p90}
    summary = {
        "method": "percentile-wise median (trimmed)",
        "n_members": len(ensemble_res.members),
        "available_counts": {k: len(bag[k]) for k in keys},
        "p10": p10, "p50": p50, "p90": p90,
    }
    return agg_out, summary


# =====================================================================================
# One-question orchestration (no GTMC1 in this pass)
# =====================================================================================

async def run_one_question_core(post: dict, *, run_id: str, seen_guard: Optional[SeenGuard] = None) -> None:
    q = post.get("question") or {}
    post_id = int(post.get("id") or post.get("post_id") or 0)
    question_id = int(q.get("id") or 0)
    title = str(q.get("title") or post.get("title") or "").strip()
    url = f"https://www.metaculus.com/questions/{question_id}/" if question_id else ""
    qtype = (q.get("type") or "binary").strip()
    description = str(q.get("description", "") or post.get("description", "") or "")
    criteria = str(q.get("resolution_criteria", "") or q.get("resolution") or "")
    units = q.get("unit") or q.get("units") or ""
    options = q.get("options") or []
    slug = _slug_for_post(post_id or question_id or 0)

    # Optional: skip if already submitted in "mini_bench" mode (one-shot)
    mark_submitted_if_mini(question_id)

    # 1) RESEARCH (cached unless user forces fresh)
    research_text = await run_research_async(
        title=title, description=description, criteria=criteria,
        qtype=qtype, options=options if isinstance(options, list) else None,
        units=str(units) if units else None, slug=slug
    )

    # 1b) Optionally append market snapshot
    if ENABLE_MARKET_SNAPSHOT:
        try:
            snap = build_market_consensus_snapshot(title=title, post=post, similarity_threshold=0.60, include_manifold=True)
            research_text = (research_text or "") + "\n\n" + snap
        except Exception:
            pass

    # 2) Build prompt & run ensemble
    if qtype == "binary":
        prompt = build_binary_prompt(title, description, research_text, criteria)
        ensemble_res = await run_ensemble_binary(prompt, DEFAULT_ENSEMBLE)
    elif qtype == "multiple_choice":
        opt_labels = [str(o) for o in (options or [])]
        prompt = build_mcq_prompt(title, opt_labels, description, research_text, criteria)
        ensemble_res = await run_ensemble_mcq(prompt, len(opt_labels), DEFAULT_ENSEMBLE)
    else:  # 'numeric' or 'discrete' — treat both via numeric prompt/parse
        prompt = build_numeric_prompt(title, str(units or ""), description, research_text, criteria)
        ensemble_res = await run_ensemble_numeric(prompt, DEFAULT_ENSEMBLE)

    # 3) Aggregate (core)
    per_rows: list[dict[str, str]] = []
    for m in ensemble_res.members:
        per_rows.append({
            "RunID": run_id,
            "RunTime": ist_iso(),
            "Type": qtype,
            "QuestionID": str(question_id),
            "Question": title,
            "ModelName": m.name,
            "Parsed": json.dumps(m.parsed) if not isinstance(m.parsed, (float,int)) else f"{float(m.parsed):.6f}",
            "OK": str(bool(m.ok)),
            "Note": m.note or "",
        })

    if qtype == "binary":
        weights = [getattr(ms, "weight", 1.0) for ms in DEFAULT_ENSEMBLE]
        p_core, bmc_summary = aggregate_binary_core(ensemble_res, weights)
        final_forecast = p_core
        agg_row = {
            "RunID": run_id, "RunTime": ist_iso(), "Type": "binary",
            "Model": "core-ensemble", "Pool": TOURNAMENT_ID, "Collection": "core",
            "URL": url, "QuestionID": str(question_id), "Question": title,
            "Binary_Prob": f"{p_core:.6f}",
            "MCQ_Probs": "", "Numeric_P10": "", "Numeric_P50": "", "Numeric_P90": "",
            "GTMC1": "0", "GTMC1_Signal": "", "BMC_Summary": json.dumps(bmc_summary),
            "Explanation_Short": "core weighted mean of ensemble",
        }
    elif qtype == "multiple_choice":
        weights = [getattr(ms, "weight", 1.0) for ms in DEFAULT_ENSEMBLE]
        vec, bmc_summary = aggregate_mcq_core(ensemble_res, len(options or []), weights)
        final_forecast = {str(options[i]): float(vec[i]) for i in range(len(options))}
        agg_row = {
            "RunID": run_id, "RunTime": ist_iso(), "Type": "multiple_choice",
            "Model": "core-ensemble", "Pool": TOURNAMENT_ID, "Collection": "core",
            "URL": url, "QuestionID": str(question_id), "Question": title,
            "Binary_Prob": "", "MCQ_Probs": json.dumps(final_forecast),
            "Numeric_P10": "", "Numeric_P50": "", "Numeric_P90": "",
            "GTMC1": "0", "GTMC1_Signal": "", "BMC_Summary": json.dumps(bmc_summary),
            "Explanation_Short": "core average of ensemble vectors",
        }
        # also write MCQ wide row for convenience
        write_mcq_wide_row(run_id, ist_iso(), url, question_id, title, [str(o) for o in options], list(vec))
    else:
        agg_percentiles, bmc_summary = aggregate_numeric_core(ensemble_res)
        final_forecast = agg_percentiles
        agg_row = {
            "RunID": run_id, "RunTime": ist_iso(), "Type": "numeric",
            "Model": "core-ensemble", "Pool": TOURNAMENT_ID, "Collection": "core",
            "URL": url, "QuestionID": str(question_id), "Question": title,
            "Binary_Prob": "", "MCQ_Probs": "",
            "Numeric_P10": _fmt_float_or_blank(agg_percentiles.get("P10")),
            "Numeric_P50": _fmt_float_or_blank(agg_percentiles.get("P50")),
            "Numeric_P90": _fmt_float_or_blank(agg_percentiles.get("P90")),
            "GTMC1": "0", "GTMC1_Signal": "", "BMC_Summary": json.dumps(bmc_summary),
            "Explanation_Short": "percentile-wise median (trimmed)",
        }

    # 4) LOG FILE (detailed)
    os.makedirs(FORECAST_LOG_DIR, exist_ok=True)
    log_path = os.path.join(FORECAST_LOG_DIR, f"{run_id}__q{question_id}.md")
    with open(log_path, "w", encoding="utf-8") as lf:
        append_detailed_forecast_log(
            log_file_handle=lf, run_id=run_id, now=ist_iso(),
            post_id=post_id, question_id=question_id, title=title, url=url,
            qtype=qtype, slug=slug, research_text=research_text,
            ensemble_result=ensemble_res, used_gtmc1=False, actors=None,
            gtmc1_signal={}, bmc_summary=bmc_summary, final_forecast=final_forecast
        )

    # 5) CSV(s)
    write_rows(agg_row, per_rows)

    # 6) Optional submit to Metaculus (binary & MCQ & numeric)
    if SUBMIT_PREDICTION:
        try:
            if qtype == "binary":
                payload = create_forecast_payload(float(final_forecast), "binary")
            elif qtype == "multiple_choice":
                # Metaculus expects probabilities per category in *order*
                pm = [float(final_forecast[str(opt)]) for opt in options]
                payload = create_forecast_payload(pm, "multiple_choice")
            else:
                # For numeric, submit a continuous CDF: Metaculus supports a discretized CDF; here we
                # provide three points (P10,P50,P90) to a simple piecewise CDF; if your API wrapper
                # expects a dense CDF, adapt this to your in-repo format.
                pf = final_forecast  # dict
                cdf = {"p10": float(pf["P10"]), "p50": float(pf["P50"]), "p90": float(pf["P90"])}
                payload = create_forecast_payload(cdf, "numeric")
            post_question_prediction(question_id, payload)
            print(f"Submitted forecast for Q{question_id}.")
        except Exception as e:
            print(f"[submit] failed for Q{question_id}: {e!r}")


# =====================================================================================
# CLI / main
# =====================================================================================

def _print_start_banner(limit: int):
    print("🚀 Spagbot ensemble starting…")
    print(f"Mode: test_questions | METACULUS_TOKEN set: {bool(METACULUS_TOKEN)}")
    print(f"USE_OPENROUTER={int(USE_OPENROUTER)} | USE_GOOGLE={int(USE_GOOGLE)} | ENABLE_GROK={int(ENABLE_GROK)}")
    base = OPENAI_BASE_URL or "[default]"
    print(f"OPENAI_BASE_URL={base} | GEMINI_MODEL={GEMINI_MODEL} | XAI_GROK_ID={XAI_GROK_ID}")
    print(f"FORECAST_TEMP={FORECAST_TEMP} | FORECAST_TOP_P={FORECAST_TOP_P} | RESEARCH_TEMP={RESEARCH_TEMP} | RESEARCH_TOP_P={RESEARCH_TOP_P}")
    print(f"Limit: {limit}\n" + "-"*90)

def parse_args():
    ap = argparse.ArgumentParser(description="Spagbot — core ensemble forecaster")
    ap.add_argument("--mode", choices=["test_questions","tournament"], default="test_questions")
    ap.add_argument("--limit", type=int, default=5, help="Max number of questions to run")
    ap.add_argument("--fresh-research", action="store_true", help="Bypass research cache")
    ap.add_argument("--submit", action="store_true", help="Submit forecasts to Metaculus")
    return ap.parse_args()

async def main_async():
    args = parse_args()
    limit = int(args.limit or 5)

    # wiring: allow CLI to override env submission and research cache
    global SUBMIT_PREDICTION, DISABLE_RESEARCH_CACHE
    if args.submit:
        SUBMIT_PREDICTION = True
    if args.fresh_research:
        DISABLE_RESEARCH_CACHE = True

    _print_start_banner(limit)

    # Collect posts (open questions) from the configured tournament
    try:
        posts_resp = list_posts_from_tournament(TOURNAMENT_ID, offset=0, count=limit)
        posts = posts_resp.get("results") or []
    except Exception as e:
        print(f"[error] listing tournament posts: {e!r}")
        return

    run_id = f"{ist_stamp()}__{TOURNAMENT_ID}"

    # Iterate
    for i, post in enumerate(posts[:limit], 1):
        try:
            pid = post.get("id") or post.get("post_id")
            if pid:
                post = get_post_details(int(pid))
        except Exception:
            # fallback: use the shallow post data we already have
            pass

        q = post.get("question") or {}
        title = str(q.get("title") or post.get("title") or "").strip()
        qid = int(q.get("id") or 0)
        print(f"\n{'-'*90}\n[{i}/{limit}] ❓ {title}  (QID: {qid})")

        await run_one_question_core(post, run_id=run_id, seen_guard=None)

    print(f"\n✅ Spagbot run complete at {ist_iso()}")

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[ctrl-c] aborted")

if __name__ == "__main__":
    main()