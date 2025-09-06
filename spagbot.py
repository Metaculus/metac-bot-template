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
        spec_gpt4o = ModelSpec(name="ChatGPT-Research-4o", provider="openrouter", model_id=OPENROUTER_FALLBACK_ID, temperature=0.2, timeout_s=GPT5_CALL_TIMEOUT_SEC)
        text1 = await _call_openrouter(spec_gpt4o, prompt)
        if not text1.startswith("[error]") and text1.strip():
            llm_text = (text1 or "").strip()
            _write_cache("research_llm", slug, {"text": llm_text, "provider": spec_gpt4o.model_id})
            print(f"Research by: {spec_gpt4o.model_id}")
        else:
            spec_gem = ModelSpec(name="Gemini-Research", provider="google", model_id=GEMINI_MODEL, temperature=0.2, timeout_s=GEMINI_CALL_TIMEOUT_SEC)
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
            0.2,
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
            0.2,
            GPT5_CALL_TIMEOUT_SEC,
            True,
            1.0,
        )
    )

# Gemini via Google (direct)
if USE_GOOGLE:
    DEFAULT_ENSEMBLE.append(ModelSpec("Gemini", "google", GEMINI_MODEL, 0.2, GEMINI_CALL_TIMEOUT_SEC, True, 0.9))
# Grok via xAI (direct)
if ENABLE_GROK:
    DEFAULT_ENSEMBLE.append(ModelSpec("Grok", "xai", XAI_GROK_ID, 0.2, GROK_CALL_TIMEOUT_SEC, True, 0.8))

# ---- Provider-specific call helpers ----

async def _call_openrouter(spec: ModelSpec, prompt: str) -> str:
    client = _get_or_client()
    if client is None:
        return "[error] OpenRouter client not available"
    try:
        async with llm_semaphore:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=spec.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=spec.temperature,
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
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature)}
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
        async with llm_semaphore:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=spec.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=spec.temperature,
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
{{
  "policy_continuum": "Short one-sentence description of the 0–100 axis.",
  "actors": [
    {{"name":"Government","position":62,"capability":70,"salience":80,"risk_threshold":0.04}},
    {{"name":"Opposition","position":35,"capability":60,"salience":85,"risk_threshold":0.05}}
  ]
}}
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
    content.append("\n" + ("-" * 60) + "\n")

    if bmc_summary:
        content.append("# BAYES-MC DETAILS")
        summary_to_log = {k: v for k, v in bmc_summary.items() if k != "samples"}
        content.append(json.dumps(summary_to_log, indent=2))
        content.append("\n" + ("-" * 60) + "\n")

    content.append("# MODEL REASONING (RAW)")
    if hasattr(ensemble_result, "members"):
        for member in ensemble_result.members:
            content.append(f"\n## {member.name} Reasoning")
            content.append("```")
            content.append(member.raw_text)
            content.append("```")

    try:
        log_file_handle.write("\n".join(content))
    except Exception as e:
        logger.error(f"Failed to append detailed forecast log for {slug}: {e}")

# =====================================================================================
# Forecast pipeline
# =====================================================================================

def _stage_label(model_name: str) -> str:
    name = (model_name or "").lower()
    if "gpt" in name: return "ChatGPT"
    if "claude" in name: return "Claude"
    if "grok" in name: return "Grok"
    if "gemini" in name or "google" in name: return "Google"
    return model_name or "Model"

def _short_reason(error_text: str) -> str:
    t = (error_text or "").lower()
    for key in ["timeout", "404", "not found", "overloaded", "model_not_found", "no allowed providers", "bad request", "api key", "missing"]:
        if key in t:
            return key
    return "error"

def _collect_stage_statuses(er) -> dict[str, str]:
    out = {"ChatGPT": "absent", "Claude": "absent", "Grok": "absent", "Google": "absent"}
    for m in getattr(er, "members", []):
        label = _stage_label(m.name)
        if label not in out: continue
        if m.ok:
            out[label] = "ok"
        else:
            raw = getattr(m, "raw_text", "") or ""
            out[label] = f"failed: {_short_reason(raw)}"
    return out

async def forecast_one(question_id: int, post_id: int, run_id: str, detailed_log_handle) -> None:
    post_details = get_post_details(post_id)
    q = post_details["question"]
    qtype = q["type"]
    title = q["title"]
    desc  = q.get("description") or ""
    crit  = q.get("resolution_criteria") or (q.get("fine_print") or "")
    units = q.get("unit") or ""
    url   = f"https://www.metaculus.com/questions/{post_id}/"
    now   = ist_iso()
    slug  = _slug_for_post(post_id)

    # Pre-init so Python never sees these as 'unbound' on binary/MCQ code paths.
    bmc_summary: dict = {}
    y = np.array([])

    print("\n" + "-"*90)
    print(f"{PRINT_PREFIX} Question: {title}")
    print(f"URL: {url} | Type: {qtype}")

    # ----- 1. Research Report -----
    research = await run_research_async(
        title=title,
        description=desc,
        criteria=crit,
        qtype=qtype,
        options=q.get("options") if isinstance(q.get("options"), list) else [],
        units=units,
        slug=slug,
    )
    # --- Append community/market consensus to the research brief ---------------
    try:
        snapshot_md = await asyncio.to_thread(
            build_market_consensus_snapshot,
            title=title,
            post=post_details,
            similarity_threshold=0.60,
            include_manifold=True,
        )
        research = (research or "") + "\n\n---\n" + snapshot_md

        manifold_found = "No sufficiently similar market found" not in snapshot_md
        metaculus_found = "error extracting forecast" not in snapshot_md
        print(f"Prediction markets found: Yes (Metaculus: {'1' if metaculus_found else '0'}, Manifold: {'1' if manifold_found else '0'})")

    except Exception as e:
        print(f"Prediction markets found: No (Error: {e})")

    research_for_prompt = research
    today = ist_date()
    per_rows: List[Dict[str, str]] = []

    # ----- 2. Ensemble Forecasting -----
    if qtype == "binary":
        prompt = BINARY_PROMPT.format(title=title, background=desc, criteria=crit, today=today, research=research_for_prompt)
        er = await run_ensemble_binary(prompt, DEFAULT_ENSEMBLE)
        binary_evidences: List[BMC.BinaryEvidence] = []
        for m in er.members:
            if m.ok and m.parsed is not None:
                binary_evidences.append(BMC.BinaryEvidence(p=float(m.parsed), w=1.0))
            per_rows.append({"RunID": run_id, "RunTime": now, "Type": "binary", "QuestionID": str(question_id), "Question": title, "ModelName": m.name, "Parsed": _fmt_float_or_blank(m.parsed), "OK": "1" if m.ok else "0", "Note": m.note})
        exp_short = f"Binary ensemble ({len(binary_evidences)}/{len(er.members)} models OK)"

    elif qtype == "multiple_choice":
        opts: List[str] = q["options"]
        prompt = MCQ_PROMPT.format(title=title, options=opts, background=desc, criteria=crit, today=today, research=research_for_prompt)
        er = await run_ensemble_mcq(prompt, n_options=len(opts), ensemble=DEFAULT_ENSEMBLE)
        mcq_evidences: List[BMC.MCQEvidence] = []
        for m in er.members:
            if m.ok and m.parsed is not None:
                mcq_evidences.append(BMC.MCQEvidence(probs=sanitize_mcq_vector(m.parsed), w=1.0))
            per_rows.append({"RunID": run_id, "RunTime": now, "Type": "multiple_choice", "QuestionID": str(question_id), "Question": title, "ModelName": m.name, "Parsed": json.dumps(m.parsed), "OK": "1" if m.ok else "0", "Note": m.note})
        exp_short = f"MCQ ensemble ({len(mcq_evidences)}/{len(er.members)} models OK)"

    else:  # numeric / discrete
        prompt = NUMERIC_PROMPT.format(title=title, units=units or "N/A", background=desc, criteria=crit, today=today, research=research_for_prompt)
        er = await run_ensemble_numeric(prompt, DEFAULT_ENSEMBLE)
        numeric_evidences: List[BMC.NumericEvidence] = []
        for m in er.members:
            if m.ok and m.parsed:
                try:
                    p10, p50, p90 = _coerce_p10_p50_p90(dict(m.parsed))
                    numeric_evidences.append(BMC.NumericEvidence(p10=p10, p50=p50, p90=p90, w=1.0))
                except Exception:
                    pass
            per_rows.append({"RunID": run_id, "RunTime": now, "Type": qtype, "QuestionID": str(question_id), "Question": title, "ModelName": m.name, "Parsed": json.dumps(m.parsed), "OK": "1" if m.ok else "0", "Note": m.note})
        exp_short = f"Numeric ensemble ({len(numeric_evidences)}/{len(er.members)} models OK)"

    statuses = _collect_stage_statuses(er)
    status_line = " | ".join([f"{model}: {status}" for model, status in statuses.items() if status != 'absent'])
    print(f"\nEnsemble Status: {status_line}")

    # ----- 3. GTMC1 (strategic + binary only) -----
    used_gtmc1 = 0
    gtmc1_signal: Dict[str, str | float | int | None] = {}
    actors: Optional[List[Dict[str, float]]] = None
    if qtype == "binary" and looks_strategic(title):
        actors, fail_reason = await extract_actors_via_llm(title, desc, research_for_prompt, slug)
        if actors:
            try:
                sig, _df = GTMC1.run_monte_carlo_from_actor_table(
                    actors, num_runs=40, log_dir="gtmc_logs", run_slug=slug
                )
                gtmc1_signal = {
                    "coalition_rate": float(sig.get("coalition_rate", 0.0)),
                    "dispersion": str(sig.get("dispersion", "n/a")),
                    "median_of_final_medians": (
                        None if sig.get("median_of_final_medians") is None 
                        else float(sig["median_of_final_medians"])
                    ),
                    "median_rounds": (
                        None if sig.get("median_rounds") is None 
                        else int(sig["median_rounds"])
                    ),
                    "exceedance_ge_50": (None if sig.get("exceedance_ge_50") is None else float(sig["exceedance_ge_50"])),
                    "iqr": (None if sig.get("iqr") is None else float(sig["iqr"])),
                    "runs_csv": sig.get("runs_csv"),
                }
                used_gtmc1 = 1

                if used_gtmc1 and qtype == "binary":
                    pos  = gtmc1_signal.get("median_of_final_medians")
                    disp = gtmc1_signal.get("dispersion", "n/a")
                    rnds = gtmc1_signal.get("median_rounds", "n/a")

                    def _as01_local(x):
                        try:
                            if x is None:
                                return 0.0
                            if isinstance(x, str):
                                xs = x.strip()
                                if xs.endswith("%"):
                                    return max(0.0, min(1.0, float(xs[:-1]) / 100.0))
                                val = float(xs)
                                return val if val <= 1.0 else val / 100.0
                            val = float(x)
                            return val if val <= 1.0 else val / 100.0
                        except Exception:
                            return 0.0

                    coal_p = _as01_local(gtmc1_signal.get("coalition_rate"))
                    ex_p = gtmc1_signal.get("exceedance_ge_50")
                    try:
                        ex_str = f"{float(ex_p):.2%}"
                    except (TypeError, ValueError):
                        ex_str = "n/a"

                    gtmc_brief = (
                        "### GTMC1 Game-Theoretic Signal\n"
                        f"- Coalition rate (YES): {coal_p:.2%}\n"
                        f"- Exceedance (median ≥ 50): {ex_str}\n"
                        f"- Median equilibrium position: {pos if pos is not None else 'n/a'} on 0–100 axis\n"
                        f"- Dispersion: {disp}\n"
                        f"- Rounds to converge (median): {rnds}\n"
                        "Interpretation: treat as bargaining-outcome evidence. If this contradicts your earlier view, explain why and update."
                    )

                    prompt = BINARY_PROMPT.format(
                        title=title, background=desc, criteria=crit, today=today,
                        research=research_for_prompt + "\n\n---\n" + gtmc_brief
                    )
                    er = await run_ensemble_binary(prompt, DEFAULT_ENSEMBLE)

                    binary_evidences = []
                    for m in er.members:
                        if m.ok and m.parsed is not None:
                            binary_evidences.append(BMC.BinaryEvidence(p=float(m.parsed), w=1.0))
                        per_rows.append({
                            "RunID": run_id,
                            "RunTime": now,
                            "Type": "binary",
                            "QuestionID": str(question_id),
                            "Question": title,
                            "ModelName": m.name,
                            "Parsed": _fmt_float_or_blank(m.parsed),
                            "OK": "1" if m.ok else "0",
                            "Note": m.note,
                        })
                    exp_short = f"Binary ensemble+GTMC1 brief ({len(binary_evidences)}/{len(er.members)} models OK)"

                # Console prints
                lines = []
                lines.append("")  # newline
                lines.append("GTMC1 activated: Yes")

                pos  = gtmc1_signal.get("median_of_final_medians")
                coal = gtmc1_signal.get("coalition_rate")
                iqr  = gtmc1_signal.get("iqr", "n/a")
                disp = gtmc1_signal.get("dispersion", "n/a")
                rnds = gtmc1_signal.get("median_rounds", "n/a")
                runs = gtmc1_signal.get("runs_csv", "n/a")
                ex_p = gtmc1_signal.get("exceedance_ge_50")

                def _as01_local2(x):
                    if x is None:
                        return None
                    try:
                        return (x / 100.0) if (isinstance(x, (int, float)) and x > 1) else float(x)
                    except Exception:
                        return None

                ex_p01 = _as01_local2(ex_p)
                coal01 = _as01_local2(coal)

                lines.append(f"  exceedance_ge_50 (P[YES]): {ex_p01:.2%}" if ex_p01 is not None else "  exceedance_ge_50 (P[YES]): n/a")
                lines.append(f"  coalition_rate: {coal01:.2%}" if coal01 is not None else "  coalition_rate: n/a")
                lines.append(f"  median_of_final_medians: {pos if pos is not None else 'n/a'}")
                lines.append(f"  dispersion (IQR): {disp} (iqr={iqr})")
                lines.append(f"  median_rounds: {rnds}")
                lines.append(f"  runs_csv: {runs}")

                print("\n".join(lines))
            except Exception as e:
                print(f"\nGTMC1 activated: No (simulation error: {e})")
        else:
            print(f"\nGTMC1 activated: No (actor extraction failed: {fail_reason})")
    else:
        print("\nGTMC1 activated: No (not a strategic question)")

    # ----- 4. Bayes-MC Aggregation -----
    if qtype == "binary":
        base_evidences = binary_evidences if "binary_evidences" in locals() else []
        evidences = list(base_evidences)

        if used_gtmc1:
            def _as01(x):
                if x is None:
                    return None
                try:
                    return (x / 100.0) if (isinstance(x, (int, float)) and x > 1) else float(x)
                except Exception:
                    return None

            ex_p  = gtmc1_signal.get("exceedance_ge_50")
            pos   = gtmc1_signal.get("median_of_final_medians")
            coal  = gtmc1_signal.get("coalition_rate")

            ex_p01, pos_p01, coal_p01 = _as01(ex_p), _as01(pos), _as01(coal)

            if ex_p01 is not None:
                gtmc_raw_p = _clip01(ex_p01)
            elif pos_p01 is not None:
                gtmc_raw_p = _clip01(pos_p01)
            elif coal_p01 is not None:
                gtmc_raw_p = _clip01(coal_p01)
            else:
                gtmc_raw_p = 0.5

            gtmc1_signal["p_used_for_aggregation"] = gtmc_raw_p

            gtmc_base_w = 0.6
            gtmc_cal_w = _APPLY_CAL_W(
                raw_weight=gtmc_base_w,
                question_type="binary",
                top_prob=gtmc_raw_p
            )
            evidences.append(BMC.BinaryEvidence(p=gtmc_raw_p, w=gtmc_cal_w))

        post = BMC.update_binary_with_mc(
            prior=BMC.BinaryPrior(alpha=0.1, beta=0.1),
            evidences=evidences,
            n_samples=20000,
            seed=42
        )
        final_p = float(post["p50"])
        bmc_summary = {k: v for k, v in post.items() if k != "samples"}
        final_payload = create_forecast_payload(final_p, "binary")
        print(f"\nFinal Forecast (binary): {final_p:.2%}")

    elif qtype == "multiple_choice":
        opts: List[str] = q["options"]
        K = len(opts)
        evs = locals().get("mcq_evidences", [])
        post = BMC.update_mcq_with_mc(
            prior=BMC.DirichletPrior(alphas=[0.1] * K),
            evidences=evs, n_samples=20000, seed=42
        )
        mean_vec = sanitize_mcq_vector([float(x) for x in post["mean"]])
        final_dict = {opts[i]: mean_vec[i] for i in range(K)}
        bmc_summary = {k: v for k, v in post.items() if k != "samples"}
        final_vec = [final_dict[o] for o in opts]
        final_payload = create_forecast_payload(final_vec, "multiple_choice")
        print("\nFinal Forecast (MCQ): " + ", ".join([f"{k}={v:.1%}" for k, v in final_dict.items()]))

        option_labels = list(final_dict.keys())
        option_probs = [final_dict[k] for k in option_labels]
        ensure_mcq_wide_csv()
        write_mcq_wide_row(
            run_id=run_id, run_time=now, url=url, question_id=question_id,
            question_title=title, option_labels=option_labels, option_probs=option_probs,
        )

    else:  # numeric / discrete
        evs = locals().get("numeric_evidences", [])
        post = BMC.update_numeric_with_mc(evidences=evs, n_samples=20000, seed=42)
        bmc_summary = {k: v for k, v in post.items() if k != "samples"}
        print(f"\nFinal Forecast (numeric): P10={bmc_summary.get('p10', 0):.3g}, "
              f"P50={bmc_summary.get('p50', 0):.3g}, P90={bmc_summary.get('p90', 0):.3g}")
        y = np.asarray(post["samples"])

    # ----- Build CDF payload for numeric/discrete -----
    if y.size > 0:
        if qtype == "discrete":
            K = (
                q.get("inbound_outcome_count")
                or (len(q.get("options", [])) if isinstance(q.get("options"), list) else None)
                or q.get("num_outcomes")
                or 0
            )
            try:
                K = int(K)
            except Exception:
                K = 0

            if K <= 1:
                lo, hi = float(np.percentile(y, 0.5)), float(np.percentile(y, 99.5))
                xs = np.linspace(lo, hi, 201).tolist()
                ys = (np.searchsorted(np.sort(y), xs, side="right") / len(y)).tolist()
                final_payload = create_forecast_payload({"xs": xs, "ys": ys}, "discrete")
            else:
                cats = np.clip(np.rint(y).astype(int), 0, K - 1)
                counts = np.bincount(cats, minlength=K)
                cdf_vals = np.cumsum(counts) / float(len(y))
                xs = list(range(K + 1))
                ys = [0.0] + cdf_vals.tolist()
                final_payload = create_forecast_payload({"xs": xs, "ys": ys}, "discrete")

        else:
            lo, hi = float(np.percentile(y, 0.5)), float(np.percentile(y, 99.5))
            xs = np.linspace(lo, hi, 201).tolist()
            ys = (np.searchsorted(np.sort(y), xs, side="right") / len(y)).tolist()
            final_payload = create_forecast_payload({"xs": xs, "ys": ys}, "numeric")
    else:
        final_payload = create_forecast_payload({"xs": [-1, 0, 1], "ys": [0, 0.5, 1]}, "numeric")

    # ----- 5. Logging and Submission -----
    numeric_p10, numeric_p50, numeric_p90 = (None, None, None)
    if qtype in ("numeric", "discrete"):
        numeric_p10 = bmc_summary.get("p10")
        numeric_p50 = bmc_summary.get("p50")
        numeric_p90 = bmc_summary.get("p90")

    pool = os.getenv("RUN_PLAYLIST", os.getenv("TOURNAMENT_ID", "tournament"))
    collection = (
        os.getenv("TOURNAMENT_ID")  # you already use this for posts API
        or (os.getenv("METACULUS_COLLECTION_SLUG_CUP") if pool == "cup" else os.getenv("METACULUS_COLLECTION_SLUG_TOURNAMENT"))
        or "unknown"
    )

    agg_row = {
        
        "RunID": run_id, "RunTime": now, "Type": qtype, "Model": "ensemble+gtmc1+bmc",
        "Pool": pool, "Collection": collection,
        "URL": url, "QuestionID": str(question_id), "Question": title,
        "Binary_Prob": _fmt_float_or_blank(locals().get("final_p")),
        "MCQ_Probs": json.dumps(locals().get("final_dict")) if "final_dict" in locals() else "",
        "Numeric_P10": _fmt_float_or_blank(numeric_p10),
        "Numeric_P50": _fmt_float_or_blank(numeric_p50),
        "Numeric_P90": _fmt_float_or_blank(numeric_p90),
        "GTMC1": str(int(used_gtmc1)),
        "GTMC1_Signal": json.dumps(gtmc1_signal),
        "BMC_Summary": json.dumps({k: v for k, v in (bmc_summary or {}).items() if k != "samples"}),
        "Explanation_Short": exp_short,
    }
    write_rows(agg_row, per_rows)

    final_forecast_for_log = {}
    if qtype == "binary":
        final_forecast_for_log = final_p
    elif qtype == "multiple_choice":
        final_forecast_for_log = final_dict
    else:
        final_forecast_for_log = bmc_summary

    append_detailed_forecast_log(
        log_file_handle=detailed_log_handle,
        run_id=run_id,
        now=now,
        post_id=post_id,
        question_id=question_id,
        title=title,
        url=url,
        qtype=qtype,
        slug=slug,
        research_text=research,
        ensemble_result=er,
        used_gtmc1=bool(used_gtmc1),
        actors=actors,
        gtmc1_signal=gtmc1_signal,
        bmc_summary=bmc_summary,
        final_forecast=final_forecast_for_log,
    )

    if SUBMIT_PREDICTION:
        try:
            post_question_prediction(question_id, final_payload)
            post_question_comment(post_id, f"Automated forecast by Spagbot.\n\n{exp_short}")
            mark_submitted_if_mini(question_id)
        except Exception as e:
            logger.warning(f"POST failed: {e}")

# =====================================================================================
# Runner
# =====================================================================================

# === Metaculus playlist fetch ===============================================
def _metaculus_fetch_open_ids_for_collection(collection_slug: str, *, limit: int = 200) -> list[tuple[int, str]]:
    """
    Returns a list of (qid, url) for OPEN questions in a given Metaculus collection.
    Uses APIv2. Requires METACULUS_TOKEN in env.
    """
    token = os.getenv("METACULUS_TOKEN", "").strip()
    headers = {"Authorization": f"Token {token}"} if token else {}
    # status=open ensures we don't fetch resolved/closed
    url = "https://www.metaculus.com/api2/questions/"
    params = {"collection": collection_slug, "status": "open", "limit": limit}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if not r.ok:
            print(f"[metaculus] fetch collection '{collection_slug}' HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        results = data.get("results", [])
        out = []
        for it in results:
            qid = it.get("id")
            url = it.get("url") or it.get("page_url") or f"https://www.metaculus.com/questions/{qid}/"
            if isinstance(qid, int):
                out.append((qid, url))
        return out
    except Exception as e:
        print(f"[metaculus] error fetching collection '{collection_slug}': {e}")
        return []


def get_playlist_question_ids(playlist: str) -> list[tuple[int, str]]:
    """
    Resolves the chosen playlist ('tournament'/'cup'/'all') to a list of (qid, url).
    Uses env slugs:
      METACULUS_COLLECTION_SLUG_TOURNAMENT
      METACULUS_COLLECTION_SLUG_CUP
    """
    if playlist == "all":
        # Merge both lists (de-dup by qid)
        t_slug = os.getenv("METACULUS_COLLECTION_SLUG_TOURNAMENT", "").strip()
        c_slug = os.getenv("METACULUS_COLLECTION_SLUG_CUP", "").strip()
        ids = []
        if t_slug:
            ids.extend(_metaculus_fetch_open_ids_for_collection(t_slug))
        if c_slug:
            ids.extend(_metaculus_fetch_open_ids_for_collection(c_slug))
        # de-dup by qid, preserve first URL seen
        seen, uniq = set(), []
        for qid, u in ids:
            if qid not in seen:
                uniq.append((qid, u))
                seen.add(qid)
        return uniq

    if playlist == "cup":
        c_slug = os.getenv("METACULUS_COLLECTION_SLUG_CUP", "").strip()
        return _metaculus_fetch_open_ids_for_collection(c_slug) if c_slug else []

    # default: tournament
    t_slug = os.getenv("METACULUS_COLLECTION_SLUG_TOURNAMENT", "").strip()
    return _metaculus_fetch_open_ids_for_collection(t_slug) if t_slug else []

async def main_async(mode: str, limit: int = 0):
    ts = ist_stamp()
    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    run_log_path = os.path.join(RUN_LOG_DIR, f"{spagbot_safe_filename('spagbot_run_'+ts)}.txt") if 'spagbot_safe_filename' in globals() else os.path.join(RUN_LOG_DIR, f"spagbot_run_{ts}.txt")

    # ANCHOR: "playlist selection"
    # Determine playlist from env (set by CLI --playlist or workflow RUN_PLAYLIST)
    playlist = os.getenv("RUN_PLAYLIST", "tournament")
    ids = get_playlist_question_ids(playlist)
    if not ids:
        print(f"[WARN] No open questions found for playlist='{playlist}'. "
                f"Check collection slugs and token.")
        return

        # If you already have a function that iterates over questions, adapt:
        # Here we build a minimal iterator that matches your forecast loop:
        questions = []
        for qid, url in ids:
            # minimal shape needed downstream
            questions.append({
                "id": qid,
                "url": url,
                "type": "binary",  # will be overwritten by fetch_post_details if needed
                "title": None,     # lazy-fill later
            })

    fh = logging.FileHandler(run_log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    class Tee:
        def __init__(self, path):
            self.f = open(path, "a", encoding="utf-8")
        def write(self, s: str):
            try:
                sys.__stdout__.write(s)
            except Exception:
                pass
            try:
                self.f.write(s)
            except Exception:
                pass
        def flush(self):
            try:
                sys.__stdout__.flush()
            except Exception:
                pass
            try:
                self.f.flush()
            except Exception:
                pass
        def close(self):
            try:
                self.f.close()
            except Exception:
                pass

    tee = Tee(run_log_path)
    sys.stdout = tee
    sys.stderr = tee

    try:
        print(f"{PRINT_PREFIX} Spagbot ensemble starting…")
        os.makedirs(FORECAST_LOG_DIR, exist_ok=True)

        if mode == "test_questions":
            open_qs = [(578, 578), (14333, 14333), (22427, 22427), (38880, 38880)]
        else:
            open_qs = get_open_question_ids_from_tournament()

        if is_mini_bench():
            already = _load_one_shot_ids()
            if already:
                before = len(open_qs)
                open_qs = [(q, p) for (q, p) in open_qs if q not in already]
                skipped = before - len(open_qs)
                if skipped > 0:
                    print(f"Mini-Bench one-shot guard: skipping {skipped} already-submitted question(s).")

        if isinstance(limit, int) and limit > 0:
            open_qs = open_qs[:limit]
            print(f"\n⚡ Limiter active: running only the first {limit} question(s).")

        run_id = ts
        detailed_log_filename = os.path.join(FORECAST_LOG_DIR, f"{run_id}_reasoning.log")
        with open(detailed_log_filename, "w", encoding="utf-8") as detailed_log_f:
            detailed_log_f.write(f"📊 Forecast Run {run_id}\n")
            detailed_log_f.write(f"Timestamp: {ist_iso()}\n")
            detailed_log_f.write("=" * 60 + "\n")

            for qid, pid in open_qs:
                try:
                    await forecast_one(qid, pid, run_id=run_id, detailed_log_handle=detailed_log_f)
                    try:
                        detailed_log_f.flush()
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[Error on qid={qid} pid={pid}] {e}")
                    try:
                        detailed_log_f.write("\n" + ("-" * 60) + "\n")
                        detailed_log_f.write(f"❌ FAILED Question (qid={qid}, pid={pid})\n")
                        detailed_log_f.write(f"Error: {e}\n")
                        detailed_log_f.write("No detailed reasoning captured due to exception.\n")
                        detailed_log_f.flush()
                    except Exception:
                        pass

        print("\n✅ forecasts.csv updated")
        print("✅ forecasts_by_model.csv updated")
        print(f"✅ Detailed reasoning log created: {detailed_log_filename}")

    finally:
        try:
            tee.flush()
            tee.close()
        except Exception:
            pass
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def main():
    """
    CLI entrypoint for Spagbot.
    Parses CLI args, sets environment flags for downstream functions,
    and dispatches to the async main loop.
    """
    parser = argparse.ArgumentParser(description="Run Spagbot (Ensemble + GTMC1 + Bayes-MC).")

    # Required/standard args
    parser.add_argument(
        "--mode",
        type=str,
        default="test_questions",
        choices=["test_questions", "run"],
        help="Which overall run mode to use. 'test_questions' runs the built-in test set; 'run' performs a full run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions to run (0 = no limit; run all).",
    )
    parser.add_argument(
        "--fresh-research",
        action="store_true",
        help="Ignore cached research and fetch anew for every question.",
    )

    # ANCHOR: "playlist CLI"
    # New optional selector; defaults to env RUN_PLAYLIST (else 'tournament').
    # This *only* sets an environment variable that downstream code can read.
    parser.add_argument(
        "--playlist",
        choices=["tournament", "cup", "all"],
        default=os.getenv("RUN_PLAYLIST", "tournament"),
        help="Which question pool to run against. Overrides RUN_PLAYLIST env if provided.",
    )

    args = parser.parse_args()

    # Resolve the playlist, print it for visibility, and expose via env
    playlist = getattr(args, "playlist", os.getenv("RUN_PLAYLIST", "tournament"))
    print(f"Playlist: {playlist}")
    os.environ["RUN_PLAYLIST"] = playlist  # make available to all called functions

    # Optional: disable research cache if explicitly requested
    if args.fresh_research:
        os.environ["SPAGBOT_DISABLE_RESEARCH_CACHE"] = "1"
        global DISABLE_RESEARCH_CACHE
        DISABLE_RESEARCH_CACHE = True

    # Dispatch to async entrypoint (signature unchanged)
    asyncio.run(main_async(args.mode, args.limit))

if __name__ == "__main__":
    main()
    print("🚀 Spagbot run complete")
