from __future__ import annotations
"""
cli.py — Spagbot runner (unified CSV + ablation support)

WHAT THIS FILE DOES (high level, in plain English)
--------------------------------------------------
- Talks to Metaculus to fetch questions (either your test IDs or a tournament list).
- For each question:
  1) Runs the RESEARCH step to build a compact research brief.
  2) Classifies the question (primary/secondary topic + "strategic?" score).
     - If it's strategic *and* the question is binary, we try GTMC1.
  3) Builds a forecasting prompt and asks each LLM model in your ensemble for a forecast.
  4) Aggregates model outputs with a Bayesian Monte Carlo layer ("BMC"); optionally fuses GTMC1 for binary.
  5) Records *everything* into ONE wide CSV row via io_logs.write_unified_row(...).
  6) (Optional) Submits the forecast to Metaculus if --submit is used.

- Additionally, it runs an **ablation** pass ("no-research") so you can quantify the
  value of the research component. Those results are logged into dedicated CSV columns.

- It also logs three ensemble **variants** for diagnostics:
  (a) no_gtmc1            → BMC aggregation without the GTMC1 signal,
  (b) uniform_weights     → treat all LLMs equally,
  (c) no_bmc_no_gtmc1     → a very simple average of model outputs (no BMC, no GTMC1).
"""

import argparse
import asyncio
import json
import os
import re
import time
from contextlib import ExitStack
from typing import Optional, List, Dict, Any, Tuple
import inspect

import numpy as np
import requests

import json
from pathlib import Path


def _safe_json_load(s: str):
    try:
        import json as _json
        return _json.loads(s)
    except Exception:
        return None


def _as_dict(x):
    """
    Normalize possibly-string/None payloads into dicts.
    - dict -> dict
    - JSON string -> parsed dict (if possible)
    - anything else -> {}
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            obj = _safe_json_load(s)
            if isinstance(obj, dict):
                return obj
    return {}


# ---- Spagbot internals (all relative imports) --------------------------------
from .config import (
    TOURNAMENT_ID, AUTH_HEADERS, API_BASE_URL,
    ist_iso, ist_stamp, SUBMIT_PREDICTION, METACULUS_HTTP_TIMEOUT,
)
from .prompts import build_binary_prompt, build_numeric_prompt, build_mcq_prompt
from .providers import DEFAULT_ENSEMBLE, _get_or_client, llm_semaphore
from .ensemble import EnsembleResult, MemberOutput, run_ensemble_binary, run_ensemble_mcq, run_ensemble_numeric
from .aggregate import aggregate_binary, aggregate_mcq, aggregate_numeric
from .research import run_research_async

# --- Corrected seen_guard import ---
try:
    from . import seen_guard
except ImportError as e:
    print(f"[warn] seen_guard not available ({e!r}); continuing without duplicate protection.")
    seen_guard = None

# Robust import for topic_classify: prefer package module; fall back to repo root
try:
    from .topic_classify import should_run_gtmc1  # expected location
except Exception:
    try:
        from topic_classify import should_run_gtmc1  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import 'topic_classify'. Move topic_classify.py into 'spagbot/' "
            "or keep it at repo root (this file supports both)."
        ) from e

from . import GTMC1

# --- seen_guard import shim (ensures a callable filter_unseen_posts exists) ---
try:
    try:
        # When cli.py is executed as a module
        from .seen_guard import SeenGuard  # type: ignore
    except Exception:
        # When cli.py is executed as a script from repo root
        from seen_guard import SeenGuard  # type: ignore

    _sg = SeenGuard()

    def filter_unseen_posts(posts):
        # Adapter to old call-site name; calls the actual class method.
        return _sg.filter_fresh_posts(posts)

except Exception as e:
    print(f"[seen_guard] disabled ({e}); processing all posts returned.")
    def filter_unseen_posts(posts):
        return posts
# --- end seen_guard import shim ---



# Unified CSV helpers (single file)
from .io_logs import ensure_unified_csv, write_unified_row, write_human_markdown, finalize_and_commit

# --------------------------------------------------------------------------------
# Small utility helpers (safe JSON, timing, clipping, etc.)
# --------------------------------------------------------------------------------

# --- SeenGuard wiring (robust to different shapes/APIs) -----------------------
def _load_seen_guard():
    """
    Try to load a SeenGuard instance from seen_guard.py in a robust way.
    Will look for common instance names and fall back to constructing SeenGuard.
    Returns: guard instance or None
    """
    try:
        import seen_guard as sg_mod
    except Exception:
        return None

    # Prefer a ready-made instance exported from the module
    for attr in ("_GUARD", "GUARD", "guard"):
        guard = getattr(sg_mod, attr, None)
        if guard is not None:
            return guard

    # Fallback: instantiate if class is available
    try:
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None:
            cooldown = int(os.getenv("SEEN_COOLDOWN_HOURS", "24"))
            path = os.getenv("SEEN_GUARD_PATH", "forecast_logs/state/seen_forecasts.jsonl")
            return SG(Path(path), cooldown_hours=cooldown)
    except Exception:
        pass

    return None


def _apply_seen_guard(guard, posts):
    """
    Call the first matching method on guard to filter posts.
    Accepts either a return of (posts, dup_count) or just posts.
    """
    if not guard or not posts:
        return posts, 0

    candidates = [
        "filter_fresh_posts",
        "filter_unseen_posts",
        "filter_posts",
        "filter_recent_posts",
        "filter_new_posts",
        "filter",  # very generic, last
    ]

    last_err = None
    for name in candidates:
        if hasattr(guard, name):
            fn = getattr(guard, name)
            try:
                # Try simple positional call
                result = fn(posts)
            except TypeError:
                # Try kwargs form if implemented that way
                try:
                    result = fn(posts=posts)
                except Exception as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

            # Normalize return
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                return result
            if isinstance(result, list):
                return result, 0

            # Unexpected return shape; treat as no-op
            return posts, 0

    # If we got here, no callable matched or all failed
    if last_err:
        raise last_err
    return posts, 0
# ----------------------------------------------------------------------------- 

# Time in milliseconds since start_time
def _ms(start_time: float) -> int:
    return int(round((time.time() - start_time) * 1000))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _sanitize_markdown_chunks(chunks: List[Any]) -> List[str]:
    """Return a list of strings suitable for ``"\n\n".join(...)``.

    The markdown builder collects many diagnostic entries, some of which
    originate from optional integrations (GTMC1 raw dumps, prediction-market
    lookups, etc.).  When any of those helpers return ``None`` we previously
    propagated the ``None`` directly into the markdown list.  Later, when we
    attempted to join the chunks we hit ``TypeError: sequence item X: expected
    str instance, NoneType found``.  This helper drops ``None`` entries and
    coerces any remaining values to strings so the join is always safe.
    """

    sanitized: List[str] = []
    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, str):
            sanitized.append(chunk)
            continue
        try:
            sanitized.append(str(chunk))
        except Exception:
            # If ``str(chunk)`` itself fails we silently drop the entry; the
            # surrounding debug output already makes it clear something odd
            # happened, and failing to write the human log is worse.
            continue
    return sanitized

def _maybe_dump_raw_gtmc1(content: str, *, run_id: str, question_id: int) -> Optional[str]:
    """
    If SPAGBOT_DEBUG_RAW=1, write the raw LLM JSON-ish text we received for the
    GTMC1 actor table to a file in gtmc_logs/ and return the path. Otherwise None.
    """
    if os.getenv("SPAGBOT_DEBUG_RAW", "0") != "1":
        return None
    try:
        os.makedirs("gtmc_logs", exist_ok=True)
        path = os.path.join("gtmc_logs", f"{run_id}_q{question_id}_actors_raw.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    except Exception:
        return None

# --------------------------------------------------------------------------------
# Robust wrapper for the classifier: supports both sync and async implementations
# --------------------------------------------------------------------------------
async def _run_classifier_safe(title: str, description: str, criteria: str, *, slug: str):
    """
    Calls should_run_gtmc1(...) whether it's sync or async, and normalizes outputs.
    Returns (use_gtmc1: bool, cls_info: dict).
    """
    import asyncio as _aio

    def _dictify(obj):
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            d = _safe_json_load(obj)
            return d if isinstance(d, dict) else {}
        return {}

    try:
        # Handle both async and sync classifier implementations + older signatures
        if _aio.iscoroutinefunction(should_run_gtmc1):
            try:
                res = await should_run_gtmc1(title, description, criteria, slug=slug)
            except TypeError:
                res = await should_run_gtmc1(title, description, criteria)
        else:
            try:
                res = should_run_gtmc1(title, description, criteria, slug=slug)
            except TypeError:
                res = should_run_gtmc1(title, description, criteria)
    except Exception:
        # On any classifier failure, fall back safely
        return False, {}

    # Normalize shapes:
    # (a) tuple -> (flag, info)
    if isinstance(res, tuple) and len(res) == 2:
        use_flag = bool(res[0])
        info = _dictify(res[1])
        return use_flag, info

    # (b) dict -> infer flag from field
    if isinstance(res, dict):
        return bool(res.get("is_strategic", False)), res

    # (c) JSON string -> parse
    if isinstance(res, str):
        info = _dictify(res)
        return bool(info.get("is_strategic", False)), info

    # Unknown shape -> safe default
    return False, {}

# --------------------------------------------------------------------------------
# Calibration weights loader (optional). You periodically run update_calibration.py
# to produce calibration_weights.json, which we use here to weight models per class.
# --------------------------------------------------------------------------------

def _load_calibration_weights() -> Dict[str, Any]:
    path = os.getenv("CALIB_WEIGHTS_PATH", "calibration_weights.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _choose_weights_for_question(calib: Dict[str, Any], class_primary: str, qtype: str) -> Tuple[Dict[str, float], str]:
    model_names = [ms.name for ms in DEFAULT_ENSEMBLE]
    # 1) class-conditional
    try:
        by_class = calib.get("by_class", {})
        w = by_class.get(class_primary or "", {}).get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"class_conditional:{class_primary}:{qtype}"
    except Exception:
        pass
    # 2) global
    try:
        glob = calib.get("global", {})
        w = glob.get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"global:{qtype}"
    except Exception:
        pass
    # 3) uniform
    return ({m: 1.0 for m in model_names}, "uniform")

# --------------------------------------------------------------------------------
# Metaculus API helpers (GET posts, build payloads, POST forecasts)
# --------------------------------------------------------------------------------

def list_posts_from_tournament(tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50) -> dict:
    params = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(["binary", "multiple_choice", "numeric", "discrete"]),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    r = requests.get(url, params=params, timeout=METACULUS_HTTP_TIMEOUT, **AUTH_HEADERS)
    if not r.ok:
        raise RuntimeError(r.text)
    return json.loads(r.content)

def get_post_details(post_id: int) -> dict:
    url = f"{API_BASE_URL}/posts/{post_id}/"
    r = requests.get(url, timeout=METACULUS_HTTP_TIMEOUT, **AUTH_HEADERS)
    if not r.ok:
        raise RuntimeError(r.text)
    return json.loads(r.content)

def _get_possibilities(q: dict) -> dict:
    return (q.get("possibilities") or q.get("range") or {})

def _get_options_list(q: dict) -> List[str]:
    if isinstance(q.get("options"), list):
        out = []
        for opt in q["options"]:
            if isinstance(opt, dict):
                out.append(str(opt.get("label") or opt.get("name") or ""))
            else:
                out.append(str(opt))
        return out
    poss = _get_possibilities(q)
    if isinstance(poss.get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["options"]]
    if isinstance(poss.get("scale", {}).get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["scale"]["options"]]
    return []

def _is_discrete(q: dict) -> bool:
    poss = _get_possibilities(q)
    q_type = (poss.get("type") or q.get("type") or "").lower()
    if q_type == "discrete":
        return True
    if q_type == "numeric" and isinstance(poss.get("scale", {}).get("values"), list):
        return True
    return False

def _discrete_values(q: dict) -> List[float]:
    poss = _get_possibilities(q)
    values = poss.get("scale", {}).get("values") or poss.get("values")
    if not values:
        return []
    return [float(v) for v in values]

def _build_payload_for_submission(question_type: str, forecast: Any) -> dict:
    if question_type == "binary":
        return {"probability_yes": float(forecast), "probability_yes_per_category": None, "continuous_cdf": None}
    if question_type == "multiple_choice":
        if isinstance(forecast, dict):
            return {"probability_yes": None, "probability_yes_per_category": forecast, "continuous_cdf": None}
        elif isinstance(forecast, list):
            return {"probability_yes": None, "probability_yes_per_category": {str(i): float(p) for i, p in enumerate(forecast)}, "continuous_cdf": None}
        else:
            raise ValueError("MCQ forecast must be list or dict")
    return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": forecast}

def post_forecast(question_id: int, payload: dict) -> Tuple[int, Optional[str]]:
    url = f"{API_BASE_URL}/questions/forecast/"
    r = requests.post(
        url,
        json=[{"question": question_id, **payload}],
        timeout=METACULUS_HTTP_TIMEOUT,
        **AUTH_HEADERS,
    )
    if r.ok:
        return r.status_code, None
    return r.status_code, r.text

# --------------------------------------------------------------------------------
# Simple, no-BMC fallback aggregators for the diagnostic variant "no_bmc_no_gtmc1"
# --------------------------------------------------------------------------------

def _simple_average_binary(members: List[MemberOutput]) -> Optional[float]:
    vals = [float(m.parsed) for m in members if m.ok and isinstance(m.parsed, (int, float))]
    if not vals:
        return None
    return float(np.mean([_clip01(v) for v in vals]))

def _simple_average_mcq(members: List[MemberOutput], n_opts: int) -> Optional[List[float]]:
    vecs: List[List[float]] = []
    for m in members:
        if m.ok and isinstance(m.parsed, list) and len(m.parsed) == n_opts:
            v = np.asarray(m.parsed, dtype=float)
            v = np.clip(v, 0.0, 1.0)
            s = float(v.sum())
            if s > 0:
                vecs.append((v / s).tolist())
    if not vecs:
        return None
    mean = np.mean(np.asarray(vecs), axis=0)
    mean = np.clip(mean, 1e-9, 1.0)
    mean = mean / float(mean.sum())
    return mean.tolist()

def _simple_average_numeric(members: List[MemberOutput]) -> Optional[Dict[str, float]]:
    p10s, p50s, p90s = [], [], []
    for m in members:
        if m.ok and isinstance(m.parsed, dict):
            d = m.parsed
            if "P10" in d and "P90" in d:
                p10s.append(float(d["P10"]))
                p90s.append(float(d["P90"]))
                p50s.append(float(d.get("P50", 0.5*(float(d["P10"]) + float(d["P90"])))))
    if not p10s:
        return None
    return {
        "P10": float(np.mean(p10s)),
        "P50": float(np.mean(p50s)) if p50s else 0.5 * (float(np.mean(p10s)) + float(np.mean(p90s))),
        "P90": float(np.mean(p90s)),
    }

# --------------------------------------------------------------------------------
# Core orchestration for ONE question → produce a single CSV row
# --------------------------------------------------------------------------------

async def _run_one_question_body(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    submit_ok: bool,
    calib: Dict[str, Any],
    seen_guard_state: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
) -> None:
    t_start_total = time.time()
    _post_original = post
    try:
    
        post = _as_dict(post)
        q = _as_dict(post.get("question") if isinstance(post, dict) else {})

        post_id = int((post.get("id") or post.get("post_id") or 0) or 0)
        question_id = int((q.get("id") or 0) or 0)
    
        seen_guard_enabled = bool(seen_guard_state.get("enabled", False))
        seen_guard_lock_acquired = seen_guard_state.get("lock_acquired")
        seen_guard_lock_error = str(seen_guard_state.get("lock_error") or "")
        
        title = str(q.get("title") or post.get("title") or "").strip()
        url = f"https://www.metaculus.com/questions/{question_id}/" if question_id else ""
        qtype = (q.get("type") or "binary").strip()
        description = str(post.get("description") or q.get("description") or "")
        criteria = str(q.get("resolution_criteria") or q.get("fine_print") or q.get("resolution") or "")
        units = q.get("unit") or q.get("units") or ""
        tournament_id = post.get("tournaments") or q.get("tournaments") or TOURNAMENT_ID
    
        # Options / discrete values
        options = _get_options_list(q)
        n_options = len(options) if qtype == "multiple_choice" else 0
        discrete_values = _discrete_values(q) if qtype in ("numeric", "discrete") and _is_discrete(q) else []
    
        # ------------------ 1) Research step (LLM brief + sources appended) ---------
        t0 = time.time()
        research_text, research_meta = await run_research_async(
            title=title,
            description=description,
            criteria=criteria,
            qtype=qtype,
            options=options if qtype == "multiple_choice" else None,
            units=str(units) if units else None,
            slug=f"q{question_id}",
        )
    
        t_research_ms = _ms(t0)
    
    
        # ------------------ 2) Topic/strategic classification (for GTMC1 gate) -----
        use_gtmc1, cls_info = await _run_classifier_safe(title, description, criteria, slug=f"q{question_id}")
        cls_info = _as_dict(cls_info)
        class_primary = (cls_info or {}).get("primary") or ""
        class_secondary = (cls_info or {}).get("secondary") or ""
        is_strategic = bool((cls_info or {}).get("is_strategic", False))
        strategic_score = float((cls_info or {}).get("strategic_score", 0.0))
        classifier_source = (cls_info or {}).get("source") or ""
        classifier_rationale = (cls_info or {}).get("rationale") or ""
        classifier_cost = float(cls_info.get("cost_usd", 0.0) or 0.0)
    
        # ------------------ 3) Optional GTMC1 (binary + strategic) ------------------
        gtmc1_active = bool(use_gtmc1 and qtype == "binary")
        actors_table: Optional[List[Dict[str, Any]]] = None
        gtmc1_signal: Dict[str, Any] = {}
        gtmc1_policy_sentence: str = ""
        t_gtmc1_ms = 0
    
        # Raw-dump debugging fields (only populated on failure / deactivation)
        gtmc1_raw_dump_path: str = ""
        gtmc1_raw_excerpt: str = ""
        gtmc1_raw_reason: str = ""
    
        if gtmc1_active:
            try:
                from .config import OPENROUTER_FALLBACK_ID
                client = _get_or_client()
                if client is None:
                    gtmc1_active = False
                else:
                    prompt = f"""You are a research analyst preparing inputs for a Bruce Bueno de Mesquita-style
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
                    t_gt0 = time.time()
                    async with llm_semaphore:
                        resp = await client.chat.completions.create(
                            model=OPENROUTER_FALLBACK_ID,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                    text = (resp.choices[0].message.content or "").strip()
                    raw_text_for_debug = text  # keep exactly what the LLM sent
                    try:
                        data = json.loads(re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S))
                    except Exception:
                        data = {}
                        gtmc1_active = False
                        gtmc1_raw_reason = "json_parse_error"
                        # Dump raw if requested; otherwise keep a short excerpt for the human log
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            # Use the same limit used elsewhere for model raw content
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
    
                    actors = data.get("actors") or []
                    gtmc1_policy_sentence = str(data.get("policy_continuum") or "").strip()
                    cleaned: List[Dict[str, Any]] = []
                    for a in actors:
                        try:
                            nm = str(a.get("name") or "").strip()
                            pos = float(a.get("position")); cap = float(a.get("capability"))
                            sal = float(a.get("salience")); thr = float(a.get("risk_threshold"))
                            if not nm: continue
                            if not (0.0 <= pos <= 100.0): continue
                            if not (0.0 <= cap <= 100.0): continue
                            if not (0.0 <= sal <= 100.0): continue
                            if not (0.0 <= thr <= 0.10): continue
                            cleaned.append({
                                "name": nm, "position": pos, "capability": cap,
                                "salience": sal, "risk_threshold": thr
                            })
                        except Exception:
                            continue
                    if len(cleaned) >= 3:
                        actors_table = cleaned
                        gtmc1_signal, _df_like = await asyncio.to_thread(
                            GTMC1.run_monte_carlo_from_actor_table,
                            actor_rows=actors_table,
                            num_runs=60,
                            log_dir="gtmc_logs",
                            run_slug=f"q{question_id}",
                        )
                    else:
                        gtmc1_active = False
                        gtmc1_raw_reason = "actors_lt_3"
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
            except Exception:
                gtmc1_active = False
                t_gtmc1_ms = 0
    
        # ------------------ 4) Build main prompts (WITH research) -------------------
        if qtype == "binary":
            main_prompt = build_binary_prompt(title, description, research_text, criteria)
        elif qtype == "multiple_choice":
            main_prompt = build_mcq_prompt(title, options, description, research_text, criteria)
        else:
            main_prompt = build_numeric_prompt(title, str(units or ""), description, research_text, criteria)
    
        # ------------------ 5) Ensemble calls (WITH research) -----------------------
        t0 = time.time()
        if qtype == "binary":
            ens_res = await run_ensemble_binary(main_prompt, DEFAULT_ENSEMBLE)
        elif qtype == "multiple_choice":
            ens_res = await run_ensemble_mcq(main_prompt, n_options, DEFAULT_ENSEMBLE)
        else:
            ens_res = await run_ensemble_numeric(main_prompt, DEFAULT_ENSEMBLE)
        t_ensemble_ms = _ms(t0)
    
        # ------------------ 6) Choose calibration weights & aggregate ---------------
        calib_weights_map, weights_profile = _choose_weights_for_question(
            _load_calibration_weights(), class_primary=class_primary, qtype=qtype
        )
    
        # MAIN aggregation (with optional GTMC1 for binary)
        if qtype == "binary":
            final_main, bmc_summary = aggregate_binary(ens_res, gtmc1_signal if gtmc1_active else None, calib_weights_map)
        elif qtype == "multiple_choice":
            vec_main, bmc_summary = aggregate_mcq(ens_res, n_options, calib_weights_map)
            final_main = {options[i]: vec_main[i] for i in range(n_options)} if n_options else {}
        else:
            quantiles_main, bmc_summary = aggregate_numeric(ens_res, calib_weights_map)
            final_main = dict(quantiles_main)
    
        # ------------------ 7) Diagnostic variants (WITH research) ------------------
        if qtype == "binary":
            v_nogtmc1, _ = aggregate_binary(ens_res, None, calib_weights_map)
            v_uniform, _ = aggregate_binary(ens_res, gtmc1_signal if gtmc1_active else None, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_simple = _simple_average_binary(ens_res.members)
        elif qtype == "multiple_choice":
            v_nogtmc1_vec, _ = aggregate_mcq(ens_res, n_options, calib_weights_map)
            v_nogtmc1 = {options[i]: v_nogtmc1_vec[i] for i in range(n_options)} if n_options else {}
            v_uniform_vec, _ = aggregate_mcq(ens_res, n_options, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_uniform = {options[i]: v_uniform_vec[i] for i in range(n_options)} if n_options else {}
            v_simple_vec = _simple_average_mcq(ens_res.members, n_options)
            v_simple = {options[i]: v_simple_vec[i] for i in range(n_options)} if (n_options and v_simple_vec) else {}
        else:
            v_nogtmc1, _ = aggregate_numeric(ens_res, calib_weights_map)
            v_uniform, _ = aggregate_numeric(ens_res, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            v_simple = _simple_average_numeric(ens_res.members) or {}
    
        # ------------------ 8) Ablation pass: NO RESEARCH ---------------------------
        if qtype == "binary":
            ab_prompt = build_binary_prompt(title, description, "", criteria)
            ens_res_ab = await run_ensemble_binary(ab_prompt, DEFAULT_ENSEMBLE)
            ab_main, _ = aggregate_binary(ens_res_ab, None, calib_weights_map)
            ab_uniform, _ = aggregate_binary(ens_res_ab, None, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_simple = _simple_average_binary(ens_res_ab.members)
        elif qtype == "multiple_choice":
            ab_prompt = build_mcq_prompt(title, options, description, "", criteria)
            ens_res_ab = await run_ensemble_mcq(ab_prompt, n_options, DEFAULT_ENSEMBLE)
            ab_vec, _ = aggregate_mcq(ens_res_ab, n_options, calib_weights_map)
            ab_main = {options[i]: ab_vec[i] for i in range(n_options)} if n_options else {}
            ab_uniform_vec, _ = aggregate_mcq(ens_res_ab, n_options, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_uniform = {options[i]: ab_uniform_vec[i] for i in range(n_options)} if n_options else {}
            ab_simple_vec = _simple_average_mcq(ens_res_ab.members, n_options)
            ab_simple = {options[i]: ab_simple_vec[i] for i in range(n_options)} if (n_options and ab_simple_vec) else {}
        else:
            ab_prompt = build_numeric_prompt(title, str(units or ""), description, "", criteria)
            ens_res_ab = await run_ensemble_numeric(ab_prompt, DEFAULT_ENSEMBLE)
            ab_main, _ = aggregate_numeric(ens_res_ab, calib_weights_map)
            ab_uniform, _ = aggregate_numeric(ens_res_ab, {m.name: 1.0 for m in DEFAULT_ENSEMBLE})
            ab_simple = _simple_average_numeric(ens_res_ab.members) or {}
    
        # ------------------ 9) Submission (optional) --------------------------------
        submit_status_code = ""
        submit_error = ""
        explanation_short = ""
        t_submit_ms = 0
    
        if submit_ok and question_id:
            t_sub0 = time.time()
            try:
                if qtype == "binary" and isinstance(final_main, float):
                    payload = _build_payload_for_submission("binary", _clip01(final_main))
                    code, err = post_forecast(question_id, payload)
                    submit_status_code = str(code)
                    submit_error = "" if err is None else err[:280]
    
                elif qtype == "multiple_choice" and isinstance(final_main, dict):
                    probs = [float(final_main.get(lbl, 0.0)) for lbl in options]
                    s = sum(probs)
                    if s <= 0 and options:
                        probs = [1.0 / len(options)] * len(options)
                    elif s > 0:
                        probs = [p / s for p in probs]
                    label_map = {str(options[i]): float(probs[i]) for i in range(len(options))}
                    payload = _build_payload_for_submission("multiple_choice", label_map)
                    code, err = post_forecast(question_id, payload)
                    submit_status_code = str(code)
                    submit_error = "" if err is None else err[:280]
    
                elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
                    pass
            except Exception as e:
                submit_status_code = submit_status_code or "EXC"
                submit_error = (str(e) or "")[:280]
    
            # Console confirmation of submit
            if submit_status_code:
                if submit_status_code.isdigit() and int(submit_status_code) == 201:
                    print("Submit: 201 Created ✅")
                else:
                    print(f"Submit: {submit_status_code} ❌ {submit_error[:120]}")
            t_submit_ms = _ms(t_sub0)
    
        # ------------------ 10) Build ONE wide CSV row and write it -----------------
        ensure_unified_csv()
    
        row: Dict[str, Any] = {
            # Run metadata
            "run_id": run_id,
            "run_time_iso": ist_iso(),
            "purpose": purpose,
            "git_sha": os.getenv("GIT_SHA", ""),
            "config_profile": "default",
            "weights_profile": "class_calibration",
            "openrouter_models_json": [
                {"name": ms.name, "provider": ms.provider, "model_id": ms.model_id, "weight": ms.weight}
                for ms in DEFAULT_ENSEMBLE
            ],
    
            # Question metadata
            "question_id": str(question_id),
            "question_url": url,
            "question_title": title,
            "question_type": qtype,
            "tournament_id": tournament_id if isinstance(tournament_id, str) else str(tournament_id),
            "created_time_iso": post.get("creation_time") or q.get("creation_time") or "",
            "closes_time_iso": post.get("close_time") or q.get("close_time") or "",
            "resolves_time_iso": post.get("scheduled_resolve_time") or q.get("scheduled_resolve_time") or "",
    
            # Classification
            "class_primary": class_primary,
            "class_secondary": class_secondary or "",
            "is_strategic": str(is_strategic),
            "strategic_score": f"{strategic_score:.3f}",
            "classifier_source": classifier_source,
            "classifier_rationale": classifier_rationale,
    
            # Research
            "research_llm": research_meta.get("research_llm", ""),
            "research_source": research_meta.get("research_source", ""),
            "research_query": research_meta.get("research_query", ""),
            "research_n_raw": str(research_meta.get("research_n_raw", "")),
            "research_n_kept": str(research_meta.get("research_n_kept", "")),
            "research_cached": research_meta.get("research_cached", ""),
            "research_error": research_meta.get("research_error", ""),
    
    
            # Options/values
            "n_options": str(n_options if qtype == "multiple_choice" else 0),
            "options_json": options if qtype == "multiple_choice" else "",
            "discrete_values_json": discrete_values if (qtype in ("numeric", "discrete") and discrete_values) else "",
        }
    
        row["seen_guard_triggered"] = (
            "1"
            if seen_guard_enabled and bool(seen_guard_lock_acquired)
            else ("0" if seen_guard_enabled else "")
        )
    
        # Per-model outputs
        for i, ms in enumerate(DEFAULT_ENSEMBLE):
            mo: Optional[MemberOutput] = None
            if isinstance(ens_res, EnsembleResult) and i < len(ens_res.members):
                mo = ens_res.members[i]
    
            ok = bool(mo and mo.ok)
            row[f"model_ok__{ms.name}"] = "1" if ok else "0"
            row[f"model_time_ms__{ms.name}"] = str(getattr(mo, "elapsed_ms", 0) or "")
    
            if ok and mo is not None:
                if qtype == "binary" and isinstance(mo.parsed, (float, int)):
                    row[f"binary_prob__{ms.name}"] = f"{_clip01(float(mo.parsed)):.6f}"
                elif qtype == "multiple_choice" and isinstance(mo.parsed, list):
                    row[f"mcq_json__{ms.name}"] = mo.parsed
                elif qtype in ("numeric", "discrete") and isinstance(mo.parsed, dict):
                    p10 = _safe_float(mo.parsed.get("P10"))
                    p50 = _safe_float(mo.parsed.get("P50"))
                    p90 = _safe_float(mo.parsed.get("P90"))
                    if p10 is not None: row[f"numeric_p10__{ms.name}"] = f"{p10:.6f}"
                    if p50 is not None: row[f"numeric_p50__{ms.name}"] = f"{p50:.6f}"
                    if p90 is not None: row[f"numeric_p90__{ms.name}"] = f"{p90:.6f}"
    
            row[f"cost_usd__{ms.name}"] = f"{getattr(mo,'cost_usd',0.0):.6f}" if mo else ""
    
        # Ensemble (main)
        if qtype == "binary" and isinstance(final_main, float):
            row["binary_prob__ensemble"] = f"{_clip01(final_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(final_main, dict):
            row["mcq_json__ensemble"] = final_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble"] = f"{_clip01(float(final_main.get(options[j], 0.0))):.6f}"
        elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in final_main:
                    row[f"numeric_{k.lower()}__ensemble"] = f"{float(final_main[k]):.6f}"
    
        # Variants (WITH research)
        def _fill_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_variant("no_gtmc1", v_nogtmc1)
        _fill_variant("uniform_weights", v_uniform)
        if qtype == "binary":
            _fill_variant("no_bmc_no_gtmc1", v_simple)  # float for binary
        else:
            _fill_variant("no_bmc_no_gtmc1", v_simple if isinstance(v_simple, dict) else v_simple)
    
        # Ablation (NO research)
        row["ablation_no_research"] = "1"
        if qtype == "binary" and isinstance(ab_main, float):
            row["binary_prob__ensemble_no_research"] = f"{_clip01(ab_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(ab_main, dict):
            row["mcq_json__ensemble_no_research"] = ab_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble_no_research"] = f"{_clip01(float(ab_main.get(options[j], 0.0))):.6f}"
        elif qtype in ("numeric", "discrete") and isinstance(ab_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in ab_main:
                    row[f"numeric_{k.lower()}__ensemble_no_research"] = f"{float(ab_main[k]):.6f}"
    
        def _fill_ablation_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_no_research_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_no_research_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_no_research_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_ablation_variant("no_gtmc1", ab_main)
        _fill_ablation_variant("uniform_weights", ab_uniform)
        _fill_ablation_variant("no_bmc_no_gtmc1", ab_simple if isinstance(ab_simple, dict) else ({"P50": ab_simple} if isinstance(ab_simple, float) else ab_simple))
    
        # Diagnostics, timings, submission, weights used
        row.update({
            "gtmc1_active": "1" if gtmc1_active else "0",
            "actors_cached": "0",
            "gtmc1_actor_count": str(len(actors_table) if actors_table else 0),
            "gtmc1_coalition_rate": (gtmc1_signal.get("coalition_rate") if gtmc1_signal else ""),
            "gtmc1_exceedance_ge_50": (gtmc1_signal.get("exceedance_ge_50") if gtmc1_signal else ""),
            "gtmc1_dispersion": (gtmc1_signal.get("dispersion") if gtmc1_signal else ""),
            "gtmc1_median_rounds": (gtmc1_signal.get("median_rounds") if gtmc1_signal else ""),
            "gtmc1_num_runs": (gtmc1_signal.get("num_runs") if gtmc1_signal else ""),
            "gtmc1_policy_sentence": gtmc1_policy_sentence or "",
            "gtmc1_signal_json": gtmc1_signal or "",
    
            "bmc_summary_json": "",
    
            "cdf_steps_clamped": "",
            "cdf_upper_open_adjusted": "",
            "prob_sum_renormalized": "",
    
            "t_research_ms": str(t_research_ms),
            "t_ensemble_ms": str(t_ensemble_ms),
            "t_gtmc1_ms": str(t_gtmc1_ms),
            "t_submit_ms": str(t_submit_ms),
            "t_total_ms": str(_ms(t_start_total)),
    
            "explanation_short": explanation_short,
            "submit_confirm": "1" if (submit_ok and submit_status_code and submit_status_code.isdigit() and int(submit_status_code) < 400) else "0",
            "submit_status_code": submit_status_code,
            "submit_error": submit_error,
    
            "resolved": "",
            "resolved_time_iso": "",
            "resolved_outcome_label": "",
            "resolved_value": "",
            "score_brier": "",
            "score_log": "",
            "score_crps": "",
    
            "score_brier__no_research": "",
            "score_log__no_research": "",
            "score_crps__no_research": "",
    
            "weights_profile_applied": weights_profile,
            "weights_per_model_json": calib_weights_map,
            "dedupe_hash": "",
            "seen_guard_triggered": "",
        })
    
        # Human-readable markdown log
        MAX_RAW_CHARS = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS","5000"))
        RESEARCH_MAX = int(os.getenv("HUMAN_LOG_RESEARCH_MAX_CHARS","20000"))
        md = []
        md.append(f"# {title} (QID: {question_id})")
        md.append(f"- Type: {qtype}")
        md.append(f"- URL: {url}")
        md.append(f"- Classifier: {class_primary} | strategic={is_strategic} (score={strategic_score:.2f})")
        md.append("### SeenGuard")
        lock_status = "n/a"
        if seen_guard_enabled:
            lock_status = "acquired" if seen_guard_lock_acquired else "not_acquired"
        md.append(f"- enabled={seen_guard_enabled} | lock_status={lock_status}")
        if seen_guard_run_report:
            before = seen_guard_run_report.get("before")
            skipped = seen_guard_run_report.get("skipped")
            after = seen_guard_run_report.get("after")
            md.append(f"- run_filter: before={before} | skipped={skipped} | after={after}")
            if seen_guard_run_report.get("error"):
                md.append(f"- filter_error={seen_guard_run_report['error']}")
        debug_note = "lock disabled"
        if seen_guard_enabled:
            debug_note = "lock acquired" if seen_guard_lock_acquired else "lock fallback"
        if seen_guard_lock_error:
            debug_note += f" | error={seen_guard_lock_error}"
        md.append(f"- debug_note={debug_note}")
    
        md.append("## Research (summary)")
        md.append((research_text or "").strip()[:RESEARCH_MAX])
        # Research (debug)
        try:
            _r_src   = research_meta.get("research_source","")
            _r_llm   = research_meta.get("research_llm","")
            _r_q     = research_meta.get("research_query","")
            _r_raw   = research_meta.get("research_n_raw","")
            _r_kept  = research_meta.get("research_n_kept","")
            _r_cache = research_meta.get("research_cached","")
            _r_err   = research_meta.get("research_error","")
            md.append("### Research (debug)")
            _r_cost  = research_meta.get("research_cost_usd", 0.0)
            md.append(
                f"- source={_r_src} | llm={_r_llm} | cached={_r_cache} | "
                f"n_raw={_r_raw} | n_kept={_r_kept} | cost=${float(_r_cost):.6f}"
            )
    
            if _r_q:
                md.append(f"- query: {_r_q}")
            if _r_err:
                md.append(f"- error: {_r_err}")
        except Exception:
            pass
    
        # --- GTMC1 (debug) --------------------------------------------------------
        try:
            md.append("### GTMC1 (debug)")
            # Basic flags
            md.append(f"- strategic_class={is_strategic} | strategic_score={strategic_score:.2f} | source={classifier_source}")
            md.append(f"- gtmc1_active={gtmc1_active} | qtype={qtype} | t_ms={t_gtmc1_ms}")
    
            # Actor extraction outcome
            _n_actors = len(actors_table) if actors_table else 0
            md.append(f"- actors_parsed={_n_actors}")
    
            # Key Monte Carlo outputs (if any)
            _sig = gtmc1_signal or {}
            _ex = _sig.get("exceedance_ge_50")
            _coal = _sig.get("coalition_rate")
            _med = _sig.get("median_of_final_medians")
            _disp = _sig.get("dispersion")
    
            md.append(f"- exceedance_ge_50={_ex} | coalition_rate={_coal} | median={_med} | dispersion={_disp}")
            _runs_csv = _sig.get("runs_csv")
            if _runs_csv:
                md.append(f"- runs_csv={_runs_csv}")
            _meta_json = _sig.get("meta_json")
            if _meta_json:
                md.append(f"- meta_json={_meta_json}")
    
            # If GTMC1 was expected but didn’t apply, say why (best effort).
            if use_gtmc1 and qtype == "binary" and not gtmc1_active:
                md.append("- note=GTMC1 gate opened (strategic) but deactivated later (client/JSON/actors<3).")
            # If we captured raw (on failure), surface it.
            if gtmc1_raw_reason:
                md.append(f"- raw_reason={gtmc1_raw_reason}")
            if gtmc1_raw_dump_path or gtmc1_raw_excerpt:
                md.append("### GTMC1 (raw)")
                if gtmc1_raw_dump_path:
                    md.append(f"- raw_file={gtmc1_raw_dump_path}")
                if gtmc1_raw_excerpt:
                    md.append("```json")
                    md.append(gtmc1_raw_excerpt)
                    md.append("```")
        except Exception as _gtmc1_dbg_ex:
            md.append(f"- gtmc1_debug_error={type(_gtmc1_dbg_ex).__name__}: {str(_gtmc1_dbg_ex)[:200]}")
        # --------------------------------------------------------------------------
    
        # --- GTMC1 (actors used) ---------------------------------------------------
        # Show the actual table we fed into GTMC1 so you can audit inputs later.
        if gtmc1_active and actors_table:
            try:
                md.append("### GTMC1 (actors used)")
                md.append("| Actor | Position | Capability | Salience | Risk thresh |")
                md.append("|---|---:|---:|---:|---:|")
                for a in actors_table:
                    md.append(
                        f"| {a['name']} | {float(a['position']):.0f} | "
                        f"{float(a['capability']):.0f} | {float(a['salience']):.0f} | "
                        f"{float(a['risk_threshold']):.3f} |"
                    )
            except Exception as _gtmc1_tbl_ex:
                md.append(f"- actors_table_render_error={type(_gtmc1_tbl_ex).__name__}: {str(_gtmc1_tbl_ex)[:160]}")
    
        # --- Ensemble outputs (compact) --------------------------------------------
        try:
            md.append("### Ensemble (model outputs)")
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                _line = f"- {m.name}: ok={m.ok} t_ms={getattr(m,'elapsed_ms',0)}"
                if qtype == "binary" and m.ok and isinstance(m.parsed, (float, int)):
                    _line += f" p={_clip01(float(m.parsed)):.4f}"
                elif qtype == "multiple_choice" and m.ok and isinstance(m.parsed, list):
                    # just show top-3
                    try:
                        vec = [float(x) for x in m.parsed]
                        idxs = np.argsort(vec)[::-1][:3]
                        _line += " top3=" + ", ".join([f"{options[i]}:{_clip01(vec[i]):.3f}" for i in idxs])
                    except Exception:
                        pass
                elif qtype in ("numeric", "discrete") and m.ok and isinstance(m.parsed, dict):
                    p10 = _safe_float(m.parsed.get("P10"))
                    p50 = _safe_float(m.parsed.get("P50"))
                    p90 = _safe_float(m.parsed.get("P90"))
                    if p10 is not None and p90 is not None:
                        if p50 is None:
                            p50 = 0.5 * (p10 + p90)
                        _line += f" P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}"
                md.append(_line)
        except Exception as _ens_dbg_ex:
            md.append(f"- ensemble_debug_error={type(_ens_dbg_ex).__name__}: {str(_ens_dbg_ex)[:200]}")
    
        # --- Per-model details: reasoning + usage/cost --------------------------------
        try:
            MODEL_RAW_MAX = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
            md.append("")
            md.append("### Per-model (raw + usage/cost)")
    
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                md.append(f"#### {m.name}")
                md.append(
                    f"- ok={m.ok} | t_ms={getattr(m,'elapsed_ms',0)} | "
                    f"tokens: prompt={getattr(m,'prompt_tokens',0)}, "
                    f"completion={getattr(m,'completion_tokens',0)}, "
                    f"total={getattr(m,'total_tokens',0)} | "
                    f"cost=${float(getattr(m,'cost_usd',0.0)):.6f}"
                )
                if getattr(m, "error", None):
                    md.append(f"- error={str(m.error)[:240]}")
                if getattr(m, "raw_text", None):
                    raw = (m.raw_text or "").strip()
                    if raw:
                        md.append("```md")
                        md.append(raw[:MODEL_RAW_MAX])
                        md.append("```")
        except Exception as _pm_ex:
            md.append(f"- per_model_dump_error={type(_pm_ex).__name__}: {str(_pm_ex)[:200]}")
    
        # --- Aggregation summary (BMC) ---------------------------------------------
        try:
            md.append("### Aggregation (BMC)")
            # Make the BMC summary JSON-safe and also visible in the human log
            bmc_json = {}
            if isinstance(bmc_summary, dict):
                # strip large arrays already removed; copy select keys if present
                for k in ("mean", "var", "std", "n_evidence", "p10", "p50", "p90"):
                    if k in bmc_summary:
                        bmc_json[k] = bmc_summary[k]
            # Put a human line:
            if qtype == "binary" and isinstance(final_main, float):
                md.append(f"- final_probability={_clip01(final_main):.4f}")
            elif qtype == "multiple_choice" and isinstance(final_main, dict):
                # show top-3
                items = sorted(final_main.items(), key=lambda kv: kv[1], reverse=True)[:3]
                md.append("- final_top3=" + ", ".join([f"{k}:{_clip01(float(v)):.3f}" for k, v in items]))
            elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
                _p10 = final_main.get("P10"); _p50 = final_main.get("P50"); _p90 = final_main.get("P90")
                md.append(f"- final_quantiles: P10={_p10}, P50={_p50}, P90={_p90}")
            md.append(f"- bmc_summary={json.dumps(bmc_json)}")
        except Exception as _bmc_dbg_ex:
            md.append(f"- bmc_debug_error={type(_bmc_dbg_ex).__name__}: {str(_bmc_dbg_ex)[:200]}")
    
        # --------------------------------------------------------------------------
        # Attach BMC summary into CSV row (JSON), then persist both CSV + human log
        # --------------------------------------------------------------------------
        try:
            if isinstance(bmc_summary, dict):
                row["bmc_summary_json"] = {k: v for k, v in bmc_summary.items() if k != "samples"}
        except Exception:
            # keep whatever default is in row already
            pass
    
        # Write human-readable markdown file
        try:
            safe_md = _sanitize_markdown_chunks(md)
            if len(safe_md) < len(md):
                dropped = len(md) - len(safe_md)
                print(f"[warn] Dropped {dropped} non-string markdown line(s) for Q{question_id}.")
            write_human_markdown(run_id=run_id, question_id=question_id, content="\n\n".join(safe_md))
        except Exception as _md_ex:
            print(f"[warn] failed to write human markdown for Q{question_id}: {type(_md_ex).__name__}: {str(_md_ex)[:180]}")
    
        # Finally, write the unified CSV row
        write_unified_row(row)
        print("✔ logged to forecasts.csv")
        return
    
    
    except Exception as _e:
        _post_t = type(_post_original).__name__
        try:
            _q_t = type(q).__name__
        except Exception:
            _q_t = "unknown"
        try:
            _cls_t = type(cls_info).__name__
        except Exception:
            _cls_t = "unknown"
        raise RuntimeError(f"run_one_question failed (post={_post_t}, q={_q_t}, cls_info={_cls_t})") from _e


async def run_one_question(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    submit_ok: bool,
    calib: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
) -> None:
    q = post.get("question") or {}
    question_id = int(q.get("id") or post.get("id") or post.get("post_id") or 0)

    seen_guard_state: Dict[str, Any] = {
        "enabled": bool(seen_guard),
        "lock_acquired": None,
        "lock_error": "",
    }

    lock_stack = ExitStack()
    try:
        if seen_guard:
            try:
                acquired = lock_stack.enter_context(seen_guard.lock(question_id))
                seen_guard_state["lock_acquired"] = bool(acquired)
                if not acquired:
                    print(f"[seen_guard] QID {question_id} is locked by another process; skipping.")
                    return
            except Exception as _sg_lock_ex:
                seen_guard_state["lock_error"] = f"{type(_sg_lock_ex).__name__}: {str(_sg_lock_ex)[:160]}"
                seen_guard_state["lock_acquired"] = False
                print(f"[seen_guard] lock error for QID {question_id}: {seen_guard_state['lock_error']}")

        await _run_one_question_body(
            post=post,
            run_id=run_id,
            purpose=purpose,
            submit_ok=submit_ok,
            calib=calib,
            seen_guard_state=seen_guard_state,
            seen_guard_run_report=seen_guard_run_report,
        )
    finally:
        lock_stack.close()


# ==============================================================================
# Top-level runner (fetch posts, iterate, submit, and commit logs)
# ==============================================================================

async def run_job(mode: str, limit: int, submit: bool, purpose: str) -> None:
    """
    Fetch a batch of posts and process them one by one.
    Supports:
      - mode="tournament": uses TOURNAMENT_ID from config
      - mode="file": reads local JSON (list of posts, dict with 'results'/'posts',
                     or dict with 'post_ids' to fetch individually)
    """
    # --- local imports to keep this function self-contained ---------------
    import os, json, inspect, importlib
    from pathlib import Path

    def _istamp():
        # Use Istanbul-tz stamp from config if available, else UTC-ish fallback
        try:
            from .config import IST_TZ
            from datetime import datetime
            return datetime.now(IST_TZ).strftime("%Y%m%d-%H%M%S")
        except Exception:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    run_id = _istamp()
    print("----------------------------------------------------------------------------------------")

    # --- load helpers from this module scope --------------------------------
    # They already exist below in this file; just reference them:
    #   list_posts_from_tournament(...), get_post_details(...)
    #   ensure_unified_csv(), run_one_question(...), _load_calibration_weights()
    #   finalize_and_commit()

    # --- load questions ------------------------------------------------------
    if mode == "tournament":
        data = list_posts_from_tournament(TOURNAMENT_ID, offset=0, count=max(1, limit))
        posts = data.get("results") or data.get("posts") or []
        print(f"[info] Retrieved {len(posts)} open post(s) from '{TOURNAMENT_ID}'.")
    elif mode == "file":
        qfile_path = (
            globals().get("QUESTIONS_FILE")
            or os.getenv("QUESTIONS_FILE")
            or "data/test_questions.json"
        )
        qfile = Path(qfile_path)
        if not qfile.exists():
            raise FileNotFoundError(f"Questions file not found: {qfile}")

        with qfile.open("r", encoding="utf-8") as f:
            data = json.load(f)

        posts = []
        if isinstance(data, list):
            # list of post objects
            posts = data
        elif isinstance(data, dict):
            # dict with posts or results
            posts = data.get("results") or data.get("posts") or []
            # NEW: dict with post_ids -> fetch each
            post_ids = data.get("post_ids") or data.get("ids") or []
            if post_ids and not posts:
                fetched = []
                for pid in post_ids[: max(1, limit)]:
                    try:
                        fetched.append(get_post_details(int(pid)))
                    except Exception as e:
                        print(f"[warn] get_post_details({pid}) failed: {type(e).__name__}: {str(e)[:120]}")
                posts = fetched
        else:
            posts = []

        print(f"[info] Loaded {len(posts)} post(s) from {qfile.as_posix()}.")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # --- SeenGuard wiring (handles both package and top-level) ---------------
    def _load_seen_guard():
        """
        Try to import a SeenGuard instance/class from spagbot.seen_guard or seen_guard.
        Return an instance or None.
        """
        sg_mod = None
        # Prefer relative (inside package)
        try:
            from . import seen_guard as _sg
            sg_mod = _sg
        except Exception:
            # Fall back to absolute names
            for modname in ("spagbot.seen_guard", "seen_guard"):
                try:
                    sg_mod = importlib.import_module(modname)
                    break
                except Exception:
                    continue

        if sg_mod is None:
            return None

        # If module exposes a ready-made instance, use it
        for attr in ("_GUARD", "GUARD", "guard"):
            guard = getattr(sg_mod, attr, None)
            if guard is not None:
                return guard

        # Else instantiate SeenGuard(csv_path/state_file/lock_dir via env defaults)
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None and inspect.isclass(SG):
            try:
                return SG()  # it reads env defaults internally
            except Exception:
                return None

        return None

    def _apply_seen_guard(guard, posts_list):
        """
        Call whichever filter method exists; normalize return to (posts, dup_count).
        """
        if guard is None or not posts_list:
            return posts_list, 0

        candidates = [
            "filter_unseen_posts",     # your current API
            "filter_fresh_posts",      # earlier suggestion
            "filter_posts",
            "filter_recent_posts",
            "filter_new_posts",
            "filter",                  # very generic, last
        ]
        last_err = None
        for name in candidates:
            if hasattr(guard, name):
                fn = getattr(guard, name)
                try:
                    # most APIs: fn(posts)
                    result = fn(posts_list)
                except TypeError:
                    try:
                        # named arg fallback
                        result = fn(posts=posts_list)
                    except Exception as e:
                        last_err = e
                        continue
                except Exception as e:
                    last_err = e
                    continue

                # normalize return
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                    return result
                if isinstance(result, list):
                    # compute naive dup_count
                    return result, max(0, len(posts_list) - len(result))

                # unexpected shape → treat as no-op
                return posts_list, 0

        if last_err:
            raise last_err
        return posts_list, 0

    # Try to activate seen guard
    seen_guard_run_report: Dict[str, Any] = {
        "enabled": False,
        "before": len(posts),
        "after": len(posts),
        "skipped": 0,
        "error": "",
    }
    try:
        guard = _load_seen_guard()
        if guard is None:
            print("[seen_guard] not active; processing all posts returned.")
        else:
            seen_guard_run_report["enabled"] = True
            seen_guard_run_report["before"] = len(posts)
            before = len(posts)
            posts, dup_count = _apply_seen_guard(guard, posts)
            after = len(posts)
            if not isinstance(dup_count, int):
                dup_count = max(0, before - after)
            seen_guard_run_report["skipped"] = int(dup_count)
            seen_guard_run_report["after"] = after
            print(f"[seen_guard] {dup_count} duplicate(s) skipped; {after} fresh post(s) remain.")
    except Exception as _sg_ex:
        seen_guard_run_report["error"] = f"{type(_sg_ex).__name__}: {str(_sg_ex)[:200]}"
        print(f"[seen_guard] disabled due to error: {type(_sg_ex).__name__}: {str(_sg_ex)[:200]}")

    # Ensure CSV exists before we start
    ensure_unified_csv()

    # Process each post
    if not posts:
        print("[info] No posts to process.")
        try:
            finalize_and_commit()
            print("[logs] finalize_and_commit: done")
        except Exception as e:
            print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")
        return

    batch = posts[: max(1, limit)]
    for idx, raw_post in enumerate(batch, start=1):
        post = raw_post
        if not isinstance(post, dict):
            post_id = None
            if isinstance(post, (int, float)):
                post_id = int(post)
            elif isinstance(post, str) and post.strip():
                try:
                    post_id = int(float(post))
                except Exception:
                    post_id = None
            if post_id is not None:
                try:
                    post = get_post_details(int(post_id))
                    if not isinstance(post, dict):
                        raise TypeError("post details response was not a dict")
                    print(f"[info] Normalized post {post_id} via get_post_details().")
                except Exception as _fetch_ex:
                    print(
                        f"[error] Could not load post details for {post_id}: "
                        f"{type(_fetch_ex).__name__}: {str(_fetch_ex)[:180]}"
                    )
                    continue
            else:
                print(
                    f"[error] Skipping entry #{idx}: unexpected post type "
                    f"{type(raw_post).__name__}"
                )
                continue

        q = post.get("question") or {}
        qid = q.get("id") or post.get("id") or "?"
        title = (q.get("title") or post.get("title") or "").strip()
        print("")
        print("----------------------------------------------------------------------------------------")
        print(f"[{idx}/{len(batch)}] ❓ {title}  (QID: {qid})")
        try:
            await run_one_question(
                post,
                run_id=run_id,
                purpose=purpose,
                submit_ok=bool(submit),
                calib=_load_calibration_weights(),
                seen_guard_run_report=seen_guard_run_report,
            )
        except Exception as e:
            print(f"[error] run_one_question failed for QID {qid}: {type(e).__name__}: {str(e)[:200]}")

    # Commit logs to git if configured
    try:
        finalize_and_commit()
        print("[logs] finalize_and_commit: done")
    except Exception as e:
        print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")


# ==============================================================================
# CLI entrypoint
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spagbot runner")
    p.add_argument("--mode", default="tournament", choices=["tournament", "file"], help="Run mode")
    p.add_argument("--limit", type=int, default=20, help="Max posts to fetch/process")
    p.add_argument("--submit", action="store_true", help="Submit forecasts to Metaculus")
    p.add_argument("--purpose", default="ad_hoc", help="String tag recorded in CSV/logs")
    p.add_argument("--questions-file", default="data/test_questions.json",
                   help="When --mode file, path to JSON with {'post_ids': [..]}")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    print("🚀 Spagbot ensemble starting…")
    print(f"Mode: {args.mode} | Limit: {args.limit} | Purpose: {args.purpose} | Submit: {bool(args.submit)}")
    try:
        asyncio.run(run_job(mode=args.mode, limit=args.limit, submit=bool(args.submit), purpose=args.purpose))
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    main()
