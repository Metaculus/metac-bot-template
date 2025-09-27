# spagbot/cli.py
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
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import requests

# ---- Spagbot internals (all relative imports) --------------------------------
from .config import (
    TOURNAMENT_ID, AUTH_HEADERS, API_BASE_URL,
    ist_iso, ist_stamp, SUBMIT_PREDICTION,
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

# Unified CSV helpers (single file)
from .io_logs import ensure_unified_csv, write_unified_row, write_human_markdown, finalize_and_commit

# --------------------------------------------------------------------------------
# Small utility helpers (safe JSON, timing, clipping, etc.)
# --------------------------------------------------------------------------------

def _ms(start_time: float) -> int:
    return int(round((time.time() - start_time) * 1000))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
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
    try:
        import asyncio as _aio
        if _aio.iscoroutinefunction(should_run_gtmc1):
            res = await should_run_gtmc1(title, description, criteria, slug=slug)
        else:
            # sync function
            try:
                res = should_run_gtmc1(title, description, criteria, slug=slug)
            except TypeError:
                # older signature without slug
                res = should_run_gtmc1(title, description, criteria)
    except Exception as e:
        return False, {"primary":"", "secondary":"", "is_strategic": False, "strategic_score": 0.0,
                       "source":"", "rationale": f"classifier error: {e}"}

    # Normalize return formats:
    # - preferred: (use_gtmc1: bool, cls_info: dict)
    # - acceptable: cls_info dict only
    if isinstance(res, tuple) and len(res) == 2:
        use_flag = bool(res[0])
        info = res[1] if isinstance(res[1], dict) else {}
        return use_flag, info
    if isinstance(res, dict):
        return bool(res.get("is_strategic", False)), res
    # Unknown shape → safe default
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
    r = requests.get(url, params=params, **AUTH_HEADERS)
    if not r.ok:
        raise RuntimeError(r.text)
    return json.loads(r.content)

def get_post_details(post_id: int) -> dict:
    url = f"{API_BASE_URL}/posts/{post_id}/"
    r = requests.get(url, **AUTH_HEADERS)
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
    r = requests.post(url, json=[{"question": question_id, **payload}], **AUTH_HEADERS)
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

async def run_one_question(post: dict, *, run_id: str, purpose: str, submit_ok: bool, calib: Dict[str, Any]) -> None:
    t_start_total = time.time()

    q = post.get("question") or {}
    post_id = int(post.get("id") or post.get("post_id") or 0)
    question_id = int(q.get("id") or 0)

    # Concurrency lock using seen_guard
    if seen_guard:
        with seen_guard.lock(question_id) as acquired:
            if not acquired:
                print(f"[seen_guard] QID {question_id} is locked by another process; skipping.")
                return
    
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
                try:
                    data = json.loads(re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S))
                except Exception:
                    data = {}
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
                t_gtmc1_ms = _ms(t_gt0)
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

    # Classifier (debug)
    try:
        md.append("### Classifier (debug)")
        md.append(f"- source={classifier_source} | is_strategic={is_strategic} | "
                  f"score={strategic_score:.2f} | cost=${classifier_cost:.6f}")

        if classifier_rationale:
            md.append(f"- rationale: {classifier_rationale}")
    except Exception:
        pass

    md.append("## Per-model")
    had_errors = []
    for m in ens_res.members:
        raw_excerpt = (m.raw_text or "")[:MAX_RAW_CHARS]
        err = f" | error={m.error}" if getattr(m, "error", "") else ""
        if getattr(m, "error", ""):
            had_errors.append(f"{m.name}: {m.error}")
        md.append(f"- **{m.name}** ok={m.ok} time={getattr(m,'elapsed_ms',0)}ms cost=${getattr(m,'cost_usd',0.0):.4f}{err} parsed={m.parsed}")
        if raw_excerpt:
            md.append("  - Raw output (excerpt):")
            md.append("    " + raw_excerpt.replace("\n","\n    "))
    if had_errors:
        md.append("### Model errors (summary)")
        for line in had_errors:
            md.append(f"- {line}")


    md.append("## Ensembles (with research)")
    if qtype == "binary":
        md.append(f"- main={final_main:.4f} | no_gtmc1={v_nogtmc1:.4f} | uniform={v_uniform:.4f} | simple={v_simple:.4f}")
    elif qtype == "multiple_choice":
        md.append(f"- main={final_main}")
    else:
        md.append(f"- main={final_main}")

    md.append("## Ablation (no research)")
    if qtype == "binary":
        md.append(f"- main={ab_main:.4f} | uniform={ab_uniform:.4f} | simple={ab_simple:.4f}")
    else:
        md.append(f"- main={ab_main}")

    if gtmc1_active:
        md.append("## GTMC1")
        md.append(f"- policy: {gtmc1_policy_sentence}")
        md.append(f"- signal: {json.dumps(gtmc1_signal, ensure_ascii=False)[:1500]}")

    row.update({
        # ...existing keys...
        "cost_usd__classifier": f"{classifier_cost:.6f}",
        "cost_usd__research": f"{float(research_meta.get('research_cost_usd',0.0)):.6f}",
    })
    # total cost = research + per-model costs
    total_cost = float(research_meta.get('research_cost_usd',0.0))
    for ms in DEFAULT_ENSEMBLE:
        try:
            total_cost += float(row.get(f"cost_usd__{ms.name}", "0") or 0)
        except Exception:
            pass
    row["cost_usd__total"] = f"{total_cost:.6f}"

    write_human_markdown(run_id, question_id, "\n\n".join(md))

    write_unified_row(row)

    # Record success → mark this question as seen so we don’t reforecast it later
    if seen_guard:
        seen_guard.mark_seen(question_id)

# --------------------------------------------------------------------------------
# Batch runners (test set or tournament)
# --------------------------------------------------------------------------------

async def run_posts(posts: List[dict], *, purpose: str, submit_ok: bool) -> None:
    ensure_unified_csv()
    calib = _load_calibration_weights()
    run_id = ist_stamp()

    # MODERN FILTERING: Filter posts at the beginning of the batch run
    if seen_guard:
        posts_to_run = seen_guard.filter_unseen_posts(posts)
    else:
        posts_to_run = posts

    if not posts_to_run:
        print("[seen_guard] All candidate posts already handled. Exiting batch.")
        return

    for i, post in enumerate(posts_to_run, 1):
        qtitle = str((post.get("question") or {}).get("title") or post.get("title") or "")
        qid = int((post.get("question") or {}).get("id") or 0)
        print(f"\n{'-'*88}\n[{i}/{len(posts_to_run)}] ❓ {qtitle}  (QID: {qid})")
        try:
            await run_one_question(post, run_id=run_id, purpose=purpose, submit_ok=submit_ok, calib=calib)
            print("✔ logged to forecasts.csv")
        except Exception as e:
            print(f"✖ error on QID {qid}: {e!r}")

    # --- Finalize & commit logs to Git (CI-safe) ---
    try:
        finalize_and_commit(
            run_id,
            forecast_rows_written=True,
            extra_paths=None,
            commit_message=os.getenv(
                "GIT_LOG_MESSAGE",
                f"chore(logs): append forecasts & run logs ({purpose})"
            ),
        )
        print("[logs] finalize_and_commit: done")
    except Exception as e:
        # Never fail the run because of git/logging issues
        print(f"[logs] finalize_and_commit skipped: {e!r}")

async def run_test_questions(limit: int, *, purpose: str, submit_ok: bool) -> None:
    DEFAULT_IDS = [578, 14333, 22427, 38195]
    env_ids = os.getenv("TEST_POST_IDS", "")
    ids = DEFAULT_IDS
    if env_ids.strip():
        try:
            ids = [int(x) for x in re.split(r"[,\s]+", env_ids.strip()) if x]
        except Exception:
            pass
    ids = ids[:max(1, limit)]
    posts: List[dict] = []
    for pid in ids:
        try:
            posts.append(get_post_details(int(pid)))
        except Exception as e:
            print(f"[warn] failed to fetch post {pid}: {e!r}")
    await run_posts(posts, purpose=purpose, submit_ok=submit_ok)

async def run_tournament(limit: int, *, purpose: str, submit_ok: bool) -> None:
    try:
        resp = list_posts_from_tournament(TOURNAMENT_ID, offset=0, count=limit)
        posts = resp.get("results") or []
        if not posts:
            print(f"[warn] No open posts returned for tournament '{TOURNAMENT_ID}'.")
            return
        print(f"[info] Retrieved {len(posts)} open post(s) from '{TOURNAMENT_ID}'.")
        await run_posts(posts, purpose=purpose, submit_ok=submit_ok)
    except Exception as e:
        print(f"[error] listing tournament posts: {e!r}")
        return

# --------------------------------------------------------------------------------
# CLI entrypoints
# --------------------------------------------------------------------------------

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test_questions", "tournament"], default="test_questions",
                        help="Where to pull questions from.")
    parser.add_argument("--limit", type=int, default=4, help="How many questions to run.")
    parser.add_argument("--pid", type=int, default=None, help="Run a single, specific post ID.")
    parser.add_argument("--submit", action="store_true", help="Actually submit to Metaculus.")
    parser.add_argument("--purpose", type=str, default="testing",
                        choices=["fall_aib_2025", "metaculus_cup", "testing"],
                        help="Internal purpose tag for logging & analysis.")
    args = parser.parse_args()

    submit_ok = args.submit or SUBMIT_PREDICTION

    print("🚀 Spagbot ensemble starting…")
    print(f"Mode: {args.mode} | Limit: {args.limit} | Purpose: {args.purpose} | Submit: {submit_ok}")
    print("-" * 88)

    if args.pid:
        try:
            post = get_post_details(int(args.pid))
            await run_posts([post], purpose=args.purpose, submit_ok=submit_ok)
        except Exception as e:
            print(f"[error] Failed to fetch or run question {args.pid}: {e!r}")
        return

    if args.mode == "test_questions":
        await run_test_questions(args.limit, purpose=args.purpose, submit_ok=submit_ok)
    else:
        await run_tournament(args.limit, purpose=args.purpose, submit_ok=submit_ok)

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[ctrl-c] aborted")