# spagbot/research.py
from __future__ import annotations
"""
research.py — External research pipeline **without Serper/AskNews**.
This version uses **Gemini 2.5 Pro** with **Grounding by Google (Google Search tool)**
for retrieval, and Gemini again for writing the research brief.

What this module does (unchanged surface):
- Extracts "anchors" (quoted phrases, proper nouns, years, numeric-with-units, keywords)
  from the question (title/description/criteria).
- Queries **Google Grounding** (via Gemini Search tool) to get source candidates.
- Ranks & filters results by anchor overlap, with a "salvage" pass if strict filtering is empty.
- Builds a compact source list and asks **Gemini 2.5 Pro** to write a research summary.
- Returns (final_text, meta). `final_text` is what your human log prints; `meta` feeds CSV.

Key outputs in meta (all strings/numbers safe to write to CSV):
  research_llm        -> "google/gemini-2.5-pro"
  research_source     -> "GoogleGrounding" | "cache" | "none"
  research_query      -> the compact query we sent to Grounding
  research_n_raw      -> number of candidates returned by Grounding (pre-filter)
  research_n_kept     -> how many survived anchor filtering/ranking
  research_cached     -> "1" if we served from cache, else "0"
  research_usage      -> {} (Gemini tool call does not return token usage here)
  research_cost_usd   -> 0.0 (we don't compute costs for research)

Environment variables used (minimal set):
  GEMINI_API_KEY   (preferred)  — your Google Generative Language API key
  GOOGLE_API_KEY   (fallback)   — used if GEMINI_API_KEY is not set
  GEMINI_MODEL                  — default "gemini-2.5-pro"
  RESEARCH_TEMP                 — default 0.20 (see config.py)
  RESEARCH_LOG_ALL_CANDIDATES   — "1" to also print pre-filter candidates into the log
  RESEARCH_REQUIRE_YEAR_IF_PRESENT — "1" require a year match when the question contains years

Caching: we keep the same cache contract as before (read_cache/write_cache from config).
This lets repeated runs avoid hitting the network when nothing changed.
"""

import os, re, json, time, math, textwrap, asyncio, hashlib
from typing import Optional, List, Dict, Any, Tuple
import requests

from .config import (
    ist_date, ist_iso,
    RESEARCH_TEMP,
    read_cache, write_cache,
    GEMINI_CALL_TIMEOUT_SEC as _GEMINI_TIMEOUT,
    GEMINI_MODEL as _GEMINI_MODEL_ENV,
)

from .prompts import build_research_prompt, _CAL_PREFIX

# --- Debug hook: last error message from research step (for human log & CSV) ---
LAST_RESEARCH_ERROR: str = ""   # set by _grounded_search / _compose_research_via_gemini
def _set_research_error(msg: str) -> None:
    global LAST_RESEARCH_ERROR
    LAST_RESEARCH_ERROR = (msg or "").strip()

# --- Rough token counter for cost estimation ---
def _rough_token_count(text: str) -> int:
    """Cheap token count heuristic: ~4 chars ≈ 1 token."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))

# =============================================================================
# RUNTIME + CACHE
# =============================================================================

def _detect_runtime_mode() -> str:
    """Identify runtime mode from argv or env (used only for cache policy)."""
    try:
        import sys
        for i, a in enumerate(sys.argv):
            if a == "--mode" and i+1 < len(sys.argv):
                return str(sys.argv[i+1]).strip()
    except Exception:
        pass
    return os.getenv("MODE", "tournament")

def _cache_key_for(title: str, description: str, criteria: str, qtype: str) -> str:
    h = hashlib.sha256()
    h.update(("|".join([title or "", description or "", criteria or "", qtype or ""])).encode("utf-8"))
    return h.hexdigest()

# =============================================================================
# SIMPLE TEXT UTILITIES
# =============================================================================

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _find_quoted_phrases(text: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r'"([^"]+)"', text or "") if m.group(1).strip()]

def _find_years(text: str) -> List[str]:
    return list({m.group(0) for m in re.finditer(r"\b(19|20|21)\d{2}\b", text or "")})

def _find_numbers_with_units(text: str) -> List[str]:
    out = []
    for m in re.finditer(
        r"\b(\d{1,4})(?:\s*[- ]\s*)?"
        r"(%|percent|per\s*cent|million|billion|trillion|m|bn|tn|k|weeks?|months?|years?|days?)\b",
        (text or "").lower()
    ):
        out.append(f"{m.group(1)} {m.group(2)}")
    return out

def _find_proper_nouns(text: str) -> List[str]:
    toks = re.findall(r"[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*", text or "")
    # Keep multi-word capitalized phrases; reduce single-word noise.
    keep = []
    for t in toks:
        if len(t.split()) >= 2:
            keep.append(t.strip())
    return list(dict.fromkeys(keep))

def _split_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", (text or "").lower()) if t and len(t) > 2]

def _extract_anchors(title: str, description: str, criteria: str) -> Dict[str, List[str]]:
    blob = " ".join([title or "", description or "", criteria or ""])
    return {
        "quotes": _find_quoted_phrases(blob),
        "years": _find_years(blob),
        "nums": _find_numbers_with_units(blob),
        "proper": _find_proper_nouns(blob),
        "tokens": _split_tokens(blob),
    }

# =============================================================================
# GROUNDING with GOOGLE via Gemini API
# =============================================================================

def _gemini_base_url(model: str) -> str:
    # REST endpoint for generateContent in v1beta
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def _gemini_api_key() -> str:
    # Prefer GEMINI_API_KEY; fallback to GOOGLE_API_KEY for convenience.
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

def _grounded_search(query: str, *, max_results: int = 12, timeout: float = None) -> List[Dict[str, Any]]:
    """Call Gemini with Google Search grounding to retrieve sources.
    Returns a list of dicts: {title, url, text, date, source}
    """
    timeout = float(_GEMINI_TIMEOUT if timeout is None else timeout)
    api_key = _gemini_api_key()
    if not api_key:
        _set_research_error("missing GEMINI_API_KEY / GOOGLE_API_KEY")
        return []
    model = (_GEMINI_MODEL_ENV or "gemini-2.5-pro").strip()

    # System-style guidance + user task. (v1beta allows only 'user' parts; keep both for clarity.)
    sys_instr = (
        "You are a research assistant. Use Google Search grounding to find recent, high-quality, "
        "authoritative sources relevant to the user's query. Return only JSON Lines (one JSON per line) "
        "with fields: title, url, source, summary, date. Prefer major outlets, official publications, "
        "and recent coverage (last 18 months when applicable)."
    )
    user_task = f"Query: {query}\nTop {max_results} sources, JSON Lines only."

    def _post(body: dict) -> tuple[int, dict]:
        """POST helper that always returns (status_code, json_or_fallback)."""
        try:
            r = requests.post(
                _gemini_base_url(model),
                params={"key": api_key},
                json=body,
                timeout=timeout,
            )
            try:
                j = r.json()
            except Exception:
                j = {"error_text": (r.text or "")[:800]}
            return r.status_code, j
        except Exception as e:
            return 599, {"error_text": f"exception: {e!r}"}

    # Shared content payload
    shared_contents = [
        {"role": "user", "parts": [{"text": sys_instr}]},
        {"role": "user", "parts": [{"text": user_task}]},
    ]
    gen_cfg = {"temperature": 0.2, "maxOutputTokens": 2048}

    # We try multiple schemata because Google's v1beta has toggled names between releases.
    attempts: List[Tuple[str, dict]] = [
        # 1) Official camelCase (most recent docs)
        ("camel_with_toolConfig_MODE_DYNAMIC", {
            "contents": shared_contents,
            "tools": [{"googleSearchRetrieval": {}}],
            "toolConfig": {"googleSearchRetrieval": {"dynamicRetrievalConfig": {"mode": "MODE_DYNAMIC"}}},
            "generationConfig": gen_cfg,
        }),
        # 2) Same but DYNAMIC (some builds reject MODE_ prefix)
        ("camel_with_toolConfig_DYNAMIC", {
            "contents": shared_contents,
            "tools": [{"googleSearchRetrieval": {}}],
            "toolConfig": {"googleSearchRetrieval": {"dynamicRetrievalConfig": {"mode": "DYNAMIC"}}},
            "generationConfig": gen_cfg,
        }),
        # 3) CamelCase with NO toolConfig (let platform choose defaults)
        ("camel_no_toolConfig", {
            "contents": shared_contents,
            "tools": [{"googleSearchRetrieval": {}}],
            "generationConfig": gen_cfg,
        }),
        # 4) Snake_case schema (older samples)
        ("snake_with_tool_config_MODE_DYNAMIC", {
            "contents": shared_contents,
            "tools": [{"google_search_retrieval": {}}],
            "tool_config": {"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "MODE_DYNAMIC"}}},
            "generationConfig": gen_cfg,
        }),
        # 5) Snake_case with DYNAMIC
        ("snake_with_tool_config_DYNAMIC", {
            "contents": shared_contents,
            "tools": [{"google_search_retrieval": {}}],
            "tool_config": {"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "DYNAMIC"}}},
            "generationConfig": gen_cfg,
        }),
        # 6) Snake_case with NO tool_config
        ("snake_no_tool_config", {
            "contents": shared_contents,
            "tools": [{"google_search_retrieval": {}}],
            "generationConfig": gen_cfg,
        }),
        # 7) Legacy 'google_search' (very old samples)
        ("legacy_google_search_simple", {
            "contents": shared_contents,
            "tools": [{"google_search": {}}],
            "generationConfig": gen_cfg,
        }),
    ]

    last_errs: List[str] = []
    data_ok: dict = {}
    status_ok: int = 0
    for attempt_name, body in attempts:
        status, data = _post(body)
        if status == 200:
            data_ok, status_ok = data, status
            last_errs.clear()
            break
        # Collect short error note for debugging
        if isinstance(data, dict):
            msg = ""
            if "error" in data and isinstance(data["error"], dict):
                msg = data["error"].get("message", "") or ""
            if not msg:
                msg = data.get("error_text", "")
            if not msg:
                msg = str(data)[:300]
            last_errs.append(f"{attempt_name}: HTTP {status} → {msg[:220]}")
        else:
            last_errs.append(f"{attempt_name}: HTTP {status}")

    if status_ok != 200:
        # Surface the top one or two error messages for your human log.
        tail = " | ".join(last_errs[:2]) if last_errs else "no provider message"
        msg = f"grounding failed: {tail}"
        print(f"[research] {msg}")
        _set_research_error(msg)
        return []

    # Parse candidates
    text_blobs: List[str] = []
    for cand in (data_ok.get("candidates") or []):
        for part in ((cand.get("content") or {}).get("parts") or []):
            t = part.get("text")
            if t:
                text_blobs.append(t)
    joined = "\n".join(text_blobs).strip()

    items: List[Dict[str, Any]] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            items.append({
                "title": obj.get("title") or obj.get("headline") or obj.get("url") or "",
                "url": obj.get("url") or "",
                "text": obj.get("summary") or obj.get("snippet") or "",
                "date": obj.get("date") or "",
                "source": obj.get("source") or "",
            })
        except Exception:
            # ignore non-JSON lines
            pass

    if not items:
        _set_research_error("no valid JSON lines returned from grounding")
    return items


# =============================================================================
# RANK & FILTER (unchanged logic, just new provider)
# =============================================================================

RESEARCH_REQUIRE_YEAR_IF_PRESENT = os.getenv("RESEARCH_REQUIRE_YEAR_IF_PRESENT", "1") in ("1","true","yes")
RESEARCH_LOG_ALL_CANDIDATES = os.getenv("RESEARCH_LOG_ALL_CANDIDATES", "0") in ("1","true","yes")

def _anchor_overlap_score(item: Dict[str, Any], anchors: Dict[str, List[str]]) -> Tuple[int, float]:
    """Return (match_count, score) — used for ranking."""
    title = _norm_space(item.get("title") or "")
    text  = _norm_space(item.get("text") or "")
    blob  = f"{title} {text}".lower()

    matches = 0
    score = 0.0
    for q in anchors.get("quotes", []):
        if q.lower() in blob:
            matches += 1; score += 3.0
    for y in anchors.get("years", []):
        if y in blob:
            matches += 1; score += 1.5
    for n in anchors.get("nums", []):
        if n.lower() in blob:
            matches += 1; score += 1.0
    for p in anchors.get("proper", []):
        if p.lower() in blob:
            matches += 1; score += 2.0
    for t in anchors.get("tokens", []):
        if t in blob:
            score += 0.25
    return matches, score

def _rank_and_filter_items(items: List[Dict[str, Any]], anchors: Dict[str, List[str]], *, min_match:int) -> List[Dict[str, Any]]:
    out: List[Tuple[float, Dict[str, Any]]] = []
    years_present = bool(anchors.get("years"))
    for it in items:
        m, s = _anchor_overlap_score(it, anchors)
        if m >= min_match:
            if RESEARCH_REQUIRE_YEAR_IF_PRESENT and years_present:
                if not re.search(r"\b(19|20|21)\d{2}\b", f"{it.get('title','')} {it.get('text','')}"):
                    continue
            out.append((s, it))
    out.sort(key=lambda t: -t[0])
    return [it for _, it in out]

# =============================================================================
# COMPOSE THE RESEARCH BRIEF (Gemini only for research)
# =============================================================================

async def _compose_research_via_gemini(prompt_text: str) -> tuple[str, str, dict, dict]:
    """Generate the research brief using Gemini 2.5 Pro.
    Returns (text, used_model_id, usage_dict, request_body)."""
    api_key = _gemini_api_key()
    if not api_key:
        _set_research_error("no GEMINI_API_KEY (or GOOGLE_API_KEY) for research composition")
        return "", "", {}, body
    model = (_GEMINI_MODEL_ENV or "gemini-2.5-pro").strip()

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": float(RESEARCH_TEMP), "maxOutputTokens": 5000},
    }
    try:
        resp = requests.post(
            _gemini_base_url(model),
            params={"key": api_key},
            json=body,
            timeout=float(_GEMINI_TIMEOUT),
        )
        if resp.status_code != 200:
            try:
                j = resp.json()
                msg = (j.get("error", {}) or {}).get("message", "")
            except Exception:
                msg = (resp.text or "")[:200]
            _set_research_error(f"compose HTTP {resp.status_code}: {msg}")
            return "", "", {}, body
        data = resp.json()

        texts = []
        for cand in (data.get("candidates") or []):
            for part in ((cand.get("content") or {}).get("parts") or []):
                t = part.get("text")
                if t:
                    texts.append(t)
        return ("\n".join(texts).strip(), f"google/{model}", {}, body)
    except Exception as e:
        _set_research_error(f"compose request error: {e!r}")
        return "", "", {}, body

# =============================================================================
# UTIL: format sources for prompt/log
# =============================================================================

def _format_sources_for_prompt(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    lines = ["### Sources"]
    for it in items[:12]:
        title = it.get("title") or "(untitled)"
        url   = it.get("url") or ""
        text  = it.get("text") or ""
        date  = it.get("date") or ""
        lines.append(f"- {title} ({url}) — {date}\n  {text}")
    return "\n".join(lines)

def _host_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url or "")
    return (m.group(1) if m else "").lower()

def _format_sources_for_log(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "### Sources\n- *(none)*"
    lines = ["### Sources"]
    for it in items[:12]:
        host = _host_of(it.get("url") or "")
        title = it.get("title") or it.get("url") or "(untitled)"
        url = it.get("url") or ""
        lines.append(f"- {title} ({host}) — {url}")
    return "\n".join(lines)

def _format_all_candidates_for_log(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "### Provider Candidates (pre-filter)\n- *(none)*"
    lines = ["### Provider Candidates (pre-filter)"]
    for it in items[:30]:  # cap to avoid massive logs
        host = _host_of(it.get("url") or "")
        title = it.get("title") or it.get("url") or "(untitled)"
        url = it.get("url") or ""
        lines.append(f"- {title} ({host}) — {url}")
    return "\n".join(lines)

# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

async def run_research_async(
    title: str,
    description: str,
    criteria: str,
    qtype: str,
    options: Optional[List[str]] = None,
    units: Optional[str] = None,
    slug: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """Perform research and return (final_text, meta). See module docstring for meta fields."""
    title = _norm_space(title)
    description = _norm_space(description)
    criteria = _norm_space(criteria)
    qtype = (qtype or "").strip()

    # 0) cache check (keyed on question text + mode to avoid cross-mode collisions)
    mode = _detect_runtime_mode()
    ck = f"{mode}:{_cache_key_for(title, description, criteria, qtype)}"
    cached = read_cache("research", ck)
    if cached:
        try:
            data = json.loads(cached)
            final_text = data.get("final_text", "")
            meta_raw: Any = data.get("meta", {}) or {}

            if isinstance(meta_raw, dict):
                meta = dict(meta_raw)
            elif isinstance(meta_raw, str):
                try:
                    meta_parsed = json.loads(meta_raw)
                    meta = dict(meta_parsed) if isinstance(meta_parsed, dict) else {}
                except Exception:
                    meta = {}
            else:
                meta = {}

            meta["research_cached"] = "1"
            return final_text, meta
        except Exception:
            pass

    # 1) anchors and compact query
    anchors = _extract_anchors(title, description, criteria)
    # Compact query: combine proper nouns, quoted phrases, and years for specificity
    query_bits = anchors.get("quotes", []) + anchors.get("proper", []) + anchors.get("years", [])
    if not query_bits:
        # fallback to key tokens
        query_bits = (anchors.get("tokens") or [])[:8]
    compact_q = "; ".join(dict.fromkeys(query_bits))[:500]

    # 2) Google Grounding fetch
    raw_items = _grounded_search(compact_q, max_results=12)
    source_tag = "GoogleGrounding" if raw_items else "none"
    query_used = compact_q

    # 3) rank & filter; salvage if empty
    picked = _rank_and_filter_items(raw_items, anchors, min_match=1)
    if not picked and raw_items:
        picked = _rank_and_filter_items(raw_items, anchors, min_match=0)

    # 4) build prompt
    if str(qtype).lower() in {"multiple","mcq","multiple_choice"}:
        units_or_options = "\n".join(str(o) for o in (options or []))
    else:
        units_or_options = units or ""

    sources_text_for_prompt = _format_sources_for_prompt(picked) if picked else ""
    prompt = build_research_prompt(
        title=title,
        qtype=qtype,
        units_or_options=units_or_options,
        background=description or "",
        criteria=criteria or "",
        today=ist_date(),
        sources_text=sources_text_for_prompt,
    )

    # 5) LLM compose (Gemini only for research)
    llm_text, used_llm, usage, req_body = await _compose_research_via_gemini(prompt)
    if not llm_text.strip():
        if picked:
            llm_text = "\n".join([f"- {it.get('title','')} ({it.get('url','')})" for it in picked])
        else:
            llm_text = "No recent external sources found; proceeding with general knowledge and base rates."

    # --- cost estimation ---
    prompt_tokens = _rough_token_count(json.dumps(req_body))
    output_tokens = _rough_token_count(llm_text)
    # Gemini 2.5 Pro rates (USD per 1K tokens)
    in_rate = 0.00125
    out_rate = 0.01
    research_cost_usd = (prompt_tokens/1000)*in_rate + (output_tokens/1000)*out_rate

    # 6) Final text for the human log: brief + source list (+ optional pre-filter dump)
    parts = [_CAL_PREFIX + llm_text, "", _format_sources_for_log(picked)]
    if RESEARCH_LOG_ALL_CANDIDATES and raw_items:
        parts += ["", _format_all_candidates_for_log(raw_items)]
    final_text = "\n".join(parts)

    # 7) cache write
    try:
        cache_blob = json.dumps({
            "final_text": final_text,
            "meta": {
                "research_llm": used_llm or "google/gemini-2.5-pro",
                "research_source": source_tag,
                "research_query": query_used,
                "research_n_raw": int(len(raw_items)),
                "research_n_kept": int(len(picked)),
                "research_cached": "0",
                "research_error": (LAST_RESEARCH_ERROR or ""),
                "research_usage": usage or {},
                "research_cost_usd": round(research_cost_usd, 6),
            },
        }, ensure_ascii=False)
        write_cache("research", ck, cache_blob)
    except Exception:
        pass

    # 8) meta (mirror cached structure)
    meta = {
        "research_llm": used_llm or "google/gemini-2.5-pro",
        "research_source": source_tag,
        "research_query": query_used,
        "research_n_raw": int(len(raw_items)),
        "research_n_kept": int(len(picked)),
        "research_cached": "0",
        "research_error": (LAST_RESEARCH_ERROR or ""),
        "research_usage": usage or {},
        "research_cost_usd": round(research_cost_usd, 6),
    }
    return final_text, meta
