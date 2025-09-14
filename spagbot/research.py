# spagbot/research.py
from __future__ import annotations
"""
research.py — External research pipeline with Serper (news/web) and optional AskNews,
returning BOTH a research brief and rich metadata for logging.

What this module does:
- Extracts "anchors" (quoted phrases, proper nouns, years, numeric-with-units, keywords)
  from the question (title/description/criteria).
- Queries providers (Serper news first, then Serper web; AskNews optional).
- Ranks & filters results by anchor overlap, with a "salvage" pass if strict filtering is empty.
- Builds a compact source list and asks an LLM to write a research summary (OpenRouter model
  from your .env; falls back to Gemini only if enabled).
- Returns (final_text, meta). `final_text` is what your human log prints; `meta` feeds CSV.

Key outputs in meta (all strings/numbers safe to write to CSV):
  research_llm        -> which LLM wrote the brief (e.g., "openai/gpt-4o")
  research_source     -> "Serper(news)" | "Serper(web)" | "AskNews" | "cache" | "none"
  research_query      -> the provider query that produced items (when Serper)
  research_n_raw      -> number of candidates returned by the provider before filtering
  research_n_kept     -> number after filtering and truncation (top-K)
  research_cached     -> "1" if cache hit, else "0"
  research_usage      -> dict with token usage (prompt/completion/total) for the research LLM
  research_cost_usd   -> float, estimated cost for the research LLM call

Tunable via .env:
  ENABLE_SERPER=1
  ENABLE_SERPER_WEB=1
  SERPER_API_KEY=...
  SERPER_LIMIT=12

  ENABLE_ASKNEWS=0/1
  ASKNEWS_CLIENT_ID=...
  ASKNEWS_SECRET=...
  ASKNEWS_STRATEGIES="news knowledge,latest news,default"
  ASKNEWS_ARTICLE_LIMIT=10
  ASKNEWS_PLAN_MAX_ARTICLES=10
  ASKNEWS_TOPK=8           # how many ranked items we keep in the final prompt/log

  MIN_ANCHOR_MATCH=2
  SALVAGE_MIN_MATCH=1
  REQUIRE_YEAR_IF_PRESENT=0/1

  RESEARCH_SNIPPET_MAX_CHARS=600  # length of each snippet shown to the LLM
  RESEARCH_LOG_ALL_CANDIDATES=0/1 # append a verbose list of pre-filter candidates at the end

  SPAGBOT_DISABLE_RESEARCH_CACHE=0/1
"""

import os, re, json, sys
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- Spagbot internals ----------------------------------------------------
from .config import (
    ist_date, ist_iso,
    RESEARCH_TEMP,
    ASKNEWS_CLIENT_ID, ASKNEWS_SECRET,
    read_cache, write_cache,
)
from .prompts import build_research_prompt, _CAL_PREFIX
from .providers import _get_or_client, _call_google, llm_semaphore, usage_to_dict, estimate_cost_usd

# Optional AskNews SDK (safe if not installed)
try:
    from asknews_sdk import AskNewsSDK  # pip install asknews
except Exception:
    AskNewsSDK = None  # type: ignore

# =============================================================================
# RUNTIME + CACHE
# =============================================================================

def _detect_runtime_mode() -> str:
    """Identify runtime mode from argv or env (used only for cache policy)."""
    try:
        if "--mode" in sys.argv:
            i = sys.argv.index("--mode")
            if i + 1 < len(sys.argv):
                return str(sys.argv[i + 1]).strip()
    except Exception:
        pass
    return os.environ.get("SPAGBOT_RUNTIME_MODE", "test_questions").strip() or "test_questions"

def _cache_allowed() -> bool:
    """
    Allow caching by default in test mode. Kill-switch via SPAGBOT_DISABLE_RESEARCH_CACHE=1.
    """
    if os.getenv("SPAGBOT_DISABLE_RESEARCH_CACHE","0").lower() in ("1","true","yes"):
        return False
    return _detect_runtime_mode() in {"test_questions"}

# =============================================================================
# ANCHOR EXTRACTION
# =============================================================================

STOPWORDS = {
    "the","a","an","of","to","in","on","for","and","or","but","with","without","via","by","at","from",
    "about","around","near","across","into","out","up","down","as","is","are","was","were","be","been",
    "it","its","this","that","those","these","which","who","whom","whose","etc","will"
}

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

def _split_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9%]+", text or "")

def _properish_phrases(text: str) -> List[str]:
    # naive: sequences of Capitalized Words length >= 2
    out: List[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\b", text or ""):
        s = m.group(1).strip()
        if len(s.split()) >= 2:
            out.append(s)
    # dedupe, keep first 12
    return list(dict.fromkeys(out))[:12]

def _salient_keywords(text: str, max_k: int = 10) -> List[str]:
    toks = [t.lower() for t in _split_tokens(text)]
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    score: Dict[str, int] = {}
    for t in toks:
        score[t] = score.get(t, 0) + 1
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k,_ in ranked][:max_k]

def extract_anchors(title: str, description: str, criteria: str) -> Dict[str, List[str]]:
    text_full = " ".join([title or "", description or "", criteria or ""])
    return {
        "quoted": _find_quoted_phrases(text_full),
        "properish": _properish_phrases(text_full),
        "years": _find_years(text_full),
        "num_units": _find_numbers_with_units(text_full),
        "keywords": _salient_keywords(text_full, max_k=10),
    }

# =============================================================================
# RANK/FILTER BY ANCHOR OVERLAP
# =============================================================================

MIN_ANCHOR_MATCH        = int(os.getenv("MIN_ANCHOR_MATCH","2"))
SALVAGE_MIN_MATCH      = int(os.getenv("SALVAGE_MIN_MATCH","1"))
REQUIRE_YEAR_IF_PRESENT = os.getenv("REQUIRE_YEAR_IF_PRESENT","0").lower() in ("1","true","yes")

RESEARCH_LOG_ALL_CANDIDATES = os.getenv("RESEARCH_LOG_ALL_CANDIDATES","0").lower() in ("1","true","yes")
RESEARCH_SNIPPET_MAX_CHARS  = int(os.getenv("RESEARCH_SNIPPET_MAX_CHARS","600"))
ASKNEWS_TOPK                = int(os.getenv("ASKNEWS_TOPK","8"))

def _host_of(url: str) -> str:
    try:
        host = urlparse(url).netloc
        if host.startswith("www."): host = host[4:]
        return host
    except Exception:
        return ""

def _anchor_overlap_score(item: Dict[str, Any], anchors: Dict[str, List[str]]) -> Tuple[int, float]:
    """
    Return (match_count, heuristic_score). We reward quoted/proper matches more.
    """
    text = f"{item.get('title','')} {item.get('text','')} {item.get('url','')}".lower()
    score = 0.0
    matches = 0
    for w in anchors.get("quoted", []):
        if w.lower() in text: score += 3.0; matches += 1
    for w in anchors.get("properish", []):
        if w.lower() in text: score += 2.0; matches += 1
    for w in anchors.get("num_units", []):
        if w.lower() in text: score += 1.5; matches += 1
    for w in anchors.get("years", []):
        if w.lower() in text: score += 1.0; matches += 1
    for w in anchors.get("keywords", []):
        if w.lower() in text: score += 0.5; matches += 1
    # light recency: if a year appears in provider item metadata
    try:
        y = re.search(r"\b(20\d{2}|19\d{2})\b", item.get("published") or "")
        if y: score += 0.25
    except Exception:
        pass
    return matches, score

def _rank_and_filter_items(items: List[Dict[str, Any]], anchors: Dict[str, List[str]], *, min_match:int) -> List[Dict[str, Any]]:
    out: List[Tuple[float, Dict[str, Any]]] = []
    years_present = bool(anchors.get("years"))
    for it in items:
        m, s = _anchor_overlap_score(it, anchors)
        if m >= min_match:
            if REQUIRE_YEAR_IF_PRESENT and years_present:
                if not re.search(r"\b(19|20|21)\d{2}\b", f"{it.get('title','')} {it.get('text','')}"):
                    continue
            out.append((s, it))
    out.sort(key=lambda t: -t[0])
    return [it for _, it in out]

# =============================================================================
# PROVIDERS: SERPER (news + web) and AskNews (optional)
# =============================================================================

ENABLE_SERPER     = os.getenv("ENABLE_SERPER","1").lower() in ("1","true","yes")
ENABLE_SERPER_WEB = os.getenv("ENABLE_SERPER_WEB","1").lower() in ("1","true","yes")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY","").strip()
SERPER_LIMIT      = int(os.getenv("SERPER_LIMIT","12"))

SERPER_NEWS_URL = "https://google.serper.dev/news"
SERPER_WEB_URL  = "https://google.serper.dev/search"

def _requests_session_with_retry(total:int=3, backoff:float=0.6) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=total, connect=total, read=total, status=total,
        backoff_factor=backoff,
        status_forcelist=(429,500,502,503,504),
        allowed_methods=("POST",)
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter); sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": "Spagbot/2.0 research.py"})
    return sess

def _serper_post(url: str, q: str, num: int) -> Optional[dict]:
    if not (ENABLE_SERPER and SERPER_API_KEY):
        return None
    try:
        sess = _requests_session_with_retry()
        r = sess.post(url, json={"q": q, "num": num}, headers={"X-API-KEY": SERPER_API_KEY, "Content-Type":"application/json"}, timeout=15)
        if r.status_code != 200:
            print(f"[research] Serper non-200 {r.status_code} | q={q[:120]!r} | head={(r.text or '')[:140]!r}")
            return None
        return r.json()
    except Exception as e:
        print(f"[research] Serper error: {type(e).__name__}: {e!r}")
        return None

def _items_from_serper_news(j: dict) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for it in (j.get("news") or []):
        items.append({
            "title": it.get("title") or "",
            "url": it.get("link") or "",
            "outlet": it.get("source") or _host_of(it.get("link") or ""),
            "published": it.get("date") or "",
            "text": (it.get("snippet") or it.get("description") or "")[:RESEARCH_SNIPPET_MAX_CHARS]
        })
    return items

def _items_from_serper_web(j: dict) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for it in (j.get("organic") or []):
        items.append({
            "title": it.get("title") or "",
            "url": it.get("link") or "",
            "outlet": _host_of(it.get("link") or ""),
            "published": "",  # web results often lack dates
            "text": (it.get("snippet") or it.get("description") or "")[:RESEARCH_SNIPPET_MAX_CHARS]
        })
    return items

def _build_provider_query_from_anchors(anchors: Dict[str, List[str]], *, limit_terms:int = 10, must_terms_cap:int = 2) -> str:
    """
    Query string: AND of up to 'must_terms_cap' quoted/proper terms, plus OR bucket of keywords.
    """
    must_terms: List[str] = []
    for bucket in ["quoted","properish","num_units","years"]:
        for s in anchors.get(bucket, []):
            s = _norm_space(s)
            if s and s not in must_terms:
                must_terms.append(s)
            if len(must_terms) >= must_terms_cap:
                break
        if len(must_terms) >= must_terms_cap:
            break
    or_terms = anchors.get("keywords", [])[:limit_terms]
    q = ""
    if must_terms:
        q += " AND ".join([f"\"{t}\"" for t in must_terms])
    if or_terms:
        q += (" AND " if q else "") + "(" + " OR ".join(or_terms) + ")"
    return q

def _serper_query_variants(anchors: Dict[str, List[str]]) -> List[str]:
    """
    Returns up to 3 queries (specific → broad):
      1) Anchored: quoted/proper ANDed + OR keywords
      2) Token AND (>=3-char tokens or 4-digit years) + OR keywords
      3) Broad OR of keywords only
    """
    # 1) Anchored
    q1 = _build_provider_query_from_anchors(anchors, limit_terms=10, must_terms_cap=2)
    # 2) Token AND
    phrases = []
    for k in ("quoted","properish","num_units","years"):
        phrases += anchors.get(k, [])
    toks: List[str] = []
    seen: Set[str] = set()
    for ph in phrases:
        for w in re.split(r"\s+", _norm_space(ph)):
            wl = w.lower()
            if re.fullmatch(r"\d{4}", w) or (len(w) >= 3 and wl not in STOPWORDS):
                if wl not in seen:
                    seen.add(wl); toks.append(w)
    q2 = ""
    if toks:
        q2 = " AND ".join(toks[:8])
    kws = anchors.get("keywords", [])[:8]
    if kws:
        q2 += (" AND " if q2 else "") + "(" + " OR ".join(kws) + ")"
    # 3) Broad OR
    q3 = "(" + " OR ".join(anchors.get("keywords", [])[:10]) + ")" if anchors.get("keywords") else ""
    # dedupe
    out: List[str] = []
    seenq: Set[str] = set()
    for s in [q1, q2, q3]:
        sl = s.strip().lower()
        if sl and sl not in seenq:
            seenq.add(sl); out.append(s)
    return out[:12]

def _serper_fetch_with_meta(anchors: Dict[str, List[str]]) -> tuple[list[Dict[str, Any]], str, str]:
    """
    Return (items, source_label, query_used)
    source_label: "Serper(news)" | "Serper(web)" | "Serper" | "none"
    query_used: the first query variant that yielded items (or "")
    """
    if not (ENABLE_SERPER and SERPER_API_KEY):
        return [], "none", ""
    variants = _serper_query_variants(anchors)
    # Try news first
    for q in variants:
        j = _serper_post(SERPER_NEWS_URL, q, SERPER_LIMIT)
        items = _items_from_serper_news(j) if j else []
        print(f"[research] Serper(news) received {len(items)} | q={q[:120]!r}")
        if items:
            return items, "Serper(news)", q
    # Then web
    if ENABLE_SERPER_WEB:
        for q in variants:
            j = _serper_post(SERPER_WEB_URL, q, SERPER_LIMIT)
            items = _items_from_serper_web(j) if j else []
            print(f"[research] Serper(web)  received {len(items)} | q={q[:120]!r}")
            if items:
                return items, "Serper(web)", q
    return [], "Serper", variants[0] if variants else ""

def _asknews_client():
    try:
        if not (os.getenv("ENABLE_ASKNEWS","0").lower() in ("1","true","yes")):
            return None
        if not ASKNEWS_CLIENT_ID or not ASKNEWS_SECRET:
            return None
        if AskNewsSDK is None:
            return None
        return AskNewsSDK(ASKNEWS_CLIENT_ID, ASKNEWS_SECRET)
    except Exception:
        return None

def _fetch_asknews_candidates(title: str, description: str, criteria: str, anchors: Dict[str, List[str]]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Query AskNews using a compact anchor-driven query. Returns (items, hit_429_flag).
    """
    client = _asknews_client()
    if client is None:
        return [], False

    anchor_q = _build_provider_query_from_anchors(anchors, limit_terms=8, must_terms_cap=3)
    query = anchor_q or f"{title} {description or ''}".strip()
    per_request = min(int(os.getenv("ASKNEWS_ARTICLE_LIMIT","10")), int(os.getenv("ASKNEWS_PLAN_MAX_ARTICLES","10")))
    hit_429 = False
    strategies = [s.strip() for s in os.getenv("ASKNEWS_STRATEGIES","news knowledge,latest news,default").split(",") if s.strip()]

    for strategy in strategies:
        try:
            resp = client.search(query=query, strategy=strategy, size=per_request)
            # Normalize response (SDK returns a dataclass-like thing)
            if hasattr(resp, "as_dict"): res_dict = resp.as_dict  # type: ignore[attr-defined]
            elif isinstance(resp, dict): res_dict = resp
            else:
                try:
                    res_dict = json.loads(getattr(resp, "as_string"))
                except Exception:
                    res_dict = None
            items: List[Dict[str, Any]] = []
            if isinstance(res_dict, dict):
                arts = res_dict.get("articles") or res_dict.get("result") or []
                for a in arts:
                    items.append({
                        "title": a.get("title") or "",
                        "url": a.get("url") or "",
                        "outlet": a.get("source") or a.get("source_name") or _host_of(a.get("url") or ""),
                        "published": a.get("published_datetime") or a.get("date") or "",
                        "text": a.get("content") or a.get("text") or a.get("description") or a.get("snippet") or ""
                    })
            print(f"[research] AskNews call: strategy='{strategy}' size={per_request} -> received={len(items)} | query={query[:140]}")
            if items:
                return items, hit_429
        except Exception as e:
            msg = getattr(e,"message",repr(e))
            print(f"[research] AskNews error (strategy='{strategy}'): {type(e).__name__}: {msg}")
            if "429" in msg or (hasattr(e,"status_code") and getattr(e,"status_code")==429):
                hit_429 = True
                if os.getenv("ASKNEWS_BREAK_ON_429","1").lower() in ("1","true","yes"):
                    print("[research]   got 429; skipping remaining strategies to avoid hammering.")
                    break
    return [], hit_429

# =============================================================================
# LLM COMPOSITION
# =============================================================================

async def _compose_research_via_llm(prompt_text: str) -> tuple[str, str, dict]:
    """
    Try OpenRouter first (OPENROUTER_FALLBACK_ID), then Gemini Flash as fallback.
    Returns (text, used_model_id, usage_dict).
    """
    client = _get_or_client()
    if client is not None and os.getenv("OPENROUTER_FALLBACK_ID"):
        try:
            async with llm_semaphore:
                resp = await client.chat.completions.create(
                    model=os.getenv("OPENROUTER_FALLBACK_ID"),
                    messages=[{"role":"user","content":prompt_text}],
                    temperature=RESEARCH_TEMP,
                )
            text = (resp.choices[0].message.content or "").strip()
            usage = usage_to_dict(getattr(resp, "usage", None))
            return text, os.getenv("OPENROUTER_FALLBACK_ID",""), usage
        except Exception:
            pass
    # Gemini fallback (returns text only; usage/cost not available via this helper)
    try:
        gtxt = await _call_google(prompt_text, model="gemini-2.0-flash", temperature=RESEARCH_TEMP)
        if gtxt:
            return gtxt.strip(), "google/gemini-2.0-flash", {}
    except Exception:
        pass
    return "", "", {}

# =============================================================================
# UTIL: format sources for prompt/log
# =============================================================================

def _format_sources_for_prompt(items: List[Dict[str, Any]]) -> str:
    """
    Compact markdown list passed to the LLM (title/outlet/date/url + snippet).
    """
    if not items:
        return ""
    lines = []
    for it in items[:ASKNEWS_TOPK]:
        host = _host_of(it.get("url") or "")
        title = it.get("title") or it.get("url") or "(untitled)"
        url = it.get("url") or ""
        published = it.get("published") or ""
        snippet = (it.get("text") or "").strip()
        if snippet:
            snippet = snippet[:RESEARCH_SNIPPET_MAX_CHARS]
            lines.append(f"- {title} ({host}; {published}) — {url}\n  {snippet}")
        else:
            lines.append(f"- {title} ({host}; {published}) — {url}")
    return "\n".join(lines)

def _format_sources_for_log(items: List[Dict[str, Any]]) -> str:
    """
    Human-readable list (we include ALL kept items). Your .txt log will show this fully.
    """
    if not items:
        return "### Sources (autofetched)\n- *(none)*"
    lines = ["### Sources (autofetched)"]
    for it in items[:ASKNEWS_TOPK]:
        host = _host_of(it.get("url") or "")
        title = it.get("title") or it.get("url") or "(untitled)"
        url = it.get("url") or ""
        lines.append(f"- {title} ({host}) — {url}")
    return "\n".join(lines)

def _format_all_candidates_for_log(items: List[Dict[str, Any]]) -> str:
    """
    Optional verbose section listing pre-filter candidates for transparency/debug.
    Enable with RESEARCH_LOG_ALL_CANDIDATES=1
    """
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
    """
    Perform research and return (final_text, meta). See module docstring for meta fields.
    """
    allow_cache = _cache_allowed()
    cache_key = (slug or _norm_space(title))[:80]

    # 1) cache read
    if allow_cache:
        cached = read_cache("research_llm", cache_key)
        if isinstance(cached, dict) and cached.get("text"):
            final_text = _CAL_PREFIX + cached["text"]
            meta = {
                "research_llm": "",
                "research_source": "cache",
                "research_query": "",
                "research_n_raw": 0,
                "research_n_kept": 0,
                "research_cached": "1",
                "research_usage": {},
                "research_cost_usd": 0.0,
            }
            return final_text, meta

    # 2) Anchors + provider fetch
    anchors = extract_anchors(title, description, criteria)
    print(f"[research] Anchors: quoted={anchors['quoted']} years={anchors['years']} nums={anchors['num_units']} proper={anchors['properish']} keywords={anchors['keywords']}")

    raw_items: List[Dict[str, Any]] = []
    source_tag = "none"
    query_used = ""

    # Try AskNews first if enabled & configured
    ask_items: List[Dict[str, Any]] = []
    try:
        ask_items, _ = _fetch_asknews_candidates(title, description, criteria, anchors)
    except Exception as e:
        print(f"[research] AskNews fetch failed: {type(e).__name__}: {e!r}")
        ask_items = []

    if ask_items:
        raw_items = ask_items
        source_tag = "AskNews"
        anchor_q = _build_provider_query_from_anchors(anchors, limit_terms=8, must_terms_cap=3)
        query_used = anchor_q or f"{title} {description or ''}".strip()
    else:
        serper_items, source_tag, query_used = _serper_fetch_with_meta(anchors)
        raw_items = serper_items

    print(f"[research] Candidates fetched: {len(raw_items)} (source={source_tag})")

    # 3) strict filter → salvage
    strict = _rank_and_filter_items(raw_items, anchors, min_match=MIN_ANCHOR_MATCH)
    print(f"[research] After anchor filters: {len(strict)} items remain (min_match={MIN_ANCHOR_MATCH}, years_required={REQUIRE_YEAR_IF_PRESENT})")

    picked: List[Dict[str, Any]] = strict[:ASKNEWS_TOPK]
    if not picked and raw_items:
        salvage = _rank_and_filter_items(raw_items, anchors, min_match=SALVAGE_MIN_MATCH)
        if salvage:
            print(f"[research] Salvage pass: {len(salvage)} item(s) accepted with min_match={SALVAGE_MIN_MATCH}.")
            picked = salvage[:ASKNEWS_TOPK]

    # 4) Build prompt for the research LLM
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

    # 5) LLM compose
    llm_text, used_llm, usage = await _compose_research_via_llm(prompt)
    if not llm_text.strip():
        if picked:
            llm_text = "\n".join([f"- {it.get('title','')} ({it.get('url','')})" for it in picked])
        else:
            llm_text = "No recent external sources found; proceeding with general knowledge and base rates."

    # 6) Final text for the human log: brief + source list (+ optional pre-filter dump)
    parts = [_CAL_PREFIX + llm_text, "", _format_sources_for_log(picked)]
    if RESEARCH_LOG_ALL_CANDIDATES and raw_items:
        parts += ["", _format_all_candidates_for_log(raw_items)]
    final_text = "\n".join(parts)

    # 7) cache write
    if allow_cache and llm_text:
        try:
            write_cache("research_llm", cache_key, {"text": llm_text, "ts": ist_iso()})
        except Exception:
            pass

    # 8) meta
    cost = float(estimate_cost_usd(used_llm or "", usage)) if used_llm else 0.0
    meta = {
        "research_llm": used_llm or "",
        "research_source": source_tag,
        "research_query": query_used,
        "research_n_raw": int(len(raw_items)),
        "research_n_kept": int(len(picked)),
        "research_cached": "0",
        "research_usage": usage,
        "research_cost_usd": cost,
    }
    return final_text, meta
