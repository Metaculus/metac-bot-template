# ANCHOR: topic_classify (paste whole file)
from __future__ import annotations
"""
topic_classify.py — Lightweight topic & GTMC1-activation classifier

What this does
--------------
- Primary: Uses a low-temperature LLM (via your existing OpenRouter client) to classify a Metaculus
  question into a compact taxonomy AND return an `is_strategic` flag suitable for GTMC1 activation.
- Fallback: If the LLM is not available/errors, falls back to a deterministic keyword rule.
- Caching: Stores per-question results so repeated runs are fast, cheap, and reproducible.

Why this helps
--------------
- Your current keyword-only gate is brittle (misses edge phrasing; false positives on noisy words).
- An LLM prompt with explicit class definitions + strict JSON schema gives better recall/precision.
- We keep it reproducible: temperature=0.1, stable prompt, and cached by slug.

Integration points
------------------
- Import `should_run_gtmc1(...)` in cli.py and use it instead of the old `looks_strategic`.
- Optionally log the returned `classification` dict for transparency.

Environment knobs (optional)
----------------------------
- SPAGBOT_USE_LLM_CLASSIFIER=1  (default: on; set 0 to force keyword fallback)
- SPAGBOT_DISABLE_CLASSIFIER_CACHE=1  (default: cache enabled)

Dependencies
------------
- Reuses your existing OpenRouter client & semaphore from providers.py
- Reuses read_cache / write_cache from config.py
"""

import os, re, json, asyncio
from typing import Dict, Any, Optional, Tuple

# Reuse your infra
from .providers import _get_or_client, llm_semaphore
from .config import read_cache, write_cache
from .providers import OPENROUTER_FALLBACK_ID  # model id you already use

# ---------------------------
# Taxonomy (keep concise)
# ---------------------------
CLASS_DEFS: Dict[str, str] = {
    "geopolitics": "Interstate relations, war/peace, sanctions, diplomacy, alliances, security.",
    "politics":    "Domestic governance: elections, parties, legislation, courts, public policy.",
    "technology":  "Computing/AI/space/robotics. Technical progress, benchmarks, launches.",
    "science":     "Non-AI science: physics, astronomy, materials, experimental results.",
    "economy":     "Macro/markets: inflation, GDP, FX, trade, central banks, commodities.",
    "business":    "Company-level outcomes: earnings, launches, adoption, M&A, product milestones.",
    "health":      "Medicine/biotech/epidemics: trials, approvals, public health metrics.",
    "environment": "Climate, emissions, weather extremes, disasters, ecology, energy mix.",
    "society":     "Demographics, education, culture, crime, norms, social media, surveys.",
    "sports":      "Sports leagues/tournaments/matches, medals, records.",
    "energy":      "Oil/gas/OPEC, electricity markets, renewables, nuclear power (non-weapons).",
    "xrisk":       "Global catastrophic/existential risk: extinction, nuclear exchange, engineered bio.",
    "other":       "Anything else or meta-questions about forecasting/platform methods."
}

# Minimal keyword fallback (only used if LLM is off/unavailable)
_FALLBACK_PATTERNS: Dict[str, str] = {
    "geopolitics": r"\b(war|conflict|invasion|cease[- ]?fire|airstrike|sanction|treaty|diplom|alliance|NATO|UN)\b",
    "politics":    r"\b(election|vote|parliament|congress|senate|minister|legislation|referendum)\b",
    "technology":  r"\b(AI|artificial intelligence|LLM|benchmark|GPT|transformer|SpaceX|Starship|satellite|launch)\b",
    "economy":     r"\b(GDP|inflation|CPI|unemployment|recession|central bank|interest rate|currency|FX)\b",
    "business":    r"\b(revenue|earnings|profit|IPO|merger|acquisition|product launch|users)\b",
    "health":      r"\b(trial|phase\s?(I|II|III)|FDA|EMA|vaccine|pandemic|epidemic|infection)\b",
    "environment": r"\b(climate|emission|CO2|hurricane|heatwave|wildfire|flood|drought)\b",
    "society":     r"\b(population|fertility|crime|education|survey|social media)\b",
    "sports":      r"\b(World Cup|Olympics|league|tournament|playoffs?)\b",
    "energy":      r"\b(oil|gas|OPEC|pipeline|renewable|solar|wind|grid|megawatt|nuclear)\b",
    "xrisk":       r"\b(extinct(ion)?|existential risk|x[- ]?risk|nuclear war|engineered pathogen)\b",
}

# Strategic definition for GTMC1 activation:
#   A question is strategic if it primarily concerns interstate bargaining/conflict,
#   sanctions/alliances, or multi-actor political bargaining with coercive leverage.
def _fallback_is_strategic(cls: str, blob: str) -> bool:
    if cls in ("geopolitics", "politics", "xrisk"):
        return True
    # Also catch words like "ceasefire", "sanction", "alliance" even if class drifted
    return bool(re.search(r"\b(cease[- ]?fire|sanction|alliance|invasion|occupation|airstrike)\b", blob, flags=re.I))

def _fallback_classify(title: str, description: str, criteria: str) -> Dict[str, Any]:
    blob = " ".join([title or "", description or "", criteria or ""]).strip()
    scores: Dict[str, float] = {}
    for cls, pat in _FALLBACK_PATTERNS.items():
        if re.search(pat, blob, flags=re.I):
            scores[cls] = 1.0
    primary = max(scores, key=scores.get) if scores else "other"
    return {
        "source": "keywords",
        "primary": primary,
        "secondary": None,
        "is_strategic": _fallback_is_strategic(primary, blob),
        "strategic_score": 1.0 if _fallback_is_strategic(primary, blob) else 0.0,
        "rationale": "keyword fallback",
        "scores": scores,
    }

def _build_llm_prompt(title: str, description: str, criteria: str) -> str:
    defs = "\n".join([f"- {k}: {v}" for k,v in CLASS_DEFS.items()])
    return f"""You are a classification assistant for Metaculus questions.

Task: Assign the question to ONE primary class (from the list) and optionally ONE secondary class.
Also decide if the question is STRATEGIC (game-theoretic bargaining/coalitions/interstate conflict)
to determine whether a bargaining simulator (GTMC1) should run.

Classes:
{defs}

Rules:
- Output STRICT JSON ONLY. No commentary. Keys exactly as specified.
- "primary" must be one of: {", ".join(CLASS_DEFS.keys())}
- "secondary" is either one of those or null.
- "is_strategic": true/false. Use true for interstate conflict/alliances/sanctions/war OR
  multi-actor political bargaining where coalition, coercion, or side-payments are central.
- "strategic_score": a number 0..1 reflecting confidence in strategic framing.
- "rationale": one short sentence.

Input:
TITLE: {title}
DESCRIPTION: {description}
CRITERIA: {criteria}

JSON schema:
{{
  "primary": "geopolitics|politics|technology|science|economy|business|health|environment|society|sports|energy|xrisk|other",
  "secondary": "geopolitics|politics|technology|science|economy|business|health|environment|society|sports|energy|xrisk|other|null",
  "is_strategic": true,
  "strategic_score": 0.0,
  "rationale": "string (<= 140 chars)"
}}
"""

async def classify_topics(
    title: str, description: str, criteria: str, slug: str
) -> Dict[str, Any]:
    """
    Returns a dict with fields: source, primary, secondary, is_strategic, strategic_score, rationale, scores?
    Uses cache if available. LLM first (if enabled), else keyword fallback.
    """
    # Cache first (accept dict or JSON string)
    if os.getenv("SPAGBOT_DISABLE_CLASSIFIER_CACHE","0").lower() not in ("1","true","yes"):
        cached = read_cache("topic_classify", slug)
        if isinstance(cached, dict) and cached.get("primary"):
            return cached
        if isinstance(cached, (str, bytes)):
            try:
                parsed = json.loads(cached)
                if isinstance(parsed, dict) and parsed.get("primary"):
                    return parsed
            except Exception:
                pass

    use_llm = os.getenv("SPAGBOT_USE_LLM_CLASSIFIER","1").lower() in ("1","true","yes")
    client = _get_or_client() if use_llm else None

    if client is None:
        result = _fallback_classify(title, description, criteria)
    else:
        prompt = _build_llm_prompt(title, description, criteria)
        try:
            async with llm_semaphore:
                resp = await client.chat.completions.create(
                    model=OPENROUTER_FALLBACK_ID,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # low temp for reproducibility
                )
            text = (resp.choices[0].message.content or "").strip()
            # strip fences if present
            text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S)
            data = json.loads(text)
            # validate minimal keys
            primary = str(data.get("primary") or "other")
            if primary not in CLASS_DEFS: primary = "other"
            secondary = data.get("secondary")
            if secondary not in CLASS_DEFS: secondary = None
            is_strategic = bool(data.get("is_strategic"))
            strategic_score = float(data.get("strategic_score") or 0.0)
            rationale = (data.get("rationale") or "").strip()[:140]
            result = {
                "source": "llm",
                "primary": primary,
                "secondary": secondary,
                "is_strategic": is_strategic,
                "strategic_score": max(0.0, min(1.0, strategic_score)),
                "rationale": rationale,
                "scores": data.get("scores") or {},  # optional
            }
        except Exception as e:
            # On any error, fall back
            result = _fallback_classify(title, description, criteria)
            result["rationale"] = f"fallback after LLM error: {e!r}"

    # Cache write (best-effort)
    try:
        write_cache("topic_classify", slug, result)
    except Exception:
        pass

    return result

async def should_run_gtmc1(
    title: str, description: str, criteria: str, slug: str
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience wrapper: returns (use_gtmc1_boolean, classification_dict).
    """
    info = await classify_topics(title, description, criteria, slug)
    # You can add a threshold if you want to be stricter, e.g., score >= 0.45
    threshold = float(os.getenv("GTMC1_ACTIVATION_THRESHOLD", "0.35"))
    decision = bool(info.get("is_strategic")) and float(info.get("strategic_score", 0.0)) >= threshold
    return decision, info
