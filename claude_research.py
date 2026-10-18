"""Research with live web search, through the Anthropic API directly.

Why this exists
---------------
Until now the researcher was Claude Haiku answering from training memory: no
search, no news, nothing dated after its cutoff. On event questions that is
how the bot ended up confidently wrong about things that had already happened
or were about to, and on price questions it centred distributions on a level
remembered from months ago. Metaculus' Fall 2025 survey found the number of
distinct research sources the strongest predictor of a winning bot (r=0.42).

The Anthropic API runs web search server-side (``web_search_20260209``): the
model issues the searches, reads the results and writes the rundown in one
request, with no search provider of our own to call or pay separately.

Active only when ``ANTHROPIC_API_KEY`` is set (a repo secret). Any failure
raises, and main.run_research falls back to the researcher it used before, so
a missing key, an exhausted credit balance or an outage costs research
quality on that question, never the forecast.

Cost, logged per call so it can be read off the Actions log: Sonnet 5.5 at
$2 / $10 per million input / output tokens plus web search requests. The
per-call cap on searches is MAX_SEARCHES.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Engine:
    model: str
    web_search: str          # tool type this model accepts
    price_in: float          # USD per input token (logged estimate only)
    price_out: float
    fallbacks: bool          # server-side refusal fallback available


#: Sonnet 5.5 takes the dynamic-filtering web search; Haiku 5.5 only the basic
#: variant and has no server-side refusal fallback. Haiku 5.5 is ~20x cheaper
#: per token ($0.10/$0.50 vs $2/$10 per M), so the same money buys more
#: searches; which reads better is decided by an A/B on Test Bot, not assumed.
ENGINES = {
    "sonnet": Engine("claude-sonnet-5-5", "web_search_20260209", 2.0 / 1e6, 10.0 / 1e6, True),
    "haiku": Engine("claude-haiku-5-5", "web_search_20250305", 0.10 / 1e6, 0.50 / 1e6, False),
}
#: Haiku since 2026-10-18. A/B on the same nine practice questions (Test Bot
#: runs 39091393405 vs 39091407899): Sonnet $0.243/question, 5-13 searches,
#: several "could not read the figures"; Haiku $0.172/question, 11-20
#: searches over three gap-driven rounds, 3-4x the dated facts (e.g. DDHQ,
#: Poliwave and Kalshi odds with dates where Sonnet had none). Cheaper and
#: better, so it is the default; Sonnet stays one variable away.
DEFAULT_ENGINE = "haiku"

#: Searches per round, by depth. "full" is for prize tournaments (FutureEval,
#: MiniBench, Market Pulse); "light" for the unpaid Cup, whose weekly
#: refreshes otherwise take a large share of the ~EUR100/month Claude-API
#: budget (the Max-plan credit; hard limit). A round after the first runs only
#: when the previous one named gaps and the question's spend is under
#: ROUND_MAX_PRIOR_COST.
ROUNDS = {
    ("sonnet", "full"): (8, 5),
    ("sonnet", "light"): (4,),
    ("haiku", "full"): (10, 6, 4),
    ("haiku", "light"): (6,),
    # One targeted pass on what the forecasters disagreed about
    # (disagreement.py); runs only on the questions where they did.
    ("sonnet", "crux"): (4,),
    ("haiku", "crux"): (6,),
}
ROUND_MAX_PRIOR_COST = 0.22

#: Kept for callers and tests that read the defaults.
MODEL = ENGINES[DEFAULT_ENGINE].model
ROUND1_SEARCHES = ROUNDS[(DEFAULT_ENGINE, "full")][0]
ROUND2_SEARCHES = ROUNDS[(DEFAULT_ENGINE, "full")][1]
MAX_SEARCHES = sum(ROUNDS[(DEFAULT_ENGINE, "full")])
ROUND2_MAX_PRIOR_COST = ROUND_MAX_PRIOR_COST
#: Research is reading and summarising, not the judgement the score depends
#: on; medium keeps the search loop focused and the spend predictable.
EFFORT = "medium"
MAX_TOKENS = 16000
#: A long server-side search loop pauses after ~10 iterations; each resume is
#: another request. Bounded so a stuck loop cannot spend without limit.
MAX_RESUMES = 3
PRICE_SEARCH = 10.0 / 1000
PRICE_IN, PRICE_OUT = ENGINES[DEFAULT_ENGINE].price_in, ENGINES[DEFAULT_ENGINE].price_out


def engine_name() -> str:
    name = os.environ.get("CLAUDE_RESEARCH_ENGINE", "").strip().lower() or DEFAULT_ENGINE
    return name if name in ENGINES else DEFAULT_ENGINE


SYSTEM = (
    "You research forecasting questions for a professional forecaster. Search "
    "the web for the most recent, relevant facts before answering. Prefer "
    "primary and dated sources. Report the current value of any quantity the "
    "question depends on, recent developments, scheduled events before the "
    "resolution date, and base rates for comparable events. Date every fact. "
    "If you cannot find recent information, say so rather than filling the "
    "gap from memory. Do not forecast and do not say how the question will "
    "resolve. End with one line starting 'GAPS:' listing up to three specific "
    "facts that matter for the question and that you could not find, or "
    "'GAPS: none'."
)

ROUND2 = (
    "Here is your first rundown for the question above. Search specifically "
    "for the facts listed after 'GAPS:' and report only what you find about "
    "them, dated, in the same style. If a gap cannot be filled, say so in one "
    "line. End with an updated 'GAPS:' line."
)


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _text(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _gaps(text: str) -> str:
    """The GAPS line's content, or "" when there is none or it says none."""
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("GAPS:"):
            gaps = line.split(":", 1)[1].strip()
            return "" if gaps.lower().rstrip(".") in ("", "none", "n/a") else gaps
    return ""


async def _round(client, engine: Engine, messages: list[dict], max_searches: int) -> tuple:
    """One research round: (final response, tokens_in, tokens_out, searches)."""
    tools = [{"type": engine.web_search, "name": "web_search", "max_uses": max_searches}]
    extra: dict = {}
    if engine.fallbacks:
        # A policy decline is re-run on a fallback model inside the same call
        # instead of returning nothing.
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    tokens_in = tokens_out = searches = 0
    response = None
    first_user = messages
    for _ in range(MAX_RESUMES + 1):
        response = await client.beta.messages.create(
            model=engine.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=messages,
            tools=tools,
            output_config={"effort": EFFORT},
            **extra,
        )
        usage = response.usage
        tokens_in += (usage.input_tokens or 0) + (usage.cache_read_input_tokens or 0) \
            + (usage.cache_creation_input_tokens or 0)
        tokens_out += usage.output_tokens or 0
        if usage.server_tool_use is not None:
            searches += usage.server_tool_use.web_search_requests or 0
        if response.stop_reason != "pause_turn":
            break
        # Resume: the server continues from the trailing server_tool_use block.
        messages = [*first_user, {"role": "assistant", "content": response.content}]
    return response, tokens_in, tokens_out, searches


def _cost(engine: Engine, tokens_in: int, tokens_out: int, searches: int) -> float:
    return tokens_in * engine.price_in + tokens_out * engine.price_out + searches * PRICE_SEARCH


async def research(prompt: str, depth: str = "full", engine_key: str | None = None) -> str:
    """One research rundown for ``prompt``, using live web search.

    Round one surveys the question and ends with a GAPS line; each later round
    (as many as ROUNDS allows for this engine and depth) searches only the
    gaps the previous round named, while the spend stays under
    ROUND_MAX_PRIOR_COST.
    """
    import anthropic

    key = engine_key or engine_name()
    engine = ENGINES[key]
    plan = ROUNDS.get((key, depth)) or ROUNDS[(key, "full")]
    client = anthropic.AsyncAnthropic()
    first = [{"role": "user", "content": prompt}]
    response, t_in, t_out, searches = await _round(client, engine, first, plan[0])
    if response.stop_reason == "refusal":
        raise RuntimeError("claude_research refused")
    text = _text(response.content)
    rounds = 1

    for budget in plan[1:]:
        gaps = _gaps(text)
        if not (text and gaps) or _cost(engine, t_in, t_out, searches) >= ROUND_MAX_PRIOR_COST:
            break
        follow = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": ROUND2},
        ]
        try:
            rn, i2, o2, s2 = await _round(client, engine, follow, budget)
        except Exception as exc:  # noqa: BLE001 - earlier rounds already stand
            logger.warning("claude_research_round%d_failed reason=%s", rounds + 1, type(exc).__name__)
            break
        t_in, t_out, searches = t_in + i2, t_out + o2, searches + s2
        extra_text = _text(rn.content) if rn.stop_reason != "refusal" else ""
        if not extra_text:
            break
        rounds += 1
        text = f"{text}\n\n## Follow-up on the gaps (round {rounds})\n{extra_text}"

    logger.info(
        "claude_research engine=%s depth=%s model=%s stop=%s rounds=%d searches=%d "
        "tokens_in=%d tokens_out=%d est_cost_usd=%.4f",
        key, depth, response.model, response.stop_reason, rounds, searches, t_in, t_out,
        _cost(engine, t_in, t_out, searches),
    )
    if not text:
        raise RuntimeError(f"claude_research returned no text (stop={response.stop_reason})")
    return text
