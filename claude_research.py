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

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"
#: Research is reading and summarising, not the judgement the score depends
#: on; medium keeps the search loop focused and the spend predictable.
EFFORT = "medium"
#: Two rounds, same total as the single round they replace (10 searches):
#: round one surveys the question, round two goes after the specific facts
#: round one reports missing ("GAPS:"). The top MiniBench bots iterate on
#: what is missing (FutureSearch: "dozens of searches"; Ora and nostreambot:
#: gap-fill passes); a second round aimed at named gaps buys that with no
#: more searches than before, so the spend stays where it was.
ROUND1_SEARCHES = 6
ROUND2_SEARCHES = 4
MAX_SEARCHES = ROUND1_SEARCHES + ROUND2_SEARCHES
#: Round two runs only while round one stayed under this many dollars: the
#: Claude-API budget is a hard ~EUR100/month (the Max-plan API credit).
ROUND2_MAX_PRIOR_COST = 0.20
MAX_TOKENS = 16000
#: A long server-side search loop pauses after ~10 iterations; each resume is
#: another request. Bounded so a stuck loop cannot spend without limit.
MAX_RESUMES = 3

#: USD, for the logged estimate only. Search priced per 1,000 requests.
PRICE_IN, PRICE_OUT, PRICE_SEARCH = 2.0 / 1e6, 10.0 / 1e6, 10.0 / 1000

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


async def _round(client, messages: list[dict], max_searches: int) -> tuple:
    """One research round: (final response, tokens_in, tokens_out, searches)."""
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}]
    tokens_in = tokens_out = searches = 0
    response = None
    first_user = messages
    for _ in range(MAX_RESUMES + 1):
        response = await client.beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=messages,
            tools=tools,
            output_config={"effort": EFFORT},
            # A policy decline is re-run on a fallback model inside the same
            # call instead of returning nothing.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
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


def _cost(tokens_in: int, tokens_out: int, searches: int) -> float:
    return tokens_in * PRICE_IN + tokens_out * PRICE_OUT + searches * PRICE_SEARCH


async def research(prompt: str) -> str:
    """One research rundown for ``prompt``, using live web search."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    first = [{"role": "user", "content": prompt}]
    response, t_in, t_out, searches = await _round(client, first, ROUND1_SEARCHES)
    if response.stop_reason == "refusal":
        raise RuntimeError("claude_research refused")
    text = _text(response.content)
    rounds = 1

    gaps = _gaps(text)
    if text and gaps and _cost(t_in, t_out, searches) < ROUND2_MAX_PRIOR_COST:
        follow = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": ROUND2},
        ]
        try:
            r2, i2, o2, s2 = await _round(client, follow, ROUND2_SEARCHES)
            t_in, t_out, searches = t_in + i2, t_out + o2, searches + s2
            extra = _text(r2.content) if r2.stop_reason != "refusal" else ""
            if extra:
                text = f"{text}\n\n## Follow-up on the gaps\n{extra}"
                rounds = 2
        except Exception as exc:  # noqa: BLE001 - round one already stands
            logger.warning("claude_research_round2_failed reason=%s", type(exc).__name__)

    logger.info(
        "claude_research model=%s stop=%s rounds=%d searches=%d tokens_in=%d "
        "tokens_out=%d est_cost_usd=%.4f",
        response.model, response.stop_reason, rounds, searches, t_in, t_out,
        _cost(t_in, t_out, searches),
    )
    if not text:
        raise RuntimeError(f"claude_research returned no text (stop={response.stop_reason})")
    return text
