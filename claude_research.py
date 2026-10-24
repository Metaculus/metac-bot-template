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

What the forecasters get (2026-10-23)
-------------------------------------
A web-search answer comes back cut into a text block at every citation
boundary, often mid-sentence. Until 2026-10-23 the blocks were joined with
a newline, so the forecasters read sentences broken mid-clause (Test Bot
run 39813843358: "Treasury's own FY2025 figure is" / "$7,010.0 billion"),
and every URL, title and page date the search returned was thrown away,
although the forecasting prompts tell them to weigh facts by how recent
they are. Now adjacent text blocks are joined as one passage (_text), and
the rundown ends with up to MAX_SOURCES "## Sources" lines: title, the
date the search gives for the page, URL (_sources).

A turn paused by the server (pause_turn) is resumed by sending everything
it produced so far back as the assistant turn; each response carries only
what it added. _round used to resend only the latest response, so a second
resume dropped the first part from the conversation, and only the last
response's text reached the rundown. The 155 calls logged 19-23 Oct all
ended on end_turn, but intermediate pauses were not logged; the log line's
resumes=N counts them now.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import spend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Engine:
    model: str
    web_search: str          # tool type this model accepts
    price_in: float          # USD per input token
    price_out: float
    fallbacks: bool          # server-side refusal fallback available
    cache_read: float = 0.10  # cache hit, as a multiple of price_in
    #: A request whose prompt (uncached + cache writes + cache reads) exceeds
    #: long_prompt tokens pays long_factor times every token price.
    long_prompt: int | None = None
    long_factor: float = 1.0


#: Sonnet 5.5 takes the dynamic-filtering web search; Haiku 5.5 only the basic
#: variant and has no server-side refusal fallback. Haiku 5.5 is ~20x cheaper
#: per token ($0.10/$0.50 vs $2/$10 per M), so the same money buys more
#: searches; which reads better is decided by an A/B on Test Bot, not assumed.
#: Prices as published on platform.claude.com/docs/en/about-claude/pricing
#: (read 2026-10-21): cache hits are 0.05x on Sonnet 5.5 and 0.1x on Haiku
#: 5.5, and Haiku 5.5 prompts over 100,000 tokens pay $0.50/$2.50 (5x) -- a
#: long search loop reaches that, so each request is priced on its own size.
ENGINES = {
    "sonnet": Engine("claude-sonnet-5-5", "web_search_20260209", 2.0 / 1e6, 10.0 / 1e6, True,
                     cache_read=0.05),
    "haiku": Engine("claude-haiku-5-5", "web_search_20250305", 0.10 / 1e6, 0.50 / 1e6, False,
                    long_prompt=100_000, long_factor=5.0),
}
#: Haiku since 2026-10-18. A/B on the same nine practice questions (Test Bot
#: runs 39091393405 vs 39091407899): Sonnet $0.243/question, 5-13 searches,
#: several "could not read the figures"; Haiku $0.172/question, 11-20
#: searches over three gap-driven rounds, 3-4x the dated facts (e.g. DDHQ,
#: Poliwave and Kalshi odds with dates where Sonnet had none). Cheaper and
#: better, so it is the default; Sonnet stays one variable away.
DEFAULT_ENGINE = "haiku"

#: Searches per round, by depth. "full" is for the tournaments the bot can win
#: money in (FutureEval, Market Pulse); "light" for MiniBench and the unpaid
#: Cup, so the ~EUR100/month Claude-API budget (the Max-plan credit; hard
#: limit) goes where the prizes are. A round after the first runs only
#: when the previous one named gaps and the question's spend is under
#: ROUND_MAX_PRIOR_COST.
ROUNDS = {
    ("sonnet", "full"): (8, 5),
    ("sonnet", "light"): (4,),
    # Full is for FutureEval and Market Pulse only since 2026-10-23 (MiniBench
    # went light, which paid for it); 102 production calls at this plan
    # averaged 12 searches and $0.145. It was cut to (8, 4) for a day while
    # MiniBench still ran full.
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
#: Source lines at the end of a rundown, about 600 tokens of forecaster
#: prompt (sponsored OpenRouter key). Pages the rundown cites come first.
MAX_SOURCES = 15
SOURCE_TITLE_CHARS = 120
#: "$10 per 1,000 searches, plus standard token costs for search-generated
#: content"; a search that errors is not billed (pricing page, 2026-10-21).
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
    """The prose of a turn. Adjacent text blocks are one passage cut at its
    citations, so they are joined with nothing between them; a search call,
    its results or any other block between two text blocks ends a passage,
    and passages are separated by a blank line."""
    passages: list[str] = []
    run: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            run.append(block.text or "")
        elif run:
            passages.append("".join(run))
            run = []
    if run:
        passages.append("".join(run))
    return "\n\n".join(p.strip() for p in passages if p.strip())


def _clean(value) -> str:
    return " ".join(str(value).split()) if value else ""


def _sources(content) -> tuple[str, int]:
    """A "## Sources" section for the pages behind ``content``, and how many
    lines it has ("" and 0 when there are none).

    One line per unique URL: the pages the text cites, in citation order,
    then the other search results in the order they came back, up to
    MAX_SOURCES. Citations carry the URL and title; the date (page_age, when
    the page was last updated) is only on the search results.
    """
    pages: dict[str, dict] = {}
    cited: dict[str, None] = {}   # insertion-ordered sets
    found: dict[str, None] = {}

    def page(url, title, age=None) -> str:
        url = _clean(url)
        if url:
            entry = pages.setdefault(url, {"title": "", "age": ""})
            entry["title"] = entry["title"] or _clean(title)
            entry["age"] = entry["age"] or _clean(age)
        return url

    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            for citation in getattr(block, "citations", None) or ():
                url = page(getattr(citation, "url", None), getattr(citation, "title", None))
                if url:
                    cited[url] = None
        elif kind == "web_search_tool_result":
            results = getattr(block, "content", None)
            if not isinstance(results, list):  # a search error is one object, not a list
                continue
            for result in results:
                url = page(getattr(result, "url", None), getattr(result, "title", None),
                           getattr(result, "page_age", None))
                if url:
                    found[url] = None

    order = [*cited, *(u for u in found if u not in cited)][:MAX_SOURCES]
    if not order:
        return "", 0
    n_cited = sum(1 for u in order if u in cited)
    if n_cited == len(order):
        head = "cited above"
    elif n_cited == 0:
        head = "search results"
    else:
        head = f"the first {n_cited} are cited above, the rest are other search results"
    lines = [f"## Sources ({head}; dates are when each page was last updated)"]
    for url in order:
        title = pages[url]["title"]
        if len(title) > SOURCE_TITLE_CHARS:
            title = title[:SOURCE_TITLE_CHARS - 3].rstrip() + "..."
        age = pages[url]["age"]
        lines.append("- " + " ".join(x for x in (title, f"({age})" if age else "", url) if x))
    return "\n".join(lines), len(order)


def _gaps(text: str) -> str:
    """The GAPS line's content, or "" when there is none or it says none."""
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("GAPS:"):
            gaps = line.split(":", 1)[1].strip()
            return "" if gaps.lower().rstrip(".") in ("", "none", "n/a") else gaps
    return ""


#: Prompt caching (2026-10-21, after Anthropic flagged a low cache-hit rate
#: on this org). With caching on the request, the server-side web search loop
#: writes a cache entry after each batch of search results, so every later
#: step of the same request -- and each pause_turn resume and follow-up round,
#: which repeat the system prompt and question -- re-reads that context at
#: a fraction of the full input price (Engine.cache_read). Writes cost 1.25x.
#: Minimum cacheable prefix on Haiku 5.5 / Sonnet 5.5 is 512 tokens; the
#: prompt alone is above that.
CACHE = {"type": "ephemeral"}
CACHE_WRITE = 1.25


def request_cost(engine: Engine, uncached: int, cache_write: int, cache_read: int,
                 out: int, searches: int) -> float:
    """USD for one API request, as billed."""
    factor = 1.0
    if engine.long_prompt is not None and uncached + cache_write + cache_read > engine.long_prompt:
        factor = engine.long_factor
    tokens = ((uncached + CACHE_WRITE * cache_write + engine.cache_read * cache_read) * engine.price_in
              + out * engine.price_out)
    return tokens * factor + searches * PRICE_SEARCH


@dataclass
class Tally:
    """Token and search counts, and their price, across every request of one
    research call."""

    engine: Engine
    uncached: int = 0
    cache_write: int = 0
    cache_read: int = 0
    out: int = 0
    searches: int = 0
    usd: float = 0.0
    resumes: int = 0   # requests that continued a paused turn

    def add(self, usage) -> None:
        uncached = usage.input_tokens or 0
        write = usage.cache_creation_input_tokens or 0
        read = usage.cache_read_input_tokens or 0
        out = usage.output_tokens or 0
        searches = 0
        if usage.server_tool_use is not None:
            searches = usage.server_tool_use.web_search_requests or 0
        self.uncached += uncached
        self.cache_write += write
        self.cache_read += read
        self.out += out
        self.searches += searches
        self.usd += request_cost(self.engine, uncached, write, read, out, searches)

    @property
    def tokens_in(self) -> int:
        return self.uncached + self.cache_write + self.cache_read


@dataclass(frozen=True)
class Turn:
    """One round's assistant turn: the blocks of every request it took, in
    order, and the last request's stop reason and model."""

    stop_reason: str
    model: str
    content: list


async def _round(client, engine: Engine, messages: list[dict], max_searches: int,
                 tally: Tally) -> Turn:
    """One research round, resumed while the server pauses it; adds every
    request's usage to ``tally`` and returns the whole turn."""
    tools = [{"type": engine.web_search, "name": "web_search", "max_uses": max_searches}]
    extra: dict = {}
    if engine.fallbacks:
        # A policy decline is re-run on a fallback model inside the same call
        # instead of returning nothing.
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    content: list = []
    response = None
    for attempt in range(MAX_RESUMES + 1):
        if attempt:
            # Resume: everything the turn has produced goes back unchanged as
            # the assistant turn, and the server continues from its trailing
            # server_tool_use block. Append-only, so the cache written by the
            # previous request is this one's prefix.
            tally.resumes += 1
        response = await client.beta.messages.create(
            model=engine.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[*messages, {"role": "assistant", "content": content}] if content else messages,
            tools=tools,
            output_config={"effort": EFFORT},
            cache_control=CACHE,
            **extra,
        )
        tally.add(response.usage)
        # A continued turn's response holds only the blocks it added.
        content = [*content, *response.content]
        if response.stop_reason != "pause_turn":
            break
    return Turn(response.stop_reason, response.model, content)


async def research(prompt: str, depth: str = "full", engine_key: str | None = None) -> str:
    """One research rundown for ``prompt``, using live web search.

    Round one surveys the question and ends with a GAPS line; each later round
    (as many as ROUNDS allows for this engine and depth) searches only the
    gaps the previous round named, while the spend stays under
    ROUND_MAX_PRIOR_COST. The rundown ends with the pages behind it (_sources).
    """
    import anthropic

    key = engine_key or engine_name()
    engine = ENGINES[key]
    plan = ROUNDS.get((key, depth)) or ROUNDS[(key, "full")]
    client = anthropic.AsyncAnthropic()
    tally = Tally(engine)
    try:
        return await _research(client, key, engine, plan, depth, prompt, tally)
    finally:
        # Recorded even when the call fails: the requests already made are billed.
        if tally.usd:
            spend.record(provider="anthropic", model=engine.model, usd=tally.usd,
                         tokens_in=tally.tokens_in, tokens_out=tally.out,
                         searches=tally.searches)


async def _research(client, key: str, engine: Engine, plan: tuple, depth: str,
                    prompt: str, tally: Tally) -> str:
    first = [{"role": "user", "content": prompt}]
    response = await _round(client, engine, first, plan[0], tally)
    if response.stop_reason == "refusal":
        raise RuntimeError("claude_research refused")
    text = _text(response.content)
    used = list(response.content)  # the blocks behind the text, for its sources
    rounds = 1

    for budget in plan[1:]:
        gaps = _gaps(text)
        if not (text and gaps) or tally.usd >= ROUND_MAX_PRIOR_COST:
            break
        follow = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": ROUND2},
        ]
        try:
            rn = await _round(client, engine, follow, budget, tally)
        except Exception as exc:  # noqa: BLE001 - earlier rounds already stand
            logger.warning("claude_research_round%d_failed reason=%s", rounds + 1, type(exc).__name__)
            break
        extra_text = _text(rn.content) if rn.stop_reason != "refusal" else ""
        if not extra_text:
            break
        rounds += 1
        used.extend(rn.content)
        text = f"{text}\n\n## Follow-up on the gaps (round {rounds})\n{extra_text}"

    # Optional: a failure here costs the source list, never the rundown.
    sources, n_sources = "", 0
    if text:
        try:
            sources, n_sources = _sources(used)
        except Exception as exc:  # noqa: BLE001
            logger.warning("claude_research_sources_failed reason=%s", type(exc).__name__)

    logger.info(
        "claude_research engine=%s depth=%s model=%s stop=%s rounds=%d resumes=%d "
        "searches=%d sources=%d tokens_in=%d cache_read=%d cache_write=%d tokens_out=%d "
        "est_cost_usd=%.4f",
        key, depth, response.model, response.stop_reason, rounds, tally.resumes,
        tally.searches, n_sources, tally.tokens_in, tally.cache_read, tally.cache_write,
        tally.out, tally.usd,
    )
    if not text:
        raise RuntimeError(f"claude_research returned no text (stop={response.stop_reason})")
    return f"{text}\n\n{sources}" if sources else text
