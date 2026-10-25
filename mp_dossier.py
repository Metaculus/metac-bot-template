"""Earnings dossiers for Market Pulse companies, researched ahead of time.

Why this exists
---------------
Market Pulse 26Q4 is spot scored at each question's close, and its numeric
groups ask for a company's first-reported revenue, GAAP diluted EPS and
guidance. On the first six resolved subquestions the human community beat the
field by +31 to +37 a question against the bot's ~17, and the biggest misses
were accounting, not markets: Tesla resolved at a GAAP 0.22 against a 0.451
street consensus, and Meta's GAAP EPS fell below the question's whole range.
The per-question research (Haiku, about twelve searches) rarely gets as far
as a company's own reconciliation items.

So once, ahead of the closes, every company with an open Market Pulse
question gets one deep research pass (Opus 5.5 with dynamic-filtering web
search, two gap-driven rounds) into a dossier of dated, sourced FACTS: the
release date, consensus, the company's guidance, the items that will move GAAP
away from the non-GAAP consensus this quarter, the recent history and what
changed since the last report. It never forecasts. The owner funded the first
run with Claude credit that would otherwise have lapsed (2026-10-25).

Where it lives: mp_dossiers.yaml writes the JSON to the Actions cache, never
the repository; run_bot_on_tournament.yaml restores it and section() adds a
company's dossier to the research of that company's Market Pulse questions,
below the fresh research, which outranks it. Logs carry counts and dollars
only. Nothing in the bot path raises.

    python mp_dossier.py build --budget 18 --out mp_dossiers.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Sequence

import market_quant

logger = logging.getLogger(__name__)

ENGINE = "opus"
DEPTH = "dossier"
EFFORT = "high"
#: A second round runs while the first cost less than this (claude_research's
#: own cap is sized for Haiku's ~$0.15 calls).
MAX_PRIOR_COST = 3.0
#: Held back before each company so the total stays under the budget: about
#: the most one two-round Opus dossier has to cost.
EST_COST = 2.5
#: A dossier older than this is left out; the next quarter needs a new one.
MAX_AGE = timedelta(days=21)
#: Characters of dossier added to a question's research.
MAX_CHARS = 6000


_METRIC_WORDS = re.compile(r"revenue|margin|expense|guidance|income|earnings|eps|cash|capex", re.I)


def _company(question: Any) -> tuple[str, str] | None:
    """(key, display name) of the one company a question is about: from its
    group option, else from the group title (the NVIDIA guidance group's
    options are metrics, not companies)."""
    option = str(getattr(question, "group_question_option", "") or "").strip()
    title = str(getattr(question, "question_text", "") or "")
    for text in (option, title):
        if not text:
            continue
        symbol = market_quant.ticker(SimpleNamespace(group_question_option=None, question_text=text))
        if symbol:
            return symbol, (option if text is option else symbol)
    if option and not re.search(r"\d", option) and not _METRIC_WORDS.search(option):
        return option.lower(), option  # a private company (SpaceX): researched by name
    return None


def _metric(question: Any) -> str:
    text = " ".join(str(getattr(question, a, "") or "") for a in ("question_text", "resolution_criteria"))
    option = str(getattr(question, "group_question_option", "") or "").strip()
    if re.search(r"earnings per share|\bEPS\b", text, re.I):
        gaap = " (GAAP diluted, as first reported)" if re.search(r"(?<!non-)\bGAAP\b", text) else ""
        return "earnings per share" + gaap
    if re.search(r"\brevenues?\b", text, re.I) and not re.search(r"guidance", text, re.I):
        return "total revenue, as first reported"
    if re.search(r"guidance", text, re.I):
        return f"forward guidance for {option or 'the next quarter'}"
    return option or "the quantities in the question"


def plan(questions: Sequence[Any]) -> dict[str, dict]:
    """One entry per company: its name, what is asked about it and its
    earliest close, so the dossiers are built soonest-close first."""
    out: dict[str, dict] = {}
    for question in questions:
        company = _company(question)
        if company is None:
            continue
        key, name = company
        entry = out.setdefault(key, {"name": name, "metrics": [], "close": None})
        metric = _metric(question)
        if metric not in entry["metrics"]:
            entry["metrics"].append(metric)
        close = getattr(question, "close_time", None)
        if close is not None and (entry["close"] is None or close < entry["close"]):
            entry["close"] = close
    return out


def prompt(key: str, entry: dict, today: datetime) -> str:
    asked = "; ".join(entry["metrics"])
    label = entry["name"] if entry["name"] == key else f"{entry['name']} ({key})"
    return (
        f"Today is {today:%Y-%m-%d}. Build a research dossier on {label} ahead of its next "
        f"quarterly earnings release, for a forecaster who must predict: {asked}.\n\n"
        "Report, each fact with its date and source:\n"
        "1. The confirmed date and time of the release, and the fiscal quarter it covers.\n"
        "2. The analyst consensus for revenue and EPS (say whether it is GAAP or non-GAAP), "
        "the number of analysts and the range, and how it moved in the last month.\n"
        "3. The company's own guidance for the quarter, and how it compared with consensus.\n"
        "4. Every item that will make GAAP diluted EPS differ from the non-GAAP consensus this "
        "quarter: stock-based compensation, amortization of acquired intangibles, gains or "
        "losses on equity investments and mark-to-market of stakes (e.g. in AI companies), "
        "equity-method results, restructuring, impairments, legal settlements, one-off tax "
        "items -- with amounts where they were disclosed or estimated.\n"
        "5. GAAP and non-GAAP EPS, and revenue, for the last 8 quarters, with how each compared "
        "with consensus at the time.\n"
        "6. Anything material since the last report: deals, write-downs, lawsuits, segment "
        "trends, currency, management commentary.\n\n"
        "Facts only. Do not forecast, give a range of your own or say what the result will be. "
        "Say plainly where a figure could not be found."
    )


async def build(questions: Sequence[Any], budget_usd: float, today: datetime,
                research: Callable[..., Awaitable[str]] | None = None) -> tuple[dict, float]:
    """The dossiers, soonest close first, until the next one could pass the
    budget; and the dollars spent. One company failing does not stop the rest."""
    if research is None:
        import claude_research

        research = claude_research.research
    entries = sorted(plan(questions).items(),
                     key=lambda kv: kv[1]["close"] or datetime.max.replace(tzinfo=timezone.utc))
    spent = 0.0
    dossiers: dict[str, dict] = {}

    def add(cost: float) -> None:
        nonlocal spent
        spent += cost

    for key, entry in entries:
        if spent + EST_COST > budget_usd:
            logger.info("mp_dossier_budget_reached spent=%.2f left_out=%d", spent,
                        len(entries) - len(dossiers))
            break
        try:
            text = await research(prompt(key, entry, today), depth=DEPTH, engine_key=ENGINE,
                                  max_prior_cost=MAX_PRIOR_COST, effort=EFFORT, on_cost=add)
        except Exception as exc:  # noqa: BLE001 - the next company still gets its dossier
            logger.warning("mp_dossier_failed company=%s reason=%s", key, type(exc).__name__)
            continue
        dossiers[key] = {"t": today.isoformat(timespec="seconds"), "name": entry["name"],
                         "metrics": entry["metrics"], "text": text}
        logger.info("mp_dossier_built company=%s chars=%d spent=%.2f", key, len(text), spent)
    return dossiers, spent


def save(path: str, dossiers: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dossiers, handle, ensure_ascii=False)


def load(path: str | None = None) -> dict:
    path = path or os.environ.get("MP_DOSSIERS_PATH", "").strip()
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _market_pulse(question: Any) -> bool:
    import tournaments

    slugs = {str(s) for s in (getattr(question, "tournament_slugs", None) or [])}
    return str(tournaments.MARKET_PULSE.id) in slugs


def section(question: Any, now: datetime | None = None, path: str | None = None) -> str:
    """The dossier of a Market Pulse question's company, for its research, or
    "" (no dossier, too old, another tournament, or any error)."""
    try:
        if not _market_pulse(question):
            return ""
        company = _company(question)
        entry = load(path).get(company[0]) if company else None
        if not entry or not entry.get("text"):
            return ""
        made = datetime.fromisoformat(entry["t"])
        now = now or datetime.now(timezone.utc)
        if now - made > MAX_AGE:
            return ""
        text = entry["text"][:MAX_CHARS]
        return (f"## Earnings dossier: {entry['name']} (researched {made:%Y-%m-%d}; the research "
                f"above is newer and outranks it where they differ)\n{text}")
    except Exception as exc:  # noqa: BLE001 - optional context, never fail a forecast
        logger.warning("mp_dossier_section_failed reason=%s", type(exc).__name__)
        return ""


def _open_questions() -> list:
    import tournaments
    from publication import PublishingClient

    return PublishingClient().get_all_open_questions_from_tournament(tournaments.MARKET_PULSE.id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build Market Pulse earnings dossiers")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--budget", type=float, required=True, help="hard cap in USD")
    b.add_argument("--out", default="mp_dossiers.json")
    b.add_argument("--dry-run", action="store_true", help="print the plan, research nothing")
    args = parser.parse_args(argv)
    questions = _open_questions()
    entries = plan(questions)
    print(f"mp_dossier_plan open_questions={len(questions)} companies={len(entries)}")
    for key, entry in sorted(entries.items(), key=lambda kv: str(kv[1]["close"])):
        print(f"  {key}: {len(entry['metrics'])} metric(s), closes {entry['close']}")
    if args.dry_run:
        return 0
    dossiers, spent = asyncio.run(build(questions, args.budget, datetime.now(timezone.utc)))
    save(args.out, dossiers)
    print(f"mp_dossier_done dossiers={len(dossiers)} spent_usd={spent:.2f}")
    return 0 if dossiers else 1


if __name__ == "__main__":
    sys.exit(main())
