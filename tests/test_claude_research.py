"""Web-search research through the Anthropic API, with a fake client.

No network and no spend: a stand-in ``anthropic`` module records each request
and replays scripted responses, so the loop (pause_turn resumes, refusals,
empty answers, cost accounting) is exercised exactly as production runs it.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import claude_research as cr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def response(stop="end_turn", text="rundown", searches=2, tin=1000, tout=200, cread=0, cwrite=0):
    content = [NS(type="server_tool_use"), NS(type="text", text=text)] if text else []
    usage = NS(input_tokens=tin, output_tokens=tout, cache_read_input_tokens=cread,
               cache_creation_input_tokens=cwrite,
               server_tool_use=NS(web_search_requests=searches))
    return NS(stop_reason=stop, content=content, usage=usage, model=cr.MODEL)


class FakeAnthropic:
    """Installs itself as the ``anthropic`` module for the test's duration."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        fake = self

        class Messages:
            async def create(self, **kwargs):
                fake.calls.append(kwargs)
                return fake.responses.pop(0)

        class Client:
            def __init__(self, *a, **k):
                self.beta = NS(messages=Messages())

        self.module = types.ModuleType("anthropic")
        self.module.AsyncAnthropic = Client

    def __enter__(self):
        self._patch = mock.patch.dict(sys.modules, {"anthropic": self.module})
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def run(coro):
    return asyncio.run(coro)


class ResearchLoopTests(unittest.TestCase):
    def test_returns_the_text_and_requests_web_search(self):
        with FakeAnthropic(response(text="BTC was $61,200 on 2026-10-14.")) as fake:
            out = run(cr.research("What will BTC close at?", engine_key="sonnet"))
        self.assertEqual(out, "BTC was $61,200 on 2026-10-14.")
        call = fake.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-5-5")
        self.assertEqual(call["tools"][0]["type"], "web_search_20260209")
        self.assertEqual(call["tools"][0]["max_uses"], cr.ROUNDS[("sonnet", "full")][0])
        self.assertEqual(call["output_config"], {"effort": cr.EFFORT})
        self.assertEqual(call["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", call["betas"])

    def test_pause_turn_is_resumed_with_the_paused_turn_appended(self):
        paused = response(stop="pause_turn", text="partial")
        with FakeAnthropic(paused, response(text="final")) as fake:
            out = run(cr.research("q"))
        # Text written before the pause is part of the rundown (it was lost
        # until 2026-10-23, when only the last response's text was kept).
        self.assertEqual(out, "partial\n\nfinal")
        self.assertEqual(len(fake.calls), 2)
        resumed = fake.calls[1]["messages"]
        self.assertEqual([m["role"] for m in resumed], ["user", "assistant"])
        self.assertEqual(resumed[1]["content"], paused.content)
        for block, sent in zip(paused.content, resumed[1]["content"]):
            self.assertIs(sent, block)  # sent back unchanged

    def test_a_second_resume_sends_back_the_whole_turn(self):
        """Each response carries only what it added, so a second resume must
        send both paused parts; it used to send only the latest."""
        one = response(stop="pause_turn", text="part one")
        two = response(stop="pause_turn", text="part two")
        with FakeAnthropic(one, two, response(text="part three")) as fake:
            out = run(cr.research("q"))
        self.assertEqual(len(fake.calls), 3)
        second = fake.calls[1]["messages"]
        third = fake.calls[2]["messages"]
        self.assertEqual(second[1]["content"], one.content)
        self.assertEqual([m["role"] for m in third], ["user", "assistant"])
        self.assertEqual(third[1]["content"], [*one.content, *two.content])
        self.assertEqual(out, "part one\n\npart two\n\npart three")

    def test_follow_up_rounds_keep_their_paused_parts_too(self):
        script = [response(text="r1\nGAPS: x"), response(stop="pause_turn", text="early"),
                  response(text="late\nGAPS: none")]
        with FakeAnthropic(*script) as fake:
            out = run(cr.research("q"))
        resumed = fake.calls[2]["messages"]
        self.assertEqual([m["role"] for m in resumed], ["user", "assistant", "user", "assistant"])
        self.assertEqual(resumed[3]["content"], script[1].content)
        self.assertIn("early\n\nlate", out)

    def test_resumes_are_logged(self):
        script = [response(stop="pause_turn"), response(stop="pause_turn"), response(text="x")]
        with FakeAnthropic(*script):
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                run(cr.research("q", depth="light"))
        self.assertIn("resumes=2", "\n".join(logs.output))
        with FakeAnthropic(response(text="x")):
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                run(cr.research("q", depth="light"))
        self.assertIn("resumes=0", "\n".join(logs.output))

    def test_resumed_requests_are_all_billed_and_cached(self):
        script = [response(stop="pause_turn", tin=1000, tout=100, searches=3),
                  response(tin=2000, tout=300, searches=1, cread=500)]
        with FakeAnthropic(*script) as fake:
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                run(cr.research("q", engine_key="sonnet", depth="light"))
        for call in fake.calls:
            self.assertEqual(call["cache_control"], {"type": "ephemeral"})
        e = cr.ENGINES["sonnet"]
        usd = (cr.request_cost(e, 1000, 0, 0, 100, 3) + cr.request_cost(e, 2000, 0, 500, 300, 1))
        line = "\n".join(logs.output)
        self.assertIn("searches=4", line)
        self.assertIn(f"est_cost_usd={usd:.4f}", line)

    def test_resumes_are_bounded(self):
        script = [response(stop="pause_turn")] * (cr.MAX_RESUMES + 5)
        with FakeAnthropic(*script) as fake:
            run(cr.research("q"))
        self.assertEqual(len(fake.calls), cr.MAX_RESUMES + 1)

    def test_a_refusal_raises_so_the_caller_falls_back(self):
        with FakeAnthropic(response(stop="refusal")):
            with self.assertRaises(RuntimeError):
                run(cr.research("q"))

    def test_an_empty_answer_raises(self):
        with FakeAnthropic(response(text="")):
            with self.assertRaises(RuntimeError):
                run(cr.research("q"))

    def test_cost_is_logged_without_content(self):
        with FakeAnthropic(response(text="SECRET-RUNDOWN", searches=4, tin=50_000, tout=3_000)):
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                run(cr.research("q", engine_key="sonnet"))
        line = "\n".join(logs.output)
        self.assertIn("searches=4", line)
        self.assertIn("est_cost_usd=0.1700", line)  # 0.10 + 0.03 + 0.04
        self.assertNotIn("SECRET-RUNDOWN", line)


class SecondRoundTests(unittest.TestCase):
    def test_named_gaps_get_a_second_targeted_round(self):
        first = response(text="BTC at $81k.\nGAPS: ETF flows last week; options expiry date")
        with FakeAnthropic(first, response(text="ETF flows: +$1.2B (2026-10-16).\nGAPS: none")) as fake:
            out = run(cr.research("q"))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[1]["tools"][0]["max_uses"], cr.ROUND2_SEARCHES)
        self.assertEqual(fake.calls[0]["model"], cr.MODEL)
        roles = [m["role"] for m in fake.calls[1]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertIn("GAPS:", fake.calls[1]["messages"][2]["content"])
        self.assertIn("Follow-up on the gaps", out)
        self.assertIn("+$1.2B", out)

    def test_no_gaps_means_one_round(self):
        with FakeAnthropic(response(text="All found.\nGAPS: none")) as fake:
            run(cr.research("q"))
        self.assertEqual(len(fake.calls), 1)

    def test_an_expensive_first_round_skips_the_second(self):
        pricey = response(text="x\nGAPS: something", tin=200_000)  # $0.40 of Sonnet input
        with FakeAnthropic(pricey) as fake:
            run(cr.research("q", engine_key="sonnet"))
        self.assertEqual(len(fake.calls), 1)

    def test_a_failed_second_round_keeps_the_first(self):
        first = response(text="Round one facts.\nGAPS: one thing")
        fake = FakeAnthropic(first)  # second create() pops from an empty list -> IndexError
        with fake:
            out = run(cr.research("q"))
        self.assertIn("Round one facts.", out)

    def test_prize_questions_get_more_searches_than_the_cup(self):
        full = sum(cr.ROUNDS[(cr.DEFAULT_ENGINE, "full")])
        light = sum(cr.ROUNDS[(cr.DEFAULT_ENGINE, "light")])
        self.assertEqual(cr.MAX_SEARCHES, full)
        self.assertGreater(full, light)

    def test_light_depth_is_one_round_even_with_gaps(self):
        with FakeAnthropic(response(text="x\nGAPS: something")) as fake:
            run(cr.research("q", depth="light"))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["tools"][0]["max_uses"],
                         cr.ROUNDS[(cr.DEFAULT_ENGINE, "light")][0])

    def test_haiku_engine_uses_the_basic_search_and_no_fallbacks(self):
        with FakeAnthropic(response(text="x\nGAPS: none")) as fake:
            run(cr.research("q", engine_key="haiku"))
        call = fake.calls[0]
        self.assertEqual(call["model"], "claude-haiku-5-5")
        self.assertEqual(call["tools"][0]["type"], "web_search_20250305")
        self.assertNotIn("fallbacks", call)
        self.assertNotIn("betas", call)

    def test_haiku_runs_every_planned_targeted_round(self):
        plan = cr.ROUNDS[("haiku", "full")]
        script = [response(text=f"r{i}\nGAPS: more", tin=1000) for i in range(len(plan))]
        with FakeAnthropic(*script) as fake:
            out = run(cr.research("q", engine_key="haiku"))
        self.assertEqual([c["tools"][0]["max_uses"] for c in fake.calls], list(plan))
        self.assertIn(f"(round {len(plan)})", out)

    def test_prize_tournaments_get_full_research_and_the_rest_light(self):
        """The owner put the budget on FutureEval and Market Pulse
        (2026-10-23); MiniBench and the Cup run light, except during a
        research boost before the credit renews (2026-10-24), never while
        over budget."""
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            src = handle.read()
        fe = src.index('spend.set_tournament("futureeval")')
        mb = src.index('spend.set_tournament("minibench")')
        self.assertIn("template_bot.research_depth = prize_depth", src[fe:mb])
        self.assertIn("template_bot.research_depth = other_depth", src[mb:mb + 200])
        cup = src.index('spend.set_tournament("cup")')
        self.assertIn("template_bot.research_depth = other_depth", src[cup - 200:cup])
        self.assertIn('boost = prize_depth == "full" and spend.boost_active()', src)
        self.assertIn('other_depth = "full" if boost else "light"', src)
        self.assertGreater(sum(cr.ROUNDS[("haiku", "full")]), sum(cr.ROUNDS[("haiku", "light")]))

    def test_haiku_is_the_production_default(self):
        self.assertEqual(cr.DEFAULT_ENGINE, "haiku")
        self.assertEqual(cr.MODEL, "claude-haiku-5-5")

    def test_engine_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, {"CLAUDE_RESEARCH_ENGINE": "haiku"}):
            self.assertEqual(cr.engine_name(), "haiku")
        with mock.patch.dict(os.environ, {"CLAUDE_RESEARCH_ENGINE": "nonsense"}):
            self.assertEqual(cr.engine_name(), cr.DEFAULT_ENGINE)

    def test_gaps_parsing(self):
        self.assertEqual(cr._gaps("a\nGAPS: x; y"), "x; y")
        self.assertEqual(cr._gaps("a\nGAPS: none."), "")
        self.assertEqual(cr._gaps("no line"), "")


def text(value, *cites):
    return NS(type="text", text=value, citations=list(cites) or None)


def cite(url, title):
    return NS(type="web_search_result_location", url=url, title=title,
              encrypted_index="x", cited_text="...")


def results(*pages):
    """A web_search_tool_result block; each page is (url, title, page_age)."""
    return NS(type="web_search_tool_result", tool_use_id="srvtoolu_1",
              content=[NS(type="web_search_result", url=u, title=t, page_age=a,
                          encrypted_content="x") for u, t, a in pages])


def blocks_response(content, stop="end_turn"):
    usage = NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
               cache_creation_input_tokens=0, server_tool_use=NS(web_search_requests=1))
    return NS(stop_reason=stop, content=content, usage=usage, model=cr.MODEL)


class TextFidelityTests(unittest.TestCase):
    """Synthetic pages and facts only (example.org): no real question's
    research goes in a fixture."""

    def test_citation_split_blocks_are_one_sentence(self):
        content = [
            text("I will look this up."),
            NS(type="server_tool_use", id="srvtoolu_1"),
            results(("https://example.org/a", "A", "May 2, 2026")),
            text("The widget index stood at "),
            text("412.5 on 2026-05-01", cite("https://example.org/a", "A")),
            text(", up from 400.\nGAPS: none"),
        ]
        self.assertEqual(
            cr._text(content),
            "I will look this up.\n\nThe widget index stood at 412.5 on 2026-05-01, up from 400.\n"
            "GAPS: none",
        )

    def test_sources_list_cited_pages_first_with_their_dates(self):
        content = [
            NS(type="server_tool_use", id="srvtoolu_1"),
            results(("https://example.org/a", "Page A", "May 2, 2026"),
                    ("https://example.org/b", "Page  B\n", None),
                    ("https://example.org/c", "Page C", "3 days ago")),
            text("Fact one", cite("https://example.org/c", "Page C")),
            text(" and fact two.", cite("https://example.org/c", "Page C"),
                 cite("https://example.org/d", "Page D")),
        ]
        section, n = cr._sources(content)
        self.assertEqual(n, 4)
        lines = section.splitlines()
        self.assertIn("first 2 are cited above", lines[0])
        self.assertEqual(lines[1:], [
            "- Page C (3 days ago) https://example.org/c",
            "- Page D https://example.org/d",
            "- Page A (May 2, 2026) https://example.org/a",
            "- Page B https://example.org/b",
        ])

    def test_sources_are_capped_and_skip_search_errors(self):
        many = [(f"https://example.org/{i}", "T" * 300, None) for i in range(cr.MAX_SOURCES + 5)]
        error = NS(type="web_search_tool_result", tool_use_id="srvtoolu_2",
                   content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded"))
        section, n = cr._sources([error, results(*many), NS(type="text", text="x")])
        self.assertEqual(n, cr.MAX_SOURCES)
        self.assertEqual(len(section.splitlines()), cr.MAX_SOURCES + 1)
        self.assertIn("(search results;", section.splitlines()[0])
        self.assertLess(max(len(line) for line in section.splitlines()), 200)
        self.assertEqual(cr._sources([text("no search at all")]), ("", 0))

    def test_the_rundown_ends_with_its_sources_and_round_two_does_not_see_them(self):
        first = blocks_response([
            results(("https://example.org/a", "Page A", "May 2, 2026")),
            text("Index at 412.5", cite("https://example.org/a", "Page A")),
            text(".\nGAPS: last week's figure"),
        ])
        second = blocks_response([
            results(("https://example.org/b", "Page B", "May 9, 2026")),
            text("Last week: 409.", cite("https://example.org/b", "Page B")),
            text("\nGAPS: none"),
        ])
        with FakeAnthropic(first, second) as fake:
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                out = run(cr.research("q"))
        self.assertIn("Index at 412.5.\nGAPS:", out)
        body, sources = out.split("\n\n## Sources", 1)
        self.assertIn("## Follow-up on the gaps (round 2)\nLast week: 409.", body)
        self.assertIn("- Page A (May 2, 2026) https://example.org/a", sources)
        self.assertIn("- Page B (May 9, 2026) https://example.org/b", sources)
        prior = fake.calls[1]["messages"][1]["content"]
        self.assertNotIn("Sources", prior)
        self.assertNotIn("example.org", prior)
        line = "\n".join(logs.output)
        self.assertIn("sources=2", line)
        self.assertNotIn("example.org", line)

    def test_a_dropped_round_lends_no_sources(self):
        first = blocks_response([text("Round one.\nGAPS: one thing")])
        refused = blocks_response([results(("https://example.org/z", "Z", None)),
                                   text("nope")], stop="refusal")
        with FakeAnthropic(first, refused):
            out = run(cr.research("q"))
        self.assertEqual(out, "Round one.\nGAPS: one thing")

    def test_a_broken_source_list_keeps_the_rundown(self):
        with FakeAnthropic(response(text="Facts.\nGAPS: none")):
            with mock.patch.object(cr, "_sources", side_effect=ValueError("boom")):
                with self.assertLogs("claude_research", level=logging.INFO) as logs:
                    out = run(cr.research("q"))
        self.assertEqual(out, "Facts.\nGAPS: none")
        self.assertIn("claude_research_sources_failed reason=ValueError", "\n".join(logs.output))


class PromptCachingTests(unittest.TestCase):
    """Anthropic flagged a low cache-hit rate (2026-10-21); every research
    request now asks for automatic caching, and the logged cost prices cache
    writes and reads as billed."""

    def test_every_request_asks_for_caching(self):
        script = [response(stop="pause_turn"), response(text="a\nGAPS: x"), response(text="b")]
        with FakeAnthropic(*script) as fake:
            run(cr.research("q"))
        self.assertGreaterEqual(len(fake.calls), 3)
        for call in fake.calls:
            self.assertEqual(call["cache_control"], {"type": "ephemeral"})

    def test_cached_tokens_are_priced_as_billed(self):
        e = cr.ENGINES["sonnet"]
        t = cr.Tally(e)
        t.add(response(tin=10_000, cwrite=20_000, cread=60_000, tout=1_000, searches=3).usage)
        # Sonnet 5.5 cache hits are 0.05x, writes 1.25x (pricing page, 2026-10-21).
        expected = (10_000 + 1.25 * 20_000 + 0.05 * 60_000) * e.price_in + 1_000 * e.price_out + 0.03
        self.assertAlmostEqual(t.usd, expected)
        self.assertEqual(t.tokens_in, 90_000)

    def test_reads_make_the_same_tokens_cheaper(self):
        e = cr.ENGINES["haiku"]
        cold = cr.request_cost(e, 90_000, 0, 0, 0, 0)
        warm = cr.request_cost(e, 10_000, 20_000, 60_000, 0, 0)
        self.assertLess(warm, cold * 0.5)

    def test_long_haiku_prompts_pay_the_long_context_price(self):
        # Haiku 5.5: $0.10/$0.50 per M up to 100,000 prompt tokens, $0.50/$2.50
        # above, counting cache reads and writes; each request on its own.
        e = cr.ENGINES["haiku"]
        short = cr.request_cost(e, 10_000, 0, 90_000, 1_000, 0)
        long = cr.request_cost(e, 10_000, 0, 91_000, 1_000, 0)
        self.assertAlmostEqual(short, (10_000 + 0.1 * 90_000) * 0.10e-6 + 1_000 * 0.50e-6)
        self.assertAlmostEqual(long, ((10_000 + 0.1 * 91_000) * 0.10e-6 + 1_000 * 0.50e-6) * 5)
        self.assertAlmostEqual(cr.request_cost(e, 0, 0, 0, 0, 4), 0.04)
        t = cr.Tally(e)
        for _ in range(3):
            t.add(response(tin=40_000, tout=0, searches=0).usage)
        self.assertAlmostEqual(t.usd, 3 * 40_000 * 0.10e-6)

    def test_cache_counts_are_logged(self):
        with FakeAnthropic(response(text="x\nGAPS: none", cread=5000, cwrite=800)):
            with self.assertLogs("claude_research", level=logging.INFO) as logs:
                run(cr.research("q"))
        line = "\n".join(logs.output)
        self.assertIn("cache_read=5000", line)
        self.assertIn("cache_write=800", line)


class EnabledTests(unittest.TestCase):
    def test_off_without_the_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            self.assertFalse(cr.enabled())

    def test_on_with_the_key(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            self.assertTrue(cr.enabled())


class WiringTests(unittest.TestCase):
    def setUp(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            self.src = handle.read()

    def test_research_tries_claude_first_and_falls_back(self):
        start = self.src.index("async def run_research")
        body = self.src[start:self.src.index("return research", start)]
        self.assertLess(body.index("claude_research.research("),
                        body.index("researcher.invoke(prompt)"))
        self.assertIn("except Exception", body)

    def test_the_workflow_passes_the_key_to_every_forecasting_step(self):
        import yaml

        with io.open(os.path.join(ROOT, ".github", "workflows",
                                  "run_bot_on_tournament.yaml"), encoding="utf-8") as h:
            steps = yaml.safe_load(h)["jobs"]["forecast_job"]["steps"]
        for name in ("Run bot", "Forecast new Metaculus Cup questions (practice)",
                     "Forecast Market Pulse questions (prize)"):
            step = next(s for s in steps if s.get("name") == name)
            self.assertIn("ANTHROPIC_API_KEY", step["env"], name)


if __name__ == "__main__":
    unittest.main()
