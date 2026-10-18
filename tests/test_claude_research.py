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


def response(stop="end_turn", text="rundown", searches=2, tin=1000, tout=200):
    content = [NS(type="server_tool_use"), NS(type="text", text=text)] if text else []
    usage = NS(input_tokens=tin, output_tokens=tout, cache_read_input_tokens=0,
               cache_creation_input_tokens=0,
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
        self.assertEqual(out, "final")
        self.assertEqual(len(fake.calls), 2)
        resumed = fake.calls[1]["messages"]
        self.assertEqual([m["role"] for m in resumed], ["user", "assistant"])
        self.assertIs(resumed[1]["content"], paused.content)

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

    def test_haiku_can_run_three_targeted_rounds(self):
        script = [response(text="a\nGAPS: one", tin=1000), response(text="b\nGAPS: two", tin=1000),
                  response(text="c\nGAPS: none", tin=1000)]
        with FakeAnthropic(*script) as fake:
            out = run(cr.research("q", engine_key="haiku"))
        self.assertEqual([c["tools"][0]["max_uses"] for c in fake.calls],
                         list(cr.ROUNDS[("haiku", "full")]))
        self.assertIn("(round 3)", out)

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
