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
            out = run(cr.research("What will BTC close at?"))
        self.assertEqual(out, "BTC was $61,200 on 2026-10-14.")
        call = fake.calls[0]
        self.assertEqual(call["model"], cr.MODEL)
        self.assertEqual(call["tools"][0]["type"], "web_search_20260209")
        self.assertEqual(call["tools"][0]["max_uses"], cr.MAX_SEARCHES)
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
                run(cr.research("q"))
        line = "\n".join(logs.output)
        self.assertIn("searches=4", line)
        self.assertIn("est_cost_usd=0.1700", line)  # 0.10 + 0.03 + 0.04
        self.assertNotIn("SECRET-RUNDOWN", line)


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
        self.assertLess(body.index("claude_research.research(prompt)"),
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
