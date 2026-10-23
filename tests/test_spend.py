"""The API spend ledger, its WhatsApp messages and the budget throttle.

See spend.py for why it exists. Everything runs against a temporary ledger;
nothing here reaches the network or WhatsApp.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import spend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)  # a Monday
ALLOWED_KEYS = {"t", "run", "tour", "q", "phase", "provider", "model", "in", "out", "searches", "usd"}
#: Written only when the provider reports it: a count, never content.
OPTIONAL_KEYS = {"reasoning"}


def call(t, usd, provider="anthropic", q=1, run="r1", tour="minibench", phase="research",
         model="claude-haiku-5-5"):
    return {"t": t.isoformat(timespec="seconds"), "run": run, "tour": tour, "q": q,
            "phase": phase, "provider": provider, "model": model, "in": 1, "out": 1,
            "searches": 0, "usd": usd}


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.jsonl")
        self.summary = os.path.join(self.dir.name, "summary.md")
        self.env = mock.patch.dict(os.environ, {
            "SPEND_LEDGER_PATH": self.path, "GITHUB_RUN_ID": "r9",
            "GITHUB_STEP_SUMMARY": self.summary, "CLAUDE_BUDGET_USD": "100",
            "SPEND_ALERT_THRESHOLDS": "0.8,1.0", "SPEND_THROTTLE_AT": "1.0",
        })
        self.env.start()
        spend.set_tournament(None)

    def tearDown(self):
        self.env.stop()
        spend.set_tournament(None)
        self.dir.cleanup()

    def write(self, *lines):
        with open(self.path, "a", encoding="utf-8") as h:
            for line in lines:
                h.write(json.dumps(line) + "\n")


class RecordTests(LedgerCase):
    def test_nothing_is_written_without_a_ledger_path(self):
        with mock.patch.dict(os.environ, {"SPEND_LEDGER_PATH": ""}):
            spend.record(provider="anthropic", model="m", usd=1.0)
        self.assertFalse(os.path.exists(self.path))

    def test_a_line_carries_question_phase_and_tournament_and_nothing_else(self):
        spend.set_tournament("cup")
        with spend.question(NS(id_of_post=46224)), spend.phase("research"):
            spend.record(provider="anthropic", model="claude-haiku-5-5", usd=0.15,
                         tokens_in=1000, tokens_out=200, searches=4)
        [line] = spend.load()
        self.assertEqual(set(line), ALLOWED_KEYS)
        self.assertEqual((line["q"], line["phase"], line["tour"], line["run"]),
                         (46224, "research", "cup", "r9"))
        self.assertEqual((line["usd"], line["searches"]), (0.15, 4))

    def test_the_crux_follow_up_keeps_its_label(self):
        with spend.phase(spend.CRUX):
            with spend.phase("forecast"), spend.phase("parser"):
                spend.record(provider="openrouter", model="m", usd=0.01)
        with spend.phase("forecast"), spend.phase("parser"):
            spend.record(provider="openrouter", model="m", usd=0.01)
        self.assertEqual([l["phase"] for l in spend.load()], ["crux", "parser"])

    def test_concurrent_questions_keep_their_own_labels(self):
        async def one(post, label):
            with spend.question(NS(id_of_post=post)), spend.phase(label):
                await asyncio.sleep(0)
                spend.record(provider="openrouter", model="m", usd=0.01)

        async def both():
            await asyncio.gather(one(1, "forecast"), one(2, "summary"))

        asyncio.run(both())
        self.assertEqual(sorted((l["q"], l["phase"]) for l in spend.load()),
                         [(1, "forecast"), (2, "summary")])

    def test_a_broken_ledger_never_raises(self):
        with mock.patch.dict(os.environ, {"SPEND_LEDGER_PATH": os.path.join(self.path, "x", "y")}):
            spend.record(provider="anthropic", model="m", usd=1.0)

    def test_litellm_responses_are_recorded_with_their_price(self):
        response = NS(_hidden_params={"response_cost": 0.021},
                      usage=NS(prompt_tokens=900, completion_tokens=300))
        spend._record_response("openrouter/openai/gpt-5.6-sol", response)
        responses_api = NS(_hidden_params={"response_cost": 0.004},
                           usage=NS(input_tokens=50, output_tokens=10))
        spend._record_response("openrouter/openai/gpt-4o-mini", responses_api)
        first, second = spend.load()
        self.assertEqual((first["provider"], first["model"], first["usd"], first["in"], first["out"]),
                         ("openrouter", "openai/gpt-5.6-sol", 0.021, 900, 300))
        self.assertEqual((second["in"], second["out"], second["usd"]), (50, 10, 0.004))

    def test_reasoning_tokens_are_recorded_as_a_count_when_reported(self):
        """C1 (2026-10-23): is Opus reasoning at effort "high"? Chat
        completions report it under completion_tokens_details, the responses
        API under output_tokens_details; litellm hands either over as an
        object or a dict."""
        chat = NS(_hidden_params={"response_cost": 0.05},
                  usage=NS(prompt_tokens=900, completion_tokens=700,
                           completion_tokens_details=NS(reasoning_tokens=412)))
        spend._record_response("openrouter/anthropic/claude-opus-5.5", chat)
        responses_api = NS(_hidden_params={"response_cost": 0.01},
                           usage=NS(input_tokens=50, output_tokens=60,
                                    output_tokens_details={"reasoning_tokens": 0}))
        spend._record_response("openrouter/openai/gpt-5.6-sol", responses_api)
        silent = NS(_hidden_params={"response_cost": 0.001},
                    usage=NS(prompt_tokens=5, completion_tokens=5, completion_tokens_details=None))
        spend._record_response("openrouter/anthropic/claude-haiku-4.5", silent)
        opus, sol, haiku = spend.load()
        self.assertEqual(opus["reasoning"], 412)
        self.assertEqual(sol["reasoning"], 0, "a reported zero is kept: it is the answer")
        self.assertNotIn("reasoning", haiku, "not reported is not zero")
        for line in (opus, sol, haiku):
            self.assertLessEqual(set(line), ALLOWED_KEYS | OPTIONAL_KEYS)

    def test_a_malformed_reasoning_count_is_left_out(self):
        for value in (None, "412", True, -3):
            usage = NS(completion_tokens_details=NS(reasoning_tokens=value))
            self.assertIsNone(spend.reasoning_tokens(usage), value)
        self.assertIsNone(spend.reasoning_tokens(None))

    def test_the_sdk_calls_are_wrapped_once_and_still_return_the_response(self):
        module = types.ModuleType("forecasting_tools.ai_models.general_llm")

        async def acompletion(**kwargs):
            return NS(_hidden_params={"response_cost": 0.5}, usage=NS(prompt_tokens=1, completion_tokens=2))

        module.acompletion = acompletion
        package = types.ModuleType("forecasting_tools.ai_models")
        package.general_llm = module
        with mock.patch.dict(sys.modules, {"forecasting_tools.ai_models": package,
                                           "forecasting_tools.ai_models.general_llm": module}):
            spend.install_litellm_recorder()
            spend.install_litellm_recorder()
            with spend.phase("summary"):
                response = asyncio.run(module.acompletion(model="openrouter/anthropic/claude-haiku-4.5"))
        self.assertEqual(response.usage.completion_tokens, 2)
        [line] = spend.load()
        self.assertEqual((line["phase"], line["model"], line["usd"]),
                         ("summary", "anthropic/claude-haiku-4.5", 0.5))

    def test_the_real_general_llm_is_recorded_in_the_callers_context(self):
        """End to end through forecasting_tools and litellm (mock response,
        no network). Test Bot run 39570809053 recorded none of its 132
        OpenRouter calls through a litellm success callback; this is the
        path that replaced it."""
        from tests._real_forecasting_tools import real_forecasting_tools

        try:
            with real_forecasting_tools():
                from forecasting_tools.ai_models import general_llm
        except ImportError:
            self.skipTest("forecasting_tools not installed")
        saved = general_llm.acompletion, getattr(general_llm, "aresponses", None)
        try:
            with real_forecasting_tools(), mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
                spend.install_litellm_recorder()
                llm = general_llm.GeneralLlm(model="openrouter/openai/gpt-4o-mini",
                                             mock_response="ok " * 10)

                async def two():
                    async def one(post, label):
                        with spend.question(NS(id_of_post=post)), spend.phase(label):
                            await llm.invoke("hi")
                    await asyncio.gather(one(1, "forecast"), one(2, "parser"))

                asyncio.run(two())
        finally:
            general_llm.acompletion = saved[0]
            if saved[1] is not None:
                general_llm.aresponses = saved[1]
        lines = sorted(spend.load(), key=lambda l: l["q"])
        self.assertEqual([(l["q"], l["phase"], l["provider"]) for l in lines],
                         [(1, "forecast", "openrouter"), (2, "parser", "openrouter")])
        self.assertTrue(all(l["usd"] > 0 for l in lines))


class ResearchRecordingTests(LedgerCase):
    def test_claude_research_records_its_cost_even_when_it_fails(self):
        import claude_research as cr
        from tests.test_claude_research import FakeAnthropic, response, run

        with FakeAnthropic(response(stop="refusal", searches=3, tin=20_000, tout=500)):
            with self.assertRaises(RuntimeError):
                run(cr.research("q"))
        [line] = spend.load()
        self.assertEqual((line["provider"], line["model"], line["searches"]),
                         ("anthropic", "claude-haiku-5-5", 3))
        self.assertAlmostEqual(line["usd"], 20_000 * 0.10e-6 + 500 * 0.50e-6 + 0.03, places=6)


class BudgetTests(LedgerCase):
    def test_only_this_months_anthropic_spend_counts(self):
        self.write(call(NOW - timedelta(days=30), 50.0),
                   call(NOW - timedelta(days=2), 30.0),
                   call(NOW - timedelta(days=1), 9.0, provider="openrouter"))
        self.assertAlmostEqual(spend.month_spend(spend.load(), NOW), 30.0)

    def test_the_throttle_starts_at_the_budget(self):
        self.write(call(NOW - timedelta(days=1), 99.0))
        self.assertFalse(spend.over_budget(NOW))
        self.write(call(NOW, 1.0))
        self.assertTrue(spend.over_budget(NOW))
        with mock.patch.dict(os.environ, {"CLAUDE_BUDGET_USD": "200"}):
            self.assertFalse(spend.over_budget(NOW))

    def test_a_budget_can_be_scoped_to_one_month(self):
        with mock.patch.dict(os.environ, {"CLAUDE_BUDGET_USD": "60@2026-10"}):
            self.assertEqual(spend.budget_usd(NOW), 60.0)
            self.assertEqual(spend.budget_usd(datetime(2026, 11, 1, tzinfo=timezone.utc)), 100.0)
            self.write(call(NOW - timedelta(days=1), 61.0))
            self.assertTrue(spend.over_budget(NOW))
            self.assertIn("de 60,00 $", spend.month_line(spend.load(), NOW))
        for raw in ("", "lots", "x@2026-10"):
            with mock.patch.dict(os.environ, {"CLAUDE_BUDGET_USD": raw}):
                self.assertEqual(spend.budget_usd(NOW), 100.0)

    def test_a_missing_or_corrupt_ledger_never_throttles(self):
        self.assertFalse(spend.over_budget(NOW))
        with open(self.path, "w") as h:
            h.write("not json\n{\"t\": \"garbage\"}\n")
        self.assertFalse(spend.over_budget(NOW))

    def test_thresholds_come_from_the_environment(self):
        with mock.patch.dict(os.environ, {"SPEND_ALERT_THRESHOLDS": "1.0, 0.5"}):
            self.assertEqual(spend.thresholds(), (0.5, 1.0))
        with mock.patch.dict(os.environ, {"SPEND_ALERT_THRESHOLDS": "nonsense"}):
            self.assertEqual(spend.thresholds(), spend.DEFAULT_THRESHOLDS)


class AfterRunTests(LedgerCase):
    def after(self, now=NOW, ok=True):
        sent = []

        def send(text):
            sent.append(text)
            return ok

        with mock.patch("sys.stdout", io.StringIO()):
            changed = spend.after_run(now, send=send)
        return changed, sent

    def test_each_threshold_alerts_once_a_month(self):
        self.write(call(NOW - timedelta(days=3), 81.0))
        changed, sent = self.after()
        self.assertTrue(changed)
        self.assertEqual(len(sent), 1)
        self.assertIn("81,00 $ de 100,00 $", sent[0])
        self.assertIn("80 %", sent[0])
        self.assertNotIn("modo ligero", sent[0])
        self.assertEqual(self.after()[1], [])
        self.write(call(NOW, 20.0))
        [stop] = self.after()[1]
        self.assertIn("🛑", stop)
        self.assertIn("modo ligero", stop)
        self.assertEqual(self.after()[1], [])

    def test_crossing_two_thresholds_at_once_sends_one_message(self):
        self.write(call(NOW - timedelta(days=3), 105.0))
        _, sent = self.after()
        self.assertEqual(len(sent), 1)
        self.assertIn("100 %", sent[0])
        self.assertEqual(self.after()[1], [])

    def test_a_failed_send_is_retried_next_run(self):
        self.write(call(NOW - timedelta(days=3), 85.0))
        self.assertEqual(len(self.after(ok=False)[1]), 1)
        self.assertEqual(len(self.after()[1]), 1)

    def test_a_new_month_starts_over(self):
        self.write(call(NOW - timedelta(days=3), 85.0))
        self.after()
        nov = datetime(2026, 11, 3, tzinfo=timezone.utc)
        self.assertEqual(self.after(nov)[1], [])
        self.write(call(nov, 81.0))
        self.assertEqual(len(self.after(nov)[1]), 1)

    def test_old_lines_are_pruned_and_the_history_start_is_kept(self):
        self.write(call(NOW - timedelta(days=100), 1.0), call(NOW - timedelta(days=5), 2.0))
        self.after()
        lines = spend.load()
        self.assertEqual([l.get("kind") for l in lines], ["start", None])
        self.assertEqual(lines[0]["t"], (NOW - timedelta(days=100)).isoformat(timespec="seconds"))

    def test_an_idle_run_changes_nothing(self):
        self.write({"t": (NOW - timedelta(days=40)).isoformat(), "kind": "start"},
                   call(NOW - timedelta(days=5), 2.0, run="r1"))
        changed, sent = self.after()
        self.assertFalse(changed)
        self.assertEqual(sent, [])

    def test_this_runs_spend_goes_to_the_step_summary(self):
        self.write(call(NOW, 0.30, run="r9", q=1), call(NOW, 0.12, run="r9", q=1, provider="openrouter"),
                   call(NOW, 0.30, run="r9", q=2), call(NOW, 5.0, run="r1", q=3))
        changed, _ = self.after()
        self.assertTrue(changed)
        with open(self.summary) as h:
            text = h.read()
        self.assertIn("Esta ejecución: Claude 0,60 $ + OpenRouter 0,12 $ (2 pronósticos, 0,36 $/pronóstico)", text)

    def test_a_corrupt_line_is_skipped_not_fatal(self):
        with open(self.path, "w") as h:
            h.write('{"t": "not a date", "usd": 1}\n{"t": "2026-10-20T00:00:00", "usd": 1}\n'
                    '{"t": "2026-10-20T00:00:00+00:00", "usd": "lots"}\nnot json\n')
        self.write(call(NOW - timedelta(days=3), 85.0))
        self.assertEqual(len(spend.load()), 1)
        self.assertEqual(len(self.after()[1]), 1)

    def test_the_cli_never_fails_the_job(self):
        with open(self.path, "w") as h:
            h.write('{"t": "not a date"}\n')
        with mock.patch("sys.stdout", io.StringIO()):
            self.assertEqual(spend.main(["spend.py", "after-run"]), 0)
            self.assertEqual(spend.main(["spend.py", "weekly"]), 0)


class MessageTests(LedgerCase):
    def week(self):
        start = NOW - timedelta(days=6)
        self.write({"t": datetime(2026, 10, 1, tzinfo=timezone.utc).isoformat(), "kind": "start"})
        self.write(call(NOW - timedelta(days=20), 20.0, q=9, run="old"))
        for i in range(10):
            t = start + timedelta(hours=i * 12)
            tour = "minibench" if i % 2 else "market_pulse"
            self.write(call(t, 0.15, q=i, run=f"r{i}", tour=tour),
                       call(t, 0.06, q=i, run=f"r{i}", tour=tour, provider="openrouter",
                            phase="forecast", model="anthropic/claude-opus-5-5"),
                       call(t, 0.01, q=i, run=f"r{i}", tour=tour, provider="openrouter",
                            phase="parser", model="openai/gpt-4o-mini"))
        self.write(call(NOW - timedelta(days=2), 0.09, q=3, run="r3", phase="crux"))

    def test_the_weekly_message(self):
        self.week()
        text = spend.weekly_message(spend.load(), NOW)
        self.assertLessEqual(len(text), 500)
        self.assertTrue(text.startswith("💸 Gasto API · 19 oct–25 oct"))
        self.assertIn("Claude (tu presupuesto): 1,59 $", text)
        self.assertIn("mes 21,59 $ de 100,00 $ (22 %)", text)
        self.assertIn("proyección", text)
        self.assertIn("OpenRouter (Metaculus): 0,70 $", text)
        self.assertIn("Torneos: MiniBench 1,19 · Market Pulse 1,10", text)
        self.assertIn("Fases: research 1,50 · forecast 0,60 · parser 0,10", text)
        self.assertIn("Modelos: haiku-5-5 1,59 · opus-5-5 0,60 · gpt-4o-mini 0,10", text)
        self.assertIn("10 pronósticos · 0,23 $ de media (Claude 0,16 $)", text)

    def test_a_restarted_history_says_from_when_it_counts(self):
        self.write({"t": datetime(2026, 10, 21, 10, 0, tzinfo=timezone.utc).isoformat(), "kind": "start"},
                   call(NOW - timedelta(days=2), 4.0))
        self.assertIn("(cuenta desde el 21 oct)", spend.weekly_message(spend.load(), NOW))

    def test_an_empty_week(self):
        text = spend.weekly_message([], NOW)
        self.assertIn("Claude (tu presupuesto): 0,00 $", text)
        self.assertIn("Sin pronósticos esta semana", text)

    def test_projection_needs_a_day_of_data(self):
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.assertIsNone(spend.projection(5.0, start + timedelta(hours=12), start))
        self.assertAlmostEqual(spend.projection(10.0, start + timedelta(days=10), start), 31.0)

    def test_the_breakdown_groups_by_provider_phase_and_model(self):
        self.week()
        text = spend.breakdown(spend.load())
        self.assertIn("anthropic research claude-haiku-5-5: 11 calls", text)
        self.assertIn("openrouter forecast anthropic/claude-opus-5-5: 10 calls", text)
        self.assertNotIn("reasoning", text, "no call reported reasoning tokens")
        self.assertEqual(spend.breakdown([]), "no calls recorded")

    def test_the_breakdown_shows_mean_reasoning_tokens_per_model(self):
        for tokens in (300, 500, None):
            line = call(NOW, 0.05, provider="openrouter", phase="forecast",
                        model="anthropic/claude-opus-5.5")
            if tokens is not None:
                line["reasoning"] = tokens
            self.write(line)
        self.write(dict(call(NOW, 0.05, provider="openrouter", phase="crux",
                             model="anthropic/claude-opus-5.5"), reasoning=1000),
                   dict(call(NOW, 0.01, provider="openrouter", phase="parser",
                             model="anthropic/claude-haiku-4.5"), reasoning=0))
        text = spend.breakdown(spend.load())
        self.assertIn("reasoning openrouter/anthropic/claude-opus-5.5: mean 600 tokens over 3 calls, max 1000",
                      text)
        self.assertIn("reasoning openrouter/anthropic/claude-haiku-4.5: mean 0 tokens over 1 calls, max 0",
                      text)

    def test_money_uses_spanish_separators(self):
        self.assertEqual(spend.money(1234.5), "1.234,50")
        self.assertEqual(spend.money(0.284), "0,28")


class WiringTests(unittest.TestCase):
    def read(self, *parts):
        with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as h:
            return h.read()

    def test_main_labels_every_phase(self):
        src = self.read("main.py")
        for label in ('spend.phase("research")', 'spend.phase("forecast")',
                      'spend.phase("summary")', 'spend.phase("parser")', "spend.phase(spend.CRUX)",
                      "spend.question(question)", "spend.install_litellm_recorder()"):
            self.assertIn(label, src)
        for tour in ("futureeval", "minibench", "cup", "market_pulse"):
            self.assertIn(f'spend.set_tournament("{tour}")', src)

    def test_main_goes_light_over_budget(self):
        src = self.read("main.py")
        start = src.index("if spend.over_budget():")
        self.assertIn('prize_depth = "light"', src[start:start + 120])
        self.assertIn("template_bot.research_depth = prize_depth", src[start:start + 300])

    def test_the_parser_wrapper_is_what_the_bot_calls(self):
        src = self.read("main.py")
        self.assertIn("structure_output as _sdk_structure_output", src)
        self.assertEqual(src.count("_sdk_structure_output("), 1)

    def test_the_production_job_keeps_the_ledger_in_the_cache_read_only(self):
        import yaml

        src = self.read(".github", "workflows", "run_bot_on_tournament.yaml")
        flow = yaml.safe_load(src)
        self.assertEqual(flow["permissions"], {"contents": "read"})
        job = flow["jobs"]["forecast_job"]
        self.assertEqual(job["env"]["SPEND_LEDGER_PATH"], "spend_ledger.jsonl")
        names = [s.get("name") for s in job["steps"]]
        restore = names.index("Restore the spend ledger")
        after = names.index("Spend alerts and run summary")
        save = names.index("Save the spend ledger")
        self.assertLess(restore, names.index("Run bot"))
        self.assertLess(names.index("Forecast Market Pulse questions (prize)"), after)
        self.assertLess(after, save)
        for index in (restore, after, save):
            self.assertTrue(job["steps"][index].get("continue-on-error"), names[index])
        self.assertEqual(job["steps"][save]["with"]["key"], job["steps"][restore]["with"]["key"])

    def test_the_weekly_report_reads_but_never_saves_the_ledger(self):
        import yaml

        steps = yaml.safe_load(self.read(".github", "workflows", "weekly_report.yaml"))["jobs"]["report"]["steps"]
        ledger_steps = [s for s in steps if s.get("with", {}).get("path") == "spend_ledger.jsonl"]
        self.assertEqual([s["uses"] for s in ledger_steps], ["actions/cache/restore@v4"])
        send = next(s for s in steps if s.get("name") == "Send the week's API spend")
        self.assertTrue(send.get("continue-on-error"))

    def test_the_ledger_is_never_committed(self):
        self.assertIn("spend_ledger.jsonl", self.read(".gitignore"))


if __name__ == "__main__":
    unittest.main()
