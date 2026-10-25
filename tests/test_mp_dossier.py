"""Market Pulse earnings dossiers: which companies, the budget, and where they
reach the research. See mp_dossier.py for why. Every question here is
synthetic and no network is touched."""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import mp_dossier as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
MP = ["market-pulse-26q4"]


def q(title, option=None, close_days=3, criteria="", slugs=MP):
    return NS(question_text=title, group_question_option=option, resolution_criteria=criteria,
              close_time=NOW + timedelta(days=close_days), tournament_slugs=slugs)


EPS = "What will the first reported earnings per share after September 2026 be for the following companies?"
REV = "What will the first reported revenues after September 2026 be for the following companies?"
NVDA = "What will be NVIDIA's forward guidance in its Q3 FY2027 earnings release?"


class PlanTests(unittest.TestCase):
    def test_one_entry_per_company_with_every_metric_and_the_earliest_close(self):
        questions = [q(EPS, "Microsoft", 3, criteria="diluted EPS in conformity with US GAAP"),
                     q(REV, "Microsoft", 2), q(EPS, "Amazon", 4), q(NVDA, "Revenue", 30),
                     q(NVDA, "Gross margin (GAAP)", 30)]
        plan = md.plan(questions)
        self.assertEqual(set(plan), {"MSFT", "AMZN", "NVDA"})
        self.assertEqual(plan["MSFT"]["close"], NOW + timedelta(days=2))
        self.assertIn("earnings per share (GAAP diluted, as first reported)", plan["MSFT"]["metrics"])
        self.assertIn("total revenue, as first reported", plan["MSFT"]["metrics"])
        self.assertEqual(len(plan["NVDA"]["metrics"]), 2)

    def test_a_private_company_is_researched_by_name_but_a_metric_never_is(self):
        self.assertEqual(md._company(q(EPS, "SpaceX")), ("spacex", "SpaceX"))
        self.assertIsNone(md._company(q("What will these be?", "Operating expenses (GAAP)")))

    def test_the_prompt_asks_for_facts_never_a_forecast(self):
        text = md.prompt("MSFT", {"name": "Microsoft", "metrics": ["x"]}, NOW)
        self.assertIn("Do not forecast", text)
        self.assertIn("GAAP", text)
        self.assertIn("Microsoft (MSFT)", text)


class BuildTests(unittest.TestCase):
    def run_build(self, questions, budget, costs, fail=()):
        calls = []

        async def research(prompt, depth, engine_key, max_prior_cost, effort, on_cost):
            calls.append((depth, engine_key, effort))
            name = prompt.split(" on ", 1)[1].split(" ahead", 1)[0]
            on_cost(costs.pop(0))
            if any(f in name for f in fail):
                raise RuntimeError("down")
            return f"dossier for {name}"

        dossiers, spent = asyncio.run(md.build(questions, budget, NOW, research=research))
        return dossiers, spent, calls

    def test_soonest_close_first_and_stops_before_the_budget(self):
        questions = [q(EPS, "Amazon", 4), q(EPS, "Microsoft", 2), q(EPS, "Apple", 8)]
        dossiers, spent, calls = self.run_build(questions, 5.5, [2.0, 2.0, 2.0])
        self.assertEqual(list(dossiers), ["MSFT", "AMZN"])
        self.assertAlmostEqual(spent, 4.0)
        self.assertEqual(calls[0], ("dossier", "opus", "high"))

    def test_a_failed_company_still_counts_its_cost_and_the_rest_go_on(self):
        questions = [q(EPS, "Microsoft", 2), q(EPS, "Amazon", 4)]
        dossiers, spent, _ = self.run_build(questions, 20, [1.5, 1.0], fail=("MSFT",))
        self.assertEqual(list(dossiers), ["AMZN"])
        self.assertAlmostEqual(spent, 2.5)


class SectionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "d.json")
        with open(self.path, "w") as h:
            json.dump({"MSFT": {"t": NOW.isoformat(), "name": "Microsoft", "metrics": [],
                                "text": "FACTS " * 2000}}, h)

    def tearDown(self):
        self.dir.cleanup()

    def test_a_market_pulse_question_gets_its_companys_dossier_capped(self):
        text = md.section(q(EPS, "Microsoft"), NOW + timedelta(days=1), self.path)
        self.assertTrue(text.startswith("## Earnings dossier: Microsoft (researched 2026-10-25"))
        self.assertLessEqual(len(text), md.MAX_CHARS + 200)

    def test_other_tournaments_old_dossiers_and_errors_get_nothing(self):
        self.assertEqual(md.section(q(EPS, "Microsoft", slugs=["fall-futureeval-2026"]), NOW, self.path), "")
        self.assertEqual(md.section(q(EPS, "Microsoft"), NOW + timedelta(days=22), self.path), "")
        self.assertEqual(md.section(q(EPS, "Amazon"), NOW, self.path), "")
        self.assertEqual(md.section(q(EPS, "Microsoft"), NOW, os.path.join(self.dir.name, "none")), "")
        with open(self.path, "w") as h:
            h.write("{broken")
        self.assertEqual(md.section(q(EPS, "Microsoft"), NOW, self.path), "")


class WiringTests(unittest.TestCase):
    def read(self, *parts):
        with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as h:
            return h.read()

    def test_main_appends_the_dossier_to_the_research(self):
        src = self.read("main.py")
        start = src.index("async def _run_research")
        body = src[start:src.index("return research", start)]
        self.assertIn("mp_dossier.section(question)", body)

    def test_production_restores_the_cache_and_the_builder_never_forecasts(self):
        import yaml

        prod = yaml.safe_load(self.read(".github", "workflows", "run_bot_on_tournament.yaml"))
        job = prod["jobs"]["forecast_job"]
        self.assertEqual(job["env"]["MP_DOSSIERS_PATH"], "mp_dossiers.json")
        names = [s.get("name") for s in job["steps"]]
        restore = job["steps"][names.index("Restore the Market Pulse dossiers")]
        self.assertTrue(restore.get("continue-on-error"))
        self.assertEqual(restore["with"]["restore-keys"].strip(), "mp-dossiers-")
        self.assertLess(names.index("Restore the Market Pulse dossiers"),
                        names.index("Forecast Market Pulse questions (prize)"))
        src = self.read(".github", "workflows", "mp_dossiers.yaml")
        flow = yaml.safe_load(src)
        self.assertEqual(flow["permissions"], {"contents": "read"})
        self.assertEqual(set((flow.get("on") or flow.get(True))), {"workflow_dispatch"})
        self.assertNotIn("main.py", src)
        save = next(s for s in flow["jobs"]["build"]["steps"] if s.get("name") == "Save the dossiers")
        self.assertTrue(save["with"]["key"].startswith("mp-dossiers-"))

    def test_the_dossiers_are_never_committed(self):
        self.assertIn("mp_dossiers.json", self.read(".gitignore"))
