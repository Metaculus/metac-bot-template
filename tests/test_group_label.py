"""Group subquestions must tell the models which option they are about.

Run 39453516101 (2026-10-20): the Market Pulse group "What will Q4 2026 EPS
be for these companies?" reached the models without "Meta", three of five
forecasts landed on another company's numbers and were rejected, and the
step failed.
"""

from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace as NS

import live_context as lc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_src() -> str:
    with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
        return h.read()


class PromptTests(unittest.TestCase):
    def test_no_prompt_reads_the_bare_question_text(self):
        """Only _question_text itself may read it; every prompt goes through it."""
        self.assertEqual(main_src().count("{question.question_text}"), 1)

    def test_every_prompt_and_the_research_use_the_option_aware_text(self):
        src = main_src()
        self.assertGreaterEqual(src.count("{self._question_text(question)}"), 7)
        self.assertIn('getattr(question, "group_question_option", None)', src)

    def test_crux_and_comment_use_it_too(self):
        src = main_src()
        self.assertIn("disagreement.crux_prompt(self._question_text(question)", src)
        self.assertIn("disagreement.short_comment(\n            self._question_text(question)", src)


class LiveContextTests(unittest.TestCase):
    def test_the_option_feeds_the_market_search(self):
        q = NS(question_text="What will Q4 2026 EPS be for these companies?",
               group_question_option="Meta")
        self.assertTrue(lc._query(q).startswith("Meta"))

    def test_the_option_is_part_of_the_scanned_text(self):
        q = NS(question_text="Will these stocks close higher?", group_question_option="(NVDA)",
               background_info="", resolution_criteria="", fine_print="")
        self.assertIn("(NVDA)", lc._text_of(q))


class PartialFailureTests(unittest.TestCase):
    def test_one_failed_question_no_longer_fails_the_step(self):
        src = main_src()
        self.assertIn("all(isinstance(r, BaseException) for r in forecast_reports)", src)
        self.assertIn("partial_failure failed=", src)

    def test_a_step_where_everything_failed_exits_without_printing_the_errors(self):
        """The SDK's RuntimeError quotes each forecaster's error, numeric
        percentiles included; re-raised, Python's default traceback printed it
        past the log redaction into the public Actions log (runs 39453516101
        and 39965881683). The step must still fail, with a count only."""
        src = main_src()
        start = src.index("template_bot.log_report_summary(forecast_reports)")
        block = src[start:src.index("partial_failure failed=", start)]
        self.assertIn("raise SystemExit(1)", block)
        self.assertIn('logger.error("all_failed count=%d"', block)
        self.assertNotRegex(block, r"\n\s*raise\s*\n")
