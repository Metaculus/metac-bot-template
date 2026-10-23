"""The caps on binary and multiple-choice forecasts, and the timing block.

See forecast_guards.py for why: the bot's worst losses were confident misses,
and its prompts never said when a question closes.
"""

from __future__ import annotations

import io
import math
import os.path
import unittest
from datetime import datetime, timedelta, timezone

import forecast_guards as fg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


class BinaryClipTests(unittest.TestCase):
    def test_extremes_are_capped_symmetrically(self):
        self.assertEqual(fg.clip_binary(0.001), fg.BINARY_MIN)
        self.assertEqual(fg.clip_binary(0.999), 1 - fg.BINARY_MIN)

    def test_the_middle_is_untouched(self):
        for p in (0.03, 0.2, 0.5, 0.8, 0.97):
            self.assertAlmostEqual(fg.clip_binary(p), p)

    def test_the_cap_is_tighter_than_the_sdk_default(self):
        self.assertGreater(fg.BINARY_MIN, 0.01)


class OptionFloorTests(unittest.TestCase):
    def test_every_option_reaches_the_floor_and_the_sum_is_one(self):
        out = fg.floor_options([0.97, 0.03, 0.0, 0.0])
        self.assertAlmostEqual(sum(out), 1.0)
        for p in out:
            self.assertGreaterEqual(p, fg.OPTION_FLOOR - 1e-12)

    def test_the_ranking_is_kept(self):
        before = [0.5, 0.3, 0.15, 0.05, 0.0]
        after = fg.floor_options(before)
        self.assertEqual(sorted(range(5), key=lambda i: -before[i])[:4],
                         sorted(range(5), key=lambda i: -after[i])[:4])

    def test_the_sdk_validator_would_leave_the_result_unchanged(self):
        """PredictedOptionList clamps to [0.01, 0.99], renormalises, and
        rejects the result if that moved an option by more than 0.05 from
        what it was GIVEN. A floored list sums to 1 inside that range, so
        the validator's own pass is a no-op."""
        for before in ([1.0, 0.0], [0.9, 0.1, 0.0], [0.97, 0.01, 0.01, 0.01],
                       [0.2] * 5, [0.0, 0.0, 1.0]):
            after = fg.floor_options(before)
            self.assertAlmostEqual(sum(after), 1.0, places=12)
            for p in after:
                self.assertTrue(0.01 <= p <= 0.99, (before, after))

    def test_many_options_shrink_the_floor(self):
        out = fg.floor_options([1.0] + [0.0] * 39)
        self.assertAlmostEqual(sum(out), 1.0)
        self.assertGreater(out[0], 0.5)

    def test_degenerate_input(self):
        self.assertEqual(fg.floor_options([]), [])
        self.assertEqual(fg.floor_options([0.0, 0.0]), [0.5, 0.5])

    def test_the_loss_it_bounds(self):
        """The arithmetic in the module docstring: 1% vs 2.5% on an option
        the field holds at 15%."""
        self.assertAlmostEqual(100 * math.log(0.01 / 0.15), -270.8, places=1)
        self.assertAlmostEqual(100 * math.log(0.025 / 0.15), -179.2, places=1)


class TimingContextTests(unittest.TestCase):
    def test_states_today_close_resolution_and_days_left(self):
        text = fg.timing_context(NOW, NOW + timedelta(days=3), NOW + timedelta(days=10))
        self.assertIn("Today is 2026-10-10 12:00 UTC.", text)
        self.assertIn("closes on 2026-10-13 12:00 UTC (3.0 days from now)", text)
        self.assertIn("resolve on 2026-10-20 12:00 UTC (10.0 days from now)", text)

    def test_missing_dates_are_left_out_not_invented(self):
        text = fg.timing_context(NOW, None, None)
        self.assertNotIn("closes", text)
        self.assertNotIn("resolve on", text)

    def test_naive_times_are_read_as_utc(self):
        text = fg.timing_context(NOW.replace(tzinfo=None),
                                 (NOW + timedelta(days=1)).replace(tzinfo=None), None)
        self.assertIn("(1.0 days from now)", text)

    def test_it_warns_against_stale_knowledge(self):
        # 2026-10-23: was "never assume the question has already resolved",
        # which told the model to ignore a dated source showing the outcome
        # had already happened between writing and opening.
        text = fg.timing_context(NOW, None, None)
        self.assertIn("never assume an outcome without a dated, primary source showing it", text)
        self.assertNotIn("never assume the question has already resolved", text)


class WiringTests(unittest.TestCase):
    def setUp(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            self.src = handle.read()

    def test_every_forecast_prompt_and_the_research_prompt_carry_the_timing(self):
        self.assertGreaterEqual(self.src.count("self._timing(question)"), 5)

    def test_the_research_prompt_no_longer_asks_for_the_resolution(self):
        """Asking a model with no sources and no date whether the question
        'would resolve Yes or No' invited it to assert the outcome."""
        self.assertNotIn("if the question would resolve Yes or No", self.src)

    def test_binary_uses_the_guard_not_the_old_one_percent_clamp(self):
        self.assertIn("forecast_guards.clip_binary(", self.src)
        self.assertNotIn("max(0.01, min(0.99,", self.src)

    def test_multiple_choice_is_floored(self):
        self.assertIn("forecast_guards.floor_options(", self.src)


if __name__ == "__main__":
    unittest.main()


class PercentileReaderTests(unittest.TestCase):
    LINES = ("Percentile 10: {}\nPercentile 20: {}\nPercentile 40: {}\n"
             "Percentile 60: {}\nPercentile 80: {}\nPercentile 90: {}")

    def read(self, *values, before=""):
        return fg.read_percentiles(before + self.LINES.format(*values))

    def test_question_46224_is_read_as_written(self):
        # gpt-4o-mini turned these into 7840 ... 8400, outside a 4M-10M range.
        out = self.read(7840000, 7945000, 8060000, 8140000, 8265000, 8400000)
        self.assertEqual(out[0], (0.1, 7840000.0))
        self.assertEqual(out[-1], (0.9, 8400000.0))
        self.assertEqual([p for p, _ in out], [0.1, 0.2, 0.4, 0.6, 0.8, 0.9])

    def test_thousands_commas_markdown_and_signs(self):
        self.assertEqual(self.read("7,550,000", "7,750,000", "8,000,000", "8,220,000",
                                   "8,550,000", "8,900,000")[0], (0.1, 7550000.0))
        text = ("**Percentile 10:** $-1.5  \n**Percentile 20:** -1\n- Percentile 40: 0\n"
                "Percentile 60: 4.25\nPercentile 80: 5%\nPercentile 90: 6")
        self.assertEqual([v for _, v in fg.read_percentiles(text)], [-1.5, -1, 0, 4.25, 5, 6])

    def test_the_last_block_wins(self):
        draft = self.LINES.format(1, 2, 3, 4, 5, 6) + "\n\nOn reflection:\n"
        self.assertEqual(self.read(10, 20, 30, 40, 50, 60, before=draft)[0], (0.1, 10.0))

    def test_anything_ambiguous_goes_to_the_parser(self):
        self.assertIsNone(self.read("7.8 trillion", 8, 9, 10, 11, 12))
        self.assertIsNone(self.read("0,5", 1, 2, 3, 4, 5))
        self.assertIsNone(self.read("XX (lowest number value)", 1, 2, 3, 4, 5))
        self.assertIsNone(self.read(5, 4, 6, 7, 8, 9))
        self.assertIsNone(fg.read_percentiles("Percentile 10: 1\nPercentile 20: 2"))

    def test_main_tries_the_verbatim_lines_before_the_parser(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            src = h.read()
        start = src.index("async def _numeric_prompt_to_forecast")
        body = src[start:src.index("def _verbatim_numeric", start)]
        self.assertLess(body.index("self._verbatim_numeric("), body.index("self._parse_numeric("))


class ProbabilityReaderTests(unittest.TestCase):
    def test_the_mandated_last_line_is_read_as_written(self):
        self.assertAlmostEqual(fg.read_probability("(a) 22 days.\nProbability: 15%"), 0.15)
        self.assertAlmostEqual(fg.read_probability("Probability: 15.5 %"), 0.155)

    def test_markdown_quotes_and_case(self):
        for text in ("**Probability:** 15%", "**Probability**: 15%", "### Probability: 15%",
                     '"Probability: 15%"', "- Probability: 15%", "> probability: 15%",
                     "Probability: 15%.", "Probability: 15%\r\n"):
            self.assertAlmostEqual(fg.read_probability(text), 0.15, msg=text)

    def test_the_whole_scale_is_accepted_and_nothing_beyond(self):
        self.assertEqual(fg.read_probability("Probability: 0%"), 0.0)
        self.assertEqual(fg.read_probability("Probability: 100%"), 1.0)
        self.assertIsNone(fg.read_probability("Probability: 101%"))

    def test_the_last_labelled_line_wins(self):
        text = "Probability: 20%\n\nOn reflection the base rate is lower.\nProbability: 12%"
        self.assertAlmostEqual(fg.read_probability(text), 0.12)

    def test_ranges_go_to_the_parser(self):
        for text in ("Probability: 35-40%", "Probability: 35% - 40%", "Probability: 35% to 40%",
                     "Probability: 35%/40%", "Probability: 35 – 40%"):
            self.assertIsNone(fg.read_probability(text), text)

    def test_a_final_range_is_not_replaced_by_an_earlier_value(self):
        """Falling back to an earlier "Probability:" line would publish a
        draft the forecaster then revised."""
        self.assertIsNone(fg.read_probability("Probability: 20%\nProbability: 30-35%"))

    def test_anything_else_unclear_goes_to_the_parser(self):
        for text in ("", "No probability here.", 'Probability: ZZ%', "Probability: 0.35",
                     "Probability: ~15%", "Probability:\n15%", "Final probability: 15%",
                     "Probability of Yes: 15%"):
            self.assertIsNone(fg.read_probability(text), text)

    def test_main_tries_the_verbatim_line_before_the_parser(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            src = h.read()
        start = src.index("async def _binary_prompt_to_forecast")
        body = src[start:src.index("def _verbatim_probability", start)]
        self.assertLess(body.index("self._verbatim_probability("), body.index("structure_output("))
