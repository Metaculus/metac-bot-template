"""The caps on binary and multiple-choice forecasts, and the timing block.

See forecast_guards.py for why: the bot's worst losses were confident misses,
and its prompts never said when a question closes.
"""

from __future__ import annotations

import io
import itertools
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


def _sdk_validator(probabilities):
    """What PredictedOptionList's validator does to a list (SDK 0.2.90),
    copied so this module stays stdlib-only: refuse a sum outside
    [0.99, 1.01], clamp to [0.01, 0.99], renormalise with the rounding on the
    last option, and refuse any option that moved by more than 0.05."""
    total = sum(probabilities)
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"sum {total}")
    clamped = [max(min(p, 0.99), 0.01) for p in probabilities]
    out = [p / sum(clamped) for p in clamped]
    out[-1] += 1.0 - sum(out)
    for before, after in zip(probabilities, out):
        if abs(after - before) > 0.05:
            raise ValueError(f"{before} -> {after}")
    return out


#: Synthetic forecasts of every shape the floor has to handle.
SHAPES = (
    [1.0, 0.0], [0.9, 0.1, 0.0], [0.97, 0.03, 0.0, 0.0], [0.97, 0.01, 0.01, 0.01],
    [0.95] + [0.01] * 5, [0.5, 0.3, 0.15, 0.05, 0.0], [0.2] * 5, [0.0, 0.0, 1.0],
    [0.4, 0.4, 0.1, 0.05, 0.03, 0.01, 0.01, 0.0], [0.6, 0.3, 0.1, 0.0],
    [1.0] + [0.0] * 39, [0.7, 0.2] + [0.1 / 58] * 58,
)


class OptionFloorTests(unittest.TestCase):
    def test_the_floor_is_two_and_a_half_percent_up_to_four_options(self):
        for n in (2, 3, 4):
            self.assertEqual(fg.option_floor(n), fg.OPTION_FLOOR)
        self.assertEqual(fg.OPTION_FLOOR, 0.025)

    def test_the_floor_is_two_percent_from_five_options(self):
        for n in (5, 6, 10, 25):
            self.assertEqual(fg.option_floor(n), fg.OPTION_FLOOR_MANY)
        self.assertEqual(fg.OPTION_FLOOR_MANY, 0.02)

    def test_many_options_shrink_the_floor_to_half_the_mass(self):
        self.assertAlmostEqual(fg.option_floor(40), 0.5 / 40)
        out = fg.floor_options([1.0] + [0.0] * 39)
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        self.assertAlmostEqual(out[0], 0.5 + 0.5 / 40)

    def test_but_never_below_the_sdks_one_percent_clamp(self):
        """At 0.5/70 the validator would lift each outsider to 1% itself and
        pull the favourite down by more than 0.05, so it refused the list."""
        self.assertEqual(fg.option_floor(70), 0.01)
        _sdk_validator(fg.floor_options([1.0] + [0.0] * 69))

    def test_with_no_room_for_the_floor_it_is_uniform(self):
        out = fg.floor_options([1.0] + [0.0] * 119)
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        for p in out:
            self.assertAlmostEqual(p, 1 / 120)
        _sdk_validator(out)

    def test_ninety_five_and_five_ones_becomes_ninety_and_five_twos(self):
        """FutureEval 46191's shape. The old mixture gave 0.83/0.033x5."""
        out = fg.floor_options([0.95] + [0.01] * 5)
        self.assertAlmostEqual(out[0], 0.90, places=12)
        for p in out[1:]:
            self.assertAlmostEqual(p, 0.02, places=12)

    def test_every_option_reaches_the_floor_and_the_sum_is_exactly_one(self):
        for before in SHAPES:
            after = fg.floor_options(before)
            floor = fg.option_floor(len(before))
            self.assertEqual(len(after), len(before))
            self.assertAlmostEqual(sum(after), 1.0, places=12)
            for p in after:
                self.assertGreaterEqual(p, floor - 1e-12, (before, after))

    def test_the_ranking_is_kept(self):
        """Options above the floor keep their strict order; the ones lifted
        to it tie, and none of them passes an option that was above it."""
        for before in SHAPES:
            after = fg.floor_options(before)
            floor = fg.option_floor(len(before))
            for i, j in itertools.permutations(range(len(before)), 2):
                if before[i] > before[j]:
                    self.assertGreaterEqual(after[i], after[j] - 1e-12)
                if before[i] > before[j] and before[j] > floor:
                    self.assertGreater(after[i], after[j])

    def test_the_shortfall_comes_from_the_options_above_in_proportion(self):
        before = [0.6, 0.3, 0.1, 0.0]
        after = fg.floor_options(before)
        floor = fg.option_floor(4)
        self.assertEqual(after[3], floor)
        kept = [(a - floor) / (b - floor) for a, b in zip(after[:3], before[:3])]
        for share in kept:
            self.assertAlmostEqual(share, kept[0], places=12)
        self.assertAlmostEqual(kept[0], 1 - 0.025 / 0.925, places=12)
        self.assertEqual(max(range(3), key=lambda i: before[i] - after[i]), 0)

    def test_a_forecast_that_clears_the_floor_is_left_alone(self):
        for before in ([0.2] * 5, [0.5, 0.3, 0.2], [0.9, 0.025, 0.025, 0.05]):
            for a, b in zip(fg.floor_options(before), before):
                self.assertAlmostEqual(a, b, places=12)

    def test_input_that_does_not_sum_to_one_is_renormalised_first(self):
        out = fg.floor_options([1.9] + [0.02] * 5)
        self.assertAlmostEqual(out[0], 0.90, places=12)
        self.assertAlmostEqual(sum(out), 1.0, places=12)

    def test_the_sdk_validator_would_leave_the_result_unchanged(self):
        """PredictedOptionList clamps to [0.01, 0.99], renormalises, and
        rejects the result if that moved an option by more than 0.05 from
        what it was GIVEN. A floored list sums to 1 inside that range, so
        the validator's own pass is a no-op."""
        for before in SHAPES:
            after = fg.floor_options(before)
            for p in after:
                self.assertTrue(0.01 <= p <= 0.99, (before, after))
            for a, b in zip(_sdk_validator(after), after):
                self.assertAlmostEqual(a, b, places=12)

    def test_the_mean_of_floored_forecasts_still_clears_the_floor(self):
        """The SDK averages the five floored lists option by option."""
        five = [fg.floor_options(p) for p in (
            [0.95, 0.01, 0.01, 0.01, 0.01, 0.01], [0.8, 0.2, 0, 0, 0, 0],
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [1, 0, 0, 0, 0, 0], [0.6, 0, 0.4, 0, 0, 0])]
        mean = [sum(column) / 5 for column in zip(*five)]
        self.assertAlmostEqual(sum(mean), 1.0, places=12)
        for p in mean:
            self.assertGreaterEqual(p, fg.OPTION_FLOOR_MANY - 1e-12)

    def test_an_explicit_floor_is_still_capped(self):
        self.assertEqual(fg.option_floor(3, 0.05), 0.05)
        self.assertAlmostEqual(fg.option_floor(20, 0.05), 0.025)
        out = fg.floor_options([1.0, 0.0, 0.0], floor=0.05)
        self.assertAlmostEqual(out[1], 0.05, places=12)
        self.assertAlmostEqual(out[0], 0.90, places=12)

    def test_degenerate_input(self):
        self.assertEqual(fg.floor_options([]), [])
        self.assertEqual(fg.floor_options([0.0, 0.0]), [0.5, 0.5])
        out = fg.floor_options([-0.1, 0.6, 0.5])
        self.assertEqual(out[0], fg.OPTION_FLOOR)
        self.assertAlmostEqual(sum(out), 1.0, places=12)

    def test_the_loss_it_bounds(self):
        """The arithmetic in the constants' comments: 1% vs 2.5% vs 2% on an
        option the field holds at 15%."""
        self.assertAlmostEqual(100 * math.log(0.01 / 0.15), -270.8, places=1)
        self.assertAlmostEqual(100 * math.log(0.025 / 0.15), -179.2, places=1)
        self.assertAlmostEqual(100 * math.log(0.02 / 0.15), -201.5, places=1)

    def test_the_break_even_against_the_old_mixture(self):
        """OPTION_FLOOR_MANY's comment: on 95%/1%x5 the lift beats the old
        mixture unless the outsiders together win more than about 13% of the
        time, and an outsider winning costs about 50 points more."""
        old_top = 0.025 + (1 - 6 * 0.025) * 0.95
        old_out = 0.025 + (1 - 6 * 0.025) * 0.01
        self.assertAlmostEqual(old_top, 0.8325)
        new = fg.floor_options([0.95] + [0.01] * 5)
        gain = math.log(new[0] / old_top)
        loss = math.log(old_out / new[1])
        self.assertAlmostEqual(gain / (gain + loss), 0.13, places=2)
        self.assertAlmostEqual(100 * math.log(new[1] / old_out), -51.6, places=1)


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
        self.assertIn("never assume the question has already resolved",
                      fg.timing_context(NOW, None, None))


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
