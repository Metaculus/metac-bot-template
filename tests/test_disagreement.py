"""Disagreement follow-up and the short public comment (disagreement.py)."""

from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace as NS

import disagreement as d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mc(**probs):
    return NS(predicted_options=[NS(option_name=k, probability=v) for k, v in probs.items()])


class SpreadTests(unittest.TestCase):
    def test_binary_spread(self):
        self.assertAlmostEqual(d.spread([0.2, 0.3, 0.6]), 0.4)

    def test_multiple_choice_uses_the_worst_option(self):
        self.assertAlmostEqual(d.spread([mc(a=0.7, b=0.3), mc(a=0.3, b=0.7)]), 0.4)

    def test_numeric_or_mixed_is_never_flagged(self):
        self.assertEqual(d.spread([object(), object()]), 0.0)
        self.assertEqual(d.spread([0.5]), 0.0)

    def test_threshold(self):
        self.assertFalse(d.needs_crux([0.30, 0.50]))
        self.assertTrue(d.needs_crux([0.20, 0.55]))


class ExtremesTests(unittest.TestCase):
    def test_binary_lowest_and_highest(self):
        preds = [NS(prediction_value=v, reasoning=str(v)) for v in (0.4, 0.1, 0.7)]
        low, high = d.extremes(preds)
        self.assertEqual((low.reasoning, high.reasoning), ("0.1", "0.7"))

    def test_multiple_choice_most_different_pair(self):
        preds = [NS(prediction_value=mc(a=0.5, b=0.5), reasoning="m"),
                 NS(prediction_value=mc(a=0.9, b=0.1), reasoning="x"),
                 NS(prediction_value=mc(a=0.1, b=0.9), reasoning="y")]
        low, high = d.extremes(preds)
        self.assertEqual({low.reasoning, high.reasoning}, {"x", "y"})


class PromptAndCommentTests(unittest.TestCase):
    def test_crux_prompt_carries_both_sides_and_the_dates(self):
        p = d.crux_prompt("Will X?", "Today is 2026-10-18.", "a" * 5000 + "LOWEND", "HIGHEND")
        self.assertIn("Today is 2026-10-18.", p)
        self.assertIn("LOWEND", p)
        self.assertIn("HIGHEND", p)
        self.assertLess(len(p), 4500)
        self.assertIn("GAPS:", p)

    def test_comment_is_short_and_names_no_model(self):
        c = d.short_comment("Will X?", "37.0%", ["30%", "40%"], "s" * 9000, "r" * 9000)
        self.assertLess(len(c), 7500)
        self.assertIn("Final prediction:** 37.0%", c)
        self.assertTrue(c.startswith("#"), "ForecastReport rejects explanations not starting with #")
        for model in ("opus", "gpt", "claude", "haiku", "sonnet"):
            self.assertNotIn(model, c.lower())

    def test_closest_to_the_aggregate(self):
        self.assertEqual(d.closest_to([0.1, 0.4, 0.8], 0.42), 1)
        self.assertEqual(d.closest_to([object()], object()), 0)


class WiringTests(unittest.TestCase):
    def setUp(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            self.src = h.read()

    def test_the_bot_overrides_research_and_comment(self):
        self.assertIn("async def _research_and_make_predictions(self, question", self.src)
        self.assertIn("disagreement.needs_crux(values)", self.src)
        self.assertIn('depth="crux"', self.src)
        self.assertIn("def _create_comment(self", self.src)

    def test_no_model_metadata_in_comments(self):
        self.assertIn("extra_metadata_in_explanation=False", self.src)

    def test_the_cup_gets_light_research(self):
        start = self.src.index('elif run_mode == "metaculus_cup":')
        end = self.src.index('elif run_mode == "market_pulse":')
        # Light outside a research boost (spend.boost_active).
        self.assertIn("template_bot.research_depth = other_depth", self.src[start:end])
        self.assertIn('other_depth = "full" if boost else "light"', self.src)
