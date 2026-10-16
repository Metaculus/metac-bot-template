"""Which tournaments the bot forecasts on, and the alarm for a stale season.

On 2026-10-01 production was still polling summer-futureeval-2026 (33022),
because main.py read forecasting_tools' CURRENT_AI_COMPETITION_ID and the
locked SDK predates the fall season. Every run was green and found nothing.
These tests pin the targets in tournaments.py and make the end of a season
visible instead of silent.
"""

from __future__ import annotations

import io
import os.path
import re
import unittest
from datetime import date, datetime, timezone

import tournaments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def code_lines(src: str) -> str:
    """The source without comment lines, so prose about a constant does not
    count as using it."""
    return "\n".join(line for line in src.splitlines() if not line.strip().startswith("#"))


class PinnedTargetsTests(unittest.TestCase):
    def test_the_seasonal_tournament_is_fall_futureeval_2026(self):
        """'The project ID is 33121 (or "fall-futureeval-2026")', from the
        tournament page."""
        self.assertEqual(tournaments.FUTUREEVAL.id, 33121)
        self.assertIn("fall-futureeval-2026", tournaments.FUTUREEVAL.url)
        self.assertTrue(tournaments.FUTUREEVAL.prize_eligible)

    def test_minibench_uses_the_alias_metaculus_keeps_current(self):
        self.assertEqual(tournaments.MINIBENCH.id, "minibench")
        self.assertIsNone(tournaments.MINIBENCH.ends_on)

    def test_the_cup_is_the_fall_2026_season_and_pays_no_bot(self):
        self.assertEqual(tournaments.METACULUS_CUP.id, 33108)
        self.assertIn("metaculus-cup-fall-2026", tournaments.METACULUS_CUP.url)
        self.assertFalse(tournaments.METACULUS_CUP.prize_eligible)

    def test_market_pulse_is_the_bot_eligible_26q4_challenge(self):
        """'compete for prizes using a forecasting bot account' and 'The ID
        for this tournament is "market-pulse-26q4"', from its page."""
        self.assertEqual(tournaments.MARKET_PULSE.id, "market-pulse-26q4")
        self.assertTrue(tournaments.MARKET_PULSE.prize_eligible)
        self.assertEqual(tournaments.MARKET_PULSE.ends_on, date(2026, 12, 31))

    def test_the_market_pulse_mode_targets_the_pinned_project(self):
        src = code_lines(read("main.py"))
        self.assertIn("tournaments.MARKET_PULSE.id", src)

    def test_main_reads_no_sdk_current_constant(self):
        """The SDK constants move only when the SDK is upgraded, which is
        exactly how 33022 outlived its season."""
        src = code_lines(read("main.py"))
        self.assertIsNone(
            re.search(r"\bCURRENT_[A-Z_]+_ID\b", src),
            "main.py must take tournament ids from tournaments.py",
        )

    def test_tournament_mode_forecasts_the_season_and_minibench(self):
        src = code_lines(read("main.py"))
        self.assertIn("forecast_on_tournament(\n                tournaments.FUTUREEVAL.id", src)
        self.assertIn("forecast_on_tournament(\n                tournaments.MINIBENCH.id", src)

    def test_the_cup_mode_targets_the_pinned_cup(self):
        src = code_lines(read("main.py"))
        self.assertIn("tournaments.METACULUS_CUP.id", src)

    def test_the_research_lab_counts_the_live_season(self):
        """Coverage for the season being played must be measurable without
        remembering to pass --tournaments."""
        src = read("research", "fetch_own_track_record.py")
        match = re.search(r"^DEFAULT_UNIVERSE_TOURNAMENTS = (\[.*\])$", src, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertIn(f'"{tournaments.FUTUREEVAL.id}"', match.group(1))
        self.assertIn('"minibench"', match.group(1))


class SeasonEndTests(unittest.TestCase):
    def test_the_last_day_still_counts_as_inside_the_season(self):
        end = tournaments.FUTUREEVAL.ends_on
        self.assertEqual(tournaments.ended((tournaments.FUTUREEVAL,), end), [])

    def test_the_day_after_the_last_day_is_over(self):
        self.assertEqual(
            tournaments.ended((tournaments.FUTUREEVAL,), date(2027, 1, 7)),
            [tournaments.FUTUREEVAL],
        )

    def test_an_alias_never_goes_stale(self):
        self.assertEqual(tournaments.ended((tournaments.MINIBENCH,), date(2100, 1, 1)), [])

    def test_a_warning_names_the_season_and_the_fix(self):
        warnings = tournaments.season_warnings(date(2027, 1, 7))
        self.assertEqual(len(warnings), 3)  # FutureEval, the Cup and Market Pulse
        for line in warnings:
            self.assertTrue(line.startswith("::warning::"))
            self.assertIn("tournaments.py", line)
        self.assertTrue(any("fall-futureeval-2026" in line for line in warnings))

    def test_no_warning_while_every_season_is_running(self):
        self.assertEqual(tournaments.season_warnings(date(2026, 10, 1)), [])

    def test_main_prints_the_warnings_at_startup(self):
        src = code_lines(read("main.py"))
        self.assertIn("tournaments.season_warnings(", src)

    def test_the_pinned_seasons_are_still_running(self):
        """A dated tripwire, on purpose. When this fails, the season in
        tournaments.py is over: pin the next one's project id (from its
        tournament page) and its last day. Do not move the date forward to
        make it pass."""
        today = datetime.now(timezone.utc).date()
        over = tournaments.ended(tournaments.ALL, today)
        self.assertEqual(
            over,
            [],
            "season over, rotate tournaments.py: "
            + ", ".join(f"{p.id} ended {p.ends_on}" for p in over),
        )


if __name__ == "__main__":
    unittest.main()
