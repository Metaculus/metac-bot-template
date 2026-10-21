"""Market Pulse: forecast new questions, refresh the ones about to close.

The tournament is spot scored at each question's close, so the forecast that
matters is the one standing then. These pin the selection rule in
market_pulse.py and the production wiring that runs it.
"""

from __future__ import annotations

import io
import os.path
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import market_pulse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


@dataclass
class Q:
    """The three attributes select() reads, nothing else."""

    already_forecasted: bool
    close_time: datetime | None = None
    timestamp_of_my_last_forecast: datetime | None = None


def ago(hours: float) -> datetime:
    return NOW - timedelta(hours=hours)


def ahead(hours: float) -> datetime:
    return NOW + timedelta(hours=hours)


class SelectionTests(unittest.TestCase):
    def test_a_question_never_forecast_is_selected(self):
        q = Q(already_forecasted=False, close_time=ahead(24 * 30))
        self.assertEqual(market_pulse.select([q], NOW), [q])

    def test_a_forecast_question_far_from_close_is_left_alone(self):
        q = Q(True, close_time=ahead(24 * 10), timestamp_of_my_last_forecast=ago(24 * 5))
        self.assertEqual(market_pulse.select([q], NOW), [])

    def test_a_stale_forecast_on_a_closing_question_is_refreshed(self):
        q = Q(True, close_time=ahead(20), timestamp_of_my_last_forecast=ago(24 * 5))
        self.assertEqual(market_pulse.select([q], NOW), [q])

    def test_a_recent_refresh_is_not_repeated_on_the_next_run(self):
        """Runs arrive ~6 times a day; without MIN_AGE every one of them
        inside the window would re-forecast the same question."""
        q = Q(True, close_time=ahead(20), timestamp_of_my_last_forecast=ago(4))
        self.assertEqual(market_pulse.select([q], NOW), [])

    def test_the_window_edges_are_inclusive(self):
        q = Q(
            True,
            close_time=NOW + market_pulse.REFRESH_WINDOW,
            timestamp_of_my_last_forecast=NOW - market_pulse.MIN_AGE,
        )
        self.assertEqual(market_pulse.select([q], NOW), [q])

    def test_unknown_close_or_last_forecast_is_not_refreshed(self):
        self.assertEqual(
            market_pulse.select(
                [
                    Q(True, close_time=None, timestamp_of_my_last_forecast=ago(48)),
                    Q(True, close_time=ahead(5), timestamp_of_my_last_forecast=None),
                ],
                NOW,
            ),
            [],
        )

    def test_a_timestamp_that_raises_counts_as_unknown(self):
        """The SDK property raises ValueError when its data disagrees with
        already_forecasted; one odd question must not abort the run."""

        class Raising:
            already_forecasted = True
            close_time = ahead(5)

            @property
            def timestamp_of_my_last_forecast(self):
                raise ValueError("inconsistent")

        self.assertEqual(market_pulse.select([Raising()], NOW), [])

    def test_naive_datetimes_are_read_as_utc(self):
        q = Q(
            True,
            close_time=ahead(20).replace(tzinfo=None),
            timestamp_of_my_last_forecast=ago(30).replace(tzinfo=None),
        )
        self.assertEqual(market_pulse.select([q], NOW), [q])

    def test_new_questions_come_before_refreshes(self):
        refresh = Q(True, close_time=ahead(10), timestamp_of_my_last_forecast=ago(30))
        new = Q(False)
        self.assertEqual(market_pulse.select([refresh, new], NOW), [new, refresh])


class StaleSelectionTests(unittest.TestCase):
    """The Cup's weekly refresh: new questions plus week-old forecasts."""

    def test_new_questions_are_selected(self):
        q = Q(False)
        self.assertEqual(market_pulse.select_stale([q], NOW), [q])

    def test_a_forecast_younger_than_a_week_is_left_alone(self):
        q = Q(True, close_time=ahead(2), timestamp_of_my_last_forecast=ago(24 * 6))
        self.assertEqual(market_pulse.select_stale([q], NOW), [])

    def test_a_week_old_forecast_is_refreshed_whatever_the_close(self):
        far = Q(True, close_time=ahead(24 * 60), timestamp_of_my_last_forecast=ago(24 * 7))
        self.assertEqual(market_pulse.select_stale([far], NOW), [far])

    def test_unknown_last_forecast_is_left_alone(self):
        q = Q(True, close_time=ahead(5), timestamp_of_my_last_forecast=None)
        self.assertEqual(market_pulse.select_stale([q], NOW), [])

    def test_new_come_before_stale(self):
        stale = Q(True, timestamp_of_my_last_forecast=ago(24 * 30))
        new = Q(False)
        self.assertEqual(market_pulse.select_stale([stale, new], NOW), [new, stale])

    def test_the_default_age_is_a_week(self):
        self.assertEqual(market_pulse.CUP_MAX_AGE.days, 7)


class WiringTests(unittest.TestCase):
    STEP = "Forecast Market Pulse questions (prize)"

    def _steps(self) -> dict:
        import yaml

        with io.open(
            os.path.join(ROOT, ".github", "workflows", "run_bot_on_tournament.yaml"),
            encoding="utf-8",
        ) as handle:
            steps = yaml.safe_load(handle)["jobs"]["forecast_job"]["steps"]
        return {s["name"]: (i, s) for i, s in enumerate(steps) if "name" in s}

    def test_production_runs_market_pulse_after_the_pin_step_and_the_window(self):
        """After the window, or its early exit never fires: Market Pulse
        always has open questions."""
        steps = self._steps()
        index, step = steps[self.STEP]
        self.assertLess(steps["Pin LLM models"][0], index)
        self.assertLess(steps["Run bot"][0], index)
        self.assertEqual(step["run"].strip(), "poetry run python main.py --mode market_pulse")
        self.assertIn("timeout-minutes", step)

    def test_a_failed_cup_step_does_not_skip_the_prize_step(self):
        steps = self._steps()
        self.assertLess(steps["Forecast new Metaculus Cup questions (practice)"][0],
                        steps[self.STEP][0])
        self.assertEqual(steps[self.STEP][1].get("if"), "${{ !cancelled() }}")

    def test_market_pulse_gets_the_same_credentials_as_run_bot(self):
        steps = self._steps()
        run_env = set(steps["Run bot"][1]["env"])
        expected = {n for n in run_env if not n.startswith(("POLL_", "CALLMEBOT_"))}
        self.assertEqual(set(steps[self.STEP][1]["env"]), expected)

    def test_the_mode_selects_before_forecasting(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            src = handle.read()
        start = src.index('elif run_mode == "market_pulse":')
        end = src.index('elif run_mode == "test_questions":')
        branch = src[start:end]
        self.assertIn("market_pulse.select(", branch)
        self.assertIn("forecast_questions(", branch)
        self.assertNotIn("forecast_on_tournament(", branch)


if __name__ == "__main__":
    unittest.main()


class LabelFixRefreshTests(unittest.TestCase):
    """Group subquestions forecast before the group option reached the
    prompts (main._question_text) are redone once, then left alone."""

    def q(self, last, option="Meta", close_days=60):
        return Q(True, close_time=ahead(24 * close_days),
                 timestamp_of_my_last_forecast=last) if option is None else \
            type("G", (), {"already_forecasted": True, "close_time": ahead(24 * close_days),
                           "timestamp_of_my_last_forecast": last,
                           "group_question_option": option})()

    def test_market_pulse_redoes_a_pre_fix_group_forecast_far_from_close(self):
        old = self.q(market_pulse.REFORECAST_BEFORE - timedelta(days=3))
        self.assertEqual(market_pulse.select([old], NOW + timedelta(days=30)), [old])

    def test_a_post_fix_forecast_is_not_redone(self):
        new = self.q(market_pulse.REFORECAST_BEFORE + timedelta(hours=1))
        self.assertEqual(market_pulse.select([new], NOW + timedelta(days=30)), [])

    def test_plain_questions_are_unaffected(self):
        plain = self.q(market_pulse.REFORECAST_BEFORE - timedelta(days=3), option=None)
        self.assertEqual(market_pulse.select([plain], NOW + timedelta(days=30)), [])

    def test_the_cup_redoes_pre_fix_group_forecasts_too(self):
        old = self.q(market_pulse.REFORECAST_BEFORE - timedelta(days=1))
        self.assertEqual(market_pulse.select_stale([old], market_pulse.REFORECAST_BEFORE), [old])
