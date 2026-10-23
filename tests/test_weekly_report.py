"""The Monday WhatsApp report: standings, weekly activity, and the message.

Pure functions only; the Metaculus calls go through MetaculusReadClient,
whose GET-only guarantee is tested elsewhere.
"""

from __future__ import annotations

import io
import os
import unittest
from datetime import date, datetime, timezone

from research import weekly_report as wr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def entry(rank, score, uid=1, name="someone", excluded=False):
    return {"rank": rank, "score": score, "excluded": excluded,
            "user": {"id": uid, "username": name}}


class StandingTests(unittest.TestCase):
    def test_finds_the_bot_and_counts_only_ranked_rows(self):
        payload = {"entries": [
            entry(None, 900, name="metac-autocast", excluded=True),
            entry(1, 783.2), entry(2, 700),
            entry(3, 66.94, uid=wr.BOT_USER_ID, name=wr.BOT_USERNAME),
            entry(None, -5),  # listed but unranked: not a position
        ]}
        self.assertEqual(wr.standing_from(payload), {"rank": 3, "total": 3, "score": 66.9})

    def test_matches_by_username_too(self):
        payload = {"entries": [entry(5, -13, uid=None, name=wr.BOT_USERNAME)]}
        self.assertEqual(wr.standing_from(payload)["rank"], 5)

    def test_falls_back_to_user_entry(self):
        payload = {"entries": [entry(1, 10)], "userEntry": {"rank": 59, "score": 67}}
        self.assertEqual(wr.standing_from(payload)["rank"], 59)

    def test_a_list_of_leaderboards_uses_the_primary_one(self):
        """The real endpoint answers with a list (seen 2026-10-17)."""
        payload = [
            {"is_primary_leaderboard": False, "entries": [entry(9, 1, uid=wr.BOT_USER_ID)]},
            {"is_primary_leaderboard": True, "entries": [entry(1, 783), entry(2, 700)],
             "userEntry": {"rank": 59, "score": 67.2, "user": {"id": wr.BOT_USER_ID}}},
        ]
        self.assertEqual(wr.standing_from(payload), {"rank": 59, "total": 2, "score": 67.2})

    def test_prize_money_is_read_from_the_rows(self):
        rows = [dict(entry(1, 10), prize=500), dict(entry(2, 11), prize=120),
                dict(entry(3, 5, uid=wr.BOT_USER_ID), prize=0), entry(4, 1)]
        s = wr.standing_from({"entries": rows})
        self.assertEqual((s["rank"], s.get("prize"), s["paid_ranks"]), (3, None, 2))
        self.assertEqual(s["paid_score"], 11.0)
        self.assertIn("sin premio (cobran hasta el 2, a 6.0 pts)", wr._fmt_prize(s))
        rows[2]["prize"] = 37.6
        s = wr.standing_from({"entries": rows})
        self.assertEqual(s["prize"], 38)
        self.assertIn("💰 38 $", wr._fmt_prize(s))
        self.assertEqual(wr._fmt_prize({"rank": 1}), "")

    def test_an_empty_leaderboard_is_none(self):
        self.assertIsNone(wr.standing_from([{"is_primary_leaderboard": True, "entries": []}]))

    def test_absent_or_unranked_is_none(self):
        self.assertIsNone(wr.standing_from({"entries": [entry(1, 10)]}))
        self.assertIsNone(wr.standing_from(None))
        self.assertIsNone(wr.standing_from({"entries": [entry(None, 3, uid=wr.BOT_USER_ID)]}))


class ActivityTests(unittest.TestCase):
    SINCE = datetime(2026, 10, 12, tzinfo=timezone.utc)

    def post(self, forecast_at=None, resolved_at=None, resolution="yes", project="33121"):
        q = {"my_forecasts": {"latest": {"start_time": forecast_at}},
             "actual_resolve_time": resolved_at, "resolution": resolution}
        return {"projects": {"tournament": [{"id": int(project) if project.isdigit() else 0,
                                              "slug": project}]}, "question": q}

    def test_counts_this_weeks_forecasts_and_resolutions(self):
        posts = [
            self.post("2026-10-13T10:00:00Z", "2026-10-15T00:00:00Z"),
            self.post(1760400000.0),            # epoch seconds, 2025: too old
            self.post("2026-10-01T10:00:00Z", "2026-10-01T00:00:00Z"),
            self.post("2026-10-14T10:00:00Z", project="33108"),  # other tournament
        ]
        self.assertEqual(wr.activity(posts, {"33121"}, self.SINCE),
                         {"forecast": 1, "resolved": 1})

    def test_unresolved_close_is_not_a_resolution(self):
        posts = [self.post(None, "2026-10-15T00:00:00Z", resolution=None)]
        self.assertEqual(wr.activity(posts, {"33121"}, self.SINCE)["resolved"], 0)

    def test_group_posts_count_each_subquestion(self):
        q = {"my_forecasts": {"latest": {"start_time": "2026-10-13T00:00:00Z"}}}
        post = {"projects": {"tournament": [{"slug": "market-pulse-26q4"}]},
                "group_of_questions": {"questions": [q, q, q]}}
        self.assertEqual(wr.activity([post], {"market-pulse-26q4"}, self.SINCE)["forecast"], 3)


class MessageTests(unittest.TestCase):
    ROWS = [
        {"key": "futureeval", "label": "FutureEval ($50k)",
         "standing": {"rank": 41, "total": 180, "score": 85.0},
         "activity": {"forecast": 7, "resolved": 2}},
        {"key": "market_pulse", "label": "Market Pulse ($7,5k)", "standing": None,
         "activity": {"forecast": 0, "resolved": 0}},
    ]

    def test_compares_in_positions_and_points_not_percent(self):
        prev = {"standings": {"futureeval": {"rank": 53, "total": 175, "score": 25.0}}}
        msg = wr.render(date(2026, 10, 12), date(2026, 10, 18), self.ROWS, prev)
        self.assertIn("puesto 41/180 (▲12)", msg)
        self.assertIn("+85 pts (▲60)", msg)
        self.assertNotIn("%", msg)

    def test_a_drop_shows_down_arrows(self):
        prev = {"standings": {"futureeval": {"rank": 30, "score": 120.0}}}
        msg = wr.render(date(2026, 10, 12), date(2026, 10, 18), self.ROWS, prev)
        self.assertIn("(▼11)", msg)
        self.assertIn("(▼35)", msg)

    def test_no_leaderboard_says_so_and_still_reports_activity(self):
        msg = wr.render(date(2026, 10, 12), date(2026, 10, 18), self.ROWS, None)
        self.assertIn("Market Pulse ($7,5k): sin clasificación aún", msg)
        self.assertIn("esta semana: 7 pronosticadas, 2 resueltas", msg)
        self.assertIn("primera semana", msg)


class TargetTests(unittest.TestCase):
    def test_previous_minibench_round_slug(self):
        keys = {t.key: t.project for t in wr.targets(date(2026, 10, 19))}
        self.assertEqual(keys["minibench_prev"], "minibench-2026-10-05")
        self.assertEqual(keys["minibench"], "minibench")

    def test_every_tournament_the_bot_plays_is_reported(self):
        import tournaments

        projects = {t.project for t in wr.targets(date(2026, 10, 19))}
        for p in tournaments.ALL:
            self.assertIn(str(p.id), projects)


class WiringTests(unittest.TestCase):
    def read(self, *parts):
        with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as h:
            return h.read()

    def test_per_question_alerts_are_gone_from_main(self):
        self.assertNotIn("handle_run(", self.read("main.py"))

    def test_the_failure_alert_stays(self):
        self.assertIn("Notify WhatsApp on workflow failure",
                      self.read(".github", "workflows", "run_bot_on_tournament.yaml"))

    def test_the_report_workflow_is_manual_read_only_and_never_forecasts(self):
        import yaml

        wf = yaml.safe_load(self.read(".github", "workflows", "weekly_report.yaml"))
        triggers = wf.get(True) or wf.get("on")
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        self.assertEqual(wf["permissions"], {"contents": "read"})
        src = self.read(".github", "workflows", "weekly_report.yaml")
        self.assertNotIn("main.py", src)
        self.assertNotIn("poetry install", src)
