"""The health check: coverage audit, FutureEval idle alert, Dagu silence.

See research/health_check.py for why it exists. Every Metaculus and GitHub
call goes to a fake; every post, question and run here is synthetic (ids in
the 900000s, no real question's forecast or community prediction anywhere),
because the repository is public.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from research import health_check as hc
from research.metaculus_read_api import MetaculusReadError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
#: Synthetic forecast and community values. They must never reach a log
#: line or a message.
SECRET_VALUES = ("0.913579", "0.246802", "0.135791")


def ts(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def question(qid, status="closed", closes=NOW - timedelta(hours=3), mine=None):
    """One question as the API lays it out. ``mine``: None for a listing
    (my_forecasts null), True/False for a detail with or without a forecast
    of the bot's."""
    q = {"id": qid, "status": status, "scheduled_close_time": ts(closes),
         "actual_close_time": ts(closes) if status != "open" else None,
         "aggregations": {"recency_weighted": {"latest": {"centers": [float(SECRET_VALUES[1])]}}}}
    if mine is True:
        q["my_forecasts"] = {"latest": {"forecast_values": [float(SECRET_VALUES[0])]},
                             "history": [{"forecast_values": [float(SECRET_VALUES[0])]}]}
    elif mine is False:
        q["my_forecasts"] = {"latest": None, "history": []}
    else:
        q["my_forecasts"] = None
    return q


def post(pid, *questions, status=None):
    if len(questions) == 1 and status is None:
        return {"id": pid, "status": questions[0]["status"], "question": questions[0]}
    return {"id": pid, "status": status or "open",
            "group_of_questions": {"questions": list(questions)}}


class FakeClient:
    """iter_posts and get_json, as MetaculusReadClient has them. Nothing else:
    a write method called by mistake fails the test with AttributeError."""

    def __init__(self, listings=None, details=None, fail_listing=(), fail_detail=()):
        self.listings = listings or {}
        self.details = details or {}
        self.fail_listing = set(fail_listing)
        self.fail_detail = set(fail_detail)
        self.params = []
        self.detail_calls = []

    def iter_posts(self, params, max_pages=200):
        self.params.append((dict(params), max_pages))
        if params["tournaments"] in self.fail_listing:
            raise MetaculusReadError("GET /posts/ failed (status=503)", status=503)
        return iter(self.listings.get(params["tournaments"], []))

    def get_json(self, path, params=None):
        self.detail_calls.append(path)
        post_id = int(path.strip("/").split("/")[-1])
        if post_id in self.fail_detail:
            raise MetaculusReadError("GET failed (status=500)", status=500)
        return self.details[post_id]


class AuditTests(unittest.TestCase):
    def test_counts_questions_closed_in_the_window_and_confirms_each_miss(self):
        listing = [
            post(900001, question(1, mine=None)),                           # forecast (detail)
            post(900002, question(2, mine=None)),                           # missing (detail)
            post(900003, question(3, closes=NOW - timedelta(hours=30))),   # outside the window
            post(900004, question(4, status="open", closes=NOW + timedelta(hours=2))),
        ]
        details = {900001: post(900001, question(1, mine=True)),
                   900002: post(900002, question(2, mine=False))}
        client = FakeClient({33121: listing}, details)
        result = hc.audit(client, 33121, NOW)
        self.assertEqual((result.closed, result.missing, result.missing_posts, result.unchecked),
                         (2, 1, [900002], 0))
        self.assertEqual(result.open_now, 1)
        self.assertEqual(sorted(client.detail_calls), ["/posts/900001/", "/posts/900002/"])
        params, max_pages = client.params[0]
        self.assertEqual(params["statuses"], ["open", "closed", "resolved"])
        self.assertEqual(max_pages, hc.MAX_PAGES)

    def test_a_group_is_audited_per_subquestion_while_the_post_is_open(self):
        """Market Pulse: the group stays open while its subquestions close one
        earnings day at a time."""
        listing = [post(900010,
                        question(11), question(12), question(13, status="open", closes=NOW + timedelta(days=3)),
                        question(14, status="upcoming", closes=NOW + timedelta(days=9)),
                        status="open")]
        details = {900010: post(900010, question(11, mine=True), question(12, mine=False),
                                question(13, status="open", mine=False), status="open")}
        result = hc.audit(FakeClient({"market-pulse-26q4": listing}, details), "market-pulse-26q4", NOW)
        self.assertEqual((result.closed, result.missing, result.missing_posts), (2, 1, [900010]))
        self.assertEqual(result.open_now, 1)

    def test_a_listing_that_shows_the_forecast_needs_no_detail_call(self):
        client = FakeClient({"minibench": [post(900020, question(20, mine=True))]})
        result = hc.audit(client, "minibench", NOW)
        self.assertEqual((result.closed, result.missing), (1, 0))
        self.assertEqual(client.detail_calls, [])

    def test_what_cannot_be_confirmed_is_unchecked_not_missing(self):
        listing = [post(900030, question(30)), post(900031, question(31))]
        details = {900031: post(900031, question(31, mine=None))}
        result = hc.audit(FakeClient({1: listing}, details, fail_detail={900030}), 1, NOW)
        self.assertEqual((result.closed, result.missing, result.unchecked), (2, 0, 2))
        self.assertIn("unchecked=2", result.line())

    def test_detail_reads_are_bounded(self):
        listing = [post(900100 + i, question(100 + i)) for i in range(hc.MAX_DETAILS + 5)]
        details = {900100 + i: post(900100 + i, question(100 + i, mine=True))
                   for i in range(hc.MAX_DETAILS + 5)}
        client = FakeClient({1: listing}, details)
        result = hc.audit(client, 1, NOW)
        self.assertEqual(len(client.detail_calls), hc.MAX_DETAILS)
        self.assertEqual((result.missing, result.unchecked), (0, 5))

    def test_the_log_line_carries_only_counts_and_ids(self):
        listing = [post(900040, question(40)), post(900041, question(41))]
        details = {900040: post(900040, question(40, mine=False)),
                   900041: post(900041, question(41, mine=False))}
        line = hc.audit(FakeClient({33121: listing}, details), 33121, NOW).line()
        self.assertEqual(line, "coverage_audit tournament=33121 closed=2 missing=2 ids=[900040,900041]")


class IdleTests(unittest.TestCase):
    def test_hours_since_the_last_open_question(self):
        self.assertEqual(hc.idle_hours(hc.Audit("1", open_now=2), NOW), 0.0)
        self.assertEqual(hc.idle_hours(hc.Audit("1", last_closed=NOW - timedelta(hours=40)), NOW), 40.0)
        self.assertIsNone(hc.idle_hours(hc.Audit("1"), NOW))

    def test_the_latest_close_is_read_from_the_whole_listing(self):
        listing = [post(900050, question(50, closes=NOW - timedelta(hours=50))),
                   post(900051, question(51, closes=NOW - timedelta(hours=38)))]
        result = hc.audit(FakeClient({1: listing}), 1, NOW)
        self.assertEqual(result.closed, 0)
        self.assertAlmostEqual(hc.idle_hours(result, NOW), 38.0)


class DailyTests(unittest.TestCase):
    def run_daily(self, listings, details=None, ok=True, **client_kwargs):
        sent, logs = [], []

        def send(text):
            sent.append(text)
            return ok

        client = FakeClient(listings, details, **client_kwargs)
        done = hc.daily(client, NOW, send, logs.append)
        return done, sent, logs

    def quiet(self):
        """FutureEval busy, everything forecast."""
        return {33121: [post(900060, question(60, mine=True)),
                        post(900061, question(61, status="open", closes=NOW + timedelta(hours=1)))],
                "minibench": [], "market-pulse-26q4": []}

    def test_a_healthy_day_sends_nothing_and_is_done(self):
        done, sent, logs = self.run_daily(self.quiet())
        self.assertTrue(done)
        self.assertEqual(sent, [])
        self.assertIn("coverage_audit tournament=33121 closed=1 missing=0 ids=[]", logs)
        self.assertIn("futureeval_idle hours=0 alert=0", logs)

    def test_misses_and_an_idle_futureeval_send_one_message(self):
        listings = {33121: [post(900070, question(70, closes=NOW - timedelta(hours=40)))],
                    "minibench": [post(900071, question(71))],
                    "market-pulse-26q4": [post(900072, question(72), question(73), status="open")]}
        details = {900071: post(900071, question(71, mine=False)),
                   900072: post(900072, question(72, mine=False), question(73, mine=False), status="open")}
        done, sent, logs = self.run_daily(listings, details)
        self.assertTrue(done)
        [message] = sent
        self.assertIn("MiniBench: 1 de 1 (post 900071)", message)
        self.assertIn("Market Pulse: 2 de 2 (post 900072)", message)
        self.assertIn("FutureEval (33121) lleva 40 h sin ninguna pregunta abierta", message)
        self.assertIn("tournaments.py", message)
        self.assertIn("futureeval_idle hours=40 alert=1", logs)

    def test_a_futureeval_with_no_question_at_all_alerts(self):
        listings = {**self.quiet(), 33121: []}
        _done, [message], logs = self.run_daily(listings)
        self.assertIn("no muestra ninguna pregunta", message)
        self.assertIn("futureeval_idle hours=none alert=1", logs)

    def test_a_failed_send_is_retried_on_the_next_run(self):
        listings = dict(self.quiet(), minibench=[post(900080, question(80))])
        details = {900080: post(900080, question(80, mine=False))}
        done, sent, _logs = self.run_daily(listings, details, ok=False)
        self.assertFalse(done)
        self.assertEqual(len(sent), 1)

    def test_one_unreadable_tournament_does_not_stop_the_others(self):
        done, sent, logs = self.run_daily(self.quiet(), fail_listing={"minibench"})
        self.assertFalse(done, "a partial read with nothing to send is retried")
        self.assertEqual(sent, [])
        self.assertIn("health_check_failed part=coverage tournament=minibench "
                      "reason=MetaculusReadError status=503", logs)
        self.assertTrue(any(l.startswith("coverage_audit tournament=market-pulse-26q4") for l in logs))

    def test_no_forecast_or_community_value_reaches_a_log_or_a_message(self):
        listings = {33121: [post(900090, question(90, mine=True))],
                    "minibench": [post(900091, question(91))],
                    "market-pulse-26q4": [post(900092, question(92), question(93), status="open")]}
        details = {900091: post(900091, question(91, mine=False)),
                   900092: post(900092, question(92, mine=True), question(93, mine=False), status="open")}
        _done, sent, logs = self.run_daily(listings, details)
        text = "\n".join(logs + sent)
        for value in SECRET_VALUES:
            self.assertNotIn(value, text)
            self.assertNotIn(value.rstrip("0")[:5], text)


def run_entry(rid, event, created, started=None, ended=None):
    return {"id": rid, "event": event, "created_at": ts(created),
            "run_started_at": ts(started or created), "updated_at": ts(ended or created)}


class DaguSilenceTests(unittest.TestCase):
    T = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)

    def at(self, minutes):
        return self.T + timedelta(minutes=minutes)

    def test_the_first_cron_run_after_a_silent_dispatch_alerts(self):
        runs = [run_entry(2, "schedule", self.at(127)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(7))]
        self.assertEqual(hc.dagu_silence(runs, 2), (120.0, 0.0))

    def test_a_short_gap_is_not_silence(self):
        runs = [run_entry(2, "schedule", self.at(40)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(7))]
        minutes, _before = hc.dagu_silence(runs, 2)
        self.assertLess(minutes, hc.DAGU_SILENT_MINUTES)

    def test_time_another_run_was_active_does_not_count(self):
        """Dagu does not launch while a run is active: a long dispatch run, or
        a cron run queued behind one, is not silence."""
        runs = [run_entry(3, "schedule", self.at(95), started=self.at(95)),
                run_entry(2, "schedule", self.at(30), started=self.at(30), ended=self.at(95)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(31))]
        self.assertEqual(hc.dagu_silence(runs, 3), (0.0, 0.0))

    def test_a_cron_run_queued_right_after_the_dispatch_does_not_hide_the_silence(self):
        runs = [run_entry(3, "schedule", self.at(300)),
                run_entry(2, "schedule", self.at(8), ended=self.at(50)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(7))]
        minutes, before = hc.dagu_silence(runs, 3)
        self.assertEqual((minutes, before), (251.0, 1.0))
        self.assertTrue(minutes > hc.DAGU_SILENT_MINUTES >= before)

    def test_a_silence_already_reported_is_not_reported_again(self):
        runs = [run_entry(3, "schedule", self.at(400)),
                run_entry(2, "schedule", self.at(180), ended=self.at(190)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(7))]
        minutes, before = hc.dagu_silence(runs, 3)
        self.assertGreater(before, hc.DAGU_SILENT_MINUTES)
        self.assertGreater(minutes, before)

    def test_runs_created_after_this_one_are_ignored(self):
        runs = [run_entry(3, "schedule", self.at(140)),
                run_entry(2, "schedule", self.at(127)),
                run_entry(1, "workflow_dispatch", self.at(0), ended=self.at(7))]
        self.assertEqual(hc.dagu_silence(runs, 2), (120.0, 0.0))

    def test_no_verdict_without_this_run_or_a_dispatch(self):
        runs = [run_entry(2, "schedule", self.at(127)), run_entry(1, "schedule", self.at(0))]
        self.assertIsNone(hc.dagu_silence(runs, 2))
        self.assertIsNone(hc.dagu_silence(runs, 99))


class DaguCheckTests(unittest.TestCase):
    T = DaguSilenceTests.T
    ENV = {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "owner/repo", "GITHUB_RUN_ID": "2"}

    def check(self, runs, env=None):
        sent, logs, asked = [], [], []

        def fetch(repo, token):
            asked.append(repo)
            return runs

        def send(text):
            sent.append(text)
            return True

        hc.dagu(self.ENV if env is None else env, fetch, send, logs.append)
        return sent, logs, asked

    def test_one_whatsapp_with_the_minutes_and_the_dagu_link(self):
        runs = [run_entry(2, "schedule", self.T + timedelta(minutes=127)),
                run_entry(1, "workflow_dispatch", self.T, ended=self.T + timedelta(minutes=7))]
        [message], logs, asked = self.check(runs)
        self.assertIn("Dagu lleva 120 min sin lanzar el bot", message)
        self.assertIn(hc.DAGU_URL, message)
        self.assertIn("dagu_silent minutes=120", logs)
        self.assertEqual(asked, ["owner/repo"])

    def test_a_healthy_dagu_sends_nothing(self):
        runs = [run_entry(2, "schedule", self.T + timedelta(minutes=20)),
                run_entry(1, "workflow_dispatch", self.T, ended=self.T + timedelta(minutes=7))]
        sent, logs, _ = self.check(runs)
        self.assertEqual(sent, [])
        self.assertIn("dagu_check idle_minutes=13 alert=0", logs)

    def test_without_the_job_context_nothing_is_fetched(self):
        sent, logs, asked = self.check([], env={})
        self.assertEqual((sent, asked), ([], []))
        self.assertIn("dagu_check skipped reason=no_github_context", logs)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FetchRunsTests(unittest.TestCase):
    def test_one_authenticated_get_of_this_workflows_runs(self):
        seen = []

        def opener(request, timeout):
            seen.append(request)
            return FakeResponse(json.dumps({"total_count": 2, "workflow_runs": [{"id": 1}, "junk"]}).encode())

        self.assertEqual(hc.fetch_runs("owner/repo", "tok", opener=opener), [{"id": 1}])
        [request] = seen
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url,
                         "https://api.github.com/repos/owner/repo/actions/workflows/"
                         "run_bot_on_tournament.yaml/runs?per_page=30")
        self.assertEqual(request.get_header("Authorization"), "Bearer tok")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.marker = os.path.join(self.dir.name, "health_check.done")
        self.output = os.path.join(self.dir.name, "output")

    def tearDown(self):
        self.dir.cleanup()

    def quiet_client(self):
        return FakeClient({33121: [post(900200, question(200, status="open",
                                                          closes=NOW + timedelta(hours=1)))]})

    def go(self, client=None, env=None, marker_day=None, fetch=None):
        if marker_day:
            with open(self.marker, "w") as handle:
                handle.write(marker_day + "\n")
        logs, sent = [], []
        environ = {"METACULUS_TOKEN": "x", "GITHUB_OUTPUT": self.output}
        environ.update(env or {})
        code = hc.run(["--marker", self.marker], now=NOW, client=client or self.quiet_client(),
                      fetch=fetch or (lambda repo, token: []),
                      send=lambda text: sent.append(text) or True, environ=environ, log=logs.append)
        return code, logs, sent

    def output_text(self):
        if not os.path.exists(self.output):
            return ""
        with open(self.output) as handle:
            return handle.read()

    def test_a_completed_day_writes_the_marker_and_the_cache_key(self):
        code, logs, _sent = self.go(marker_day="2026-10-22")
        self.assertEqual(code, 0)
        with open(self.marker) as handle:
            self.assertEqual(handle.read().strip(), "2026-10-23 2026-10-23T12:00:00Z")
        self.assertEqual(self.output_text(), "daily_key=health-2026-10-23\n")

    def test_the_daily_checks_run_once_a_day(self):
        client = self.quiet_client()
        _code, logs, _sent = self.go(client=client, marker_day="2026-10-23")
        self.assertIn("health_check daily=skipped reason=already_ran_today", logs)
        self.assertEqual(client.params, [])
        self.assertEqual(self.output_text(), "")

    def test_an_incomplete_day_is_not_marked(self):
        client = FakeClient({}, fail_listing={33121})
        self.go(client=client, marker_day="2026-10-22")
        with open(self.marker) as handle:
            self.assertEqual(handle.read().strip(), "2026-10-22")
        self.assertEqual(self.output_text(), "")

    def test_without_a_token_nothing_is_read(self):
        logs = []
        code = hc.run(["--marker", self.marker], now=NOW, environ={}, log=logs.append,
                      send=lambda text: True)
        self.assertEqual(code, 0)
        self.assertIn("health_check daily=skipped reason=no_metaculus_token", logs)

    def test_the_dagu_check_runs_on_cron_runs_only(self):
        asked = []

        def fetch(repo, token):
            asked.append(repo)
            return []

        env = dict(DaguCheckTests.ENV)
        self.go(env=env, marker_day="2026-10-23", fetch=fetch)
        self.assertEqual(asked, [])
        self.go(env=dict(env, GITHUB_EVENT_NAME="workflow_dispatch"), fetch=fetch)
        self.assertEqual(asked, [])
        _code, logs, _sent = self.go(env=dict(env, GITHUB_EVENT_NAME="schedule"), fetch=fetch)
        self.assertEqual(asked, ["owner/repo"])
        self.assertIn("dagu_check skipped reason=no_dispatch_or_run_listed", logs)

    def test_nothing_it_does_can_fail_the_job(self):
        def broken(repo, token):
            raise OSError("network down")

        code, logs, _sent = self.go(env=dict(DaguCheckTests.ENV, GITHUB_EVENT_NAME="schedule"),
                                    fetch=broken)
        self.assertEqual(code, 0)
        self.assertIn("health_check_failed part=dagu reason=OSError", logs)


class WindowTests(unittest.TestCase):
    """Each audit starts where the last completed one ended, so audits chain
    however late each day's runs: with a fixed "last 24 h", a miss was
    reported two days running after a late check, and one that closed between
    an early check and a late one the next day was never audited
    (2026-10-23 review)."""

    D = datetime(2026, 10, 24, tzinfo=timezone.utc)

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.marker = os.path.join(self.dir.name, "health_check.done")

    def client(self, closes):
        listing = {33121: [post(900300, question(300, closes=closes)),
                           post(900301, question(301, status="open", closes=closes + timedelta(days=9)))]}
        return FakeClient(listing, {900300: post(900300, question(300, closes=closes, mine=False))})

    def day(self, now, closes):
        sent, logs = [], []
        hc.run(["--marker", self.marker], now=now, client=self.client(closes),
               send=lambda text: sent.append(text) or True, environ={"METACULUS_TOKEN": "x"},
               log=logs.append)
        return sent, logs

    def test_a_miss_is_reported_once_after_a_late_check(self):
        closes = self.D + timedelta(hours=2)
        first, _ = self.day(self.D + timedelta(hours=5), closes)
        second, _ = self.day(self.D + timedelta(days=1, minutes=5), closes)
        self.assertEqual((len(first), len(second)), (1, 0))

    def test_a_miss_between_an_early_and_a_late_check_is_reported(self):
        closes = self.D + timedelta(hours=3)
        first, _ = self.day(self.D + timedelta(minutes=5), closes)
        second, logs = self.day(self.D + timedelta(days=1, hours=6), closes)
        self.assertEqual((len(first), len(second)), (0, 1))
        self.assertIn("coverage_window hours=30", logs)
        self.assertIn("Cerradas en las últimas 30 h", second[0])

    def test_the_window_reaches_back_at_most_72_hours(self):
        with open(self.marker, "w") as handle:
            handle.write("2026-10-14 2026-10-14T06:00:00Z\n")
        missed_long_ago = self.D - timedelta(hours=80)
        sent, logs = self.day(self.D + timedelta(hours=1), missed_long_ago)
        self.assertEqual(sent, [])
        self.assertIn("coverage_window hours=72", logs)
        sent, _ = self.day(self.D + timedelta(days=1, hours=1), self.D - timedelta(hours=60))
        self.assertEqual(sent, [], "already inside the previous audit")

    def test_a_marker_without_an_instant_means_the_last_24_hours(self):
        with open(self.marker, "w") as handle:
            handle.write("2026-10-23\n")
        sent, logs = self.day(self.D + timedelta(hours=12), self.D - timedelta(hours=10))
        self.assertIn("coverage_window hours=24", logs)
        self.assertEqual(len(sent), 1)
        with open(self.marker) as handle:
            self.assertEqual(handle.read().strip(), "2026-10-24 2026-10-24T12:00:00Z")

    def test_a_failed_audit_keeps_the_start_for_the_next_run(self):
        with open(self.marker, "w") as handle:
            handle.write("2026-10-23 2026-10-23T08:00:00Z\n")
        failing = FakeClient({}, fail_listing={33121})
        hc.run(["--marker", self.marker], now=self.D + timedelta(hours=1), client=failing,
               send=lambda text: True, environ={"METACULUS_TOKEN": "x"}, log=lambda _: None)
        sent, logs = self.day(self.D + timedelta(hours=9), self.D)
        self.assertIn("coverage_window hours=25", logs)
        self.assertEqual(len(sent), 1)


class DaguRetryTests(unittest.TestCase):
    """--dagu-marker: one WhatsApp per silence, retried until one goes out.
    Judged on the run order alone, a failed send or a cancelled run left the
    rest of an outage silent (2026-10-23 review)."""

    T = DaguSilenceTests.T
    ENV = dict(DaguCheckTests.ENV, GITHUB_EVENT_NAME="schedule")

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.dagu_marker = os.path.join(self.dir.name, "dagu_alert.done")
        self.output = os.path.join(self.dir.name, "output")
        self.marker = os.path.join(self.dir.name, "health_check.done")
        with open(self.marker, "w") as handle:  # the daily pair already ran today
            handle.write(self.T.date().isoformat() + "\n")

    def runs(self, current, minutes, dispatch=1):
        """This cron run ``minutes`` after a dispatch ended, with an earlier
        cron run already past the threshold in between."""
        end = self.T + timedelta(minutes=7)
        return [run_entry(current, "schedule", end + timedelta(minutes=minutes)),
                run_entry(current - 1, "schedule", end + timedelta(minutes=95)),
                run_entry(dispatch, "workflow_dispatch", self.T, ended=end)]

    def cron(self, current, minutes, ok=True, dispatch=1):
        sent, logs = [], []
        env = dict(self.ENV, GITHUB_RUN_ID=str(current), GITHUB_OUTPUT=self.output)
        hc.run(["--marker", self.marker, "--dagu-marker", self.dagu_marker],
               now=self.T, client=FakeClient({}), environ=env, log=logs.append,
               fetch=lambda repo, token: self.runs(current, minutes, dispatch),
               send=lambda text: sent.append(text) or ok)
        return sent, logs

    def output_text(self):
        if not os.path.exists(self.output):
            return ""
        with open(self.output) as handle:
            return handle.read()

    def test_a_failed_send_is_retried_by_the_next_cron_run(self):
        sent, logs = self.cron(3, 120, ok=False)
        self.assertEqual(len(sent), 1)
        self.assertIn("dagu_alert sent=0", logs)
        self.assertFalse(os.path.exists(self.dagu_marker))
        self.assertNotIn("dagu_key", self.output_text())
        sent, _ = self.cron(4, 180)
        self.assertEqual(len(sent), 1, "the old rule saw before > 90 and stayed silent")
        with open(self.dagu_marker) as handle:
            self.assertEqual(handle.read().strip(), "1")
        self.assertIn("dagu_key=dagu-alerted-1\n", self.output_text())

    def test_once_it_went_out_the_same_silence_is_not_reported_again(self):
        with open(self.dagu_marker, "w") as handle:
            handle.write("1\n")
        sent, logs = self.cron(5, 240)
        self.assertEqual(sent, [])
        self.assertIn("dagu_check idle_minutes=240 alert=0", logs)

    def test_a_new_silence_after_a_new_dispatch_is_reported(self):
        with open(self.dagu_marker, "w") as handle:
            handle.write("1\n")
        sent, _ = self.cron(9, 120, dispatch=7)
        self.assertEqual(len(sent), 1)
        with open(self.dagu_marker) as handle:
            self.assertEqual(handle.read().strip(), "7")

    def test_a_healthy_dagu_writes_nothing(self):
        sent, _ = self.cron(3, 20)
        self.assertEqual(sent, [])
        self.assertFalse(os.path.exists(self.dagu_marker))


class ReadOnlyTests(unittest.TestCase):
    def test_the_module_only_reads(self):
        """Metaculus through the GET-only client, GitHub through one GET."""
        with io.open(os.path.join(ROOT, "research", "health_check.py"), encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn("MetaculusReadClient", src)
        self.assertNotIn("forecasting_tools", src)
        self.assertNotIn('method="POST"', src)
        self.assertEqual(src.count("urllib.request.Request("), 1)
        self.assertIn('method="GET"', src)


if __name__ == "__main__":
    unittest.main()
