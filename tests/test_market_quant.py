"""GAAP diluted EPS against the consensus: the eps_gaap Live data section.

Every figure here is synthetic (round numbers on a company with no open
Market Pulse question): the repository is public, and real consensus or gap
figures for an open question would publish the bot's prior. A fake fetcher
stands in for Yahoo, shaped like the responses read on 2026-10-23
(fundamentals-timeseries, quoteSummary earningsTrend + earningsHistory).
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import sys
import unittest
import urllib.error
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

import market_quant as mq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
CRITERIA = ("This question resolves to the diluted earnings per share reported for the quarter, "
            "calculated in conformity with US GAAP, as first published by the company.")


def eps_question(option="Intel", lower=0.0, upper=2.0, close=datetime(2026, 10, 30, 10, tzinfo=timezone.utc)):
    return NS(question_text="What will Q3 2026 EPS be for these companies?", group_question_option=option,
              resolution_criteria=CRITERIA, fine_print="", background_info="",
              lower_bound=lower, upper_bound=upper, close_time=close)


def timeseries(rows):
    return {"timeseries": {"result": [{
        "meta": {"symbol": ["INTC"], "type": ["quarterlyDilutedEPS"]},
        "timestamp": [1] * len(rows),
        "quarterlyDilutedEPS": [
            {"dataId": 1, "asOfDate": day, "periodType": "3M", "currencyCode": "USD",
             "reportedValue": {"raw": value, "fmt": f"{value:.2f}"}} for day, value in rows]}],
        "error": None}}


def summary(estimates, avg=1.0, end="2026-09-30", analysts=10):
    return {"quoteSummary": {"result": [{
        "earningsTrend": {"trend": [
            {"period": "0q", "endDate": end, "earningsEstimate": {
                "avg": {"raw": avg, "fmt": f"{avg:.2f}"}, "low": {"raw": avg - 0.1, "fmt": ""},
                "high": {"raw": avg + 0.1, "fmt": ""}, "numberOfAnalysts": {"raw": analysts, "fmt": ""}}},
            {"period": "+1q", "endDate": "2026-12-31", "earningsEstimate": {"avg": {"raw": 9.0, "fmt": ""}}}]},
        "earningsHistory": {"history": [
            {"quarter": {"raw": 0, "fmt": day}, "epsEstimate": {"raw": est, "fmt": ""},
             # the street "actual": never what the gap is measured from
             "epsActual": {"raw": est + 0.05, "fmt": ""}} for day, est in estimates]}}],
        "error": None}}


#: Synthetic history: GAAP sits exactly 0.30 under the consensus every quarter.
GAAP = [("2025-06-30", 0.45), ("2025-09-30", 0.50), ("2025-12-31", 0.55),
        ("2026-03-31", 0.60), ("2026-06-30", 0.65)]
ESTIMATES = [("2025-09-30", 0.80), ("2025-12-31", 0.85), ("2026-03-31", 0.90), ("2026-06-30", 0.95)]


class FakeYahoo:
    def __init__(self, ts=None, qs=None, crumb=b"synthCrumb01", fail=()):
        self.routes = {
            mq.COOKIE_URL: b"",
            mq.CRUMB_URL: crumb,
            mq.TIMESERIES_URL: ts if ts is not None else timeseries(GAAP),
            mq.SUMMARY_URL: qs if qs is not None else summary(ESTIMATES),
        }
        self.fail, self.calls = set(fail), []

    def __call__(self, url, headers):
        self.calls.append(url)
        for prefix, body in self.routes.items():
            if url.startswith(prefix):
                if prefix in self.fail:
                    raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
                return body if isinstance(body, bytes) else json.dumps(body).encode()
        raise OSError("no route: " + url)


class WhichQuestionTests(unittest.TestCase):
    def test_gaap_eps_questions_only(self):
        self.assertEqual(mq.kind(eps_question()), "eps_gaap")
        self.assertIsNone(mq.kind(NS(question_text="What will Intel's revenue be?",
                                     resolution_criteria="Total revenue under US GAAP.", fine_print="")))
        self.assertIsNone(mq.kind(NS(question_text="What will Intel's EPS be?",
                                     resolution_criteria="Adjusted (non-GAAP) diluted EPS.", fine_print="")))

    def test_company_from_the_group_option(self):
        self.assertEqual(mq.ticker(eps_question("Intel")), "INTC")
        self.assertEqual(mq.ticker(eps_question("Meta Platforms (META)")), "META")
        self.assertEqual(mq.ticker(eps_question("Advanced Micro Devices")), "AMD")

    def test_no_company_or_several_is_none(self):
        self.assertIsNone(mq.ticker(eps_question("SpaceX")))
        self.assertIsNone(mq.ticker(NS(question_text="Will Apple or Microsoft report higher EPS?",
                                       group_question_option="")))
        self.assertIsNone(mq.ticker(NS(question_text="What will diluted EPS (GAAP) be?",
                                       group_question_option="")))


class GapTests(unittest.TestCase):
    def test_gaps_pair_gaap_with_the_estimate_of_the_time(self):
        history = mq.gaps([(mq._day(d), v) for d, v in GAAP], [(mq._day(d), e) for d, e in ESTIMATES])
        self.assertEqual([str(d) for d, _ in history], [d for d, _ in ESTIMATES])
        for _, gap in history:
            self.assertAlmostEqual(gap, -0.30)

    def test_quarters_a_few_days_apart_are_the_same_quarter(self):
        history = mq.gaps([(mq._day("2026-06-27"), 1.0)], [(mq._day("2026-06-30"), 1.2)])
        self.assertAlmostEqual(history[0][1], -0.2)

    def test_stability_rule(self):
        self.assertTrue(mq.stable([-0.30, -0.32, -0.28, -0.31]))
        self.assertTrue(mq.stable([-0.30, -0.32, 0.05, -0.31]))       # 3 of 4 the same sign
        self.assertFalse(mq.stable([-0.30, 0.10, 0.05, -0.31]))       # 2 of 4
        self.assertTrue(mq.stable([-0.30] * 6 + [0.30] * 2))          # 6 of 8, no spread
        self.assertFalse(mq.stable([-0.30] * 5 + [0.30] * 3))         # 5 of 8, no spread
        self.assertFalse(mq.stable([0.20, 0.60, 0.35, 0.90]))         # same sign, wide spread
        self.assertFalse(mq.stable([-0.30, -0.30, -0.30]))            # too few quarters
        self.assertFalse(mq.stable([0.0, 0.0, 0.0, 0.0]))

    def test_prior_is_centred_on_consensus_plus_gap_with_a_floor(self):
        p10, p50, p90 = mq.prior(1.0, [-0.30] * 4)
        self.assertAlmostEqual(p50, 0.70, places=6)
        self.assertLess(p10, p50)
        self.assertLess(p50, p90)
        # a history with no spread still leaves the 3% floor's width
        self.assertGreater(p90 - p10, 2 * 1.5 * 0.03)
        self.assertAlmostEqual(p50 - p10, p90 - p50, places=6)

    def test_t_cdf_matches_tables(self):
        self.assertAlmostEqual(mq._t_cdf(1.5332, 4), 0.90, places=4)
        self.assertAlmostEqual(mq._t_cdf(1.8856, 2), 0.90, places=4)
        self.assertAlmostEqual(mq._t_cdf(0.0, 4), 0.5)


class SectionTests(unittest.TestCase):
    def test_stable_gap_gets_consensus_history_and_prior(self):
        web = FakeYahoo()
        out = mq.section(eps_question(), web, NOW)
        self.assertIn("Yahoo symbol INTC", out)
        self.assertIn("Consensus EPS for the quarter ending 2026-09-30: 1.000 "
                      "(10 analysts; low 0.90, high 1.10)", out)
        self.assertIn("2025-09-30 -0.30", out)
        self.assertIn("median -0.30", out)
        self.assertIn("same sign in 4 of 4 quarters", out)
        self.assertIn("P50 0.70", out)

    def test_requests_timeseries_without_crumb_and_summary_with_it(self):
        web = FakeYahoo()
        mq.section(eps_question(), web, NOW)
        ts = [u for u in web.calls if u.startswith(mq.TIMESERIES_URL)][0]
        qs = [u for u in web.calls if u.startswith(mq.SUMMARY_URL)][0]
        self.assertIn("type=quarterlyDilutedEPS", ts)
        self.assertNotIn("crumb", ts)
        self.assertIn("crumb=synthCrumb01", qs)
        self.assertIn("earningsTrend", qs)
        self.assertLess(web.calls.index(mq.COOKIE_URL), web.calls.index(mq.CRUMB_URL))

    def test_unstable_gap_gets_no_prior(self):
        estimates = [("2025-09-30", 0.20), ("2025-12-31", 0.85), ("2026-03-31", 0.30), ("2026-06-30", 1.20)]
        out = mq.section(eps_question(), FakeYahoo(qs=summary(estimates)), NOW)
        self.assertIn("has not been stable", out)
        self.assertNotIn("P10", out)
        self.assertIn("Consensus EPS", out)

    def test_refused_crumb_leaves_the_gaap_history(self):
        for web in (FakeYahoo(fail={mq.CRUMB_URL}), FakeYahoo(crumb=b"<html>blocked</html>")):
            out = mq.section(eps_question(), web, NOW)
            self.assertIn("GAAP diluted EPS, by quarter end: 2025-06-30 0.45", out)
            self.assertNotIn("Consensus", out)
            self.assertNotIn("P10", out)

    def test_missing_gaap_history_keeps_the_consensus_but_no_prior(self):
        out = mq.section(eps_question(), FakeYahoo(fail={mq.TIMESERIES_URL}), NOW)
        self.assertIn("Consensus EPS", out)
        self.assertNotIn("P10", out)

    def test_consensus_for_an_already_reported_quarter_is_dropped(self):
        out = mq.section(eps_question(), FakeYahoo(qs=summary(ESTIMATES, end="2026-06-30")), NOW)
        self.assertNotIn("Consensus EPS", out)
        self.assertNotIn("P10", out)
        self.assertIn("has been stable", out)

    def test_consensus_for_a_quarter_ending_after_close_is_dropped(self):
        question = eps_question(close=datetime(2026, 9, 1, tzinfo=timezone.utc))
        self.assertNotIn("Consensus EPS", mq.section(question, FakeYahoo(), NOW))

    def test_consensus_far_from_the_range_means_the_wrong_company(self):
        self.assertEqual(mq.section(eps_question(lower=10.0, upper=12.0), FakeYahoo(), NOW), "")

    def test_other_questions_fetch_nothing(self):
        web = FakeYahoo()
        self.assertEqual(mq.section(NS(question_text="Will it rain in Paris?", resolution_criteria="",
                                       fine_print=""), web, NOW), "")
        self.assertEqual(mq.section(eps_question("SpaceX"), web, NOW), "")
        self.assertEqual(web.calls, [])

    def test_never_raises(self):
        def explode(url, headers):
            raise RuntimeError("down")
        self.assertEqual(mq.section(eps_question(), explode, NOW), "")
        self.assertEqual(mq.section(object(), FakeYahoo(), NOW), "")
        garbage = FakeYahoo(ts={"timeseries": None}, qs={"quoteSummary": {"result": "x"}})
        self.assertEqual(mq.section(eps_question(), garbage, NOW), "")

    def test_logs_carry_flags_and_counts_only(self):
        with self.assertLogs("market_quant", level="INFO") as logs:
            mq.section(eps_question(), FakeYahoo(), NOW)
        text = "\n".join(logs.output)
        self.assertIn("stable=1 prior=1", text)
        for figure in ("1.000", "0.70", "0.30", "0.45", "INTC", "P10"):
            self.assertNotIn(figure, text)


class SessionTests(unittest.TestCase):
    def test_cookie_page_404_is_not_an_error_elsewhere_it_is(self):
        def not_found(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        opener = mock.Mock()
        opener.open.side_effect = not_found
        with mock.patch("urllib.request.build_opener", return_value=opener):
            fetch = mq._session()
        self.assertEqual(fetch(mq.COOKIE_URL, {}), b"")
        with self.assertRaises(urllib.error.HTTPError):
            fetch(mq.CRUMB_URL, {})

    def test_each_session_has_its_own_cookie_jar(self):
        with mock.patch("urllib.request.build_opener") as build:
            mq._session()
            mq._session()
        jars = [call.args[0].cookiejar for call in build.call_args_list]
        self.assertIsNot(jars[0], jars[1])

    def test_standard_library_only(self):
        with io.open(os.path.join(ROOT, "market_quant.py"), encoding="utf-8") as h:
            tree = ast.parse(h.read())
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue(names <= set(sys.stdlib_module_names) | {"__future__"}, names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
