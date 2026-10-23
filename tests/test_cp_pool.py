"""Market Pulse: the pool with the revealed human community prediction.

See cp_pool.py for why. The questions are built by the SDK's own group
unpacking (MetaculusClient._unpack_group_question) from a post in the shape
of group 46006 -- a group post whose subquestions carry ``cp_reveal_time``
and ``aggregations.recency_weighted.latest`` -- so the tests read the CP from
where discovery really puts it. Every number is synthetic: the repository is
public, and no CP or forecast of a real open question may appear here.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import io
import os
import re
import unittest
from datetime import datetime, timedelta, timezone

import cp_pool
import forecast_guards
import numeric_shape
from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from forecasting_tools.data_models.multiple_choice_report import (
        PredictedOption,
        PredictedOptionList,
    )
    from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVEAL = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
AFTER = REVEAL + timedelta(hours=40)
BEFORE = REVEAL - timedelta(hours=1)
URL = "https://www.metaculus.com/questions/900046"
Z = (-1.2816, -0.8416, -0.2533, 0.2533, 0.8416, 1.2816)
LEVELS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
LOG_LINE = re.compile(r"^cp_pool applied=[01] reason=[a-z_]+(?::[A-Za-z]+)? url=\S+$")


def latest(forecast_values, forecaster_count=23, end_time=None):
    """recency_weighted.latest as the posts API returns it (synthetic)."""
    return {
        "start_time": REVEAL.timestamp() - 3600,
        "end_time": end_time,
        "forecast_values": forecast_values,
        "forecaster_count": forecaster_count,
        "interval_lower_bounds": None,
        "centers": None,
        "interval_upper_bounds": None,
        "means": None,
        "histogram": None,
    }


def subquestion(kind, aggregation, reveal=REVEAL, **extra):
    question = {
        "id": 900047,
        "title": "What will the synthetic quantity be for these companies?",
        "label": "Company A",
        "type": kind,
        "status": "open",
        "open_time": "2026-10-20T12:00:00Z",
        "cp_reveal_time": reveal.isoformat() if reveal else None,
        "scheduled_close_time": "2026-11-03T10:00:00Z",
        "scheduled_resolve_time": "2026-11-04T10:00:00Z",
        "include_bots_in_aggregates": False,
        "unit": "$",
        "my_forecasts": {"history": [], "latest": None},
        "aggregations": {
            "recency_weighted": {"history": [], "latest": aggregation},
            "unweighted": {"history": [], "latest": aggregation},
        },
    }
    question.update(extra)
    return question


def group_post(sub):
    return {
        "id": 900046,
        "title": "What will the synthetic quantity be for these companies?",
        "group_of_questions": {
            "description": "Synthetic background.",
            "resolution_criteria": "Synthetic criteria.",
            "fine_print": "",
            "questions": [sub],
        },
    }


def unpack(sub):
    with real_forecasting_tools():
        (question,) = MetaculusClient._unpack_group_question(group_post(copy.deepcopy(sub)))
    return question


def numeric_sub(aggregation, lb=1.0, ub=3.0, open_lower=True, open_upper=True,
                outcomes=200, zero_point=None, **extra):
    scaling = {"range_min": lb, "range_max": ub, "zero_point": zero_point,
               "inbound_outcome_count": outcomes, "nominal_min": None, "nominal_max": None}
    return subquestion("numeric" if outcomes == 200 else "discrete", aggregation,
                       scaling=scaling, open_lower_bound=open_lower,
                       open_upper_bound=open_upper, **extra)


def distribution(question, centre, sigma):
    values = [centre + sigma * z for z in Z]
    return numeric_shape.build(NumericDistribution,
                               [Percentile(percentile=p, value=v) for p, v in zip(LEVELS, values)],
                               question)


def cdf_values(question, centre, sigma):
    """A valid synthetic CP: the grid heights of a standardised SDK CDF."""
    return [p.percentile for p in distribution(question, centre, sigma).get_cdf()]


def numeric_case(centre_cp=2.4, **kwargs):
    """(question, our aggregate, CP heights) on one synthetic subquestion."""
    shape_only = unpack(numeric_sub(None, **kwargs))
    cp = cdf_values(shape_only, centre_cp, 0.1)
    question = unpack(numeric_sub(latest(cp), **kwargs))
    return question, distribution(question, 1.8, 0.1), cp


def heights(d):
    return [p.percentile for p in d.get_cdf()]


def set_latest(question, aggregation):
    question.api_json["question"]["aggregations"]["recency_weighted"]["latest"] = aggregation


class RealSdkTestCase(unittest.TestCase):
    """Each test inside the real-package window: other test modules leave a
    stub forecasting_tools in sys.modules, and the SDK imports lazily inside
    from_question; see tests/_real_forecasting_tools.py."""

    def setUp(self):
        super().setUp()
        window = real_forecasting_tools()
        window.__enter__()
        self.addCleanup(window.__exit__, None, None, None)


class PoolLogMixin:
    def pool(self, aggregate, question, weight=0.6, now=AFTER):
        with self.assertLogs(cp_pool.logger, level="INFO") as logs:
            out = cp_pool.apply(aggregate, question, weight, now)
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertRegex(message, LOG_LINE)
        return out, message


class WeightTests(RealSdkTestCase):
    def test_off_unless_set(self):
        self.assertEqual(cp_pool.weight_from_env({}), 0.0)
        self.assertEqual(cp_pool.weight_from_env({cp_pool.ENV_WEIGHT: ""}), 0.0)
        self.assertEqual(cp_pool.weight_from_env({cp_pool.ENV_WEIGHT: " 0 "}), 0.0)

    def test_a_weight_is_read(self):
        self.assertEqual(cp_pool.weight_from_env({cp_pool.ENV_WEIGHT: "0.6"}), 0.6)
        self.assertEqual(cp_pool.weight_from_env({cp_pool.ENV_WEIGHT: "1"}), 1.0)

    def test_anything_else_is_off_with_a_warning(self):
        for raw in ("abc", "1.5", "-0.2", "nan", "inf"):
            with self.subTest(raw=raw), self.assertLogs(cp_pool.logger, level="WARNING"):
                self.assertEqual(cp_pool.weight_from_env({cp_pool.ENV_WEIGHT: raw}), 0.0)

    def test_weight_zero_is_a_silent_no_op(self):
        question, ours, _cp = numeric_case()
        with self.assertNoLogs(cp_pool.logger):
            self.assertIs(cp_pool.apply(ours, question, 0.0, AFTER), ours)


class NumericPoolTests(PoolLogMixin, RealSdkTestCase):
    def assertValid(self, pooled, question):
        cdf = heights(pooled)
        self.assertEqual(len(cdf), question.cdf_size)
        steps = [b - a for a, b in zip(cdf, cdf[1:])]
        self.assertGreater(min(steps), 0)
        self.assertLess(max(steps), 0.2 * 200 / (question.cdf_size - 1))
        if not question.open_lower_bound:
            self.assertEqual(cdf[0], 0)
        if not question.open_upper_bound:
            self.assertAlmostEqual(cdf[-1], 1, places=9)
        return cdf

    def test_the_sdk_puts_the_cp_where_the_pool_reads_it(self):
        question, _ours, cp = numeric_case()
        self.assertEqual(question.group_question_option, "Company A")
        self.assertEqual(question.cp_reveal_time, REVEAL)
        self.assertEqual(cp_pool.latest_cp(question, AFTER)["forecast_values"], cp)

    def test_the_cdfs_are_mixed_at_the_weight(self):
        question, ours, cp = numeric_case()
        pooled, message = self.pool(ours, question, 0.6)
        self.assertEqual(message, f"cp_pool applied=1 reason=ok url={URL}")
        declared = [p.percentile for p in pooled.declared_percentiles]
        for got, c, o in zip(declared, cp, heights(ours)):
            self.assertAlmostEqual(got, 0.6 * c + 0.4 * o, places=9)
        self.assertValid(pooled, question)

    def test_the_pooled_median_lies_between_the_two(self):
        question, ours, _cp = numeric_case()
        pooled, _ = self.pool(ours, question)

        def median(d):
            return d.get_percentiles_at_target_heights([0.5])[0].value

        cp_question = unpack(numeric_sub(None))
        self.assertTrue(median(ours) < median(pooled) < median(distribution(cp_question, 2.4, 0.1)))

    def test_closed_bounds_stay_valid(self):
        question, ours, _cp = numeric_case(open_lower=False, open_upper=False)
        pooled, message = self.pool(ours, question)
        self.assertIn("applied=1", message)
        self.assertValid(pooled, question)

    def test_a_log_scaled_question_stays_valid(self):
        kwargs = dict(lb=10.0, ub=10_000.0, zero_point=0.0, open_lower=False, open_upper=True)
        shape_only = unpack(numeric_sub(None, **kwargs))
        self.assertEqual(shape_only.zero_point, 0.0)
        cp = cdf_values(shape_only, 800, 150)
        question = unpack(numeric_sub(latest(cp), **kwargs))
        pooled, message = self.pool(distribution(question, 300, 60), question)
        self.assertIn("applied=1", message)
        self.assertValid(pooled, question)

    def test_a_discrete_question_stays_valid(self):
        kwargs = dict(lb=-0.5, ub=20.5, outcomes=21, open_lower=False, open_upper=True)
        shape_only = unpack(numeric_sub(None, **kwargs))
        cp = cdf_values(shape_only, 12, 2)
        question = unpack(numeric_sub(latest(cp), **kwargs))
        self.assertEqual(question.cdf_size, 22)
        pooled, message = self.pool(distribution(question, 8, 2), question)
        self.assertIn("applied=1", message)
        self.assertValid(pooled, question)

    def test_a_date_question_stays_valid(self):
        start = datetime(2027, 1, 1, tzinfo=timezone.utc)
        end = datetime(2028, 1, 1, tzinfo=timezone.utc)
        sub = subquestion("date", None, scaling={
            "range_min": start.timestamp(), "range_max": end.timestamp(), "zero_point": None,
            "inbound_outcome_count": 200}, open_lower_bound=False, open_upper_bound=True)
        shape_only = unpack(sub)
        day = 86400.0
        cp = cdf_values(shape_only, start.timestamp() + 200 * day, 20 * day)
        sub["aggregations"]["recency_weighted"]["latest"] = latest(cp)
        question = unpack(sub)
        pooled, message = self.pool(distribution(question, start.timestamp() + 150 * day, 30 * day),
                                    question)
        self.assertIn("applied=1", message)
        self.assertTrue(pooled.is_date)
        self.assertValid(pooled, question)

    def test_a_cp_on_another_grid_is_ignored(self):
        question, ours, cp = numeric_case()
        set_latest(question, latest(cp[::2]))
        out, message = self.pool(ours, question)
        self.assertIs(out, ours)
        self.assertEqual(message, f"cp_pool applied=0 reason=shape url={URL}")

    def test_a_decreasing_cp_is_ignored(self):
        question, ours, cp = numeric_case()
        set_latest(question, latest(cp[::-1]))
        out, message = self.pool(ours, question)
        self.assertIs(out, ours)
        self.assertIn("reason=shape", message)


class GateTests(PoolLogMixin, RealSdkTestCase):
    def test_nothing_before_the_reveal(self):
        question, ours, _cp = numeric_case()
        out, message = self.pool(ours, question, now=BEFORE)
        self.assertIs(out, ours)
        self.assertEqual(message, f"cp_pool applied=0 reason=not_revealed url={URL}")

    def test_nothing_without_a_reveal_time(self):
        question, ours, _cp = numeric_case()
        question.cp_reveal_time = None
        out, message = self.pool(ours, question)
        self.assertIs(out, ours)
        self.assertIn("reason=not_revealed", message)

    def test_nothing_without_a_cp(self):
        """Before the reveal, or listed without with_cp, there is no latest."""
        _q, ours, _cp = numeric_case()
        for aggregations in (None, {}, {"recency_weighted": {"latest": None}},
                             {"recency_weighted": {"latest": latest([])}}):
            question = unpack(numeric_sub(None))
            question.api_json["question"]["aggregations"] = aggregations
            with self.subTest(aggregations=aggregations):
                out, message = self.pool(ours, question)
                self.assertIs(out, ours)
                self.assertIn("reason=no_cp", message)

    def test_nothing_with_fewer_than_ten_forecasters(self):
        question, ours, cp = numeric_case()
        for count in (9, None, "23"):
            set_latest(question, latest(cp, forecaster_count=count))
            with self.subTest(count=count):
                out, message = self.pool(ours, question)
                self.assertIs(out, ours)
                self.assertIn("reason=few_forecasters", message)
        set_latest(question, latest(cp, forecaster_count=cp_pool.MIN_FORECASTERS))
        _out, message = self.pool(ours, question)
        self.assertIn("applied=1", message)

    def test_nothing_from_an_aggregation_that_has_ended(self):
        question, ours, cp = numeric_case()
        set_latest(question, latest(cp, end_time=AFTER.timestamp() - 60))
        out, message = self.pool(ours, question)
        self.assertIs(out, ours)
        self.assertIn("reason=stale", message)

    def test_any_error_keeps_our_forecast(self):
        question, _ours, _cp = numeric_case()

        class Broken:
            def get_cdf(self):
                raise RuntimeError("synthetic")

        broken = Broken()
        with self.assertLogs(cp_pool.logger, level="WARNING") as logs:
            self.assertIs(cp_pool.apply(broken, question, 0.6, AFTER), broken)
        self.assertEqual(logs.records[0].getMessage(),
                         f"cp_pool applied=0 reason=error:RuntimeError url={URL}")


class BinaryAndMultipleChoiceTests(PoolLogMixin, RealSdkTestCase):
    def test_binary_mixes_the_probability_then_clips(self):
        question = unpack(subquestion("binary", latest([0.7, 0.3])))
        pooled, message = self.pool(0.8, question, 0.6)
        self.assertAlmostEqual(pooled, 0.6 * 0.3 + 0.4 * 0.8)
        self.assertIn("applied=1", message)
        question = unpack(subquestion("binary", latest([1.0, 0.0])))
        pooled, _ = self.pool(0.04, question, 1.0)
        self.assertEqual(pooled, forecast_guards.BINARY_MIN)

    def test_binary_needs_two_values(self):
        question = unpack(subquestion("binary", latest([0.3])))
        out, message = self.pool(0.8, question)
        self.assertEqual(out, 0.8)
        self.assertIn("reason=shape", message)

    def test_binary_falls_back_to_the_centre(self):
        """The field the SDK itself reads for a binary CP."""
        aggregation = latest(None)
        aggregation["centers"] = [0.3]
        question = unpack(subquestion("binary", aggregation))
        self.assertEqual(question.community_prediction_at_access_time, 0.3)
        pooled, message = self.pool(0.8, question, 0.5)
        self.assertAlmostEqual(pooled, 0.55)
        self.assertIn("applied=1", message)

    def mc(self, cp_values, options=("Low", "Mid", "High")):
        return unpack(subquestion("multiple_choice", latest(list(cp_values)),
                                  options=list(options), group_variable=""))

    def ours(self, **probabilities):
        with real_forecasting_tools():
            return PredictedOptionList(predicted_options=[
                PredictedOption(option_name=n, probability=p) for n, p in probabilities.items()])

    def test_options_are_mixed_by_name_then_floored(self):
        question = self.mc([0.5, 0.3, 0.2])
        ours = self.ours(High=0.9, Low=0.05, Mid=0.05)  # a different order
        with real_forecasting_tools():
            pooled, message = self.pool(ours, question, 0.5)
        self.assertIn("applied=1", message)
        self.assertIsInstance(pooled, PredictedOptionList)
        got = pooled.to_dict()
        expected = forecast_guards.floor_options([0.5 * 0.2 + 0.5 * 0.9, 0.5 * 0.5 + 0.5 * 0.05,
                                                  0.5 * 0.3 + 0.5 * 0.05])
        for name, p in zip(("High", "Low", "Mid"), expected):
            self.assertAlmostEqual(got[name], p)
        self.assertAlmostEqual(sum(got.values()), 1.0)

    def test_mismatched_options_are_ignored(self):
        question = self.mc([0.5, 0.5], options=("Low", "High"))
        ours = self.ours(Low=0.3, Mid=0.3, High=0.4)
        out, message = self.pool(ours, question)
        self.assertIs(out, ours)
        self.assertIn("reason=shape", message)


# ------------------------------------------------------------------- main.py


def main_src() -> str:
    with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
        return handle.read()


class MainWiringTests(RealSdkTestCase):
    def test_off_by_default_and_set_only_in_market_pulse(self):
        src = main_src()
        self.assertIn("    cp_pool_weight = 0.0\n", src)
        self.assertEqual(src.count("cp_pool_weight = cp_pool.weight_from_env()"), 1)
        start = src.index('elif run_mode == "market_pulse":')
        end = src.index('elif run_mode == "test_questions":')
        self.assertIn("template_bot.cp_pool_weight = cp_pool.weight_from_env()", src[start:end])

    def test_the_pool_runs_after_the_aggregate_and_only_with_a_weight(self):
        src = main_src()
        body = src[src.index("async def _aggregate_predictions"):src.index("def _create_comment")]
        self.assertLess(body.index("numeric_shape.mean_of_cdfs"), body.index("cp_pool.apply("))
        self.assertLess(body.index("super()._aggregate_predictions"), body.index("cp_pool.apply("))
        self.assertIn("if self.cp_pool_weight > 0:", body)

    def test_the_bot_pools_a_revealed_numeric_question(self):
        with real_forecasting_tools():
            path = os.path.join(ROOT, "main.py")
            spec = importlib.util.spec_from_file_location("main_cp_pool", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        bot = object.__new__(module.SummerTemplateBot2026)
        question, ours, _cp = numeric_case()
        question.cp_reveal_time = datetime.now(timezone.utc) - timedelta(hours=1)
        preds = [ours, distribution(question, 1.9, 0.1), distribution(question, 1.7, 0.1)]
        with real_forecasting_tools():
            plain = asyncio.run(bot._aggregate_predictions(preds, question))
            bot.cp_pool_weight = 0.6
            with self.assertLogs(cp_pool.logger, level="INFO") as logs:
                pooled = asyncio.run(bot._aggregate_predictions(preds, question))
        self.assertEqual(heights(plain), heights(numeric_shape.mean_of_cdfs(preds, question)))
        self.assertNotEqual(heights(pooled), heights(plain))
        self.assertEqual([r.getMessage() for r in logs.records],
                         [f"cp_pool applied=1 reason=ok url={URL}"])


if __name__ == "__main__":
    unittest.main()
