"""Numeric forecasts: synthesized tails, the out-of-range rescue, the mean of CDFs.

See numeric_shape.py for why. Every case is checked against the REAL
forecasting-tools validators, because what the SDK accepts is the property
that matters: a shape it rejects loses the forecast. All numbers are
synthetic (the repository is public); the out-of-range case mirrors the shape
of the 2026-10-20 Meta failure on made-up values.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import math
import os
import random
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import numeric_shape as ns
from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from forecasting_tools.data_models.numeric_report import (
        NumericDistribution,
        NumericReport,
        Percentile,
    )
    from forecasting_tools.data_models.questions import (
        BinaryQuestion,
        DateQuestion,
        DiscreteQuestion,
        NumericQuestion,
    )

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://www.metaculus.com/questions/1"
LEVELS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
#: Standard normal quantiles at LEVELS.
Z = (-1.2816, -0.8416, -0.2533, 0.2533, 0.8416, 1.2816)


def numeric(lb, ub, open_lower=True, open_upper=True, cdf_size=201, zero_point=None):
    cls = NumericQuestion if cdf_size == 201 else DiscreteQuestion
    return cls(question_text="synthetic", id_of_post=1, id_of_question=2,
               page_url=URL,
               lower_bound=lb, upper_bound=ub, open_lower_bound=open_lower,
               open_upper_bound=open_upper, unit_of_measure="", zero_point=zero_point,
               cdf_size=cdf_size)


def below_range_question():
    """An open 10-14 range: the SDK's 25% window ends at 9."""
    return numeric(10.0, 14.0)


def percentiles(values, levels=LEVELS):
    return [Percentile(percentile=p, value=v) for p, v in zip(levels, values)]


def normal_set(centre, sigma, skew=1.0):
    return [centre + sigma * z * (skew if z > 0 else 1.0) for z in Z]


def synthetic_sets(seed=20261023, count=10):
    """Ten five-forecaster sets: different centres, spreads and skews, with
    the forecasters disagreeing about both centre and width."""
    rnd = random.Random(seed)
    sets = []
    for _ in range(count):
        centre = rnd.uniform(-50, 200)
        sigma = abs(centre) * rnd.uniform(0.02, 0.3) + 0.5
        skew = rnd.uniform(0.6, 1.6)
        sets.append([normal_set(centre + rnd.gauss(0, sigma * 0.3),
                                sigma * rnd.uniform(0.5, 2.0), skew) for _ in range(5)])
    return sets


def build(values, question):
    return ns.build(NumericDistribution, percentiles(values), question)


def heights(distribution):
    return [p.percentile for p in distribution.get_cdf()]


def cdf_at(distribution, x):
    cdf = distribution.get_cdf()
    for a, b in zip(cdf, cdf[1:]):
        if a.value <= x <= b.value:
            share = (x - a.value) / (b.value - a.value)
            return a.percentile + (b.percentile - a.percentile) * share
    raise ValueError(x)


class RealSdkTestCase(unittest.TestCase):
    """Each test inside the real-package window: other test modules leave a
    stub forecasting_tools in sys.modules, and the SDK imports lazily inside
    from_question; see tests/_real_forecasting_tools.py."""

    def setUp(self):
        super().setUp()
        window = real_forecasting_tools()
        window.__enter__()
        self.addCleanup(window.__exit__, None, None, None)


class ValidCdfMixin:
    def assertValidCdf(self, distribution, question):
        cdf = heights(distribution)
        self.assertEqual(len(cdf), question.cdf_size)
        steps = [b - a for a, b in zip(cdf, cdf[1:])]
        self.assertGreater(min(steps), 0)
        self.assertLessEqual(max(steps), 0.2 * 200 / (question.cdf_size - 1) + 1e-9)
        if not question.open_lower_bound:
            self.assertEqual(cdf[0], 0)
        if not question.open_upper_bound:
            self.assertAlmostEqual(cdf[-1], 1, places=9)


class SdkValidationTests(ValidCdfMixin, RealSdkTestCase):
    """The prototype's 50 cases (10 sets x 5 bound settings), plus log-scaled
    and date questions: every forecaster and the aggregate must pass the
    SDK's validators, closed and open bounds, discrete and continuous."""

    SETTINGS = [  # open lower, open upper, cdf size, padding in spans
        (True, True, 201, 1.0),
        (False, False, 201, 0.2),
        (False, True, 201, 0.05),
        (True, True, 21, 0.5),
        (False, False, 201, 0.0),
    ]

    def test_the_fifty_cases_validate(self):
        cases = 0
        for rows in synthetic_sets():
            lo, hi = min(r[0] for r in rows), max(r[-1] for r in rows)
            span = hi - lo
            for open_lower, open_upper, size, pad in self.SETTINGS:
                with self.subTest(case=cases):
                    q = numeric(lo - pad * span, hi + pad * span, open_lower, open_upper, size)
                    preds = [build(r, q) for r in rows]
                    for p in preds:
                        self.assertValidCdf(p, q)
                    self.assertValidCdf(ns.mean_of_cdfs(preds, q), q)
                cases += 1
        self.assertEqual(cases, 50)

    LOG_SETTINGS = [  # open lower, open upper, range as multiples of the forecasts' span
        (True, True, 0.5, 2.0),
        (False, False, 0.9, 1.1),
        (False, True, 0.2, 1.01),
        (True, True, 1.0, 1.0),
    ]

    def test_log_scaled_questions_validate(self):
        rnd = random.Random(7)
        for case in range(5):
            centre = math.exp(rnd.uniform(0, 12))
            rows = []
            for _ in range(5):
                shift, sigma = rnd.gauss(0, 0.2), rnd.uniform(0.1, 1.0)
                rows.append([centre * math.exp(shift + sigma * z) for z in Z])
            lo, hi = min(r[0] for r in rows), max(r[-1] for r in rows)
            for open_lower, open_upper, f_lo, f_hi in self.LOG_SETTINGS:
                with self.subTest(case=case, open_lower=open_lower, open_upper=open_upper):
                    q = numeric(lo * f_lo, hi * f_hi, open_lower, open_upper, zero_point=0.0)
                    preds = [build(r, q) for r in rows]
                    self.assertValidCdf(ns.mean_of_cdfs(preds, q), q)

    def test_date_questions_validate(self):
        start = datetime(2027, 1, 1, tzinfo=timezone.utc)
        rnd = random.Random(11)
        for open_lower, open_upper in [(True, True), (False, False), (False, True)]:
            q = DateQuestion(question_text="synthetic", id_of_post=1, id_of_question=2,
                             lower_bound=start, upper_bound=start + timedelta(days=730),
                             open_lower_bound=open_lower, open_upper_bound=open_upper)
            preds = []
            for _ in range(5):
                centre, sigma = rnd.uniform(100, 600), rnd.uniform(5, 60)
                values = [(start + timedelta(days=centre + sigma * z)).timestamp() for z in Z]
                preds.append(ns.build(NumericDistribution, percentiles(values), q))
            with self.subTest(open_lower=open_lower, open_upper=open_upper):
                if open_lower and open_upper:
                    self.assertTrue(all(len(p.declared_percentiles) > 6 for p in preds))
                self.assertValidCdf(ns.mean_of_cdfs(preds, q), q)


class TailTests(RealSdkTestCase):
    VALUES = normal_set(100, 10)

    def shaped(self, question, values=None):
        out, moved = ns.shape(percentiles(values or self.VALUES), question)
        self.assertFalse(moved)
        return {round(p.percentile, 4): p.value for p in out}

    def test_p2_5_and_p97_5_follow_a_t3_fit_to_each_side(self):
        out = self.shaped(numeric(0, 300, False, False))
        p50 = (out[0.4] + out[0.6]) / 2
        self.assertAlmostEqual(out[0.975], p50 + ns.TAIL_C * (out[0.9] - p50))
        self.assertAlmostEqual(out[0.025], p50 - ns.TAIL_C * (p50 - out[0.1]))

    def test_the_declared_six_are_kept_as_written(self):
        out = self.shaped(numeric(0, 300))
        self.assertEqual([out[p] for p in LEVELS], self.VALUES)

    def test_a_log_scaled_question_fits_the_tails_in_log_space(self):
        values = [100 * math.exp(0.3 * z) for z in Z]
        out = self.shaped(numeric(10, 1000, False, False, zero_point=0.0), values)
        log50 = (math.log(out[0.4]) + math.log(out[0.6])) / 2
        self.assertAlmostEqual(math.log(out[0.975]) - log50,
                               ns.TAIL_C * (math.log(out[0.9]) - log50))
        self.assertGreater(out[0.025], 0)

    def test_an_open_bound_tail_stays_inside_the_sdk_limit(self):
        out = self.shaped(numeric(99, 101), [96.0, 99.7, 99.9, 100.1, 100.3, 103.0])
        self.assertLessEqual(out[0.975], 101 + ns.OPEN_TAIL_LIMIT * 2)
        self.assertGreaterEqual(out[0.025], 99 - ns.OPEN_TAIL_LIMIT * 2)

    def test_a_closed_bound_tail_is_used_only_strictly_inside(self):
        far = self.shaped(numeric(0, 300, False, False))
        self.assertTrue(0 < far[0.025] < far[0.1])
        self.assertTrue(far[0.9] < far[0.975] < 300)
        near = self.shaped(numeric(75, 125, False, False))
        self.assertNotIn(0.025, near)
        self.assertNotIn(0.975, near)

    def test_other_declared_levels_are_left_alone(self):
        levels = (0.05, 0.25, 0.5, 0.75, 0.95)
        out, moved = ns.shape(percentiles([80, 93, 100, 107, 120], levels), numeric(0, 300))
        self.assertEqual([p.percentile for p in out], list(levels))
        self.assertFalse(moved)

    def test_the_density_past_p90_is_no_longer_a_cliff(self):
        """Mean density in the p80-p90 width just past p90, against inside
        p80-p90: at least a fifth with the tails; the SDK alone falls far
        below that when the bound is far away."""
        values = self.VALUES
        width = values[5] - values[4]

        def ratio(distribution):
            inner = cdf_at(distribution, values[5]) - cdf_at(distribution, values[4])
            outer = cdf_at(distribution, values[5] + width) - cdf_at(distribution, values[5])
            return outer / inner

        for q in (numeric(0, 300), numeric(0, 300, False, False), numeric(50, 200),
                  numeric(70, 130), numeric(0, 140, False, False)):
            with self.subTest(lb=q.lower_bound, ub=q.upper_bound, closed=not q.open_upper_bound):
                self.assertGreaterEqual(ratio(build(values, q)), 0.2)
        sdk = NumericDistribution.from_question(percentiles(values), numeric(0, 300))
        self.assertLess(ratio(sdk), 0.1)


class BoundMassTests(RealSdkTestCase):
    """The tails past an open bound never leave less mass beyond it than the
    SDK's own construction on the six declared percentiles, and the density
    never rises towards a bound (2026-10-23 review: a bare p97.5 inside an
    open range cut the mass past the bound from 5% to 1.3% and put a bump
    against it; one just inside a closed bound piled 2.5% against it)."""

    def beyond(self, distribution):
        h = heights(distribution)
        return h[0], 1 - h[-1]

    def cases(self):
        rnd = random.Random(20261024)
        for _ in range(60):
            centre, sigma = rnd.uniform(1.2, 2.8), rnd.uniform(0.02, 0.6)
            yield numeric(1.0, 3.0), normal_set(centre, sigma, rnd.uniform(0.6, 1.6))
        for vals in ([1.85, 1.90, 1.97, 2.03, 2.10, 2.15], [1.70, 1.80, 1.94, 2.06, 2.20, 2.30],
                     [1.40, 1.60, 1.88, 2.12, 2.40, 2.60], [1.5, 1.7, 1.95, 2.05, 2.3, 2.5]):
            yield numeric(1.0, 3.0), vals
        for _ in range(10):
            centre, sigma = math.exp(rnd.uniform(2, 4)), rnd.uniform(0.05, 0.5)
            yield (numeric(5.0, 200.0, zero_point=0.0),
                   [centre * math.exp(sigma * z) for z in Z])

    def test_an_open_bound_keeps_at_least_the_sdks_mass_past_it(self):
        for i, (q, vals) in enumerate(self.cases()):
            with self.subTest(case=i):
                old = NumericDistribution.from_question(percentiles(vals), q)
                new = build(vals, q)
                (old_low, old_high), (new_low, new_high) = self.beyond(old), self.beyond(new)
                self.assertGreaterEqual(new_low, old_low - 1e-9)
                self.assertGreaterEqual(new_high, old_high - 1e-9)

    def test_a_date_question_keeps_its_mass_after_the_upper_bound(self):
        start = datetime(2026, 11, 1, tzinfo=timezone.utc)
        q = DateQuestion(question_text="synthetic", id_of_post=1, id_of_question=2,
                         lower_bound=start, upper_bound=start + timedelta(days=425),
                         open_lower_bound=False, open_upper_bound=True)
        centre = start + timedelta(days=120)
        vals = [(centre + timedelta(days=d)).timestamp() for d in (-60, -30, -5, 10, 40, 80)]
        old = NumericDistribution.from_question(percentiles(vals), q)
        new = ns.build(NumericDistribution, percentiles(vals), q)
        self.assertGreaterEqual(1 - heights(new)[-1], 1 - heights(old)[-1] - 1e-9)

    def assertFallsTowardsTheBound(self, distribution, start, upper=True):
        """Per-bin mass from ``start`` to the bound never rises."""
        cdf = distribution.get_cdf()
        pmf = [(b.value, b.percentile - a.percentile) for a, b in zip(cdf, cdf[1:])]
        tail = [m for x, m in pmf if x > start] if upper else [m for x, m in pmf if x < start][::-1]
        for a, b in zip(tail, tail[1:]):
            self.assertLessEqual(b, a + 1e-9)

    def test_no_bump_against_an_open_bound(self):
        vals = [1.5, 1.7, 1.95, 2.05, 2.3, 2.5]
        prediction = build(vals, numeric(1.0, 3.0))
        self.assertFallsTowardsTheBound(prediction, 2.5 + 0.02)
        self.assertFallsTowardsTheBound(prediction, 1.5 - 0.02, upper=False)

    def test_no_mass_piled_against_a_closed_bound(self):
        q = numeric(0.0, 100.0, False, False)
        for vals in ([40, 50, 58, 64, 72, 80], [50, 56, 61, 65, 75, 80]):
            with self.subTest(vals=vals):
                out, _moved = ns.shape(percentiles(vals), q)
                self.assertNotIn(0.975, [round(p.percentile, 4) for p in out])
                self.assertFallsTowardsTheBound(build(vals, q), 80 + 0.5)

    def test_the_tail_inside_an_open_range_is_a_rescaled_t3(self):
        out, _moved = ns.shape(percentiles(normal_set(100, 10)), numeric(0, 300))
        top = [(p.percentile, p.value) for p in out if p.percentile > 0.9]
        self.assertEqual(top[-1], (0.95, 300.0))
        self.assertTrue(all(b[0] > a[0] and b[1] > a[1] for a, b in zip(top, top[1:])))
        self.assertTrue(all(v < 300 for _lvl, v in top[:-1]))


class OutOfRangeTests(RealSdkTestCase):
    """Same shape as Meta on 2026-10-20, synthetic numbers: a forecaster
    wholly below an open range, past the SDK's 25% window."""

    BELOW = [5.0, 5.6, 6.6, 7.6, 8.4, 8.8]
    INSIDE = [[10.8, 11.1, 11.5, 11.8, 12.2, 12.5], [11.0, 11.3, 11.6, 11.9, 12.3, 12.6]]

    def test_the_sdk_rejects_the_set_as_written(self):
        with self.assertRaises(ValueError):
            NumericDistribution.from_question(percentiles(self.BELOW), below_range_question())

    def test_it_is_now_accepted_with_its_mass_below_the_range(self):
        out, moved = ns.shape(percentiles(self.BELOW), below_range_question())
        self.assertTrue(moved)
        values = [p.value for p in out]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        declared = [p.value for p in out if p.percentile in LEVELS]
        self.assertTrue(all(10.0 - 0.25 * 4 <= v < 10.0 for v in declared))
        prediction = build(self.BELOW, below_range_question())
        self.assertGreater(heights(prediction)[0], 0.9)

    def test_after_aggregation_most_of_the_mass_stays_below(self):
        preds = [build(self.BELOW, below_range_question()) for _ in range(3)]
        preds += [build(r, below_range_question()) for r in self.INSIDE]
        aggregate = ns.mean_of_cdfs(preds, below_range_question())
        self.assertGreater(heights(aggregate)[0], 0.5)

    def test_the_rescue_is_logged_as_a_flag_only(self):
        with self.assertLogs(ns.logger, level="INFO") as logs:
            build(self.BELOW, below_range_question())
        self.assertEqual(logs.output, [
            f"INFO:numeric_shape:numeric_clamped=1 url={URL}"])

    def test_a_set_the_sdk_accepts_is_not_moved(self):
        partly = [8.0, 9.5, 10.5, 11.0, 11.5, 12.0]
        out, moved = ns.shape(percentiles(partly), below_range_question())
        self.assertFalse(moved)
        self.assertEqual([p.value for p in out if p.percentile in LEVELS], partly)

    def test_a_log_scaled_set_above_the_range_is_rescued(self):
        q = numeric(10.0, 100.0, open_lower=False, zero_point=0.0)  # window ends at 122.5
        above = [130.0, 140.0, 155.0, 170.0, 190.0, 200.0]
        with self.assertRaises(ValueError):
            NumericDistribution.from_question(percentiles(above), q)
        out, moved = ns.shape(percentiles(above), q)
        self.assertTrue(moved)
        prediction = build(above, q)
        self.assertLess(heights(prediction)[-1], 0.1)

    def test_a_closed_bound_is_not_rescued(self):
        q = numeric(10.0, 14.0, open_lower=False)
        out, moved = ns.shape(percentiles(self.BELOW), q)
        self.assertFalse(moved)
        with self.assertRaises(ValueError):
            build(self.BELOW, q)

    def test_a_fat_tail_past_twice_the_range_is_brought_inside_it(self):
        """Roadmap A3, "always keep values within +/-1.9R": an open-range set
        with values in range but p10 more than 2 ranges below it was refused
        whole, on the verbatim and the parser path alike."""
        q = numeric(5.345, 7.345)  # R = 2: the SDK refuses values below 1.345
        fat = [1.2, 4.0, 5.6, 6.0, 6.4, 6.8]
        with self.assertRaises(ValueError):
            NumericDistribution.from_question(percentiles(fat), q)
        out, moved = ns.shape(percentiles(fat), q)
        self.assertTrue(moved)
        declared = [p.value for p in out if p.percentile in LEVELS]
        self.assertAlmostEqual(declared[0], 5.345 - ns.OPEN_TAIL_LIMIT * 2)
        self.assertEqual(declared[1:], fat[1:])
        with self.assertLogs(ns.logger, level="INFO") as logs:
            prediction = build(fat, q)
        self.assertIn(f"INFO:numeric_shape:numeric_clamped=1 url={URL}", logs.output)
        self.assertGreater(heights(prediction)[0], 0.15)

    def test_a_far_value_on_either_side_keeps_the_order(self):
        q = numeric(10.0, 14.0)  # 2x limit: 2 and 22
        for values in ([0.5, 1.5, 11.0, 12.0, 13.0, 14.0], [10.5, 11.0, 12.0, 13.0, 23.0, 30.0]):
            with self.subTest(values=values):
                out, moved = ns.shape(percentiles(values), q)
                self.assertTrue(moved)
                declared = [p.value for p in out if p.percentile in LEVELS]
                self.assertTrue(all(b > a for a, b in zip(declared, declared[1:])))
                self.assertTrue(all(10.0 - 2 * 4 <= v <= 14.0 + 2 * 4 for v in declared))
                build(values, q).get_cdf()

    def test_a_far_value_past_a_closed_bound_is_left_alone(self):
        q = numeric(5.345, 7.345, open_lower=False)
        fat = [1.2, 4.0, 5.6, 6.0, 6.4, 6.8]
        _out, moved = ns.shape(percentiles(fat), q)
        self.assertFalse(moved)
        with self.assertRaises(ValueError):
            build(fat, q)

    def test_a_unit_error_is_left_for_the_parser(self):
        """Values past the SDK's own 2x limit are a unit error, not a
        forecast outside the range: still rejected, so main.py falls back to
        the parser as before."""
        for factor in (1000, 0.001):
            values = [v * factor for v in normal_set(12, 0.3)]
            with self.subTest(factor=factor):
                _out, moved = ns.shape(percentiles(values), below_range_question())
                self.assertFalse(moved)
                with self.assertRaises(ValueError):
                    build(values, below_range_question())


class MeanOfCdfsTests(RealSdkTestCase):
    def preds(self, *rows):
        return [build(r, numeric(0, 300)) for r in rows]

    def test_it_is_the_pointwise_mean_of_the_cdfs(self):
        preds = self.preds(normal_set(100, 10), normal_set(110, 20), normal_set(95, 5),
                           normal_set(105, 12))
        aggregate = ns.mean_of_cdfs(preds, numeric(0, 300))
        expected = [sum(col) / 4 for col in zip(*(heights(p) for p in preds))]
        declared = [p.percentile for p in aggregate.declared_percentiles]
        for a, b in zip(declared, expected):
            self.assertAlmostEqual(a, b, places=9)

    def test_the_wider_forecasters_are_heard(self):
        """Three near-identical narrow forecasts and two wide ones: the
        median is the narrow CDF, the mean is not."""
        narrow, wide = normal_set(100, 5), normal_set(100, 20)
        preds = self.preds(narrow, narrow, narrow, wide, wide)
        mean = ns.mean_of_cdfs(preds, numeric(0, 300))
        median = asyncio.run(NumericReport.aggregate_predictions(preds, numeric(0, 300)))

        def width(d):
            ps = d.get_percentiles_at_target_heights([0.1, 0.9])
            return ps[1].value - ps[0].value

        self.assertGreater(width(mean), 1.2 * width(median))

    def test_a_lone_far_forecast_is_dropped_when_four_remain(self):
        near = [normal_set(c, 8) for c in (98, 100, 102, 101)]
        preds = self.preds(*near, normal_set(250, 8))
        with self.assertLogs(ns.logger, level="INFO") as logs:
            aggregate = ns.mean_of_cdfs(preds, numeric(0, 300))
        self.assertIn("INFO:numeric_shape:numeric_aggregate method=mean forecasts=5 dropped=1",
                      logs.output)
        expected = [sum(col) / 4 for col in zip(*(heights(p) for p in preds[:4]))]
        declared = [p.percentile for p in aggregate.declared_percentiles]
        for a, b in zip(declared, expected):
            self.assertAlmostEqual(a, b, places=9)
        # With the far forecast kept, a fifth of the mass would sit past 150.
        self.assertGreater(cdf_at(aggregate, 150), 0.9)

    def test_nothing_is_dropped_from_three(self):
        preds = self.preds(normal_set(100, 8), normal_set(101, 8), normal_set(250, 8))
        with self.assertLogs(ns.logger, level="INFO") as logs:
            ns.mean_of_cdfs(preds, numeric(0, 300))
        self.assertIn("INFO:numeric_shape:numeric_aggregate method=mean forecasts=3 dropped=0",
                      logs.output)

    def test_forecasts_on_different_grids_are_refused(self):
        other = build(normal_set(100, 10), numeric(0, 400))
        with self.assertRaises(ValueError):
            ns.mean_of_cdfs([build(normal_set(100, 10), numeric(0, 300)), other],
                            numeric(0, 300))


# ------------------------------------------------------------------- main.py


def load_main():
    """main.py without its __main__ block, on the real SDK."""
    with real_forecasting_tools():
        path = os.path.join(ROOT, "main.py")
        spec = importlib.util.spec_from_file_location("main_numeric", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def main_src() -> str:
    with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
        return handle.read()


def method_body(src: str, name: str) -> str:
    start = src.index(f"def {name}(")
    following = re.search(r"\n    (?:async )?def |\n    #{5,}", src[start + 1:])
    return src[start:start + 1 + following.start()] if following else src[start:]


class MainWiringTests(RealSdkTestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()
        # The aggregation hook needs no state from __init__.
        cls.bot = object.__new__(cls.main.SummerTemplateBot2026)

    def aggregate(self, predictions, question):
        with real_forecasting_tools():
            return asyncio.run(self.bot._aggregate_predictions(predictions, question))

    def test_every_numeric_construction_goes_through_the_shape(self):
        src = main_src()
        for name in ("_verbatim_numeric", "_parse_numeric", "_date_prompt_to_forecast"):
            with self.subTest(method=name):
                body = method_body(src, name)
                self.assertIn("numeric_shape.build(NumericDistribution,", "".join(body.split()))
                self.assertNotIn("NumericDistribution.from_question(", body)

    def test_numeric_questions_aggregate_by_the_mean(self):
        q = numeric(0, 300)
        spreads = ((100, 5), (100, 5), (100, 5), (100, 20), (104, 20))
        preds = [build(normal_set(c, s), q) for c, s in spreads]
        ours = self.aggregate(preds, q)
        self.assertEqual(heights(ours), heights(ns.mean_of_cdfs(preds, q)))

    def test_a_failure_falls_back_to_the_sdk_median(self):
        q = numeric(0, 300)
        preds = [build(normal_set(100 + i, 10), q) for i in range(5)]
        with mock.patch.object(ns, "mean_of_cdfs", side_effect=RuntimeError("boom")), \
                self.assertLogs(self.main.logger, level="WARNING") as logs:
            ours = self.aggregate(preds, q)
        median = asyncio.run(NumericReport.aggregate_predictions(preds, q))
        self.assertEqual(heights(ours), heights(median))
        self.assertEqual(logs.output, [
            "WARNING:main_numeric:numeric_mean_failed reason=RuntimeError falling_back=median"])

    def test_binary_questions_keep_the_sdk_median(self):
        q = BinaryQuestion(question_text="synthetic", id_of_post=1, id_of_question=2)
        self.assertEqual(self.aggregate([0.2, 0.3, 0.9], q), 0.3)

    def test_the_verbatim_path_rescues_an_out_of_range_set(self):
        text = "\n".join(f"Percentile {round(p * 100)}: {v}"
                         for p, v in zip(LEVELS, OutOfRangeTests.BELOW))
        bot_class = self.main.SummerTemplateBot2026
        with self.assertLogs(ns.logger, level="INFO") as logs:
            prediction = bot_class._verbatim_numeric(text, below_range_question())
        self.assertIsNotNone(prediction)
        self.assertIn(f"INFO:numeric_shape:numeric_clamped=1 url={URL}", logs.output)

    def test_a_shaping_failure_keeps_the_set_as_given(self):
        q = numeric(0, 300)
        with mock.patch.object(ns, "shape", side_effect=RuntimeError("boom")), \
                self.assertLogs(ns.logger, level="WARNING") as logs:
            prediction = build(normal_set(100, 10), q)
        self.assertEqual([p.value for p in prediction.declared_percentiles], normal_set(100, 10))
        self.assertEqual(logs.output, [
            f"WARNING:numeric_shape:numeric_shape_failed reason=RuntimeError url={URL}"])


if __name__ == "__main__":
    unittest.main()
