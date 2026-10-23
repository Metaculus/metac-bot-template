"""Numeric forecasts: smooth tails on each forecaster, and the mean of their CDFs.

Why this exists
---------------
What the SDK (forecasting-tools 0.2.90) did with a numeric, discrete or date
question until 2026-10-23:

* From the six declared percentiles it draws each CDF as straight lines and
  spreads what lies beyond p10/p90 uniformly out to the bound; on an open
  bound it pins p95 (or p5) exactly at the bound. The density therefore falls
  20-40x just past p10/p90 (numeric_report.py:422-471), and that cliff is
  what gets posted.
* It aggregates the five CDFs by their pointwise MEDIAN (numeric_report.py:
  633). Three of the five calls are Opus 5.5 and its three samples are
  near-duplicates, so the median is Opus's own narrow CDF and the two wider
  GPT-5.6 Sol samples are ignored.

Both matter more now than before: since the 18-Oct prompt changes the bot's
10-90 width on 17 closed FutureEval questions is 0.78x the community's (it was
1.13x), so the risk has flipped from too wide to too narrow. A Monte Carlo on
10 logged five-forecaster sets (2026-10-23) found the mean of the CDFs beats
the SDK median by +0.04 to +0.25 nats per question, and the mean plus the
synthesized tails below by +0.08 to +0.51, in every setting tried; tails
synthesized in code were never worse in 12 scenarios, while asking the model
for p1/p99 lost when the model was overconfident and the range narrow (an
LLM's extreme percentiles are its worst calibrated output). Continuous peer
scores are 50*ln, so that is roughly +2.5 to +7.5 points per question.

What it does
------------
:func:`shape` -- for each forecaster, before ``from_question``:

* Tails (roadmap A2): adds p2.5 and p97.5 from a Student-t with 3 degrees of
  freedom fitted to each side, ``p97.5 = p50 + TAIL_C * (p90 - p50)`` and its
  mirror, computed in the question's own scale (log space when the question
  has a ``zero_point``). What happens next depends on the bound:

  - Open bound, tail past it: kept, within ``OPEN_TAIL_LIMIT`` ranges of the
    bound (inside the SDK's 2x "far exceeding the bounds" rejection), as long
    as it leaves at least the SDK's own mass past the bound.
  - Open bound, tail inside the range (the usual case), or past it but too
    close to leave that mass: the SDK pins the level halfway between the top
    declared level and 1 at the bound, so a bare p97.5 inside the range
    would cut the mass past the bound from the SDK's 5% to 1.25% and pile
    1.25% into the strip between p97.5 and the bound. Outcomes past the
    range are not rare (1 of the first 6 Market Pulse resolutions), so the
    t(3) is instead drawn inside the range at ``INSIDE_LEVELS`` and scaled to
    reach exactly the SDK's 95% (5%) at the bound, which is declared
    explicitly: the mass past the bound is never below what the SDK gave,
    and the density never rises towards the bound.
  - Closed bound: used only when it lands strictly inside the range and the
    2.5% the SDK then puts between it and the bound is no denser than the
    tail itself; otherwise that side is left as the SDK builds it. When the
    t(3) tail would cross the bound, the SDK's own uniform 10% between p90
    and the bound is already fatter than that tail, and a tail just short of
    it would pile the last 2.5% against the bound. (The roadmap's
    alternative, a tail at the midpoint between the bound and p90, halves
    the density right at the bound: about 35 points, 50*ln 2, lost on a
    discrete question that resolves at its bound.)

  The tails never decide whether the SDK accepts a set.
* Out-of-range sets (roadmap A3): on 2026-10-20 (run 39453516101) three of
  five Meta EPS forecasts, at 2.8-4.7 on an open 5.345-7.345 range, were
  rejected because no percentile lay within the range +/- 25%; the question
  then failed the SDK's 3-of-5 rule, and the truth was 3.83, below the range.
  One of the first six Market Pulse resolutions landed outside its range. A
  set the SDK would reject for exactly that reason has its open-side values
  moved to ``CLAMP_WINDOW`` ranges past the bound, kept in order, so the mass
  stays on the side the forecaster put it. A set the SDK would reject only
  for an open-side value more than 2 ranges past the bound, with other
  values in its window, has that value brought to ``OPEN_TAIL_LIMIT`` ranges
  past the bound, as the roadmap's "always keep values within +/-1.9R"
  asks: it is a fat tail, and as written it lost the whole set on both the
  verbatim and the parser path, which reads the same text. Any other set
  the SDK accepts keeps its declared values exactly as written, and so does
  a set wholly outside the window with a value beyond the 2x limit (a unit
  error, which the parser fallback is there for) or one past a closed
  bound. Either move is logged as ``numeric_clamped=1``.

:func:`mean_of_cdfs` -- the aggregate: the mean of the forecasters' CDFs on
the question's grid (a linear pool, which is how Metaculus aggregates and
which can never score below the average of its members under the log score),
after dropping a forecast whose median sits more than ``OUTLIER_IQRS`` pooled
IQRs from the median of the medians -- a unit or label error -- as long as at
least ``MIN_KEPT`` remain.

The SDK objects are duck-typed (``percentile``/``value``, ``get_cdf()``,
``from_question``) and the module imports nothing from forecasting_tools, so
it loads without the SDK. Any failure here degrades to the SDK's own
construction with a one-line warning: :func:`build` falls back by itself, and
main._aggregate_predictions falls back to the SDK median.
"""

from __future__ import annotations

import logging
import math
from statistics import median
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: p97.5 = p50 + TAIL_C * (p90 - p50): the ratio of a t(3)'s 97.5% and 90%
#: quantiles, 3.182 / 1.638. A normal would give 1.53. If a Test Bot run shows
#: silly tails, about 1.75 is the fallback.
TAIL_C = 1.94
TAIL_LEVELS = (0.025, 0.975)
#: The declared extremes the tails are fitted to: the prompts ask for
#: percentiles 10..90 (forecast_guards.PERCENTILES).
FIT_LEVELS = (0.1, 0.9)
#: On an open bound a tail stays within this many ranges past the bound; the
#: SDK rejects any value more than 2 ranges out.
OPEN_TAIL_LIMIT = 1.9
#: Levels of the t(3) drawn inside an open range when a bare p97.5 would
#: leave less than the SDK's mass past the bound; rescaled before use (see
#: _inside_tail). Dense near p90, where the density is highest.
INSIDE_LEVELS = (0.925, 0.95, 0.975, 0.99)
#: Declared levels closer than this are not added: the SDK refuses steps
#: under 5e-05.
MIN_STEP = 1e-3
#: Out-of-range rescue: open-side values are moved to this many ranges past
#: the bound, inside the SDK's 25% window.
CLAMP_WINDOW = 0.2
SDK_WIGGLE = 0.25
SDK_FAR = 2.0
#: A set lying wholly beyond one bound, by this factor or more against that
#: bound, is a unit error (thousands written for millions), not a forecast.
#: The 2x rule alone misses it on a wide range: question 46224's parser wrote
#: 7,840 for 7,840,000 against a 4M-10M range, which is under two ranges
#: below the lower bound, and the rescue would have posted 90% "below the
#: range" (2026-10-23 review). Left for the SDK to reject, as before.
UNIT_ERROR_FACTOR = 20.0
OUTLIER_IQRS = 3.0
MIN_KEPT = 4


# ------------------------------------------------------------------- scale


def _float(bound: Any) -> float:
    """Date questions carry datetimes; their forecasts are timestamps."""
    return float(bound.timestamp()) if hasattr(bound, "timestamp") else float(bound)


class _Scale:
    """The question's own axis: nominal value <-> location in [0, 1].

    The same mapping as NumericDistribution._nominal_location_to_cdf_location:
    linear, or logarithmic around ``zero_point``.
    """

    def __init__(self, question: Any):
        self.lb = _float(question.lower_bound)
        self.ub = _float(question.upper_bound)
        self.width = self.ub - self.lb
        zero = getattr(question, "zero_point", None)
        self.zero = None if zero is None else float(zero)
        if not self.width > 0:
            raise ValueError("question range is empty")
        if self.zero is not None:
            if not self.zero < self.lb:
                raise ValueError("zero point is not below the range")
            self.ratio = math.log((self.ub - self.zero) / (self.lb - self.zero))

    def loc(self, x: float) -> float:
        if self.zero is None:
            return (x - self.lb) / self.width
        return math.log((x - self.zero) / (self.lb - self.zero)) / self.ratio

    def value(self, loc: float) -> float:
        if self.zero is None:
            return self.lb + self.width * loc
        return self.zero + (self.lb - self.zero) * math.exp(self.ratio * loc)

    @property
    def sdk_buffer(self) -> float:
        """What the SDK moves values near a closed bound to (numeric_report.py:
        _add_explicit_upper_lower_bound_percentiles)."""
        return 1.0 if self.width > 100 else 0.01 * self.width


# ------------------------------------------------------------------- shape


Point = tuple[float, float]


def _interp(levels: Sequence[float], locs: Sequence[float], level: float) -> float:
    for (l0, x0), (l1, x1) in zip(zip(levels, locs), zip(levels[1:], locs[1:])):
        if l0 <= level <= l1:
            return x0 + (x1 - x0) * (level - l0) / (l1 - l0)
    raise ValueError("level outside the declared percentiles")


def _t3_cdf(z: float) -> float:
    """The Student-t CDF with 3 degrees of freedom (closed form)."""
    x = z / math.sqrt(3.0)
    return 0.5 + (math.atan(x) + x / (1.0 + x * x)) / math.pi


def _t3_quantile(p: float) -> float:
    lo, hi = 0.0, 1e3
    for _ in range(200):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if _t3_cdf(m) < p else (lo, m)
    return (lo + hi) / 2


#: TAIL_C is the ratio of the t(3)'s 97.5% and 90% quantiles, 3.182 / 1.638.
_T3_Q975 = _t3_quantile(TAIL_LEVELS[1])
_INSIDE_Q = tuple(_t3_quantile(p) for p in INSIDE_LEVELS)


def _inside_tail(mid: float, edge: float, edge_level: float, tail: float,
                 cap: float) -> list[tuple[float, float | None]]:
    """The t(3) whose 97.5% point is ``tail``, drawn at INSIDE_LEVELS inside
    the range and rescaled to run from ``edge_level`` at the declared extreme
    to ``cap`` at the bound, which is declared too (location None). A t(3)
    beyond its median is concave, so the density falls all the way to the
    bound."""
    spread = (tail - mid) / _T3_Q975
    f_edge = _t3_cdf((edge - mid) / spread)
    share = (cap - edge_level) / (_t3_cdf((1.0 - mid) / spread) - f_edge)
    points: list[tuple[float, float | None]] = []
    last = edge_level
    for q in _INSIDE_Q:
        x = mid + spread * q
        if not edge < x < 1.0:
            continue
        level = edge_level + (_t3_cdf(q) - f_edge) * share
        if level - last >= MIN_STEP and cap - level >= MIN_STEP:
            points.append((level, x))
            last = level
    return points + [(cap, None)]


def _past(mid: float, edge: float, edge_level: float, tail: float,
          open_bound: bool, clear: bool) -> list[tuple[float, float | None]]:
    """(level, location) of the points to declare past one declared extreme,
    the location None for the bound itself. In the upper side's frame:
    locations grow towards the bound at 1, levels towards 1; the lower side
    is mirrored into it. See the module docstring for the cases."""
    top = TAIL_LEVELS[1]
    if not (edge > mid and tail > edge):
        return []
    if not open_bound:
        if not clear:
            return []
        # The SDK runs straight from the tail to the bound at level 1.
        if (1.0 - tail) / (1.0 - top) < (tail - edge) / (top - edge_level):
            return []
        return [(top, tail)]
    if edge >= 1.0:
        return [(top, tail)]  # past the bound already: nothing in range changes
    # The level the SDK pins at an open bound past the declared extreme.
    cap = 1.0 - 0.5 * (1.0 - edge_level)
    if tail > 1.0:
        at_bound = edge_level + (top - edge_level) * (1.0 - edge) / (tail - edge)
        if at_bound <= cap:
            return [(top, tail)]
    return _inside_tail(mid, edge, edge_level, tail, cap)


def _value(scale: _Scale, loc: float) -> float | None:
    """The nominal value at ``loc``, or None where it cannot be one: past
    the float range, or at or below a log scale's zero point."""
    try:
        value = scale.value(loc)
    except OverflowError:
        return None
    if not math.isfinite(value) or (scale.zero is not None and not value > scale.zero):
        return None
    return value


def _side(scale: _Scale, points: list[tuple[float, float | None]], bound: float,
          mirror: bool) -> list[Point]:
    """(level, value) for one side's points from _past, in increasing order;
    empty when any of them has no value."""
    out = []
    for level, x in points:
        value = bound if x is None else _value(scale, 1.0 - x if mirror else x)
        if value is None:
            return []
        out.append((1.0 - level, value) if mirror else (level, value))
    return out[::-1] if mirror else out


def _tails(levels: list[float], values: list[float], question: Any,
           scale: _Scale) -> tuple[list[Point], list[Point]]:
    """The (level, value) points to declare below p10 and above p90, each
    side in increasing order; empty for a side left to the SDK. Worked out
    in the question's own [0, 1] locations, so a tail far out on a log scale
    never has to be a number first."""
    if not (math.isclose(levels[0], FIT_LEVELS[0]) and math.isclose(levels[-1], FIT_LEVELS[1])):
        return [], []
    locs = [scale.loc(v) for v in values]
    mid = _interp(levels, locs, 0.5)
    t_low = mid - TAIL_C * (mid - locs[0])
    t_high = mid + TAIL_C * (locs[-1] - mid)
    reach = OPEN_TAIL_LIMIT * scale.width
    if question.open_upper_bound:
        t_high = min(t_high, scale.loc(scale.ub + reach))
    if question.open_lower_bound and (scale.zero is None or scale.lb - reach > scale.zero):
        t_low = max(t_low, scale.loc(scale.lb - reach))
    high, low = _value(scale, t_high), _value(scale, t_low)
    buffer = scale.sdk_buffer
    upper = _past(mid, locs[-1], levels[-1], t_high, question.open_upper_bound,
                  high is not None and high < scale.ub - buffer)
    lower = _past(1.0 - mid, 1.0 - locs[0], 1.0 - levels[0], 1.0 - t_low,
                  question.open_lower_bound, low is not None and low > scale.lb + buffer)
    return (_side(scale, lower, scale.lb, mirror=True),
            _side(scale, upper, scale.ub, mirror=False))


def _ordered(points: Sequence[Point]) -> bool:
    return all(b[0] > a[0] and b[1] > a[1] for a, b in zip(points, points[1:]))


def _inside_sdk_window(values: Sequence[float], scale: _Scale) -> bool:
    """Whether the SDK's _check_too_far_from_bounds finds a value in range."""
    margin = SDK_WIGGLE * scale.width
    return any(scale.lb - margin <= v <= scale.ub + margin for v in values)


def _ordered_with_eps(out: list[float], eps: float) -> list[float]:
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def _clip_far_tail(values: list[float], question: Any,
                   scale: _Scale) -> tuple[list[float], bool]:
    """Open-side values past the SDK's 2x limit, in a set with values in its
    window, brought to OPEN_TAIL_LIMIT ranges past the bound, kept in order."""
    lb, ub, width = scale.lb, scale.ub, scale.width
    eps = width * 1e-6
    n = len(values)
    out = list(values)
    if question.open_lower_bound:
        edge = lb - OPEN_TAIL_LIMIT * width
        if scale.zero is None or edge > scale.zero:
            out = [edge + i * eps if v < lb - SDK_FAR * width else v for i, v in enumerate(out)]
    if question.open_upper_bound:
        edge = ub + OPEN_TAIL_LIMIT * width
        out = [edge - (n - 1 - i) * eps if v > ub + SDK_FAR * width else v
               for i, v in enumerate(out)]
    if out == values:
        return values, False
    return _ordered_with_eps(out, eps), True


def _looks_like_unit_error(values: Sequence[float], scale: _Scale) -> bool:
    """Every value beyond one positive bound by UNIT_ERROR_FACTOR or more."""
    if scale.lb > 0 and all(v < scale.lb for v in values):
        return max(values) * UNIT_ERROR_FACTOR <= scale.lb
    if scale.ub > 0 and all(v > scale.ub for v in values):
        return min(values) >= scale.ub * UNIT_ERROR_FACTOR
    return False


def _rescue_out_of_range(values: list[float], question: Any,
                         scale: _Scale) -> tuple[list[float], bool]:
    """Move open-side values into the SDK's window, only when it would
    otherwise reject the set for having none there; or, when it has some
    there, bring an open-side value past the SDK's 2x limit back inside it."""
    lb, ub, width = scale.lb, scale.ub, scale.width
    if _inside_sdk_window(values, scale):
        return _clip_far_tail(values, question, scale)
    if any(v < lb - SDK_FAR * width or v > ub + SDK_FAR * width for v in values):
        return values, False
    if _looks_like_unit_error(values, scale):
        return values, False
    out = list(values)
    eps = width * 1e-6
    n = len(out)
    if question.open_lower_bound:
        edge = lb - CLAMP_WINDOW * width
        if scale.zero is None or edge > scale.zero:
            out = [edge + i * eps if v < edge else v for i, v in enumerate(out)]
    if question.open_upper_bound:
        edge = ub + CLAMP_WINDOW * width
        out = [edge - (n - 1 - i) * eps if v > edge else v for i, v in enumerate(out)]
    if out == values:
        return values, False
    return _ordered_with_eps(out, eps), True


def shape(percentiles: Sequence[Any], question: Any) -> tuple[list[Any], bool]:
    """The percentiles to build one forecaster's distribution from, and
    whether an out-of-range set was moved into the range (see the module
    docstring). The tails are fitted to the values as declared, before any
    move, so a rescued set keeps the forecaster's own tail."""
    ps = sorted(percentiles, key=lambda p: p.percentile)
    if len(ps) < 2:
        return list(ps), False
    scale = _Scale(question)
    make = type(ps[0])
    levels = [float(p.percentile) for p in ps]
    declared = [float(p.value) for p in ps]
    lower, upper = _tails(levels, declared, question, scale)
    values, moved = _rescue_out_of_range(declared, question, scale)
    out = [make(percentile=lvl, value=v) for lvl, v in zip(levels, values)]
    if not _inside_sdk_window(values, scale):
        # Still a set the SDK rejects (a closed bound, or a unit error): the
        # tails must not be what makes it pass.
        return out, moved
    # A rescued set has moved away from the values the tails were fitted
    # to: a point that would now sit among the declared ones is left out.
    lower = [(lvl, v) for lvl, v in lower if v < values[0]]
    upper = [(lvl, v) for lvl, v in upper if v > values[-1]]
    if lower and _ordered(lower + [(levels[0], values[0])]):
        out = [make(percentile=lvl, value=v) for lvl, v in lower] + out
    if upper and _ordered([(levels[-1], values[-1])] + upper):
        out += [make(percentile=lvl, value=v) for lvl, v in upper]
    return out, moved


def build(distribution: Any, percentiles: Sequence[Any], question: Any) -> Any:
    """``distribution.from_question`` on :func:`shape` of the percentiles.

    Any failure of the shaping falls back to the percentiles exactly as
    given, so this raises only what ``from_question`` raised before it: a set
    the SDK rejects still reaches the caller's parser fallback as a
    ValueError. ``numeric_clamped=1`` marks a rescued set (a count, never a
    value).
    """
    url = getattr(question, "page_url", None)
    try:
        shaped, moved = shape(percentiles, question)
        prediction = distribution.from_question(shaped, question)
        prediction.get_cdf()  # validate now, not at publication
    except Exception as exc:  # noqa: BLE001 - the unshaped set is the old behaviour
        prediction = distribution.from_question(list(percentiles), question)
        logger.warning("numeric_shape_failed reason=%s url=%s", type(exc).__name__, url)
        return prediction
    if moved:
        logger.info("numeric_clamped=1 url=%s", url)
    return prediction


# --------------------------------------------------------------- aggregate


def _quantile(heights: Sequence[float], q: float) -> float:
    """Location in [0, 1] where a CDF on the question's grid reaches ``q``."""
    n = len(heights)
    if q <= heights[0]:
        return 0.0
    for i in range(1, n):
        if heights[i] >= q:
            step = heights[i] - heights[i - 1]
            frac = (q - heights[i - 1]) / step if step > 0 else 0.0
            return (i - 1 + frac) / (n - 1)
    return 1.0


def outliers(cdfs: Sequence[Sequence[float]]) -> list[int]:
    """Indices of forecasts whose median is more than OUTLIER_IQRS pooled IQRs
    from the median of medians; empty unless MIN_KEPT would still remain."""
    n = len(cdfs)
    if n - 1 < MIN_KEPT:
        return []
    points = len(cdfs[0])
    pooled = [sum(c[i] for c in cdfs) / n for i in range(points)]
    iqr = max(_quantile(pooled, 0.75) - _quantile(pooled, 0.25), 1.0 / (points - 1))
    medians = [_quantile(c, 0.5) for c in cdfs]
    centre = median(medians)
    far = [i for i, m in enumerate(medians) if abs(m - centre) > OUTLIER_IQRS * iqr]
    return far if n - len(far) >= MIN_KEPT else []


def mean_of_cdfs(predictions: Sequence[Any], question: Any) -> Any:
    """The linear pool of the forecasters' distributions, as one
    distribution of the same type, built and validated the way the SDK
    builds its median (``from_question`` on the full grid)."""
    if not predictions:
        raise ValueError("no predictions to aggregate")
    cdfs = [p.get_cdf() for p in predictions]
    grid = [pt.value for pt in cdfs[0]]
    if any([pt.value for pt in c] != grid for c in cdfs):
        raise ValueError("the forecasts are not on the same grid")
    heights = [[pt.percentile for pt in c] for c in cdfs]
    dropped = outliers(heights)
    kept = [h for i, h in enumerate(heights) if i not in dropped]
    mean = [sum(h[i] for h in kept) / len(kept) for i in range(len(grid))]
    make = type(cdfs[0][0])
    aggregate = type(predictions[0]).from_question(
        [make(value=x, percentile=m) for x, m in zip(grid, mean)], question
    )
    aggregate.get_cdf()  # validate now, not at publication
    logger.info("numeric_aggregate method=mean forecasts=%d dropped=%d",
                len(predictions), len(dropped))
    return aggregate
