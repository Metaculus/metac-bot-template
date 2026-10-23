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
  has a ``zero_point``). On an open bound the tail may go past the bound but
  stays within ``OPEN_TAIL_LIMIT`` ranges of it, inside the SDK's 2x "far
  exceeding the bounds" rejection. On a closed bound it is used only when it
  lands strictly inside the range: when the t(3) tail would cross the bound,
  the SDK's own uniform 10% between p10 and the bound is already fatter than
  that tail, so that side is left as it is. (The roadmap's alternative, a
  tail at the midpoint between the bound and p10, halves the density right
  at the bound: about 35 points, 50*ln 2, lost on a discrete question that
  resolves at its bound.) The tails never decide whether the SDK accepts a
  set.
* Out-of-range sets (roadmap A3): on 2026-10-20 (run 39453516101) three of
  five Meta EPS forecasts, at 2.8-4.7 on an open 5.345-7.345 range, were
  rejected because no percentile lay within the range +/- 25%; the question
  then failed the SDK's 3-of-5 rule, and the truth was 3.83, below the range.
  One of the first six Market Pulse resolutions landed outside its range. A
  set the SDK would reject for exactly that reason has its open-side values
  moved to ``CLAMP_WINDOW`` ranges past the bound, kept in order, so the mass
  stays on the side the forecaster put it. A set the SDK already accepts
  keeps its declared values exactly as written, and so does a set with a
  value beyond the SDK's own 2x limit (a unit error, which the parser
  fallback is there for) or one past a closed bound.

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
#: Out-of-range rescue: open-side values are moved to this many ranges past
#: the bound, inside the SDK's 25% window.
CLAMP_WINDOW = 0.2
SDK_WIGGLE = 0.25
SDK_FAR = 2.0
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


def _interp(levels: Sequence[float], locs: Sequence[float], level: float) -> float:
    for (l0, x0), (l1, x1) in zip(zip(levels, locs), zip(levels[1:], locs[1:])):
        if l0 <= level <= l1:
            return x0 + (x1 - x0) * (level - l0) / (l1 - l0)
    raise ValueError("level outside the declared percentiles")


def _tails(levels: list[float], values: list[float], question: Any,
           scale: _Scale) -> tuple[float | None, float | None]:
    """(p2.5, p97.5) from the declared values, or None for a side left alone."""
    if not (math.isclose(levels[0], FIT_LEVELS[0]) and math.isclose(levels[-1], FIT_LEVELS[1])):
        return None, None
    locs = [scale.loc(v) for v in values]
    mid = _interp(levels, locs, 0.5)
    low = scale.value(mid - TAIL_C * (mid - locs[0]))
    high = scale.value(mid + TAIL_C * (locs[-1] - mid))
    buffer = scale.sdk_buffer
    if question.open_lower_bound:
        low = max(low, scale.lb - OPEN_TAIL_LIMIT * scale.width)
    elif not low > scale.lb + buffer:
        low = None
    if question.open_upper_bound:
        high = min(high, scale.ub + OPEN_TAIL_LIMIT * scale.width)
    elif not high < scale.ub - buffer:
        high = None
    if low is not None and not low < values[0]:
        low = None
    if high is not None and not high > values[-1]:
        high = None
    return low, high


def _inside_sdk_window(values: Sequence[float], scale: _Scale) -> bool:
    """Whether the SDK's _check_too_far_from_bounds finds a value in range."""
    margin = SDK_WIGGLE * scale.width
    return any(scale.lb - margin <= v <= scale.ub + margin for v in values)


def _rescue_out_of_range(values: list[float], question: Any,
                         scale: _Scale) -> tuple[list[float], bool]:
    """Move open-side values into the SDK's window, only when it would
    otherwise reject the set for having none there."""
    lb, ub, width = scale.lb, scale.ub, scale.width
    if _inside_sdk_window(values, scale):
        return values, False
    if any(v < lb - SDK_FAR * width or v > ub + SDK_FAR * width for v in values):
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
    for i in range(1, n):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out, True


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
    low, high = _tails(levels, declared, question, scale)
    values, moved = _rescue_out_of_range(declared, question, scale)
    out = [make(percentile=lvl, value=v) for lvl, v in zip(levels, values)]
    if not _inside_sdk_window(values, scale):
        # Still a set the SDK rejects (a closed bound, or a unit error): the
        # tails must not be what makes it pass.
        return out, moved
    if low is not None and low < values[0]:
        out.insert(0, make(percentile=TAIL_LEVELS[0], value=low))
    if high is not None and high > values[-1]:
        out.append(make(percentile=TAIL_LEVELS[1], value=high))
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
