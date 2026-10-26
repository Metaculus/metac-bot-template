"""Market Pulse: pool the final forecast with the revealed human community prediction.

Why this exists
---------------
Market Pulse is spot scored: only the forecast standing at each question's
close counts, and market_pulse.select already refreshes every question in its
last 6 hours. The tournament's community prediction (CP) is human only
(``include_bots_in_aggregates=false``) and is revealed 20-70 hours before
close, so it is visible at every final refresh. On the first six resolved
subquestions (three companies' revenue and EPS, as of 2026-10-23) the CP
beat the field on all six, by +31 to +37 peer points a question, against
the bot's ~17. The bot never read its own question's CP: live_context only
follows explicit links.

A linear pool cannot do worse than its parts on average: the log of a
mixture is at least the mixture of the logs, so for any weight w the pooled
peer score is at least ``w * CP's + (1 - w) * ours`` -- binary, multiple
choice and continuous alike. The gain is at least w times the CP's edge over
the bot, and the pool removes the catastrophic misses that sink a spot-scored
total. The rules found say only "enter once" and nothing about the revealed
CP; humans see the same CP, and the pool runs with no human in the loop.

What it does
------------
After the five forecasts are aggregated (main._aggregate_predictions), and
only when ``MP_CP_POOL_WEIGHT`` is above 0 and the run is ``--mode
market_pulse``: if the CP has been revealed (``now >= cp_reveal_time``) and
has at least ``MIN_FORECASTERS`` forecasters, publish ``w * CP + (1 - w) *
ours`` -- on the probabilities for binary and multiple choice (then
forecast_guards.clip_binary / floor_options), on the CDF grid for numeric,
discrete and date questions. Anything missing, unexpected or invalid keeps
our own forecast.

The CP is read from ``question.api_json["question"]["aggregations"]
["recency_weighted"]["latest"]``: the SDK lists questions with
``with_cp=true`` and copies each group subquestion's JSON there. That the
listing carries the full 201-point ``forecast_values`` of a numeric
subquestion has not been checked against a real Market Pulse listing (the
API answered 403 without the token), so when the listing has no CP, or not
in the shape the question's grid needs, the post is read again as the
roadmap says: GET /api/posts/{id_of_post}/ through the bot's own
authenticated client (:func:`detail_fetcher`), once per post per run, and
the log says ``reason=ok_detail``. The pooled CDF is rebuilt with ``from_question`` and the
SDK's usual standardisation, the way the SDK rebuilds its own aggregate,
rather than with ``standardize_cdf=False``: standardising again moves about
1% of the mass to uniform, but it keeps every step a margin above the
5e-05 minimum Metaculus accepts (a convex mix of two CDFs that sit at that
floor sits at it too), and it keeps the SDK's own "too concentrated" check,
which ``standardize_cdf=False`` skips. The 0.2 per-bin cap is checked again
here before anything is returned.

Off by default (weight 0) and to stay off: asked whether a bot may use the
revealed CP, Metaculus answered (2026-10-26) that the CP may be used in any
way short of copying it -- the wisdom of the crowd needs each forecaster's
own reasoning -- and recommended giving it to the model as an input rather
than weighting with it. live_context.own_community does that; this module
remains only as the CP reader (latest_cp). The weight is still a repo
variable (MP_CP_POOL_WEIGHT), and 0 is the only value to use. The log carries only ``cp_pool applied=0|1 reason=...``:
the repository is public, so no CP or forecast value ever reaches it.

Standard library only; the SDK objects are duck-typed, as in numeric_shape.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import forecast_guards

logger = logging.getLogger(__name__)

ENV_WEIGHT = "MP_CP_POOL_WEIGHT"
MIN_FORECASTERS = 10
#: Largest step a 200-bin CDF may take (NumericDefaults.MAX_NUMERIC_PMF_VALUE);
#: scaled for discrete questions with fewer bins, as the SDK does.
MAX_STEP = 0.2


#: Listing reasons a fresh read of the post can cure: no CP in the listing,
#: or not in the shape this question's grid needs.
REFETCH = ("no_cp", "shape")

Fetch = Callable[[int], list]


class _Skip(Exception):
    """Keep our own forecast, for the reason given."""


def weight_from_env(environ: Mapping[str, str] | None = None) -> float:
    """The CP weight in [0, 1]; 0 (off) when unset, empty or not a number
    in range."""
    raw = (os.environ if environ is None else environ).get(ENV_WEIGHT, "").strip()
    if not raw:
        return 0.0
    try:
        weight = float(raw)
    except ValueError:
        weight = math.nan
    if not 0.0 <= weight <= 1.0:
        logger.warning("cp_pool_weight_invalid variable=%s using=0", ENV_WEIGHT)
        return 0.0
    return weight


def _utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def detail_fetcher(client: Any) -> Fetch:
    """``fetch(post_id)``: the post's questions from GET /posts/{id}/, read
    through the bot's authenticated MetaculusClient, group subquestions
    unpacked the way discovery unpacks them. Each post is read at most once
    per run -- a Market Pulse group's subquestions share one post -- and a
    failure is remembered too, so a down API costs one wait, not one per
    subquestion."""
    cache: dict[int, Any] = {}

    def fetch(post_id: int) -> list:
        if post_id not in cache:
            try:
                found = client.get_question_by_post_id(
                    post_id, group_question_mode="unpack_subquestions")
                cache[post_id] = found if isinstance(found, list) else [found]
            except Exception as exc:  # noqa: BLE001 - re-raised below, every time
                cache[post_id] = exc
        result = cache[post_id]
        if isinstance(result, Exception):
            raise result
        return result

    return fetch


def _detail_api(question: Any, fetch: Fetch) -> dict:
    """This subquestion's JSON from a fresh read of its post."""
    post_id = getattr(question, "id_of_post", None)
    if not isinstance(post_id, int):
        raise _Skip("no_cp")
    found = fetch(post_id)
    wanted = getattr(question, "id_of_question", None)
    match = [q for q in found if getattr(q, "id_of_question", None) == wanted]
    if not match and len(found) == 1:
        match = found
    if not match:
        raise _Skip("no_cp")
    return getattr(match[0], "api_json", None) or {}


def latest_cp(question: Any, now: datetime, api: dict | None = None) -> dict:
    """The revealed, current CP aggregation with enough forecasters, from
    ``api`` (a fresh read of the post) or the question's own listing JSON."""
    reveal = getattr(question, "cp_reveal_time", None)
    if reveal is None or _utc(now) < _utc(reveal):
        raise _Skip("not_revealed")
    if api is None:
        api = getattr(question, "api_json", None) or {}
    latest = (((api.get("question") or {}).get("aggregations") or {})
              .get("recency_weighted") or {}).get("latest")
    if not isinstance(latest, dict) or not (latest.get("forecast_values") or latest.get("centers")):
        raise _Skip("no_cp")
    end = latest.get("end_time")
    if isinstance(end, (int, float)) and end < _utc(now).timestamp():
        raise _Skip("stale")
    count = latest.get("forecaster_count")
    if not isinstance(count, int) or count < MIN_FORECASTERS:
        raise _Skip("few_forecasters")
    return latest


def _probabilities(values: Any, n: int) -> list[float]:
    if not isinstance(values, list) or len(values) != n:
        raise _Skip("shape")
    out = []
    for v in values:
        if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            raise _Skip("shape")
        out.append(float(v))
    return out


def _pool_binary(ours: float, latest: dict, w: float) -> float:
    """``forecast_values`` is [P(no), P(yes)]; ``centers`` [P(yes)] is what
    the SDK itself reads for BinaryQuestion.community_prediction_at_access_time."""
    if latest.get("forecast_values"):
        _no, yes = _probabilities(latest["forecast_values"], 2)
    else:
        (yes,) = _probabilities(latest.get("centers"), 1)
    return forecast_guards.clip_binary(w * yes + (1 - w) * float(ours))


def _pool_multiple_choice(ours: Any, latest: dict, question: Any, w: float,
                          api: dict | None = None) -> Any:
    api = (getattr(question, "api_json", None) or {}) if api is None else api
    api_options = (api.get("question") or {}).get("options")
    names = api_options or list(getattr(question, "options", []) or [])
    cp = dict(zip(names, _probabilities(latest.get("forecast_values"), len(names))))
    options = ours.predicted_options
    if len(options) != len(cp) or any(o.option_name not in cp for o in options):
        raise _Skip("shape")
    mixed = [w * cp[o.option_name] + (1 - w) * o.probability for o in options]
    floored = forecast_guards.floor_options(mixed)
    make = type(options[0])
    return type(ours)(predicted_options=[
        make(option_name=o.option_name, probability=p) for o, p in zip(options, floored)
    ])


def _pool_cdf(ours: Any, latest: dict, question: Any, w: float) -> Any:
    grid = ours.get_cdf()
    n = len(grid)
    cp = _probabilities(latest.get("forecast_values"), n)
    if any(b < a for a, b in zip(cp, cp[1:])):
        raise _Skip("shape")
    if (not question.open_lower_bound and cp[0] > 1e-9) or \
            (not question.open_upper_bound and cp[-1] < 1 - 1e-9):
        raise _Skip("shape")
    make = type(grid[0])
    mixed = [make(value=pt.value, percentile=w * c + (1 - w) * pt.percentile)
             for pt, c in zip(grid, cp)]
    try:
        pooled = type(ours).from_question(mixed, question)
        heights = [pt.percentile for pt in pooled.get_cdf()]
    except ValueError as exc:
        raise _Skip("invalid") from exc
    cap = MAX_STEP * 200 / (n - 1)
    if len(heights) != n or max(b - a for a, b in zip(heights, heights[1:])) >= cap:
        raise _Skip("invalid")
    return pooled


def _pool(aggregate: Any, question: Any, weight: float, now: datetime,
          api: dict | None = None) -> Any:
    latest = latest_cp(question, now, api)
    if isinstance(aggregate, bool):
        raise _Skip("type")
    if isinstance(aggregate, (int, float)):
        return _pool_binary(aggregate, latest, weight)
    if hasattr(aggregate, "predicted_options"):
        return _pool_multiple_choice(aggregate, latest, question, weight, api)
    if hasattr(aggregate, "get_cdf"):
        return _pool_cdf(aggregate, latest, question, weight)
    raise _Skip("type")


def apply(aggregate: Any, question: Any, weight: float,
          now: datetime | None = None, fetch: Fetch | None = None) -> Any:
    """``weight * CP + (1 - weight) * aggregate``, or ``aggregate`` itself
    when the CP cannot be used. With ``fetch`` (:func:`detail_fetcher`), a
    listing without a usable CP is followed by a fresh read of the post; the
    reasons logged then end in ``_detail``. Never raises."""
    if not weight > 0:
        return aggregate
    url = getattr(question, "page_url", None)
    now = now or datetime.now(timezone.utc)
    suffix = ""
    try:
        try:
            pooled = _pool(aggregate, question, weight, now)
        except _Skip as skip:
            if fetch is None or str(skip) not in REFETCH:
                raise
            suffix = "_detail"
            pooled = _pool(aggregate, question, weight, now, _detail_api(question, fetch))
    except _Skip as skip:
        logger.info("cp_pool applied=0 reason=%s%s url=%s", skip, suffix, url)
        return aggregate
    except Exception as exc:  # noqa: BLE001 - never lose the forecast over the pool
        logger.warning("cp_pool applied=0 reason=error:%s url=%s", type(exc).__name__, url)
        return aggregate
    logger.info("cp_pool applied=1 reason=ok%s url=%s", suffix, url)
    return pooled
