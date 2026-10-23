"""Guards on what a forecast may say, and the dates every prompt must carry.

Why these exist
---------------
On 141 resolved questions (2026-10-07) the bot's mean peer score was -0.81,
and most of the deficit came from a handful of confident misses near -50 each:
an option the bot all but excluded won a multiple-choice question, and two
event questions went the way the bot had called at the extremes. Peer score is
100 * ln(p / G) against the other forecasters' geometric mean G, so the loss
from a probability near zero on what happens is unbounded in practice, while
the gain from being extreme and right is small. Capping is the one technique
Metaculus' Fall 2025 survey found separating prize winners from the rest
(r=+0.48); extremizing went the other way in Spring 2026.

The prompts also never said when a question closes or resolves, only today's
date, so "time left" -- the first thing each prompt asks the model to state --
had to be guessed from the question text.

Standard library only, so research/ and the tests can import it without the
SDK.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

#: Binary forecasts stay inside [BINARY_MIN, 1 - BINARY_MIN]. At 3%, a miss
#: against a field at 50% costs 100*ln(0.03/0.5) = -281 instead of -391 at 1%;
#: a correct call gives up 100*ln(0.97/0.99) = -2. The SDK's own clamp is 1%.
BINARY_MIN = 0.03

#: Every multiple-choice option keeps at least this much, on questions with
#: up to four options. Giving 1% to an option the field puts at 15% costs
#: -271 when it wins; 2.5% costs -179, 2% costs -201.
OPTION_FLOOR = 0.025

#: The floor from five options up. Five or more floors at 2.5% reserve
#: 12.5% or more of the mass, which a confident favourite pays for: on
#: FutureEval 46191 (closed, six options) the bot field held the favourite
#: near 93% and each outsider at 1.4-2%, while the old mixture (see
#: floor_options) posted all five of our forecasts at 82-83%, as it maps
#: 95%/1%x5 to 0.83/0.033x5. Lifting to 2% gives 0.90/0.02x5 instead
#: (2026-10-23). That loses only if the outsiders together win more than
#: about 13% of the time the model says 95%, and costs about 50 more points
#: when one of them does.
OPTION_FLOOR_MANY = 0.02

#: The SDK's PredictedOptionList clamps every option into [0.01, 0.99], so a
#: floor below 1% is moved there anyway (and rejected if that shifts the
#: favourite by more than 0.05).
_SDK_OPTION_MIN = 0.01

#: The percentiles every numeric prompt asks for, one "Percentile N: value"
#: line each.
PERCENTILES = (10, 20, 40, 60, 80, 90)

_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_PERCENTILE_LINE = re.compile(
    rf"^[ \t*_>-]*Percentile[ \t]+(\d+)[ \t*_]*:[ \t*_]*\$?[ \t]*({_NUMBER})[ \t]*%?[ \t*_\r]*$",
    re.IGNORECASE | re.MULTILINE,
)


def clip_binary(probability: float) -> float:
    return max(BINARY_MIN, min(1 - BINARY_MIN, probability))


def option_floor(n: int, floor: float | None = None) -> float:
    """The floor ``floor_options`` puts under each of ``n`` options.

    OPTION_FLOOR up to four options, OPTION_FLOOR_MANY from five, unless
    ``floor`` is given. Then at most 0.5/n, so the floors never reserve more
    than half the mass, and never below the SDK's own 1% clamp; at 100
    options or more that leaves no room, and the caller gets uniform.
    """
    if floor is None:
        floor = OPTION_FLOOR if n <= 4 else OPTION_FLOOR_MANY
    return max(min(floor, 0.5 / max(n, 1)), _SDK_OPTION_MIN)


def floor_options(probabilities: list[float], floor: float | None = None) -> list[float]:
    """Lift every option to the floor, paid for by the options above it.

    Options under the floor (``option_floor``) go up to it, and the total
    shortfall comes off the options above it in proportion to how far above
    they are, ``p - floor``. So the favourite pays most, nothing ends below
    the floor, the ranking is kept (options at or under the floor tie) and a
    forecast that already clears the floor comes back unchanged. Until
    2026-10-23 this was the mixture ``f + (1 - n*f) * p``, which taxed every
    option, the favourite most, even when nothing was under the floor:
    95%/1%x5 became 0.83/0.033x5 where the bot field had about 0.93 and
    1.4-2% per outsider (see OPTION_FLOOR_MANY); lifting gives 0.90/0.02x5.

    The input is renormalised first and the result sums to exactly 1 inside
    [0.01, 0.99], which matters because the SDK's PredictedOptionList
    validator clamps to that range, renormalises, and rejects the result if
    that moved any option by more than 0.05; a list like this passes it
    unchanged. Negative inputs count as 0, and all-zero input is uniform.
    """
    n = len(probabilities)
    if n == 0:
        return []
    clean = [max(0.0, p) for p in probabilities]
    total = sum(clean)
    floor = option_floor(n, floor)
    if total <= 0 or n * floor >= 1:
        out = [1.0 / n] * n
    else:
        p = [x / total for x in clean]
        short = sum(floor - x for x in p if x < floor)
        # excess - short = sum(p - floor) = 1 - n*floor > 0, so the mass
        # above the floor always covers the shortfall, and every option
        # above it keeps (1 - short/excess) > 0 of its distance to the floor.
        excess = sum(x - floor for x in p if x > floor)
        out = p if short <= 0 else [
            floor if x <= floor else x - short * (x - floor) / excess for x in p
        ]
    # Rounding goes on the favourite, the one option it cannot push under
    # the floor.
    top = max(range(n), key=out.__getitem__)
    out[top] += 1.0 - sum(out)
    return out


def read_percentiles(text: str) -> list[tuple[float, float]] | None:
    """The six "Percentile N: value" lines, read verbatim, as (0.1, value)...

    The forecasters already write their percentiles in the question's units,
    as the prompt asks. Handing them to the gpt-4o-mini parser added a step
    that can only get them wrong: on question 46224 (2026-10-21, units of
    millions) it turned 7840000 into 7840 for three of five forecasters, every
    value fell outside the range and the question was lost. None, so the
    caller falls back to that parser, unless all six lines hold a plain
    number (thousands commas, "$" or "%" allowed; no "million" or "7.8B") and
    the values increase. The last line for each percentile wins.
    """
    found: dict[int, float] = {}
    for match in _PERCENTILE_LINE.finditer(text):
        found[int(match.group(1))] = float(match.group(2).replace(",", ""))
    if not set(PERCENTILES) <= set(found):
        return None
    values = [found[p] for p in PERCENTILES]
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        return None
    return [(p / 100, v) for p, v in zip(PERCENTILES, values)]


def _date(moment: datetime | None) -> str | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def timing_context(
    now: datetime,
    close_time: datetime | None,
    resolution_time: datetime | None,
) -> str:
    """The dates a forecaster needs, stated rather than left to be inferred.

    Ends with the stale-knowledge warning: the forecasting models were trained
    well before today, and the research step may be answering from the same
    memory, so an event "has not happened yet" in the model's head says
    nothing about whether it happened last week.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    lines = [f"Today is {_date(now)}."]
    for label, moment in (
        ("Forecasting on this question closes", close_time),
        ("The question is scheduled to resolve", resolution_time),
    ):
        stated = _date(moment)
        if stated is None:
            continue
        aware = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        days = (aware - now).total_seconds() / 86400
        lines.append(f"{label} on {stated} ({days:.1f} days from now).")
    lines.append(
        "Your training data ends well before today. Treat any event in the "
        "question window as unknown unless the research gives a dated source "
        "for it, and never assume the question has already resolved."
    )
    return "\n".join(lines)
