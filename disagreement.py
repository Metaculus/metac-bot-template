"""When the five forecasts disagree, research the disagreement before deciding.

Why this exists
---------------
Compared with the MiniBench 2026-09-21 top 10 (2026-10-18), the bot's worst
binary misses came from forecasts far from where the field clustered. Ora, one
of the self-described top bots, researches "the crux" when its samples spread
too widely; Metaculus' own Autocast synthesises two forecasters. With Opus 5.5
and GPT-5.6 Sol now voting, a wide spread usually means the two read the same
research differently -- which more targeted research can settle.

Pure functions only (no SDK, no network), so the rule is testable offline;
main.py does the calling.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Largest gap between the highest and lowest of the five forecasts (on any
#: option, for multiple choice) that is still treated as agreement. Wider than
#: this and a targeted research pass runs, then all five forecast again.
SPREAD_THRESHOLD = 0.25

#: Characters of each opposing rationale shown to the crux researcher.
RATIONALE_TAIL = 1500


def _binary_value(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _options(value: Any) -> dict[str, float] | None:
    options = getattr(value, "predicted_options", None)
    if options is None:
        return None
    return {o.option_name: float(o.probability) for o in options}


def spread(values: Sequence[Any]) -> float:
    """Max-minus-min across forecasts: on the probability for binary, on the
    worst option for multiple choice. 0 for anything else (numeric, date)."""
    binary = [v for v in (_binary_value(x) for x in values) if v is not None]
    if len(binary) == len(values) and len(binary) >= 2:
        return max(binary) - min(binary)
    options = [o for o in (_options(x) for x in values) if o is not None]
    if len(options) == len(values) and len(options) >= 2:
        names = set().union(*options)
        return max(
            max(o.get(n, 0.0) for o in options) - min(o.get(n, 0.0) for o in options)
            for n in names
        )
    return 0.0


def needs_crux(values: Sequence[Any]) -> bool:
    return spread(values) > SPREAD_THRESHOLD


def extremes(predictions: Sequence[Any]) -> tuple[Any, Any] | None:
    """(lowest, highest) binary predictions, or the two most different MC
    ones, by their prediction_value. None when there is no comparable pair."""
    values = [p.prediction_value for p in predictions]
    binary = [_binary_value(v) for v in values]
    if all(b is not None for b in binary) and len(binary) >= 2:
        order = sorted(range(len(binary)), key=lambda i: binary[i])
        return predictions[order[0]], predictions[order[-1]]
    options = [_options(v) for v in values]
    if all(o is not None for o in options) and len(options) >= 2:
        best = (0.0, 0, 1)
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                names = set(options[i]) | set(options[j])
                d = max(abs(options[i].get(n, 0) - options[j].get(n, 0)) for n in names)
                best = max(best, (d, i, j))
        return predictions[best[1]], predictions[best[2]]
    return None


def crux_prompt(question_text: str, timing: str, low_reasoning: str, high_reasoning: str) -> str:
    return (
        "Two forecasters read the same research on the question below and "
        "reached very different conclusions. Find the facts that would settle "
        "the disagreement -- the specific claims they weigh differently, what "
        "has or has not happened yet, the scheduled events and the dates -- and "
        "report what you find, dated. Do not forecast.\n\n"
        f"{timing}\n\nQuestion: {question_text}\n\n"
        f"Lower forecast's reasoning (end):\n{low_reasoning[-RATIONALE_TAIL:]}\n\n"
        f"Higher forecast's reasoning (end):\n{high_reasoning[-RATIONALE_TAIL:]}\n\n"
        "End with one line starting 'GAPS:' (or 'GAPS: none')."
    )


#: Characters of the median forecaster's reasoning kept in the public comment.
COMMENT_RATIONALE = 3500
COMMENT_SUMMARY = 3000


def short_comment(question_text: str, final: str, forecaster_values: Sequence[str],
                  summary: str, rationale: str) -> str:
    """The comment Metaculus requires under each forecast, kept short.

    The SDK's default comment pasted the full research plus all five
    rationales (~16,000 characters on question 45745, against 900-4,000 for
    the top bots) and named the models. Metaculus treats long bot comments
    that add no value as spam; this keeps the forecast, the research summary
    and the reasoning closest to the final answer.
    """
    bullets = "\n".join(f"- Forecaster {i + 1}: {v}" for i, v in enumerate(forecaster_values))
    # ForecastReport's validator requires the explanation to start with "#".
    return (
        f"# Summary\n**Question:** {question_text}\n**Final prediction:** {final}\n\n"
        f"### Individual forecasts\n{bullets}\n\n"
        f"### Research summary\n{summary.strip()[:COMMENT_SUMMARY]}\n\n"
        f"### Reasoning (forecaster closest to the final prediction)\n"
        f"{rationale.strip()[:COMMENT_RATIONALE]}"
    )


def closest_to(values: Sequence[Any], final: Any) -> int:
    """Index of the forecast nearest the aggregate (binary); 0 otherwise."""
    b = [_binary_value(v) for v in values]
    f = _binary_value(final)
    if f is None or any(x is None for x in b) or not b:
        return 0
    return min(range(len(b)), key=lambda i: abs(b[i] - f))
