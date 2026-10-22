"""What the bot spends on APIs: one ledger line per paid call, summed for WhatsApp.

Why this exists
---------------
The Claude API key runs on the ~$100/month credit of the owner's Max plan,
a hard limit. Nothing recorded the spend: the SDK prices every litellm call
into a per-question ``final_cost`` and drops it, and claude_research calls
the Anthropic SDK directly, which litellm never sees -- so even that figure
left out the research, the one part that comes out of the budget. Here both
are recorded, a month of history survives in the Actions cache
(run_bot_on_tournament.yaml restores and saves it, weekly_report.yaml reads
it), the owner gets one WhatsApp alert the first time a month crosses each
threshold, and main.py drops to light research once the budget is spent.

Every line holds a time, the run id, the tournament, the question's post id,
the phase, the provider, the model, tokens, searches and dollars -- never a
forecast, a rationale or anything else that says what the bot predicted. The
repository is public and so are its cache and step summaries.

Recording is off unless SPEND_LEDGER_PATH is set (the production workflow
sets it), so tests and local runs write nothing. Nothing here may fail a run:
every entry point swallows its own errors. Standard library only;
forecasting_tools and litellm are imported inside the functions that use them.

    python spend.py after-run   # end of each production job: alerts, summary
    python spend.py weekly      # Monday message, printed to stdout
"""

from __future__ import annotations

import calendar
import contextvars
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: The provider whose spend counts against the budget. OpenRouter traffic
#: (forecasts, parser, summary) runs on Metaculus' sponsored key and is
#: reported, not budgeted.
BUDGET_PROVIDER = "anthropic"
DEFAULT_BUDGET_USD = 100.0
DEFAULT_THRESHOLDS = (0.8, 1.0)
DEFAULT_THROTTLE_AT = 1.0
#: Lines older than this are dropped at the end of each job; two months keep
#: the weekly window and the current month whole.
KEEP_DAYS = 70
#: The disagreement follow-up owns everything it triggers (research and the
#: five fresh forecasts), so phases set inside it do not relabel its calls.
CRUX = "crux"
NOTIFY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "notify.py")

_question: contextvars.ContextVar[int | None] = contextvars.ContextVar("spend_question", default=None)
_phase: contextvars.ContextVar[str | None] = contextvars.ContextVar("spend_phase", default=None)
_tournament: str | None = None
_warned = False


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def budget_usd(now: datetime | None = None) -> float:
    """CLAUDE_BUDGET_USD, in dollars. "60@2026-10" applies 60 to October 2026
    only and the default to every other month: the ledger started on
    2026-10-22, after ~$40 of October had been spent, and a plain 60 would
    have had to be removed by hand on 1 November."""
    raw = os.environ.get("CLAUDE_BUDGET_USD", "").strip()
    amount, _, month = raw.partition("@")
    if month and month.strip() != _month(now or datetime.now(timezone.utc)):
        return DEFAULT_BUDGET_USD
    try:
        return float(amount) if amount.strip() else DEFAULT_BUDGET_USD
    except ValueError:
        return DEFAULT_BUDGET_USD


def thresholds() -> tuple[float, ...]:
    raw = os.environ.get("SPEND_ALERT_THRESHOLDS", "").strip()
    try:
        parsed = tuple(sorted(float(x) for x in raw.split(",") if x.strip()))
    except ValueError:
        parsed = ()
    return parsed or DEFAULT_THRESHOLDS


def throttle_at() -> float:
    return _float_env("SPEND_THROTTLE_AT", DEFAULT_THROTTLE_AT)


def ledger_path() -> str | None:
    return os.environ.get("SPEND_LEDGER_PATH", "").strip() or None


############################## RECORDING ##############################

def set_tournament(label: str | None) -> None:
    """main.py runs one tournament per asyncio.run, so a module global is
    enough; questions inside a run are told apart by the context variable."""
    global _tournament
    _tournament = label


@contextmanager
def question(q: Any):
    token = _question.set(getattr(q, "id_of_post", None))
    try:
        yield
    finally:
        _question.reset(token)


@contextmanager
def phase(name: str):
    if _phase.get() == CRUX:
        yield
        return
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


def _warn_once(what: str, exc: BaseException) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning("spend_%s_failed reason=%s", what, type(exc).__name__)


def record(*, provider: str, model: str, usd: float, tokens_in: int = 0,
           tokens_out: int = 0, searches: int = 0) -> None:
    """Append one paid call to the ledger. Never raises."""
    path = ledger_path()
    if path is None:
        return
    try:
        line = {
            "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run": os.environ.get("GITHUB_RUN_ID"),
            "tour": _tournament,
            "q": _question.get(),
            "phase": _phase.get() or "other",
            "provider": provider,
            "model": model,
            "in": int(tokens_in or 0),
            "out": int(tokens_out or 0),
            "searches": int(searches or 0),
            "usd": round(float(usd or 0.0), 6),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
    except Exception as exc:  # noqa: BLE001 - spend tracking never fails a run
        _warn_once("record", exc)


def _response_cost(response: Any) -> float:
    """The larger of litellm's two prices for a response, as forecasting_tools'
    LitellmCostTracker takes them: the cost litellm stored on the response,
    and the one it computes from the response's model and usage. Either can
    be missing for a model the cost map lacks; an unpriced call records $0."""
    hidden = getattr(response, "_hidden_params", None) or {}
    stored = hidden.get("response_cost") if isinstance(hidden, dict) else None
    try:
        import litellm

        computed = litellm.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001
        computed = None
    return float(max(stored or 0.0, computed or 0.0))


def _record_response(model: Any, response: Any) -> None:
    try:
        name = str(model or "unknown")
        provider, _, rest = name.partition("/")
        usage = getattr(response, "usage", None)
        if not _logged_first_call:
            _note_first_call(name)
        record(
            provider=provider if rest else "litellm",
            model=rest or name,
            usd=_response_cost(response),
            tokens_in=getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0,
        )
    except Exception as exc:  # noqa: BLE001
        _warn_once("litellm", exc)


_logged_first_call = False


def _note_first_call(model: str) -> None:
    global _logged_first_call
    _logged_first_call = True
    logger.info("spend_first_llm_call model=%s", model)


def _recorded(call):
    async def wrapper(*args, **kwargs):
        response = await call(*args, **kwargs)
        _record_response(kwargs.get("model"), response)
        return response

    wrapper._spend_recorded = True
    return wrapper


def install_litellm_recorder() -> None:
    """Record every call forecasting_tools' GeneralLlm makes through litellm
    (forecasts, parser, summary, fallback research), priced as the SDK prices
    its own final_cost (the response's ``response_cost``).

    Wraps the two litellm functions general_llm.py imports instead of adding
    a litellm success callback: litellm 1.80 runs async callbacks on a
    background logging worker, and on Test Bot run 39570809053 not one of the
    132 OpenRouter calls reached it. The wrapper records in the caller's own
    task, so the question and phase set around the call are the ones seen.
    """
    if ledger_path() is None:
        return
    try:
        from forecasting_tools.ai_models import general_llm

        wrapped = []
        for name in ("acompletion", "aresponses"):
            call = getattr(general_llm, name, None)
            if call is not None and not getattr(call, "_spend_recorded", False):
                setattr(general_llm, name, _recorded(call))
                wrapped.append(name)
        logger.info("spend_recorder_installed wrapped=%s", ",".join(wrapped) or "none")
    except Exception as exc:  # noqa: BLE001
        _warn_once("install", exc)


############################## READING ##############################

def _valid(item: Any) -> bool:
    """One bad line must not break every later summary, so lines without a
    timezone-aware time or with a non-numeric amount are skipped."""
    if not isinstance(item, dict) or not isinstance(item.get("t"), str):
        return False
    try:
        if datetime.fromisoformat(item["t"]).tzinfo is None:
            return False
    except ValueError:
        return False
    return isinstance(item.get("usd", 0.0), (int, float))


def load(path: str | None = None) -> list[dict]:
    path = path or ledger_path()
    if not path or not os.path.exists(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            try:
                item = json.loads(raw)
            except ValueError:
                continue
            if _valid(item):
                lines.append(item)
    return lines


def _when(line: dict) -> datetime:
    return datetime.fromisoformat(line["t"])


def calls(lines: Iterable[dict]) -> list[dict]:
    return [l for l in lines if l.get("kind") is None]


def month_start(now: datetime) -> datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _month(now: datetime) -> str:
    return now.strftime("%Y-%m")


def total(lines: Iterable[dict], provider: str | None = None) -> float:
    return sum(l.get("usd", 0.0) for l in lines if provider is None or l.get("provider") == provider)


def between(lines: Iterable[dict], start: datetime, end: datetime) -> list[dict]:
    return [l for l in lines if start <= _when(l) < end]


def month_spend(lines: list[dict], now: datetime) -> float:
    """Budgeted (Anthropic) dollars recorded since the 1st of now's month."""
    return total(between(calls(lines), month_start(now), now + timedelta(seconds=1)), BUDGET_PROVIDER)


def projection(spent: float, now: datetime, since: datetime) -> float | None:
    """Month-end spend at the pace since ``since``; None before a day of data."""
    days = calendar.monthrange(now.year, now.month)[1]
    elapsed = (now - since).total_seconds() / 86400
    if elapsed < 1:
        return None
    remaining = (month_start(now) + timedelta(days=days) - now).total_seconds() / 86400
    return spent + spent / elapsed * remaining


def history_start(lines: list[dict]) -> datetime | None:
    starts = [_when(l) for l in lines if l.get("kind") == "start"]
    return min(starts) if starts else None


def counted_since(lines: list[dict], now: datetime) -> datetime:
    """When this month's figure starts: the 1st, or later if the history
    began (or was lost and restarted) after it."""
    start = history_start(lines)
    first = month_start(now)
    return max(first, start) if start else first


def over_budget(now: datetime | None = None) -> bool:
    """True once this month's Anthropic spend reaches SPEND_THROTTLE_AT of
    the budget; main.py then runs light research. False on any error."""
    try:
        now = now or datetime.now(timezone.utc)
        return month_spend(load(), now) >= budget_usd(now) * throttle_at()
    except Exception as exc:  # noqa: BLE001
        _warn_once("budget", exc)
        return False


############################## MESSAGES ##############################

def money(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def short_model(name: str) -> str:
    name = name.split("/")[-1]
    return name[len("claude-"):] if name.startswith("claude-") else name


TOURNAMENT_NAMES = {
    "futureeval": "FutureEval", "minibench": "MiniBench", "cup": "Cup",
    "market_pulse": "Market Pulse", "test": "Pruebas",
}
SPANISH_MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def _day(moment: datetime) -> str:
    return f"{moment.day} {SPANISH_MONTHS[moment.month - 1]}"


def _top(lines: list[dict], key, limit: int) -> str:
    sums: dict[str, float] = defaultdict(float)
    for line in lines:
        sums[key(line)] += line.get("usd", 0.0)
    ranked = sorted(sums.items(), key=lambda kv: -kv[1])[:limit]
    return " · ".join(f"{name} {money(usd)}" for name, usd in ranked if usd >= 0.005) or "—"


def forecasts(lines: Iterable[dict]) -> int:
    """Question forecasts paid for: one per question per run."""
    return len({(l.get("run"), l.get("q")) for l in lines if l.get("q") is not None})


def month_line(lines: list[dict], now: datetime) -> str:
    spent = month_spend(lines, now)
    budget = budget_usd(now)
    since = counted_since(lines, now)
    text = f"mes {money(spent)} $ de {money(budget)} $ ({spent / budget:.0%})".replace("%", " %")
    projected = projection(spent, now, since)
    if projected is not None:
        text += f" · proyección {money(projected)} $"
    if since > month_start(now):
        text += f" (cuenta desde el {_day(since)})"
    return text


def weekly_message(lines: list[dict], now: datetime) -> str:
    week = between(calls(lines), now - timedelta(days=7), now + timedelta(seconds=1))
    claude = total(week, BUDGET_PROVIDER)
    other = total(week) - claude
    n = forecasts(week)
    out = [
        f"💸 Gasto API · {_day(now - timedelta(days=7))}–{_day(now - timedelta(days=1))}",
        f"Claude (tu presupuesto): {money(claude)} $ · {month_line(lines, now)}",
        f"OpenRouter (Metaculus): {money(other)} $",
        "Torneos: " + _top(week, lambda l: TOURNAMENT_NAMES.get(l.get("tour"), l.get("tour") or "otros"), 4),
        "Fases: " + _top(week, lambda l: l.get("phase") or "other", 3),
        "Modelos: " + _top(week, lambda l: short_model(l.get("model") or "?"), 3),
    ]
    if n:
        out.append(f"{n} pronósticos · {money((claude + other) / n)} $ de media (Claude {money(claude / n)} $)")
    else:
        out.append("Sin pronósticos esta semana")
    return "\n".join(out)


def run_line(lines: list[dict], run_id: str | None, now: datetime) -> str:
    mine = [l for l in calls(lines) if run_id and l.get("run") == run_id]
    claude = total(mine, BUDGET_PROVIDER)
    other = total(mine) - claude
    n = forecasts(mine)
    each = f", {money((claude + other) / n)} $/pronóstico" if n else ""
    return (f"💸 Esta ejecución: Claude {money(claude)} $ + OpenRouter {money(other)} $ "
            f"({n} pronósticos{each}) · Claude {month_line(lines, now)}")


def alert_message(spent: float, level: float, now: datetime, lines: list[dict]) -> str:
    budget = budget_usd(now)
    head = "🛑" if level >= 1 else "⚠️"
    text = (f"{head} METACULUS BOT · gasto Claude este mes: {money(spent)} $ de "
            f"{money(budget)} $ ({spent / budget:.0%}), pasado el umbral del {level:.0%}.").replace("%", " %")
    projected = projection(spent, now, counted_since(lines, now))
    if projected is not None:
        text += f" Proyección a fin de mes: {money(projected)} $."
    if spent >= budget * throttle_at():
        text += " Desde la próxima ejecución la investigación pasa a modo ligero (6 búsquedas) hasta fin de mes."
    return text


def due_alert(lines: list[dict], now: datetime) -> tuple[float, list[float]] | None:
    """(highest threshold crossed, every crossed one not yet alerted) this
    month, or None. Several crossed at once send one message."""
    spent = month_spend(lines, now)
    month = _month(now)
    sent = {l.get("threshold") for l in lines if l.get("kind") == "alert" and l.get("month") == month}
    crossed = [t for t in thresholds() if spent >= budget_usd(now) * t and t not in sent]
    return (max(crossed), crossed) if crossed else None


############################## END OF JOB ##############################

def _send(text: str) -> bool:
    try:
        done = subprocess.run([sys.executable, NOTIFY_SCRIPT, "whatsapp", "--message", text],
                              timeout=60, check=False)
        return done.returncode == 0
    except Exception as exc:  # noqa: BLE001
        _warn_once("send", exc)
        return False


def _rewrite(path: str, lines: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    os.replace(tmp, path)


def after_run(now: datetime | None = None, send=_send) -> bool:
    """Prune, mark where the history starts, send a due alert, write this
    run's line to the step summary. Returns whether the ledger changed."""
    path = ledger_path()
    if path is None:
        return False
    now = now or datetime.now(timezone.utc)
    lines = load(path)
    run_id = os.environ.get("GITHUB_RUN_ID")
    changed = any(l.get("run") == run_id for l in calls(lines))
    kept = [l for l in lines if _when(l) >= now - timedelta(days=KEEP_DAYS) or l.get("kind") == "start"]
    if history_start(kept) is None:
        first = min((_when(l) for l in lines), default=now)
        kept.insert(0, {"t": first.isoformat(timespec="seconds"), "kind": "start"})
    if kept != lines:
        _rewrite(path, kept)
        lines, changed = kept, True
    due = due_alert(lines, now)
    if due is not None:
        level, crossed = due
        if send(alert_message(month_spend(lines, now), level, now, lines)):
            stamp = now.isoformat(timespec="seconds")
            with open(path, "a", encoding="utf-8") as handle:
                for t in crossed:
                    handle.write(json.dumps({"t": stamp, "kind": "alert", "month": _month(now), "threshold": t}) + "\n")
            lines, changed = load(path), True
    line = run_line(lines, run_id, now)
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return changed


def breakdown(lines: list[dict]) -> str:
    """Calls, tokens and dollars per provider, phase and model -- for Test
    Bot's log, where the question is whether every path is recorded."""
    rows: dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0.0])
    for line in calls(lines):
        row = rows[(line.get("provider"), line.get("phase"), line.get("model"))]
        row[0] += 1
        row[1] += line.get("in", 0)
        row[2] += line.get("out", 0)
        row[3] += line.get("usd", 0.0)
    out = [f"{p} {ph} {m}: {n} calls, {i} in, {o} out, ${usd:.4f}"
           for (p, ph, m), (n, i, o, usd) in sorted(rows.items(), key=lambda kv: -kv[1][3])]
    return "\n".join(out) or "no calls recorded"


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    command = argv[1] if len(argv) > 1 else ""
    try:
        if command == "after-run":
            changed = after_run()
            output = os.environ.get("GITHUB_OUTPUT")
            if output:
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(f"changed={'true' if changed else 'false'}\n")
        elif command == "weekly":
            print(weekly_message(load(), datetime.now(timezone.utc)))
        elif command == "breakdown":
            print(breakdown(load()))
        else:
            print("usage: python spend.py after-run|weekly|breakdown", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001 - never fail the job
        logger.warning("spend_%s_failed reason=%s", command or "cli", type(exc).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
