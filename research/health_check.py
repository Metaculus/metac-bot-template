#!/usr/bin/env python3
"""Health check: questions closed without a forecast, a FutureEval that stopped
opening questions, and a Dagu that stopped dispatching.

Why this exists
---------------
Every coverage loss so far was "not running", never "failed to post" (audit
of 388 production runs, 27 Sep - 23 Oct 2026). The bot forecast 45 of the
first 65 FutureEval questions, and the 20 it missed went unnoticed for days:
7 opened while it still polled the finished summer season (33022, until
1 Oct 21:16Z), 13 fell in GitHub-cron gaps of up to 43 h. Nothing failed,
because "no open questions" is a normal state. Since Dagu on the owner's home
server started dispatching the workflow (14 Oct) it is 29 of 29, median 2.8
min after opening -- which makes that one machine, with no redundancy, the
whole coverage. Cron alone delivers ~6.35 runs a day, roughly 60-69% of
FutureEval's 3-hour questions.

Three checks, and a WhatsApp only when one of them finds something:

  coverage  (daily) questions of FutureEval, MiniBench and Market Pulse whose
            scheduled close fell in the last 24 h and that carry no forecast
            of the bot's.
  idle      (daily) FutureEval with no open question for more than 36 h: a
            stale tournament id, as 33022 was, or the season change on
            6 Jan 2027. 36 and not 24: FutureEval opens ~3.3 questions a
            day, but Metaculus did leave 2-4 Oct without one.
  dagu      (every cron-triggered run) Dagu idle for more than 90 minutes.

The daily pair runs when --marker holds an earlier day than today (UTC), or
with --daily; the workflow keeps the marker in the Actions cache as
health-YYYY-MM-DD. The Dagu check cannot be daily: once Dagu stops, cron runs
are the only ones left, and the first of them is the one that has to say so.

Logs carry ids and counts only -- never a forecast, a community prediction or
anything else a question's page shows: the repository is public.

    coverage_audit tournament=33121 closed=4 missing=1 ids=[46400]
    futureeval_idle hours=40 alert=1
    dagu_silent minutes=120

Read only: Metaculus through MetaculusReadClient (GET only, raises on any
other verb), GitHub through one GET of this workflow's runs with the job's own
token (permissions: actions: read). Standard library only, like the rest of
research/. Never fails the job: each check catches its own errors, and the CLI
exits 0 whatever the checks find or fail on (the workflow step is
continue-on-error as a second layer).

    python research/health_check.py --marker health_check.done
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.metaculus_read_api import MetaculusReadClient, MetaculusReadError  # noqa: E402
from research.weekly_report import _parse, _questions  # noqa: E402

#: Questions whose scheduled close falls this far back are audited each day.
WINDOW = timedelta(hours=24)
FUTUREEVAL_IDLE_HOURS = 36
#: Dagu ticks every 30 minutes and skips a tick while a run is active or one
#: started under 25 minutes ago (all 408 ticks of 15-23 Oct are explained by
#: those two rules), so 90 idle minutes is three ticks missed in a row.
DAGU_SILENT_MINUTES = 90

STATUSES = ("open", "closed", "resolved")
#: 2,000 posts a tournament; FutureEval's whole season is 300-400.
MAX_PAGES = 20
#: The /posts/ listing leaves my_forecasts null, so a question it cannot vouch
#: for is re-read from /posts/<id>/. Bounded per tournament and day; the rest
#: are reported as unchecked, not as missing.
MAX_DETAILS = 60
#: Shown in the WhatsApp message only. ids are listed up to this many.
MAX_IDS = 8

GITHUB_API = "https://api.github.com"
WORKFLOW_FILE = "run_bot_on_tournament.yaml"
#: Enough to reach the last dispatch through ~4 days of cron-only runs; by
#: then the first of them has already alerted.
RUNS_PER_PAGE = 30
#: Dagu's web UI, reachable only from inside the owner's tailnet.
DAGU_URL = "https://gl65.tail415b45.ts.net"
NOTIFY_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "scripts", "notify.py")


############################## COVERAGE ##############################


def forecast_state(question: dict[str, Any]) -> bool | None:
    """True when the bot has a forecast on the question, False when the
    payload says it has none, None when the payload does not say (the
    listing's my_forecasts is null)."""
    mine = question.get("my_forecasts")
    if not isinstance(mine, dict):
        return None
    return bool(mine.get("latest") or mine.get("history"))


@dataclass
class Audit:
    """One tournament's last 24 h. Counts and ids only."""

    tournament: str
    closed: int = 0
    missing: int = 0
    missing_posts: list[int] = field(default_factory=list)
    unchecked: int = 0
    open_now: int = 0
    last_closed: datetime | None = None

    def line(self) -> str:
        text = (f"coverage_audit tournament={self.tournament} closed={self.closed} "
                f"missing={self.missing} ids=[{','.join(str(i) for i in self.missing_posts)}]")
        return text + (f" unchecked={self.unchecked}" if self.unchecked else "")


def audit(client: MetaculusReadClient, tournament: str | int, now: datetime) -> Audit:
    """Questions of one tournament that closed in the last 24 h, and how many
    of them the bot never forecast.

    Per question, not per post: a Market Pulse group stays open while its
    subquestions close one earnings day at a time, which is why open posts
    are listed too. The same listing says whether anything is open now and
    when the latest question closed, which the FutureEval idle check reads.
    """
    params = {"tournaments": tournament, "statuses": list(STATUSES), "order_by": "-published_at"}
    result = Audit(str(tournament))
    pending: dict[int, list[Any]] = {}
    for post in client.iter_posts(params, max_pages=MAX_PAGES):
        for question in _questions(post):
            status = question.get("status") or post.get("status")
            if status == "open":
                result.open_now += 1
                continue
            if status not in ("closed", "resolved"):
                continue
            ended = _parse(question.get("actual_close_time")) or _parse(question.get("scheduled_close_time"))
            if ended is not None and ended <= now and (result.last_closed is None or ended > result.last_closed):
                result.last_closed = ended
            closes = _parse(question.get("scheduled_close_time")) or ended
            if closes is None or not (now - WINDOW <= closes <= now):
                continue
            result.closed += 1
            if forecast_state(question) is not True and isinstance(post.get("id"), int):
                pending.setdefault(post["id"], []).append(question.get("id"))
    missing_posts = set()
    for index, (post_id, question_ids) in enumerate(sorted(pending.items())):
        if index >= MAX_DETAILS:
            result.unchecked += len(question_ids)
            continue
        try:
            detail = client.get_json(f"/posts/{post_id}/")
        except MetaculusReadError:
            result.unchecked += len(question_ids)
            continue
        found = _questions(detail) if isinstance(detail, dict) else []
        by_id = {q.get("id"): q for q in found}
        for question_id in question_ids:
            question = by_id.get(question_id)
            if question is None and len(found) == 1:
                question = found[0]
            state = forecast_state(question or {})
            if state is False:
                result.missing += 1
                missing_posts.add(post_id)
            elif state is None:
                result.unchecked += 1
    result.missing_posts = sorted(missing_posts)
    return result


def idle_hours(futureeval: Audit, now: datetime) -> float | None:
    """Hours since FutureEval last had an open question; 0 while one is
    open, None when the tournament shows no question at all."""
    if futureeval.open_now:
        return 0.0
    if futureeval.last_closed is None:
        return None
    return (now - futureeval.last_closed).total_seconds() / 3600


def _ids(posts: list[int]) -> str:
    shown = ", ".join(str(i) for i in posts[:MAX_IDS])
    return shown + (" …" if len(posts) > MAX_IDS else "")


def daily_message(audits: list[tuple[str, Audit]], futureeval_id: Any,
                  idle: float | None, idle_alert: bool) -> str | None:
    """The WhatsApp for the daily pair, or None when there is nothing to say."""
    lines = []
    gaps = [(label, a) for label, a in audits if a.missing]
    if gaps:
        lines.append("Cerradas en las últimas 24 h sin pronóstico del bot:")
        lines += [f"• {label}: {a.missing} de {a.closed} (post {_ids(a.missing_posts)})" for label, a in gaps]
    if idle_alert:
        if idle is None:
            lines.append(f"FutureEval ({futureeval_id}) no muestra ninguna pregunta.")
        else:
            lines.append(f"FutureEval ({futureeval_id}) lleva {int(idle)} h sin ninguna pregunta abierta.")
        lines.append("¿Id de torneo caducado o cambio de temporada? Revisa tournaments.py.")
    if not lines:
        return None
    return "\n".join(["🩺 METACULUS BOT · revisión diaria", *lines])


def _reason(exc: BaseException) -> str:
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    return type(exc).__name__ + (f" status={status}" if status else "")


def daily(client: MetaculusReadClient, now: datetime, send: Callable[[str], bool],
          log: Callable[[str], None]) -> bool:
    """The coverage audit and the FutureEval idle check. True when the day
    can be marked done: every tournament was read and nothing was due, or
    the alert went out (a partial read that already alerted is not retried,
    so a flaky tournament cannot repeat the message on every run)."""
    import tournaments

    targets = ((tournaments.FUTUREEVAL, "FutureEval"), (tournaments.MINIBENCH, "MiniBench"),
               (tournaments.MARKET_PULSE, "Market Pulse"))
    audits: list[tuple[str, Audit]] = []
    futureeval = None
    complete = True
    for project, label in targets:
        try:
            result = audit(client, project.id, now)
        except Exception as exc:  # noqa: BLE001 - one tournament must not stop the rest
            complete = False
            log(f"health_check_failed part=coverage tournament={project.id} reason={_reason(exc)}")
            continue
        log(result.line())
        audits.append((label, result))
        if project is tournaments.FUTUREEVAL:
            futureeval = result
    idle, idle_alert = None, False
    if futureeval is not None:
        idle = idle_hours(futureeval, now)
        idle_alert = idle is None or idle > FUTUREEVAL_IDLE_HOURS
        shown = "none" if idle is None else str(int(idle))
        log(f"futureeval_idle hours={shown} alert={int(idle_alert)}")
    message = daily_message(audits, tournaments.FUTUREEVAL.id, idle, idle_alert)
    if message is None:
        return complete
    sent = send(message)
    log(f"health_check_alert sent={int(sent)}")
    return sent


############################## DAGU ##############################


def _times(run: dict[str, Any]) -> tuple[datetime | None, datetime | None, datetime | None]:
    """(created, started, ended). A finished run's updated_at is its end."""
    created = _parse(run.get("created_at"))
    started = _parse(run.get("run_started_at")) or created
    ended = _parse(run.get("updated_at")) or started
    return created, started, ended


def dagu_silence(runs: list[dict[str, Any]], current_id: int) -> tuple[float, float] | None:
    """(idle minutes at this run's start, idle minutes at the previous run's
    start), counted from the end of the last workflow_dispatch run; None when
    the listing cannot say (this run or a dispatch is not in it).

    Only time with NO run active counts as idle, because Dagu does not launch
    while one is (138 of 408 ticks skipped for that, 15-23 Oct). A dispatch run
    of 100 minutes on a MiniBench launch day is therefore not 100 minutes of
    silence, and neither is a cron run queued behind it.

    The second number makes the alert fire once per silence, on the first
    cron-triggered run past the threshold: it is 0 when the previous run was
    the dispatch itself, and above the threshold once an earlier cron run has
    already alerted.
    """
    current = next((r for r in runs if r.get("id") == current_id), None)
    if current is None:
        return None
    created, started, _ended = _times(current)
    if created is None or started is None:
        return None
    earlier = [r for r in runs if r.get("id") != current_id
               and _times(r)[0] is not None and _times(r)[0] <= created]
    earlier.sort(key=lambda r: _times(r)[0], reverse=True)
    since_dispatch = []
    dispatch = None
    for run in earlier:
        if run.get("event") == "workflow_dispatch":
            dispatch = run
            break
        since_dispatch.append(run)
    if dispatch is None or _times(dispatch)[2] is None:
        return None
    busy_until = _times(dispatch)[2]
    idle = before = 0.0
    for run in sorted(since_dispatch, key=lambda r: _times(r)[1] or created) + [current]:
        _created, run_started, run_ended = _times(run)
        if run is current:
            before = idle
        if run_started is not None:
            idle += max(0.0, (run_started - busy_until).total_seconds() / 60)
        if run is not current and run_ended is not None:
            busy_until = max(busy_until, run_ended)
    return idle, before


def fetch_runs(repo: str, token: str, opener: Callable[..., Any] = urllib.request.urlopen) -> list[dict]:
    """This workflow's most recent runs, newest first. One GET."""
    url = f"{GITHUB_API}/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs?per_page={RUNS_PER_PAGE}"
    request = urllib.request.Request(url, method="GET", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "seergiii-bot-health-check",
    })
    with opener(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    runs = data.get("workflow_runs") if isinstance(data, dict) else None
    return [r for r in runs or [] if isinstance(r, dict)]


def dagu_message(minutes: float) -> str:
    return (f"⏰ METACULUS BOT · Dagu lleva {int(minutes)} min sin lanzar el bot; "
            f"solo quedan los eventos de cron de GitHub. Revísalo: {DAGU_URL}")


def dagu(environ: Any, fetch: Callable[[str, str], list[dict]], send: Callable[[str], bool],
         log: Callable[[str], None]) -> None:
    token = environ.get("GITHUB_TOKEN", "")
    repo = environ.get("GITHUB_REPOSITORY", "")
    run_id = environ.get("GITHUB_RUN_ID", "")
    if not (token and repo and run_id.isdigit()):
        log("dagu_check skipped reason=no_github_context")
        return
    silence = dagu_silence(fetch(repo, token), int(run_id))
    if silence is None:
        log("dagu_check skipped reason=no_dispatch_or_run_listed")
        return
    minutes, before = silence
    alert = minutes > DAGU_SILENT_MINUTES >= before
    log(f"dagu_check idle_minutes={int(minutes)} alert={int(alert)}")
    if alert:
        log(f"dagu_silent minutes={int(minutes)}")
        log(f"dagu_alert sent={int(send(dagu_message(minutes)))}")


############################## CLI ##############################


def _send(text: str) -> bool:
    try:
        done = subprocess.run([sys.executable, NOTIFY_SCRIPT, "whatsapp", "--message", text],
                              timeout=60, check=False)
        return done.returncode == 0
    except Exception:  # noqa: BLE001 - a failed alert never fails the job
        return False


def _marker_day(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _log(text: str) -> None:
    print(text, flush=True)


def run(argv: list[str] | None = None, *, now: datetime | None = None,
        client: MetaculusReadClient | None = None,
        fetch: Callable[[str, str], list[dict]] = fetch_runs,
        send: Callable[[str], bool] = _send, environ: Any = None,
        log: Callable[[str], None] = _log) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--marker", help="file holding the last UTC day the daily checks ran")
    ap.add_argument("--daily", action="store_true", help="run the daily checks whatever the marker says")
    args = ap.parse_args(argv)
    environ = os.environ if environ is None else environ
    now = now or datetime.now(timezone.utc)
    today = now.date().isoformat()

    if args.daily or (args.marker and _marker_day(args.marker) != today):
        token = environ.get("METACULUS_TOKEN", "")
        done = False
        if client is None and not token.strip():
            log("health_check daily=skipped reason=no_metaculus_token")
        else:
            try:
                done = daily(client or MetaculusReadClient(token, timeout=30, max_retries=3),
                             now, send, log)
            except Exception as exc:  # noqa: BLE001
                log(f"health_check_failed part=daily reason={_reason(exc)}")
        if done and args.marker:
            try:
                with open(args.marker, "w", encoding="utf-8") as handle:
                    handle.write(today + "\n")
                output = environ.get("GITHUB_OUTPUT")
                if output:
                    with open(output, "a", encoding="utf-8") as handle:
                        handle.write(f"daily_key=health-{today}\n")
            except OSError as exc:
                log(f"health_check_failed part=marker reason={_reason(exc)}")
    else:
        log("health_check daily=skipped reason=already_ran_today")

    if environ.get("GITHUB_EVENT_NAME") == "schedule":
        try:
            dagu(environ, fetch, send, log)
        except Exception as exc:  # noqa: BLE001
            log(f"health_check_failed part=dagu reason={_reason(exc)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except SystemExit as exc:  # argparse on a bad flag
        return 0 if exc.code in (0, None) else 2
    except Exception as exc:  # noqa: BLE001 - never fail the job
        print(f"health_check_failed part=cli reason={type(exc).__name__}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
