#!/usr/bin/env python3
"""Monday WhatsApp report: standings per tournament, compared with last week.

Replaces the per-question WhatsApp alerts (one message every time a question
opened, which told the owner nothing about how the bot was doing). Sent once
a week, Monday 09:00 Madrid, dispatched by Dagu on gl65.

What it reports, per tournament the bot plays:
  * leaderboard position and total score, with the change since last week
    in positions and points -- never as a percentage: scores go negative,
    and -10 -> +5 as "-150%" would read as a collapse;
  * when Metaculus has no leaderboard yet, what can be measured instead:
    questions forecast and resolved this week.

Read-only by construction: everything goes through MetaculusReadClient
(GET only, raises on any other verb) with the bot's own token. The previous
week's numbers come from a small JSON file the workflow keeps in the Actions
cache; a missing file just means "no comparison this time".

Standard library only, like the rest of research/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.metaculus_read_api import MetaculusReadClient, MetaculusReadError  # noqa: E402

BOT_USER_ID = 306913
BOT_USERNAME = "seergiii-bot"

#: MiniBench rounds run back to back every 14 days. A finished round is
#: renamed to "minibench-YYYY-MM-DD"; the live one is just "minibench".
MINIBENCH_EPOCH = date(2026, 8, 10)
MINIBENCH_DAYS = 14


@dataclass(frozen=True)
class Target:
    key: str        # stable key in the state file
    label: str      # shown in the message
    project: str    # numeric id or slug


def targets(today: date) -> list[Target]:
    import tournaments

    rounds = (today - MINIBENCH_EPOCH).days // MINIBENCH_DAYS
    last_done = MINIBENCH_EPOCH + timedelta(days=MINIBENCH_DAYS * (rounds - 1))
    return [
        Target("futureeval", "FutureEval ($50k)", str(tournaments.FUTUREEVAL.id)),
        Target("minibench", "MiniBench (ronda actual)", "minibench"),
        Target("minibench_prev", f"MiniBench {last_done:%d-%b}",
               f"minibench-{last_done.isoformat()}"),
        Target("market_pulse", "Market Pulse ($7,5k)", str(tournaments.MARKET_PULSE.id)),
        Target("cup", "Metaculus Cup", str(tournaments.METACULUS_CUP.id)),
    ]


# ------------------------------------------------------------- leaderboard


def project_id(client: MetaculusReadClient, project: str) -> int | None:
    if project.isdigit():
        return int(project)
    try:
        data = client.get_json(f"/projects/tournaments/{project}/")
    except MetaculusReadError:
        return None
    pid = data.get("id") if isinstance(data, dict) else None
    return int(pid) if isinstance(pid, int) else None


def _entry_user(entry: dict[str, Any]) -> tuple[Any, Any]:
    user = entry.get("user") or {}
    if isinstance(user, dict):
        return user.get("id"), user.get("username")
    return entry.get("user_id"), entry.get("username")


def standing_from(payload: Any) -> dict[str, Any] | None:
    """{rank, total, score} for the bot, or None when it is not listed.

    Defensive about shape: the entries list may sit under "entries" or
    "results", the bot may also be echoed as "userEntry", and rows Metaculus
    excludes from prizes (its own reference bots) do not count towards the
    total, matching what the page shows as numbered ranks.
    """
    if isinstance(payload, list):
        # The endpoint answers with every leaderboard of the project (seen
        # 2026-10-17: a one-element list); the primary one is what the page
        # shows.
        boards = [b for b in payload if isinstance(b, dict)]
        payload = next((b for b in boards if b.get("is_primary_leaderboard")),
                       boards[0] if boards else None)
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries") or payload.get("results") or []
    # The total is the last numbered position, as the page shows it: the
    # entries list also carries unranked and excluded rows (236 entries
    # against 136 positions on minibench-2026-09-21).
    ranks = [e.get("rank") for e in entries
             if isinstance(e, dict) and not e.get("excluded") and isinstance(e.get("rank"), int)]
    mine = None
    for e in entries:
        uid, name = _entry_user(e) if isinstance(e, dict) else (None, None)
        if uid == BOT_USER_ID or name == BOT_USERNAME:
            mine = e
            break
    if mine is None and isinstance(payload.get("userEntry"), dict):
        mine = payload["userEntry"]
    if mine is None or mine.get("rank") is None:
        return None
    score = mine.get("score")
    return {
        "rank": int(mine["rank"]),
        "total": max(ranks) if ranks else None,
        "score": round(float(score), 1) if score is not None else None,
    }


def fetch_standing(client: MetaculusReadClient, project: str) -> tuple[dict | None, str]:
    """(standing, note). The note says why there is none, for the log."""
    pid = project_id(client, project)
    if pid is None:
        return None, "project not found"
    try:
        payload = client.get_json(f"/leaderboards/project/{pid}/")
    except MetaculusReadError as exc:
        return None, f"leaderboard unavailable ({getattr(exc, 'status', None)})"
    standing = standing_from(payload)
    return standing, "ok" if standing else "bot not on leaderboard"


# ---------------------------------------------------------------- activity


def _parse(ts: Any) -> datetime | None:
    if ts in (None, ""):
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _post_projects(post: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    projects = post.get("projects") or {}
    groups: Iterable[Any] = []
    if isinstance(projects, dict):
        groups = [projects.get("tournament") or [], projects.get("question_series") or [],
                  [projects.get("default_project")] if projects.get("default_project") else []]
    for group in groups:
        for p in group or []:
            if isinstance(p, dict):
                for k in ("id", "slug"):
                    if p.get(k) is not None:
                        out.add(str(p[k]))
    return out


def _questions(post: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(post.get("question"), dict):
        return [post["question"]]
    group = post.get("group_of_questions") or {}
    return [q for q in (group.get("questions") or []) if isinstance(q, dict)]


def activity(posts: list[dict[str, Any]], project_keys: set[str],
             since: datetime) -> dict[str, int]:
    """Questions forecast and resolved since ``since`` in one tournament."""
    forecast = resolved = 0
    for post in posts:
        if not (_post_projects(post) & project_keys):
            continue
        for q in _questions(post):
            latest = ((q.get("my_forecasts") or {}).get("latest") or {})
            started = _parse(latest.get("start_time"))
            if started and started >= since:
                forecast += 1
            done = _parse(q.get("actual_resolve_time"))
            if done and done >= since and q.get("resolution") not in (None, ""):
                resolved += 1
    return {"forecast": forecast, "resolved": resolved}


# ----------------------------------------------------------------- message


def _delta_rank(now: int, before: int | None) -> str:
    if before is None:
        return ""
    moved = before - now
    return " (=)" if moved == 0 else f" ({'▲' if moved > 0 else '▼'}{abs(moved)})"


def _delta_score(now: float | None, before: float | None) -> str:
    if now is None or before is None:
        return ""
    diff = round(now - before, 1)
    return " (=)" if diff == 0 else f" ({'▲' if diff > 0 else '▼'}{abs(diff):g})"


def _fmt_score(score: float | None) -> str:
    return "?" if score is None else f"{score:+g} pts"


def render(week_start: date, week_end: date, rows: list[dict[str, Any]],
           previous: dict[str, Any] | None) -> str:
    prev = (previous or {}).get("standings", {})
    lines = [f"📊 Bot Metaculus · semana {week_start:%d/%m}–{week_end:%d/%m}", ""]
    for row in rows:
        s, act = row.get("standing"), row.get("activity") or {}
        if s:
            p = prev.get(row["key"]) or {}
            total = f"/{s['total']}" if s.get("total") else ""
            lines.append(
                f"• {row['label']}: puesto {s['rank']}{total}"
                f"{_delta_rank(s['rank'], p.get('rank'))} · "
                f"{_fmt_score(s.get('score'))}{_delta_score(s.get('score'), p.get('score'))}"
            )
        else:
            lines.append(f"• {row['label']}: sin clasificación aún")
        if act:
            lines.append(f"   esta semana: {act.get('forecast', 0)} pronosticadas, "
                         f"{act.get('resolved', 0)} resueltas")
    if not previous:
        lines += ["", "(primera semana: sin comparativa todavía)"]
    return "\n".join(lines)


# -------------------------------------------------------------------- main


def build(client: MetaculusReadClient, today: date,
          log: Callable[[str], None] = print) -> tuple[list[dict[str, Any]], date, date]:
    week_end = today - timedelta(days=1)
    week_start = today - timedelta(days=7)
    since = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc)
    posts = client.get_posts_forecasted_by(BOT_USER_ID)
    log(f"posts_forecasted_by_bot={len(posts)}")
    # The listing carries resolutions but not the bot's own forecasts
    # (my_forecasts is null there); those come only from /posts/<id>/. Fetch
    # just the posts that can hold a forecast made this week: still open, or
    # closed since the week began.
    detailed = 0
    for i, post in enumerate(posts):
        closed = _parse(post.get("actual_close_time"))
        if post.get("status") == "open" or (closed and closed >= since):
            try:
                posts[i] = client.get_json(f"/posts/{post['id']}/")
                detailed += 1
            except MetaculusReadError:
                pass
    log(f"posts_detailed={detailed}")
    rows = []
    for t in targets(today):
        standing, note = fetch_standing(client, t.project)
        keys = {t.project}
        pid = project_id(client, t.project) if not t.project.isdigit() else int(t.project)
        if pid is not None:
            keys.add(str(pid))
        act = activity(posts, keys, since)
        log(f"target={t.key} project={t.project} standing={standing} note={note} activity={act}")
        rows.append({"key": t.key, "label": t.label, "standing": standing, "activity": act})
    return rows, week_start, week_end


def _shape(value: Any, depth: int = 0) -> Any:
    """Keys and types only, two levels deep, plus list lengths: enough to
    see how a payload is laid out without dumping it."""
    if depth > 2:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: _shape(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [f"len={len(value)}", _shape(value[0], depth + 1)] if value else ["len=0"]
    return type(value).__name__


def probe(client: MetaculusReadClient, today: date) -> None:
    """Print the layout of the payloads this report reads (dry runs only)."""
    for t in targets(today):
        pid = project_id(client, t.project)
        print(f"PROBE project {t.project} -> id {pid}")
        if pid is None:
            continue
        try:
            payload = client.get_json(f"/leaderboards/project/{pid}/")
        except MetaculusReadError as exc:
            print(f"PROBE leaderboard {pid}: error status={getattr(exc, 'status', None)}")
            continue
        print(f"PROBE leaderboard {pid} shape: {json.dumps(_shape(payload))[:1500]}")
        entries = (payload.get("entries") or payload.get("results") or []) if isinstance(payload, dict) else []
        mine = [e for e in entries if BOT_USERNAME in json.dumps(e)]
        print(f"PROBE leaderboard {pid} bot rows: {json.dumps(mine)[:600]}")
    posts = client.get_posts_forecasted_by(BOT_USER_ID)
    if posts:
        print(f"PROBE post shape: {json.dumps(_shape(posts[0]))[:2500]}")
        q = (_questions(posts[0]) or [{}])[0]
        print(f"PROBE question keys: {sorted(q.keys())}")
        print(f"PROBE my_forecasts: {json.dumps(q.get('my_forecasts'))[:400]}")
        print(f"PROBE projects: {json.dumps(posts[0].get('projects'))[:600]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True, help="last week's JSON; rewritten unless --dry-run")
    ap.add_argument("--out", required=True, help="where to write the message text")
    ap.add_argument("--dry-run", action="store_true", help="do not update --state")
    args = ap.parse_args(argv)

    token = os.environ.get("METACULUS_TOKEN", "")
    client = MetaculusReadClient(token)
    today = datetime.now(timezone.utc).date()
    if args.dry_run:
        probe(client, today)
    rows, week_start, week_end = build(client, today)

    previous = None
    if os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as handle:
            previous = json.load(handle)
    message = render(week_start, week_end, rows, previous)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(message + "\n")
    print(message)

    if not args.dry_run:
        state = {"week_end": week_end.isoformat(),
                 "standings": {r["key"]: r["standing"] for r in rows if r["standing"]}}
        with open(args.state, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
