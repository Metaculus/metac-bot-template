"""GAAP diluted EPS against the analyst consensus, for EPS questions.

Why this exists
---------------
Market Pulse 26Q4's EPS subquestions (group 45995) resolve on GAAP diluted
EPS as first reported, while the consensus the research finds -- the figure
every earnings preview quotes -- is the street one, usually non-GAAP. The two
can differ a lot: Tesla's Q3 2026 resolved at a GAAP 0.22 against a 0.451
consensus, and Meta's GAAP 3.83 landed below a 5.345-7.345 range. For some
companies the gap repeats quarter after quarter (stock-based compensation,
amortisation of acquired intangibles); for others it comes from one-off
items or from revaluing investments, which no history predicts.

A backtest on 116 company-quarters (2021-2026, run 2026-10-23) of "consensus
plus the company's median past GAAP-minus-consensus gap" against a
consensus-centred normal (sd 0.35) gained +36 mean and +26 median
(100 x ln density at the outcome), nearly all of it on the companies whose
gap is stable; on those whose gap comes from investment gains it gained
nothing or lost.

What it adds -- phase 1, text in the Live data block only
  * the consensus for the quarter being forecast (Yahoo earningsTrend "0q":
    mean, analyst count, low and high);
  * GAAP diluted EPS for the last quarters (Yahoo fundamentals-timeseries,
    quarterlyDilutedEPS, no crumb needed) and, for each quarter, GAAP minus
    the consensus at the time (earningsHistory's epsEstimate -- its epsActual
    is the street figure and is never used);
  * a prior P10/P50/P90 only when the gap is stable: its MAD (normal-
    consistent, 1.4826 x the median absolute deviation) below 0.35 x |median
    gap| and the same sign in at least three quarters of every four (the
    roadmap's "6 of 8"; Yahoo gives 4). On the 2026-10-23 data that holds
    for two of the seven companies backtested.

Nothing here moves a forecast by itself: the forecasters read it beside the
research. A CDF blend is roadmap B1 phase 2 and needs its own backtest.

Yahoo's consensus sits behind a cookie and a crumb (fc.yahoo.com answers 404
but sets the A3 cookie, then /v1/test/getcrumb returns the crumb). That works
from a home connection and is untested on GitHub's runners; when it fails the
section shows the GAAP history alone.

Never raises; costs no Claude money. Logs carry flags and counts only -- no
EPS figure, consensus or prior, because the repository is public. Standard
library only.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import math
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

TIMEOUT = 10
UA = "Mozilla/5.0 (seergiii-bot research; +https://github.com/BySergiMM/metac-bot-template)"

Fetch = Callable[[str, dict], bytes]

COOKIE_URL = "https://fc.yahoo.com/"
CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
TIMESERIES_URL = "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/"
CRUMB = re.compile(r"[^\s<>{}\"']{4,64}")

#: Gap stability, as set in the roadmap (B1, 2026-10-23).
STABLE_MAD = 0.35
SAME_SIGN = 0.75
MIN_GAPS = 4
MAX_GAPS = 8
#: Floors on the prior's spread (as in the backtest): 3% of the consensus or
#: 2 cents, whichever is larger, however tight the history looks.
FLOOR_FRAC = 0.03
FLOOR_ABS = 0.02
#: Two feeds date the same fiscal quarter a few days apart when it ends on a
#: Saturday (Apple's late-September quarter); within this many days they are
#: the same quarter.
QUARTER_SLACK_DAYS = 20


def _session() -> Fetch:
    """A fetch with its own cookie jar, one per question: the crumb is tied
    to the A3 cookie, so two questions sharing a jar could swap cookies under
    each other's crumbs."""
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def fetch(url: str, headers: dict) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
        try:
            with opener.open(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https hosts
                return resp.read(600_000)
        except urllib.error.HTTPError as exc:
            if url == COOKIE_URL and exc.code == 404:
                return b""  # answers 404 and sets the cookie all the same
            raise

    return fetch


def _json(fetch: Fetch, url: str) -> Any:
    return json.loads(fetch(url, {}).decode("utf-8", "replace"))


# ------------------------------------------------------------------ which question

EPS = re.compile(r"\bearnings per share\b|\bEPS\b", re.I)
GAAP = re.compile(r"(?<!non-)(?<!non )\bGAAP\b", re.I)


def kind(question: Any) -> str | None:
    """The question's kind: "eps_gaap" when it resolves on GAAP earnings per share, else None."""
    text = "\n".join(str(getattr(question, a, "") or "")
                     for a in ("question_text", "resolution_criteria", "fine_print"))
    if EPS.search(text) and GAAP.search(text):
        return "eps_gaap"
    return None


#: Companies by the name a group option uses ("Meta", "Nvidia"). US-listed,
#: reporting US GAAP in dollars -- a foreign filer's EPS would not compare.
COMPANIES = {
    "advanced micro devices": "AMD", "amd": "AMD", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "apple": "AAPL", "broadcom": "AVGO", "intel": "INTC",
    "meta platforms": "META", "meta": "META", "microsoft": "MSFT", "netflix": "NFLX",
    "nvidia": "NVDA", "oracle": "ORCL", "palantir": "PLTR", "salesforce": "CRM",
    "tesla": "TSLA",
}
TICKER = re.compile(r"\(([A-Z]{1,5})\)")
NOT_TICKERS = {"EPS", "GAAP", "USD", "SEC", "US", "USA", "AI", "IPO", "ADR", "ADS"}


def ticker(question: Any) -> str | None:
    """The one company the question is about, from its group option (the
    group title names no company) or else its text; None when it names none
    or several."""
    option = str(getattr(question, "group_question_option", "") or "").strip()
    text = option or str(getattr(question, "question_text", "") or "")
    found = {t for t in TICKER.findall(text) if t not in NOT_TICKERS}
    low = text.lower()
    found |= {sym for name, sym in COMPANIES.items() if re.search(rf"\b{re.escape(name)}\b", low)}
    return next(iter(found)) if len(found) == 1 else None


# ------------------------------------------------------------------------ data

def _day(raw: Any) -> date | None:
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _raw(field: Any) -> float | None:
    value = field.get("raw") if isinstance(field, dict) else field
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def gaap_history(symbol: str, fetch: Fetch, now: datetime) -> list[tuple[date, float]]:
    """GAAP diluted EPS by fiscal quarter end, oldest first (Yahoo gives ~5)."""
    end = int(now.timestamp())
    data = _json(fetch, TIMESERIES_URL + urllib.parse.quote(symbol) + "?" + urllib.parse.urlencode(
        {"type": "quarterlyDilutedEPS", "period1": end - 3 * 366 * 86400, "period2": end}))
    rows: dict[date, float] = {}
    for result in (data.get("timeseries") or {}).get("result") or []:
        for row in (result or {}).get("quarterlyDilutedEPS") or []:
            if not isinstance(row, dict) or (row.get("periodType") or "3M") != "3M" \
                    or (row.get("currencyCode") or "USD") != "USD":
                continue
            day, value = _day(row.get("asOfDate")), _raw(row.get("reportedValue"))
            if day and value is not None:
                rows[day] = value
    return sorted(rows.items())


def street(symbol: str, fetch: Fetch) -> dict | None:
    """Yahoo's earningsTrend and earningsHistory, or None when the crumb is refused."""
    try:
        fetch(COOKIE_URL, {})
    except Exception:  # noqa: BLE001 - the crumb request below says whether it worked
        pass
    crumb = fetch(CRUMB_URL, {}).decode("utf-8", "replace").strip()
    if not CRUMB.fullmatch(crumb):
        return None
    data = _json(fetch, SUMMARY_URL + urllib.parse.quote(symbol) + "?" + urllib.parse.urlencode(
        {"modules": "earningsTrend,earningsHistory", "crumb": crumb}))
    result = ((data.get("quoteSummary") or {}).get("result") or [None])[0]
    return result if isinstance(result, dict) else None


def consensus(summary: dict) -> dict | None:
    """The current quarter's ("0q") EPS consensus."""
    for row in (summary.get("earningsTrend") or {}).get("trend") or []:
        if not isinstance(row, dict) or row.get("period") != "0q":
            continue
        est = row.get("earningsEstimate") or {}
        avg, end = _raw(est.get("avg")), _day(row.get("endDate"))
        if avg is None or end is None:
            return None
        analysts = _raw(est.get("numberOfAnalysts"))
        return {"end": end, "avg": avg, "analysts": int(analysts) if analysts else None,
                "low": _raw(est.get("low")), "high": _raw(est.get("high"))}
    return None


def past_estimates(summary: dict) -> list[tuple[date, float]]:
    """The consensus each past quarter had when it was reported."""
    out = []
    for row in (summary.get("earningsHistory") or {}).get("history") or []:
        if not isinstance(row, dict):
            continue
        quarter = row.get("quarter")
        day = _day(quarter.get("fmt")) if isinstance(quarter, dict) else None
        if day is None and _raw(quarter) is not None:
            day = datetime.fromtimestamp(_raw(quarter), timezone.utc).date()
        estimate = _raw(row.get("epsEstimate"))
        if day and estimate is not None:
            out.append((day, estimate))
    return sorted(out)


def gaps(gaap: Sequence[tuple[date, float]],
         estimates: Sequence[tuple[date, float]]) -> list[tuple[date, float]]:
    """GAAP minus the consensus of the time, per quarter both feeds have, oldest first."""
    out = []
    for day, estimate in sorted(estimates):
        same = [value for d, value in gaap if abs((d - day).days) <= QUARTER_SLACK_DAYS]
        if len(same) == 1:
            out.append((day, same[0] - estimate))
    return out[-MAX_GAPS:]


# ----------------------------------------------------------------------- prior

def _mad(values: Sequence[float]) -> float:
    centre = statistics.median(values)
    return 1.4826 * statistics.median([abs(v - centre) for v in values])


def stable(values: Sequence[float]) -> bool:
    """The gap repeats: small spread around its median, and the same sign
    in at least three quarters of every four."""
    if len(values) < MIN_GAPS:
        return False
    centre = statistics.median(values)
    if centre == 0:
        return False
    same_sign = sum(1 for v in values if v * centre > 0)
    return _mad(values) < STABLE_MAD * abs(centre) and same_sign >= math.ceil(SAME_SIGN * len(values))


def _t_cdf(z: float, df: int) -> float:
    """Student-t CDF in closed form, for 2 or 4 degrees of freedom."""
    if df == 2:
        return 0.5 + z / (2 * math.sqrt(2 + z * z))
    return 0.5 + 0.5 * (z / math.sqrt(4 + z * z)) * (1 + 2 / (4 + z * z))


def _quantile(parts: Sequence[tuple[float, float, float, int]], p: float) -> float:
    lo = min(m - 100 * s for _, m, s, _ in parts)
    hi = max(m + 100 * s for _, m, s, _ in parts)
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(w * _t_cdf((mid - m) / s, df) for w, m, s, df in parts) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def prior(consensus_eps: float, values: Sequence[float]) -> tuple[float, float, float]:
    """P10, P50, P90 of GAAP EPS: consensus plus the typical gap (half the
    last four quarters' median, half the whole history's), a t(4) core as wide
    as the gaps' spread (floored), and 5% of a much wider t(2) for one-off
    items -- more when the history already holds some. The backtest's shape."""
    centre_gap = statistics.median(values)
    centre = consensus_eps + 0.5 * statistics.median(values[-4:]) + 0.5 * centre_gap
    mad = _mad(values)
    spread = max(mad, FLOOR_FRAC * abs(consensus_eps), FLOOR_ABS)
    outliers = sum(1 for v in values if abs(v - centre_gap) > 3 * max(mad, 1e-9))
    w_wide = min(0.35, 0.05 + outliers / len(values)) if outliers else 0.05
    wide = max(statistics.pstdev(values), 3 * spread, 0.1 * abs(consensus_eps))
    parts = [(1 - w_wide, centre, spread, 4), (w_wide, centre, wide, 2)]
    p10, p50, p90 = (_quantile(parts, p) for p in (0.1, 0.5, 0.9))
    return p10, p50, p90


# --------------------------------------------------------------------- section

def _right_quarter(cons: dict, reported: Sequence[date], question: Any) -> bool:
    """The consensus is for the quarter the question is about: one later
    than any quarter either feed shows as reported, and ended by the time the
    question closes."""
    if reported and cons["end"] <= max(reported) + timedelta(days=QUARTER_SLACK_DAYS):
        return False
    close = getattr(question, "close_time", None)
    return not (isinstance(close, datetime) and cons["end"] > close.date())


def _near_range(value: float, question: Any) -> bool:
    """A consensus far outside the question's range means the wrong company
    or share class, not a surprise (the range is centred near consensus)."""
    lo, hi = getattr(question, "lower_bound", None), getattr(question, "upper_bound", None)
    if not all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in (lo, hi)) or hi <= lo:
        return True
    return lo - 0.5 * (hi - lo) <= value <= hi + 0.5 * (hi - lo)


def _section(question: Any, fetch: Fetch | None, now: datetime) -> str:
    if kind(question) != "eps_gaap":
        return ""
    symbol = ticker(question)
    if not symbol:
        logger.info("market_quant eps_gaap mapped=0")
        return ""
    fetch = fetch or _session()
    try:
        gaap = gaap_history(symbol, fetch, now)
    except Exception as exc:  # noqa: BLE001 - the consensus may still come
        logger.warning("market_quant gaap_history_failed reason=%s", type(exc).__name__)
        gaap = []
    try:
        summary = street(symbol, fetch)
    except Exception as exc:  # noqa: BLE001 - the GAAP history alone still helps
        logger.warning("market_quant consensus_failed reason=%s", type(exc).__name__)
        summary = None
    cons = consensus(summary) if summary else None
    if cons and not _near_range(cons["avg"], question):
        logger.info("market_quant eps_gaap mapped=1 range_check=0")
        return ""
    estimates = past_estimates(summary) if summary else []
    if cons and not _right_quarter(cons, [d for d, _ in gaap] + [d for d, _ in estimates], question):
        cons = None
    history = gaps(gaap, estimates)
    values = [g for _, g in history]
    is_stable = stable(values)
    give_prior = bool(cons) and is_stable

    lines = [f"- Yahoo symbol {symbol}. GAAP diluted EPS is what resolves this question. The analyst "
             "consensus, like most EPS figures in earnings previews and the news, is the street "
             "figure, which usually leaves out items GAAP keeps: stock-based compensation, "
             "amortisation of acquired intangibles, gains and losses on investments, one-off tax items."]
    if cons:
        extra = [f"{cons['analysts']} analysts"] if cons["analysts"] else []
        if cons["low"] is not None and cons["high"] is not None:
            extra.append(f"low {cons['low']:.2f}, high {cons['high']:.2f}")
        lines.append(f"- Consensus EPS for the quarter ending {cons['end']}: {cons['avg']:.3f}"
                     + (f" ({'; '.join(extra)})" if extra else "") + " [Yahoo earningsTrend, now]")
    if gaap:
        lines.append("- GAAP diluted EPS, by quarter end: "
                     + "; ".join(f"{d} {v:.2f}" for d, v in gaap[-MAX_GAPS:])
                     + " [Yahoo fundamentals-timeseries]")
    if values:
        lines.append("- GAAP minus the consensus at the time, oldest first: "
                     + "; ".join(f"{d} {g:+.2f}" for d, g in history)
                     + f"; median {statistics.median(values):+.2f} [Yahoo earningsHistory]")
    if is_stable:
        same = sum(1 for v in values if v * statistics.median(values) > 0)
        text = (f"- The gap has been stable (same sign in {same} of {len(values)} quarters, little "
                "spread around its median).")
        if give_prior:
            p10, p50, p90 = prior(cons["avg"], values)
            text += (f" Prior from the consensus plus that gap: P10 {p10:.2f}, P50 {p50:.2f}, "
                     f"P90 {p90:.2f}. Move its centre only for dated news the history cannot know "
                     "(one-off charges, gains or losses on investments, tax items).")
        lines.append(text)
    elif 0 < len(values) < MIN_GAPS:
        lines.append(f"- Only {len(values)} quarter(s) where both feeds overlap: too few to call the gap "
                     "stable, so no prior is given.")
    elif values:
        lines.append(f"- The gap has not been stable, so no prior is given: GAAP can land far from the "
                     f"consensus on one-off items (past gaps ran from {min(values):+.2f} to "
                     f"{max(values):+.2f}). Keep a wide tail on the side such items would push.")
    logger.info("market_quant eps_gaap mapped=1 gaap_quarters=%d consensus=%d gap_quarters=%d "
                "stable=%d prior=%d", len(gaap), int(bool(cons)), len(values), int(is_stable),
                int(give_prior))
    if not gaap and not cons:
        return ""
    return "\n".join(lines)


def section(question: Any, fetch: Fetch | None = None, now: datetime | None = None) -> str:
    """The eps_gaap text for the Live data block, or "". Never raises.

    `fetch` is injected by the tests; None opens a cookie-keeping session.
    """
    try:
        return _section(question, fetch, now or datetime.now(timezone.utc))
    except Exception as exc:  # noqa: BLE001 - optional context never fails a run
        logger.warning("market_quant eps_gaap_failed reason=%s", type(exc).__name__)
        return ""
