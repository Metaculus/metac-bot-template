"""Live data appended to the research before the forecasters read it.

Why this exists
---------------
Comparing the bot with the MiniBench 2026-09-21 top 10 (2026-10-18): the top
bots read the number a question is about instead of searching the news for
it. MiniBench generates many questions from data series -- prices, official
statistics, or another Metaculus question's community prediction -- and the
open/self-described top bots pull those directly (nostreambot: yfinance and
FRED plus a Polymarket/Kalshi/Manifold snapshot; Ora and GreeneiBot2 fetch
the pages named in the resolution criteria). Metaculus' Fall 2025 survey also
found winners checking prediction markets and similar questions (34% vs 0%).

What it adds, all free, keyless or the bot's own token, GET only:
  * Metaculus questions linked from the question: their current community
    prediction (bot token, the same read the SDK already does);
  * crypto spot prices (CoinGecko) and stock/index closes with a one-month
    range (Yahoo chart API), for assets the question names;
  * for a question resolving on GAAP diluted EPS: the analyst consensus and
    the company's history of GAAP results against it (market_quant);
  * possibly related prediction markets (Polymarket, Kalshi, Manifold), labelled as
    such: the forecaster must judge whether they match the question;
  * the text of up to three pages linked from the resolution criteria.

Hygiene (2026-10-23)
--------------------
The prompts present this block as outranking the research, so a wrong number
here is a wrong anchor. Test Bot run 39813843358 showed every kind: a
Dogecoin price on a federal-outlays question whose background mentions the
DOGE agency; Polymarket markets resolved years ago shown at 100/0; Wikipedia
and Wayback pages whose first 1,250 characters were page chrome; "earnings
per share" counted as a price question and fetching share prices; linked
numeric questions shown "in internal scale". Each source now filters for
those cases, and any filter that fails falls back to the old behaviour.

Never raises: every source has a timeout, and one that fails is left out.
Costs no Claude-API money -- the extra text reaches the forecasters on the
sponsored OpenRouter key. Standard library only.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

import market_quant

logger = logging.getLogger(__name__)

TIMEOUT = 10
MAX_SECTION = 2500
MAX_TOTAL = 7000
MAX_PAGES = 3
UA = "Mozilla/5.0 (seergiii-bot research; +https://github.com/BySergiMM/metac-bot-template)"

Fetch = Callable[[str, dict], bytes]


def _http(url: str, headers: dict) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https hosts
        return resp.read(600_000)


def _json(fetch: Fetch, url: str, headers: dict | None = None) -> Any:
    return json.loads(fetch(url, headers or {}).decode("utf-8", "replace"))


def _text_of(question: Any) -> str:
    return "\n".join(
        str(getattr(question, a, "") or "")
        for a in ("question_text", "group_question_option", "background_info",
                  "resolution_criteria", "fine_print")
    )


# ----------------------------------------------------------------- Metaculus

METACULUS_LINK = re.compile(r"metaculus\.com/questions/(\d+)")


def _real_units(q: dict, location: float) -> float | None:
    """A point on Metaculus' internal 0-1 axis in the question's own units.

    The API gives continuous aggregates on that axis; until 2026-10-23 they
    were shown raw ("0.42 (internal scale)"), which reads as a value. Same
    formula as the SDK's _cdf_location_to_nominal_location: linear between
    range_min and range_max, or logarithmic when zero_point is set.
    """
    scaling = q.get("scaling") or {}
    lo, hi, zero = scaling.get("range_min"), scaling.get("range_max"), scaling.get("zero_point")
    if lo is None or hi is None:
        return None
    lo, hi, x = float(lo), float(hi), float(location)
    if zero is None:
        return lo + (hi - lo) * x
    ratio = (hi - float(zero)) / (lo - float(zero))
    if ratio <= 0 or ratio == 1:
        return None
    return lo + (hi - lo) * (ratio ** x - 1) / (ratio - 1)


def _shown_value(q: dict, value: float) -> str:
    if q.get("type") == "date":
        return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
    shown = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    unit = str(q.get("unit") or "").strip()
    return f"{shown} {unit}".strip()


def _community(q: dict, latest: dict) -> str | None:
    """The community prediction in words, or None when it cannot be stated in
    real units (better nothing than a number on the wrong scale)."""
    kind = q.get("type")
    centers = latest.get("centers")
    if kind == "binary":
        if isinstance(centers, list) and centers and centers[0] is not None:
            return f"{float(centers[0]):.1%}"
        return None
    if kind == "multiple_choice":
        options, values = q.get("options"), latest.get("forecast_values")
        if not (isinstance(options, list) and isinstance(values, list) and len(options) == len(values)):
            return None
        top = sorted(zip(options, values), key=lambda ov: -float(ov[1] or 0))[:4]
        return ", ".join(f"{o} {float(v or 0):.0%}" for o, v in top)
    if kind in ("numeric", "discrete", "date"):
        if not (isinstance(centers, list) and centers and centers[0] is not None):
            return None
        median = _real_units(q, centers[0])
        if median is None:
            return None
        shown = f"median {_shown_value(q, median)}"
        lows, highs = latest.get("interval_lower_bounds"), latest.get("interval_upper_bounds")
        if isinstance(lows, list) and isinstance(highs, list) and lows and highs \
                and lows[0] is not None and highs[0] is not None:
            low, high = _real_units(q, lows[0]), _real_units(q, highs[0])
            if low is not None and high is not None:
                shown += f" (middle 50%: {_shown_value(q, low)} to {_shown_value(q, high)})"
        return shown
    return None


def metaculus_links(question: Any, fetch: Fetch) -> str:
    own = str(getattr(question, "id_of_post", "") or "")
    ids = [i for i in dict.fromkeys(METACULUS_LINK.findall(_text_of(question))) if i != own][:3]
    token = os.environ.get("METACULUS_TOKEN", "")
    if not ids or not token:
        return ""
    lines = []
    for pid in ids:
        try:
            post = _json(fetch, f"https://www.metaculus.com/api/posts/{pid}/",
                         {"Authorization": f"Token {token}"})
        except Exception:  # noqa: BLE001
            continue
        q = post.get("question") or {}
        latest = (((q.get("aggregations") or {}).get("recency_weighted") or {}).get("latest") or {})
        try:
            shown = _community(q, latest)
        except Exception as exc:  # noqa: BLE001 - an odd shape drops the line, never the block
            logger.warning("live_context linked_question_unreadable reason=%s", type(exc).__name__)
            shown = None
        if shown is None:
            continue
        lines.append(f"- Metaculus question {pid} \"{post.get('title', '')[:120]}\": "
                     f"community prediction now {shown}")
    return "\n".join(lines)


# -------------------------------------------------------------------- prices

CRYPTO = {
    "bitcoin": "bitcoin", "btc": "bitcoin", "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana", "xrp": "ripple", "ripple": "ripple",
    "dogecoin": "dogecoin", "doge": "dogecoin", "zcash": "zcash", "zec": "zcash",
    "cardano": "cardano", "bnb": "binancecoin", "litecoin": "litecoin",
}
INDICES = {"s&p 500": "^GSPC", "nasdaq": "^IXIC", "dow jones": "^DJI",
           "vix": "^VIX", "nikkei": "^N225", "ftse 100": "^FTSE"}
TICKER = re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z]{1,2})?|[0-9]{4}\.HK|[A-Z]{4}[0-9])\)")

#: Coin keys that are also ordinary words: "DOGE" is the US Department of
#: Government Efficiency as often as Dogecoin, "sol" is Spanish for sun. They
#: count only in a question that asks about a price.
AMBIGUOUS_COINS = {"doge", "sol"}


def _own_text(question: Any) -> str:
    """What the question itself asks -- its text and group option, not the
    background. A Dogecoin price reached a federal-outlays question (q46224,
    Test Bot run 39813843358) because its background mentions DOGE."""
    return "\n".join(str(getattr(question, a, "") or "") for a in ("question_text", "group_question_option"))


def crypto_prices(question: Any, fetch: Fetch) -> str:
    words = set(re.findall(r"[a-z]+", _own_text(question).lower()))
    priced = _asks_a_price(question)
    ids = sorted({cg for w, cg in CRYPTO.items()
                  if w in words and (w not in AMBIGUOUS_COINS or priced)})[:4]
    if not ids:
        return ""
    try:
        data = _json(fetch, "https://api.coingecko.com/api/v3/simple/price?" + urllib.parse.urlencode(
            {"ids": ",".join(ids), "vs_currencies": "usd", "include_24hr_change": "true"}))
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(
        f"- {cid}: ${row['usd']:,} (24h {row.get('usd_24h_change', 0):+.1f}%) [CoinGecko, now]"
        for cid, row in data.items() if isinstance(row, dict) and "usd" in row
    )


def _yahoo_symbol(raw: str) -> str:
    if re.fullmatch(r"[A-Z]{4}[0-9]", raw):   # B3 tickers such as BHIA3
        return raw + ".SA"
    return raw


PRICE_WORDS = re.compile(r"\b(stock|shares?|close[sd]?|closing|price|etf|index|trad(e|es|ing)|futures)\b", re.I)

#: Accounting figures quoted "per share" are not share prices: "earnings per
#: share" matched \bshares?\b, so Market Pulse EPS questions with a ticker in
#: the text got a one-month share-price range as their live data.
PER_SHARE = re.compile(r"\b(earnings|income|dividends?)\s+per\s+share\b", re.I)


def _asks_a_price(question: Any) -> bool:
    text = PER_SHARE.sub(" ", str(getattr(question, "question_text", "") or ""))
    return bool(PRICE_WORDS.search(text))


def stock_prices(question: Any, fetch: Fetch) -> str:
    text = _text_of(question)
    # Parenthesised capitals are tickers only in a question about prices:
    # elsewhere "(WHO)", "(CDC)" or "(FAA)" would fetch some unrelated
    # company that happens to trade under that symbol.
    if not _asks_a_price(question):
        return ""
    symbols = [_yahoo_symbol(t) for t in TICKER.findall(text)]
    low = text.lower()
    symbols += [sym for name, sym in INDICES.items() if name in low]
    lines = []
    for sym in list(dict.fromkeys(symbols))[:4]:
        try:
            data = _json(fetch, "https://query1.finance.yahoo.com/v8/finance/chart/"
                         + urllib.parse.quote(sym) + "?range=1mo&interval=1d")
            res = data["chart"]["result"][0]
            closes = [c for c in res["indicators"]["quote"][0]["close"] if c is not None]
            meta = res.get("meta", {})
        except Exception:  # noqa: BLE001
            continue
        if not closes:
            continue
        lines.append(
            f"- {sym}: last close {closes[-1]:,.2f} {meta.get('currency', '')}; "
            f"1-month range {min(closes):,.2f}–{max(closes):,.2f} [Yahoo, now]"
        )
    return "\n".join(lines)


def eps_gaap(question: Any, fetch: Fetch) -> str:
    """GAAP diluted EPS history against the analyst consensus; see market_quant.

    Yahoo's consensus needs a cookie and a crumb, and _http keeps no cookies,
    so a real run hands market_quant None and it opens its own session; a
    test's fake fetch is passed through.
    """
    return market_quant.section(question, None if fetch is _http else fetch)


# ---------------------------------------------------------- prediction markets

STOP = set("will the a an of in on at to by for be is are was were what how many much "
           "which who when before after between and or than more less any this that "
           "with from as its their have has does do 2026 2027".split())


def _query(question: Any) -> str:
    text = " ".join(str(getattr(question, a, "") or "") for a in ("group_question_option", "question_text"))
    words = re.findall(r"[A-Za-z0-9$&.-]+", text)
    return " ".join([w for w in words if w.lower() not in STOP][:8])


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if len(w) >= 4 and w not in STOP}


#: Shared content words a market title needs with the question to be shown.
#: Polymarket's search returns loosely related events (an NFL fantasy market
#: for a Tour de France question in Test Bot run 39079742977); the forecaster
#: discarded them, but they cost tokens and invite a wrong anchor.
MIN_OVERLAP = 2


def _related(question_words: set[str], title: str) -> bool:
    return len(question_words & _content_words(title)) >= MIN_OVERLAP


#: A price this close to 0 or 1 on a market past its end date is a resolved
#: market, whatever its flags say.
SETTLED = 0.995


def _when(raw: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _polymarket_live(event: dict, market: dict, prices: list) -> bool:
    """Still trading. Polymarket's search returns resolved markets too: Test
    Bot run 39813843358 showed "Will Donald Trump be President of the USA on
    May 31, 2023?" at No 100% and a 2025 Argentina market at 100/0."""
    if event.get("closed") or market.get("closed") or market.get("archived") \
            or market.get("active") is False:
        return False
    try:
        top = max(float(p) for p in prices)
    except (TypeError, ValueError):
        return True
    end = _when(market.get("endDate") or event.get("endDate"))
    return not (top >= SETTLED and end is not None and end < datetime.now(timezone.utc))


def _kalshi_open(markets: list) -> list:
    """Markets without a result, or none when every price is 0 or 1 -- a
    settled event Kalshi's search still lists."""
    live = [m for m in markets if not m.get("result")]
    prices = [float(m.get("last_price_dollars") or 0) for m in live]
    if not live or all(p <= 0 or p >= 1 for p in prices):
        return []
    return live


def prediction_markets(question: Any, fetch: Fetch) -> str:
    q = _query(question)
    if not q:
        return ""
    qwords = _content_words(str(getattr(question, "question_text", "") or ""))
    lines = []
    try:
        data = _json(fetch, "https://gamma-api.polymarket.com/public-search?"
                     + urllib.parse.urlencode({"q": q, "limit_per_type": 3}))
        for ev in (data.get("events") or [])[:3]:
            for m in (ev.get("markets") or [])[:2]:
                prices = m.get("outcomePrices")
                prices = json.loads(prices) if isinstance(prices, str) else prices
                outcomes = m.get("outcomes")
                outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                if prices and outcomes and _related(qwords, m.get("question", "")) \
                        and _polymarket_live(ev, m, prices):
                    shown = ", ".join(f"{o} {float(p):.0%}" for o, p in zip(outcomes, prices))
                    lines.append(f"- Polymarket \"{m.get('question', '')[:140]}\": {shown}")
    except Exception:  # noqa: BLE001
        pass
    try:
        data = _json(fetch, "https://api.manifold.markets/v0/search-markets?"
                     + urllib.parse.urlencode({"term": q, "limit": 3}))
        for m in data[:3]:
            if (m.get("probability") is not None and not m.get("isResolved")
                    and _related(qwords, m.get("question", ""))):
                lines.append(f"- Manifold \"{m.get('question', '')[:140]}\": "
                             f"{m['probability']:.0%} ({m.get('uniqueBettorCount', '?')} traders)")
    except Exception:  # noqa: BLE001
        pass
    try:
        data = _json(fetch, "https://api.elections.kalshi.com/v1/search/series?"
                     + urllib.parse.urlencode({"query": q, "page_size": 3}))
        for ev in (data.get("current_page") or [])[:3]:
            title = ev.get("event_title") or ev.get("series_title") or ""
            if not _related(qwords, title):
                continue
            markets = sorted(_kalshi_open(ev.get("markets") or []),
                             key=lambda m: -float(m.get("last_price_dollars") or 0))[:4]
            shown = ", ".join(f"{m.get('yes_subtitle') or m.get('ticker')} "
                              f"{float(m.get('last_price_dollars') or 0):.0%}" for m in markets)
            if shown:
                lines.append(f"- Kalshi \"{title[:140]}\": {shown}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines[:8])


# ---------------------------------------------------------- resolution sources

URL = re.compile(r"https?://[^\s)\]>\"']+")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|footer|header|aside|noscript)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def _page_text(raw: str) -> str:
    """The page's own content: its <main>, else its longest <article>, else
    the whole page -- the 1,250 characters shown per page used to be mostly
    menus and cookie banners."""
    for tag in ("main", "article"):
        blocks = re.findall(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", raw)
        if blocks:
            text = _strip_html(max(blocks, key=len))
            if len(text) > 200:
                return text
    return _strip_html(raw)


WAYBACK_SNAPSHOT = re.compile(r"https?://web\.archive\.org/web/(\d{4,14})(?:[a-z]{2}_)?/(https?://.+)", re.I)
WIKIPEDIA = re.compile(r"https?://([a-z-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)", re.I)


def _wayback(url: str) -> str | None:
    """A Wayback snapshot as the archived page itself ("id_": no toolbar, no
    rewritten links), or None for the archive's own pages -- its home page,
    calendars and wildcard listings, which are nothing but chrome."""
    snapshot = WAYBACK_SNAPSHOT.fullmatch(url)
    if not snapshot:
        return None
    return f"https://web.archive.org/web/{snapshot.group(1)}id_/{snapshot.group(2)}"


def _wikipedia(url: str, question: Any, fetch: Fetch) -> str:
    """The article's plain text from the API, cut to the paragraphs that
    share content words with the question (most shared first, then in page
    order); the lead when none does. The rendered page's first 1,250
    characters were navigation."""
    match = WIKIPEDIA.match(url)
    api = f"https://{match.group(1).lower()}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "format": "json", "formatversion": 2, "titles": urllib.parse.unquote(match.group(2))})
    pages = _json(fetch, api)["query"]["pages"]
    page = pages[0] if isinstance(pages, list) else next(iter(pages.values()))
    paragraphs = [p.strip() for p in str(page.get("extract") or "").split("\n")
                  if p.strip() and not p.strip().startswith("==")]
    if not paragraphs:
        raise ValueError("no extract")
    words = _content_words(_own_text(question))
    scored = [(len(words & _content_words(p)), i) for i, p in enumerate(paragraphs)]
    chosen = [i for overlap, i in sorted(scored, key=lambda s: (-s[0], s[1])) if overlap >= MIN_OVERLAP]
    picked, size = [], 0
    for i in chosen or range(len(paragraphs)):
        if size >= MAX_SECTION // 2:
            break
        picked.append(i)
        size += len(paragraphs[i]) + 1
    return " ".join(paragraphs[i] for i in sorted(picked))


def _source_text(url: str, question: Any, fetch: Fetch) -> str:
    if WIKIPEDIA.match(url):
        try:
            return _wikipedia(url, question, fetch)
        except Exception as exc:  # noqa: BLE001 - the plain page is the old behaviour
            logger.warning("live_context wikipedia_api_failed reason=%s fallback=page", type(exc).__name__)
    return _page_text(fetch(url, {}).decode("utf-8", "replace"))


def resolution_sources(question: Any, fetch: Fetch) -> str:
    text = "\n".join(str(getattr(question, a, "") or "") for a in ("resolution_criteria", "fine_print"))
    urls = [u.rstrip(".,;") for u in URL.findall(text) if "metaculus.com" not in u]
    # Wayback links: the snapshot itself, never the archive's own pages.
    urls = [_wayback(u) if "web.archive.org" in u.lower() else u for u in urls]
    parts = []
    for url in [u for u in dict.fromkeys(urls) if u][:MAX_PAGES]:
        try:
            body = _source_text(url, question, fetch)
        except Exception:  # noqa: BLE001
            continue
        if len(body) > 200:
            parts.append(f"[{url}]\n{body[:MAX_SECTION // 2]}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------- assemble

SECTIONS = (
    ("Linked Metaculus questions", metaculus_links),
    ("Current crypto prices", crypto_prices),
    ("Current stock/index prices", stock_prices),
    ("GAAP diluted EPS against the analyst consensus (this question resolves on GAAP)", eps_gaap),
    ("Possibly related prediction markets (check they ask the same thing before using them)",
     prediction_markets),
    ("Text of pages named in the resolution criteria (fetched now)", resolution_sources),
)


def build(question: Any, fetch: Fetch = _http) -> str:
    """The live-data block, or "" when nothing applied. Never raises."""
    blocks, used = [], []
    for title, source in SECTIONS:
        try:
            body = source(question, fetch)
        except Exception:  # noqa: BLE001 - one bad source never costs the rest
            body = ""
        if body:
            blocks.append(f"### {title}\n{body[:MAX_SECTION]}")
            used.append(source.__name__)
    logger.info("live_context sources=%s", ",".join(used) or "none")
    if not blocks:
        return ""
    out = "## Live data retrieved just now by the bot (not by the research assistant)\n\n" + "\n\n".join(blocks)
    return out[:MAX_TOTAL]
