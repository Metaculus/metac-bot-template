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
  * possibly related prediction markets (Polymarket, Manifold), labelled as
    such: the forecaster must judge whether they match the question;
  * the text of up to three pages linked from the resolution criteria.

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
from typing import Any, Callable

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
        for a in ("question_text", "background_info", "resolution_criteria", "fine_print")
    )


# ----------------------------------------------------------------- Metaculus

METACULUS_LINK = re.compile(r"metaculus\.com/questions/(\d+)")


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
        centers = latest.get("centers")
        value = centers[0] if isinstance(centers, list) and centers else None
        if value is None:
            continue
        shown = f"{value:.1%}" if q.get("type") == "binary" else f"{value} (internal scale)"
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


def crypto_prices(question: Any, fetch: Fetch) -> str:
    words = set(re.findall(r"[a-z]+", _text_of(question).lower()))
    ids = sorted({cg for w, cg in CRYPTO.items() if w in words})[:4]
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


def stock_prices(question: Any, fetch: Fetch) -> str:
    text = _text_of(question)
    # Parenthesised capitals are tickers only in a question about prices:
    # elsewhere "(WHO)", "(CDC)" or "(FAA)" would fetch some unrelated
    # company that happens to trade under that symbol.
    if not PRICE_WORDS.search(str(getattr(question, "question_text", "") or "")):
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


# ---------------------------------------------------------- prediction markets

STOP = set("will the a an of in on at to by for be is are was were what how many much "
           "which who when before after between and or than more less any this that "
           "with from as its their have has does do 2026 2027".split())


def _query(question: Any) -> str:
    words = re.findall(r"[A-Za-z0-9$&.-]+", str(getattr(question, "question_text", "") or ""))
    return " ".join([w for w in words if w.lower() not in STOP][:8])


def prediction_markets(question: Any, fetch: Fetch) -> str:
    q = _query(question)
    if not q:
        return ""
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
                if prices and outcomes:
                    shown = ", ".join(f"{o} {float(p):.0%}" for o, p in zip(outcomes, prices))
                    lines.append(f"- Polymarket \"{m.get('question', '')[:140]}\": {shown}")
    except Exception:  # noqa: BLE001
        pass
    try:
        data = _json(fetch, "https://api.manifold.markets/v0/search-markets?"
                     + urllib.parse.urlencode({"term": q, "limit": 3}))
        for m in data[:3]:
            if m.get("probability") is not None and not m.get("isResolved"):
                lines.append(f"- Manifold \"{m.get('question', '')[:140]}\": "
                             f"{m['probability']:.0%} ({m.get('uniqueBettorCount', '?')} traders)")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines[:6])


# ---------------------------------------------------------- resolution sources

URL = re.compile(r"https?://[^\s)\]>\"']+")


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|footer|header)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def resolution_sources(question: Any, fetch: Fetch) -> str:
    text = "\n".join(str(getattr(question, a, "") or "") for a in ("resolution_criteria", "fine_print"))
    urls = [u.rstrip(".,;") for u in URL.findall(text) if "metaculus.com" not in u]
    parts = []
    for url in list(dict.fromkeys(urls))[:MAX_PAGES]:
        try:
            body = _strip_html(fetch(url, {}).decode("utf-8", "replace"))
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
