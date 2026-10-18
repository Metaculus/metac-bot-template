"""Live data block: prices, linked Metaculus questions, markets, sources.

A fake fetcher stands in for the network, so these run offline and pin what
each source asks for and how it reads the answer.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from types import SimpleNamespace as Q
from unittest import mock

import live_context as lc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def q(text, criteria="", fine="", background="", post=1):
    return Q(question_text=text, resolution_criteria=criteria, fine_print=fine,
             background_info=background, id_of_post=post)


class FakeWeb:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        for prefix, body in self.routes.items():
            if url.startswith(prefix):
                if isinstance(body, Exception):
                    raise body
                return body if isinstance(body, bytes) else json.dumps(body).encode()
        raise OSError("no route: " + url)


YAHOO = {"chart": {"result": [{"meta": {"currency": "USD"},
                               "indicators": {"quote": [{"close": [300.0, None, 320.5, 310.25]}]}}]}}


class PriceTests(unittest.TestCase):
    def test_stock_ticker_in_a_price_question(self):
        web = FakeWeb({"https://query1.finance.yahoo.com/v8/finance/chart/AAPL": YAHOO})
        out = lc.stock_prices(q("What will Apple's (AAPL) stock close at?"), web)
        self.assertIn("AAPL: last close 310.25 USD; 1-month range 300.00–320.50", out)

    def test_acronyms_are_not_tickers_outside_price_questions(self):
        web = FakeWeb({})
        self.assertEqual(lc.stock_prices(q("Will the (WHO) declare a PHEIC?"), web), "")
        self.assertEqual(web.calls, [])

    def test_brazilian_ticker_gets_the_b3_suffix(self):
        web = FakeWeb({"https://query1.finance.yahoo.com/v8/finance/chart/BHIA3.SA": YAHOO})
        self.assertIn("BHIA3.SA", lc.stock_prices(q("What will (BHIA3) shares close at?"), web))

    def test_index_by_name(self):
        web = FakeWeb({"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC": YAHOO})
        self.assertIn("^GSPC", lc.stock_prices(q("What will the S&P 500 index close at?"), web))

    def test_crypto_by_name(self):
        web = FakeWeb({"https://api.coingecko.com/": {"bitcoin": {"usd": 81157, "usd_24h_change": -0.5}}})
        out = lc.crypto_prices(q("Will Bitcoin close above $80,000?"), web)
        self.assertIn("bitcoin: $81,157 (24h -0.5%)", out)


class MetaculusTests(unittest.TestCase):
    POST = {"title": "Will X happen?", "question": {"type": "binary",
            "aggregations": {"recency_weighted": {"latest": {"centers": [0.37]}}}}}

    def test_linked_question_gets_its_community_prediction_with_the_token(self):
        web = FakeWeb({"https://www.metaculus.com/api/posts/123/": self.POST})
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "t"}):
            out = lc.metaculus_links(q("Will the CP on metaculus.com/questions/123 exceed 40%?"), web)
        self.assertIn("community prediction now 37.0%", out)
        self.assertEqual(web.calls[0][1], {"Authorization": "Token t"})

    def test_the_question_itself_is_skipped(self):
        web = FakeWeb({})
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "t"}):
            lc.metaculus_links(q("see metaculus.com/questions/9", post=9), web)
        self.assertEqual(web.calls, [])


class MarketTests(unittest.TestCase):
    def test_polymarket_and_manifold_are_labelled(self):
        web = FakeWeb({
            "https://gamma-api.polymarket.com/": {"events": [{"markets": [
                {"question": "Democrats win House?", "outcomes": '["Yes","No"]',
                 "outcomePrices": '["0.8","0.2"]'}]}]},
            "https://api.elections.kalshi.com/": {"current_page": [
                {"event_title": "Which party will Democrats win the House?",
                 "markets": [{"yes_subtitle": "Democratic Party", "last_price_dollars": "0.9010"},
                             {"yes_subtitle": "Republican Party", "last_price_dollars": "0.1000"}]}]},
            "https://api.manifold.markets/": [
                {"question": "Will Democrats win the House?", "probability": 0.91,
                 "uniqueBettorCount": 40, "isResolved": False}],
        })
        out = lc.prediction_markets(q("Will Democrats win the House in 2026?"), web)
        self.assertIn('Polymarket "Democrats win House?": Yes 80%, No 20%', out)
        self.assertIn('Manifold "Will Democrats win the House?": 91% (40 traders)', out)
        self.assertIn("Kalshi", out)
        self.assertIn("Democratic Party 90%", out)

    def test_unrelated_markets_are_dropped(self):
        web = FakeWeb({
            "https://gamma-api.polymarket.com/": {"events": [{"markets": [
                {"question": "Will Josh Allen average 22.0+ PPR fantasy points?",
                 "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]'}]}]},
            "https://api.manifold.markets/": [],
        })
        self.assertEqual(lc.prediction_markets(
            q("Will the winner of the 2026 Tour de France be from Australia?"), web), "")

    def test_one_market_failing_keeps_the_other(self):
        web = FakeWeb({"https://gamma-api.polymarket.com/": OSError("ssl"),
                       "https://api.manifold.markets/": [
                           {"question": "Will the comet happen visible?", "probability": 0.5,
                            "isResolved": False}]})
        self.assertIn("Manifold", lc.prediction_markets(q("Will the comet become visible?"), web))


class SourceTests(unittest.TestCase):
    def test_resolution_pages_are_fetched_and_stripped(self):
        page = b"<html><script>x()</script><p>Cases reported: 1,482 as of Oct 10</p>" + b" pad" * 80 + b"</html>"
        web = FakeWeb({"https://www.cdc.gov/": page})
        out = lc.resolution_sources(q("Q?", criteria="Per https://www.cdc.gov/westnile/data."), web)
        self.assertIn("Cases reported: 1,482 as of Oct 10", out)
        self.assertNotIn("x()", out)
        self.assertEqual(web.calls[0][0], "https://www.cdc.gov/westnile/data")

    def test_metaculus_urls_are_not_resolution_sources(self):
        web = FakeWeb({})
        lc.resolution_sources(q("Q?", criteria="https://www.metaculus.com/questions/5/"), web)
        self.assertEqual(web.calls, [])


class BuildTests(unittest.TestCase):
    def test_nothing_applicable_is_empty(self):
        self.assertEqual(lc.build(q("Will it rain in Paris?"), FakeWeb({})), "")

    def test_a_broken_source_never_raises(self):
        def explode(url, headers):
            raise RuntimeError("down")
        self.assertEqual(lc.build(q("Will Bitcoin (BTC) price close above 1?"), explode), "")

    def test_output_is_capped(self):
        page = b"<p>" + b"word " * 20000 + b"</p>"
        web = FakeWeb({"https://a.example/": page, "https://b.example/": page,
                       "https://c.example/": page})
        out = lc.build(q("Q?", criteria="https://a.example/1 https://b.example/2 https://c.example/3"), web)
        self.assertLessEqual(len(out), lc.MAX_TOTAL)

    def test_prompts_ask_for_a_base_rate_first_and_a_pre_mortem(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            src = h.read()
        self.assertGreaterEqual(src.count("(a0) Before using the research"), 4)
        self.assertGreaterEqual(src.count("Pre-mortem"), 4)
        self.assertGreaterEqual(src.count('"Live data" section'), 4)

    def test_main_appends_it_to_the_research(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            src = h.read()
        start = src.index("async def run_research")
        body = src[start:src.index("return research", start)]
        self.assertIn("asyncio.to_thread(live_context.build, question)", body)
