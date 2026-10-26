"""Live data block: prices, linked Metaculus questions, markets, sources.

A fake fetcher stands in for the network, so these run offline and pin what
each source asks for and how it reads the answer.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from datetime import datetime, timezone
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


class LiveDataHygieneTests(unittest.TestCase):
    """The noise Test Bot run 39813843358 put in the Live data block."""

    COINS = {"https://api.coingecko.com/": {"dogecoin": {"usd": 0.2}, "solana": {"usd": 150}}}

    def test_a_coin_in_the_background_only_is_not_fetched(self):
        web = FakeWeb(self.COINS)
        out = lc.crypto_prices(q("Will federal outlays fall in FY2026?",
                                 background="DOGE (Department of Government Efficiency) promised cuts."), web)
        self.assertEqual(out, "")
        self.assertEqual(web.calls, [])

    def test_bare_doge_and_sol_need_a_price_question(self):
        web = FakeWeb(self.COINS)
        self.assertEqual(lc.crypto_prices(q("Will DOGE cut 100,000 federal jobs?"), web), "")
        self.assertEqual(lc.crypto_prices(q("Will the Sol model top the leaderboard?"), web), "")
        self.assertEqual(web.calls, [])
        self.assertIn("dogecoin", lc.crypto_prices(q("Will the DOGE price close above $1?"), web))
        self.assertIn("dogecoin", lc.crypto_prices(q("Will Dogecoin be in the top 10 coins?"), web))

    def test_the_group_option_names_the_coin(self):
        web = FakeWeb(self.COINS)
        question = Q(question_text="What will these coins be worth on Dec 31?", group_question_option="Solana",
                     background_info="", resolution_criteria="", fine_print="")
        self.assertIn("solana", lc.crypto_prices(question, web))

    def test_earnings_per_share_is_not_a_share_price(self):
        web = FakeWeb({"https://query1.finance.yahoo.com/": YAHOO})
        for text in ("What will Intel (INTC) report as diluted earnings per share?",
                     "What will Intel (INTC) report as net income per share?",
                     "What dividends per share will Intel (INTC) declare?"):
            self.assertEqual(lc.stock_prices(q(text), web), "")
        self.assertEqual(web.calls, [])
        self.assertIn("INTC", lc.stock_prices(q("Will Intel (INTC) shares close above $30?"), web))

    @staticmethod
    def polymarket(**market):
        return {"https://gamma-api.polymarket.com/": {"events": [{"markets": [dict(
            {"question": "Will Democrats win the House?", "outcomes": '["Yes","No"]',
             "outcomePrices": '["0.6","0.4"]'}, **market)]}]},
            "https://api.manifold.markets/": [], "https://api.elections.kalshi.com/": {}}

    def test_resolved_polymarket_markets_are_dropped(self):
        question = q("Will Democrats win the House in 2026?")
        for flags in ({"closed": True}, {"active": False}, {"archived": True},
                      {"outcomePrices": '["1","0"]', "endDate": "2023-05-31T00:00:00Z"},
                      {"outcomePrices": '["0.0005","0.9995"]', "endDate": "2025-01-01T00:00:00Z"}):
            self.assertEqual(lc.prediction_markets(question, FakeWeb(self.polymarket(**flags))), "", flags)

    def test_live_polymarket_markets_stay_even_when_lopsided(self):
        question = q("Will Democrats win the House in 2026?")
        self.assertIn("Yes 60%", lc.prediction_markets(question, FakeWeb(self.polymarket(
            closed=False, active=True, endDate="2099-11-03T00:00:00Z"))))
        self.assertIn("Yes 100%", lc.prediction_markets(question, FakeWeb(self.polymarket(
            outcomePrices='["0.996","0.004"]', endDate="2099-11-03T00:00:00Z"))))

    def test_settled_kalshi_events_are_dropped(self):
        def kalshi(markets):
            return FakeWeb({"https://gamma-api.polymarket.com/": {}, "https://api.manifold.markets/": [],
                            "https://api.elections.kalshi.com/": {"current_page": [
                                {"event_title": "Which party will win the House?", "markets": markets}]}})
        question = q("Which party will win the House in 2026?")
        settled = [{"yes_subtitle": "Democratic Party", "last_price_dollars": "1.0000"},
                   {"yes_subtitle": "Republican Party", "last_price_dollars": "0.0000"}]
        self.assertEqual(lc.prediction_markets(question, kalshi(settled)), "")
        resolved = [{"yes_subtitle": "Democratic Party", "last_price_dollars": "0.6", "result": "yes"}]
        self.assertEqual(lc.prediction_markets(question, kalshi(resolved)), "")
        live = [{"yes_subtitle": "Democratic Party", "last_price_dollars": "0.6000", "result": ""},
                {"yes_subtitle": "Republican Party", "last_price_dollars": "0.0000", "result": ""}]
        self.assertIn("Democratic Party 60%", lc.prediction_markets(question, kalshi(live)))


class LinkedNumericTests(unittest.TestCase):
    """Linked continuous questions in real units, never "internal scale"."""

    @staticmethod
    def post(kind, scaling, latest, **extra):
        return {"title": "Linked", "question": dict(
            {"type": kind, "scaling": scaling,
             "aggregations": {"recency_weighted": {"latest": latest}}}, **extra)}

    def shown(self, post):
        web = FakeWeb({"https://www.metaculus.com/api/posts/77/": post})
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "t"}):
            return lc.metaculus_links(q("See metaculus.com/questions/77 for the related forecast."), web)

    def test_linear_numeric(self):
        out = self.shown(self.post("numeric", {"range_min": 0, "range_max": 200, "zero_point": None},
                                   {"centers": [0.25], "interval_lower_bounds": [0.2],
                                    "interval_upper_bounds": [0.3]}, unit="GW"))
        self.assertIn("median 50 GW (middle 50%: 40 GW to 60 GW)", out)
        self.assertNotIn("internal scale", out)

    def test_log_scaled_numeric(self):
        out = self.shown(self.post("numeric", {"range_min": 1, "range_max": 1000, "zero_point": 0},
                                   {"centers": [0.5]}))
        self.assertIn("median 31.62", out)

    def test_large_values_and_dates(self):
        out = self.shown(self.post("discrete", {"range_min": 0, "range_max": 1e6, "zero_point": None},
                                   {"centers": [0.5]}))
        self.assertIn("median 500,000", out)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        end = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
        out = self.shown(self.post("date", {"range_min": start, "range_max": end, "zero_point": None},
                                   {"centers": [0.5]}))
        self.assertIn("median 2026-07-02", out)

    def test_without_scaling_it_is_dropped(self):
        self.assertEqual(self.shown(self.post("numeric", {}, {"centers": [0.4]})), "")

    def test_multiple_choice_names_its_options(self):
        out = self.shown(self.post("multiple_choice", {}, {"forecast_values": [0.2, 0.7, 0.1]},
                                   options=["A", "B", "C"]))
        self.assertIn("B 70%, A 20%, C 10%", out)
        self.assertEqual(self.shown(self.post("multiple_choice", {}, {"forecast_values": [0.5, 0.5]},
                                              options=["A", "B", "C"])), "")


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

    def test_wayback_root_pages_are_skipped_and_snapshots_read_raw(self):
        page = b"<p>Archived count: 77 as of March</p>" + b" pad" * 80
        web = FakeWeb({"https://web.archive.org/web/20250301000000id_/": page})
        out = lc.resolution_sources(q("Q?", criteria=(
            "See the Wayback Machine (https://web.archive.org/) or https://web.archive.org/web/*/example.org "
            "and https://web.archive.org/web/20250301000000/https://example.org/stats.")), web)
        self.assertEqual([u for u, _ in web.calls],
                         ["https://web.archive.org/web/20250301000000id_/https://example.org/stats"])
        self.assertIn("Archived count: 77", out)

    def test_main_content_wins_over_page_chrome(self):
        chrome = b"<div>Menu Home About Subscribe Cookie settings Accept all</div>" * 30
        page = (b"<html><body>" + chrome + b"<main><h1>Report</h1><p>Cases reported: 1,482 as of Oct 10."
                + b" detail" * 40 + b"</p></main>" + chrome + b"</body></html>")
        out = lc.resolution_sources(q("Q?", criteria="https://www.cdc.gov/data"), FakeWeb({"https://www.cdc.gov/": page}))
        self.assertIn("Cases reported: 1,482", out)
        self.assertNotIn("Cookie settings", out)

    def test_longest_article_when_there_is_no_main(self):
        page = (b"<nav>Menu</nav><article>Teaser</article><div>Sidebar " + b"x " * 300 + b"</div>"
                + b"<article><p>The agency counted 512 cases this week." + b" more" * 60 + b"</p></article>")
        out = lc.resolution_sources(q("Q?", criteria="https://a.example/p"), FakeWeb({"https://a.example/": page}))
        self.assertTrue(out.split("\n", 1)[1].startswith("The agency counted 512 cases"))

    WIKI_API = "https://en.wikipedia.org/w/api.php?"

    def test_wikipedia_keeps_the_paragraphs_about_the_question(self):
        extract = "\n".join([
            "The Grand Tour is a cycling race held every July in France." + " filler" * 20,
            "", "== History ==",
            "The race was first organised in 1903 by a newspaper." + " filler" * 20,
            "== Winners ==",
            "Australian riders have finished on the podium twice; no Australian winner since Cadel Evans."
            + " filler" * 20,
        ])
        web = FakeWeb({self.WIKI_API: {"query": {"pages": [{"pageid": 1, "title": "Grand Tour",
                                                           "extract": extract}]}}})
        out = lc.resolution_sources(q("Will the Grand Tour winner be Australian?",
                                      criteria="Per https://en.wikipedia.org/wiki/Grand_Tour."), web)
        self.assertTrue(web.calls[0][0].startswith(self.WIKI_API))
        self.assertIn("explaintext=1", web.calls[0][0])
        self.assertIn("Australian riders", out)
        self.assertNotIn("first organised in 1903", out)
        self.assertNotIn("== History ==", out)

    def test_wikipedia_never_cuts_the_best_match_for_a_weaker_one(self):
        """The budget is filled most relevant first, then put in page order:
        a weaker paragraph earlier on the page used to push the best match
        past the cut resolution_sources makes."""
        filler = "Unrelated history of the region and its geography. " * 16
        mild = "The Zorbian parliament building was renovated in 1990 and hosts tours. " * 11
        key = ("The turnip subsidy bill passed its second reading in the Zorbian "
               "parliament; the final vote is scheduled. ") * 8
        extract = "\n".join([filler, mild, filler, key.strip()])
        web = FakeWeb({self.WIKI_API: {"query": {"pages": [{"pageid": 1, "extract": extract}]}}})
        question = q("Will the Zorbian parliament pass the turnip subsidy bill before 2027?",
                     criteria="Per https://en.wikipedia.org/wiki/Zorbian_parliament.")
        out = lc.resolution_sources(question, web)
        body = out.split("\n", 1)[1]
        self.assertLessEqual(len(body), lc.MAX_SECTION // 2)
        self.assertIn(key.strip(), body)
        self.assertNotIn("Unrelated history", body)
        # What is left of the weaker paragraph still comes first, as on the page.
        self.assertTrue(body.startswith("The Zorbian parliament building"))
        self.assertTrue(body.endswith("is scheduled."))

    def test_wikipedia_api_failure_falls_back_to_the_page(self):
        page = b"<p>Plain page text about the race.</p>" + b" pad" * 80
        web = FakeWeb({self.WIKI_API: OSError("api down"), "https://en.wikipedia.org/wiki/": page})
        with self.assertLogs("live_context", level="WARNING"):
            out = lc.resolution_sources(q("Q?", criteria="https://en.wikipedia.org/wiki/Race"), web)
        self.assertIn("Plain page text", out)


class BuildTests(unittest.TestCase):
    def test_nothing_applicable_is_empty(self):
        self.assertEqual(lc.build(q("Will it rain in Paris?"), FakeWeb({})), "")

    def test_a_broken_source_never_raises(self):
        def explode(url, headers):
            raise RuntimeError("down")
        self.assertEqual(lc.build(q("Will Bitcoin (BTC) price close above 1?"), explode), "")

    def test_eps_section_is_in_the_block_and_the_sources_line(self):
        question = q("What will Q3 2026 EPS be for these companies?",
                     criteria="Diluted earnings per share in conformity with US GAAP.")
        with mock.patch("market_quant.section", return_value="- synthetic line") as section, \
                self.assertLogs("live_context", level="INFO") as logs:
            out = lc.build(question, FakeWeb({}))
        self.assertIn("### GAAP diluted EPS against the analyst consensus", out)
        self.assertIn("- synthetic line", out)
        self.assertIn("eps_gaap", "\n".join(logs.output))
        self.assertIsNotNone(section.call_args.args[1])

    def test_the_real_fetch_hands_market_quant_its_own_session(self):
        question = q("Q?")
        with mock.patch("market_quant.section", return_value="") as section:
            lc.eps_gaap(question, lc._http)
            self.assertIsNone(section.call_args.args[1])
            web = FakeWeb({})
            lc.eps_gaap(question, web)
            self.assertIs(section.call_args.args[1], web)

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


class OwnCommunityTests(unittest.TestCase):
    """The revealed community forecast on the question itself, as data for
    the prompt (Metaculus' advice, 2026-10-26: use it, never copy it, and
    prefer it as an input to weighting with it). Synthetic numbers."""

    NOW = __import__("datetime").datetime(2026, 10, 27, 12, 0, tzinfo=__import__("datetime").timezone.utc)

    def question(self, latest=None, revealed_hours_ago=5, qtype="numeric", post=1, qid=11):
        from datetime import timedelta
        api = {"question": {"type": qtype, "unit": "B $",
                            "scaling": {"range_min": 100, "range_max": 200, "zero_point": None},
                            "aggregations": {"recency_weighted": {"latest": latest}}}}
        return Q(question_text="What will revenue be?", id_of_post=post, id_of_question=qid,
                 api_json=api, cp_reveal_time=self.NOW - timedelta(hours=revealed_hours_ago),
                 resolution_criteria="", fine_print="", background_info="")

    LATEST = {"centers": [0.5], "interval_lower_bounds": [0.4], "interval_upper_bounds": [0.6],
              "forecaster_count": 25, "end_time": None}

    def test_a_revealed_cp_is_shown_in_real_units_as_an_input(self):
        out = lc.own_community(self.question(dict(self.LATEST)), FakeWeb({}), self.NOW)
        self.assertIn("25 forecasters", out)
        self.assertIn("median 150 B $ (middle 50%: 140 B $ to 160 B $)", out)
        self.assertIn("not an answer to copy", out)

    def test_hidden_thin_or_missing_cps_add_nothing(self):
        self.assertEqual(lc.own_community(self.question(dict(self.LATEST), revealed_hours_ago=-5),
                                          FakeWeb({}), self.NOW), "")
        few = dict(self.LATEST, forecaster_count=4)
        self.assertEqual(lc.own_community(self.question(few), FakeWeb({}), self.NOW), "")
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": ""}):
            self.assertEqual(lc.own_community(self.question(None), FakeWeb({}), self.NOW), "")

    def test_a_group_subquestion_is_read_from_its_post_when_the_listing_lacks_it(self):
        sub = {"id": 11, "type": "numeric", "unit": "B $",
               "scaling": {"range_min": 100, "range_max": 200, "zero_point": None},
               "aggregations": {"recency_weighted": {"latest": dict(self.LATEST)}}}
        web = FakeWeb({"https://www.metaculus.com/api/posts/7/": {
            "group_of_questions": {"questions": [dict(sub, id=10), sub]}}})
        with mock.patch.dict(os.environ, {"METACULUS_TOKEN": "t"}):
            out = lc.own_community(self.question(None, post=7, qid=11), web, self.NOW)
        self.assertIn("median 150 B $", out)

    def test_every_prompt_says_to_weigh_it_not_copy_it(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as h:
            self.assertEqual(h.read().count("never an answer to copy"), 4)
