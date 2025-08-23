from datetime import datetime

import pytest

from metaculus_bot.research_providers import _dedup_articles_by_url, _normalize_url_for_dedup


class DummyArticle:
    def __init__(self, url: str, title: str = "") -> None:
        self.article_url = url
        self.eng_title = title
        self.pub_date = datetime.now()


def test_normalize_url_basic_tracking_and_slashes() -> None:
    u1 = "https://EXAMPLE.com/path/?utm_source=x&utm_medium=y&gclid=123&a=1&b=2#frag"
    u2 = "https://example.com/path?a=1&b=2/"
    n1 = _normalize_url_for_dedup(u1)
    n2 = _normalize_url_for_dedup(u2)
    assert n1 == n2


def test_normalize_url_mobile_and_amp() -> None:
    u1 = "https://m.news.com/article/amp"
    u2 = "https://news.com/article"
    assert _normalize_url_for_dedup(u1) == _normalize_url_for_dedup(u2)


def test_dedup_articles_by_url_preserves_order_and_keeps_non_url_items() -> None:
    items = [
        DummyArticle("https://site.com/a?utm_campaign=z"),
        DummyArticle("https://site.com/a/"),  # duplicate of first after normalization
        {"article_url": "https://m.other.com/b/amp"},
        DummyArticle("https://other.com/b"),  # duplicate of previous after normalization
        {"no_url": True},  # should be kept
    ]
    deduped = _dedup_articles_by_url(items)
    # Expect first occurrence of each URL + the item without URL
    assert len(deduped) == 3
    assert isinstance(deduped[0], DummyArticle)
    assert isinstance(deduped[1], dict)
    assert isinstance(deduped[2], dict) and deduped[2].get("no_url")
