import importlib
import logging

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.setenv("RESOLVER_DEBUG", "1")
    monkeypatch.setenv("UNHCR_TEST_ISO3", "GRC")
    monkeypatch.delenv("RESOLVER_SKIP_UNHCR", raising=False)
    monkeypatch.delenv("RESOLVER_MAX_RESULTS", raising=False)
    yield
    monkeypatch.delenv("RESOLVER_DEBUG", raising=False)
    monkeypatch.delenv("UNHCR_TEST_ISO3", raising=False)


def _fake_cfg():
    return {
        "base_url": "https://api.unhcr.org/population/v1/",
        "endpoints": {"asylum_applications": "asylum-applications/"},
        "user_agent": "pytest-agent/1.0",
        "window_days": 400,
        "params": {
            "granularity": "month",
            "years_back": 1,
            "include_years": [2025],
        },
        "paging": {"max_pages": 1},
        "debug": {"debug_every": 1},
        "defaults": {"max_results": 50},
        "test_iso3": "GRC",
    }


def _fake_registries():
    countries = pd.DataFrame(
        [
            {"iso3": "GRC", "country_name": "Greece", "country_norm": "greece"},
            {"iso3": "ITA", "country_name": "Italy", "country_norm": "italy"},
        ]
    )
    shocks = pd.DataFrame(
        [
            {"hazard_code": "DI", "hazard_label": "Displacement influx", "hazard_class": "human-induced"}
        ]
    )
    return countries, shocks


class _FakeResponse:
    def __init__(self, payload, url):
        self._payload = payload
        self.status_code = 200
        self.url = url

    def json(self):
        return self._payload


def test_unhcr_debug_logs(monkeypatch, caplog):
    from resolver.ingestion import unhcr_client as module

    module = importlib.reload(module)
    monkeypatch.setattr(module, "load_cfg", _fake_cfg)
    monkeypatch.setattr(module, "load_registries", _fake_registries)
    module.RESOLVER_DEBUG = True
    module.logger.setLevel(logging.DEBUG)

    payload = {
        "results": [
            {
                "coa_iso": "GRC",
                "country_name": "Greece",
                "value": 12,
                "year": 2025,
                "month": 1,
                "coo_iso": "SYR",
            },
            {
                "coa": "ITA",
                "country_of_asylum": "Italy",
                "applications": "17",
                "year": 2025,
                "month": "02",
                "coo": "AFG",
            },
        ]
    }

    def _fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(payload, "https://api.unhcr.org/mock?page=1")

    monkeypatch.setattr(module.requests, "get", _fake_get)

    caplog.set_level(logging.DEBUG, logger=module.logger.name)
    rows, counters = module.make_rows()

    assert rows, "expected at least one row from fake payload"
    assert counters["final_rows"] >= 1
    text = caplog.text
    assert "raw_count" in text
    assert "final_rows" in text
    summary = module._format_summary(counters)
    assert "raw_count=" in summary
    assert "final_rows=" in summary
