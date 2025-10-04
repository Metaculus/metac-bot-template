from __future__ import annotations

import importlib

import pandas as pd


def test_unhcr_make_rows_handles_json_error(monkeypatch):
    monkeypatch.delenv("RESOLVER_SKIP_UNHCR", raising=False)

    import resolver.ingestion.unhcr_client as unhcr_client

    unhcr_client = importlib.reload(unhcr_client)

    monkeypatch.setattr(
        unhcr_client,
        "load_cfg",
        lambda: {
            "base_url": "https://example.com",
            "endpoints": {"asylum_applications": "/apps"},
            "user_agent": "pytest",
            "window_days": 30,
            "params": {
                "cf_type": "ISO",
                "coo_all": "true",
                "coa_all": "true",
            },
            "defaults": {"max_results": 10, "debug_every": 1, "max_pages": 2},
            "page_size": 5,
        },
    )

    def _fake_registries():
        countries = pd.DataFrame(
            [
                {
                    "country_name": "Exampleland",
                    "iso3": "EXA",
                    "country_norm": "exampleland",
                }
            ]
        )
        shocks = pd.DataFrame(
            [
                {
                    "hazard_code": "DI",
                    "hazard_label": "Displacement influx",
                    "hazard_class": "conflict",
                }
            ]
        )
        return countries, shocks

    monkeypatch.setattr(unhcr_client, "load_registries", _fake_registries)

    class _Response:
        status_code = 200
        url = "https://example.com/apps?page=1"

        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr(unhcr_client.requests, "get", lambda *_, **__: _Response())

    rows, counters = unhcr_client.make_rows()

    assert rows == []
    assert counters == unhcr_client.Counter()
