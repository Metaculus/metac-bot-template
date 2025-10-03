"""Smoke test for WFP mVAM client import and collection."""

import importlib
from pathlib import Path


def test_collect_rows_disabled_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WFP_MVAM_ALLOW_PERCENT", "1")

    config_path = tmp_path / "wfp_mvam_sources.yml"
    config_path.write_text("enabled: false\nsources: []\n", encoding="utf-8")

    client = importlib.import_module("resolver.ingestion.wfp_mvam_client")
    monkeypatch.setattr(client, "SOURCES_CONFIG", Path(config_path))

    rows = client.collect_rows()

    assert rows == []
