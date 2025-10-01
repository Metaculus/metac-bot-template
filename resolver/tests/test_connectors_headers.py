"""
Guarantee each connector writes a CSV with the canonical header,
even if it's empty (fail-soft). This prevents pipeline stalls and
catches accidental column drift early.

Connectors covered (import and run main):
- IFRC GO:    resolver/ingestion/ifrc_go_client.py
- ReliefWeb:  resolver/ingestion/reliefweb_client.py
(Add more here as you create them.)

We don't assert row count — only that the file exists and has the exact columns.
"""

from pathlib import Path
import importlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"

CANON = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","value","unit",
    "as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","method","confidence",
    "revision","ingested_at"
]

def _assert_header(csv_path: Path):
    assert csv_path.exists(), f"missing {csv_path}"
    df = pd.read_csv(csv_path, dtype=str)
    assert list(df.columns) == CANON, f"{csv_path} columns differ: {list(df.columns)}"

def test_ifrc_go_header(tmp_path, monkeypatch):
    # Allow skipping network in CI: if env set, just ensure header exists after main()
    mod = importlib.import_module("resolver.ingestion.ifrc_go_client")
    mod.main()
    _assert_header(STAGING / "ifrc_go.csv")

def test_reliefweb_header(tmp_path, monkeypatch):
    mod = importlib.import_module("resolver.ingestion.reliefweb_client")
    mod.main()
    _assert_header(STAGING / "reliefweb.csv")
