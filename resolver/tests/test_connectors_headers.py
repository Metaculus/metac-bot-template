from pathlib import Path
import csv
import importlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"

CANON = [
    "event_id","country_name","iso3",
    "hazard_code","hazard_label","hazard_class",
    "metric","series_semantics","value","unit",
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
    # Hermetic: skip network and force header-only CSV if needed
    monkeypatch.setenv("RESOLVER_SKIP_IFRCGO", "1")
    mod = importlib.import_module("resolver.ingestion.ifrc_go_client")
    mod.main()
    _assert_header(STAGING / "ifrc_go.csv")

def test_reliefweb_header(tmp_path, monkeypatch):
    # Hermetic: skip network and force header-only CSV (WAF/proxy-safe)
    monkeypatch.setenv("RESOLVER_SKIP_RELIEFWEB", "1")
    mod = importlib.import_module("resolver.ingestion.reliefweb_client")
    mod.main()
    _assert_header(STAGING / "reliefweb.csv")


def test_unhcr_header(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_UNHCR", "1")
    mod = importlib.import_module("resolver.ingestion.unhcr_client")
    mod.main()
    _assert_header(STAGING / "unhcr.csv")


def test_hdx_header(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_HDX", "1")
    mod = importlib.import_module("resolver.ingestion.hdx_client")
    mod.main()
    _assert_header(STAGING / "hdx.csv")


def test_who_phe_header(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_WHO", "1")
    mod = importlib.import_module("resolver.ingestion.who_phe_client")
    tmp_out = Path(tmp_path) / "who_phe.csv"
    mod.OUT_PATH = tmp_out
    mod.main()
    _assert_header(tmp_out)


def test_dtm_header_written(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")
    from resolver.ingestion import dtm_client

    dtm_client.OUT_DIR = Path(tmp_path)
    dtm_client.OUT_PATH = Path(tmp_path) / "dtm.csv"
    assert dtm_client.main() is False

    with open(dtm_client.OUT_PATH, newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    assert row == dtm_client.CANONICAL_HEADERS


def test_ipc_header_written(tmp_path, monkeypatch):
    monkeypatch.setenv("RESOLVER_SKIP_IPC", "1")
    mod = importlib.import_module("resolver.ingestion.ipc_client")
    tmp_out = Path(tmp_path) / "ipc.csv"
    mod.OUT_PATH = tmp_out
    mod.main()
    _assert_header(tmp_out)
