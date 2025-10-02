import csv
import importlib

CANONICAL = [
    "source","source_event_id","as_of_date","country_iso3","country_name",
    "hazard_code","hazard_label","hazard_class","metric_name","metric_unit",
    "series_semantics","value","evidence_url","evidence_label",
]


def test_unhcr_odp_header(tmp_path, monkeypatch):
    mod = importlib.import_module("resolver.ingestion.unhcr_odp_client")
    out = tmp_path / "unhcr_odp.csv"
    monkeypatch.setattr(mod, "OUT_PATH", out)
    monkeypatch.setenv("RESOLVER_SKIP_UNHCR_ODP", "1")

    rc = mod.main()
    assert rc == 0
    assert out.exists(), "connector must write a CSV file"
    with open(out, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == CANONICAL, "CSV header must match canonical schema"
