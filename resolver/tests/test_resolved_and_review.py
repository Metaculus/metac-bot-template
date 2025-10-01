from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from resolver.tests.test_utils import EXPORTS, REVIEW, read_csv

REQUIRED_RESOLVED = {
    "iso3","hazard_code","hazard_label","hazard_class",
    "metric","value","unit","as_of_date","publication_date",
    "publisher","source_type","source_url","doc_title",
    "definition_text","event_id"
}

def test_resolved_shapes_if_present():
    csv = EXPORTS / "resolved.csv"
    if not csv.exists():
        return
    df = read_csv(csv)
    missing = REQUIRED_RESOLVED - set(df.columns)
    assert not missing, f"resolved.csv missing {missing}"
    # value numeric
    assert pd.to_numeric(df["value"], errors="coerce").notna().all()

def test_resolved_jsonl_if_present():
    jl = EXPORTS / "resolved.jsonl"
    if not jl.exists():
        return
    # ensure it parses line-by-line JSON and has key fields
    with open(jl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            assert "iso3" in obj and "hazard_code" in obj and "value" in obj, f"line {i} missing essentials"

def test_review_queue_if_present():
    rq = REVIEW / "review_queue.csv"
    if not rq.exists():
        return
    df = read_csv(rq)
    expected_cols = {
        "iso3","hazard_code","metric","value","precedence_tier",
        "conflict","low_confidence","media_in_need","proxy_used","tier_risk","date_anomaly","needs_review",
        "analyst_decision","override_value","override_source_url","override_notes"
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"review_queue.csv missing {missing}"
