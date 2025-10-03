from __future__ import annotations
from pathlib import Path
import re
from resolver.tests.test_utils import load_countries, load_shocks

def test_countries_registry_has_iso3_and_names():
    c = load_countries()
    assert {"country_name", "iso3"}.issubset(set(c.columns))
    assert not c["iso3"].isnull().any()
    assert not c["country_name"].isnull().any()
    assert c["iso3"].str.len().eq(3).all()

def test_shocks_registry_columns_and_scope():
    s = load_shocks()
    assert {"hazard_code","hazard_label","hazard_class"}.issubset(set(s.columns))
    assert not s.empty
    # scope: no earthquakes (we don't forecast them here)
    labels = s["hazard_label"].str.lower().tolist()
    assert not any("earthquake" in x for x in labels)
    # classes must be in allowed set
    allowed = {"natural","human-induced","epidemic","other","multi"}
    assert set(s["hazard_class"]).issubset(allowed)


def test_conflict_hazard_entries_present_without_cessation():
    shocks = load_shocks()
    codes = set(shocks["hazard_code"].tolist())
    assert "ACO" in codes, "conflict onset hazard missing"
    assert "ACE" in codes, "conflict escalation hazard missing"
    assert not shocks["hazard_label"].str.lower().str.contains("cessation").any()
