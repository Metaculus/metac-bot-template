from __future__ import annotations
import math, datetime as dt
import pandas as pd
from resolver.tests.test_utils import (
    load_schema, load_countries, load_shocks, read_csv,
    discover_export_files, today_iso, require_columns
)

def _is_iso_date(s: str) -> bool:
    try:
        dt.date.fromisoformat(s); return True
    except Exception:
        return False

def _num_ok(x: str) -> bool:
    try:
        return float(x) >= 0
    except Exception:
        return False

def test_exports_facts_contract_if_present():
    schema = load_schema()
    req = set(schema["required"])
    enum_metric = set(schema["enums"]["metric"])
    enum_unit = set(schema["enums"]["unit"])
    countries = load_countries()
    shocks = load_shocks()
    cset = set(countries["iso3"])
    sset = set(shocks["hazard_code"])

    # prefer current canonical facts
    paths = [p for p in discover_export_files() if p.name == "facts.csv" and "state" not in str(p)]
    if not paths:
        # skip if no exports/facts.csv yet
        return

    df = read_csv(paths[0])

    # columns present
    missing = req - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    # registry membership
    assert df["iso3"].isin(cset).all(), "Some iso3 not in countries registry"
    assert df["hazard_code"].isin(sset).all(), "Some hazard_code not in shocks registry"

    # labels/classes match the shocks registry rows
    reg = shocks.set_index("hazard_code")
    bad_label = []
    bad_class = []
    for _, r in df.iterrows():
        code = r["hazard_code"]
        if code in reg.index:
            if r["hazard_label"] != reg.loc[code, "hazard_label"]:
                bad_label.append(code)
            if r["hazard_class"] != reg.loc[code, "hazard_class"]:
                bad_class.append(code)
    assert not bad_label, f"hazard_label mismatch for codes: {set(bad_label)}"
    assert not bad_class, f"hazard_class mismatch for codes: {set(bad_class)}"

    # metric & unit enums
    assert df["metric"].isin(enum_metric).all()
    assert df["unit"].isin(enum_unit).all()

    # numeric values >= 0
    assert df["value"].map(_num_ok).all(), "Non-numeric or negative values present"

    # date sanity
    assert df["as_of_date"].map(_is_iso_date).all()
    assert df["publication_date"].map(_is_iso_date).all()
    assert (df["as_of_date"] <= df["publication_date"]).all()
    assert (df["publication_date"] <= today_iso()).all()

    # governance rules
    media_in_need = df[(df["source_type"] == "media") & (df["metric"] == "in_need")]
    assert media_in_need.empty, "Policy: media cannot be the source for in_need"

    cases_wrong_unit = df[(df["metric"] == "cases") & (df["unit"] != "persons_cases")]
    assert cases_wrong_unit.empty, "Policy: cases must use unit=persons_cases"

def test_remote_first_state_exports_are_valid_if_present():
    # Validate any committed state exports CSVs (PR or daily). Skip gracefully if none.
    schema = load_schema()
    req = set(schema["required"])
    enum_metric = set(schema["enums"]["metric"])
    enum_unit = set(schema["enums"]["unit"])
    countries = load_countries()
    shocks = load_shocks()
    cset = set(countries["iso3"]); sset = set(shocks["hazard_code"])

    paths = [p for p in discover_export_files() if p.name == "facts.csv" and "state" in str(p)]
    if not paths:
        return

    for p in paths:
        df = read_csv(p)
        missing = req - set(df.columns)
        assert not missing, f"{p}: missing columns {missing}"
        assert df["iso3"].isin(cset).all(), f"{p}: iso3 not in registry"
        assert df["hazard_code"].isin(sset).all(), f"{p}: hazard_code not in registry"
        assert df["metric"].isin(enum_metric).all(), f"{p}: metric enum"
        assert df["unit"].isin(enum_unit).all(), f"{p}: unit enum"
