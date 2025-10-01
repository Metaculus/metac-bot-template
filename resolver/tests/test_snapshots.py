from __future__ import annotations
from pathlib import Path
from resolver.tests.test_utils import SNAPS, read_parquet

def test_any_snapshot_parquet_reads_and_has_core_columns():
    if not SNAPS.exists():
        return
    # take any facts.parquet present
    paths = list(SNAPS.glob("*/facts.parquet"))
    if not paths:
        return
    df = read_parquet(paths[0])
    core = {"iso3","hazard_code","metric","value","as_of_date","publication_date"}
    assert core.issubset(set(df.columns))
