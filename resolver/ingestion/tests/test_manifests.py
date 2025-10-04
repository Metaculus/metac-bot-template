from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion._manifest import (
    count_csv_rows,
    ensure_manifest_for_csv,
    load_manifest,
    manifest_path_for,
)


def test_manifest_creation_and_repair(tmp_path: Path) -> None:
    csv_path = tmp_path / "example.csv"
    csv_path.write_text("a,b\n", encoding="utf-8")

    manifest = ensure_manifest_for_csv(csv_path)
    manifest_path = manifest_path_for(csv_path)

    assert manifest_path.exists(), "manifest file should be created"
    assert manifest["format"] == "csv"
    assert manifest["row_count"] == 0
    assert manifest["data_path"] == csv_path.name
    assert "sha256" in manifest
    assert "generated_at" in manifest

    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    updated = ensure_manifest_for_csv(csv_path)

    assert updated["row_count"] == 3
    assert updated.get("sha256") != manifest.get("sha256")

    corrupt = load_manifest(manifest_path) or {}
    corrupt["row_count"] = 999
    manifest_path.write_text(json.dumps(corrupt), encoding="utf-8")

    recount = count_csv_rows(csv_path)
    reloaded = load_manifest(manifest_path)
    assert reloaded is not None
    assert reloaded["row_count"] != recount

    repaired = ensure_manifest_for_csv(csv_path)
    assert repaired["row_count"] == recount
    fixed = load_manifest(manifest_path)
    assert fixed is not None
    assert fixed["row_count"] == recount
