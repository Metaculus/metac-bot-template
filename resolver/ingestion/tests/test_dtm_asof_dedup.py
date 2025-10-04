from __future__ import annotations

import os
import time
from pathlib import Path

from resolver.ingestion import dtm_client


def _make_candidate(row: dict[str, str], file_ctx: dict[str, str], value: int) -> dict[str, str | int]:
    return {
        "as_of": dtm_client._extract_record_as_of(row, file_ctx),
        "value": value,
    }


def _apply_admin_dedup(candidates: list[dict[str, str | int]]) -> dict[str, str | int]:
    key = ("NGA", "Borno", "2025-09-01", "dtm_source")
    dedup: dict[tuple[str, str, str, str], dict[str, str | int]] = {}
    for record in candidates:
        existing = dedup.get(key)
        if existing and not dtm_client._is_candidate_newer(existing["as_of"], record["as_of"]):
            continue
        dedup[key] = record
    return dedup[key]


def test_row_field_timestamp_beats_filename(tmp_path: Path) -> None:
    first_file = tmp_path / "dtm_first.csv"
    second_file = tmp_path / "dtm_second.csv"
    first_file.write_text("first", encoding="utf-8")
    second_file.write_text("second", encoding="utf-8")

    first_candidate = _make_candidate(
        {"as_of": "2025-09-30"},
        {"filename": first_file.name, "path": str(first_file)},
        100,
    )
    second_candidate = _make_candidate(
        {"as_of": "2025-10-03"},
        {"filename": second_file.name, "path": str(second_file)},
        999,
    )

    result = _apply_admin_dedup([first_candidate, second_candidate])
    assert result["as_of"] == "2025-10-03"
    assert result["value"] == 999


def test_filename_timestamp_fallback(tmp_path: Path) -> None:
    sept_file = tmp_path / "dtm_2025-09.csv"
    oct_file = tmp_path / "dtm_2025-10.csv"
    sept_file.write_text("sept", encoding="utf-8")
    oct_file.write_text("oct", encoding="utf-8")

    first_candidate = _make_candidate(
        {},
        {"filename": sept_file.name, "path": str(sept_file)},
        123,
    )
    second_candidate = _make_candidate(
        {},
        {"filename": oct_file.name, "path": str(oct_file)},
        456,
    )

    result = _apply_admin_dedup([first_candidate, second_candidate])
    assert result["as_of"] == "2025-10-01"
    assert result["value"] == 456


def test_file_mtime_fallback(tmp_path: Path) -> None:
    older_file = tmp_path / "dtm_old.csv"
    newer_file = tmp_path / "dtm_new.csv"
    older_file.write_text("old", encoding="utf-8")
    newer_file.write_text("new", encoding="utf-8")

    old_ts = time.time() - 7 * 24 * 3600
    new_ts = time.time()
    os.utime(older_file, (old_ts, old_ts))
    os.utime(newer_file, (new_ts, new_ts))

    first_candidate = _make_candidate(
        {},
        {"filename": older_file.name, "path": str(older_file)},
        25,
    )
    second_candidate = _make_candidate(
        {},
        {"filename": newer_file.name, "path": str(newer_file)},
        75,
    )

    result = _apply_admin_dedup([first_candidate, second_candidate])
    assert result["value"] == 75
    assert result["as_of"] == second_candidate["as_of"]


def test_equal_as_of_keeps_first_candidate(tmp_path: Path) -> None:
    file_one = tmp_path / "dtm_equal_one.csv"
    file_two = tmp_path / "dtm_equal_two.csv"
    file_one.write_text("one", encoding="utf-8")
    file_two.write_text("two", encoding="utf-8")

    first_candidate = _make_candidate(
        {"as_of": "2025-11-01"},
        {"filename": file_one.name, "path": str(file_one)},
        10,
    )
    second_candidate = _make_candidate(
        {"as_of": "2025-11-01"},
        {"filename": file_two.name, "path": str(file_two)},
        999,
    )

    result = _apply_admin_dedup([first_candidate, second_candidate])
    assert result["value"] == 10
    assert result["as_of"] == "2025-11-01"
