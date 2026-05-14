from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_template_runs import export_from_forecast_report, export_run


class _Q:
    def __init__(self) -> None:
        self.id = 123
        self.question_text = "Will X happen?"
        self.close_time = "2026-06-01"
        self.scheduled_resolve_time = "2026-07-01"


class _Report:
    def __init__(self) -> None:
        self.question = _Q()
        self.prediction = 0.64
        self.explanation = "reasoning"


def test_export_run_appends_daily_jsonl(tmp_path: Path) -> None:
    path = export_run(
        question_id=1,
        question_text="Q",
        p_yes=0.5,
        reasoning="R",
        close_date="2026-06-01",
        resolution_date="2026-07-01",
        outcome=None,
        output_dir=tmp_path,
    )
    assert path.name.startswith("runs-")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["question_id"] == 1
    assert rows[0]["posted_probability"] == 0.5


def test_export_from_forecast_report_writes_expected_fields(tmp_path: Path) -> None:
    out = export_from_forecast_report(_Report(), tmp_path)
    assert out is not None
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows[0]["question_id"] == 123
    assert rows[0]["question_text"] == "Will X happen?"
    assert rows[0]["posted_probability"] == 0.64
