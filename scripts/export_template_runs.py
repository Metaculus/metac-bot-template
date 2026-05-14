from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _daily_path(output_dir: Path, when: datetime | None = None) -> Path:
    now = when or datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"runs-{now.date().isoformat()}.jsonl"


def export_run(
    question_id: int,
    question_text: str,
    p_yes: float,
    reasoning: str,
    close_date: str,
    resolution_date: str,
    outcome: bool | None,
    output_dir: Path,
) -> Path:
    """Append one run record to output_dir/runs-YYYY-MM-DD.jsonl."""
    if not (0.0 <= float(p_yes) <= 1.0):
        raise ValueError("p_yes must be in [0,1]")

    row = {
        "question_id": int(question_id),
        "question_text": question_text,
        "run_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "posted_probability": float(p_yes),
        "close_date": close_date,
        "resolution_date": resolution_date,
        "resolved_outcome": outcome,
        "reasoning": reasoning,
    }

    path = _daily_path(output_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")
    return path


def _first_number(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value and isinstance(value[0], (int, float)):
        return float(value[0])
    return None


def export_from_forecast_report(report, output_dir: Path) -> Path | None:
    """Best-effort extraction from forecasting-tools ForecastReport objects."""
    question = getattr(report, "question", None)
    if question is None:
        return None

    question_id = getattr(question, "id_of_post", None) or getattr(question, "id", None)
    question_text = getattr(question, "question_text", None) or getattr(question, "title", "")
    close_dt = getattr(question, "close_time", None) or getattr(question, "scheduled_close_time", None)
    resolve_dt = getattr(question, "scheduled_resolve_time", None) or getattr(question, "resolve_time", None) or close_dt

    if question_id is None or not question_text or close_dt is None or resolve_dt is None:
        return None

    prediction = getattr(report, "prediction", None)
    p_yes = _first_number(prediction)
    if p_yes is None:
        prediction_obj = getattr(report, "prediction_value", None)
        p_yes = _first_number(prediction_obj)
    if p_yes is None:
        return None

    reasoning = getattr(report, "explanation", "") or getattr(report, "report", "") or ""

    close_date = close_dt.date().isoformat() if hasattr(close_dt, "date") else str(close_dt)
    resolution_date = resolve_dt.date().isoformat() if hasattr(resolve_dt, "date") else str(resolve_dt)

    return export_run(
        question_id=int(question_id),
        question_text=str(question_text),
        p_yes=float(p_yes),
        reasoning=str(reasoning),
        close_date=close_date,
        resolution_date=resolution_date,
        outcome=None,
        output_dir=output_dir,
    )
