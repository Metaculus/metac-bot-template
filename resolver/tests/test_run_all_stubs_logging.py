import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "ingestion" / "run_all_stubs.py"


@pytest.mark.allow_network
def test_run_all_stubs_creates_structured_logs(tmp_path):
    log_dir = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "RUNNER_LOG_DIR": str(log_dir),
            "RESOLVER_INCLUDE_STUBS": "1",
            "RUNNER_LOG_LEVEL": "INFO",
            "RESOLVER_INGESTION_MODE": "stubs",
        }
    )
    project_root = str(SCRIPT.parents[2])
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_root}:{existing_path}" if existing_path else project_root
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--connector",
            "ifrc_go_stub",
            "--retries",
            "1",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    log_files = sorted(log_dir.glob("ingest_*.log"))
    jsonl_files = sorted(log_dir.glob("ingest_*.jsonl"))
    assert log_files, "expected plain log output"
    assert jsonl_files, "expected jsonl log output"

    run_id = log_files[0].stem.split("_", 1)[1]
    connector_log = log_dir / run_id / "ifrc_go_stub.log"
    assert connector_log.exists(), "expected per-connector log file"

    with jsonl_files[0].open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert any(rec.get("extra", {}).get("event") == "run_summary" for rec in records)
