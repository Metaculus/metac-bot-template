from __future__ import annotations

import importlib
from pathlib import Path


def _prepare_module(monkeypatch, tmp_path: Path, mode: str):
    monkeypatch.setenv("RESOLVER_INGESTION_MODE", mode)
    monkeypatch.delenv("RESOLVER_INCLUDE_STUBS", raising=False)
    monkeypatch.delenv("RESOLVER_FORCE_DTM_STUB", raising=False)
    monkeypatch.delenv("RESOLVER_FAIL_ON_STUB_ERROR", raising=False)

    import resolver.ingestion.run_all_stubs as run_all_stubs

    module = importlib.reload(run_all_stubs)

    monkeypatch.setattr(module, "ROOT", tmp_path)

    for name in set(module.REAL + module.STUBS):
        (tmp_path / name).write_text("", encoding="utf-8")

    forced = sorted({module.ff.norm(name) for name in module.REAL + module.STUBS})
    monkeypatch.setenv("RESOLVER_FORCE_ENABLE", ",".join(forced))

    return module


def test_runner_stubs_mode_runs_only_stubs(monkeypatch, tmp_path):
    module = _prepare_module(monkeypatch, tmp_path, "stubs")

    calls: list[str] = []

    def _record(spec, _logger):
        calls.append(spec.path.name)
        return {"status": "ok", "rows": 0, "duration_ms": 0}

    monkeypatch.setattr(module, "_run_connector", _record)

    module.main([])

    assert calls, "expected stub scripts to run"
    assert set(calls) == set(module.STUBS)

    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    importlib.reload(module)


def test_runner_real_mode_skips_stubs(monkeypatch, tmp_path):
    module = _prepare_module(monkeypatch, tmp_path, "real")

    calls: list[str] = []

    def _record(spec, _logger):
        calls.append(spec.path.name)
        return {"status": "ok", "rows": 0, "duration_ms": 0}

    monkeypatch.setattr(module, "_run_connector", _record)

    module.main([])

    assert calls, "expected real connectors to run"
    assert set(calls) == set(module.REAL)

    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    importlib.reload(module)


def test_real_mode_failure_returns_non_zero(monkeypatch, tmp_path):
    module = _prepare_module(monkeypatch, tmp_path, "real")

    module.REAL = ["failing_real.py"]
    module.STUBS = []
    (tmp_path / "failing_real.py").write_text("", encoding="utf-8")

    def _fail(spec, logger):  # pragma: no cover - signature documentation
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_run_connector", _fail)

    exit_code = module.main([])

    assert exit_code == 1

    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    importlib.reload(module)
