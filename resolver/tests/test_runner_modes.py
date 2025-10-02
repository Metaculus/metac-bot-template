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

    return module


def test_runner_stubs_mode_runs_only_stubs(monkeypatch, tmp_path):
    module = _prepare_module(monkeypatch, tmp_path, "stubs")

    calls: list[str] = []

    monkeypatch.setattr(module, "_run_script", lambda path: calls.append(path.name) or 0)

    module.main()

    assert calls, "expected stub scripts to run"
    assert set(calls) == set(module.STUBS)

    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    importlib.reload(module)


def test_runner_real_mode_skips_stubs(monkeypatch, tmp_path):
    module = _prepare_module(monkeypatch, tmp_path, "real")

    calls: list[str] = []

    monkeypatch.setattr(module, "_run_script", lambda path: calls.append(path.name) or 0)

    module.main()

    assert calls, "expected real connectors to run"
    assert set(calls) == set(module.REAL)

    monkeypatch.delenv("RESOLVER_INGESTION_MODE", raising=False)
    importlib.reload(module)
