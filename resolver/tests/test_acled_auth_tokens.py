import importlib
import os

import resolver.ingestion.acled_auth as acled_auth_module

_ENV_VARS = [
    "ACLED_ACCESS_TOKEN",
    "ACLED_TOKEN",
    "ACLED_REFRESH_TOKEN",
    "ACLED_USERNAME",
    "ACLED_PASSWORD",
]


def _reload(monkeypatch, **env):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.reload(acled_auth_module)
    return module


def test_get_access_token_accepts_opaque_token(monkeypatch):
    module = _reload(monkeypatch, ACLED_ACCESS_TOKEN="opaque-token")

    def _fail(*_args, **_kwargs):  # pragma: no cover - defensive helper
        raise AssertionError("refresh/password grant should not run for opaque tokens")

    monkeypatch.setattr(module, "_exchange_refresh", _fail)
    monkeypatch.setattr(module, "_password_grant", _fail)

    token = module.get_access_token()

    assert token == "opaque-token"


def test_get_access_token_uses_legacy_env_var(monkeypatch):
    module = _reload(monkeypatch, ACLED_TOKEN="legacy-token")

    token = module.get_access_token()

    assert token == "legacy-token"
    assert os.environ.get("ACLED_ACCESS_TOKEN") == "legacy-token"


def test_invalid_access_token_refreshes_when_possible(monkeypatch):
    module = _reload(
        monkeypatch,
        ACLED_ACCESS_TOKEN="expired-token",
        ACLED_REFRESH_TOKEN="refresh-123",
    )

    monkeypatch.setattr(module, "_jwt_is_valid", lambda _token: False)

    def _fake_refresh(refresh_token: str):
        assert refresh_token == "refresh-123"
        return {"access_token": "new-token", "refresh_token": "new-refresh"}

    monkeypatch.setattr(module, "_exchange_refresh", _fake_refresh)

    token = module.get_access_token()

    assert token == "new-token"
    assert os.environ.get("ACLED_ACCESS_TOKEN") == "new-token"
    assert os.environ.get("ACLED_REFRESH_TOKEN") == "new-refresh"
