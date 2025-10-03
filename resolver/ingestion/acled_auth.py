"""Helper utilities for authenticating against the ACLED API."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional

_TOKEN_URL = "https://acleddata.com/oauth/token"
_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}
_CLIENT_ID = "acled"
_MIN_TTL = 300  # seconds


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _jwt_exp(token: str) -> Optional[int]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:
        return None
    exp = payload.get("exp")
    try:
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _jwt_is_valid(token: str, *, min_ttl: int = _MIN_TTL) -> bool:
    exp = _jwt_exp(token)
    if not exp:
        return False
    return (exp - int(time.time())) > min_ttl


def _post(data: Dict[str, str]) -> Dict[str, str]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(_TOKEN_URL, data=body, headers=_HEADERS, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _exchange_refresh(refresh_token: str) -> Dict[str, str]:
    return _post(
        {
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "client_id": _CLIENT_ID,
        }
    )


def _password_grant(username: str, password: str) -> Dict[str, str]:
    return _post(
        {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": _CLIENT_ID,
        }
    )


def get_access_token() -> str:
    """Return a valid ACLED access token, refreshing credentials when required."""

    existing = os.environ.get("ACLED_ACCESS_TOKEN")
    if existing and _jwt_is_valid(existing):
        return existing

    refresh_token = os.environ.get("ACLED_REFRESH_TOKEN")
    if refresh_token:
        tokens = _exchange_refresh(refresh_token)
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError("ACLED refresh grant response missing access_token")
        new_refresh = tokens.get("refresh_token")
        if new_refresh:
            os.environ["ACLED_REFRESH_TOKEN"] = new_refresh
        os.environ["ACLED_ACCESS_TOKEN"] = access_token
        return access_token

    username = os.environ.get("ACLED_USERNAME")
    password = os.environ.get("ACLED_PASSWORD")
    if username and password:
        tokens = _password_grant(username, password)
        access_token = tokens.get("access_token")
        if not access_token:
            raise RuntimeError("ACLED password grant response missing access_token")
        new_refresh = tokens.get("refresh_token")
        if new_refresh:
            os.environ["ACLED_REFRESH_TOKEN"] = new_refresh
        os.environ["ACLED_ACCESS_TOKEN"] = access_token
        return access_token

    raise RuntimeError(
        "ACLED authentication failed: provide ACLED_REFRESH_TOKEN or "
        "ACLED_USERNAME/ACLED_PASSWORD credentials."
    )


def get_auth_header() -> Dict[str, str]:
    """Return an Authorization header for ACLED requests."""

    token = get_access_token()
    return {"Authorization": f"Bearer {token}"}
