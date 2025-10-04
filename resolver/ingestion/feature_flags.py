"""Helper utilities for connector enable flags and overrides.

All ingestion connector YAML configs now expose a top-level ``enable`` boolean
which defaults to ``false`` in the repository so that heavy network calls are
skipped in CI by default.  Local developers can override these guards by
setting ``RESOLVER_FORCE_ENABLE`` (for example, ``RESOLVER_FORCE_ENABLE=acled``)
without editing the YAML files.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, Set


def norm(name: str | os.PathLike[str] | None) -> str:
    """Normalise connector names for comparison.

    Names are lower-cased, spaces and hyphens are replaced with underscores, the
    ``.py``/``.yml``/``.yaml`` suffix is removed, and common connector suffixes
    like ``_client``/``_stub`` are stripped.  This allows callers to use the
    same identifier for ``RESOLVER_FORCE_ENABLE``, ``--only`` arguments, or
    configuration file stems.
    """

    if name is None:
        return ""
    value = str(name).strip().lower()
    if not value:
        return ""
    value = value.replace(" ", "_").replace("-", "_")
    for suffix in (".py", ".yml", ".yaml"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for suffix in ("_client", "_stub"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.strip("_")


def parse_force_enable(env_val: str | None) -> Set[str]:
    """Parse ``RESOLVER_FORCE_ENABLE`` environment variable into a set."""

    if not env_val:
        return set()
    entries = (part.strip() for part in str(env_val).split(","))
    normalised = {norm(part) for part in entries if part.strip()}
    return {item for item in normalised if item}


def _coerce_cfg(cfg: object) -> Dict[str, object]:
    return cfg if isinstance(cfg, dict) else {}


def is_enabled(
    name: str,
    cfg: Dict[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return ``True`` if the connector should run.

    ``cfg`` should be the parsed YAML mapping for the connector.  ``env``
    defaults to :mod:`os.environ`.  ``RESOLVER_FORCE_ENABLE`` always wins over
    the config flag.
    """

    if env is None:
        env = os.environ
    cfg_dict = _coerce_cfg(cfg)
    cfg_enable = bool(cfg_dict.get("enable", False))
    forced = norm(name) in parse_force_enable(env.get("RESOLVER_FORCE_ENABLE"))
    return forced or cfg_enable


def explain_enable(
    name: str,
    cfg: Dict[str, object] | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return a short reason string for why a connector is enabled/disabled."""

    if env is None:
        env = os.environ
    cfg_dict = _coerce_cfg(cfg)
    cfg_enable = bool(cfg_dict.get("enable", False))
    if norm(name) in parse_force_enable(env.get("RESOLVER_FORCE_ENABLE")):
        return "forced_by_env"
    return "config_enabled" if cfg_enable else "config_disabled"
