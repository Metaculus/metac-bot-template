"""Deterministic ID helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable


def stable_digest(parts: Iterable[object], length: int = 12, algorithm: str = "sha1") -> str:
    """Return a stable hexadecimal digest derived from ``parts``."""

    text = "|".join(str(part or "") for part in parts)
    if algorithm == "sha256":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    else:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if length > 0:
        return digest[:length]
    return digest
