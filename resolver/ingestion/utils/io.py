"""I/O helpers shared across ingestion connectors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def ensure_headers(path: Path, headers: Sequence[str]) -> None:
    """Write a header-only CSV to ``path`` ensuring parent directories exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
