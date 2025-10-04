from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def manifest_path_for(data_path: Path) -> Path:
    """Return the manifest path for a CSV file."""

    return data_path.with_suffix(f"{data_path.suffix}.meta.json")


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 hash for the given file."""

    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_manifest(
    data_path: Path,
    *,
    row_count: int,
    sha256: str | None = None,
    schema_version: str | None = None,
    source_id: str | None = None,
) -> Path:
    """Write a manifest JSON file describing the CSV at ``data_path``."""

    manifest_path = manifest_path_for(data_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": "csv",
        "row_count": int(row_count),
        "data_path": data_path.name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if sha256 is not None:
        payload["sha256"] = sha256
    if schema_version is not None:
        payload["schema_version"] = schema_version
    if source_id is not None:
        payload["source_id"] = source_id

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    return manifest_path


def load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    """Load a manifest JSON file if it exists and is valid."""

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning(
            "Failed to parse manifest JSON", extra={"manifest_path": str(manifest_path)}
        )
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Failed to load manifest", extra={"manifest_path": str(manifest_path), "error": str(exc)}
        )
    return None


def count_csv_rows(data_path: Path) -> int:
    """Count the number of data rows (excluding header) in a CSV file."""

    try:
        with data_path.open("r", encoding="utf-8") as fh:
            # Discard header line
            next(fh, None)
            return sum(1 for _ in fh)
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return 0


def ensure_manifest_for_csv(
    data_path: Path,
    *,
    schema_version: str | None = None,
    source_id: str | None = None,
) -> Dict[str, Any]:
    """Ensure a manifest exists and is up to date for the CSV at ``data_path``."""

    row_count = count_csv_rows(data_path)
    sha256 = compute_sha256(data_path)
    manifest_path = write_csv_manifest(
        data_path,
        row_count=row_count,
        sha256=sha256,
        schema_version=schema_version,
        source_id=source_id,
    )
    manifest = load_manifest(manifest_path)
    return manifest or {
        "format": "csv",
        "row_count": row_count,
        "data_path": data_path.name,
        "sha256": sha256,
    }
