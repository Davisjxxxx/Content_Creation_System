"""File-integrity helpers: content hashes for identity references and render outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or None when it cannot be read."""
    target = Path(str(path)).expanduser()
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def hash_files(paths: list[str]) -> dict[str, str]:
    """Hash every readable file; unreadable or missing files are skipped."""
    hashes: dict[str, str] = {}
    for path in paths:
        value = sha256_file(path)
        if value:
            hashes[str(path)] = value
    return hashes


def verify_hashes(expected: dict[str, str]) -> dict[str, Any]:
    """Recompute recorded hashes. Reports files that match, changed, or went missing."""
    results: dict[str, str] = {}
    for path, recorded in (expected or {}).items():
        current = sha256_file(path)
        if current is None:
            results[path] = "missing"
        elif current == recorded:
            results[path] = "verified"
        else:
            results[path] = "changed"
    return {
        "ok": bool(results) and all(state == "verified" for state in results.values()),
        "results": results,
        "verified": sum(1 for state in results.values() if state == "verified"),
        "changed": sorted(path for path, state in results.items() if state == "changed"),
        "missing": sorted(path for path, state in results.items() if state == "missing"),
    }
