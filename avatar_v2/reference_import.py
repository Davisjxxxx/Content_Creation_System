"""Manual import of externally generated reference images (Midjourney and similar tools).

Midjourney offers no official public API, and its terms prohibit automated access, so Avatar V2
does not automate it: no Discord bots, no browser scraping, no third-party "Midjourney API"
endpoints. Images you generated and downloaded yourself are imported here, validated, hashed,
copied unchanged into a managed store with a provenance record, and registered only as
*pending* candidates. Nothing is accepted into an identity automatically.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .integrity import sha256_file

SOURCES = {
    "midjourney": "Midjourney (manual export from your own account)",
    "other_ai": "Another image generator (manual export)",
    "photo": "Photograph you are authorized to use",
}
FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MIN_SIDE = 512
RECOMMENDED_SIDE = 1024
MAX_SIDE = 8192
MAX_BYTES = 50 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_image(path: Path) -> dict[str, Any]:
    """Return image facts or raise ValueError with a plain-language reason."""
    if not path.is_file():
        raise ValueError("file not found")
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError("larger than 50 MB")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            fmt = (image.format or "").upper()
            width, height = image.size
            frames = getattr(image, "n_frames", 1)
    except Exception as exc:  # PIL raises many exception types for broken files
        raise ValueError(f"not a readable image ({type(exc).__name__})") from exc
    if fmt not in FORMATS:
        raise ValueError(f"unsupported format {fmt or 'unknown'}; use PNG, JPEG or WebP")
    if frames and frames > 1:
        raise ValueError("animated images are not accepted as identity references")
    if min(width, height) < MIN_SIDE:
        raise ValueError(f"{width}x{height} is too small; the short side must be at least {MIN_SIDE} px")
    if max(width, height) > MAX_SIDE:
        raise ValueError(f"{width}x{height} exceeds {MAX_SIDE} px")
    ratio = width / height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError(f"aspect ratio {ratio:.2f} is outside 0.25-4.0")
    warnings = []
    if min(width, height) < RECOMMENDED_SIDE:
        warnings.append(f"short side below {RECOMMENDED_SIDE} px; upscale or re-export at higher resolution for identity work")
    return {"format": fmt, "width": width, "height": height, "bytes": size, "warnings": warnings}


def import_images(
    paths: list[str],
    *,
    store_dir: Path,
    source: str,
    attested: bool,
    prompt: str = "",
    job_reference: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Validate, hash and copy images into ``store_dir`` with provenance sidecars."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Choose one of: {', '.join(SOURCES)}.")
    target_dir = Path(store_dir) / source
    target_dir.mkdir(parents=True, exist_ok=True)
    imported: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in paths or []:
        path = Path(str(raw)).expanduser()
        try:
            facts = validate_image(path)
        except ValueError as exc:
            rejected.append({"file": path.name, "reason": str(exc)})
            continue
        digest = sha256_file(path)
        if not digest or digest in seen:
            rejected.append({"file": path.name, "reason": "duplicate of another selected file" if digest else "unreadable"})
            continue
        seen.add(digest)
        stored = target_dir / f"{digest[:16]}{FORMATS[facts['format']]}"
        sidecar = stored.with_suffix(stored.suffix + ".json")
        duplicate = stored.exists()
        if duplicate and (stored.is_symlink() or sha256_file(stored) != digest):
            rejected.append({"file": path.name, "reason": "Existing import has conflicting content."})
            continue
        if not duplicate:
            from .ownership import copy_file_exclusive

            stored = copy_file_exclusive(path, target_dir, stored.name)
            sidecar = stored.with_suffix(stored.suffix + ".json")
        provenance = {
            "sha256": digest,
            "source": source,
            "source_label": SOURCES[source],
            "original_name": path.name,
            "imported_at": _now(),
            "attested": bool(attested),
            "prompt": prompt[:4000],
            "job_reference": job_reference[:500],
            "notes": notes[:1000],
            **{key: facts[key] for key in ("format", "width", "height", "bytes")},
        }
        if not duplicate or not sidecar.exists():
            from .safe_output import publish_output
            try:
                publish_output(sidecar, lambda path: path.write_text(json.dumps(provenance, indent=2),
                                                                     encoding="utf-8"), rename_on_collision=False)
            except FileExistsError:
                rejected.append({"file": path.name, "reason": "Provenance destination is already occupied."})
                continue
        imported.append(
            {
                **provenance,
                "path": str(stored.resolve()),
                "already_imported": duplicate,
                "warnings": facts["warnings"],
            }
        )
    return {"imported": imported, "rejected": rejected}


def list_imports(store_dir: Path) -> list[dict[str, Any]]:
    items = []
    for sidecar in sorted(Path(store_dir).glob("*/*.json")):
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        image = Path(str(sidecar)[: -len(".json")])
        record["path"] = str(image)
        record["exists"] = image.is_file()
        record["verified"] = record["exists"] and sha256_file(image) == record.get("sha256")
        items.append(record)
    return sorted(items, key=lambda item: item.get("imported_at", ""), reverse=True)
