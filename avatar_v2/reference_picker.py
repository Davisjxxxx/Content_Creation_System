"""Shared local reference-image validation, access control, and controlled job staging (Sprint 4).

Allows users to select starter images and reference assets directly from local filesystem
directories (Pictures, Downloads, external storage, etc.) without importing them into My Stuff,
while preventing unrestricted filesystem access through frontend-supplied paths and preserving
original source files untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)
SUPPORTED_PIL_FORMATS: frozenset[str] = frozenset(
    {"PNG", "JPEG", "MPO", "WEBP", "BMP", "TIFF"}
)
MAX_REFERENCE_FILE_BYTES: int = 50 * 1024 * 1024  # 50 MiB
MAX_REFERENCE_PIXELS: int = 40_000_000  # ~6324x6324 decompression-bomb guard
MIN_REFERENCE_DIMENSION: int = 64

# Workflow compatibility constraints for reference images.
WORKFLOW_REFERENCE_LIMITS: dict[str, dict[str, Any]] = {
    "single_start": {"max_images": 1, "min_dim": 64, "label": "Start image"},
    "image_to_video": {"max_images": 1, "min_dim": 128, "label": "Image-to-video starter image"},
    "image_edit": {"max_images": 2, "min_dim": 64, "label": "Image editing"},
    "avatar_generation": {"max_images": 4, "min_dim": 128, "label": "Avatar generation"},
    "image_to_image": {"max_images": 2, "min_dim": 64, "label": "Image-to-image"},
    "character_reference": {"max_images": 4, "min_dim": 128, "label": "Character reference"},
    "multi_reference": {"max_images": 6, "min_dim": 64, "label": "Multi-reference workflow"},
    "guided": {"max_images": 4, "min_dim": 64, "label": "Guided generation"},
    "script": {"max_images": 8, "min_dim": 64, "label": "Script reference assets"},
}


@dataclass(frozen=True)
class ValidatedReference:
    """Validation result for one local reference image."""

    ok: bool
    path: str
    name: str
    width: int = 0
    height: int = 0
    format: str = ""
    size_bytes: int = 0
    sha256: str = ""
    source: str = "local"  # "local" | "my_stuff"
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_image(
    raw_path: str | Path,
    *,
    workflow: str = "single_start",
    source: str = "local",
) -> ValidatedReference:
    """Validate format, readability, dimensions, file size, and workflow compatibility.

    Never modifies the original file. Handles missing, moved, corrupted, or inaccessible files gracefully.
    """
    text = str(raw_path or "").strip()
    if not text:
        return ValidatedReference(ok=False, path="", name="", source=source, error="No image path was provided.")

    candidate = Path(text).expanduser()
    name = candidate.name or text
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return ValidatedReference(
            ok=False,
            path=str(candidate),
            name=name,
            source=source,
            error=f"Image '{name}' is missing or was moved.",
        )
    except PermissionError:
        return ValidatedReference(
            ok=False,
            path=str(candidate),
            name=name,
            source=source,
            error=f"Permission denied reading '{name}'.",
        )
    except OSError as exc:
        return ValidatedReference(
            ok=False,
            path=str(candidate),
            name=name,
            source=source,
            error=f"Could not access '{name}': {exc}",
        )

    if not resolved.is_file():
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            source=source,
            error=f"'{resolved.name}' is not a regular image file.",
        )

    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            source=source,
            error=(
                f"Unsupported image format '{suffix or 'unknown'}'. "
                "Choose a PNG, JPEG, WebP, BMP, or TIFF file."
            ),
        )

    try:
        size_bytes = resolved.stat().st_size
    except OSError as exc:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            source=source,
            error=f"Could not read file size for '{resolved.name}': {exc}",
        )

    if size_bytes <= 0:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            source=source,
            error=f"Image '{resolved.name}' is empty (0 bytes).",
        )
    if size_bytes > MAX_REFERENCE_FILE_BYTES:
        mb = round(size_bytes / (1024 * 1024), 1)
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            source=source,
            error=f"Image '{resolved.name}' is too large ({mb} MB; limit is 50 MB).",
        )

    try:
        with Image.open(resolved) as opened:
            fmt = str(opened.format or "").upper()
            width, height = opened.size
            if fmt and fmt not in SUPPORTED_PIL_FORMATS:
                return ValidatedReference(
                    ok=False,
                    path=str(resolved),
                    name=resolved.name,
                    size_bytes=size_bytes,
                    source=source,
                    error=f"Unsupported decoded image format '{fmt}'.",
                )
            if width * height > MAX_REFERENCE_PIXELS:
                return ValidatedReference(
                    ok=False,
                    path=str(resolved),
                    name=resolved.name,
                    width=width,
                    height=height,
                    format=fmt,
                    size_bytes=size_bytes,
                    source=source,
                    error=f"Image '{resolved.name}' exceeds safe pixel limit ({width}x{height}).",
                )
            # Fully decode a transposed copy in memory to catch truncated/corrupted files cleanly.
            transposed = ImageOps.exif_transpose(opened)
            width, height = transposed.size
            transposed.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            size_bytes=size_bytes,
            source=source,
            error=f"Image '{resolved.name}' appears corrupted or unreadable ({type(exc).__name__}).",
        )

    wf_rule = WORKFLOW_REFERENCE_LIMITS.get(workflow, WORKFLOW_REFERENCE_LIMITS["single_start"])
    min_dim = int(wf_rule.get("min_dim", MIN_REFERENCE_DIMENSION))
    if min(width, height) < min_dim:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            width=width,
            height=height,
            format=fmt,
            size_bytes=size_bytes,
            source=source,
            error=(
                f"Image '{resolved.name}' is too small ({width}x{height}) for {wf_rule['label']} "
                f"(minimum {min_dim}px on shortest edge)."
            ),
        )

    warnings: list[str] = []
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > 3.0:
        warnings.append(f"Extreme aspect ratio ({width}x{height}) may be cropped by the model.")
    if min(width, height) < 384:
        warnings.append(f"Low resolution ({width}x{height}); higher-resolution photos give cleaner results.")

    try:
        sha256 = _sha256_file(resolved)
    except OSError as exc:
        return ValidatedReference(
            ok=False,
            path=str(resolved),
            name=resolved.name,
            width=width,
            height=height,
            format=fmt,
            size_bytes=size_bytes,
            source=source,
            error=f"Could not hash '{resolved.name}': {exc}",
        )

    return ValidatedReference(
        ok=True,
        path=str(resolved),
        name=resolved.name,
        width=width,
        height=height,
        format=fmt or suffix.lstrip(".").upper(),
        size_bytes=size_bytes,
        sha256=sha256,
        source=source,
        warnings=warnings,
    )


def validate_reference_batch(
    paths: Iterable[str | Path],
    *,
    workflow: str = "single_start",
    source: str = "local",
) -> dict[str, Any]:
    """Validate a list of one or more reference images against workflow multiplicity & format limits."""
    wf_rule = WORKFLOW_REFERENCE_LIMITS.get(workflow, WORKFLOW_REFERENCE_LIMITS["single_start"])
    max_images = int(wf_rule.get("max_images", 1))
    raw_list = [str(p) for p in paths if str(p or "").strip()]
    errors: list[str] = []
    if len(raw_list) > max_images:
        errors.append(
            f"{wf_rule['label']} supports at most {max_images} image{'s' if max_images == 1 else ''}; "
            f"keeping the first {max_images}."
        )
        raw_list = raw_list[:max_images]

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_list:
        checked = validate_local_image(raw, workflow=workflow, source=source)
        if not checked.ok:
            errors.append(checked.error)
            continue
        if checked.path in seen:
            continue
        seen.add(checked.path)
        items.append(checked.public())

    return {
        "ok": bool(items) and not (len(raw_list) > 0 and not items),
        "workflow": workflow,
        "max_images": max_images,
        "items": items,
        "errors": errors,
    }


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


class LocalReferenceManager:
    """Tracks user-approved external reference paths and handles controlled job staging.

    Security boundary:
    - The frontend cannot arbitrarily read, thumbnail, or stage files from the host filesystem.
    - External paths must either live inside an approved Avatar V2 media root (My Stuff / archive)
      or have been explicitly selected by the user via the native OS file dialog and registered here.
    - Staging deduplicates by SHA-256 inside an ephemeral staging directory and records original
      provenance without modifying original user files or creating permanent duplicates.
    """

    def __init__(
        self,
        *,
        staging_dir: Path,
        trusted_roots: Iterable[Path] = (),
        max_staged_files: int = 128,
    ) -> None:
        self.staging_dir = Path(staging_dir)
        self._trusted_roots: list[Path] = [Path(r).expanduser() for r in trusted_roots]
        self._max_staged_files = max(8, int(max_staged_files))
        self._lock = threading.RLock()
        self._approved: dict[str, ValidatedReference] = {}

    def add_trusted_root(self, root: Path) -> None:
        with self._lock:
            p = Path(root).expanduser()
            if p not in self._trusted_roots:
                self._trusted_roots.append(p)

    def register_selected(
        self,
        paths: Iterable[str | Path],
        *,
        workflow: str = "single_start",
    ) -> dict[str, Any]:
        """Validate and approve paths returned by the native OS file picker."""
        batch = validate_reference_batch(paths, workflow=workflow, source="local")
        with self._lock:
            for item in batch["items"]:
                ref = ValidatedReference(**item)
                self._approved[ref.path] = ref
        return batch

    def is_path_authorized(self, raw_path: str | Path) -> bool:
        """True if the path was explicitly selected by the user or resides under a trusted media root."""
        text = str(raw_path or "").strip()
        if not text:
            return False
        try:
            resolved = Path(text).expanduser().resolve(strict=False)
        except OSError:
            return False
        with self._lock:
            if str(resolved) in self._approved:
                return True
            if _is_within_root(resolved, self.staging_dir):
                return True
            if any(_is_within_root(resolved, root) for root in self._trusted_roots):
                # Trusted roots may authorize ONLY media assets — never app configs, logs, or ledgers (RC-03).
                if resolved.is_dir():
                    return True
                suffix = resolved.suffix.lower()
                return suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
            return False

    def authorize_existing(self, raw_path: str | Path, *, workflow: str = "single_start") -> ValidatedReference:
        """Verify that a path is authorized AND still exists/valid on disk."""
        if not self.is_path_authorized(raw_path):
            return ValidatedReference(
                ok=False,
                path=str(raw_path or ""),
                name=Path(str(raw_path or "")).name,
                error="That image path was not selected through Avatar V2. Use Browse Local Files or My Stuff.",
            )
        source = "my_stuff"
        resolved = Path(str(raw_path)).expanduser().resolve(strict=False)
        with self._lock:
            if str(resolved) in self._approved:
                source = "local"
        checked = validate_local_image(resolved, workflow=workflow, source=source)
        if checked.ok and source == "local":
            with self._lock:
                self._approved[checked.path] = checked
        return checked

    def stage_for_job(
        self,
        raw_path: str | Path,
        *,
        workflow: str = "single_start",
        require_authorization: bool = True,
    ) -> dict[str, Any]:
        """Stage a validated external reference image into a controlled deduplicated staging cache.

        - Original source files are NEVER modified.
        - Files already inside trusted Avatar V2 roots (My Stuff) are referenced directly without copying.
        - External files are staged once per content SHA-256 so queued jobs remain resilient if the
          user moves/unmounts the original folder, while avoiding permanent duplicate clutter.
        """
        if require_authorization:
            checked = self.authorize_existing(raw_path, workflow=workflow)
        else:
            checked = validate_local_image(raw_path, workflow=workflow, source="local")
        if not checked.ok:
            raise ValueError(checked.error)

        resolved = Path(checked.path)
        with self._lock:
            in_trusted = any(_is_within_root(resolved, root) for root in self._trusted_roots)
        if in_trusted:
            return {
                "staged_path": str(resolved),
                "original_path": str(resolved),
                "copied": False,
                "sha256": checked.sha256,
                "width": checked.width,
                "height": checked.height,
                "format": checked.format,
                "source": "my_stuff",
            }

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        suffix = resolved.suffix.lower()
        target = self.staging_dir / f"ref_{checked.sha256[:20]}{suffix}"
        meta_path = self.staging_dir / f"ref_{checked.sha256[:20]}.json"

        with self._lock:
            copied = False
            if not target.is_file() or _sha256_file(target) != checked.sha256:
                tmp = self.staging_dir / f".tmp_{checked.sha256[:20]}_{os.getpid()}{suffix}"
                shutil.copy2(resolved, tmp)
                os.replace(tmp, target)
                copied = True
            else:
                try:
                    os.utime(target, None)
                except OSError:
                    pass

            provenance = {
                "original_path": str(resolved),
                "original_name": resolved.name,
                "staged_path": str(target),
                "sha256": checked.sha256,
                "width": checked.width,
                "height": checked.height,
                "format": checked.format,
                "size_bytes": checked.size_bytes,
                "staged_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            self._prune_staging_locked()

        return {
            "staged_path": str(target),
            "original_path": str(resolved),
            "copied": copied,
            "sha256": checked.sha256,
            "width": checked.width,
            "height": checked.height,
            "format": checked.format,
            "source": "local",
        }

    def _prune_staging_locked(self) -> None:
        """Keep the ephemeral staging cache bounded by LRU mtime."""
        if not self.staging_dir.is_dir():
            return
        media_files = [
            p
            for p in self.staging_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        if len(media_files) <= self._max_staged_files:
            return
        media_files.sort(key=lambda p: p.stat().st_mtime)
        for stale in media_files[: len(media_files) - self._max_staged_files]:
            try:
                stale.unlink(missing_ok=True)
                stale.with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                pass


def default_browse_directory(remembered: str | None = None) -> str:
    """Resolve the best initial directory for native image browsing (Pictures/Downloads/Home)."""
    if remembered:
        p = Path(remembered).expanduser()
        if p.is_dir():
            return str(p)
    for candidate in (Path.home() / "Pictures", Path.home() / "Downloads", Path.home()):
        if candidate.is_dir():
            return str(candidate)
    return str(Path.cwd())
