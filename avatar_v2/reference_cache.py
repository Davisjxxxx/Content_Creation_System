from __future__ import annotations

import re
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
MAX_CACHE_FILES = 500


def _words(path: Path) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).split())


def _has(words: set[str], *values: str) -> bool:
    return any(value in words for value in values)


def infer_reference_role(path: Path) -> tuple[str, str] | None:
    """Infer a conservative reference role from a descriptive filename.

    This intentionally does not pretend to perform visual recognition. Files that
    do not state their view remain unclassified for review.
    """

    words = _words(path)
    left = _has(words, "left", "l")
    right = _has(words, "right", "r")
    front = _has(words, "front", "frontal")
    rear = _has(words, "rear", "back", "behind")

    if _has(words, "hair", "hairstyle"):
        for token, role in (
            ("loose", "loose"),
            ("tied", "tied"),
            ("wind", "wind_example"),
            ("front", "front"),
            ("back", "back"),
            ("side", "side"),
        ):
            if token in words:
                return "hair_refs", role

    if _has(words, "wardrobe", "outfit", "clothing", "look"):
        return "wardrobe_refs", "look"

    if _has(words, "hand", "hands"):
        return "anatomy_refs", "hands"
    if _has(words, "foot", "feet"):
        return "anatomy_refs", "feet"
    if _has(words, "pelvis", "pelvic", "genital", "genitals"):
        if rear:
            return "anatomy_refs", "pelvis_rear"
        if _has(words, "threequarter", "three", "quarter", "34"):
            return "anatomy_refs", "pelvis_three_quarter"
        return "anatomy_refs", "pelvis_front"
    if _has(words, "torso", "navel", "belly", "abdomen"):
        return "anatomy_refs", "torso_rear" if rear else "torso_front"
    if _has(words, "skin", "freckles", "moles", "scar", "tattoo") and _has(
        words, "face", "head", "portrait"
    ):
        return "anatomy_refs", "face_skin"

    face = _has(words, "face", "head", "headshot", "portrait", "selfie", "identity")
    if face:
        if _has(words, "eyesclosed", "closed") and _has(words, "eye", "eyes", "eyesclosed"):
            return "identity_refs", "eyes_closed"
        if _has(words, "expression", "expressions", "smile", "laugh"):
            return "identity_refs", "expressions"
        three_quarter = _has(words, "threequarter", "three", "quarter", "34", "¾")
        if three_quarter and left:
            return "identity_refs", "three_quarter_left"
        if three_quarter and right:
            return "identity_refs", "three_quarter_right"
        if _has(words, "profile", "side") and left:
            return "identity_refs", "profile_left"
        if _has(words, "profile", "side") and right:
            return "identity_refs", "profile_right"
        if front or not (left or right or rear):
            return "identity_refs", "front"

    body = _has(words, "body", "fullbody", "full", "standing", "figure", "pose")
    if body:
        if _has(words, "seated", "sitting", "sit"):
            return "body_refs", "seated"
        if _has(words, "walking", "walk", "stride"):
            return "body_refs", "walking"
        top = _has(words, "top", "overhead", "high")
        low = _has(words, "low", "bottom", "upward")
        if top:
            return "body_refs", "top_rear" if rear else "top_front"
        if low:
            return "body_refs", "low_rear" if rear else "low_front"
        if rear:
            return "body_refs", "rear"
        if left:
            return "body_refs", "left"
        if right:
            return "body_refs", "right"
        if front or not (left or right or rear):
            return "body_refs", "full_front"
    return None


def scan_reference_cache(root: str | Path) -> dict[str, Any]:
    cache_dir = Path(root).expanduser().resolve()
    if not cache_dir.is_dir():
        raise ValueError("reference cache folder does not exist")

    media = {"images": [], "gifs": [], "videos": [], "audio": [], "unsupported": []}
    assignments: dict[str, dict[str, str]] = {
        "identity_refs": {},
        "body_refs": {},
        "hair_refs": {},
        "wardrobe_refs": {},
        "anatomy_refs": {},
    }
    unclassified: list[str] = []
    duplicates: list[str] = []
    files = sorted((path for path in cache_dir.rglob("*") if path.is_file()), key=lambda p: str(p).lower())
    truncated = len(files) > MAX_CACHE_FILES
    for path in files[:MAX_CACHE_FILES]:
        suffix = path.suffix.lower()
        value = str(path)
        if suffix in IMAGE_EXTENSIONS:
            media["images"].append(value)
            inferred = infer_reference_role(path)
            if inferred is None:
                unclassified.append(value)
                continue
            group, role = inferred
            if role == "look":
                role = f"look_{min(len(assignments[group]) + 1, 3)}"
            if role in assignments[group]:
                duplicates.append(value)
                continue
            assignments[group][role] = value
        elif suffix in GIF_EXTENSIONS:
            media["gifs"].append(value)
        elif suffix in VIDEO_EXTENSIONS:
            media["videos"].append(value)
        elif suffix in AUDIO_EXTENSIONS:
            media["audio"].append(value)
        else:
            media["unsupported"].append(value)

    primary = assignments["identity_refs"].get("front") or next(iter(media["images"]), "")
    mapped = sum(len(group) for group in assignments.values())
    warnings: list[str] = []
    if unclassified:
        warnings.append(f"{len(unclassified)} image(s) need angle review because their filenames do not describe a view.")
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate role candidate(s) were left for review.")
    if media["gifs"] or media["videos"]:
        warnings.append("GIF/video files were classified as motion examples, not body identity references.")
    if truncated:
        warnings.append(f"Only the first {MAX_CACHE_FILES} files were analyzed.")
    return {
        "cache_dir": str(cache_dir),
        "media": media,
        "assignments": assignments,
        "primary_source": primary,
        "unclassified_images": unclassified,
        "duplicate_role_candidates": duplicates,
        "mapped_images": mapped,
        "truncated": truncated,
        "warnings": warnings,
        "angle_method": "descriptive_filename",
    }
