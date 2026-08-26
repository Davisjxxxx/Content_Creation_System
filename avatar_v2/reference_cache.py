from __future__ import annotations

import base64
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
GIF_EXTENSIONS = {".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
MAX_CACHE_FILES = 500
DEFAULT_VISION_URL = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "qwen3-vl:2b-instruct"
MIN_VISUAL_CONFIDENCE = 0.60

VISION_ROLE_MAP: dict[str, tuple[str, str]] = {
    "identity_front": ("identity_refs", "front"),
    "identity_three_quarter_left": ("identity_refs", "three_quarter_left"),
    "identity_three_quarter_right": ("identity_refs", "three_quarter_right"),
    "identity_profile_left": ("identity_refs", "profile_left"),
    "identity_profile_right": ("identity_refs", "profile_right"),
    "identity_expressions": ("identity_refs", "expressions"),
    "body_full_front": ("body_refs", "full_front"),
    "body_rear": ("body_refs", "rear"),
    "body_left": ("body_refs", "left"),
    "body_right": ("body_refs", "right"),
    "body_top_front": ("body_refs", "top_front"),
    "body_top_rear": ("body_refs", "top_rear"),
    "body_low_front": ("body_refs", "low_front"),
    "body_low_rear": ("body_refs", "low_rear"),
    "body_seated": ("body_refs", "seated"),
    "body_walking": ("body_refs", "walking"),
    "anatomy_torso_front": ("anatomy_refs", "torso_front"),
    "anatomy_torso_rear": ("anatomy_refs", "torso_rear"),
    "anatomy_pelvis_front": ("anatomy_refs", "pelvis_front"),
    "anatomy_pelvis_three_quarter": ("anatomy_refs", "pelvis_three_quarter"),
    "anatomy_pelvis_rear": ("anatomy_refs", "pelvis_rear"),
    "anatomy_hands": ("anatomy_refs", "hands"),
    "anatomy_feet": ("anatomy_refs", "feet"),
}

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "framing": {
            "type": "string",
            "enum": ["face_closeup", "upper_body", "full_body", "body_detail", "other"],
        },
        "facing": {
            "type": "string",
            "enum": ["front", "rear", "left_profile", "right_profile", "three_quarter_left", "three_quarter_right", "unknown"],
        },
        "camera_pitch": {
            "type": "string",
            "enum": ["overhead", "eye_level", "low_angle", "unknown"],
        },
        "region": {
            "type": "string",
            "enum": ["face", "full_body", "torso", "pelvis", "buttocks", "hands", "feet", "skin", "other"],
        },
        "pose": {"type": "string", "enum": ["standing", "seated", "walking", "other"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "visible_evidence": {"type": "string"},
    },
    "required": ["framing", "facing", "camera_pitch", "region", "pose", "confidence", "visible_evidence"],
}

VISION_PROMPT = """Classify only the camera/view geometry visible in this image, never its filename. Ignore phone and social-media UI. Do not identify the person and do not infer or report sex, gender, name, ethnicity, or identity. Report only framing, facing direction, camera pitch, visible body region, pose, confidence, and short geometric evidence. face_closeup means face dominant; upper_body ends near waist; full_body has knees or feet visible; body_detail has one region dominant. REAR means the person's back faces the camera even if their head turns. OVERHEAD looks down; LOW_ANGLE looks up. LEFT and RIGHT mean the person's visible side. Use unknown or other below 0.60 confidence. Keep visible_evidence under 16 words and non-sexual. Return only schema-valid JSON."""


def _visual_facts_to_role(content: dict[str, Any]) -> str:
    framing = str(content.get("framing") or "other")
    facing = str(content.get("facing") or "unknown")
    pitch = str(content.get("camera_pitch") or "unknown")
    region = str(content.get("region") or "other")
    pose = str(content.get("pose") or "other")

    if region == "hands":
        return "anatomy_hands"
    if region == "feet":
        return "anatomy_feet"
    if region in {"pelvis", "buttocks"}:
        if facing == "rear" or region == "buttocks":
            return "anatomy_pelvis_rear"
        if facing in {"three_quarter_left", "three_quarter_right", "left_profile", "right_profile"}:
            return "anatomy_pelvis_three_quarter"
        return "anatomy_pelvis_front"

    face_dominant = framing == "face_closeup" or (region in {"face", "skin"} and framing != "full_body")
    if face_dominant:
        return {
            "front": "identity_front",
            "rear": "unclassified",
            "left_profile": "identity_profile_left",
            "right_profile": "identity_profile_right",
            "three_quarter_left": "identity_three_quarter_left",
            "three_quarter_right": "identity_three_quarter_right",
        }.get(facing, "unclassified")

    body_visible = framing == "full_body" or region == "full_body"
    if body_visible:
        if pose == "seated":
            return "body_seated"
        if pose == "walking":
            return "body_walking"
        rear = facing == "rear"
        if pitch == "overhead":
            return "body_top_rear" if rear else "body_top_front"
        if pitch == "low_angle":
            return "body_low_rear" if rear else "body_low_front"
        if rear:
            return "body_rear"
        if facing in {"left_profile", "three_quarter_left"}:
            return "body_left"
        if facing in {"right_profile", "three_quarter_right"}:
            return "body_right"
        if facing == "front":
            return "body_full_front"
    if region == "torso" or (framing == "body_detail" and region == "skin"):
        return "anatomy_torso_rear" if facing == "rear" else "anatomy_torso_front"
    return "unclassified"


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


def _local_vision_url(base_url: str) -> str:
    parsed = urlparse(str(base_url).strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("visual classification is restricted to a loopback Ollama server")
    return str(base_url).rstrip("/")


def ollama_vision_available(
    base_url: str = DEFAULT_VISION_URL,
    model: str = DEFAULT_VISION_MODEL,
    *,
    client: httpx.Client | None = None,
) -> bool:
    url = _local_vision_url(base_url)

    def check(active_client: httpx.Client) -> bool:
        response = active_client.get(f"{url}/api/tags")
        response.raise_for_status()
        names = {str(item.get("name") or "") for item in response.json().get("models") or []}
        return model in names

    if client is not None:
        return check(client)
    with httpx.Client(timeout=5) as active_client:
        return check(active_client)


def classify_reference_image(
    path: Path,
    *,
    base_url: str = DEFAULT_VISION_URL,
    model: str = DEFAULT_VISION_MODEL,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = _local_vision_url(base_url)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [encoded]}],
        "format": VISION_SCHEMA,
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 160},
    }

    def classify(active_client: httpx.Client) -> dict[str, Any]:
        response = active_client.post(f"{url}/api/chat", json=payload, timeout=45)
        response.raise_for_status()
        content = json.loads(response.json()["message"]["content"])
        role = _visual_facts_to_role(content)
        try:
            confidence = max(0.0, min(float(content.get("confidence") or 0), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "role": role,
            "confidence": confidence,
            "visible_evidence": str(content.get("visible_evidence") or "").strip()[:500],
            "visual_facts": {
                key: str(content.get(key) or "")
                for key in ("framing", "facing", "camera_pitch", "region", "pose")
            },
        }

    if client is not None:
        return classify(client)
    with httpx.Client(timeout=120) as active_client:
        return classify(active_client)


def scan_reference_cache(
    root: str | Path,
    *,
    use_visual: bool = False,
    visual_classifier: Callable[[Path], dict[str, Any]] | None = None,
    vision_base_url: str = DEFAULT_VISION_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> dict[str, Any]:
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
    low_confidence: list[str] = []
    visual_failures: list[str] = []
    classification_results: list[dict[str, Any]] = []
    assignment_scores: dict[tuple[str, str], float] = {}
    vision_error = ""
    vision_client: httpx.Client | None = None
    created_classifier = False
    if use_visual and visual_classifier is None:
        try:
            vision_client = httpx.Client(timeout=120)
            if ollama_vision_available(vision_base_url, vision_model, client=vision_client):
                created_classifier = True
                visual_classifier = lambda path: classify_reference_image(
                    path,
                    base_url=vision_base_url,
                    model=vision_model,
                    client=vision_client,
                )
            else:
                vision_error = f"local Ollama model {vision_model} is not installed"
        except Exception as exc:
            vision_error = f"{type(exc).__name__}: {exc}"
            if vision_client is not None:
                vision_client.close()
                vision_client = None
    files = sorted((path for path in cache_dir.rglob("*") if path.is_file()), key=lambda p: str(p).lower())
    truncated = len(files) > MAX_CACHE_FILES
    try:
        for path in files[:MAX_CACHE_FILES]:
            suffix = path.suffix.lower()
            value = str(path)
            if suffix in IMAGE_EXTENSIONS:
                media["images"].append(value)
                filename_role = infer_reference_role(path)
                inferred = None
                score = 0.0
                method = "unclassified"
                if visual_classifier is not None:
                    try:
                        result = visual_classifier(path)
                        visual_role = str(result.get("role") or "unclassified")
                        confidence = max(0.0, min(float(result.get("confidence") or 0), 1.0))
                        classification_results.append(
                            {
                                "path": value,
                                "role": visual_role,
                                "confidence": confidence,
                                "visible_evidence": str(result.get("visible_evidence") or "")[:500],
                                "visual_facts": dict(result.get("visual_facts") or {}),
                            }
                        )
                        if visual_role in VISION_ROLE_MAP and confidence >= MIN_VISUAL_CONFIDENCE:
                            inferred = VISION_ROLE_MAP[visual_role]
                            score = confidence
                            method = "local_vision"
                        else:
                            low_confidence.append(value)
                    except Exception as exc:
                        visual_failures.append(value)
                        classification_results.append(
                            {"path": value, "role": "error", "confidence": 0.0, "visible_evidence": f"{type(exc).__name__}: {exc}"[:500]}
                        )
                if inferred is None and filename_role is not None:
                    inferred = filename_role
                    score = 0.55 if visual_classifier is not None else 0.80
                    method = "filename_fallback" if visual_classifier is not None else "descriptive_filename"
                if inferred is None:
                    unclassified.append(value)
                    continue
                group, role = inferred
                if role == "look":
                    role = f"look_{min(len(assignments[group]) + 1, 3)}"
                key = (group, role)
                if role in assignments[group]:
                    if score > assignment_scores.get(key, 0):
                        duplicates.append(assignments[group][role])
                        assignments[group][role] = value
                        assignment_scores[key] = score
                    else:
                        duplicates.append(value)
                    continue
                assignments[group][role] = value
                assignment_scores[key] = score
                if classification_results and classification_results[-1].get("path") == value:
                    classification_results[-1]["selected_method"] = method
            elif suffix in GIF_EXTENSIONS:
                media["gifs"].append(value)
            elif suffix in VIDEO_EXTENSIONS:
                media["videos"].append(value)
            elif suffix in AUDIO_EXTENSIONS:
                media["audio"].append(value)
            else:
                media["unsupported"].append(value)
    finally:
        if created_classifier and vision_client is not None:
            try:
                vision_client.post(
                    f"{_local_vision_url(vision_base_url)}/api/generate",
                    json={"model": vision_model, "keep_alive": 0},
                    timeout=10,
                )
            except Exception:
                pass
            vision_client.close()

    primary = (
        assignments["identity_refs"].get("front")
        or next(iter(assignments["identity_refs"].values()), "")
        or next(iter(media["images"]), "")
    )
    mapped = sum(len(group) for group in assignments.values())
    warnings: list[str] = []
    if classification_results:
        warnings.append(
            f"{len(classification_results) - len(visual_failures)} image(s) were inspected locally by {vision_model}; no images were uploaded."
        )
    if vision_error:
        warnings.append(f"Local visual classifier unavailable ({vision_error}); descriptive-filename fallback was used.")
    if visual_failures:
        warnings.append(f"Visual analysis failed for {len(visual_failures)} image(s); filename fallback was attempted.")
    if low_confidence:
        warnings.append(f"{len(low_confidence)} image(s) had low-confidence visual results and need review unless a filename supplied the role.")
    if unclassified:
        warnings.append(f"{len(unclassified)} image(s) still need review because neither vision nor filenames produced a confident role.")
    if duplicates:
        warnings.append(f"{len(duplicates)} lower-ranked duplicate role candidate(s) were left for review.")
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
        "low_confidence_images": low_confidence,
        "visual_failures": visual_failures,
        "classification_results": classification_results,
        "duplicate_role_candidates": duplicates,
        "mapped_images": mapped,
        "truncated": truncated,
        "warnings": warnings,
        "angle_method": "local_ollama_vision" if classification_results else "descriptive_filename",
        "vision_model": vision_model if classification_results else "",
        "visual_images_analyzed": len(classification_results) - len(visual_failures),
        "vision_error": vision_error,
    }
