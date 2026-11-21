from __future__ import annotations

import base64
import hashlib
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
MIN_SOURCE_QUALITY = 0.60
MIN_IDENTITY_CONFIDENCE = 0.65

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
        "usable_regions": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["face", "full_body", "torso", "pelvis", "buttocks", "hands", "feet", "skin"],
            },
            "maxItems": 8,
        },
        "pose": {"type": "string", "enum": ["standing", "seated", "walking", "other"]},
        "face_visibility": {"type": "string", "enum": ["clear", "partial", "absent", "unknown"]},
        "phone_ui": {"type": "boolean"},
        "watermark": {"type": "boolean"},
        "heavy_occlusion": {"type": "boolean"},
        "source_quality": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "framing", "facing", "camera_pitch", "region", "usable_regions", "pose", "face_visibility",
        "phone_ui", "watermark", "heavy_occlusion", "source_quality", "confidence",
    ],
}

VISION_PROMPT = """Classify only view geometry and source suitability visible in this image, never its filename. Do not identify the person and do not infer or report sex, gender, name, ethnicity, or identity. Return only the requested JSON fields. Source quality measures whether the visible face or body geometry is usable as an avatar reference: complete relevant region, useful detail, and minimal crop or occlusion. Report phone UI and watermark separately. Do not lower source quality solely for phone chrome or a watermark when the subject geometry remains clear and detailed. `region` is the dominant region. `usable_regions` must list every separate region whose geometry is clearly visible and detailed enough to constrain generation; do not omit a complete region merely because another region is dominant. A named region is usable only when that region itself is inside the frame: hands require complete fingers and palm/back, feet require complete toes and heel, face requires the complete face, torso requires chest through waist, and pelvis requires the complete visible pelvic region. Never infer a region from nearby limbs, pose, or what would exist beyond the crop. Do not include any cropped, covered, blurred, or tiny region. Face visibility is clear whenever the entire face is sharp and unobstructed, even inside an upper-body or full-body image; partial means the face itself is cropped, covered, blurred, or too small to compare. face_closeup means face dominant; upper_body ends near waist; full_body has knees or feet visible; body_detail has one region dominant. REAR means the person's back faces the camera even if their head turns. OVERHEAD looks down; LOW_ANGLE looks up. LEFT and RIGHT mean the person's visible side. Use unknown or other below 0.60 confidence. Return only schema-valid JSON."""

IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "same_subject": {"type": "string", "enum": ["same", "different", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["same_subject", "confidence"],
}

IDENTITY_PROMPT = """Compare Picture 1 and Picture 2 only to decide whether they show the same visual subject. This is a private consistency check, not identification: never name, recognize, search for, or report demographic traits about the person. Ignore clothing, pose, background, phone UI, hairstyle changes, and expression. Compare intrinsic visible face geometry and stable distinguishing features. Return different when visible geometry conflicts, uncertain when either image lacks enough comparable face evidence, and same only with clear support. Return only the requested schema-valid JSON fields."""


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        # A visible pelvis is not automatically a neutral anatomy-sheet view.
        # Seated and pitched-camera sources change the apparent geometry too
        # much to label as a straight-on/three-quarter/rear canonical slot.
        if pitch != "eye_level" or pose not in {"standing", "other"}:
            return "unclassified"
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


def _visual_facts_to_roles(content: dict[str, Any]) -> list[str]:
    """Return every supported profile role visibly evidenced by one image."""

    roles: list[str] = []

    def add(role: str) -> None:
        if role in VISION_ROLE_MAP and role not in roles:
            roles.append(role)

    add(_visual_facts_to_role(content))
    visible = content.get("usable_regions") or []
    if not isinstance(visible, list):
        visible = []
    face_visibility = str(content.get("face_visibility") or "unknown")
    for region in visible:
        region = str(region)
        if region in {"face", "skin"} and face_visibility not in {"clear", "partial"}:
            continue
        candidate = dict(content)
        candidate["region"] = region
        if region in {"face", "skin"}:
            candidate["framing"] = "face_closeup"
        elif region == "full_body":
            candidate["framing"] = "full_body"
        else:
            candidate["framing"] = "body_detail"
        add(_visual_facts_to_role(candidate))
    return roles


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


def _encoded_reference_image(path: Path) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _quality_result(result: dict[str, Any], role: str) -> tuple[float, list[str], bool]:
    try:
        raw_quality = max(0.0, min(float(result.get("quality_score", result.get("source_quality", 1.0))), 1.0))
    except (TypeError, ValueError):
        raw_quality = 0.0
    facts = dict(result.get("visual_facts") or {})
    phone_ui = bool(result.get("phone_ui", facts.get("phone_ui", False)))
    watermark = bool(result.get("watermark", facts.get("watermark", False)))
    heavy_occlusion = bool(result.get("heavy_occlusion", facts.get("heavy_occlusion", False)))
    face_visibility = str(
        result.get("face_visibility")
        or facts.get("face_visibility")
        or ("clear" if role.startswith("identity_") else "unknown")
    )
    # The model reports UI and watermarks separately and already scores whether
    # the subject geometry is usable. Do not subtract the same concern twice.
    quality = raw_quality
    reasons: list[str] = []
    if quality < MIN_SOURCE_QUALITY:
        reasons.append("source quality below 60% after UI/watermark penalties")
    if heavy_occlusion:
        reasons.append("subject is heavily occluded")
    if role.startswith("identity_") and face_visibility != "clear":
        reasons.append("identity views require a clearly visible complete face")
    return quality, reasons, not reasons


def classify_reference_image(
    path: Path,
    *,
    base_url: str = DEFAULT_VISION_URL,
    model: str = DEFAULT_VISION_MODEL,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = _local_vision_url(base_url)
    encoded = _encoded_reference_image(path)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": VISION_PROMPT, "images": [encoded]}],
        "format": VISION_SCHEMA,
        "stream": False,
        "keep_alive": "10m" if client is not None else 0,
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 240},
    }

    def classify(active_client: httpx.Client) -> dict[str, Any]:
        response = active_client.post(f"{url}/api/chat", json=payload, timeout=45)
        response.raise_for_status()
        content = json.loads(response.json()["message"]["content"])
        roles = _visual_facts_to_roles(content)
        role = roles[0] if roles else "unclassified"
        try:
            confidence = max(0.0, min(float(content.get("confidence") or 0), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            source_quality = max(0.0, min(float(content.get("source_quality") or 0), 1.0))
        except (TypeError, ValueError):
            source_quality = 0.0
        visual_facts = {
            key: str(content.get(key) or "")
            for key in ("framing", "facing", "camera_pitch", "region", "pose", "face_visibility")
        }
        visual_facts["usable_regions"] = [
            str(value) for value in (content.get("usable_regions") or []) if str(value)
        ][:8]
        visual_facts.update(
            {
                "phone_ui": bool(content.get("phone_ui")),
                "watermark": bool(content.get("watermark")),
                "heavy_occlusion": bool(content.get("heavy_occlusion")),
            }
        )
        return {
            "role": role,
            "roles": roles,
            "confidence": confidence,
            "source_quality": source_quality,
            "face_visibility": visual_facts["face_visibility"],
            "phone_ui": visual_facts["phone_ui"],
            "watermark": visual_facts["watermark"],
            "heavy_occlusion": visual_facts["heavy_occlusion"],
            "visible_evidence": ", ".join(
                str(visual_facts[key]) for key in ("framing", "facing", "camera_pitch", "region", "pose")
            )[:500],
            "visual_facts": visual_facts,
        }

    if client is not None:
        return classify(client)
    with httpx.Client(timeout=120) as active_client:
        return classify(active_client)


def compare_reference_identity(
    anchor: Path,
    candidate: Path,
    *,
    base_url: str = DEFAULT_VISION_URL,
    model: str = DEFAULT_VISION_MODEL,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = _local_vision_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": IDENTITY_PROMPT,
                "images": [_encoded_reference_image(anchor), _encoded_reference_image(candidate)],
            }
        ],
        "format": IDENTITY_SCHEMA,
        "stream": False,
        "keep_alive": "10m" if client is not None else 0,
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 256},
    }

    def compare(active_client: httpx.Client) -> dict[str, Any]:
        response = active_client.post(f"{url}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        content = json.loads(response.json()["message"]["content"])
        verdict = str(content.get("same_subject") or "uncertain")
        if verdict not in {"same", "different", "uncertain"}:
            verdict = "uncertain"
        try:
            confidence = max(0.0, min(float(content.get("confidence") or 0), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "same_subject": verdict,
            "confidence": confidence,
            "visible_evidence": "local non-identifying same-subject comparison",
        }

    if client is not None:
        return compare(client)
    with httpx.Client(timeout=120) as active_client:
        return compare(active_client)


def scan_reference_cache(
    root: str | Path,
    *,
    selected_files: list[str | Path] | tuple[str | Path, ...] | None = None,
    use_visual: bool = False,
    visual_classifier: Callable[[Path], dict[str, Any]] | None = None,
    identity_comparator: Callable[[Path, Path], dict[str, Any]] | None = None,
    vision_base_url: str = DEFAULT_VISION_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
    single_subject_confirmed: bool = False,
) -> dict[str, Any]:
    selection_mode = "files" if selected_files is not None else "folder"
    if selected_files is not None:
        files = sorted(
            {Path(value).expanduser().resolve() for value in selected_files},
            key=lambda path: str(path).lower(),
        )
        if not files:
            raise ValueError("select at least one reference image")
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"selected reference image does not exist: {missing[0]}")
        unsupported = [path for path in files if path.suffix.lower() not in IMAGE_EXTENSIONS]
        if unsupported:
            raise ValueError(f"selected reference file is not a supported still image: {unsupported[0].name}")
        cache_dir = Path(root).expanduser().resolve() if str(root).strip() else files[0].parent
    else:
        cache_dir = Path(root).expanduser().resolve()
        if not cache_dir.is_dir():
            raise ValueError("reference cache folder does not exist")
        files = sorted((path for path in cache_dir.rglob("*") if path.is_file()), key=lambda path: str(path).lower())

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
    quality_rejected: list[str] = []
    source_warning_images: list[str] = []
    identity_quarantined: list[str] = []
    duplicate_content_images: list[str] = []
    content_digest_owner: dict[str, str] = {}
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
                identity_comparator = lambda anchor, candidate: compare_reference_identity(
                    anchor,
                    candidate,
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
    truncated = len(files) > MAX_CACHE_FILES
    try:
        for path in files[:MAX_CACHE_FILES]:
            suffix = path.suffix.lower()
            value = str(path)
            if suffix in IMAGE_EXTENSIONS:
                media["images"].append(value)
                digest = _file_digest(path)
                if digest in content_digest_owner:
                    duplicate_content_images.append(value)
                else:
                    content_digest_owner[digest] = value
                filename_role = infer_reference_role(path)
                inferred = None
                score = 0.0
                method = "unclassified"
                if visual_classifier is not None:
                    try:
                        result = visual_classifier(path)
                        visual_role = str(result.get("role") or "unclassified")
                        raw_roles = result.get("roles") or [visual_role]
                        visual_roles = list(
                            dict.fromkeys(
                                str(role)
                                for role in raw_roles
                                if str(role) in VISION_ROLE_MAP
                            )
                        )
                        confidence = max(0.0, min(float(result.get("confidence") or 0), 1.0))
                        role_quality = {
                            role: _quality_result(result, role)
                            for role in visual_roles
                        }
                        eligible_roles = [
                            role
                            for role, (_, _, passed) in role_quality.items()
                            if passed and confidence >= MIN_VISUAL_CONFIDENCE
                        ]
                        quality_score = max(
                            (quality for quality, _, passed in role_quality.values() if passed),
                            default=_quality_result(result, visual_role)[0],
                        )
                        quality_pass = bool(eligible_roles)
                        quality_reasons = list(
                            dict.fromkeys(
                                reason
                                for _, reasons, passed in role_quality.values()
                                if not passed
                                for reason in reasons
                            )
                        )
                        facts = dict(result.get("visual_facts") or {})
                        source_flags = [
                            label
                            for label, present in (
                                ("phone_ui", bool(result.get("phone_ui", facts.get("phone_ui", False)))),
                                ("watermark", bool(result.get("watermark", facts.get("watermark", False)))),
                            )
                            if present
                        ]
                        classification_results.append(
                            {
                                "path": value,
                                "role": visual_role,
                                "roles": visual_roles,
                                "eligible_roles": eligible_roles,
                                "confidence": confidence,
                                "quality_score": quality_score,
                                "quality_pass": quality_pass,
                                "quality_reasons": quality_reasons,
                                "source_flags": source_flags,
                                "face_visibility": str(
                                    result.get("face_visibility")
                                    or (result.get("visual_facts") or {}).get("face_visibility")
                                    or ("clear" if visual_role.startswith("identity_") else "unknown")
                                ),
                                "visible_evidence": str(result.get("visible_evidence") or "")[:500],
                                "visual_facts": dict(result.get("visual_facts") or {}),
                            }
                        )
                        if not quality_pass:
                            quality_rejected.append(value)
                        elif source_flags:
                            source_warning_images.append(value)
                        if eligible_roles:
                            inferred = VISION_ROLE_MAP[eligible_roles[0]]
                            score = confidence * quality_score
                            method = "local_vision"
                        else:
                            low_confidence.append(value)
                    except Exception as exc:
                        visual_failures.append(value)
                        classification_results.append(
                            {"path": value, "role": "error", "confidence": 0.0, "visible_evidence": f"{type(exc).__name__}: {exc}"[:500]}
                        )
                quality_blocked = bool(
                    classification_results
                    and classification_results[-1].get("path") == value
                    and not classification_results[-1].get("quality_pass", True)
                )
                if inferred is None and filename_role is not None and not quality_blocked:
                    inferred = filename_role
                    score = 0.55 if visual_classifier is not None else 0.80
                    method = "filename_fallback" if visual_classifier is not None else "descriptive_filename"
                if inferred is None:
                    unclassified.append(value)
                    continue
                inferred_roles = [
                    VISION_ROLE_MAP[role]
                    for role in (
                        classification_results[-1].get("eligible_roles") or []
                        if classification_results and classification_results[-1].get("path") == value
                        else []
                    )
                ] or [inferred]
                for group, role in inferred_roles:
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

        identity_anchor = ""
        identity_gate_active = identity_comparator is not None
        if classification_results:
            anchor_candidates = [
                item
                for item in classification_results
                if item.get("quality_pass")
                and item.get("face_visibility") in {"clear", "partial"}
                and float(item.get("confidence") or 0) >= MIN_VISUAL_CONFIDENCE
            ]
            if anchor_candidates:
                anchor_item = max(
                    anchor_candidates,
                    key=lambda item: float(item.get("confidence") or 0) * float(item.get("quality_score") or 0),
                )
                identity_anchor = str(anchor_item["path"])
            for item in classification_results:
                eligible_roles = [
                    str(role) for role in (item.get("eligible_roles") or []) if str(role) in VISION_ROLE_MAP
                ]
                base_eligible = bool(
                    item.get("quality_pass")
                    and eligible_roles
                    and float(item.get("confidence") or 0) >= MIN_VISUAL_CONFIDENCE
                )
                item["identity_gate_pass"] = not identity_gate_active
                item["identity_status"] = "not_checked"
                item["identity_confidence"] = 0.0
                if identity_gate_active and base_eligible:
                    if not identity_anchor:
                        item["identity_status"] = "no_clear_anchor"
                    elif item["path"] == identity_anchor:
                        item["identity_gate_pass"] = True
                        item["identity_status"] = "anchor"
                        item["identity_confidence"] = 1.0
                    elif item.get("face_visibility") not in {"clear", "partial"}:
                        if single_subject_confirmed:
                            item["identity_gate_pass"] = True
                            item["identity_status"] = "confirmed_no_face"
                            item["identity_confidence"] = MIN_IDENTITY_CONFIDENCE
                        else:
                            item["identity_status"] = "unverified_no_face"
                    else:
                        try:
                            comparison = identity_comparator(Path(identity_anchor), Path(str(item["path"])))
                            verdict = str(comparison.get("same_subject") or "uncertain")
                            identity_confidence = max(
                                0.0, min(float(comparison.get("confidence") or 0), 1.0)
                            )
                            item["identity_status"] = verdict
                            item["identity_confidence"] = identity_confidence
                            item["identity_evidence"] = str(comparison.get("visible_evidence") or "")[:500]
                            item["identity_gate_pass"] = bool(
                                verdict == "same" and identity_confidence >= MIN_IDENTITY_CONFIDENCE
                            )
                        except Exception as exc:
                            item["identity_status"] = "comparison_error"
                            item["identity_evidence"] = f"{type(exc).__name__}: {exc}"[:500]
                    if not item["identity_gate_pass"]:
                        identity_quarantined.append(str(item["path"]))
                item["generation_eligible"] = bool(base_eligible and item["identity_gate_pass"])

            if identity_gate_active:
                for group in ("identity_refs", "body_refs", "anatomy_refs"):
                    assignments[group] = {}
                gated_scores: dict[tuple[str, str], float] = {}
                for item in classification_results:
                    if not item.get("generation_eligible"):
                        continue
                    for visual_role in item.get("eligible_roles") or []:
                        group, role = VISION_ROLE_MAP[str(visual_role)]
                        key = (group, role)
                        score = (
                            float(item.get("confidence") or 0)
                            * float(item.get("quality_score") or 0)
                            * max(float(item.get("identity_confidence") or 0), MIN_IDENTITY_CONFIDENCE)
                        )
                        if score > gated_scores.get(key, -1):
                            assignments[group][role] = str(item["path"])
                            gated_scores[key] = score
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
    if quality_rejected:
        warnings.append(f"{len(set(quality_rejected))} image(s) failed the source-quality gate and were held out of automatic generation.")
    if source_warning_images:
        warnings.append(
            f"{len(set(source_warning_images))} usable image(s) include phone UI or a watermark; crop them when practical to reduce transfer artifacts."
        )
    if identity_gate_active and not identity_anchor:
        warnings.append("Identity gate found no clean, complete face anchor; Quick Build must stop for review.")
    elif identity_gate_active and identity_quarantined:
        warnings.append(
            f"{len(set(identity_quarantined))} image(s) were held because the local same-subject comparison failed or lacked enough face evidence."
        )
    if single_subject_confirmed:
        warnings.append(
            "Single-avatar cache confirmation admitted usable body-only views without a comparable face; visible face mismatches remain held."
        )
    if duplicate_content_images:
        warnings.append(
            f"{len(duplicate_content_images)} selected image path(s) repeat byte-identical content and add no new visual evidence."
        )
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
        "selection_mode": selection_mode,
        "selected_file_count": len(files) if selection_mode == "files" else 0,
        "selected_image_count": len(media["images"]),
        "distinct_image_count": len(content_digest_owner),
        "duplicate_content_images": duplicate_content_images,
        "media": media,
        "assignments": assignments,
        "primary_source": primary,
        "unclassified_images": unclassified,
        "low_confidence_images": low_confidence,
        "quality_rejected_images": sorted(set(quality_rejected)),
        "source_warning_images": sorted(set(source_warning_images)),
        "identity_quarantined_images": sorted(set(identity_quarantined)),
        "identity_anchor": identity_anchor,
        "identity_gate_active": identity_gate_active,
        "single_subject_confirmed": bool(single_subject_confirmed),
        "generation_eligible_images": [
            str(item["path"]) for item in classification_results if item.get("generation_eligible")
        ],
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
