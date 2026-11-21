"""Runway developer API adapter: an explicit, opt-in cloud provider.

Built against the Runway developer API reference (https://docs.dev.runwayml.com/api,
``X-Runway-Version: 2024-11-06``), the pricing guide, the input-limits guide and the
task-failure guide, all verified on 2026-11-21. Only models whose request schemas were
verified that day are registered here; Runway adds models often, so extend
``MODELS`` deliberately rather than passing arbitrary model ids through.

Privacy and cost rules this module enforces:

* The API key is sent only to the official API host. Presigned upload URLs and output
  download URLs are contacted with a separate client that carries no credentials.
* Request bodies are redacted before they are persisted (inline data URIs never reach
  queue state or history).
* A task whose server-side cost estimate exceeds the job's credit cap is cancelled
  immediately.
* Images are re-encoded without EXIF metadata before upload when metadata is present.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import mimetypes
import random
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from PIL import Image

from ..models import ProviderConfig, RenderJob
from ..integrity import sha256_file
from ..secret_store import get_secret

RUNWAY_API_BASE = "https://api.dev.runwayml.com"
RUNWAY_API_VERSION = "2024-11-06"
ALLOWED_BASE_URLS = frozenset({RUNWAY_API_BASE})
DOCS_VERIFIED = "2026-11-21"
CREDIT_USD = 0.01

DATA_URI_LIMIT = 5 * 1024 * 1024  # applies to the encoded data URI string
UPLOAD_LIMIT_BYTES = 200 * 1024 * 1024
UPLOAD_MIN_BYTES = 512
DOWNLOAD_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
POLL_MIN_INTERVAL_S = 5.0
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

IMAGE_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
VIDEO_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}
AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}
OUTPUT_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "application/zip": ".zip",
}

_TAG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,15}$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


# --------------------------------------------------------------------------- errors


class RunwayError(RuntimeError):
    """Base error. ``error_category``/``error_message`` feed the queue's error classifier."""

    error_category = "cloud_error"
    error_message = "The Runway job failed. The retained detail explains why."

    def __init__(self, message: str, *, receipt: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt: dict[str, Any] = dict(receipt or {})


class RunwayConfigError(RunwayError):
    error_category = "cloud_not_configured"
    error_message = "Runway is not enabled or has no API key. Open Advanced → Runway cloud provider."


class RunwayValidationError(RunwayError, ValueError):
    error_category = "cloud_invalid_request"
    error_message = "The Runway request does not meet the selected model's documented limits."


class RunwayBudgetError(RunwayError):
    error_category = "cloud_budget"
    error_message = "Runway estimated more credits than the per-job cap allows, so the task was cancelled."


class RunwayAPIError(RunwayError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        receipt: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, receipt=receipt)
        self.status_code = status_code
        if status_code is None:
            self.error_category = "cloud_unreachable"
            self.error_message = "Runway could not be reached. Check the network connection and try again."
        elif status_code == 401:
            self.error_category = "cloud_auth"
            self.error_message = "Runway rejected the API key. Replace it under Advanced → Runway cloud provider."
        elif status_code == 429:
            self.error_category = "cloud_rate_limited"
            self.error_message = "Runway rate-limited the request. Wait a moment, then retry."
        elif status_code == 400:
            self.error_category = "cloud_invalid_request"
            self.error_message = "Runway rejected the request parameters. The detail names the field."
        else:
            self.error_category = "cloud_api_error"
            self.error_message = "Runway returned an error. The retained detail has the status and message."


class RunwayTaskFailed(RunwayError):
    def __init__(
        self,
        task_id: str,
        failure: str | None,
        failure_code: str | None,
        *,
        receipt: dict[str, Any] | None = None,
    ) -> None:
        info = classify_failure(failure_code)
        super().__init__(
            f"Runway task {task_id} failed ({failure_code or 'no failure code'}): {failure or 'no reason given'}",
            receipt=receipt,
        )
        self.task_id = task_id
        self.failure_code = failure_code
        self.retryable = info["retryable"]
        self.error_category = f"cloud_{info['category']}"
        self.error_message = info["advice"]


class RunwayTaskCancelled(RunwayError):
    error_category = "cancelled"
    error_message = "The Runway task was cancelled."


class RunwayTimeout(RunwayError, TimeoutError):
    error_category = "timeout"
    error_message = "The Runway task did not finish within the configured timeout; cancellation was requested."


def classify_failure(code: str | None) -> dict[str, Any]:
    """Map a documented ``failureCode`` to a category, retry guidance and user advice."""
    value = (code or "").strip().upper()
    if value.startswith("SAFETY."):
        return {
            "category": "moderation",
            "retryable": False,
            "advice": "Runway's moderation rejected an input or output. Revise the prompt or references; "
            "retrying unchanged will fail again.",
        }
    if value.startswith("INPUT_PREPROCESSING.SAFETY"):
        return {
            "category": "moderation",
            "retryable": False,
            "advice": "Runway's moderation rejected the prompt text. Rephrase it before retrying.",
        }
    if value == "INPUT_PREPROCESSING.INTERNAL":
        return {
            "category": "transient",
            "retryable": True,
            "advice": "Runway's moderation step could not run. Retry after a short delay.",
        }
    if value.startswith("INTERNAL.BAD_OUTPUT"):
        return {
            "category": "bad_output",
            "retryable": True,
            "advice": "Runway discarded the output for quality reasons, often caused by logos, watermarks or "
            "overlaid text in the inputs, or by asking for text. Remove those and retry.",
        }
    if value.startswith("ASSET."):
        return {
            "category": "invalid_asset",
            "retryable": False,
            "advice": "Runway could not use one of the inputs (dimensions, duration or format). Replace it.",
        }
    if value.startswith("THIRD_PARTY"):
        return {
            "category": "provider_unavailable",
            "retryable": True,
            "advice": "The model's upstream provider is unavailable or shedding load. Wait, then retry.",
        }
    if not value or value.startswith("INTERNAL"):
        return {
            "category": "internal",
            "retryable": True,
            "advice": "Runway hit an internal error. Retry after a short delay.",
        }
    return {
        "category": "task_failed",
        "retryable": False,
        "advice": "The Runway task failed. The retained failure code and reason explain why.",
    }


# --------------------------------------------------------------------------- model registry

GEN4_IMAGE_RATIOS = (
    "1920:1080", "1080:1920", "1024:1024", "1360:768", "1080:1080", "1168:880", "1440:1080",
    "1080:1440", "1808:768", "2112:912", "1280:720", "720:1280", "720:720", "960:720", "720:960",
    "1680:720",
)
GEN4_IMAGE_720P = frozenset({"1280:720", "720:1280", "720:720", "960:720", "720:960", "1680:720"})
GEMINI_IMAGE_RATIOS = (
    "1024:1024", "832:1248", "1248:832", "864:1184", "1184:864", "896:1152", "1152:896", "768:1344",
    "1344:768", "2048:2048", "1696:2528", "2528:1696", "1792:2400", "2400:1792", "1856:2304",
    "2304:1856", "1536:2752", "2752:1536", "512:512", "416:624", "624:416", "448:576", "576:448",
    "384:672", "672:384",
)
GEN4_VIDEO_RATIOS = ("720:1280", "1280:720", "832:1104", "1104:832", "960:960", "1584:672")

AVATAR_PRESETS = (
    "game-character", "music-superstar", "game-character-man", "cat-character", "influencer",
    "tennis-coach", "human-resource", "fashion-designer", "cooking-teacher",
)
VOICE_PRESETS = (
    "victoria", "vincent", "clara", "drew", "skye", "max", "morgan", "felix", "mia", "marcus",
    "summer", "ruby", "aurora", "jasper", "leo", "adrian", "nina", "emma", "blake", "david",
    "maya", "nathan", "sam", "georgia", "petra", "adam", "zach", "violet", "roman", "luna",
)


@dataclass(frozen=True)
class RunwayModel:
    id: str
    task: str
    endpoint: str
    label: str
    summary: str
    pricing: str
    prompt_max: int = 0
    prompt_required: bool = False
    ratios: tuple[str, ...] = ()
    min_refs: int = 0
    max_refs: int = 0
    max_human_refs: int = 0
    duration_range: tuple[int, int] | None = None
    default_duration: int | None = None
    supports_seed: bool = True
    supports_last_frame: bool = False
    output_counts: tuple[int, ...] = (1,)
    resolutions: tuple[str, ...] = ()
    min_image_side: int = 0
    max_image_side: int = 0
    extras: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "endpoint": self.endpoint,
            "label": self.label,
            "summary": self.summary,
            "pricing": self.pricing,
            "prompt_max": self.prompt_max,
            "prompt_required": self.prompt_required,
            "ratios": list(self.ratios),
            "min_refs": self.min_refs,
            "max_refs": self.max_refs,
            "max_human_refs": self.max_human_refs,
            "duration_range": list(self.duration_range) if self.duration_range else None,
            "default_duration": self.default_duration,
            "supports_seed": self.supports_seed,
            "supports_last_frame": self.supports_last_frame,
            "output_counts": list(self.output_counts),
            "resolutions": list(self.resolutions),
            "extras": list(self.extras),
        }


MODELS: dict[str, RunwayModel] = {
    model.id: model
    for model in (
        RunwayModel(
            "gen4_image", "text_to_image", "/v1/text_to_image", "Gen-4 Image",
            "Photoreal stills from text, optionally anchored by up to 3 tagged references (@tag in the prompt).",
            "5 credits (720p) or 8 credits (1080p) per image",
            prompt_max=1000, prompt_required=True, ratios=GEN4_IMAGE_RATIOS, max_refs=3,
        ),
        RunwayModel(
            "gen4_image_turbo", "text_to_image", "/v1/text_to_image", "Gen-4 Image Turbo",
            "Faster, cheaper Gen-4 stills; requires 1-3 reference images.",
            "2 credits per image",
            prompt_max=1000, prompt_required=True, ratios=GEN4_IMAGE_RATIOS, min_refs=1, max_refs=3,
        ),
        RunwayModel(
            "gemini_image3.1_flash", "text_to_image", "/v1/text_to_image", "Gemini Image 3.1 Flash",
            "Up to 14 references, 5 of them marked as the human subject for character consistency; 1 or 4 outputs.",
            "5 / 7 / 11 / 16 credits per image (512px / 1K / 2K / 4K)",
            prompt_max=5500, prompt_required=True, ratios=GEMINI_IMAGE_RATIOS, max_refs=14,
            max_human_refs=5, supports_seed=False, output_counts=(1, 4),
        ),
        RunwayModel(
            "gen4.5", "image_to_video", "/v1/image_to_video", "Gen-4.5",
            "Highest-fidelity Runway image-to-video from an approved first frame, 2-10 s.",
            "12 credits per second",
            prompt_max=1000, prompt_required=True, ratios=GEN4_VIDEO_RATIOS, duration_range=(2, 10),
            default_duration=5,
        ),
        RunwayModel(
            "gen4_turbo", "image_to_video", "/v1/image_to_video", "Gen-4 Turbo",
            "Faster, cheaper image-to-video for motion tests, 2-10 s.",
            "5 credits per second",
            prompt_max=1000, ratios=GEN4_VIDEO_RATIOS, duration_range=(2, 10), default_duration=5,
        ),
        RunwayModel(
            "h3_max", "image_to_video", "/v1/image_to_video", "MiniMax H3 Max (hosted)",
            "Hosted H3: first frame plus optional last frame, 5-15 s at 480p or 768p. Use prompt expansion "
            "'disabled' for repeatable seeds.",
            "5 credits per second (480p) or 8 credits per second (768p)",
            prompt_max=6000, prompt_required=True, duration_range=(5, 15), default_duration=6,
            supports_last_frame=True, resolutions=("768p", "480p"), min_image_side=256,
        ),
        RunwayModel(
            "act_two", "character_performance", "/v1/character_performance", "Act-Two performance transfer",
            "Transfers a recorded 3-30 s performance (face, and optionally body) onto the presenter image.",
            "5 credits per second of performance",
            ratios=GEN4_VIDEO_RATIOS,
        ),
        RunwayModel(
            "gwm1_avatars", "avatar_video", "/v1/avatar_videos", "Runway Avatar video (lip-synced)",
            "Talking-presenter video from a script (TTS) or narration audio, using a Runway custom avatar.",
            "1 credit per 3 s of video, plus 1 credit per 50 script characters; minimum 2",
            supports_seed=False,
        ),
        RunwayModel(
            "magnific_precision_upscaler_v2", "image_upscale", "/v1/image_upscale", "Magnific precision upscale",
            "Upscales an approved still 2-16x. Each input side must be 300-8000 px.",
            "25 credits per image (150 when the output exceeds 4096 px)",
            supports_seed=False, min_image_side=300, max_image_side=8000,
        ),
        RunwayModel(
            "eleven_v4", "text_to_speech", "/v1/text_to_speech", "Eleven v4 narration",
            "Most natural narration; supports [pause] tags, speed control and continuity with neighbouring segments.",
            "5 credits per 1,000 characters (minimum 1)",
            prompt_max=2500, prompt_required=True,
            extras=("speed", "stability", "similarity_boost", "style", "use_speaker_boost", "language_code", "context"),
        ),
        RunwayModel(
            "eleven_v3", "text_to_speech", "/v1/text_to_speech", "Eleven v3 narration",
            "Expressive narration with audio tags such as [whispers]; up to 5,000 characters.",
            "1 credit per 50 characters",
            prompt_max=5000, prompt_required=True,
            extras=("speed", "stability", "similarity_boost", "style", "use_speaker_boost", "language_code", "normalization"),
        ),
        RunwayModel(
            "eleven_multilingual_v2", "text_to_speech", "/v1/text_to_speech", "Eleven Multilingual v2 narration",
            "Stable multilingual narration; up to 1,000 characters per request.",
            "1 credit per 50 characters",
            prompt_max=1000, prompt_required=True, supports_seed=False,
        ),
    )
}

VOICE_DESIGN_MODELS = ("eleven_ttv_v3", "eleven_multilingual_ttv_v2")
_TTS_PRESET_RE = re.compile(r"^[A-Z][A-Za-z]{1,24}$")
TTS_PRESET_SUGGESTIONS = (
    "Maya", "Serene", "Leslie", "Eleanor", "Mabel", "Sandra", "Kylie", "Lara", "Lisa", "Marlene",
    "Miriam", "Paula", "Maggie", "Katie", "Rina", "Ella", "Mariah", "Claudia",
)

TASK_ENDPOINT_TASK = {
    "still": "text_to_image",
    "variations": "text_to_image",
    "image_to_video": "image_to_video",
    "performance": "character_performance",
    "avatar_video": "avatar_video",
    "upscale": "image_upscale",
    "narration": "text_to_speech",
}


def model_spec(model_id: str) -> RunwayModel:
    try:
        return MODELS[str(model_id)]
    except KeyError as exc:
        raise RunwayValidationError(
            f"Runway model '{model_id}' is not registered in Avatar V2. Registered: {', '.join(sorted(MODELS))}."
        ) from exc


def models_payload() -> list[dict[str, Any]]:
    return [model.public() for model in MODELS.values()]


def utf16_len(text: str) -> int:
    """Runway measures prompt limits in UTF-16 code units."""
    return len(str(text).encode("utf-16-le")) // 2


def ratio_dims(ratio: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in str(ratio).split(":", 1))
    except (TypeError, ValueError) as exc:
        raise RunwayValidationError(f"Ratio '{ratio}' must look like WIDTH:HEIGHT.") from exc
    return width, height


# --------------------------------------------------------------------------- cost estimates


def _gemini_tier(ratio: str) -> tuple[int, str]:
    width, height = ratio_dims(ratio)
    megapixels = width * height / 1_000_000
    if megapixels <= 0.4:
        return 5, "512px"
    if megapixels <= 1.6:
        return 7, "1K"
    if megapixels <= 6.0:
        return 11, "2K"
    return 16, "4K"


def estimate_speech_seconds(text: str) -> float:
    """Approximate narration length at a calm teaching pace (about 150 words per minute)."""
    words = len(re.findall(r"\S+", str(text or "")))
    return max(1.0, words / 2.5)


def estimate_credits(model_id: str, params: dict[str, Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Local, documented-price estimate. Runway's own ``estimatedCost`` is authoritative once a task exists."""
    model = model_spec(model_id)
    facts = facts or {}
    upper_bound = False

    if model.id == "gen4_image":
        per = 5 if params.get("ratio") in GEN4_IMAGE_720P else 8
        credits = float(per)
        basis = f"{per} credits per image ({'720p' if per == 5 else '1080p'} tier)"
    elif model.id == "gen4_image_turbo":
        credits, basis = 2.0, "2 credits per image"
    elif model.id == "gemini_image3.1_flash":
        per, tier = _gemini_tier(str(params.get("ratio") or "1024:1024"))
        count = int(params.get("output_count") or 1)
        credits = float(per * count)
        basis = f"{per} credits per {tier} image x {count}"
    elif model.id in {"gen4.5", "gen4_turbo"}:
        rate = 12 if model.id == "gen4.5" else 5
        seconds = int(params.get("duration") or model.default_duration or 5)
        credits = float(rate * seconds)
        basis = f"{rate} credits/s x {seconds} s"
    elif model.id == "h3_max":
        rate = 5 if params.get("resolution") == "480p" else 8
        seconds = int(params.get("duration") or model.default_duration or 6)
        credits = float(rate * seconds)
        basis = f"{rate} credits/s at {params.get('resolution') or '768p'} x {seconds} s"
    elif model.id == "act_two":
        seconds = facts.get("reference_video_s")
        if seconds is None:
            seconds, upper_bound = 30.0, True
        seconds = min(max(float(seconds), 3.0), 30.0)
        credits = 5.0 * math.ceil(seconds)
        basis = f"5 credits/s x {math.ceil(seconds)} s of performance" + (" (upper bound)" if upper_bound else "")
    elif model.id == "gwm1_avatars":
        speech = params.get("speech") or {}
        if speech.get("type") == "audio":
            seconds = facts.get("speech_audio_s")
            if seconds is None:
                seconds, upper_bound = 300.0, True
            script_credits = 0
        else:
            text = str(speech.get("text") or "")
            seconds = estimate_speech_seconds(text)
            script_credits = math.ceil(len(text) / 50)
        credits = float(max(2, math.ceil(float(seconds) / 3) + script_credits))
        basis = (
            f"~{math.ceil(float(seconds))} s of video at 1 credit/3 s"
            + (f" + {script_credits} for the script" if script_credits else "")
            + (" (upper bound)" if upper_bound else "")
        )
    elif model.task == "text_to_speech":
        characters = utf16_len(str(params.get("prompt") or ""))
        if model.id == "eleven_v4":
            credits = float(max(1, math.ceil(characters * 5 / 1000)))
            basis = f"5 credits per 1,000 characters x {characters} characters"
        else:
            credits = float(max(1, math.ceil(characters / 50)))
            basis = f"1 credit per 50 characters x {characters} characters"
    elif model.id == "magnific_precision_upscaler_v2":
        scale = int(params.get("scale_factor") or 2)
        size = facts.get("image_size")
        big = bool(size) and max(size) * scale > 4096
        if not size:
            upper_bound = True
        credits = 150.0 if (big or not size) else 25.0
        basis = "150 credits (output above 4096 px)" if credits == 150 else "25 credits per image"
        if upper_bound:
            basis += " (upper bound; input size unknown)"
    else:  # pragma: no cover - registry and estimator are kept in sync by tests
        raise RunwayValidationError(f"No cost model for {model.id}")

    credits_int = int(math.ceil(credits))
    return {
        "credits": credits_int,
        "usd": round(credits_int * CREDIT_USD, 2),
        "basis": basis,
        "upper_bound": upper_bound,
        "pricing": model.pricing,
        "verified": DOCS_VERIFIED,
    }


# --------------------------------------------------------------------------- spec validation / request building


def _media_present(spec: dict[str, Any], asset_map: dict[str, Any] | None, key: str) -> bool:
    entry = (spec.get("media") or {}).get(key)
    if not entry:
        return False
    if entry.get("asset"):
        return bool((asset_map or {}).get(entry["asset"]))
    return bool(entry.get("path"))


def validate_spec(spec: dict[str, Any], asset_map: dict[str, Any] | None = None) -> None:
    """Check a job spec against the model's documented limits before any upload happens."""
    model = model_spec(spec.get("model", ""))
    params = spec.get("params") or {}
    media = spec.get("media") or {}

    prompt = str(params.get("prompt") or "")
    if model.prompt_max:
        if model.prompt_required and not prompt.strip():
            raise RunwayValidationError(f"{model.label} needs a prompt.")
        if utf16_len(prompt) > model.prompt_max:
            raise RunwayValidationError(
                f"The prompt is {utf16_len(prompt)} UTF-16 units; {model.label} allows {model.prompt_max}."
            )
    elif prompt:
        raise RunwayValidationError(f"{model.label} does not take a text prompt.")

    if model.ratios:
        ratio = params.get("ratio")
        required = model.task in {"text_to_image", "image_to_video"}
        if ratio is None and required:
            raise RunwayValidationError(f"{model.label} needs an output ratio.")
        if ratio is not None and ratio not in model.ratios:
            raise RunwayValidationError(f"Ratio {ratio} is not supported by {model.label}.")
    if model.resolutions and params.get("resolution") not in model.resolutions:
        raise RunwayValidationError(f"{model.label} supports {', '.join(model.resolutions)}.")

    if model.duration_range:
        duration = params.get("duration")
        low, high = model.duration_range
        if not isinstance(duration, int) or not low <= duration <= high:
            raise RunwayValidationError(f"{model.label} duration must be a whole number from {low} to {high} s.")

    seed = params.get("seed")
    if seed is not None:
        if not model.supports_seed:
            raise RunwayValidationError(f"{model.label} does not accept a seed.")
        if not isinstance(seed, int) or not 0 <= seed <= 4_294_967_295:
            raise RunwayValidationError("Seed must be an integer from 0 to 4294967295.")

    count = int(params.get("output_count") or 1)
    if count not in model.output_counts:
        raise RunwayValidationError(f"{model.label} output count must be one of {list(model.output_counts)}.")

    if model.task == "text_to_image":
        references = list(media.get("references") or [])
        if not model.min_refs <= len(references) <= model.max_refs:
            raise RunwayValidationError(
                f"{model.label} takes {model.min_refs}-{model.max_refs} reference images; got {len(references)}."
            )
        tags = [str(ref.get("tag")) for ref in references if ref.get("tag")]
        for tag in tags:
            if not _TAG_RE.match(tag):
                raise RunwayValidationError(
                    f"Reference tag '{tag}' must be 3-16 letters, digits or underscores and start with a letter."
                )
        if len(tags) != len(set(tags)):
            raise RunwayValidationError("Reference tags must be unique.")
        humans = sum(1 for ref in references if ref.get("subject") == "human")
        if humans and not model.max_human_refs:
            raise RunwayValidationError(f"{model.label} does not support human-subject reference marking.")
        if humans > model.max_human_refs:
            raise RunwayValidationError(f"{model.label} allows at most {model.max_human_refs} human-subject references.")
        for ref in references:
            if ref.get("subject") not in (None, "human", "object"):
                raise RunwayValidationError("Reference subject must be 'human' or 'object'.")
            if asset_map is not None and ref.get("asset") and not asset_map.get(ref["asset"]):
                raise RunwayValidationError(f"Reference {ref.get('role') or ref['asset']} has no file.")
        for tag in tags:
            if f"@{tag}" not in prompt:
                raise RunwayValidationError(f"The prompt must mention @{tag} so the tagged reference is used.")
    elif model.task == "image_to_video":
        if not _media_present(spec, asset_map, "first_frame"):
            raise RunwayValidationError(f"{model.label} needs a first-frame image (the approved presenter still).")
        if _media_present(spec, asset_map, "last_frame") and not model.supports_last_frame:
            raise RunwayValidationError(f"{model.label} does not accept a last frame.")
        mode = params.get("prompt_expansion")
        if model.id == "h3_max" and mode not in {"disabled", "balanced", "quality"}:
            raise RunwayValidationError("Prompt expansion must be disabled, balanced or quality.")
    elif model.task == "character_performance":
        if not _media_present(spec, asset_map, "character"):
            raise RunwayValidationError("Act-Two needs the presenter image or video as the character.")
        if (media.get("character") or {}).get("type", "image") not in {"image", "video"}:
            raise RunwayValidationError("Act-Two character type must be image or video.")
        if not _media_present(spec, asset_map, "reference_video"):
            raise RunwayValidationError("Act-Two needs a 3-30 s performance reference video.")
        intensity = params.get("expression_intensity", 3)
        if not isinstance(intensity, int) or not 1 <= intensity <= 5:
            raise RunwayValidationError("Expression intensity must be an integer from 1 to 5.")
    elif model.task == "avatar_video":
        avatar = params.get("avatar") or {}
        if avatar.get("type") == "custom":
            if not _UUID_RE.match(str(avatar.get("avatarId") or "")):
                raise RunwayValidationError("Choose one of your Runway custom avatars (a UUID).")
        elif avatar.get("type") == "runway-preset":
            if avatar.get("presetId") not in AVATAR_PRESETS:
                raise RunwayValidationError("Unknown Runway preset avatar.")
        else:
            raise RunwayValidationError("Avatar videos need a custom avatar id or a Runway preset.")
        speech = params.get("speech") or {}
        if speech.get("type") == "text":
            text = str(speech.get("text") or "").strip()
            if not text:
                raise RunwayValidationError("Write the script the presenter should speak.")
            if utf16_len(text) > 5000:
                raise RunwayValidationError("Avatar video scripts are limited to 5000 characters; split the lesson.")
            voice = speech.get("voice")
            if voice:
                if voice.get("type") == "preset" and voice.get("presetId") not in VOICE_PRESETS:
                    raise RunwayValidationError("Unknown Runway voice preset.")
                if voice.get("type") == "custom" and not _UUID_RE.match(str(voice.get("id") or "")):
                    raise RunwayValidationError("Custom voice ids are UUIDs.")
                if voice.get("type") not in {"preset", "custom"}:
                    raise RunwayValidationError("Voice override type must be preset or custom.")
        elif speech.get("type") == "audio":
            if not _media_present(spec, asset_map, "speech_audio"):
                raise RunwayValidationError("Select the narration audio file for the presenter.")
        else:
            raise RunwayValidationError("Avatar video speech must be a script or an audio file.")
    elif model.task == "image_upscale":
        if not _media_present(spec, asset_map, "image"):
            raise RunwayValidationError("Select the still to upscale.")
        if int(params.get("scale_factor") or 2) not in {2, 4, 8, 16}:
            raise RunwayValidationError("Upscale factor must be 2, 4, 8 or 16.")
        if params.get("flavor", "photo") not in {"sublime", "photo", "photo_denoiser"}:
            raise RunwayValidationError("Upscale flavor must be sublime, photo or photo_denoiser.")
    elif model.task == "text_to_speech":
        voice = params.get("voice") or {}
        if voice.get("type") == "runway-preset":
            if not _TTS_PRESET_RE.match(str(voice.get("presetId") or "")):
                raise RunwayValidationError("Choose a Runway narration preset voice (for example Maya or Leslie).")
        elif voice.get("type") == "custom":
            if not _UUID_RE.match(str(voice.get("id") or "")):
                raise RunwayValidationError("Custom narration voices are referenced by their UUID.")
        else:
            raise RunwayValidationError("Narration needs a preset voice or a designed custom voice.")
        for key, low, high in (("speed", 0.7, 1.2), ("stability", 0.0, 1.0), ("similarity_boost", 0.0, 1.0), ("style", 0.0, 1.0)):
            value = params.get(key)
            if value is None:
                continue
            if key not in model.extras:
                raise RunwayValidationError(f"{model.label} does not support {key.replace('_', ' ')}.")
            if not isinstance(value, (int, float)) or not low <= float(value) <= high:
                raise RunwayValidationError(f"{key.replace('_', ' ').capitalize()} must be between {low} and {high}.")
        if (params.get("previous_text") or params.get("next_text")) and "context" not in model.extras:
            raise RunwayValidationError(f"{model.label} does not support neighbouring-segment context.")
        for key in ("previous_text", "next_text"):
            if utf16_len(str(params.get(key) or "")) > model.prompt_max:
                raise RunwayValidationError(f"{key.replace('_', ' ').capitalize()} is longer than {model.prompt_max} characters.")
        normalization = params.get("apply_text_normalization")
        if normalization is not None and ("normalization" not in model.extras or normalization not in {"auto", "on", "off"}):
            raise RunwayValidationError(f"{model.label} text normalization must be auto, on or off where supported.")
        language = params.get("language_code")
        if language is not None and ("language_code" not in model.extras or not re.match(r"^[A-Za-z]{2,3}(-[A-Za-z]{2,4})?$", str(language))):
            raise RunwayValidationError("Language code must look like 'en' or 'pt-BR' and be supported by the model.")

    output_audio = params.get("output_audio")
    if output_audio is not None:
        if output_audio not in {"keep", "mute", "voiceover"}:
            raise RunwayValidationError("Output audio must be keep, mute or voiceover.")
        if model.task not in {"image_to_video", "character_performance", "avatar_video"}:
            raise RunwayValidationError(f"{model.label} does not produce video audio to replace.")
        if output_audio == "voiceover":
            local = (spec.get("local") or {}).get("voiceover") or {}
            present = bool((asset_map or {}).get(local.get("asset", ""))) if asset_map is not None else bool(local)
            if not present:
                raise RunwayValidationError("Voiceover output needs an approved narration file.")


def build_request(model: RunwayModel, params: dict[str, Any], media: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the documented JSON body from validated params and already-resolved media URIs."""
    body: dict[str, Any] = {"model": model.id}
    seed = params.get("seed")

    if model.task == "text_to_image":
        body["promptText"] = params["prompt"]
        body["ratio"] = params["ratio"]
        references = []
        for ref in media.get("references") or []:
            item: dict[str, Any] = {"uri": ref["uri"]}
            if ref.get("tag"):
                item["tag"] = ref["tag"]
            if model.max_human_refs and ref.get("subject"):
                item["subject"] = ref["subject"]
            references.append(item)
        if references:
            body["referenceImages"] = references
        if len(model.output_counts) > 1:
            body["outputCount"] = int(params.get("output_count") or 1)
    elif model.task == "image_to_video":
        first, last = media["first_frame"], media.get("last_frame")
        if model.supports_last_frame and last:
            body["promptImage"] = [{"uri": first, "position": "first"}, {"uri": last, "position": "last"}]
        else:
            body["promptImage"] = [{"uri": first, "position": "first"}]
        if params.get("prompt"):
            body["promptText"] = params["prompt"]
        if model.ratios:
            body["ratio"] = params["ratio"]
        if model.resolutions:
            body["resolution"] = params["resolution"]
        if model.id == "h3_max":
            body["promptExpansionMode"] = params.get("prompt_expansion") or "disabled"
        body["duration"] = int(params["duration"])
    elif model.task == "character_performance":
        body["character"] = {"type": media.get("character_type") or "image", "uri": media["character"]}
        body["reference"] = {"type": "video", "uri": media["reference_video"]}
        body["bodyControl"] = bool(params.get("body_control", True))
        body["expressionIntensity"] = int(params.get("expression_intensity", 3))
        if params.get("ratio"):
            body["ratio"] = params["ratio"]
    elif model.task == "avatar_video":
        body["avatar"] = dict(params["avatar"])
        speech = params["speech"]
        if speech["type"] == "audio":
            body["speech"] = {"type": "audio", "audio": media["speech_audio"]}
        else:
            spoken: dict[str, Any] = {"type": "text", "text": speech["text"]}
            if speech.get("voice"):
                spoken["voice"] = dict(speech["voice"])
            body["speech"] = spoken
    elif model.task == "image_upscale":
        body["imageUri"] = media["image"]
        body["scaleFactor"] = int(params.get("scale_factor") or 2)
        body["flavor"] = params.get("flavor") or "photo"
        for source, target in (("sharpen", "sharpen"), ("smart_grain", "smartGrain"), ("ultra_detail", "ultraDetail")):
            if params.get(source) is not None:
                body[target] = max(0, min(int(params[source]), 100))
    elif model.task == "text_to_speech":
        body["promptText"] = params["prompt"]
        body["voice"] = dict(params["voice"])
        for source, target in (
            ("speed", "speed"),
            ("stability", "stability"),
            ("similarity_boost", "similarityBoost"),
            ("style", "style"),
            ("use_speaker_boost", "useSpeakerBoost"),
            ("language_code", "languageCode"),
            ("previous_text", "previousText"),
            ("next_text", "nextText"),
            ("apply_text_normalization", "applyTextNormalization"),
        ):
            if params.get(source) not in (None, ""):
                body[target] = params[source]

    if seed is not None and model.supports_seed:
        body["seed"] = int(seed)
    return model.endpoint, body


def redact_request(value: Any) -> Any:
    """Copy a request body with inline data URIs replaced by a size marker."""
    if isinstance(value, dict):
        return {key: redact_request(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_request(item) for item in value]
    if isinstance(value, str) and value.startswith("data:"):
        return f"<inline data URI, {len(value)} chars>"
    return value


# --------------------------------------------------------------------------- media preparation


@dataclass
class PreparedMedia:
    uri: str
    method: str
    role: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    metadata_stripped: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "file": self.filename,
            "bytes": self.size_bytes,
            "content_type": self.content_type,
            "transfer": self.method,
            "sha256": self.sha256[:16],
            "metadata_stripped": self.metadata_stripped,
        }


def _image_bytes(path: Path, *, min_side: int = 0, max_side: int = 0) -> tuple[bytes, str, str, bool]:
    """Return (bytes, content_type, filename, metadata_stripped) for a Runway-compatible image."""
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if min_side and min(width, height) < min_side:
            raise RunwayValidationError(f"{path.name} is {width}x{height}; this model needs at least {min_side} px per side.")
        if max_side and max(width, height) > max_side:
            raise RunwayValidationError(f"{path.name} is {width}x{height}; this model allows at most {max_side} px per side.")
        fmt = (image.format or "").upper()
        try:
            has_exif = bool(image.info.get("exif")) or len(image.getexif()) > 0
        except Exception:
            has_exif = bool(image.info.get("exif"))
        if fmt in IMAGE_TYPES and not has_exif:
            return path.read_bytes(), IMAGE_TYPES[fmt], path.name, False

        icc = image.info.get("icc_profile")
        buffer = io.BytesIO()
        extra: dict[str, Any] = {"icc_profile": icc} if icc else {}
        if fmt == "JPEG":
            converted = image if image.mode in {"RGB", "L"} else image.convert("RGB")
            converted.save(buffer, format="JPEG", quality=95, subsampling=0, **extra)
            return buffer.getvalue(), "image/jpeg", f"{path.stem}.jpg", has_exif
        if fmt == "WEBP":
            image.save(buffer, format="WEBP", quality=95, **extra)
            return buffer.getvalue(), "image/webp", f"{path.stem}.webp", has_exif
        if image.mode not in {"RGB", "RGBA", "L", "LA"}:
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        image.save(buffer, format="PNG", **extra)
        return buffer.getvalue(), "image/png", f"{path.stem}.png", has_exif


def prepare_media(
    value: str,
    *,
    kind: str,
    role: str,
    uploader: Callable[[str, str, bytes], str] | None,
    min_side: int = 0,
    max_side: int = 0,
) -> PreparedMedia:
    """Turn a local path (or an HTTPS / Runway URI) into a URI Runway accepts.

    Small files travel inline as data URIs (the documented 5 MB encoded limit); larger files
    go through Runway's ephemeral upload flow via ``uploader(filename, content_type, data)``.
    """
    raw = str(value or "").strip()
    if not raw:
        raise RunwayValidationError(f"The {role} file is missing.")
    if raw.startswith(("runway://", "runwayml://")):
        return PreparedMedia(raw, "runway_uri", role, raw, "", 0, "")
    if raw.startswith("https://"):
        return PreparedMedia(raw, "https_url", role, raw.rsplit("/", 1)[-1][:120], "", 0, "")
    if raw.startswith(("http://", "data:")):
        raise RunwayValidationError(f"The {role} must be a local file, an HTTPS URL or a Runway upload URI.")

    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RunwayValidationError(f"The {role} file does not exist: {path.name}")

    stripped = False
    if kind == "image":
        try:
            data, content_type, filename, stripped = _image_bytes(path, min_side=min_side, max_side=max_side)
        except OSError as exc:  # includes PIL.UnidentifiedImageError
            raise RunwayValidationError(f"{path.name} could not be read as an image ({type(exc).__name__}).") from exc
    elif kind == "video":
        content_type = VIDEO_TYPES.get(path.suffix.lower(), "")
        if not content_type:
            raise RunwayValidationError(f"{path.name}: use MP4, MOV, MKV or WebM video for Runway.")
        data, filename = path.read_bytes(), path.name
    elif kind == "audio":
        content_type = AUDIO_TYPES.get(path.suffix.lower(), "")
        if not content_type:
            raise RunwayValidationError(f"{path.name}: use MP3, WAV, FLAC, M4A or AAC audio for Runway.")
        data, filename = path.read_bytes(), path.name
    else:
        raise RunwayValidationError(f"Unsupported media kind {kind}.")

    if len(data) > UPLOAD_LIMIT_BYTES:
        raise RunwayValidationError(f"{path.name} is larger than Runway's 200 MB upload limit.")
    digest = hashlib.sha256(data).hexdigest()
    prefix = f"data:{content_type};base64,"
    encoded_len = len(prefix) + 4 * math.ceil(len(data) / 3)
    if encoded_len <= DATA_URI_LIMIT:
        uri = prefix + base64.b64encode(data).decode("ascii")
        method = "inline_data_uri"
    else:
        if uploader is None:
            raise RunwayValidationError(f"{path.name} needs an ephemeral upload but no uploader is available.")
        uri = uploader(filename, content_type, data)
        method = "ephemeral_upload"
    return PreparedMedia(uri, method, role, filename, content_type, len(data), digest, stripped)


def describe_media(value: str, *, kind: str, role: str) -> dict[str, Any]:
    """Describe what an upload would send, without reading it into memory or contacting Runway."""
    raw = str(value or "")
    if raw.startswith(("runway://", "runwayml://", "https://")):
        return {"role": role, "file": raw.rsplit("/", 1)[-1][:120], "bytes": 0, "transfer": "reference by URI"}
    path = Path(raw).expanduser()
    size = path.stat().st_size if path.is_file() else 0
    transfer = "inline data URI" if size and 4 * math.ceil(size / 3) + 40 <= DATA_URI_LIMIT else "Runway ephemeral upload"
    return {"role": role, "file": path.name, "bytes": size, "kind": kind, "transfer": transfer, "exists": path.is_file()}


def media_duration_s(path: str | Path) -> float | None:
    """Return a media file's duration via ffprobe, or None when unavailable."""
    target = Path(path).expanduser()
    if not target.is_file() or not shutil.which("ffprobe"):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(target)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, OSError):
        return None


# --------------------------------------------------------------------------- HTTP client


class RunwayClient:
    """Minimal synchronous client for the documented Runway REST endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = RUNWAY_API_BASE,
        api_version: str = RUNWAY_API_VERSION,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        if not api_key:
            raise RunwayConfigError("No Runway API key is configured.")
        base = str(base_url).rstrip("/")
        if base not in ALLOWED_BASE_URLS:
            raise RunwayConfigError("Refusing to send the Runway API key to a host other than the official API.")
        self._api = httpx.Client(
            base_url=base,
            timeout=timeout_s,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": api_version,
                "Accept": "application/json",
            },
        )
        # Presigned upload and download URLs must never receive the API key.
        self._plain = httpx.Client(
            timeout=httpx.Timeout(timeout_s, read=300.0),
            transport=transport,
            follow_redirects=True,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter or (lambda: random.uniform(0.0, 1.5))

    def close(self) -> None:
        self._api.close()
        self._plain.close()

    def __enter__(self) -> "RunwayClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- plumbing

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:300]
        if not isinstance(data, dict):
            return str(data)[:300]
        detail = str(data.get("error") or data.get("message") or "")
        issues = data.get("issues")
        if isinstance(issues, list) and issues:
            parts = []
            for issue in issues[:5]:
                if isinstance(issue, dict):
                    where = ".".join(str(p) for p in issue.get("path") or [])
                    parts.append(f"{where}: {issue.get('message', '')}".strip(": "))
                else:
                    parts.append(str(issue))
            detail = f"{detail} ({'; '.join(parts)})" if detail else "; ".join(parts)
        return detail[:500]

    def _wait(self, seconds: float, cancel_event: threading.Event | None = None) -> bool:
        """Wait, returning True if cancellation was requested meanwhile."""
        seconds = max(float(seconds), 0.0)
        if self._sleep is not None:
            self._sleep(seconds)
            return bool(cancel_event and cancel_event.is_set())
        if cancel_event is not None:
            return cancel_event.wait(seconds)
        time.sleep(seconds)
        return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        context: str,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        retry_rate_limit: int = 0,
    ) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = self._api.request(method, path, json=json_body, params=params)
            except httpx.HTTPError as exc:
                raise RunwayAPIError(f"{context}: network error ({type(exc).__name__})") from exc
            if response.status_code == 429 and attempt < retry_rate_limit:
                attempt += 1
                try:
                    delay = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    delay = 2.0 * (2 ** attempt)
                self._wait(min(delay, 60.0) + self._jitter())
                continue
            if response.status_code >= 400:
                raise RunwayAPIError(
                    f"{context} failed with HTTP {response.status_code}: {self._detail(response)}".rstrip(": "),
                    status_code=response.status_code,
                )
            return response

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        if not response.content:
            return {}
        try:
            data = response.json()
        except ValueError as exc:
            raise RunwayAPIError("Runway returned a non-JSON response.", status_code=response.status_code) from exc
        return data if isinstance(data, dict) else {"data": data}

    # -- account

    def organization(self) -> dict[str, Any]:
        return self._json(self._request("GET", "/v1/organization", context="Organization lookup"))

    # -- tasks

    def create_task(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        # 429 means no task was created, so retrying is safe. Other errors are not retried
        # to avoid accidentally creating (and paying for) a duplicate task.
        data = self._json(
            self._request("POST", endpoint, context="Task creation", json_body=body, retry_rate_limit=3)
        )
        if not data.get("id"):
            raise RunwayAPIError("Runway did not return a task id.")
        return data

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._json(self._request("GET", f"/v1/tasks/{task_id}", context="Task lookup"))

    def cancel_task(self, task_id: str) -> None:
        try:
            self._request("DELETE", f"/v1/tasks/{task_id}", context="Task cancellation")
        except RunwayAPIError as exc:
            if exc.status_code != 404:
                raise

    def wait_for_task(
        self,
        task_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float = 6.0,
        cancel_event: threading.Event | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll until a terminal status. Uses >= 5 s intervals with jitter and exponential backoff."""
        interval = max(POLL_MIN_INTERVAL_S, float(poll_interval_s))
        deadline = self._monotonic() + float(timeout_s)
        errors = 0
        while True:
            task: dict[str, Any] | None = None
            try:
                task = self.get_task(task_id)
                errors = 0
            except RunwayAPIError as exc:
                if exc.status_code in {401, 403, 404}:
                    raise
                errors += 1
                if errors > 6:
                    raise
            if task is not None:
                if on_update is not None:
                    on_update(task)
                if task.get("status") in TERMINAL_STATUSES:
                    return task

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                try:
                    self.cancel_task(task_id)
                except RunwayError:
                    pass
                raise RunwayTimeout(
                    f"Runway task {task_id} did not finish within {int(timeout_s)} s; cancellation was requested."
                )
            delay = interval if not errors else min(60.0, interval * (2 ** errors))
            delay = min(delay + self._jitter(), max(remaining, 0.5))
            if self._wait(delay, cancel_event):
                try:
                    self.cancel_task(task_id)
                except RunwayError:
                    pass
                raise RunwayTaskCancelled(f"Runway task {task_id} was cancelled by the user.")

    # -- uploads and downloads

    def upload(self, filename: str, content_type: str, data: bytes) -> str:
        if len(data) < UPLOAD_MIN_BYTES:
            raise RunwayValidationError(f"{filename} is smaller than Runway's 512-byte upload minimum.")
        started = self._json(
            self._request(
                "POST",
                "/v1/uploads",
                context="Upload start",
                json_body={"filename": filename, "type": "ephemeral"},
                retry_rate_limit=3,
            )
        )
        upload_url = str(started.get("uploadUrl") or "")
        runway_uri = started.get("runwayUri") or started.get("uri")
        if not upload_url.startswith("https://") or not runway_uri:
            raise RunwayAPIError("Runway returned an incomplete upload target.")
        fields = {str(k): str(v) for k, v in (started.get("fields") or {}).items()}
        try:
            response = self._plain.post(upload_url, data=fields, files={"file": (filename, data, content_type)})
        except httpx.HTTPError as exc:
            raise RunwayAPIError(f"Upload of {filename} failed: network error ({type(exc).__name__})") from exc
        if response.status_code >= 300:
            raise RunwayAPIError(f"Upload of {filename} failed with status {response.status_code}.",
                                 status_code=response.status_code)
        return str(runway_uri)

    def download(self, url: str, destination_dir: Path, stem: str) -> Path:
        if not str(url).startswith("https://"):
            raise RunwayAPIError("Runway returned a non-HTTPS output URL; refusing to download it.")
        destination_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(url).path).suffix.lower()
        try:
            with self._plain.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise RunwayAPIError(f"Output download failed with HTTP {response.status_code}.",
                                         status_code=response.status_code)
                if not suffix or len(suffix) > 6:
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    suffix = OUTPUT_EXTENSIONS.get(content_type) or mimetypes.guess_extension(content_type) or ".bin"
                target = destination_dir / f"{stem}{suffix}"
                written = 0
                partial = target.with_name(target.name + ".part")
                try:
                    with partial.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            written += len(chunk)
                            if written > DOWNLOAD_LIMIT_BYTES:
                                raise RunwayAPIError("Output download exceeded the 4 GB safety limit.")
                            handle.write(chunk)
                    partial.replace(target)
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
        except httpx.HTTPError as exc:
            raise RunwayAPIError(f"Output download failed: network error ({type(exc).__name__})") from exc
        return target

    # -- avatars

    def list_avatars(self, *, limit: int = 100, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 100))}
        if cursor:
            params["cursor"] = cursor
        return self._json(self._request("GET", "/v1/avatars", context="Avatar list", params=params))

    def get_avatar(self, avatar_id: str) -> dict[str, Any]:
        return self._json(self._request("GET", f"/v1/avatars/{avatar_id}", context="Avatar lookup"))

    def create_avatar(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._json(
            self._request("POST", "/v1/avatars", context="Avatar creation", json_body=body, retry_rate_limit=3)
        )

    def delete_avatar(self, avatar_id: str) -> None:
        self._request("DELETE", f"/v1/avatars/{avatar_id}", context="Avatar deletion")

    # -- designed voices (text description only; this client deliberately offers no voice cloning)

    def preview_voice(self, prompt: str, model: str = "eleven_ttv_v3") -> dict[str, Any]:
        validate_voice_design("Preview", prompt, model)
        return self._json(
            self._request(
                "POST", "/v1/voices/preview", context="Voice preview",
                json_body={"prompt": prompt, "model": model}, retry_rate_limit=3,
            )
        )

    def create_designed_voice(
        self, name: str, prompt: str, *, description: str | None = None, model: str = "eleven_ttv_v3"
    ) -> dict[str, Any]:
        validate_voice_design(name, prompt, model)
        body: dict[str, Any] = {"name": name, "from": {"type": "text", "prompt": prompt, "model": model}}
        if description:
            body["description"] = description[:500]
        return self._json(
            self._request("POST", "/v1/voices", context="Voice creation", json_body=body, retry_rate_limit=3)
        )

    def get_voice(self, voice_id: str) -> dict[str, Any]:
        return self._json(self._request("GET", f"/v1/voices/{voice_id}", context="Voice lookup"))

    def list_voices(self, *, limit: int = 100) -> dict[str, Any]:
        return self._json(
            self._request("GET", "/v1/voices", context="Voice list", params={"limit": max(1, min(int(limit), 100))})
        )


def validate_voice_design(name: str, prompt: str, model: str) -> None:
    """Designed voices come from a text description; no real person's voice is involved."""
    if not 1 <= len(str(name or "").strip()) <= 100:
        raise RunwayValidationError("Voice names must be 1-100 characters.")
    if not 20 <= len(str(prompt or "").strip()) <= 1000:
        raise RunwayValidationError("Describe the voice in 20-1000 characters.")
    if model not in VOICE_DESIGN_MODELS:
        raise RunwayValidationError(f"Voice design model must be one of {', '.join(VOICE_DESIGN_MODELS)}.")


def strip_audio_track(source: Path, staging_dir: Path) -> Path:
    """Copy a video without any audio stream (used so a driving performer's voice is never uploaded)."""
    if not shutil.which("ffmpeg"):
        raise RunwayValidationError(
            "ffmpeg is required to remove the performer's audio before upload; install ffmpeg and retry."
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() if source.suffix.lower() in VIDEO_TYPES else ".mp4"
    target = staging_dir / f"{source.stem}_noaudio{suffix}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", str(target)],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise RunwayValidationError(f"Could not remove the audio track from {source.name}: {type(exc).__name__}") from exc
    return target


def replace_audio_track(video: Path, mode: str, voiceover: Path | None = None) -> Path:
    """Write a muted copy, or a copy whose audio is the approved voiceover. The original is kept."""
    if not shutil.which("ffmpeg"):
        raise RunwayValidationError("ffmpeg is required to mute or replace output audio.")
    target = video.with_name(f"{video.stem}_{'voiceover' if mode == 'voiceover' else 'muted'}{video.suffix}")
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if mode == "voiceover":
        if voiceover is None or not voiceover.is_file():
            raise RunwayValidationError("The approved voiceover file is missing.")
        command += ["-i", str(voiceover), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", str(target)]
    else:
        command += ["-map", "0:v:0", "-c:v", "copy", "-an", str(target)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RunwayValidationError(f"Could not post-process the audio of {video.name}: {type(exc).__name__}") from exc
    return target


def client_from_store(**kwargs: Any) -> RunwayClient:
    key, _source = get_secret("runway")
    if not key:
        raise RunwayConfigError(
            "No Runway API key is configured. Add one under Advanced → Runway cloud provider "
            "or set RUNWAYML_API_SECRET."
        )
    return RunwayClient(key, **kwargs)


# --------------------------------------------------------------------------- queue-facing provider

_MEDIA_KINDS = (
    ("first_frame", "image"),
    ("last_frame", "image"),
    ("image", "image"),
    ("reference_video", "video"),
    ("speech_audio", "audio"),
)


def iter_media_entries(spec: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Every media entry of a spec with its kind, in upload order (references first)."""
    media = spec.get("media") or {}
    entries: list[tuple[dict[str, Any], str]] = [(entry, "image") for entry in media.get("references") or []]
    for key, kind in _MEDIA_KINDS:
        if media.get(key):
            entries.append((media[key], kind))
    if media.get("character"):
        entries.append((media["character"], str(media["character"].get("type") or "image")))
    return entries


def describe_uploads(spec: dict[str, Any], asset_map: dict[str, Any]) -> list[dict[str, Any]]:
    """What a job would send to Runway, computed locally with no network access."""
    seen: set[tuple[str, str]] = set()
    described: list[dict[str, Any]] = []
    for entry, kind in iter_media_entries(spec):
        value = asset_map.get(entry["asset"]) if entry.get("asset") else entry.get("path")
        if not value:
            continue
        key = (str(value), kind)
        if key in seen:
            continue
        seen.add(key)
        described.append(describe_media(str(value), kind=kind, role=str(entry.get("role") or kind)))
    return described


def load_spec(job: RenderJob) -> dict[str, Any]:
    raw = job.asset_map.get("RUNWAY_SPEC")
    if not raw:
        raise RunwayValidationError("This job has no Runway request spec.")
    try:
        spec = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise RunwayValidationError("The stored Runway request spec is not valid JSON.") from exc
    if spec.get("version") != 1:
        raise RunwayValidationError("Unsupported Runway spec version.")
    return spec


class RunwayProvider:
    """Executes one queued Runway job: upload media, create task, poll, download outputs."""

    uses_local_gpu = False

    def __init__(
        self,
        config: ProviderConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._transport = transport
        self._sleep = sleep
        self._monotonic = monotonic
        self._cancel = threading.Event()
        self._task_id: str | None = None
        # Set by the render queue: called with (task_id, receipt) as soon as Runway accepts a task,
        # so the id is persisted before polling starts and can be reconciled after a restart.
        self.on_task_created: Callable[[str, dict[str, Any]], None] | None = None

    def _output_root(self) -> Path:
        return Path(self.config.output_dir or (Path.home() / ".avatar_v2" / "runs" / "runway")).expanduser()

    def _staging_dir(self) -> Path:
        return self._output_root() / "_staging"

    def _client(self) -> RunwayClient:
        return client_from_store(
            base_url=self.config.base_url or RUNWAY_API_BASE,
            transport=self._transport,
            sleep=self._sleep,
            monotonic=self._monotonic,
        )

    def interrupt(self) -> None:
        """Called from the queue thread on Cancel: stop polling and cancel the remote task."""
        self._cancel.set()
        task_id = self._task_id
        if not task_id:
            return
        try:
            with self._client() as client:
                client.cancel_task(task_id)
        except RunwayError:
            pass

    def _resolve_media(
        self,
        client: RunwayClient,
        spec: dict[str, Any],
        job: RenderJob,
        receipt: dict[str, Any],
        model: RunwayModel,
    ) -> dict[str, Any]:
        media_spec = spec.get("media") or {}
        cache: dict[tuple[str, str], PreparedMedia] = {}

        def prepare(entry: dict[str, Any], kind: str) -> str:
            value = job.asset_map.get(entry["asset"]) if entry.get("asset") else entry.get("path")
            role = str(entry.get("role") or entry.get("asset") or kind)
            if kind == "video" and model.task == "character_performance" and value:
                source = Path(str(value)).expanduser()
                if source.is_file():
                    # Act-Two only needs the motion. Never upload the driving performer's voice.
                    value = str(strip_audio_track(source.resolve(), self._staging_dir()))
                    receipt["source_audio"] = "removed before upload"
            key = (str(value), kind)
            if key not in cache:
                cache[key] = prepare_media(
                    str(value or ""),
                    kind=kind,
                    role=role,
                    uploader=client.upload,
                    min_side=model.min_image_side if kind == "image" else 0,
                    max_side=model.max_image_side if kind == "image" else 0,
                )
                receipt["uploads"].append(cache[key].public())
            return cache[key].uri

        resolved: dict[str, Any] = {}
        references = [
            {"uri": prepare(entry, "image"), "tag": entry.get("tag"), "subject": entry.get("subject")}
            for entry in media_spec.get("references") or []
        ]
        if references:
            resolved["references"] = references
        for key, kind in _MEDIA_KINDS:
            entry = media_spec.get(key)
            if not entry:
                continue
            if entry.get("optional") and not (job.asset_map.get(entry.get("asset", "")) or entry.get("path")):
                continue
            resolved[key] = prepare(entry, kind)
        character = media_spec.get("character")
        if character:
            character_type = character.get("type") or "image"
            resolved["character"] = prepare(character, character_type)
            resolved["character_type"] = character_type
        return resolved

    def _download_outputs(self, client: RunwayClient, task_id: str, urls: list[str]) -> list[str]:
        target_dir = self._output_root() / task_id
        outputs = []
        for index, url in enumerate(urls, start=1):
            outputs.append(str(client.download(str(url), target_dir, f"runway_{task_id[:8]}_{index:02d}").resolve()))
        return outputs

    @staticmethod
    def _postprocess_audio(outputs: list[str], spec: dict[str, Any], job: RenderJob, receipt: dict[str, Any]) -> list[str]:
        mode = (spec.get("params") or {}).get("output_audio") or "keep"
        if mode == "keep":
            return outputs
        voiceover = None
        if mode == "voiceover":
            asset = ((spec.get("local") or {}).get("voiceover") or {}).get("asset", "")
            voiceover = Path(str(job.asset_map.get(asset) or "")).expanduser()
        processed: list[str] = []
        originals: list[str] = []
        for output in outputs:
            path = Path(output)
            if path.suffix.lower() not in VIDEO_TYPES:
                processed.append(output)
                continue
            target = replace_audio_track(path, mode, voiceover)
            processed.append(str(target.resolve()))
            originals.append(output)
            entry: dict[str, Any] = {"source": path.name, "output": target.name, "mode": mode}
            if voiceover is not None:
                video_s, voice_s = media_duration_s(path), media_duration_s(voiceover)
                entry.update({"video_s": video_s, "voiceover_s": voice_s})
                if video_s and voice_s and voice_s > video_s + 0.25:
                    entry["warning"] = "The voiceover is longer than the video and was cut at the video's end."
            receipt.setdefault("audio_postprocess", []).append(entry)
        # Processed files come first; the untouched downloads stay available after them.
        return processed + originals

    def render(self, workflow: str, job: RenderJob) -> dict[str, Any]:
        spec = load_spec(job)
        model = model_spec(spec.get("model", ""))
        validate_spec(spec, job.asset_map)
        resume_task_id = str(job.asset_map.get("RUNWAY_RESUME_TASK_ID") or "").strip()
        receipt: dict[str, Any] = {
            "provider": "runway",
            "model": model.id,
            "task": model.task,
            "endpoint": model.endpoint,
            "estimate": spec.get("estimate"),
            "max_credits": spec.get("max_credits"),
            "identity": spec.get("identity"),
            "uploads": [],
            "task_id": None,
            "status": "PREPARING",
            "api_version": RUNWAY_API_VERSION,
        }
        try:
            with self._client() as client:
                if resume_task_id:
                    # Reconcile a task created before a restart: poll it, never create a duplicate.
                    if not _UUID_RE.match(resume_task_id):
                        raise RunwayValidationError("The stored Runway task id is not valid; cannot reconcile it.")
                    task_id = resume_task_id
                    receipt.update({"task_id": task_id, "status": "RECONCILING", "resumed": True})
                    self._task_id = task_id
                else:
                    media = self._resolve_media(client, spec, job, receipt, model)
                    endpoint, body = build_request(model, spec.get("params") or {}, media)
                    receipt["request"] = redact_request(body)
                    if self._cancel.is_set():
                        raise RunwayTaskCancelled("Cancelled before the Runway task was created.")

                    created = client.create_task(endpoint, body)
                    task_id = str(created["id"])
                    self._task_id = task_id
                    receipt["task_id"] = task_id
                    receipt["status"] = "PENDING"
                    api_estimate = (created.get("estimatedCost") or {}).get("credits")
                    receipt["api_estimated_credits"] = api_estimate
                    if self.on_task_created is not None:
                        try:
                            self.on_task_created(task_id, dict(receipt))
                        except Exception as hook_error:  # persistence trouble must not orphan the task
                            receipt["persist_warning"] = f"{type(hook_error).__name__}: {hook_error}"
                    cap = spec.get("max_credits")
                    if cap and isinstance(api_estimate, (int, float)) and api_estimate > float(cap):
                        try:
                            client.cancel_task(task_id)
                        except RunwayError:
                            pass
                        receipt["status"] = "CANCELLED"
                        receipt["credits_charged"] = 0
                        raise RunwayBudgetError(
                            f"Runway estimated {api_estimate} credits for task {task_id}, above the {cap}-credit "
                            "per-job cap; the task was cancelled before it ran."
                        )
                    if self._cancel.is_set():
                        client.cancel_task(task_id)
                        raise RunwayTaskCancelled(f"Runway task {task_id} was cancelled by the user.")

                def on_update(task: dict[str, Any]) -> None:
                    receipt["status"] = task.get("status")
                    if task.get("progress") is not None:
                        receipt["progress"] = task.get("progress")

                task = client.wait_for_task(
                    task_id,
                    timeout_s=float(self.config.timeout_s),
                    poll_interval_s=float(self.config.poll_interval_s),
                    cancel_event=self._cancel,
                    on_update=on_update,
                )
                status = str(task.get("status"))
                receipt["status"] = status
                receipt["credits_charged"] = (task.get("cost") or {}).get("credits")
                receipt["failure_code"] = task.get("failureCode")
                receipt["created_at"] = task.get("createdAt")
                if status == "FAILED":
                    raise RunwayTaskFailed(task_id, task.get("failure"), task.get("failureCode"))
                if status == "CANCELLED":
                    raise RunwayTaskCancelled(f"Runway task {task_id} was cancelled.")
                downloads = self._download_outputs(client, task_id, list(task.get("output") or []))
            receipt["downloads_sha256"] = {Path(path).name: sha256_file(path) for path in downloads}
            outputs = self._postprocess_audio(downloads, spec, job, receipt)
        except RunwayError as exc:
            exc.receipt = {**receipt, **exc.receipt}
            raise
        finally:
            self._task_id = None
            staging = self._staging_dir()
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)

        receipt["outputs"] = [Path(path).name for path in outputs]
        history = {key: value for key, value in task.items() if key != "output"}
        return {"prompt_id": task_id, "outputs": outputs, "history": history, "cloud_receipt": receipt}
