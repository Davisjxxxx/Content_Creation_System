"""The Offline Builder: FLUX.2 [klein] images + Wan 2.2 video, prompt-first and free.

``scripts/setup_offline_builder.sh`` installs the image models into the same ComfyUI used by the
local engine (``scripts/setup_local_engine.sh``). The desktop app's "Free builder" view turns one
long prompt into:

- an avatar photo (FLUX.2 klein text-to-image), and/or
- a video scene: the prompt is split into shots, each shot renders through the proven
  Wan 2.2 image-to-video pipeline, shots continue from the previous shot's last frame,
  and the last shot's completion joins them into one video.

CPU offloading is part of the design for the VRAM-intense runs: ComfyUI's DynamicVRAM streams
UNet weights between RAM and VRAM, the FLUX.2 Qwen3 text encoder always runs on the CPU, and the
Wan UMT5 encoder can be moved to the CPU per job through the "VRAM mode" option. Every job runs
on the ``low_memory`` profile, so the GPU memory manager forces a ``POST /free`` cleanup before
and after each render and records VRAM traces (see docs/GPU_MEMORY_MANAGER.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# FLUX.2 [klein] photo models. Each pairs with its own Qwen3 text encoder (GGUF, runs on CPU)
# and the shared FLUX.2 VAE.
# - 4B: distilled 4-8 steps, Apache 2.0 (commercial). The default. Encoder: Qwen3-4B.
# - 9B Base: undistilled full-capacity model, flux-non-commercial license, more steps.
#   Encoder: Qwen3-8B (the 4B's Qwen3-4B conditioning does not fit the 9B's joint attention).
PHOTO_MODELS: dict[str, dict[str, Any]] = {
    "4b": {
        "label": "FLUX.2 klein 4B (fast · commercial license)",
        "unet": "flux-2-klein-4b-Q5_K_M.gguf",
        "text_encoder": "Qwen3-4B-Q5_K_M.gguf",
        "steps": 8,
        "cfg": 1.2,
        "estimated_minutes": 2,
        "license": "apache-2.0",
    },
    "9b": {
        "label": "FLUX.2 klein 9B Base (quality · non-commercial license)",
        "unet": "flux-2-klein-base-9b-Q4_K_M.gguf",
        "text_encoder": "Qwen3-8B-Q4_K_M.gguf",
        "steps": 28,  # undistilled base: needs a full step budget
        "cfg": 3.5,
        "estimated_minutes": 6,
        "license": "flux-non-commercial",
    },
}
PHOTO_MODEL_IDS = tuple(PHOTO_MODELS)

# Shared FLUX.2 files: the VAE (168 MB). Text encoders are per model above.
SHARED_MODELS: dict[str, tuple[str, str]] = {
    "FORGE_VAE": ("vae", "flux2-vae.safetensors"),
}
# Must match scripts/setup_offline_builder.sh.

# FLUX.2 canvases: multiples of 16, portrait shapes that suit avatars.
PHOTO_SHAPES: dict[str, tuple[int, int]] = {"portrait": (896, 1280), "tall": (832, 1216), "square": (1024, 1024)}
PHOTO_SAMPLER = "euler"
PHOTO_NEGATIVE = (
    "text, words, letters, numbers, typography, captions, labels, signage, logo, watermark, "
    "blurry, distorted face, extra fingers, bad anatomy"
)

# The scene step reuses the Wan 2.2 pipeline from avatar_v2.local_engine (already verified on
# this machine at 480p / 81 frames). Wan video models and their install state live there.

# How the Wan UMT5 text encoder runs. "cpu" frees VRAM for the UNet at the cost of a slower
# prompt encode; "default" is the measured fast path on this machine.
VRAM_MODES: dict[str, dict[str, Any]] = {
    "balanced": {"label": "Balanced (default)", "wan_clip_device": "default"},
    "cpu_offload": {"label": "CPU offload (safer for heavy jobs, a bit slower)", "wan_clip_device": "cpu"},
}

SCENE_MAX_SHOTS = 12  # about a minute of video; a longer prompt becomes more scenes
PROMPT_MAX_CHARS = 8000  # "max length possible" for the prompt box; split into shots below
SHOT_TARGET_CHARS = 220  # a 5-second clip expresses roughly this much action

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$")


def split_scene_prompt(text: str, max_shots: int = SCENE_MAX_SHOTS) -> list[str]:
    """Turn one long description into ordered shot prompts (about 5 seconds each).

    Splits on sentence boundaries, packs short sentences together up to the target
    size, and hard-caps the shot count. The result is a list of trimmed, non-empty
    prompts; an empty input returns an empty list.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    sentences = [part.strip() for part in _SENTENCE_END.split(cleaned) if part.strip()]
    shots: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if buffer and len(candidate) > SHOT_TARGET_CHARS:
            shots.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
        if len(shots) >= max_shots:
            buffer = ""
            break
    if buffer and len(shots) < max_shots:
        shots.append(buffer)
    shots = shots[:max_shots]
    if not shots and cleaned:
        shots = [cleaned[:4000]]
    return shots


def photo_model_choice(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in PHOTO_MODELS else "4b"


def vram_mode_choice(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in VRAM_MODES else "balanced"


def photo_model_missing(settings: dict[str, Any], model_id: str) -> list[str]:
    """The files a photo model needs that are not installed (shared VAE counted once)."""
    comfyui_dir = Path(str(settings.get("comfyui_dir") or "")).expanduser()
    has_comfy = comfyui_dir.is_dir() and (comfyui_dir / "main.py").is_file()
    if not has_comfy:
        names = [
            PHOTO_MODELS[model_id]["unet"],
            PHOTO_MODELS[model_id]["text_encoder"],
            *(name for _sub, name in SHARED_MODELS.values()),
        ]
        return names
    required = [(PHOTO_MODELS[model_id]["unet"], "diffusion_models")]
    required.append((PHOTO_MODELS[model_id]["text_encoder"], "text_encoders"))
    required += [(name, subfolder) for subfolder, name in SHARED_MODELS.values()]
    return [
        name
        for name, subfolder in required
        if not (comfyui_dir / "models" / subfolder / name).is_file()
    ]


def inspect(settings: dict[str, Any]) -> dict[str, Any]:
    """What the Offline Builder needs per photo model, without contacting ComfyUI."""
    comfyui_dir = Path(str(settings.get("comfyui_dir") or "")).expanduser()
    has_comfy = comfyui_dir.is_dir() and (comfyui_dir / "main.py").is_file()
    has_gguf_loader = has_comfy and (comfyui_dir / "custom_nodes" / "ComfyUI-GGUF").is_dir()
    models: dict[str, dict[str, Any]] = {}
    for model_id, model in PHOTO_MODELS.items():
        missing = photo_model_missing(settings, model_id)
        models[model_id] = {
            "id": model_id,
            "label": model["label"],
            "license": model["license"],
            "minutes": model["estimated_minutes"],
            "missing_models": missing,
            "installed": bool(has_gguf_loader and not missing),
        }
    return {
        "installed": bool(has_gguf_loader and not photo_model_missing(settings, "4b")),
        "has_gguf_loader": has_gguf_loader,
        "comfyui_dir": str(comfyui_dir) if has_comfy else "",
        "models": models,
    }


def image_placeholders(model_id: str) -> dict[str, str]:
    model = PHOTO_MODELS[model_id]
    placeholders = {
        "FORGE_UNET": model["unet"],
        "FORGE_TEXT_ENCODER": model["text_encoder"],
    }
    placeholders.update({key: name for key, (_subfolder, name) in SHARED_MODELS.items()})
    return placeholders
