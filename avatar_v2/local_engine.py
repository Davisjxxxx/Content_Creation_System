"""The local engine: ComfyUI + Wan 2.2 image-to-video, sized for an 8 GB NVIDIA GPU.

``scripts/setup_local_engine.sh`` installs it into ``~/AvatarForge``. The desktop app starts ComfyUI
when it opens (if installed) and stops it on exit when the app started it. Studio's "Moving clip"
recipe can render here for free; every other recipe uses Runway, because nothing else runs well
enough on 8 GB of VRAM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_BUNDLE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
WORKFLOW = _BUNDLE_ROOT / "workflows" / "wan22_i2v_14b_gguf_api.json"
WORKFLOW_5B = _BUNDLE_ROOT / "workflows" / "wan22_i2v_5b_turbo_gguf_api.json"
ENGINE_LABEL = "Wan 2.2 14B (GGUF Q4_K_S) + Lightning 4-step"

# Workflow placeholder -> (ComfyUI models subfolder, file name). Must match setup_local_engine.sh.
MODELS: dict[str, tuple[str, str]] = {
    "WAN_HIGH_MODEL": ("diffusion_models", "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf"),
    "WAN_LOW_MODEL": ("diffusion_models", "Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf"),
    "WAN_HIGH_LORA": ("loras", "wan2.2_i2v_lightning_4step_high.safetensors"),
    "WAN_LOW_LORA": ("loras", "wan2.2_i2v_lightning_4step_low.safetensors"),
    "WAN_TEXT_ENCODER": ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "WAN_VAE": ("vae", "wan_2.1_vae.safetensors"),
}

# The Wan 2.2 TI2V 5B Turbo (Apache 2.0, 4-step distilled, no LoRAs needed) is the faster
# video option. It shares the 14B's text encoder but needs the 48-channel Wan 2.2 VAE
# (the A14B GGUF is packed for the 16-channel Wan 2.1 VAE).
# Must match scripts/setup_offline_builder.sh.
MODELS_5B: dict[str, tuple[str, str]] = {
    "WAN5B_MODEL": ("diffusion_models", "Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf"),
    "WAN5B_VAE": ("vae", "Wan2.2_VAE.safetensors"),
}

# Both video models render the same 480p shapes through the same WanImageToVideo node.
VIDEO_MODELS: dict[str, dict[str, Any]] = {
    "a14b": {
        "label": "Wan 2.2 A14B + Lightning (best quality)",
        "engine": "wan22_i2v_14b_gguf",
        "workflow": WORKFLOW,
        "models": MODELS,
        "estimated_minutes": 6,
    },
    "5b": {
        "label": "Wan 2.2 5B Turbo (faster)",
        "engine": "wan22_i2v_5b_turbo",
        "workflow": WORKFLOW_5B,
        "models": MODELS_5B,
        # Measured 2026-11-24: 277 s for a 5 s 480p-requested clip (the 48-ch Wan 2.2 VAE
        # decodes at 2x the requested canvas, so files come out at 960x1664 for "tall").
        "estimated_minutes": 5,
    },
}

# 480p sizes (multiples of 16): phone-shaped, the 3:4 photo shape, and widescreen.
SHAPES: dict[str, tuple[int, int]] = {"tall": (480, 832), "portrait": (480, 640), "wide": (832, 480)}
FPS = 16
FRAMES = 81  # Wan uses 4n+1 frames: 81 frames at 16 fps is about 5 seconds
SECONDS = round(FRAMES / FPS)
NEGATIVE = "blurry, distorted face, extra fingers, morphing, flicker, text, watermark, frozen still image"
PROFILE = "low_memory"  # forces the safe GPU-memory cleanup before every local render
# Measured on the RTX 4070 Laptop GPU (8 GB), 2026-11-22: 345 s for a 5 s 480p clip (docs/LOCAL_ENGINE.md).
ESTIMATED_MINUTES = 6


def video_model_ids() -> list[str]:
    return list(VIDEO_MODELS)


def video_model_info(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-model install state: which files each video model needs and whether they exist."""
    comfyui_dir = Path(str(settings.get("comfyui_dir") or "")).expanduser()
    has_comfy = comfyui_dir.is_dir() and (comfyui_dir / "main.py").is_file()
    out: dict[str, dict[str, Any]] = {}
    for model_id, model in VIDEO_MODELS.items():
        missing = [
            name
            for subfolder, name in model["models"].values()
            if not has_comfy or not (comfyui_dir / "models" / subfolder / name).is_file()
        ]
        out[model_id] = {
            "id": model_id,
            "label": model["label"],
            "minutes": model["estimated_minutes"],
            "missing_models": missing,
            "installed": bool(has_comfy and not missing),
        }
    return out


def inspect(settings: dict[str, Any]) -> dict[str, Any]:
    """What is installed, without contacting ComfyUI (the default A14B video model)."""
    comfyui_dir = Path(str(settings.get("comfyui_dir") or "")).expanduser()
    python = Path(str(settings.get("comfyui_python") or "")).expanduser()
    has_comfy = comfyui_dir.is_dir() and (comfyui_dir / "main.py").is_file()
    has_python = python.is_file()
    missing = [
        name for subfolder, name in MODELS.values() if not (comfyui_dir / "models" / subfolder / name).is_file()
    ] if has_comfy else [name for _subfolder, name in MODELS.values()]
    has_gguf_loader = has_comfy and (comfyui_dir / "custom_nodes" / "ComfyUI-GGUF").is_dir()
    return {
        "installed": bool(has_comfy and has_python and has_gguf_loader),
        "missing_models": missing,
        "comfyui_dir": str(comfyui_dir) if has_comfy else "",
        "workflow_present": WORKFLOW.is_file(),
        "video_models": video_model_info(settings),
    }


def model_placeholders() -> dict[str, str]:
    return {key: name for key, (_subfolder, name) in MODELS.items()}


def model_placeholders_5b() -> dict[str, str]:
    return {key: name for key, (_subfolder, name) in MODELS_5B.items()}


def _normalize_video_model_key(model_id: str) -> str:
    raw = str(model_id or "").strip().lower()
    if raw in {"5b", "wan22_5b", "wan22_i2v_5b_turbo", "wan22_ti2v_5b"}:
        return "5b"
    if raw in {"a14b", "14b", "wan22_a14b", "wan22_14b", "wan22_i2v_14b_gguf"}:
        return "a14b"
    raise ValueError(f"Unsupported local Wan video model '{model_id}'; expected 'wan22_a14b' or 'wan22_5b'.")


def workflow_for_video_model(model_id: str) -> Path:
    """Return the exact bundled workflow JSON matching ``model_id`` (RC-12)."""
    key = _normalize_video_model_key(model_id)
    return Path(VIDEO_MODELS[key]["workflow"])


def placeholders_for_video_model(model_id: str) -> dict[str, str]:
    """Return exact ComfyUI model placeholder bindings matching ``model_id`` (RC-12)."""
    key = _normalize_video_model_key(model_id)
    if key == "5b":
        return {
            **model_placeholders_5b(),
            "WAN_TEXT_ENCODER": MODELS["WAN_TEXT_ENCODER"][1],
            "WAN_VAE": MODELS_5B["WAN5B_VAE"][1],
        }
    return model_placeholders()
