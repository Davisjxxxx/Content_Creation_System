"""The local engine: ComfyUI + Wan 2.2 14B image-to-video, sized for an 8 GB NVIDIA GPU.

``scripts/setup_local_engine.sh`` installs it into ``~/AvatarForge``. The desktop app starts ComfyUI
when it opens (if installed) and stops it on exit when the app started it. Studio's "Moving clip"
recipe can render here for free; every other recipe uses Runway, because nothing else runs well
enough on 8 GB of VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

WORKFLOW = Path(__file__).resolve().parent.parent / "workflows" / "wan22_i2v_14b_gguf_api.json"
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

# 480p sizes (multiples of 16). "tall" is phone-shaped; "portrait" keeps the 3:4 photo shape.
SHAPES: dict[str, tuple[int, int]] = {"tall": (480, 832), "portrait": (480, 640)}
FPS = 16
FRAMES = 81  # Wan uses 4n+1 frames: 81 frames at 16 fps is about 5 seconds
SECONDS = round(FRAMES / FPS)
NEGATIVE = "blurry, distorted face, extra fingers, morphing, flicker, text, watermark, frozen still image"
PROFILE = "low_memory"  # forces the safe GPU-memory cleanup before every local render
# Measured on the RTX 4070 Laptop GPU (8 GB) on 2026-11-22; refreshed by the live test in the docs.
ESTIMATED_MINUTES = 6


def inspect(settings: dict[str, Any]) -> dict[str, Any]:
    """What is installed, without contacting ComfyUI."""
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
    }


def model_placeholders() -> dict[str, str]:
    return {key: name for key, (_subfolder, name) in MODELS.items()}
