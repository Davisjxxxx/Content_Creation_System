# Your GPU engine (local, free)

Avatar V2 can render the **Moving clip** recipe on this computer's NVIDIA GPU at no cost. Everything
else (photos, voiceovers, talking videos, act-it-out) uses Runway, because nothing local does those
well enough on an 8 GB GPU.

## What it is

| Part | Choice | Why |
|---|---|---|
| Runner | ComfyUI (current release) with ComfyUI-GGUF | ComfyUI's dynamic VRAM manager streams model layers between RAM and the GPU, which is what lets a 14B model run on 8 GB. |
| Video model | Wan 2.2 I2V A14B, GGUF **Q4_K_S** (high- and low-noise experts, 8.75 GB each) | The strongest image-to-video model that fits an 8 GB card; Q4_K_S keeps quality close to Q4_K_M with less memory. |
| Speed-up | Official Wan 2.2 Lightning 4-step LoRAs (lightx2v, rank 64) | 4 steps instead of 20-40, at CFG 1. Settings follow the authors' reference workflow: steps 1-2 on the high-noise expert, steps 3-4 on the low-noise expert, shift 5, euler/simple. |
| Text encoder / VAE | UMT5-XXL fp8 (6.7 GB) / Wan 2.1 VAE | The standard pair for Wan 2.2 14B. |
| Output | 480p, 81 frames at 16 fps (about 5 seconds), H.264 MP4 | What an 8 GB card renders reliably. "Phone" is 480x832; "Like her photo" is 480x640. |

Workflow: `workflows/wan22_i2v_14b_gguf_api.json`. Model list and sizes: `avatar_v2/local_engine.py`.

## Install, start and stop

```bash
scripts/setup_local_engine.sh   # about 27 GB of models plus about 7 GB for ComfyUI and PyTorch
```

- Installs into `~/AvatarForge` (`comfyui/` and `venv/`), which is where the app looks by default.
- Re-runnable: finished files are verified and skipped, partial downloads resume, and every model
  is checked against its published SHA-256. It stops before a download that would leave less than
  15 GB free (`AVATAR_V2_MIN_FREE_GB`).
- PyTorch 2.8 for CUDA 12.8 (works with the 580 driver's CUDA 13), in a Python 3.12 environment.

The app starts ComfyUI when it opens (only if it's installed) and stops it when the app closes,
if the app started it. **Settings → Your GPU engine** shows its state and has Start / Stop
buttons; Stop frees the GPU memory. Extra ComfyUI flags go in the `comfyui_args` setting
(default `--disable-auto-launch`).

## GPU memory safety

Local clips run on the `low_memory` profile, so the queue always asks ComfyUI to free memory
(`POST /free`) before rendering, records a VRAM trace during the render, and cleans up afterwards.
These are the GPU memory manager's rules from `docs/GPU_MEMORY_MANAGER.md`. Nothing ever kills
another program's GPU process.

## Measured on this computer

Measured on 2026-11-22: RTX 4070 Laptop GPU (8 GB) with 62 GB RAM and ComfyUI defaults.

| Measurement | Value |
|---|---|
| First clip (includes loading the models from disk) | _pending_ |
| Peak VRAM during the render | _pending_ |
| Pre-render cleanup | _pending_ |

Runway Gen-4.5 is still faster (about 1-2 minutes) and sharper (720p), so the recipe offers both.
