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
- PyTorch 2.14.1 built for CUDA 13.0 (needs an NVIDIA driver with CUDA 13 support, 580 or newer), in a
  Python 3.12 environment. Current ComfyUI only enables its optimized CUDA kernels on CUDA 13 builds; the
  script's comments show the override for older drivers.

The app starts ComfyUI when it opens (only if it's installed; it is ready in about 10 seconds) and
asks it to quit when the app closes, if the app started it. **Settings → Your GPU engine** shows its
state and has Start / Stop buttons. Stop frees the GPU memory. A loaded engine can take about a minute
to release its models; it is only ever asked to quit (SIGTERM), never force-killed. Extra ComfyUI flags
go in the `comfyui_args` setting (default `--disable-auto-launch`); ComfyUI's own dynamic VRAM manager
needs no tuning flags on 8 GB.

## GPU memory safety

Local clips run on the `low_memory` profile, so the queue always asks ComfyUI to free memory
(`POST /free`) before rendering, records a VRAM trace during the render, and cleans up afterwards.
These are the GPU memory manager's rules from `docs/GPU_MEMORY_MANAGER.md`. Nothing ever kills
another program's GPU process.

## Measured on this computer

Measured on 2026-11-22: RTX 4070 Laptop GPU (8 GB) with 62 GB RAM and ComfyUI defaults.

| Measurement | PyTorch 2.8 / CUDA 12.8, first run (models loaded from disk) | PyTorch 2.14.1 / CUDA 13.0 (the default) |
|---|---|---|
| Whole job, queued to finished | 390 s | **345 s (about 5¾ minutes)** |
| ComfyUI render time | 376 s | 331 s |
| Peak VRAM (whole GPU, including about 1.3 GB used by the desktop) | 7,396 MB of 8,188 | 7,503 MB of 8,188 |
| Pre-render cleanup (`POST /free`, forced by the `low_memory` profile) | requested, recorded | requested, recorded |
| Post-render cleanup (forced) | not verified (that test stopped the engine too early) | requested, freed 224 MB, recorded |
| Output | 480x832, 81 frames, 16 fps, 5.06 s, H.264 | same |

The weights stream from RAM during sampling ("0 MB loaded, 8475 MB offloaded" in ComfyUI's log),
which is what keeps the 14B model within 8 GB; 62 GB of system RAM helps. A test frame kept the
presenter's face, a natural smile and a five-finger open-hand wave.

Runway Gen-4.5 is still faster (about 1-2 minutes) and sharper (720p), so the recipe offers both.
