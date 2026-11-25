# The Free builder (offline, prompt-first)

A separate nav button next to Make: build avatars and full video scenes on this computer's GPU from
one prompt, free, with no cloud credits. Runway stays available in Make as the final quality gate.

## The flow

One prompt box. Two things can come out:

| What you make | How it works |
|---|---|
| **Avatar photo** | FLUX.2 [klein] text-to-image on the GPU. The photo lands in My stuff, where the usual actions apply ("Presenter option" makes it her for every future recipe). |
| **Video scene** | The one long prompt is split into shots (sentence boundaries, ~5 s each, up to 12 shots ≈ a minute). Each shot renders through the proven Wan 2.2 image-to-video pipeline, each shot starts from the previous shot's last frame, and the last shot's completion joins them into one video with ffmpeg. |

Start pictures are optional. With one, the scene starts from it. Without one, the builder first
makes a start photo with FLUX.2 and then starts the scene automatically — one click, two stages.

## Models (sized for the RTX 4070 Laptop, 8 GB)

### Photos (dropdown in the Free builder)

| Model | File | Steps | Speed | License |
|---|---|---|---|---|
| FLUX.2 klein 4B (default) | `flux-2-klein-4b-Q5_K_M.gguf` (3.1 GB) | 8, CFG 1.2 | ~2 min | Apache 2.0 (commercial) |
| FLUX.2 klein 9B Base | `flux-2-klein-base-9b-Q4_K_M.gguf` (5.9 GB) | 28, CFG 3.5 | ~6 min | **FLUX Non-Commercial** — testing and research only |

Each model pairs with its own Qwen3 text encoder (GGUF, runs on the **CPU**): Qwen3-4B for the
4B, Qwen3-8B for the 9B Base (the 4B's conditioning does not fit the 9B's joint attention — the
live test caught the mismatch). Both use the FLUX.2 VAE (`flux2-vae.safetensors`, 168 MB). The 4B
is a distilled 4-step-class model; the 9B Base is undistilled and needs the full step budget,
which is why it is slow on an 8 GB card.

Why FLUX.2 klein over SDXL: it is the 2026 sweet spot for small cards — modern rectified-flow
quality and prompt adherence at 4 steps, Apache 2.0, and community-verified 8 GB workflows
(ComfyUI-GGUF + CPU text encoder). SDXL checkpoints also work by pointing the workflow at a
checkpoint, but no SDXL files survived the Nov 2026 engine reinstall.

### Video (dropdown in the Free builder)

| Model | Files | Steps | Speed per 5 s shot |
|---|---|---|---|
| Wan 2.2 A14B + Lightning (best) | 2× GGUF Q4_K_S (8.75 GB each) + 2 Lightning LoRAs | 4, CFG 1 | ~6 min (measured: 345 s, 7.5 GB peak) |
| Wan 2.2 TI2V 5B Turbo (faster) | `Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf` (3.8 GB) + `Wan2.2_VAE.safetensors` (1.4 GB) | 4, CFG 1 | ~5 min (measured: 277 s) |

Both share the UMT5-XXL fp8 text encoder, and both render 480p requests / 81 frames / 16 fps
through the same `WanImageToVideo` node. The 5B's 48-channel Wan 2.2 VAE decodes at **2× the
requested canvas** — a "tall" (480×832) shot comes out as 960×1664 — so 5B files are bigger and
sharper than the 14B's. Chained shots stay consistent because every shot's start frame is resized
to the same requested canvas before rendering.

## CPU offload is part of the design

For the VRAM- and CUDA-intense runs:

- ComfyUI's DynamicVRAM streams UNet weights between the 62 GB of RAM and the 8 GB GPU
  ("0 MB loaded, 8475 MB offloaded" during the 14B render) — always on.
- The FLUX.2 Qwen3 text encoder **always** runs on the CPU (`CLIPLoader device="cpu"`), keeping
  VRAM for the UNet. This is the community-verified 8 GB setting.
- The Wan UMT5 encoder follows the **VRAM mode** dropdown:
  - *Balanced (default)*: encoder on the GPU — the measured fast path (7.5 GB peak on the 14B).
  - *CPU offload*: encoder on the CPU — slower prompts, more VRAM headroom. Use it when CUDA
    error 999/OOM appears or the desktop is busy.

## GPU cleaning during and after every job

Every Free builder job runs on the `low_memory` profile, which forces the safe GPU memory
manager's rules (`docs/GPU_MEMORY_MANAGER.md`):

- `POST /free` with `unload_models` + `free_memory` **before** the render,
- a VRAM trace with the peak recorded **during** the render,
- `POST /free` again **after** every job (forced after failures too).

Nothing ever kills another program's GPU process.

## Install

```bash
scripts/setup_local_engine.sh      # ComfyUI + the Wan 14B stack (already installed)
scripts/setup_offline_builder.sh   # FLUX.2 4B + 9B + Wan 5B Turbo (~16 GB more)
```

The second script is re-runnable, verifies every file against its published SHA-256, resumes
partial downloads, and stops before a download that would leave less than 15 GB free. Skip the
non-commercial 9B with `AVATAR_V2_FORGE_9B=0`.

## Verified on this computer

- Wan 2.2 A14B pipeline: measured 2026-11-22 (docs/LOCAL_ENGINE.md) — 345 s per 5 s clip, 7.5 GB peak.
- FLUX.2 klein 4B text-to-image: rendered live 2026-11-24 through the app's workflow builder at
  512×768, 4 steps — output `avatar_v2/forge_test_00001_.png` in ComfyUI's output folder. Log shows
  the design working as intended: Qwen3 encoder on the CPU (3.2 GB staged in RAM), UNet streamed
  by DynamicVRAM (3 GB loaded, partial unloads during sampling), VAE 160 MB.
- Wan 2.2 5B Turbo: rendered live 2026-11-24 — 81 frames, 4 steps, **277 s**, output
  `avatar_v2/forge_live_5b_00001_.mp4` (960×1664, 5.06 s, H.264). First attempt failed at VAEDecode
  with the 16-channel Wan 2.1 VAE ("got 48 channels"); the 48-channel `Wan2.2_VAE.safetensors`
  fixed it — that is why the 5B is pinned to the 2.2 VAE in the model manifest.
- FLUX.2 klein 9B Base: first live attempt failed at the sampler with a conditioning shape
  mismatch ("512x7680 vs 12288x4096") — the 9B needs the Qwen3-8B encoder, not the 4B's Qwen3-4B.
  Re-verified live with `Qwen3-8B-Q4_K_M.gguf`: 42 s at 8 steps / 512×768 (output
  `avatar_v2/forge_test_9b_00001_.png`), so the full 28-step portrait lands around the 6-minute
  estimate.

Two loader details the live schema check settled (ComfyUI 0.46 + ComfyUI-GGUF):

- The Qwen3 encoder is a GGUF, so it loads through **CLIPLoaderGGUF** (type `flux2`), which has no
  per-node device input. In this ComfyUI build text encoders run on the CPU by default — the app
  never passes `--gpu-only`, so the CPU-offload design holds. Keep `--gpu-only` out of
  `comfyui_args` to preserve it.
- The Wan UMT5 encoder loads through the regular CLIPLoader, whose optional `device` input takes
  `default`/`cpu` — that is what the VRAM mode dropdown switches.
