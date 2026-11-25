# Sprint 2 — Follow-Up Verification, RAM/Swap Forensics, Residency & ResourcePlanner Design

**Date:** 2026-11-25  
**Branch:** `sprint-1-gpu-memory-foundation` (continuing Sprint 2 on branch)  
**Live GPU Benchmark Status:** **Partially Verified** — ComfyUI PID `91038` was actively executing an owner Wan 2.2 A14B render (`queue_running` active, 5,233 MiB VRAM, 73°C, 100% GPU util) during Sprint 2 execution. Per Section 9 & Section 10 execution boundaries, the active workload was never interrupted and no live GPU benchmarks were injected on top of it.

---

## 1. Sprint 1 Follow-Up Verification

### 1.1 Restored Files Verification (`policy.py`, `avatar_designer.py`, `prompt_presets.py`)
- **`avatar_v2/policy.py`:** Was deleted in the pre-Sprint-1 working tree, which broke imports across `cli.py`, `composer.py`, `desktop.py`, `desktop_studio.py`, and 4 test suites. Restoring from `HEAD` (`88df7ec`) restored required runtime & adult-only safety guards with zero loss of legitimate code.
- **`avatar_v2/avatar_designer.py`:** Line 123 had a syntax-corrupted truncation (`apparent_age_years: int |` missing `None = None,`), causing `SyntaxError` across 9 test modules. Restored cleanly from `HEAD`.
- **`avatar_v2/prompt_presets.py`:** Had a partial string edit weakening `ADULT_PROMPT_NOTICE` ("no restrictions") inconsistent with `policy.py` and existing tests. Restored cleanly from `HEAD`.
- **Commit & Remote Preservation Verified:** Sprint 1 commit `bf400019f1cf23b59b759311d6a928292ca6dc5a` matches `origin/sprint-1-gpu-memory-foundation` identically.
- **Preserved Bundles & Environment:** `dist/AvatarV2` (active bundle), all 9 `dist/AvatarV2-prev-*` rollback directories, and `/home/jd/AvatarForge` remain untouched.

### 1.2 Exact Activation & Rollback Procedure
> [!IMPORTANT]
> **Restarting ComfyUI alone does NOT activate Avatar V2 application changes** (`render_queue.py`, `gpu_memory.py`, `desktop.py`, `resource_monitor.py`, `workflows_builder.py`). The desktop launcher (`scripts/launch_avatar_v2.sh`) runs the packaged PyInstaller binary `dist/AvatarV2/AvatarV2` unless `AVATAR_V2_FROM_SOURCE=1` is set.

- **Option A — Activate from Source (Immediate Testing):**
  1. Wait for any active render to finish and close the running Avatar V2 window cleanly.
  2. Run `AVATAR_V2_FROM_SOURCE=1 scripts/launch_avatar_v2.sh`.
- **Option B — Activate via Isolated Rebuilt Bundle (Production Desktop Icon):**
  1. Build into `dist/AvatarV2.next` (leaving active `dist/AvatarV2` untouched):
     `rm -rf build/dist_next dist/AvatarV2.next && .venv314/bin/pyinstaller --noconfirm --distpath build/dist_next --workpath build/pyinstaller_next scripts/avatar_v2.spec && mv build/dist_next/AvatarV2 dist/AvatarV2.next`
  2. Close and relaunch Avatar V2; `launch_avatar_v2.sh` automatically archives the previous bundle to `dist/AvatarV2-prev-<timestamp>` and promotes `dist/AvatarV2.next`.
- **Rollback Procedure:**
  - Source rollback: `git checkout avatar-v2-presenter-upgrade` (or `git revert <commit>`).
  - Bundle rollback: close Avatar V2, rename `dist/AvatarV2` aside, and move the desired `dist/AvatarV2-prev-<timestamp>` back to `dist/AvatarV2`.

---

## 2. Live RAM & Swap Forensics (Section 7 — Empirical `/proc` Breakdown)

Direct read-only inspection of ComfyUI PID `91038` (`/proc/91038/status`, `/proc/91038/smaps_rollup`, `/proc/91038/maps`) revealed the exact root cause of the ~27 GB idle / ~32 GB active RSS and 99.97% swap occupancy:

| Metric (Live PID `91038`) | Measured Value | Attribution |
|---|---|---|
| `VmRSS` / `Pss` (Active Wan 14B) | **32,211 MB (30.7 GiB)** (`27,597 MB` when idle) | Dominant host RAM consumer |
| `RssAnon` (`Private_Dirty`) | **28,282 MB (27.0 GiB — 88% of RSS!)** | `ComfyUI-GGUF` anonymous CPU weight tensors + `comfy-aimdo` host buffers + unbounded `--cache-ram` + glibc malloc arena retention |
| `RssFile` (`Private_Clean`) | **3,720 MB (3.55 GiB — 11.5% of RSS)** | Memory-mapped model files (`Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf`, Lightning LoRA, `umt5_xxl`, `wan_2.1_vae`) |
| `VmLck` / `VmPin` | **0 kB** | NVIDIA `cudaHostRegister` pins pages in kernel driver tables rather than `VmPin` |
| **ComfyUI `VmSwap`** | **1,968 kB (< 2 MB!)** | **ComfyUI itself is NOT swapping!** |
| **Linux PSI (`/proc/pressure/memory`)** | **`some avg10=0.00`, `full avg10=0.00`** | **Zero active memory stall / thrashing** |
| **Who occupies the 8 GiB Swap?** | `clamd` (490 MB), `mongod` (172 MB), `soffice.bin` (105 MB), `gnome-shell` (99 MB), `next-server` (67 MB), `chrome`, Docker | Cold, idle background daemons paged out by Linux when ComfyUI grew to 30–35 GiB |

**Key Conclusions:**
1. High swap percentage (`99.97%`) on this workstation was **cold background daemon swap**, NOT active ComfyUI thrashing—empirically confirming why `resource_monitor.py` gates swap severity using `pswpin`/`pswpout` rates and Linux PSI.
2. ComfyUI's `27.6 GB` idle RSS is **anonymous heap retention (`RssAnon`, 88%)** from unbounded `--cache-ram` (default = 100% of host RAM inactive!) and `ComfyUI-GGUF` host tensors—not a VRAM leak (idle GPU dropped to `224 MiB`) and not `mmap` file pages (`3.55 GiB`). Sprint 1's `--cache-ram 6 24` directly bounds this retention on the next ComfyUI restart.

---

## 3. Evaluation of the 85% VRAM Retention Threshold (Section 4)

Measured peak VRAM across the verified workstation baseline:
- **FLUX.2 klein 4B:** `5,423–5,653 MiB / 8,188 MiB` (**66–69%**)
- **FLUX.2 klein 9B:** `5,481 MiB` UNet + ~1,200 MiB desktop + CUDA context ≈ **~7,000–7,400 MiB (85.5–90.4%)**
- **Wan 2.2 A14B:** `7,169–7,649 MiB / 8,188 MiB` (**87.6–93.4%**)

**Finding:** A fixed `85%` (`6,960 MiB`) `nvidia-smi` eviction threshold **will unnecessarily evict FLUX.2 9B** between consecutive 9B stills even when zero allocation pressure exists and the identical 9B model is needed immediately for the next job!

**Recommended Retention Policy for Sprint 3 `ResourcePlanner`:**
1. **Preserve `AVATAR_V2_GPU_THRESHOLD=0.85`** as the backward-compatible `GPUMemoryManager` safety threshold when no planner hint is present or on idle/unknown transitions.
2. **Adopt Upcoming-Job + Component + Pressure-Aware Retention:**
   - **Same Model Fingerprint + Healthy Pressure:** Retain warm models across consecutive runs even above 85% `nvidia-smi` occupancy (relying on ComfyUI's `cudaMallocAsync` pool).
   - **Same Family (Shared Components):** Delegate selective partial eviction to ComfyUI's native `ModelPatcher` / `comfy-aimdo` rather than forcing global `unload_all_models()`.
   - **Cross-Family Switch (`FLUX.2 ↔ Wan 2.2 ↔ Ollama VLM`), Error/Cancel, or Emergency Resource Pressure:** Force full `POST /free (unload_models=True, free_memory=True)`.

---

## 4. Component-Aware Model Residency Matrix (Section 5)

| Workflow Transition | Shared Components | Non-Shared Components | Recommended Eviction Action |
|---|---|---|---|
| **FLUX.2 4B → FLUX.2 4B** | UNet (`4B`), TE (`Qwen3-4B`), VAE (`flux2-vae`) | None | **Retain all** (skip `/free`) |
| **FLUX.2 9B → FLUX.2 9B** | UNet (`9B`), TE (`Qwen3-8B`), VAE (`flux2-vae`) | None | **Retain all** (upcoming-job-aware, skip fixed-85% eviction) |
| **FLUX.2 4B ↔ FLUX.2 9B** | VAE (`flux2-vae`, 160 MB) | UNet (`4B` vs `9B`), TE (`Qwen3-4B` vs `Qwen3-8B`) | **Full unload** (VAE reload is <0.2 s; keeping 3–5.5 GB UNet causes OOM) |
| **Wan 2.2 A14B → Wan 2.2 A14B (Scene Shots)** | High/Low UNet, LoRAs, **TE (`umt5_xxl`, 6.27 GiB)**, VAE (`wan_2.1_vae`) | None | Evaluate ComfyUI-native retention vs `/free` in Sprint 2 live trials (potential 20–40 s/shot savings on 6–12 shot scenes!) |
| **Wan 2.2 A14B ↔ Wan 2.2 5B / Fun-Control** | **TE (`umt5_xxl`, 6.27 GiB)** | UNet (`A14B` vs `5B`), VAE (`wan_2.1` 16-ch vs `Wan2.2` 48-ch) | Let ComfyUI evict UNet/VAE while keeping `umt5_xxl` staged in RAM (`ResourcePlanner` hint; do NOT build a second competing model manager) |
| **FLUX.2 ↔ Wan 2.2** | **None** | All (UNet, TE, VAE) | **Mandatory full `POST /free (unload_models=True)`** |
| **ComfyUI ↔ Ollama GPU VLM** | **None** | Separate OS processes | **Mandatory full `POST /free` before Ollama** (or run Ollama on CPU `num_gpu=0`) |

---

## 5. Attention Backend Status (Section 6)

- **PyTorch SDPA (`Using pytorch attention`):** Verified active baseline on RTX 4070 (`sm_89`), PyTorch `2.14.1+cu130`.
- **`comfy-kitchen 0.2.42`:** Verified installed with `int8_attention_is_available=True` and `sol_attn_is_available=True`; activates automatically in ComfyUI `0.46.0` for checkpoints carrying `comfy_kitchen_int8`/`comfy_kitchen_sol` metadata (not present in current GGUF files).
- **SageAttention 2 (CUDA C++):** Deferred/rejected for production venv due to verified `nvcc 12.1 != torch cuda 13.0` mismatch.
- **SageAttention 1.0.6 (Triton wheel):** Candidate for isolated non-production venv A/B test once GPU is idle.

---

## 6. Sprint 3 `ResourcePlanner` Requirements Derived From Sprint 2

1. Consume `ResourceGuard` (`avatar_v2/resource_monitor.py`) + `GPUMemoryManager` + ComfyUI `/system_stats` + upcoming queue peek (`RenderQueue`).
2. Compute job `model_fingerprint` `(family, unet, text_encoder, vae)` to drive upcoming-job-aware retention and avoid unnecessary fixed-85% eviction of FLUX.2 9B.
3. Select internal execution profiles (`P0_FULL_GPU`, `P1_PARTIAL_GPU_CPU_TE`, `P2_DYNAMIC_STREAM`, `P3_BLOCK_OFFLOAD_TILED`, `P4_THERMAL_OR_RAM_GUARD`, `P5_AUTO_RECOVERY`) transparently behind the existing UI.
4. Enforce predictive storage checks before queueing multi-shot scenes and continuous script batches.
5. Schedule Ollama prompt compilation / VLM QA either on CPU (`num_gpu=0`) or strictly after ComfyUI `POST /free` with `keep_alive=0`.

---

## 7. Live Controlled Benchmark Commands (Ready to Run When GPU Is Idle)

Once your active ComfyUI render completes and ComfyUI is restarted with `--cache-ram 6 24`, run each isolated A/B comparison with:

```bash
# 1. FLUX.2 9B cold vs warm + retention comparison (3 trials)
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario flux2_9b --trials 3
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario flux2_9b --trials 3 --keep-loaded

# 2. FLUX.2 9B untiled vs tiled VAE
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario flux2_9b --trials 3 --tiled-vae

# 3. Wan 2.2 14B baseline vs tiled VAE vs CPU text encoder (supply an existing start PNG)
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario wan22_14b --trials 3 --init-image <start.png>
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario wan22_14b --trials 3 --tiled-vae --init-image <start.png>
.venv314/bin/python scripts/gpu_memory_benchmark.py --scenario wan22_14b --trials 3 --wan-clip-device cpu --init-image <start.png>
```
