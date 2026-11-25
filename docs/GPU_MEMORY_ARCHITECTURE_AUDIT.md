# Avatar V2 — Supplemental GPU Memory Architecture and Large Model Optimization Audit

**Audit Date:** 2026-11-25  
**Target Workstation (Verified Live):** NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB total / 7,781 MiB CUDA-usable, Ada Lovelace `sm_89`, PCIe Gen 4 × 8, Driver `580.178.04`) · Intel Core i9-13980HX (24 cores / 32 threads) · 62 GiB RAM (63,911 MiB) + 8 GiB swap · 915 GB NVMe (**47–48 GB free, 95% full**)  
**Runtime Stack (Verified Live):** ComfyUI `0.46.0` (`4c2b826`) · `comfy-aimdo 0.6.0` · `comfy-kitchen 0.2.42` (`int8_attention_is_available=True`, `sol_attn_is_available=True`) · PyTorch `2.14.1+cu130` (`cudaMallocAsync`) · CUDA `13.0` · cuDNN `92400` · Triton `3.8.0` · `ComfyUI-GGUF` (`6ea2651`) · Ollama (19 local models installed)

---

## 18. Established Operational Baseline (Independently Verified)

The owner's operational baseline—**FLUX.2 9B** and **Wan 2.2 14B** executing without CUDA OOM crashes on the 8 GB RTX 4070 Laptop GPU—is independently verified from `~/.avatar_v2/history/*.json` (116 ComfyUI records) and `~/.avatar_v2/logs/comfyui.log`.

| Workflow | Samples | Resolution / Frames / Steps | Measured Duration | Peak Total GPU VRAM (`nvidia-smi`) | Resident vs Offloaded Weights (ComfyUI Log) |
|---|---|---|---|---|---|
| **Wan 2.2 A14B + Lightning (Q4_K_S)** | 25 `done` | 480×832 / 81f / 4 steps | **255.0–520.8 s (avg 325.3 s)** | **7,169–7,649 MiB (avg 7,393 MiB)** | Earlier: `1,535 MB` loaded / `6,940 MB` offloaded. Recent: **`0.00 MB` loaded / `8,475 MB` offloaded** (`62.6–65.2 s/it`) |
| **Wan 2.2 5B Turbo (Q5_K_M)** | 1 `done` (log) | 480×832 req → 960×1664 out / 81f / 4 steps | **276.4 s** | Pending controlled trace | `2,064 MB` loaded / `1,656 MB` offloaded; 48-ch VAE `1,344 MB` staged |
| **FLUX.2 klein 4B (Q5_K_M)** | 5 `done` | 896×1280 / 8 steps (CFG 1.2) | **46.5–59.5 s (avg 53.5 s)** | **5,423–5,653 MiB (avg 5,542 MiB)** | Qwen3-4B: `3,188 MB` full load; UNet: `3,039 MB` full load (`2.89–3.09 s/it`); VAE: `160 MB` staged |
| **FLUX.2 klein 9B Base (Q4_K_M)** | 1 `done` (log) | 512×768 / 8 steps (live verification) | **40.4 s** (~3.5–6 min at 28 steps) | Pending controlled trace | Qwen3-8B: `721 MB` loaded / `6,108 MB` offloaded; UNet: **`5,481 MB` loaded / `346 MB` offloaded** (`2.99 s/it`) |

Live hardware observation during active Wan 2.2 14B generation: GPU 73–74°C at 99–100% util; Chrome GPU process 502–581 MiB VRAM; ComfyUI RSS **35.3 GiB** on 62 GiB host RAM; Swap **8,190 / 8,191 MiB (99.9% full)**; NVMe **47–48 GB free (95% full)**.

---

## 19. Existing CUDA Memory Management Audit

### Layer Ownership
1. **Avatar V2 Controller (`avatar_v2/gpu_memory.py`, `render_queue.py`, `providers/comfyui.py`, `desktop.py`):** Out-of-process `nvidia-smi` monitoring, 85% cleanup threshold (`AVATAR_V2_GPU_THRESHOLD`), 5 s peak VRAM trace (`AVATAR_V2_GPU_CHECK_INTERVAL`), `POST /free` cleanup at job boundaries, single-retry crash recovery, and scoped shutdown of the app-owned ComfyUI PID.
2. **ComfyUI 0.46 + `comfy-aimdo 0.6.0` (`model_management.py`, `memory_management.py`, `pinned_memory.py`):** Native `DynamicVRAM` (55,719 MB pinned host ceiling, 2 async CUDA offload streams, `fast_disk=True` file-slice reads) for native safetensors (`WanTEModel`, `WanVAE`, `AutoencoderKL`, `MiniMaxH3`), `soft_empty_cache()` (`synchronize + empty_cache + ipc_collect`), and auto-OOM unload.
3. **`ComfyUI-GGUF` (`ops.py`, `nodes.py`):** Bypasses `comfy-aimdo` DynamicVRAM (`fast_disk=False`) and uses legacy `ModelPatcher` partial loading + per-layer on-the-fly dequantization (`dequantize_tensor`).
4. **PyTorch `2.14.1+cu130`:** Native `cudaMallocAsync` stream-ordered allocator (low fragmentation) and PyTorch SDPA (`Using pytorch attention`).

### Why FLUX.2 9B and Wan 2.2 14B Succeed on 8 GB VRAM
- 4-bit GGUF compresses Wan 2.2 14B experts to 8.75 GB each (executed sequentially: HighNoise steps 0–2, LowNoise steps 2–4) and FLUX.2 9B to 5.91 GB (~93% resident in VRAM).
- 62 GiB host RAM absorbs staged weights while `ComfyUI-GGUF` streams offloaded layers over PCIe Gen 4 × 8 and frees temporary dequantized weights after each `F.linear`.
- Lightning 4-step CFG-1 minimizes activation memory; `cudaMallocAsync` minimizes allocator fragmentation; Avatar V2 `POST /free` prevents cross-job retention leaks.

### Five Concrete Bottlenecks & Bugs Found
1. **`/free` `unload_models=False` Bug (Fixed in Sprint 1):** `render_queue.py` sent `{"unload_models": False, "free_memory": True}` to keep FLUX.2 models warm, but ComfyUI `server.py:1299` skips falsy `unload_models` and `main.py:419` defaults `flags.get("unload_models", free_memory)` to `True`—unloading all FLUX.2 models after every single image.
2. **Wan 2.2 14B Degraded to `0.00 MB loaded`:** Desktop/Chrome VRAM + UMT5 staging + reserve headroom dropped resident UNet from 1,535 MB to 0 MB, forcing 100% PCIe streaming (`63 s/it`).
3. **FLUX.2 Qwen3 Not on CPU:** `CLIPLoaderGGUF` loads Qwen3 onto `cuda:0` (`3,188 MB` full load for 4B; `721 MB` loaded / `6,108 MB` offloaded for 8B), sometimes leaving `1,680 MB` stranded in VRAM during `Flux2` denoising.
4. **Host RAM & Swap Exhaustion (Addressed in Sprint 1):** Unbounded `--cache-ram` + 55.7 GB pinned pool pushed ComfyUI RSS to 35.3 GiB and filled 99.9% of 8 GB swap, explaining `Connection reset by peer` crashes in history (`2ec06a4e92`, `31fa7336f4`, `70ebafe9fb`).
5. **Untiled `VAEDecode` & Attention Headroom (Evaluated in Sprint 1):** All baseline workflows used untiled `VAEDecode`; SageAttention is not installed (`nvcc` is 12.1 vs PyTorch `cu130`, so SageAttention 2 source build fails, while PyPI `sageattention==1.0.6` offers a pure-Python+Triton wheel and already-installed `comfy-kitchen 0.2.42` reports `int8_attention_is_available=True`).

---

## 20–23. Dynamic Optimization, Adaptive Resource Intelligence & Larger Model Feasibility

- **Dynamic Optimization:** Fix `/free` retention; bound `--cache-ram` by host RAM (`--cache-ram 6 24` on ≥48 GB RAM); support `VAEDecodeTiled` (`tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8`); keep `cudaMallocAsync` active; never invoke `torch.cuda.empty_cache()` mid-step; benchmark `comfy-aimdo` DynamicVRAM FP8 models and isolated Triton SageAttention in Sprint 2.
- **Graduated Resource Intelligence (`avatar_v2/resource_monitor.py` → Sprint 3 `ResourcePlanner`):** Monitors VRAM (total vs ComfyUI vs external), RAM available %, process RSS/PSS, swap % + `pswpin`/`pswpout` activity, Linux PSI (`/proc/pressure/memory`), GPU temp/util with hysteresis, and predictive disk space (Normal `>=30 GB`, Warning `<30 GB`, Restricted `<20 GB`, Emergency Stop `<15 GB`).
- **Larger Model Classification Summary (Full Details in Audit Table):**
  - *1. Already operational:* FLUX.2 klein 4B (Q5_K_M, Apache 2.0), FLUX.2 klein 9B Base (Q4_K_M, Non-Comm), Wan 2.2 A14B (Q4_K_S, Apache 2.0), Wan 2.2 5B Turbo (Q5_K_M, Apache 2.0), Ollama local LLMs/VLMs (`qwen3-vl:8b-instruct`, `qwen3:8b`, `qwen3.5:9b`).
  - *2. Feasible using existing MM:* FLUX.2 klein Multi-Reference Editing (**0 GB extra disk**), FLUX.2 klein 9B Distilled 4-step (Q6_K/FP8, Non-Comm), FLUX.1 Kontext [dev] 12B GGUF (Non-Comm).
  - *3. Feasible with targeted optimization:* Wan 2.2 A14B FP8 Scaled (`comfy-aimdo` 2-stream async, Apache 2.0), Qwen-Image-Edit 20B (GGUF Q4_K_M + Lightning, Apache 2.0), Wan 2.2 Fun-Control 14B (GGUF Q4_K_S, Apache 2.0).
  - *4. Experimental with acceptable potential:* LTX-2 (FP8/INT8), FramePack (HunyuanVideo progressive I2V), MiniMax H3 (19.9B INT8 — OOM-prone without tight desktop VRAM headroom).
  - *5. Technically executable but operationally impractical:* FLUX.2 [dev] 32B (Q4_K_S ~19 GB UNet + 10–14 GB TE; >30 GB disk, ~6–12+ min/image).
- **Zero-Contention Local Intelligence:** Ollama executes strictly sequentially after ComfyUI `POST /free` with `keep_alive: 0` (already in `candidate_qa.py`) or on the 24-core i9-13980HX CPU (`options: {"num_gpu": 0}`) with 0 MB VRAM usage.

---

## 24–27. Benchmark Protocol, Acceptance Criteria & Preservation Verdict

- **Reproducible Harness:** Implemented in `scripts/gpu_memory_benchmark.py` with active-workload interlock, `--dry-run` preflight, per-trial `nvidia-smi` + `/system_stats` + RSS + PSI + swap-activity + thermal telemetry, and one-variable-at-a-time trial support for Sprint 2 Scenarios B1–B12.
- **Preservation Verdict:** Retain `avatar_v2/gpu_memory.py` as the safety foundation; fix `/free` retention and ComfyUI RAM bounds incrementally; extend with `resource_monitor.py` (Sprint 1) and `ResourcePlanner` (Sprint 3); keep all memory complexity behind the existing UI.
- **Attention Evaluation:** See `docs/SAGEATTENTION_EVALUATION.md` and `scripts/eval_attention_compat.py` for full CUDA 13.0 / `nvcc` 12.1 / `comfy-kitchen 0.2.42` / `sageattention` compatibility details.
