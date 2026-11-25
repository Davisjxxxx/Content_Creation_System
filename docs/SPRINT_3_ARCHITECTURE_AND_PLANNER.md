# Sprint 3 — Central ResourcePlanner, Prompt Compilers & Guided Mode Architecture

**Date:** 2026-11-25  
**Status:** Software-Verified (CPU-only tests passing); Live Sprint 2/3 GPU Benchmarks Carried Forward (blocked by active owner ComfyUI render PID `91038`)

---

## 1. Delivered Modules & Shared Execution Spine (Prepared for Sprint 4 Script Mode)

All three modes—**Automatic**, **Guided**, and upcoming **Script**—now share a single deterministic backend pipeline without duplicating inference or memory-management code:

```text
User Input (Automatic / Guided / Future Script SDK)
   → StructuredPromptSpec v1.0 (avatar_v2/prompt_schema.py)
   → Reusable Prompt Libraries (avatar_v2/data/prompt_libraries.json + avatar_v2/prompt_library.py)
   → Model-Specific Prompt Compilers (avatar_v2/prompt_compilers.py)
   → Component-Aware ModelRegistry (avatar_v2/model_registry.py)
   → Centralized ResourcePlanner (avatar_v2/resource_planner.py)
   → RenderQueue + GPUMemoryManager (avatar_v2/render_queue.py + gpu_memory.py)
   → ComfyUI 0.46 Native ModelPatcher / comfy-aimdo / cudaMallocAsync
```

## 2. Key Architectural Changes

1. **Component-Aware Model Registry (`avatar_v2/model_registry.py`):**
   - Verifies exact `(subfolder, filename, loader, compat_group)` identity before treating any component as reusable.
   - Confirms `umt5_xxl` (6.4 GB) is safely shared across `wan22_a14b`, `wan22_5b`, and `wan22_funcontrol`; prevents false reuse of `Qwen3-4B` (7680) vs `Qwen3-8B` (12288) and 16-ch `wan_2.1_vae` vs 48-ch `Wan2.2_VAE`.
2. **Centralized `ResourcePlanner` (`avatar_v2/resource_planner.py`):**
   - Replaces fixed-85% VRAM eviction for consecutive identical models (allowing FLUX.2 9B at ~85.5–90.4% total VRAM to stay warm under healthy pressure via `respect_retention=True`).
   - Uses `comfy_native_selective` for intra-family shared-component switches (delegating tensor eviction to ComfyUI's `ModelPatcher`/`comfy-aimdo` rather than building a competing model manager).
   - Forces `full_evict` on cross-family transitions (`FLUX.2 ↔ Wan 2.2 ↔ Ollama VLM`), failures/cancels, or intervention/emergency RAM/swap/thermal pressure.
   - Automatically selects `VAEDecodeTiled` for high-res stills (`>1024×1280`), extended video frames (`>81f`), Wan 2.2 5B 48-ch VAE, or tight VRAM (`<5.2 GB` free), and schedules Ollama VLM on CPU (`num_gpu=0`) whenever warm ComfyUI models would otherwise be evicted.
3. **Model-Specific Deterministic Compilers (`avatar_v2/prompt_compilers.py`):**
   - `Flux2PromptCompiler`: natural declarative prose for Qwen3 + Rectified Flow; strips legacy SDXL `(token:weight)` syntax with warnings; explicitly flags unsupported mandatory video motion/camera attributes instead of silently dropping them.
   - `Wan22PromptCompiler`: motion-first phrasing for UMT5-XXL + 4-step Lightning/Turbo; explicitly warns about CFG 1.0 negative-prompt limitations.
4. **Data-Driven Prompt Libraries & Collapsed Guided Mode (`avatar_v2/data/prompt_libraries.json`, `prompt_library.py`, `desktop_studio.py`, `ui/index.html`):**
   - 16 general-purpose creative categories + 3 validated scene recipes stored strictly outside execution code.
   - Integrated as one collapsed-by-default `<details id="forgeGuided">` panel in `#view-forge` with context-sensitive photo/video attributes, per-attribute Randomize (`🎲`) and Lock (`🔒`) controls, and zero technical/VRAM knobs.

## 3. Carried-Forward Live GPU Verification Gates (Pending Idle GPU)

1. Execute Sprint 2 controlled baseline vs Sprint 1 (`--cache-ram 6 24`, `/free` fix, tiled VAE) vs Sprint 3 (`ResourcePlanner` retention & compiled prompts) once ComfyUI PID `91038` is idle.
2. Measure actual cold vs warm FLUX.2 9B latency under `respect_retention=True` and Wan 2.2 14B ↔ 5B shared `umt5_xxl` transition latency.
3. Capture baseline 13-category adherence scores before Sprint 6/8 intelligence & correction work.
