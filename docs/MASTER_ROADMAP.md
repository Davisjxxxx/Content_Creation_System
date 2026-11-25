# Avatar V2 — Master Modernization Roadmap (10-Sprint Plan)

**Status:** Approved (2026-11-25)  
**Target Workstation Baseline:** NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM, `sm_89`, PCIe Gen 4 × 8) · Intel Core i9-13980HX (24C / 32T) · 62 GiB RAM · Local ComfyUI `0.46.0` + PyTorch `2.14.1+cu130` + Ollama  
**Core Preservation Guarantees:**
- Preserve working operation of **FLUX.2 9B** and **Wan 2.2 14B** on 8 GB VRAM.
- Preserve the simple, intuitive approved frontend (`Make` / `Free builder` / `My stuff` / `Settings`); keep memory management internal.
- Local-only offline execution by default; cloud providers (including Runway) require explicit user authorization.
- Preserve existing `GPUMemoryManager` safety rules (`docs/GPU_MEMORY_MANAGER.md`, `AGENTS.md`).

---

## Approved Three-Mode Frontend Interaction Model

1. **Automatic (Default):** Natural-language prompt entry, optional reference assets, automatic prompt interpretation, model selection, workflow routing, and internal resource optimization.
2. **Guided:** Optional task-applicable creative attribute dropdowns, reusable scene recipes, attribute locking, controlled randomization, and model-specific prompt compilation.
3. **Script (Replaces Manual):** Local `.py` selection (plus associated JSON/YAML configs and prompt libraries), static AST validation (no import-on-inspect), concise compatibility report, Start / Pause / Resume / Stop, progress, and output review.

---

## Ten-Sprint Master Roadmap

| Sprint | Title | Core Objectives & Acceptance Gates |
|---|---|---|
| **Sprint 1** | **GPU Memory, Stability & Performance Optimization** | Separate branch; persist audits & roadmap in `docs/`; fix `/free` `unload_models=False` bug in `render_queue.py`; add bounded `--cache-ram` ComfyUI launch args; build `avatar_v2/resource_monitor.py` (graduated RAM/swap/PSI/thermal + predictive storage guard); schema-verified `VAEDecodeTiled` support; isolated SageAttention & `comfy-kitchen` compatibility evaluation (no unverified prod venv installs); reproducible benchmark harness (`scripts/gpu_memory_benchmark.py`); regression tests. |
| **Sprint 2** | **Controlled Benchmarks & Structured Prompt Schema** | Execute idle-GPU benchmark matrix (B1–B12) changing one variable at a time (≥ 3 trials); calibrate RAM/swap/thermal/storage thresholds; fix FLUX.2 Qwen3 CPU offload; benchmark untiled vs tiled VAE, FP8 `comfy-aimdo` vs GGUF, and isolated SageAttention; define canonical **Structured Prompt Schema** (`PromptSpec`). |
| **Sprint 3** | **Central ResourcePlanner, Prompt Compilers & Guided Mode** | Build internal `avatar_v2/resource_planner.py` (`P0`–`P5` profiles + hysteresis); build model-specific prompt compilers (FLUX.2 natural language vs Wan 2.2 motion beats); integrate **Guided** mode with attribute locking, scene recipes, and controlled randomization inside the existing UI. |
| **Sprint 4** | **Script Mode, AST Inspector, Bubblewrap Isolation & Native SDK** | Build `script_inspector.py` (static AST only), `script_runner.py` with Linux `bwrap` (`--unshare-all`, `--unshare-net`, `--clearenv`, `CUDA_VISIBLE_DEVICES=""`, `prlimit`, UDS IPC), bounded continuous-generation controller (default 10 items, Start/Pause/Resume/Stop, host-enforced disk/RAM/thermal caps), Native `avatar_v2_sdk`, minimal Script mode UI, and sandbox security tests. |
| **Sprint 5** | **Optional A1111 Compatibility Adapter & Multi-Reference Editing** | Implement opt-in, local-only A1111 Compatibility Adapter (documented mappings, explicit rejection of unsupported params, mandatory `policy.py` & `RenderQueue` routing), Static Structured Prompt Extractor, migration guide for `continuous_generator_juggernaut.py`, and FLUX.2 klein Multi-Reference Editing (0 GB extra disk). |
| **Sprint 6** | **Local Offline Prompt Intelligence & Intelligent Workflow Routing** | Wire offline Ollama prompt interpretation (`qwen3:8b` / `qwen3.5:9b`, CPU `num_gpu=0` or sequential `keep_alive=0`) to compile natural-language requests into `PromptSpec` and route automatically to the highest-quality feasible local model/workflow. |
| **Sprint 7** | **Local Image & Video Quality Inspection** | Extend `candidate_qa.py` and `realism_qa.py` for automated local VLM (`qwen3-vl:8b-instruct`) + structural/temporal video QA executed strictly without competing with ComfyUI for VRAM. |
| **Sprint 8** | **Automated Corrective Generation & Bounded Regeneration** | Implement closed-loop, strictly bounded corrective regeneration (hard retry ceiling, prompt/seed/parameter adjustment based on QA diagnostics, zero unbounded loops, zero cloud spend without consent). |
| **Sprint 9** | **Larger Models, Higher Precision & Specialized Model Expansion** | After disk reclamation and benchmark proof: integrate higher-precision FLUX.2 9B Distilled (Q6_K/FP8), Wan 2.2 A14B FP8 Scaled (`comfy-aimdo`), Qwen-Image-Edit 20B (Apache 2.0), and Wan 2.2 Fun-Control 14B (Apache 2.0). |
| **Sprint 10** | **Integrated Functional, Performance, Security & Offline Acceptance** | Full end-to-end offline acceptance test suite (air-gapped network verification, security/sandbox escape resistance, multi-hour stability/leak test, UI self-test, PyInstaller bundle verification, rollback validation). |

---

## Graduated Protection Thresholds (Provisional — Calibrated in Sprint 2)

| Resource | Normal | Warning | Intervention / Restricted | Emergency Stop / Pause |
|---|---|---|---|---|
| **Available Disk Space** | `>= 30 GB` | `< 30 GB` | `< 20 GB` (Restricted) | `< 15 GB` (Hard stop + predictive check before every job) |
| **Available Physical RAM** | `>= 20%` | `< 20%` | `< 10%` | `< 5%` |
| **Swap Utilization** *(gated with swap-in/out & Linux PSI)* | `<= 50%` | `> 50%` | `> 75%` | `> 90%` (with active swap/PSI pressure) |
| **GPU Temperature** *(with recovery hysteresis)* | `< 75°C` | `>= 75°C` | `>= 80°C` | `>= 84°C` (provisional emergency pause; clears `< 78°C`) |

---

## Sprint Boundary Reporting Standard

Every sprint closes with a documented report recording: completed implementation, exact files changed, tests executed and results, before-and-after measurements, remaining limitations, identified risks, rollback procedure, git branch/commit, and verified remote preservation where available.
