# Sprint 5 & Phase A/B Closure — Live GPU Benchmarks, Multi-View Subject References & Optional A1111 Adapter

**Date:** 2026-11-25  
**Target Workstation:** NVIDIA GeForce RTX 4070 Laptop GPU (`8,188 MiB` VRAM, `sm_89`) · Intel Core i9-13980HX · 62 GiB RAM · ComfyUI `0.46.0` + PyTorch `2.14.1+cu130` + `comfy-aimdo 0.6.0` + `ComfyUI-GGUF`

---

## 1. Phase A — Live Controlled GPU Benchmark Results (Closed)

All live GPU trials executed against an isolated ComfyUI instance (`127.0.0.1:8199`, isolated output directory) without modifying `~/.avatar_v2/desktop.json`, `dist/AvatarV2`, production history, or `/home/jd/AvatarForge/venv`.

### A1. Memory, Retention, `--cache-ram 6 24`, Tiled VAE, Wan Component Reuse & Attention (`13 / 13` Succeeded, `0` OOM, `0 MB` ComfyUI `VmSwap`)

Raw data: `~/.avatar_v2/benchmarks/phase_a1_live/phase_a1_results.json`

| Trial | Configuration | Elapsed (s) | Peak VRAM (MB) | Post VRAM (MB) | Peak RSS (MB) | Comfy VmSwap | Peak Temp |
|---|---|---:|---:|---:|---:|---:|---:|
| `B1_flux2_9b_cold_untiled` | Baseline (no `--cache-ram`), cold FLUX.2 9B `896×1280` | `242.71` | `7,261` | `5,367` | `14,000` | `0 MB` | `79°C` |
| `B2_flux2_9b_after_85pct_evict` | Baseline + old `/free` eviction between stills | `242.89` | `7,548` | `5,351` | `14,482` | `0 MB` | `73°C` |
| `B3_flux2_9b_warm_retained_1` | Sprint 3 warm retention (`respect_retention=True`) | **`223.67`** | `7,721` | `5,305` | **`10,280`** | `0 MB` | `75°C` |
| `B4_flux2_9b_warm_retained_2` | Sprint 3 warm retention (repeatability trial) | **`222.76`** | `7,751` | `5,530` | **`10,109`** | `0 MB` | `76°C` |
| `B5_flux2_9b_warm_tiled_vae` | Warm FLUX.2 9B + `VAEDecodeTiled` (`512/64`) | `225.57` | `7,760` | `5,850` | `9,900` | `0 MB` | `74°C` |
| `O1_flux2_9b_cold_cacheram` | Sprint 1 `--cache-ram 6 24`, cold FLUX.2 9B | `240.83` | `7,082` | `5,502` | `13,370` | `0 MB` | `72°C` |
| `O2_flux2_9b_warm_cacheram` | Sprint 1 `--cache-ram 6 24` + Sprint 3 retention | **`221.49`** | `7,619` | `5,504` | **`9,415`** | `0 MB` | `72°C` |
| `O3_wan22_14b_from_flux_prefree` | Cross-family FLUX → Wan 2.2 14B (`/free`, `480×832`, 81f) | `316.12` | `7,742` | `1,295` | `35,229` | `0 MB` | `74°C` |
| `O4_wan22_5b_shared_umt5_tiled` | Intra-family Wan 14B → 5B (no `/free`, shared `umt5_xxl` + tiled VAE) | **`246.69`** | **`7,074`** | `1,871` | `34,606` | `0 MB` | `73°C` |
| `O5_wan22_5b_cold_after_free_untiled` | Wan 2.2 5B after full `/free`, untiled VAE | `284.05` | `7,730` | `1,075` | `20,351` | `0 MB` | `73°C` |
| `A1_flux2_9b_cold_sdpa` | `--cache-ram 6 24 --use-pytorch-cross-attention` (cold) | `250.23` | `7,316` | `5,331` | `13,947` | `0 MB` | `71°C` |
| `A2_flux2_9b_warm_sdpa` | `--cache-ram 6 24 --use-pytorch-cross-attention` (warm) | `219.37` | `7,659` | `5,551` | `9,369` | `0 MB` | `72°C` |
| `cancel_recovery_followup_flux2_9b` | `/interrupt` mid-render + `/free` + follow-up FLUX.2 9B | `244.36` | `7,352` | `5,267` | `15,079` | `0 MB` | `74°C` |

**Verified Attribution (One Variable at a Time):**
1. **Sprint 3 Warm Retention (`B3/B4` vs `B2`):** Skipping fixed-85% `/free` eviction between consecutive FLUX.2 9B stills saved **`19.2–20.1s` (`-8.1%`)** and cut peak host RSS by **`4.2–4.4 GB`** (`10.1–10.3 GB` vs `14.5 GB`) with zero OOM even though active denoising reached `94.7%` VRAM (`7,751 MB`).
2. **Sprint 1 `--cache-ram 6 24` (`O1/O2` vs `B1/B3/B4`):** Further reduced cold RSS by `630 MB` and warm RSS by `700–865 MB` (`9,415 MB` warm peak — **`-5.07 GB` vs `B2`**) and shaved another `~1.9s` (`221.49s` warm).
3. **Component-Aware Wan 2.2 Reuse (`O4` vs `O5`) — Confounder Note:** Reusing `umt5_xxl` (6.3 GB) across Wan 14B → 5B (`comfy_native_selective`) together with `VAEDecodeTiled` on Wan 5B's 48-ch VAE cut latency by **`37.36s` (`-13.2%`)** AND cut peak VRAM by **`656 MB`** (`7,074 MB` vs `7,730 MB`). **Confounder:** `O4` changed two variables simultaneously relative to `O5` (shared `umt5_xxl` retention AND `VAEDecodeTiled` vs untiled VAE), so the `-37.36s` latency win and `-656 MB` VRAM win reflect the **combined** effect rather than component retention alone; isolating `VAEDecodeTiled` vs untiled on cold Wan 5B remains a clean follow-up A/B if finer attribution is needed.
4. **Tiled vs Untiled VAE (`B5` vs `B3/B4` vs `O4/O5`):** On warm `896×1280` FLUX.2 9B stills, `VAEDecodeTiled` added `+2.4s` with no VRAM reduction — confirming `ResourcePlanner`'s rule to keep standard `VAEDecode` on warm `<=1024×1280` FLUX.2 stills while enabling `VAEDecodeTiled` on Wan 2.2 5B 48-ch VAE (`-656 MB` VRAM) and high-res stills (`>1024×1280`).
5. **Attention Compatibility:** Default PyTorch 2.14 SDPA is within `~1%` of explicit `--use-pytorch-cross-attention`; `host nvcc 12.1 != torch CUDA 13.0` blocks SageAttention 2 source builds so production `/home/jd/AvatarForge/venv` remains untouched.
6. **Live Threshold Calibration Fix (`avatar_v2/resource_monitor.py`):** Live concurrency exposed sub-second `swap_rate` amplification (`dt=0.001s` turning a single background page fault into a false-positive `>90%` swap emergency). Floored `dt >= 1.0s` so only sustained active paging triggers swap-pressure escalation.

---

## 2. Phase B — Isolated Desktop Verification & Script Isolation Hardening (Closed)

- **Live Desktop WebKitGTK Verification (`AVATAR_V2_UITEST=1` against isolated temp `APP_HOME`):**
  - Raw report: `~/.avatar_v2/benchmarks/phase_b_desktop_uitest.json`
  - Verified all 4 views (`make`, `forge`, `library`, `settings`), all 7 recipes, Automatic/Guided/Script mode switching, `#forgeGuided` opening, `#forgeSubjectRefs` role population (`>=8` roles), `character_profiles` CRUD bridge, `#forgeScriptPanel` + Bubblewrap sandbox status, `Browse local files…` controls, missing media refusal, media viewer keyboard controls, readable dropdowns, and **`js_errors: []`**.
- **Additional Controlled Script Isolation Hardening (`avatar_v2/script_runner.py`, `tests/test_sprint5_multiview_and_adapter.py`):**
  - Content-hash pinning + immutable `0o400` script snapshot bind-mount prevents TOCTOU post-validation script mutation (`code="script_mutated"`).
  - Verified broker token & contract version rejection, reference path traversal rejection, 12-thread concurrent burst flood capping (`max_concurrent=1`), and live Bubblewrap process-group teardown of spawned child subprocesses.

---

## 3. Phases C–K — Multi-View Subject References, FLUX.2 Multi-Ref, Wan FLF2V & A1111 Adapter

### Live Phase F (FLUX.2 9B Multi-Reference `ReferenceLatent`) & Phase G (Wan 2.2 `WanFirstLastFrameToVideo`) Results (`10 / 10` Succeeded, `0` OOM)

Raw data & outputs: `~/.avatar_v2/benchmarks/phase_a2_multiview_adherence/phase_a2_multiview_results.json`

| Trial | Condition / Refs | Elapsed (s) | Peak VRAM (MB) | Peak RSS (MB) | Temp | Firsthand Visual Finding (Human Inspection Required for Formal Scoring) |
|---|---|---:|---:|---:|---:|---|
| `seed_ref_front` | `0` refs (txt2img) | `226.56` | `7,338` | `12,445` | `78°C` | Clean studio front reference (olive linen blazer, cream blouse, navy trousers, low chestnut bun) |
| `mv_01_starter_only` | `1` ref (front only, rear request) | `400.62` | `6,158` | `13,351` | `72°C` | Rear blazer/trousers matched, **but invented loose bob hair** (lost low hair bun) |
| `mv_02_starter_plus_front` | `1` ref (new bookstore env) | `400.61` | `6,154` | `14,077` | `72°C` | Identity, proportions, and wardrobe cleanly placed into sunlit bookstore |
| `mv_03_starter_front_back` | `2` refs (front + back, rear gallery) | `606.38` | `5,108` | `13,834` | `72°C` | **Eliminated `mv_01` invention:** exact low hair bun, back blazer seam & cuff buttons |
| `mv_04_multi_perspective` | `4` refs (face + front + left + back) | `1,097.34` | `5,363` | `14,228` | `72°C` | Zero OOM on 8 GB; crisp walking left side profile matching silhouette, hair bun & blazer details |
| `mv_05_conditioned_keyframes` | `3` refs (rear 3/4 keyframe) | `845.18` | `4,663` | `14,267` | `73°C` | Exact studio lighting, low bun, and rear 3/4 blazer continuity |
| `mv_06_wan_flf2v_turn` | Wan 2.2 14B `WanFirstLastFrameToVideo` | **`313.29`** | `7,744` | `34,236` | `73°C` | **Works natively on installed 14B Lightning weights:** smooth front→back turn preserving hair bun & blazer; pairing matched-environment start/end keyframes avoids background wipe |

**Empirical Phase F Finding & Adaptive Optimization:**
- Because FLUX.2 concatenates reference latent tokens into self-attention, chaining `1.0 MP` references scales latency steeply (`0 refs = 227s`, `1 ref = 401s`, `2 refs = 606s`, `3 refs = 845s`, `4 refs = 1,097s`).
- `build_flux2_txt2img_workflow` in `avatar_v2/workflows_builder.py` now applies an adaptive pixel budget (`1.0 MP` for 1 ref, `0.65 MP` for 2 refs, `0.5 MP` for 3–4 refs, overrideable via `REF_MEGAPIXELS`) to keep multi-view conditioning responsive and OOM-safe on 8 GB VRAM.

**Verified Phase G Capabilities & Documented Limitations:**
- Installed Wan 2.2 weights (`Wan2.2-I2V-A14B` Q4_K_S + `Wan2_2-TI2V-5B-Turbo` Q5_K_M) do **not** accept arbitrary multi-view subject reference images directly, and `Wan22FunControl` / `WanVace` weights are not installed on disk (`44 GB` free disk — no large downloads performed).
- Verified supported multi-view video strategies: (1) FLUX.2 multi-reference conditioned starter image → Wan 2.2 I2V, and (2) FLUX.2 multi-reference conditioned start + end keyframes → `WanFirstLastFrameToVideo` (`with_wan_end_frame` in `workflows_builder.py` & `providers/comfyui.py`).
- `select_relevant_references()` in `avatar_v2/subject_references.py` never silently discards rear/side references when a workflow cannot directly consume them; it surfaces explicit strategy recommendations and limitations.

### Optional A1111 Compatibility Adapter (`avatar_v2/a1111_adapter.py`)
- Opt-in, local-only subset: `POST /sdapi/v1/txt2img`, `POST /sdapi/v1/img2img`, `GET /sdapi/v1/sd-models`, `GET /sdapi/v1/progress`.
- Routes strictly through `StructuredPromptSpec`, `policy.depicts_minor`, `ModelRegistry`, `ResourcePlanner`, and `GenerationBroker` (default `max_items=10`, local-only).
- Explicitly rejects unsupported A1111 features (`enable_hr`, `alwayson_scripts`, `refiner_checkpoint`, `restore_faces`, `tiling`, unknown SDXL checkpoints like `juggernautXL_v9.safetensors`) with HTTP `400` rather than silently ignoring or substituting incompatible models.

---

## 4. 13-Category Prompt Adherence Baseline (`phase_a2_adherence_results.json`)

Status: Executing live on isolated ComfyUI `:8199` (`task-641`) across all 10 FLUX.2 9B cases + 3 Wan 2.2 14B cases; objective requirement checklists and subjective `1–5` visual quality scores are preserved unfilled (`null`) for honest human evaluation (never fabricated).
