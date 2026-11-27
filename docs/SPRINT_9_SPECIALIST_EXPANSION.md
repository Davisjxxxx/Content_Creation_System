# Avatar V2 — Sprint 9 Specialized Model Expansion & Empirical Evaluation

**Status:** Software-verified & live-GPU benchmarked (**406 passing regression tests**, `JS syntax OK`, zero unauthorized downloads, zero OOM)  
**Hardware:** RTX 4070 Laptop GPU (`8,188 MiB` VRAM), Intel Core i9-13980HX (24C/32T), `62 GiB` RAM  
**Offline & Preservation Guarantee:** 100% local execution, zero assets deleted, `dist/AvatarV2` and rollback packages untouched, immutable Sprint 5–8 benchmark outputs preserved.

---

## 1. Reconciliation of Sprint 5–8 Evidence Inconsistencies (Section 1)

1. **`adh_04_spatial` Original Requirement Reconciled:**
   - Authoritative Sprint 2/5 benchmark definition ([adherence_benchmark.py](file:///home/jd/avatar_project/avatar_v2/adherence_benchmark.py#L292-L300)) and immutable `phase_a2_adherence_results.json` specify:
     - **Original prompt:** *"Interior studio shot: a green glass vase sits to the left of a brass lamp on a wooden table, while a tall potted fern stands behind the table near a white curtain."*
     - **Mandatory criterion `vase_left_of_lamp`:** *"Green glass vase is positioned to the left of the brass lamp"*
     - **Mandatory criterion `fern_behind_table`:** *"Potted fern stands behind the table near a white curtain"*
   - **Resolution:** **`vase-left / lamp-right`** IS the original user requirement. The earlier Sprint 6 summary note (*"brass lamp left, green vase right"*) described what FLUX.2 9B *erroneously rendered* in `adh_04_spatial_00001_.png` (a real `1/2` baseline failure). Therefore, Sprint 8's `s8_adh04_E1_edit_00001_.png` genuinely corrected `adh_04_spatial` against the **original** acceptance criteria (`1/2 -> 2/2`).
2. **Production Configuration & Harness Audit (Sprint 5 vs Sprint 8):**
   - **Production FLUX.2 9B Base (`offline_builder.py`):** `28 steps, CFG 3.5, 896×1280, Q4_K_M + Qwen3-8B-Q4_K_M + flux2-vae`. An early Sprint 8 run erroneously used 4B's `8 steps, CFG 1.2` and was discarded after visual review caught under-convergence; all final Sprint 8 and Sprint 9 FLUX.2 9B renders ran at identical `28 steps, CFG 3.5`.
   - **Harness vs Production `CorrectionCoordinator` distinction:** In Sprint 8 `scripts/eval_sprint8_corrections.py`, `adh_03_quantities` Condition D/E and `mv_01_rear_hair` executed through the **actual production `CorrectionCoordinator`** (and during the resumed `adh_03` pass, `ResourcePlanner`'s live swap-thrashing guard fired and safely blocked the attempt). By contrast, `adh_04_spatial` Conditions E1/E2 ran directly through the isolated ComfyUI harness (bypassing `CorrectionCoordinator`'s VLM gate because Sprint 8 single-pass `qwen3-vl:8b` falsely accepted `adh_04_spatial_00001_.png`).

---

## 2. Read-Only Storage Inventory & Acquisition Gate (Section 4)

No existing files were deleted and no unauthorized model downloads were initiated.

| Storage Category | Path | Measured Size | Notes / Cleanup Candidates (Read-Only) |
|---|---|---:|---|
| Primary NVMe (`/`) | `/dev/nvme0n1p2` (915G ext4) | **827G used / 43G free (96%)** | Only 13 GB above 30 GB Normal tier; 28 GB above 15 GB Emergency Stop |
| Secondary NVMe | `/dev/nvme1n1` (953.9G WD SN560) | 926.2G NTFS (unmounted) | Potential future cold storage if mounted/partitioned by owner |
| ComfyUI Generation Models | `/home/jd/AvatarForge/comfyui/models/` | **47.1 GB** | Diffusion `29G`, Text Encoders `14G`, LoRAs `2.3G`, VAE `1.8G` (0 duplicates) |
| Ollama Model Store | `/usr/share/ollama/.ollama/models/blobs` | **78.0 GB** | 19 models (`mythomax-local` 7.9G, `mythomax-l2` 7.9G, `qwen3.5:9b` 6.6G, etc.) |
| HuggingFace Cache | `~/.cache/huggingface/hub/` | **24.0 GB** | Includes cached `yolov8x-worldv2.pt` (140M) & `adetailer` face/hand/person-seg (63M) |
| Application Bundles | `dist/AvatarV2*` | **3.8 GB** | Current bundle + 10 timestamped rollback copies (`~346 MB` each; untouched) |
| Avatar V2 Home | `~/.avatar_v2/` | **2.6 GB** | `updates/` `1.5G`, `runs/` `845M`, `test_assets/` `188M`, `benchmarks/` `70M` |
| Pip Cache | `~/.cache/pip` | **997 MB** | Standard wheel cache |

All acquisition checks are enforced programmatically via `SpecialistRegistry.evaluate_acquisition()` and `ResourcePlanner.plan_specialist_execution()` ([specialist_registry.py](file:///home/jd/avatar_project/avatar_v2/specialist_registry.py), [resource_planner.py](file:///home/jd/avatar_project/avatar_v2/resource_planner.py)).

---

## 3. Specialist Localization, Masked Editing & Video Benchmarks

### 3.1 Blind 2D Grid Localization (`specialist_vision.py` — ZERO Downloads)

To solve Sprint 7/8's photographic counting and left/right spatial weaknesses without downloading unapproved weights, `run_blind_object_localization()` overlays subtle `x=0%/25%/50%/75%/100%` coordinate lines and runs a **non-leading** object inventory pass (omitting target counts and left/right hints) before deterministic Python reconciliation (`reconcile_criterion_with_detections()`).

Live benchmark across **6 real photographic cases (10 criteria)** (`~/.avatar_v2/benchmarks/sprint9_specialists/sprint9_blind_grid_results.json`, `100%` verified VRAM release):

| Photographic Case | Criterion | Ground Truth | Sprint 8 Single-Pass VLM | Sprint 9 Blind 2D Grid | Observed 2D Evidence |
|---|---|---|---|---|---|
| `adh_03_quantities` (orig) | `three_mugs` | `failed` | `passed` (FP) | **`failed` (TN)** | Counted 2 handled mugs (`x=50%, 87.5%`), excluded 2 bowls |
| `adh_03_quantities` (orig) | `two_closed_notebooks` | `failed` | `passed` (FP) | **`uncertain`** | Inconclusive open/closed state -> **0% false accept!** |
| `s8_adh03_E2_regen` | `three_mugs` | `passed` | `passed` (TP) | **`passed` (TP)** | Counted 3 handled mugs (`x=25%, 50%, 75%`) |
| `s8_adh03_E2_regen` | `two_closed_notebooks` | `passed` | `passed` (TP) | **`uncertain`** | State kept conservative for human review |
| `s8_adh03_condB_det` | `three_mugs` | `failed` | `failed` (TN) | **`failed` (TN)** | Counted >3 mugs vs required 3 |
| `s8_adh03_condB_det` | `two_closed_notebooks` | `failed` | `failed` (TN) | **`failed` (TN)** | Counted <2 notebooks |
| `s8_adh03_condC_llm` | `three_mugs` | `failed` | `failed` (TN) | **`failed` (TN)** | Counted 2 mugs vs required 3 |
| `s8_adh03_condC_llm` | `two_closed_notebooks` | `passed` | `passed` (TP) | **`uncertain`** | State kept conservative for human review |
| `adh_04_spatial` (orig) | `vase_left_of_lamp` | `failed` | `passed` (FP) | **`failed` (TN)** | Green vase `x=33.3%` vs brass lamp `x=25.0%` (contradicts left!) |
| `s8_adh04_E1_edit` | `vase_left_of_lamp` | `passed` | `uncertain` | **`passed` (TP)** | Green vase `x=25.0%` vs brass lamp `x=45.0%` (satisfies left!) |

- **Totals:** **`TP=2, TN=5, FP=0 (0.0% FAR), FN=0 (0.0% FRR), Uncertain=3`** — eliminating all 3 photographic false-acceptance failures from Sprint 8 while conservatively marking ambiguous open/closed notebook states `uncertain` (`2d_image_coordinates` explicitly distinguished from 3D depth).

### 3.2 Genuine Masked Localized Editing (`builtin:flux2_masked_edit` — ZERO Downloads)

Implemented `build_flux2_masked_edit_workflow()` in [workflows_builder.py](file:///home/jd/avatar_project/avatar_v2/workflows_builder.py) wiring ComfyUI 0.46 core nodes (`SetLatentNoiseMask` + `DifferentialDiffusion` + `ReferenceLatent` + `ImageToMask` + `ImageCompositeMasked`) with pure-Pillow feathered masks (`generate_region_mask()`).

Live RTX 4070 benchmark (`28 steps, CFG 3.5`, `896×1280`, visually verified):

| Trial | Workflow | Latency | Peak VRAM | Pixels Changed Outside Mask | Visual Inspection Outcome |
|---|---|---:|---:|---:|---|
| `s9_masked_adh04_blue_vase` | **`builtin:flux2_masked_edit`** | 437.9 s | **5,823 MiB** | **`0.0%`** (mean diff `0.000`) | **Success:** Green vase -> deep cobalt blue glass with realistic reflections; lamp, table, fern & curtain pixel-identical! |
| `s8_adh04_E1_edit` (reference) | Unmasked `ReferenceLatent` edit | 417.4 s | 6,327 MiB | `57.4%` | Good global preservation, but non-zero pixel drift across entire canvas |
| `s9_masked_adh03_bowl_to_mug` | **`builtin:flux2_masked_edit`** | 431.3 s | **5,915 MiB** | **`0.0%`** (mean diff `0.000`) | **Success:** Left handleless bowl -> white ceramic coffee mug with curved left handle & dark brewed coffee; rest of desk pixel-identical! |

### 3.3 Live Wan 2.2 A14B Video Quality & Motion Comparison (Section 9 — ZERO Downloads)

Identical starter image (`seed_front_00001_.png`), identical wave-hello prompt, `49 frames @ 480×832`, seed `92001`, tiled VAE (`~/.avatar_v2/benchmarks/sprint9_specialists/sprint9_wan_video_results.json`):

| Configuration | Steps / CFG / LoRA | Latency | Peak VRAM | Motion Energy | Max Luma Jump | Max dHash Jump | Sharpness | Visual Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`wan14b_lightning_4step_cfg1` (Default)** | 4 / 1.0 / Lightning | **225.3 s** | **6,166 MiB** | 8.88 | 1.91 | 9 | **186.3** | **Best overall:** crisp 5-finger right-hand wave, warm smile, clean hand lower at frame 48, `0` defects |
| `wan14b_lightning_6step_cfg1` | 6 / 1.0 / Lightning | 295.3 s (`+31%`) | 6,604 MiB | 8.40 | 2.13 | 8 | 183.5 | Virtually identical to 4-step; extra 70 s not justified |
| `wan14b_base_10step_cfg3_5` | 10 / 3.5 / None (Base) | 703.5 s (`3.12×`) | 6,228 MiB | 8.03 | **1.41** | **7** | 174.5 | Slightly smoother temporal step (`1.41`/`7`), slightly softer (`174.5`), slight midpoint eye squint; `3.12×` slower |

- **Empirical decision:** Retain **Wan 2.2 A14B Lightning 4-step (`CFG 1.0`)** as default; keep `without_wan_lightning()` strictly optional for cases requiring explicit `CFG > 1.0` negative prompting.

---

## 4. Specialist & Larger Model Research Matrix (Sections 5, 6, 8, 10, 11)

| Candidate | Role | Official Source & License | Storage | Min VRAM | Classification | Decision |
|---|---|---|---:|---:|---|---|
| `flux2_masked_latent_composite` | Masked editing | ComfyUI 0.46 core + installed FLUX.2 9B | 0 MB | 6.2 GB | **`already_available`** | **Enabled** (live-proven `0.0%` outside-mask drift) |
| `pillow_bbox_mask_generator` | Region masks | Built-in Pillow 11.3 (MIT/HPND) | 0 MB | 0 MB | **`already_available`** | **Enabled** (`<20 ms` CPU) |
| `qwen3_vl_8b_blind_grid` | 2D localization | Installed Ollama `qwen3-vl:8b-instruct` (Apache-2.0) | 0 MB | 5.8 GB | **`already_available`** | **Available for selective count/spatial corroboration** (`0%` FAR on 6 photo cases) |
| `wan22_a14b_base_cfg` | Video CFG>1 | Installed Wan 2.2 A14B GGUF (Apache-2.0) | 0 MB | 7.6 GB | **`already_available`** | **Optional/experimental** (`3.12×` slower than 4-step Lightning) |
| `cached_yolo_world_v2` / `adetailer` | Detection / seg | Cached `~/.cache/huggingface` (`140 + 63 MB`, GPL-3.0 runtime) | 203 MB (cached) | 1.2 GB | **`ready_for_isolated_evaluation`** | Deferred pending `ultralytics` dependency approval |
| `grounding_dino_tiny` | Open-vocab det | `IDEA-Research/grounding-dino-tiny` (Apache-2.0) | 694 MB | 1.4 GB | **`requires_limited_acquisition`** | Top deferred candidate (native `transformers` already installed in AvatarForge venv) |
| `florence_2_base_ft` | Det + OCR | `microsoft/Florence-2-base-ft` (MIT) | 460 MB | 1.2 GB | **`requires_limited_acquisition`** | Deferred behind storage/approval gate |
| `sam2_1_hiera_tiny` | Fine masks | `facebook/sam2.1-hiera-tiny` (Apache-2.0) | 156 MB | 1.1 GB | **`requires_limited_acquisition`** | Deferred (Pillow feathered masks sufficient for coarse edits) |
| `depth_anything_v2_small` | Relative depth | `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0; Base+ CC-BY-NC) | 99 MB | 0.6 GB | **`requires_limited_acquisition`** | Deferred until depth-conditioned generation weights are installed |
| `dwpose_onnx` | Pose / hands | `yzd-v/DWPose` (Apache-2.0) | 210 MB | 0.7 GB | **`requires_limited_acquisition`** | Deferred (requires `onnxruntime`) |
| `realesrgan_x2plus` | 2× upscaling | `xinntao/Real-ESRGAN` (BSD-3-Clause) | 67 MB | 0.9 GB | **`requires_limited_acquisition`** | Opt-in only; never alters identity or recovers factual detail |
| `wan22_fun_control_5b` | Video control | `alibaba-pai/Wan2.2-Fun-5B-Control` (Apache-2.0) | ~5–9.8 GB | 7.5 GB | **`requires_significant_acquisition_approval`** | Deferred until disk space freed above Normal tier |
| `qwen_image_edit_gguf_q4` | Image editing | `Qwen/Qwen-Image-Edit` (Apache-2.0) | ~18.5 GB | 12+ GB (offload on 8 GB) | **`requires_significant_acquisition_approval`** | Deferred (would breach 30 GB Warning storage tier) |
| `flux2_dev_32b_gguf_q4` | Larger still gen | `black-forest-labs/FLUX.2-dev` (Non-Commercial) | ~35+ GB | 16+ GB | **`operationally_impractical`** | Rejected (breaches 15 GB Emergency reserve + PCIe bottleneck) |
| `sdxl_controlnet_pack` | Pose/Depth | Legacy SDXL | 2.5 GB | 6 GB | **`incompatible`** | Explicitly blocked by `SpecialistRegistry.check_compatibility()` |
