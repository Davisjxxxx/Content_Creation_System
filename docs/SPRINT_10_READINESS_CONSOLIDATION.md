# Avatar V2 — Sprint 10 Readiness & Final Independent Validation Consolidation Dossier

**Prepared at:** Sprint 9 Completion (Branch `sprint-1-gpu-memory-foundation`)  
**Target Release Candidate Validation:** Sprint 10 Integrated Functional, Performance, Security & Offline Acceptance Testing (followed by one consolidated independent Astra review after Sprint 10 completes).

---

## 1. Consolidated Original Requirements & Mode Architecture

1. **Three Supported Interaction Modes (Approved Frontend Preserved):**
   - **Automatic:** Natural-language prompt entry with local offline intent interpretation (`Qwen3 8B` / `Llama 3.2 3B` / deterministic fallback) and deterministic workflow routing.
   - **Guided:** Optional collapsed structured attribute selection backed by `StructuredPromptSpec` v1.0 and data-driven prompt libraries (`avatar_v2/data/prompt_libraries.json`).
   - **Script:** Local `.py` script inspection (Static AST Inspector), structured prompt extraction, Native Python SDK (`AvatarV2Client`), Unix-socket `GenerationBroker`, and Linux Bubblewrap + `prlimit` sandboxed execution (no auto-execute on selection, TOCTOU SHA-256 hash pinning, `0o400` read-only snapshot mount).
2. **Quality-First Offline Local Execution:**
   - Default still model: **FLUX.2 Klein 9B Base** (`28 steps, CFG 3.5`).
   - Default video model: **Wan 2.2 I2V A14B + Lightning** (`4 steps, CFG 1.0`, tiled VAE).
   - Default local intelligence: Loopback-only Ollama (`127.0.0.1:11434`), `qwen3:8b` / `llama3.2:3b` for prompt intelligence, `qwen3-vl:8b-instruct` for visual inspection.
   - Zero automatic cloud escalation (`allow_cloud_escalation=False`, Runway strictly opt-in with explicit host authorization and spending caps).

---

## 2. Sprint 1–9 Implementation History & Verified Commits

| Sprint | Commit | Key Delivered Capabilities | Verification Milestone |
|---|---|---|---|
| **Sprint 1** | `bf40001` | GPU memory foundation, `/free` keep-loaded fix, bounded `--cache-ram 6 24`, `ResourceMonitor`, `VAEDecodeTiled`, attention compatibility evaluator | 345 passing tests |
| **Sprint 2** | `c065552` | `StructuredPromptSpec` v1.0, 13-category `adherence_benchmark.py`, `/proc` RAM/swap forensics, component residency analysis | 351 passing tests |
| **Sprint 3** | `366c6a4` | Component-aware `ModelRegistry`, central `ResourcePlanner`, FLUX.2 & Wan 2.2 prompt compilers, data-driven prompt library, collapsed Guided mode UI | 360+ passing tests |
| **Sprint 4** | `c9c3cfe` | Shared local reference picker (My Stuff + native Browse Local Files, SHA-256 staging), Static AST Inspector, Native SDK, `GenerationBroker`, live-verified Bubblewrap sandbox, Script mode UI | 370 passing tests + live bwrap verification |
| **Sprint 5** | `e660b1d` | Multi-view Subject Reference Sets (9 roles, multi-subject separation, conflict detection, deterministic orientation selector), `CharacterProfileStore`, chained FLUX.2 `ReferenceLatent`, `with_wan_end_frame()`, optional A1111 adapter, Phase A1/A2 live GPU benchmarks & Phase B desktop WebKitGTK UI test | 379 passing tests + 23 live GPU trials (`0` OOM) + `0` JS errors |
| **Sprint 6** | `539a7ef`, `b512a74` | 13-category immutable adherence baseline closure (`36/36` live GPU trials passed), 4-model local Ollama benchmark (8 CPU/GPU runs), `LocalOllamaClient`, `intent_interpreter.py`, `WorkflowRouter`, `GenerationVerificationRecord` | 384 passing tests + 36 cumulative live GPU trials (`0` OOM) |
| **Sprint 7** | `5955e42` | Hybrid `VisualInspector` & `VideoInspector`, verified async Ollama VRAM release (`verify_ollama_vram_released`), rear-view guardrail, image prompt-injection defense, Sprint 8 `CorrectionRecommendation` contract, non-intrusive viewer details | 395 passing tests + 114 live VLM inspections + 4 video CPU temporal benchmarks |
| **Sprint 8** | `0878b2e` | `CorrectionCoordinator` (max 2 image / max 1 video retry, 7-factor best-candidate ranking, regression rejection, cancellation, ResourcePlanner & storage guards), verification eligibility matrix, held-out unseen VLM eval (`18/18`, `0%` FAR), live Conditions A–E GPU comparison | 401 passing tests + 13 live held-out VLM runs + 6 live 28-step FLUX.2 9B trials |
| **Sprint 9** | *(current)* | Read-only storage & larger-model audit, `SpecialistRegistry` + acquisition gate, zero-download Blind 2D Grid Localization (`0%` FAR on `adh_03`/`adh_04` photographic cases), genuine FLUX.2 9B masked localized editing (`SetLatentNoiseMask` + `DifferentialDiffusion` + `ImageCompositeMasked`, **`0.0%` outside-mask pixel drift**), live Wan 2.2 A14B Lightning vs Base comparison, Sprint 10 consolidation | **406 passing tests** + live blind-grid, masked-edit & Wan video benchmarks |

---

## 3. Verified Workstation Hardware, Storage & Model Inventory

- **Hardware:** NVIDIA RTX 4070 Laptop GPU (`8,188 MiB` VRAM, `sm_89`), Intel Core i9-13980HX (24C/32T), `62 GiB` physical RAM.
- **Primary NVMe (`nvme0n1p2`, ext4, 915 GB):** `827 GB` used, **`43 GB` free (96% used)** — `13 GB` above the `30 GB` Normal tier and `28 GB` above the `15 GB` Emergency Stop reserve.
- **Secondary NVMe (`nvme1n1`, WD PC SN560, 953.9 GB):** Present with `926.2 GB` NTFS partition (unmounted in Linux) and `26 GB` `/media/jd/RESTORE`.
- **Installed Generation Models (`/home/jd/AvatarForge/comfyui/models/`, `47.1 GB`, zero duplicates):**
  - Diffusion (`29 GB`): `flux-2-klein-4b-Q5_K_M.gguf` (2.9G), `flux-2-klein-base-9b-Q4_K_M.gguf` (5.6G), `Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf` (8.2G), `Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf` (8.2G), `Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf` (3.6G).
  - Text Encoders (`14 GB`): `Qwen3-4B-Q5_K_M.gguf` (2.7G), `Qwen3-8B-Q4_K_M.gguf` (4.7G), `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (6.3G).
  - LoRAs (`2.3 GB`): `wan2.2_i2v_lightning_4step_high.safetensors` (1.2G), `wan2.2_i2v_lightning_4step_low.safetensors` (1.2G).
  - VAEs (`1.8 GB`): `flux2-vae.safetensors` (161M), `wan_2.1_vae.safetensors` (243M), `Wan2.2_VAE.safetensors` (1.4G).
- **Ollama Store (`/usr/share/ollama`, `78 GB`):** 19 models including active Avatar V2 models (`qwen3:8b`, `llama3.2:3b`, `qwen3-vl:8b-instruct`, `qwen3-vl:4b-instruct`, `qwen3-vl:2b-instruct`).
- **Other Storage Consumers (Read-Only Inventory — Untouched):** `~/.cache/huggingface` (`24 GB`, includes already-cached `yolov8x-worldv2.pt` 140 MB & `adetailer` 63 MB), `dist/` (`3.8 GB` across 11 rollback/current `AvatarV2` bundles), `~/.avatar_v2` (`2.6 GB`: `1.5 GB` updates, `845 MB` runs, `70 MB` immutable benchmarks), `~/.cache/pip` (`997 MB`).

---

## 4. Security, Sandbox & Frontend Verification Summary

- **Offline & Privacy:** All LLM/VLM calls strictly enforce loopback `127.0.0.1` / `localhost` (`ValueError` on non-loopback URLs). Zero local images, references, or inspection frames transmitted externally. `allow_cloud_escalation=False` hard-enforced.
- **Script Isolation:** Linux Bubblewrap (`--unshare-net`, read-only `/sdk` + `0o400` SHA-256-pinned script snapshot, `--clearenv`) + `prlimit` memory/CPU bounds + Unix-socket `GenerationBroker` + Static AST Inspector.
- **Prompt Injection Defense:** Image-embedded prompt-injection patterns detected, warned, and sanitized; VLM outputs parsed via closed Pydantic schemas and cannot emit arbitrary commands, workflows, or policy overrides.
- **Frontend Preservation:** Approved frontend preserved across Sprints 1–9 (`JS syntax OK`, `JS_API_METHODS` synced in `js_bridge.py`, WebKitGTK `AVATAR_V2_UITEST=1` passed with `0` JS errors in Phase B; zero separate QA/correction/specialist dashboards).

---

## 5. Remaining Known Limitations & Unresolved Risk Register

| Risk / Limitation | Severity | Current Mitigation | Planned Resolution |
|---|---|---|---|
| **Tight primary NVMe space (`43 GB` free, 96% full)** | High | Predictive storage guard (`30/20/15 GB` tiers) + `SpecialistRegistry` acquisition gate blocking unauthorized downloads | Owner-approved archival/cleanup of old `dist/AvatarV2-prev-*`, unused Ollama models, or mounting secondary NVMe before larger model acquisition |
| **Subtle object states (open vs closed) & fine hand/micro-identity nuance** | Medium | Blind 2D grid keeps inconclusive states `uncertain` (`0%` FAR); `CATEGORY_VERIFICATION_ELIGIBILITY` enforces `requires_human_review` | Optional lightweight Apache-2.0 `grounding-dino-tiny` (694 MB) / `DWPose` (210 MB) upon owner storage approval |
| **No dedicated pose/depth ControlNet or Fun-Control weights installed** | Medium | `SpecialistRegistry` explicitly blocks incompatible SDXL ControlNets and marks `pose_conditioning`/`composition_conditioning` unsupported | Evaluate `Wan2.2-Fun-5B-Control` + `Depth-Anything-V2-Small` after disk headroom increases |
| **Production bundle (`dist/AvatarV2`) still on pre-Sprint-1 build** | Medium | Source execution works cleanly; rollback packages preserved untouched | Sprint 10 controlled isolated rebuild & smoke-test (preserving rollback bundle!) |
| **Swap thrashing under prolonged multi-model sessions** | Low-Med | `ResourcePlanner` swap-rate & PSI guard (verified live in Sprint 8!) + `--cache-ram 6 24` + verified Ollama VRAM release | Sprint 10 sustained endurance stress test |

---

## 6. Rollback Procedure

1. **Application bundle rollback:** Existing working production bundle at `dist/AvatarV2` and timestamped backups (`dist/AvatarV2-prev-*`) have NEVER been overwritten; rollback requires zero file restoration unless a new Sprint 10 candidate bundle is staged.
2. **Git branch rollback:** All Sprint 1–9 work lives cleanly on `sprint-1-gpu-memory-foundation`; reverting to any sprint boundary is a clean `git checkout <commit>` (`bf40001` through Sprint 9 HEAD).
3. **Environment & model preservation:** `/home/jd/AvatarForge/venv`, `/home/jd/AvatarForge/comfyui/models/`, `~/.avatar_v2/desktop.json`, and `~/.avatar_v2/benchmarks/` remain completely intact.

---

## 7. Required Sprint 10 Final Acceptance Tests

Before invoking Astra for the single consolidated release-candidate validation after Sprint 10:
1. **Integrated end-to-end functional acceptance:** Automatic, Guided, and Script modes + local reference picker + multi-view Character Profile + FLUX.2 masked localized edit + Wan 2.2 FLF2V + VisualInspector + CorrectionCoordinator cancellation & recovery.
2. **Sustained resource & thermal endurance:** Back-to-back mixed FLUX.2 9B -> Qwen3-VL 8B -> Wan 2.2 14B -> FLUX.2 9B transitions verifying `0` OOM, `100%` VRAM release, bounded RSS/swap, and safe thermal behavior.
3. **Security & offline adversarial suite:** Non-loopback URL blocking, image prompt-injection resistance, Bubblewrap network/filesystem/TOCTOU escape tests, cloud-billing lock verification, and AST bypass attempts.
4. **Live desktop WebKitGTK UI acceptance (`AVATAR_V2_UITEST=1`):** Fresh isolated desktop run confirming `0` JS errors, viewer collapsed inspection/correction details, and zero unintended UI regressions.
5. **Isolated candidate bundle build & rollback verification:** Build isolated release-candidate bundle without overwriting existing production rollback bundles, verify startup and rollback procedure.
