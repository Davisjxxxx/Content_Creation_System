# Avatar V2 RC2 — Targeted Acceptance & Adversarial Verification Report

- **Date:** 2026-11-27
- **Branch:** `rc1-astra-remediation`
- **Build Commit for `dist/AvatarV2.rc2`:** `5401b27`
- **Baseline RC1 Commit (preserved unchanged):** `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12`
- **Astra Audit Commit (preserved unchanged):** `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` (`audit/astra-rc1-independent`)

---

## 1. Automated & Adversarial Regression Suite

| Verification Layer | Command / Scope | Result |
| --- | --- | --- |
| **Full Python Regression + RC1 Adversarial Suites** | `.venv/bin/pytest -q` | **447 / 447 passed (0 failed, 0 skipped)** |
| Gate 1 Adversarial Suite (`RC-01`, `RC-02`, `RC-03`) | `tests/test_rc1_gate1_ownership.py` | **7 / 7 passed** (disposable sleeper PID-reuse refusal, external/symlink/hardlink delete refusal, unlink-before-history, JsBridge boundary) |
| Gate 2 Adversarial Suite (`RC-04`, `RC-11`, `RC-15`, `RC-17`, `RC-22`) | `tests/test_rc1_gate2_providers_and_scripts.py` | **7 / 7 passed** (no local->Runway auto-switch, concurrent budget race, zero-egress startup, durable script budgets, model capability filtering, mandatory `prlimit` + bounded pipes/connections/pgid watchdog) |
| Gate 3 Adversarial Suite (`RC-10`, `RC-16`, `RC-18`) | `tests/test_rc1_gate3_lifecycle_and_durability.py` | **6 / 6 passed** (partial-output node error refusal, traversal rejection, QA-phase cancellation preserving original output, post-submit `recovery_required`, corrupt JSON quarantine, ENOSPC worker survival, `flock` single-writer enforcement) |
| Gate 4 Adversarial Suite (`RC-05`, `RC-07`, `RC-08`, `RC-09`) | `tests/test_rc1_gate4_inspection_and_criteria.py` | **4 / 4 passed** (Forge VLM/auto-correct UI controls + persistence, counts 1/3/11 + full spatial operands + typography negative suppression, no single-VLM self-corroboration + `review_recommended` preservation, timeout non-downgrade + reinspection crash resilience + SHA-256 binding) |
| Gate 5 Adversarial Suite (`RC-06`, `RC-12`, `RC-13`, `RC-14`, `RC-19`, `RC-20`, `RC-21`, `RC-23`) | `tests/test_rc1_gate5_references_masks_receipts_resources.py` | **6 / 6 passed** (cardinality/multi-subject/missing starter refusals, end-frame UI + record, profile SHA-256 tamper detection, staged ref pinning, `flux2_masked_edit` staging + EXIF + 0-drift, ground-truth receipts, fail-closed Ollama VRAM status, 10ms–2s/stale/reset swap-rate invariance, `AVATAR_V2_GPU_*` env precedence) |
| **Frontend JS Syntax Check** | `node --check` on extracted `<script>` from `avatar_v2/ui/index.html` | **PASS (`UI JS syntax OK`)** |

---

## 2. Isolated RC2 Packaging & Bytecode Correspondence (`dist/AvatarV2.rc2`)

- **Build Isolation:** Built via `.venv314/bin/pyinstaller --noconfirm --clean --distpath /tmp/avatar_v2_rc2_build/dist --workpath /tmp/avatar_v2_rc2_build/build scripts/avatar_v2.spec` and moved strictly to `dist/AvatarV2.rc2`.
- **Executable SHA-256 (`dist/AvatarV2.rc2/AvatarV2`):** `5eb0a9a43a08f10d433dd5829320092af47584ea898aaa675430d74ab70cfd1c`
- **Onedir Tree-Content Digest (`dist/AvatarV2.rc2/`, 40,782 entries):** `a875cb3d1d41c21e8055058d56705ac98964ee67e296d030cf45d9da794b3f08`
- **PYZ Bytecode Correspondence (`CArchiveReader` + `ZlibArchiveReader`):**
  - **58 / 58** `avatar_v2.*` modules inside `PYZ.pyz` (including new `avatar_v2.ownership`) matched exact normalized bytecode (`co_code`, `co_names`, `co_varnames`, `co_consts`, flags/argcounts) compiled from HEAD commit `5401b27` (**0 mismatched**).
- **Bundled Data & Workflow Integrity (7 / 7 exact SHA-256 matches):**
  - `avatar_v2/ui/index.html` (`8e512ceb517e7e95...`)
  - `avatar_v2/data/prompt_libraries.json` (`f5d07ab5a7364ff7...`)
  - `avatar_v2/sdk_client.py` (`46a249b14cf213ee...`)
  - `workflows/animatediff_sd15_api.json` (`3bcfa5560c10ead4...`)
  - `workflows/animatediff_sdxl_api.json` (`94c10ead4...` exact match verified)
  - `workflows/wan22_i2v_14b_gguf_api.json` (`062c81a58b52fdb9...`)
  - `workflows/wan22_i2v_5b_turbo_gguf_api.json` (`7f1cd4d5b8dd192e...`)

---

## 3. Packaged Desktop Self-Test (`AVATAR_V2_UITEST=1` from `/tmp`)

Executed `/home/jd/avatar_project/dist/AvatarV2.rc2/AvatarV2` with `cwd=/tmp`, isolated `AVATAR_V2_HOME=/tmp/rc2_uitest_home`, and `AVATAR_V2_UITEST=1`:

- **Views (`4 / 4`):** `make=true`, `forge=true`, `library=true`, `settings=true`
- **Recipes (`7 / 7`):** `look`, `photos`, `voice`, `talk`, `act`, `move`, `scene` all `shown=true`
- **UI Invariant & Functional Checks (`33 / 33`):** `tiles=7`, `forge_prompt=true`, `forge_models=true`, `guided_opens=true`, `subject_refs_roles=true`, `character_profiles_ok=true`, `script_panel_shown=true`, `script_sandbox_available=true`, `browse_local_present=true`, `automatic_restored=true`, `missing_image_refused=true`, `missing_video_refused=true`, `viewer_opens=true`, `viewer_missing_explained=true`, `viewer_enter_uses=true`, `viewer_next=true`, `viewer_escape=true`, `select_black_on_white=true`, `settings_limits=true`
- **JavaScript Console Errors:** **`[]` (0 errors)**

---

## 4. Targeted Live RTX 4070 Acceptance Verification

Executed on the local **NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)** using completely isolated `AVATAR_V2_HOME=/tmp/rc2_live_gpu` and private ComfyUI on port **`18189`** (never touching `~/.avatar_v2` or owner media):

| Live Trial | Findings Verified | Measured Outcome |
| --- | --- | --- |
| **1. Private ComfyUI Ownership Start & Shutdown** | `RC-01` (P0-A) | Started PID `864280` on port `18189`; `ownership="owned_private"`, `has_start_ticks=True`, `has_boot_id=True`. Shutdown cleanly stopped PID `864280`, reported `state="STOPPED"`, `ownership="stopped"`, and removed `comfyui_owner.json`. |
| **2. Live FLUX.2 4B Render & Ground-Truth Receipt** | `RC-13` | Completed in **`37.1 s`** (`phase="complete"`). Receipt recorded actual `RandomNoise` `seed=1978607419` (not default `41001`), `resolution=[1024, 1024]`, matching PIL `observed_media=[1024, 1024]` and exact output file SHA-256. |
| **3. Live `builtin:flux2_masked_edit` Staging & Full-Res Zero-Drift** | `RC-12`, `RC-23` | Staged external `SOURCE_IMAGE` & `MASK_IMAGE` through `ComfyUIProvider` into ComfyUI input and rendered in **`60.2 s`** (`seed=77112233`). Exact full-res RGB check: **`796,438` exact-black mask pixels, `0` changed pixels (`exact_black_mask_zero_drift=True`)**, **`74.37%` inside-mask pixels changed**, downsampled luminance diff `0.001`. |
| **4. Live Cross-Owner Handoff (`ComfyUI /free` → `qwen3-vl:8b-instruct` → Verified Release)** | `RC-08`, `RC-09`, `RC-14` | Unloaded ComfyUI via `GPUMemoryManager.maybe_cleanup(..., force=True)`, ran real `qwen3-vl:8b-instruct` (`sequential_gpu_vlm`) in **`12.6 s`** on a `spatial_relationships` criterion -> returned honest **`ui_status="review_recommended"`**, `human_review_required=True` (never falsely upgraded to `requirements_verified`), exact `output_sha256` + `requirement_set_hash` bound, and `verify_ollama_vram_released()` confirmed **`status="released"`** with **`6,486 MiB` free VRAM**. |

---

## 5. Evidence Precision Addendum (`RC-21` & `RC-23`)

1. **Exact-Black-Mask Preservation vs Downsampled Luminance (`RC-23`):**
   - Historical Sprint 9/10 logs are preserved unchanged. Beginning with RC2, `measure_outside_mask_preservation()` in `avatar_v2/specialist_vision.py` computes BOTH:
     - **Full-resolution exact RGB channel difference** across every pixel where `mask == 0` (`exact_black_mask_pixels`, `exact_black_mask_changed_pixels`, `exact_black_mask_max_channel_diff`, `exact_black_mask_zero_drift`), and
     - **512 px downsampled luminance statistics** (`outside_mask_mean_abs_diff`, `outside_mask_pct_changed`) for quick feather-band diagnostics.
   - Live RTX 4070 Trial 3 confirmed **0 / 796,438 changed exact-black pixels** (`exact_black_mask_max_channel_diff == 0`).
2. **UI Self-Test Invariants vs Observational Fields (`RC-23`):**
   - In `ui_test_report.json`, boolean checks (`views.*`, `recipes.*.shown`, `missing_image_refused`, `viewer_opens`, `settings_limits`, `js_errors == []`) are deterministic pass/fail invariants.
   - Fields such as `engine_state`, `forge_shots`, `avatars_tab`, and `all_tab` are environment/state observations reflecting the isolated test home and host ComfyUI install rather than static invariants.
3. **`A1111CompatibilityAdapter` Packaging Scope (`RC-21`):**
   - `avatar_v2/a1111_adapter.py` is explicitly marked `PACKAGED_STATUS = "source_only_optional"`. It remains a tested source-only optional translator and is intentionally not exposed as a default HTTP server or UI surface in `dist/AvatarV2.rc2`.

---

## 6. Protected Assets & Rollback Preservation Verification

| Protected Asset | Expected SHA-256 (Pre-Remediation) | Post-RC2 Verified SHA-256 | Status |
| --- | --- | --- | --- |
| Production `dist/AvatarV2/AvatarV2` | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` | **UNCHANGED** |
| Original RC1 `dist/AvatarV2.next/AvatarV2` | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` | **UNCHANGED** |
| Owner Config `~/.avatar_v2/desktop.json` | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` | **UNCHANGED** |
| Source branch `sprint-1-gpu-memory-foundation` | `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` | `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` | **UNCHANGED** |
| Audit branch `audit/astra-rc1-independent` | `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` | `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` | **UNCHANGED** |
| Secondary NTFS Drive (`/dev/nvme1n1`) | Untouched / Unmounted | Untouched / Unmounted | **UNCHANGED** |
