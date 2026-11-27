# Avatar V2 — Sprint 10 Final Integrated Acceptance & Release-Candidate Report

**Date:** 2026-11-27  
**Branch:** `sprint-1-gpu-memory-foundation`  
**Sprint 9 Base Commit:** `4692b686bc0325d9600c425a12d5bd62f023efe4`  
**Final Disposition:** **`RC_READY_FOR_INDEPENDENT_REVIEW`**  
**Promotion State:** Isolated release candidate staged at `dist/AvatarV2.next`; production `dist/AvatarV2`, owner `~/.avatar_v2/desktop.json`, installed models, and rollback bundles remain **100% untouched**.

---

## 1. Release-Candidate Identity & Packaging Verification (Section 2)

| Field | Verified Value |
|---|---|
| **Candidate Bundle Path** | `/home/jd/avatar_project/dist/AvatarV2.next/AvatarV2` |
| **Candidate Executable SHA-256** | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` |
| **Untouched Production Bundle** | `/home/jd/avatar_project/dist/AvatarV2/AvatarV2` |
| **Production Executable SHA-256** | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` (unchanged mtime `1795541526`) |
| **Untouched Owner Config SHA-256** | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` (`~/.avatar_v2/desktop.json`, unchanged mtime `1795544113`) |
| **Build Toolchain** | PyInstaller `6.22.3`, Python `3.14.4` (`.venv314`), Linux `7.0.0-38-generic` x86_64 glibc `2.43` |
| **Bundled Data Verified in `_internal/`** | `avatar_v2/ui/index.html` (134,547 B), `avatar_v2/data/prompt_libraries.json` (12,790 B), `avatar_v2/sdk_client.py` (16,439 B), all 4 `workflows/*.json` |
| **Launch Outside Repository** | Verified from `cwd=/tmp` via `AVATAR_V2_CANDIDATE=1 AVATAR_V2_HOME=/tmp/av2_rc_test_home AVATAR_V2_UITEST=1 /home/jd/avatar_project/scripts/launch_avatar_v2.sh` |
| **Packaged UI Self-Test Result** | **PASS** (`FINAL_RC_UITEST_FOUND_AT_17s`: all 4 views, all 7 recipes, all 31 checks `true` including `guided_opens`, `subject_refs_roles`, `character_profiles_ok`, `script_panel_shown`, `script_sandbox_available`, `browse_local_present`, `automatic_restored`, `all_tab: "4 items"`, `js_errors: []`) |
| **Secondary NTFS Drive (`/dev/nvme1n1`)** | Untouched (not mounted, reformatted, or written to) |

### Exact Candidate Build, Launch & Verification Commands

```bash
# 1. Clean isolated PyInstaller candidate build (never touches dist/AvatarV2)
rm -rf /tmp/avatar_v2_rc_dist /tmp/avatar_v2_rc_build
.venv314/bin/python -m PyInstaller --noconfirm --clean \
  --distpath /tmp/avatar_v2_rc_dist --workpath /tmp/avatar_v2_rc_build scripts/avatar_v2.spec
rm -rf dist/AvatarV2.next && mv /tmp/avatar_v2_rc_dist/AvatarV2 dist/AvatarV2.next

# 2. Isolated candidate launch from outside the repository
cd /tmp && AVATAR_V2_CANDIDATE=1 AVATAR_V2_HOME=/tmp/av2_rc_test_home \
  /home/jd/avatar_project/scripts/launch_avatar_v2.sh

# 3. Full regression & Sprint 10 acceptance test suite (417 tests) + UI JS syntax check
.venv314/bin/pytest -q tests/
node -e "const fs=require('fs'),html=fs.readFileSync('avatar_v2/ui/index.html','utf8'),m=html.match(/<script>([\s\S]*?)<\/script>/); new Function(m[1]); console.log('UI JS syntax OK');"

# 4. Live Sprint 10 GPU acceptance scripts (separate held-out grid eval, mixed workload, live Journey F)
.venv314/bin/python -u scripts/eval_sprint10_acceptance.py
.venv314/bin/python -u scripts/eval_sprint10_acceptance.py --journey-f-live
```

---

## 2. Defects Found, Fixed & Retested in Sprint 10

Every defect below was discovered during Sprint 10 packaging, integration auditing, or live execution, fixed surgically within frozen scope, and verified via regression tests and the rebuilt `dist/AvatarV2.next` candidate:

| # | Severity | Component | Defect Found | Fix Implemented | Retest Proof |
|---|---|---|---|---|---|
| 1 | **Critical** | [script_runner.py](file:///home/jd/avatar_project/avatar_v2/script_runner.py) | Frozen PyInstaller builds treated `sys.executable` (the GUI binary!) as Python for Script mode and mounted `_MEIPASS` | Added `resolve_script_python()` & `_resolve_sdk_client_source()` using real host `python3` / `AVATAR_V2_SCRIPT_PYTHON` and `sys._MEIPASS` | `test_journey_e_frozen_python_resolution_never_returns_frozen_binary` + live packaged `script_sandbox_available: true` |
| 2 | **High** | [desktop.py](file:///home/jd/avatar_project/avatar_v2/desktop.py) | `APP_HOME` hardcoded to `~/.avatar_v2`; bundled workflows lacked `sys._MEIPASS` lookup; missing `import sys` | Added `AVATAR_V2_HOME` override, `_bundle_root()`, and `import sys` | Packaged launch from `/tmp` wrote exclusively to `/tmp/av2_rc_test_home`; `~/.avatar_v2/desktop.json` SHA-256 unchanged |
| 3 | **High** | [launch_avatar_v2.sh](file:///home/jd/avatar_project/scripts/launch_avatar_v2.sh) | Auto-promoted `dist/AvatarV2.next` over `dist/AvatarV2` on next launch | Added explicit `AVATAR_V2_CANDIDATE=1` isolated launcher and gated promotion behind `AVATAR_V2_PROMOTE_NEXT=1` | Candidate launched without altering `dist/AvatarV2` (`8e71ad4d...`) |
| 4 | **High** | [local_engine.py](file:///home/jd/avatar_project/avatar_v2/local_engine.py), [prompt_library.py](file:///home/jd/avatar_project/avatar_v2/prompt_library.py) | Bundled Wan workflows and `prompt_libraries.json` assumed repo-relative paths | Added `sys._MEIPASS`-aware resolution | Packaged launch from `cwd=/tmp` loaded Guided prompt libraries and Wan models cleanly |
| 5 | **High** | [desktop.py](file:///home/jd/avatar_project/avatar_v2/desktop.py) | `_run_visual_inspection()` had no real production ComfyUI execution hook for `CorrectionCoordinator` (and referenced non-existent `.comfyui_client` / wrong `_script_runner` attr) | Wired real `_execute_local_correction_step(item, step)` using `ComfyUIProvider`, verified Ollama VRAM release, ComfyUI `/free`, and `self._script_run_manager._active_broker` | `test_journey_f_production_desktop_correction_coordinator_end_to_end` + **live RTX 4070 Journey F** (`sprint10_journey_f_live.json`) |
| 6 | **High** | [script_broker.py](file:///home/jd/avatar_project/avatar_v2/script_broker.py) | Original Script jobs and corrective attempts did not share the host-enforced run budget | Added thread-safe `GenerationBroker.charge_corrective_attempt()` wired into `_execute_local_correction_step()` | `test_journey_e_script_shared_budget_and_no_auto_execute` |
| 7 | **High** | [resource_monitor.py](file:///home/jd/avatar_project/avatar_v2/resource_monitor.py) | Single-step `dt = max(..., 1.0)` concealed sustained paging when polled rapidly (`0.1s`) | Replaced with rolling ~5 s deque window anchored at `>= 1.0s` span | `test_section5_swap_rate_invariant_across_rapid_and_slow_polling` (`95–105 pg/s` at both `0.1s` and `2.0s`; 1-page glitch `<10 pg/s`) |
| 8 | **Medium** | [render_queue.py](file:///home/jd/avatar_project/avatar_v2/render_queue.py) | `QueuedRender.public()` omitted `verification_record`, dropping original requirements / inspection / correction records on history reload | Included `verification_record` in `QueuedRender.public()` persisted to `history/<id>.json` | `test_journey_a_automatic_generation_history_and_binding` & live Part B history JSONs |
| 9 | **Medium** | [workflow_router.py](file:///home/jd/avatar_project/avatar_v2/workflow_router.py), [desktop_studio.py](file:///home/jd/avatar_project/avatar_v2/desktop_studio.py) | Guided `state=="locked"` attributes (including subject-level `clothing`) were not forwarded into `mandatory_acceptance_criteria`; video scenes lacked `verification_record` / optional `end_image` (`LAST_FRAME`) wiring | Mapped top-level & subject-level locked Guided attributes into `AcceptanceCriterion(..., source="guided_ui", locked=True)`; wired `verification_record` + `LAST_FRAME` + scene post-join inspection | `test_journey_b_guided_lock_randomize_survives_compilation_routing_and_correction` & `test_journey_d_video_starter_never_replaced_and_flf2v_wired` |
| 10 | **High** | [inspection_contract.py](file:///home/jd/avatar_project/avatar_v2/inspection_contract.py) | `compute_inspection_status()` returned `"requirements_verified"` even when `needs_human_corroboration` was `True` for uncorroborated `requires_human_review` criteria (caught live during Sprint 10 Part B!) | Fixed to return `("warn", "review_recommended", "Review recommended", True)` whenever any mandatory criterion requires human corroboration | `test_section5_fault_injections_planner_vlm_timeout_and_honest_status` + full 417-test suite |

---

## 3. Requirement-to-Test Matrix (Journeys A–G & Sections 4–7)

| ID | Acceptance Requirement | Verification Method | Evidence Artifact | Status |
|---|---|---|---|---|
| **J-A** | Automatic image generation -> inspect -> status -> reopen from history with bound original requirements, compiled prompt, model settings & inspection | Unit + Live RTX 4070 (`DesktopAPI.forge_go` FLUX.2 9B + Qwen3-VL 8B) + Packaged UI | `test_journey_a_*`, `sprint10_live_mixed_workload.json` (`d83c4c973c`, `270.1 s`, peak `7,569 MiB`) | **PASS** |
| **J-B** | Guided generation: set attributes, lock attribute, randomize unlocked attribute; locked requirements survive compilation, routing & correction | Automated integration (`DesktopAPI.forge_go` -> `WorkflowRouter` -> `build_corrective_prompt`) + Packaged UI (`guided_opens: true`) | `test_journey_b_*`, `ui_test_report.json` | **PASS** |
| **J-C** | Local images & reusable references: My Stuff + Browse Local Files, multi-view roles, Character Profile across restart, cancel/replace/remove, spaces & Unicode paths, missing/moved source errors, ephemeral `staged_refs` deletion does not break profiles, originals unchanged, multi-subject separation | Automated integration (`DesktopAPI.pick_local_images`, `character_profiles`, `SubjectReferenceSet`) + Packaged UI | `test_journey_c_*`, `ui_test_report.json` (`subject_refs_roles: true`, `character_profiles_ok: true`, `browse_local_present: true`) | **PASS** |
| **J-D** | Video generation: local starter + supplementary refs, Wan FLF2V (`LAST_FRAME`), endpoint vs intermediate continuity evaluated separately, explicit starter image NEVER silently replaced | Unit + Live RTX 4070 (`DesktopAPI.forge_go` Wan 2.2 A14B `49f @ 480x832`, `205.21 s`, peak `7,435 MiB`, `photo_first: false`, visually inspected 3 frames) | `test_journey_d_*`, `sprint10_live_mixed_workload.json` (`ac5e6521fa`), `vid_frames/frame_01..03.png` | **PASS** |
| **J-E** | Script mode: select/inspect never executes; native SDK bounded run; shared host budget with corrective attempts; static extractor; documented A1111 adapter; legacy explanation; frozen Python resolver; live Bubblewrap boundary test | Live Bubblewrap 0.9.0 + `prlimit` test (blocks `/dev/nvidia0`, `~/.ssh`, `1.1.1.1:53`, TOCTOU mutation detection, 5-thread concurrent burst capped at 2/2, fail-closed without bwrap) + Packaged UI | `test_journey_e_*`, `ui_test_report.json` (`script_sandbox_available: true`) | **PASS** |
| **J-F** | Automatic correction through **actual production `DesktopAPI._run_visual_inspection()` -> `CorrectionCoordinator` -> `_execute_local_correction_step()`** (not just harness), retaining both original and corrected outputs with honest UI status | Automated end-to-end + **Live RTX 4070 production trial** (`121.18 s`, Attempt 1 `targeted_image_edit` added crisp `'OPEN'` tabletop sign while preserving lamp, vase, fern & curtains; 2 outputs retained; honest `some_uncertain` status) | `test_journey_f_*`, `sprint10_journey_f_live.json` | **PASS** *(see Section 4 note on async `/free` & mask provenance)* |
| **J-G** | Persistence & interrupted-job recovery across restart without duplication or false completion | Automated `RenderQueue` persistence + Packaged RC reading copied owner history (`all_tab: "4 items"`) | `test_journey_g_*`, `ui_test_report.json` | **PASS** |
| **S4-1** | Reconciled `adh_04_spatial` requirement (`vase-left / lamp-right`) retained | Verified in `adherence_benchmark.py` and Sprint 9/10 records | `docs/SPRINT_9_SPECIALIST_EXPANSION.md` | **PASS** |
| **S4-2** | Grid-assisted Qwen3-VL localization tested on **separate photographic held-out set** (distinct from Sprint 9's 6 images) | Live Qwen3-VL 8B eval on 4 distinct Sprint 5 photos (`adh_02`, `adh_05`, `adh_06`, `adh_10`, 13 checks) | `sprint10_heldout_grid_eval.json`: **`TP=8, TN=5, FP=0, FN=0, uncertain=0`** (`13/13`), VRAM released in `112.3 ms` | **PASS** *(keep `requires_human_review` & note `bbox_norm` clamping)* |
| **S4-3** | Masked correction provenance & honest UI status (never `"Requirements verified"` while any mandatory requirement is failed/uncertain/pending/uncorroborated) | Defect #10 fixed & tested + Sprint 9 `0.0%` outside-mask drift verified with explicit `user_or_harness` mask provenance | `avatar_v2/inspection_contract.py`, `test_section5_*` | **PASS** |
| **S5-1** | Bounded mixed-workload live session (FLUX 9B -> Qwen3-VL -> Wan 14B -> VideoInspector -> FLUX 4B -> full shutdown) | Live RTX 4070 session (`0` OOM, full recovery from `1,153 MiB` baseline -> `7,569 MiB` peak -> `1,016 MiB` idle; RAM `37,189 MiB` -> `36,777 MiB`) | `sprint10_live_mixed_workload.json` (`hw_trace`) | **PASS** |
| **S5-2** | Pressure monitoring invariance at rapid (`0.1s`) vs slower (`2.0s`) polling intervals | Automated simulation proving sustained `100 pg/s` paging measures `95–105 pg/s` at both intervals without amplifying 1-page glitches | `test_section5_swap_rate_invariant_across_rapid_and_slow_polling` | **PASS** |
| **S5-3** | Fault injections: cancellation, invalid VLM JSON, VLM timeout, memory-pressure & storage-reserve refusal, budget exhaustion, offline sandbox (`--unshare-net`) | Automated fault suite + live Journey F CPU VLM `ReadTimeout` graceful degradation | `test_section5_fault_injections_*`, `test_journey_e_live_bwrap_*` | **PASS** |
| **S6** | Accurate Wan 2.2 sampling capability statements | 4-step Wan Lightning (`CFG 1.0`) kept as default; Sprint 9 10-step base trial (`703.5 s`) explicitly documented as a **tested reduced-step base configuration** (not full upstream 20–40 step base sampling) | Section 5 below | **PASS** |
| **S7** | Rollback demonstration using copied test data | Untouched `dist/AvatarV2/AvatarV2` launched against isolated `/tmp/av2_rollback_home` copy -> passed UI self-test (`js_errors: []`); live `~/.avatar_v2/desktop.json` & models untouched | Rollback UI test log + SHA-256 verification | **PASS** |

---

## 4. Inspection, Localization & Correction Evidence & Honest Limitations (Section 4)

### 4.1 Separate Photographic Held-Out Evaluation of Blind 2D Grid Localization

Evaluated `run_blind_object_localization()` (`qwen3-vl:8b-instruct` + vertical `x=0/25/50/75/100%` ruler) on **4 completely separate immutable Sprint 5 photographs** not used in Sprint 9 (`~/.avatar_v2/benchmarks/sprint10_acceptance/sprint10_heldout_grid_eval.json`):

| Held-Out Photo | Checks Tested | TP | TN | FP | FN | Uncertain | Latency | VRAM Released |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `adh_02_composition` | `1 man`, `2 men (false)`, `man left of bench`, `bench left of man (false)` | 2 | 2 | 0 | 0 | 0 | 14.39 s | Yes (`356.5 ms`) |
| `adh_05_colors` | `1 folder`, `2 folders (false)` | 1 | 1 | 0 | 0 | 0 | 9.62 s | Yes (`357.0 ms`) |
| `adh_06_clothing` | `1 eyeglasses`, `2 earrings`, `1 coffee mug (false)` | 2 | 1 | 0 | 0 | 0 | 11.73 s | Yes (`362.5 ms`) |
| `adh_10_complex_nl` | `1 lamp`, `2 windows`, `woman left of lamp`, `lamp left of woman (false)` | 3 | 1 | 0 | 0 | 0 | 16.31 s | Yes (`392.2 ms`) |
| **Sprint 10 Held-Out Total** | **13 distinct checks** | **8** | **5** | **0** | **0** | **0** | **13.0 s avg** | **100% verified** |

**Honest Engineering Caveats Observed:**
1. **Ruler-anchored `x_center_pct` & object counting generalized cleanly (`13/13`, `0.0%` FAR)**, confirming strong 2D horizontal corroboration; however, `object_quantities` and `spatial_relationships` remain conservatively classified as `requires_human_review` so single-pass VLM results without corroboration never display `"Requirements verified"` (enforced by Defect #10 fix).
2. **Raw `bbox_norm` clamping limitation:** Qwen3-VL 8B frequently emits 0–1000 scale box coordinates that clamp to `[100.0, 100.0, 100.0, 100.0]`. Consequently, `bbox_norm` from Qwen3-VL 8B is **not** used for autonomous mask generation.
3. **Mask provenance & autonomous vs manual boundary:**
   - **Autonomous production correction path (`DesktopAPI._run_visual_inspection()` -> `CorrectionCoordinator` -> `_execute_local_correction_step()`):** Uses unmasked `targeted_image_edit` (FLUX.2 `ReferenceLatent`) or `prompt_refinement` / `seed_variation` fully autonomously (live-proven in Journey F: added `'OPEN'` tabletop sign in `121.18 s`, retaining both original and corrected outputs).
   - **Masked localized editing (`builtin:flux2_masked_edit` with `0.0%` outside-mask pixel drift, Sprint 9):** Currently requires explicit **user-supplied or harness-supplied region boxes (`mask_provenance = "user_or_harness"`)** via `generate_region_mask()` rather than autonomous VLM bounding-box extraction, until an optional Apache-2.0 open-vocabulary detector (e.g. deferred `grounding-dino-tiny`) is approved.
4. **Sequential VLM reinspection settle window:** During live Journey F, calling `ComfyUIProvider.free_memory(unload_models=True)` immediately before `_re_inspect()` returned asynchronously before VRAM fully dropped, causing `choose_vision_execution_schedule` to select CPU VLM (which hit `ReadTimeout` and honestly degraded to `some_uncertain`). Waiting on `GPUMemoryManager.maybe_cleanup(..., force=True)` before sequential VLM reinspection is noted as a low-risk post-RC polish item.

---

## 5. Accurate Capability, Experimental & Deferred Matrix (Sections 1 & 6)

| Capability | Status | Verified Operational Bounds |
|---|---|---|
| **FLUX.2 Klein 9B Base (`28 steps, CFG 3.5`)** | **Supported (Default Still)** | `~270 s` cold, `~224 s` warm, peak `~7.57 GB` VRAM (`--cache-ram 6 24`), multi-ref `ReferenceLatent` |
| **FLUX.2 Klein 4B (`8 steps, CFG 1.2`)** | **Supported (Fast Still)** | `~52 s` post-video recovery, peak `~5.53 GB` VRAM |
| **Wan 2.2 I2V A14B + Lightning (`4 steps, CFG 1.0`)** | **Supported (Default Video)** | `205–225 s` (`49f @ 480x832`), peak `~7.44 GB` VRAM, tiled VAE, optional `LAST_FRAME` FLF2V; explicit starter preserved |
| **Wan 2.2 TI2V 5B Turbo (`4 steps, CFG 1.0`)** | **Supported (Fast Video)** | Verified in Sprint 5/6; faster alternative sharing `umt5_xxl` |
| **Wan 2.2 A14B Base (`10 steps, CFG 3.5`, no Lightning)** | **Experimental / Opt-In Only** | Tested reduced-step base configuration (`703.5 s`, `3.12x` slower than 4-step Lightning); **not** equivalent to full upstream 20–40 step base sampling |
| **Automatic, Guided & Script Modes + Local Picker + Character Profiles** | **Supported** | All 3 modes preserved inside approved frontend; Bubblewrap + `prlimit` + `GenerationBroker` fail-closed isolation |
| **Hybrid VisualInspector / VideoInspector & Bounded CorrectionCoordinator** | **Supported (Advisory / Bounded)** | Loopback Ollama `qwen3-vl:8b-instruct` with verified VRAM release; max 2 image / 1 video retries; never claims `"Requirements verified"` on uncorroborated human-review categories |
| **Masked Localized Editing (`builtin:flux2_masked_edit`)** | **Experimental (User/Harness Masks)** | Live-proven `0.0%` outside-mask drift; requires explicit region mask (`mask_provenance = "user_or_harness"`) |
| **Optional Local A1111 Compatibility Adapter** | **Supported (Opt-In Subset)** | Documented `txt2img`/`img2img`/`sd-models`/`progress` subset only; explicitly rejects unsupported A1111 flags (`enable_hr`, `alwayson_scripts`, etc.) |
| **Dedicated Detection / Fine Segmentation / Pose / Depth / Upscaling (`grounding-dino-tiny`, `florence-2`, `sam2.1`, `dwpose`, `depth-anything-v2`, `realesrgan`, `Wan2.2-Fun-5B-Control`)** | **Deferred** | Blocked behind `SpecialistRegistry` storage/approval gate (`43 GB` free on primary NVMe; secondary NTFS drive untouched) |
| **Storage Migration to Secondary NVMe (`/dev/nvme1n1`)** | **Deferred (Out of Scope)** | Untouched in Sprint 10 |

---

## 6. Live Mixed-Workload Resource & Recovery Summary (Section 5)

Captured from isolated live RTX 4070 session (`~/.avatar_v2/benchmarks/sprint10_acceptance/sprint10_live_mixed_workload.json`, `0` OOM failures):

| Session Checkpoint | VRAM Used / Total | Host RAM Available | Swap Used | Notes |
|---|---:|---:|---:|---|
| Before ComfyUI start | `1,153 / 8,188 MiB` | `37,189 MiB` | `8,109 MiB` | Idle desktop baseline |
| ComfyUI `18189` ready (`--cache-ram 6 24`) | `1,258 / 8,188 MiB` | `36,453 MiB` | `8,109 MiB` | Clean startup |
| FLUX.2 9B (`270.1 s`) + Qwen3-VL 8B (`15.4 s`) | Peak `7,569 MiB` -> `1,381 MiB` | `34,111 MiB` | `8,191 MiB` | ComfyUI `/free` + Ollama VRAM release verified |
| Wan 2.2 A14B (`49f`, `205.2 s`) + VideoInspector (`0.73 s`) | Peak `7,435 MiB` -> `1,372 MiB` | `24,441 MiB` | `8,191 MiB` | `max_luma_jump=1.58`, `sharpness=202.0`, `0` freezes |
| Post-video FLUX.2 4B recovery (`52.1 s`) | Peak `5,531 MiB` -> `4,323 MiB` | `24,360 MiB` | `8,191 MiB` | Retained warm per `ResourcePlanner` |
| Full shutdown & recovery | **`1,016 / 8,188 MiB`** | **`36,777 MiB`** | `8,191 MiB` | Clean baseline recovery, `0` leaked GPU processes |

---

## 7. Activation & Rollback Procedure (Section 7)

### Testing the Isolated Candidate (Current Safe State)
```bash
AVATAR_V2_CANDIDATE=1 AVATAR_V2_HOME=/tmp/avatar_v2_rc_home /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

### Controlled Promotion (ONLY After Independent Review Approval)
```bash
# 1. Confirm no running AvatarV2 instance
pgrep -fa "dist/AvatarV2/AvatarV2" || true
# 2. Explicitly promote dist/AvatarV2.next (automatically archives current dist/AvatarV2 to dist/AvatarV2-prev-<timestamp>)
AVATAR_V2_PROMOTE_NEXT=1 /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

### Immediate Rollback (Verified in Sprint 10)
```bash
# If promoted and rollback is desired:
mv dist/AvatarV2 dist/AvatarV2-failed-rc
mv dist/AvatarV2-prev-<timestamp> dist/AvatarV2
# Until promoted, dist/AvatarV2 is already untouched and launches normally via:
/home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

---

## 8. Final Disposition

**Disposition:** **`RC_READY_FOR_INDEPENDENT_REVIEW`**

- All **417 automated regression and Sprint 10 acceptance tests pass** (`0` failures), UI JS syntax is clean, all 10 Sprint 10 defects were fixed and retested, the rebuilt isolated candidate `dist/AvatarV2.next` (`2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee`) passed live packaged UI verification from `/tmp` with `0` JS errors, separate held-out grid localization scored `13/13` (`0%` FAR), live production mixed-workload & Journey F correction completed with `0` OOM and clean GPU recovery, and rollback to untouched `dist/AvatarV2` (`8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893`) was verified cleanly.
- Ready for one consolidated independent review by Astra when explicitly requested by the owner.
