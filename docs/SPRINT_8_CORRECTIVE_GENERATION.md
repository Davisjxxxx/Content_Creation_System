# Avatar V2 — Sprint 8 Bounded Automated Corrective Generation

**Status:** Software-verified & live-GPU benchmarked (401 passing regression tests, held-out unseen VLM eval, and controlled live FLUX.2 9B + Qwen3-VL 8B Conditions A–E comparison)  
**Hardware:** RTX 4070 Laptop GPU (8,188 MiB VRAM), Intel Core i9-13980HX (24C/32T), 62 GiB RAM  
**Offline guarantee:** 100% local execution (`127.0.0.1` ComfyUI + loopback `127.0.0.1:11434` Ollama, zero automatic cloud escalation, zero new model downloads)

---

## 1. Architecture & Implemented Components

Sprint 8 builds strictly on existing Avatar V2 Sprint 1–7 infrastructure (`GenerationVerificationRecord`, `VisualInspectionResult`, `CorrectionRecommendation`, `CorrectionBudget`, `StructuredPromptSpec`, `ModelRegistry`, `ResourcePlanner`, `RenderQueue`, `VisualInspector`, `VideoInspector`) without adding competing execution systems or redesigning the frontend:

1. **Requirement-Specific Verification Eligibility & Inspector Calibration ([inspection_contract.py](file:///home/jd/avatar_project/avatar_v2/inspection_contract.py), [visual_inspector.py](file:///home/jd/avatar_project/avatar_v2/visual_inspector.py)):**
   - Added explicit `CATEGORY_VERIFICATION_ELIGIBILITY` distinguishing `deterministic_reliable`, `hybrid_corroborated`, and `requires_human_review` categories so weak categories (`object_quantities`, `spatial_relationships`, `fine_hand_anatomy`, `facial_identity_subtle`, `micro_detail_realism`, `video_motion`) are never silently auto-certified as fully verified without independent corroboration.
   - Added structured `localized_objects` (`label`, `state`, `horizontal_side`, `x_center_pct`) cross-checks (`_deterministic_cross_check_vlm_criterion`) for exact typography, exact counts, open/closed object states, and viewer-left/viewer-right spatial coordinates, plus conservative `reconcile_inspector_judgments()` (never silently accepts on disagreement).
2. **Verified Capability Matrix & Bounded Correction Coordinator ([correction_engine.py](file:///home/jd/avatar_project/avatar_v2/correction_engine.py)):**
   - **Verified installed strategies:** `targeted_image_edit` (FLUX.2 native `ReferenceLatent` edit conditioning on the prior candidate + preserved user refs, `is_localized_edit=True`), `additional_reference_conditioning` (`back_body`, `left_side`, `right_side`), `reference_adjustment`, `prompt_refinement` (full regeneration, honestly flagged `is_localized_edit=False`), `seed_variation`, `keyframe_correction`, `video_regeneration` (bounded max-1), `alternative_approved_workflow`, and `human_review_escalation`.
   - **Honestly marked unsupported (never fabricated):** `inpainting` (no masked inpaint weights installed), `composition_conditioning` (`controlnet/` empty), `pose_conditioning` (no pose/FunControl weights installed), and localized video temporal inpainting.
   - **Intent preservation (`build_corrective_prompt`):** Anchors every attempt to `record.original_prompt`, locked criteria, and previously satisfied criteria; never weakens requirements to satisfy the inspector and never silently replaces user-provided starter images.
   - **7-factor best-candidate ranker (`score_candidate`, `is_candidate_strict_improvement`):** Ranks by (1) net mandatory satisfaction, (2) zero regressions on previously satisfied criteria & zero new defects, (3) locked attribute preservation, (4) reference consistency, (5) visual quality, (6) video temporal consistency, and (7) latency. Rejects trade-off regressions (`rejected_regression`) rather than auto-replacing the prior best.
   - **Host-enforced budgets & guards (`CorrectionCoordinator`):** Provisional defaults of max `2` image retries and max `1` video retry, `allow_cloud_escalation=False`, `ResourcePlanner` admission + predictive storage check before every attempt, user cancellation check (`cancel_event`) at every boundary, complete `CorrectionAttemptRecord` & `CorrectionSessionReport` provenance preserving original and all candidate outputs, and `plan_batched_inspection_and_correction()` minimizing FLUX/Wan eviction cycles.
3. **Non-Intrusive Frontend & Queue Wiring ([desktop.py](file:///home/jd/avatar_project/avatar_v2/desktop.py), [desktop_studio.py](file:///home/jd/avatar_project/avatar_v2/desktop_studio.py), [index.html](file:///home/jd/avatar_project/avatar_v2/ui/index.html)):**
   - Concise statuses (`Generating`, `Checking quality`, `Improving result`, `Correction complete`, `Some requirements remain uncertain`, `Review recommended`) and collapsed `<details id="viewerInspection">` inside the existing media viewer. Zero new dashboards, zero agent panels, zero changes to `JS_API_METHODS`.

---

## 2. Separate Calibration vs Held-Out Evaluation (Live Qwen3-VL 8B)

Raw JSON preserved at `~/.avatar_v2/benchmarks/sprint8_corrections/sprint8_heldout_eval.json` (100% verified VRAM release after run).

### 2.1 Held-Out Unseen Evaluation Set (11 Brand-New Cases, 18 Criteria — Never Used in Sprint 7)

Includes 4 unseen immutable Sprint 5 outputs (`adh_02_composition`, `adh_05_colors`, `adh_06_clothing` with real failed `auburn_half_up` short bob, `mv_04_multi_perspective` vs `seed_left`) + 7 unseen synthetic ground-truth fixtures (5-triangle count pass/fail, purple-diamond/cyan-square spatial pass/fail, `"PHASE 08"` vs `"PHSAE 08"` typography, unwanted yellow circle):

| Category (Held-Out Unseen) | TP | TN | FP (FAR) | FN (FRR) | Uncertain | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| `clothing_appearance` (`adh_06`) | 3 | 1 | 0 (0.0%) | 0 (0.0%) | 0 | — |
| `colors` (`adh_05`) | 3 | 0 | 0 (N/A) | 0 (0.0%) | 0 | — |
| `composition` (`adh_02`) | 2 | 0 | 0 (N/A) | 0 (0.0%) | 0 | — |
| `missing_or_unwanted` (synthetic) | 0 | 1 | 0 (0.0%) | 0 (N/A) | 0 | — |
| `multiview_reference` (`mv_04` vs `seed_left`) | 2 | 0 | 0 (N/A) | 0 (0.0%) | 0 | — |
| `object_quantities` (synthetic) | 1 | 1 | 0 (0.0%) | 0 (0.0%) | 0 | — |
| `spatial_relationships` (synthetic) | 1 | 1 | 0 (0.0%) | 0 (0.0%) | 0 | — |
| `typography` (synthetic, corroborated) | 1 | 1 | 0 (0.0%) | 0 (0.0%) | 0 | — |
| **Held-Out Total (11 cases / 18 criteria)** | **13** | **5** | **0 (0.0%)** | **0 (0.0%)** | **0** | **10.77 s** |

### 2.2 Hard Photographic Calibration Recheck (Honest Disclosure of Persistent VLM Weakness)

On the 2 hardest Sprint 7 photographic failure cases (`adh_03_quantities` with handleless bowls + open notebook, and `adh_04_spatial` with angled brass lamp), single-pass `qwen3-vl:8b-instruct` STILL exhibited `FP=3, TN=0` (`TP=1`) because it hallucinated both coordinates and closed-notebook state on those specific photographs (`corr=False`). This empirically justifies keeping `object_quantities` and `spatial_relationships` classified as `requires_human_review` until dedicated Sprint 9 object detectors (e.g., YOLO/GroundingDINO) are evaluated.

---

## 3. Controlled Live GPU Generation & Correction Comparison (Conditions A–E)

Executed on RTX 4070 (`8,188 MiB`) with isolated ComfyUI (`127.0.0.1:18189`, `--cache-ram 6 24`) using exact production FLUX.2 9B Base parameters (`28 steps, CFG 3.5`, `896×1280`). Every original and candidate image was **independently verified by direct visual inspection** (not solely by VLM scores). Total storage overhead for all 6 new 896×1280 PNG candidates: **9.37 MB**. Zero OOM errors; GPU thermal max `73°C`.

### 3.1 `adh_03_quantities` (Target: Exactly 3 Handled Ceramic Mugs + Exactly 2 Closed Notebooks)

| Condition | Strategy | Independent Visual Ground Truth | Satisfied | Preserved vs Original | Notes / Representative Finding |
|---|---|---|---:|---|---|
| **A (Sprint 5 Original)** | Immutable baseline | 2 handled mugs + 2 bowls; 1 open + 1 closed notebook | **0/2** | Baseline | Real Sprint 5 count + state failure (untouched) |
| **B (Deterministic Compile)** | `compile_prompt_for_model`, no correction | 4 handled mugs + 1 closed blue notebook | **0/2** | N/A (new scene) | Crisp photorealism; still missed exact counts |
| **C (Local LLM Prompt)** | `interpret_natural_language(use_llm=True)`, no correction | 2 handled mugs + 2 cleanly closed hardcover notebooks | **1/2** | N/A (new scene) | Improved closed-notebook adherence (`0/2 -> 1/2`), still short 1 mug |
| **E1 (Targeted Image Edit)** | FLUX.2 `ReferenceLatent` edit on Cond A (`is_localized_edit=True`) | **Closed BOTH notebooks & removed 1 bowl**; 2 mugs + 1 bowl left | **1/2** | **High** (same oak desk, mugs & lighting) | Localized edit reliably fixed notebook state without altering scene |
| **D/E2 (Bounded Correction Attempt 2)** | `prompt_refinement` full regen (`is_localized_edit=False`) | **EXACTLY 3 handled ceramic coffee mugs + EXACTLY 2 closed hardcover notebooks** | **2/2** | New clean layout | **Complete adherence fix (`0/2 -> 2/2`)**, zero bowls, zero open pages |

### 3.2 `adh_04_spatial` (Target: Green Glass Vase LEFT of Brass Lamp + Potted Fern Behind Table Near White Curtain)

| Condition | Strategy | Latency | Peak VRAM | Independent Visual Ground Truth | Satisfied | Preservation vs Original |
|---|---|---:|---:|---|---:|---|
| **A (Sprint 5 Original)** | Immutable baseline | 229.6 s | 7,446 MiB | Lamp LEFT, vase RIGHT (failed); fern behind table near curtain (passed) | **1/2** | Baseline |
| **E1 (Targeted Image Edit)** | FLUX.2 `ReferenceLatent` edit (`is_localized_edit=True`) | **417.4 s** | **6,327 MiB** | **Green vase clearly LEFT, brass lamp clearly RIGHT + EXACT original fern, curtain & wooden table!** | **2/2** | **Near-pixel preservation of satisfied table, fern, curtain & lighting!** |
| **E2 (Full Regeneration)** | `prompt_refinement` (`is_localized_edit=False`) | **226.8 s** | **7,383 MiB** | Green vase LEFT, brass lamp RIGHT + fern behind table near curtain | **2/2** | Low (new boxy table, different plant & lamp, different room corner) |

### 3.3 Failed Approaches, Confounders Caught & Live Guardrail Proof

1. **Confounder caught & fixed:** An initial benchmark pass mistakenly used 4B's distilled `8 steps, CFG 1.2` on `flux-2-klein-base-9b`, causing severe under-convergence on `txt2img` while `ReferenceLatent` edits still converged. Visual inspection caught the confounder immediately; all final trials above were re-run cleanly at production `28 steps, CFG 3.5`.
2. **Live `ResourcePlanner` swap-thrashing guard verification:** During the resumed `adh_03_quantities` session, host swap momentarily crossed `90%` with active paging; `CorrectionCoordinator` recorded `blocked_resource_or_policy` (`Active swap thrashing detected above 90% swap utilization`) and safely retained the original output without crashing.
3. **Inspector false-acceptance / false-uncertainty limitations observed live:**
   - On `mv_01_rear_hair`, adding a 3rd criterion caused `qwen3-vl:8b-instruct` to falsely pass `rear_hair_match` (`3/3`), causing early exit without triggering the `seed_back` correction (though Sprint 5's `mv_03_starter_front_back` already demonstrated live that `seed_back` `ReferenceLatent` conditioning corrects the loose bob to the exact low hair bun).
   - On `s8_adh04_E1_edit`, `qwen3-vl:8b-instruct` returned `Some requirements uncertain` (`0/2`) even though direct visual inspection confirmed `2/2` — reinforcing why Sprint 8 never silently discards candidates solely on uncertain VLM scores and preserves all candidates for human review.

---

## 4. Verification Boundary: Simulated Tests vs Live Trials

- **Simulated & deterministic regression suite ([test_sprint8_correction_engine.py](file:///home/jd/avatar_project/tests/test_sprint8_correction_engine.py)):** Covers all 23 Section 15 test cases (strategy validation, unsupported actions rejected, locked/mandatory requirement preservation, count/text/reference/multi-view/video strategies, 7-factor ranking, regression rejection, retry budget exhaustion, cancellation, execution failure recovery, ResourcePlanner & storage guards, offline enforcement, localized coordinate/state cross-check, inspector disagreement, batched scheduling, frontend & SDK compatibility). Full suite: **401 passed, 0 failed** + `JS syntax OK`.
- **Live GPU & VLM benchmarks ([eval_sprint8_corrections.py](file:///home/jd/avatar_project/scripts/eval_sprint8_corrections.py)):** 13 live `qwen3-vl:8b-instruct` calibration/held-out evaluations (`18/18` on held-out unseen) + 6 live 28-step FLUX.2 9B GPU generations across Conditions A–E (`0` OOM, `9.37 MB` total candidate storage, verified Ollama VRAM release).
