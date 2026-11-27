"""Bounded Automated Corrective Generation Engine (Sprint 8).

Coordinates validated corrective generation strictly using existing Avatar V2 infrastructure:
- Evaluates every decision against the ORIGINAL user requirements in ``GenerationVerificationRecord``
  (never relaxing or rewriting mandatory requirements to appease the inspector).
- Distinguishes verified installed capabilities (FLUX.2 native ``ReferenceLatent`` targeted edit,
  prompt refinement, seed variation, multi-view reference conditioning, keyframe correction,
  Wan 2.2 bounded video regeneration) from unsupported capabilities (dedicated masked inpainting,
  ControlNet pose/composition conditioning, localized video temporal inpainting) — NEVER inventing
  unsupported workflows or auto-downloading models.
- Ranks original and corrected candidates on 7 explicit criteria, rejecting regressions where
  fixing one requirement breaks another previously satisfied mandatory requirement.
- Enforces hard host budgets (default max 2 image retries, max 1 video retry, zero automatic cloud
  escalation), ResourcePlanner admission, predictive storage checks, and clean user cancellation
  at every execution boundary while preserving ALL candidate outputs and provenance.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .inspection_contract import (
    AcceptanceCriterion,
    CorrectionAttemptRecord,
    CorrectionBudget,
    CorrectionRecommendation,
    CorrectionSessionReport,
    CriterionAssessment,
    GenerationVerificationRecord,
    ReferenceProvenanceRecord,
    UI_INSPECTION_LABELS,
    VisualInspectionResult,
    compute_inspection_status,
)
from .model_registry import DEFAULT_REGISTRY, ModelRegistry
from .resource_monitor import ResourceSnapshot, StorageCheck
from .resource_planner import ExecutionPlan, ResourcePlanner, WorkloadRequest
from .visual_inspector import sanitize_untrusted_text

# --------------------------------------------------------------------------- Verified Capability Matrix (Section 3)

SUPPORTED_CORRECTION_STRATEGIES: dict[str, dict[str, Any]] = {
    "masked_localized_edit": {
        "supported": True,
        "is_localized_edit": True,
        "media": ("image",),
        "models": ("flux2_9b", "flux2_4b"),
        "workflow": "builtin:flux2_masked_edit",
        "description": (
            "True mask-conditioned FLUX.2 latent edit (SetLatentNoiseMask + DifferentialDiffusion + "
            "ImageCompositeMasked) guaranteeing zero pixel drift outside the feathered edit mask."
        ),
    },
    "targeted_image_edit": {
        "supported": True,
        "is_localized_edit": True,
        "media": ("image",),
        "models": ("flux2_9b", "flux2_4b"),
        "workflow": "builtin:flux2_txt2img",
        "description": (
            "Unmasked FLUX.2 ReferenceLatent conditioned edit using the prior candidate + preserved "
            "user source references (distinct from pixel-locked masked_localized_edit)."
        ),
    },
    "additional_reference_conditioning": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image",),
        "models": ("flux2_9b", "flux2_4b"),
        "workflow": "builtin:flux2_txt2img",
        "description": "Add orientation-matched subject reference(s) (e.g., back_body, left_side, right_side) via chained ReferenceLatent.",
    },
    "reference_adjustment": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image",),
        "models": ("flux2_9b", "flux2_4b"),
        "workflow": "builtin:flux2_txt2img",
        "description": "Re-prioritize orientation-appropriate user source references without modifying original files.",
    },
    "prompt_refinement": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image", "video"),
        "models": ("flux2_9b", "flux2_4b", "wan22_a14b", "wan22_5b"),
        "workflow": "same",
        "description": "Complete regeneration with refined prompt emphasizing failed requirements while preserving satisfied criteria.",
    },
    "seed_variation": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image", "video"),
        "models": ("flux2_9b", "flux2_4b", "wan22_a14b", "wan22_5b"),
        "workflow": "same",
        "description": "Controlled deterministic seed retry keeping prompt and references intact.",
    },
    "keyframe_correction": {
        "supported": True,
        "is_localized_edit": True,
        "media": ("video",),
        "models": ("flux2_9b", "flux2_4b"),
        "workflow": "builtin:flux2_txt2img",
        "description": "Revise the FLUX.2 ending keyframe before re-running WanFirstLastFrameToVideo.",
    },
    "video_regeneration": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("video",),
        "models": ("wan22_a14b", "wan22_5b"),
        "workflow": "workflows/wan22_i2v_14b_gguf_api.json",
        "description": "Bounded single full-video regeneration (localized temporal video inpainting is not installed).",
    },
    "alternative_approved_workflow": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image", "video"),
        "models": ("flux2_9b", "flux2_4b", "wan22_a14b", "wan22_5b"),
        "workflow": "approved_registry_fallback",
        "description": "Switch to another verified ModelRegistry workflow (e.g., Wan FLF2V or fallback quantization under pressure).",
    },
    "human_review_escalation": {
        "supported": True,
        "is_localized_edit": False,
        "media": ("image", "video"),
        "models": (),
        "workflow": "none",
        "description": "Stop automated retries and request human visual review with supporting inspection evidence.",
    },
    # Explicitly UNSUPPORTED strategies on current installed local models (never fabricated!):
    "inpainting": {
        "supported": False,
        "is_localized_edit": True,
        "media": ("image",),
        "models": (),
        "workflow": "unsupported",
        "limitation_note": "No dedicated masked inpainting model/ControlNet is installed; use FLUX.2 ReferenceLatent targeted_image_edit or defer to Sprint 9.",
    },
    "composition_conditioning": {
        "supported": False,
        "is_localized_edit": False,
        "media": ("image",),
        "models": (),
        "workflow": "unsupported",
        "limitation_note": "No depth/canny/layout ControlNet model is installed in ComfyUI/models/controlnet/.",
    },
    "pose_conditioning": {
        "supported": False,
        "is_localized_edit": False,
        "media": ("image", "video"),
        "models": (),
        "workflow": "unsupported",
        "limitation_note": "No OpenPose/DWPose/FunControl pose conditioning weights are installed.",
    },
}


def validate_correction_strategy(strategy: str, *, media_kind: Literal["image", "video"]) -> tuple[bool, str]:
    """Validate that a proposed correction strategy is genuinely supported by installed local workflows."""
    meta = SUPPORTED_CORRECTION_STRATEGIES.get(str(strategy))
    if meta is None:
        return False, f"Unknown or unapproved correction strategy '{strategy}'."
    if not meta.get("supported"):
        return False, str(meta.get("limitation_note") or f"Correction strategy '{strategy}' is not installed.")
    if media_kind not in meta.get("media", ()):
        return False, f"Correction strategy '{strategy}' does not apply to media kind '{media_kind}'."
    return True, "ok"


# --------------------------------------------------------------------------- Strategy Selection & Intent-Preserving Prompting


@dataclass(frozen=True)
class PlannedCorrectionStep:
    strategy: str
    is_localized_edit: bool
    model_id: str
    workflow: str
    revised_prompt: str
    reference_paths: list[str]
    reference_changes: list[str]
    seed: int
    failed_criterion_ids: list[str]
    supporting_evidence: list[str]
    preserved_satisfied_criteria: list[str]
    expected_improvement: str
    generation_settings: dict[str, Any]
    mask_path: str = ""
    mask_provenance: Literal["", "user", "harness"] = ""
    requires_human_review: bool = False
    stop_reason: str = ""


def _locked_requirements(record: GenerationVerificationRecord) -> list[str]:
    return [c.description for c in record.mandatory_acceptance_criteria if c.locked]


def build_corrective_prompt(
    record: GenerationVerificationRecord,
    *,
    failed_assessments: list[CriterionAssessment],
    satisfied_assessments: list[CriterionAssessment],
    strategy: str,
    is_localized_edit: bool,
) -> str:
    """Build a corrective prompt anchored to ORIGINAL user intent, locked attributes & satisfied criteria.

    Never relaxes or removes a mandatory requirement merely to satisfy the inspector.
    """
    orig = sanitize_untrusted_text(record.original_prompt, max_len=650)
    fixes: list[str] = []
    for a in failed_assessments:
        req = sanitize_untrusted_text(a.original_requirement or a.criterion_id, max_len=160)
        cat = a.category.lower()
        if "quantity" in cat or "count" in cat:
            fixes.append(f"strictly enforce {req} with clear separation between each individual item")
        elif "spatial" in cat:
            fixes.append(f"strictly place elements so that {req} (judged from viewer perspective)")
        elif "typography" in cat or "text" in cat:
            fixes.append(f"render crisp, exact-spelling text satisfying: {req}")
        elif any(tok in cat for tok in ("multiview", "reference", "hair", "clothing", "identity")):
            fixes.append(f"faithfully match the supplied subject reference so that {req}")
        elif "temporal" in cat or "video" in cat:
            fixes.append(f"maintain smooth, continuous environment, lighting, and subject motion satisfying: {req}")
        else:
            fixes.append(f"satisfy {req}")

    preserve_bits = [
        sanitize_untrusted_text(a.original_requirement, max_len=110)
        for a in satisfied_assessments
        if a.original_requirement
    ][:4]
    locked_bits = [sanitize_untrusted_text(x, max_len=110) for x in _locked_requirements(record)][:4]
    preserve_all = list(dict.fromkeys([*locked_bits, *preserve_bits]))

    if is_localized_edit and strategy == "targeted_image_edit":
        parts = [
            "Edit the first reference image (the current candidate) to fix ONLY the following failed requirement(s): "
            + "; ".join(fixes)
            + ".",
            f"Original user request to honor completely: {orig}",
        ]
        if preserve_all:
            parts.append("Keep every already-satisfied element unchanged: " + "; ".join(preserve_all) + ".")
        return " ".join(parts)

    parts = [orig]
    if fixes:
        parts.append("Critical corrective requirements: " + "; ".join(fixes) + ".")
    if preserve_all:
        parts.append("Preserve all locked and previously satisfied elements: " + "; ".join(preserve_all) + ".")
    if record.expected_output.expected_exclusions:
        parts.append("Do not include: " + ", ".join(record.expected_output.expected_exclusions) + ".")
    return " ".join(parts)


def select_correction_step(
    record: GenerationVerificationRecord,
    inspection: VisualInspectionResult,
    *,
    current_best_path: str,
    attempt_number: int,
    prior_strategies: list[str],
    base_seed: int = 41001,
    allow_targeted_edit: bool = True,
    registry: ModelRegistry = DEFAULT_REGISTRY,
    prefer_full_regeneration: bool = False,
    explicit_mask_path: str | None = None,
    mask_provenance: Literal["", "user", "harness"] = "",
) -> PlannedCorrectionStep:
    """Select the least disruptive verified correction strategy for failed mandatory criteria."""
    media_kind = record.expected_output.media_kind
    failed = [a for a in inspection.criterion_assessments if a.mandatory and a.status == "failed"]
    satisfied = [a for a in inspection.criterion_assessments if a.status == "passed"]
    uncertain = [a for a in inspection.criterion_assessments if a.mandatory and a.status in {"uncertain", "unverifiable", "pending"}]

    preserved_ids = [f"{a.criterion_id}: {a.original_requirement}" for a in satisfied]
    failed_ids = [a.criterion_id for a in failed]
    evidence = [
        sanitize_untrusted_text(a.failure_description or a.observed_evidence or a.confidence_limitations, max_len=200)
        for a in (failed or uncertain)
    ]

    # Stop early if there are no failed mandatory requirements (either all passed, or only uncertain needing human review).
    if not failed:
        return PlannedCorrectionStep(
            strategy="human_review_escalation",
            is_localized_edit=False,
            model_id=record.selected_model_id,
            workflow=record.selected_workflow,
            revised_prompt=record.original_prompt,
            reference_paths=[],
            reference_changes=[],
            seed=base_seed,
            failed_criterion_ids=[],
            supporting_evidence=evidence,
            preserved_satisfied_criteria=preserved_ids,
            expected_improvement="Human visual review required for remaining uncertain criteria.",
            generation_settings={},
            requires_human_review=True,
            stop_reason="only_uncertain_criteria_remaining" if uncertain else "all_mandatory_satisfied",
        )

    # Validate recommendations strictly against closed action types.
    recs = list(inspection.correction_recommendations)
    rec_types = {r.action_type for r in recs}

    user_refs = [r for r in record.reference_requirements if r.origin != "generated_intermediate"]
    user_ref_paths = [r.path for r in user_refs if r.path and Path(r.path).expanduser().is_file()]
    user_roles = {r.role for r in user_refs}

    model_spec = registry.get(record.selected_model_id)
    model_id = model_spec.model_id if model_spec else ("flux2_9b" if media_kind == "image" else "wan22_a14b")
    workflow = model_spec.workflow if model_spec else record.selected_workflow
    next_seed = int(base_seed) + attempt_number * 137

    # --- Video correction path (bounded max-1 regeneration using _MEIPASS-aware Wan workflow; RC-12) ---
    # Note: Single-step PNG-only keyframe_correction is disabled because VideoInspector requires an MP4 output.
    if media_kind == "video":
        from . import local_engine

        wan_workflow = str(local_engine.WORKFLOW) if local_engine.WORKFLOW.is_file() else workflow
        strategy = "video_regeneration"
        prompt = build_corrective_prompt(
            record, failed_assessments=failed, satisfied_assessments=satisfied, strategy=strategy, is_localized_edit=False
        )
        return PlannedCorrectionStep(
            strategy=strategy,
            is_localized_edit=False,
            model_id=model_id,
            workflow=wan_workflow,
            revised_prompt=prompt,
            reference_paths=user_ref_paths[:1],  # Never silently replace user starter_image!
            reference_changes=["Preserved original user starter image verbatim for bounded video regeneration."],
            seed=next_seed,
            failed_criterion_ids=failed_ids,
            supporting_evidence=evidence,
            preserved_satisfied_criteria=preserved_ids,
            expected_improvement="Reduce temporal discontinuity via bounded single full-video regeneration.",
            generation_settings={
                "width": record.expected_output.width,
                "height": record.expected_output.height,
                "frames": record.expected_output.frames,
                "seed": next_seed,
            },
        )

    # --- Still image correction path (localized edit priority over full regeneration!) ---
    # 1. Rear / side multi-view mismatch: prioritize orientation-matched subject references!
    if rec_types & {"add_back_body_reference", "add_side_reference", "select_subject_references"}:
        needed_roles: list[str] = []
        for r in recs:
            needed_roles.extend(r.target_reference_roles)
        matched_refs = [r.path for r in user_refs if r.role in needed_roles and Path(r.path).expanduser().is_file()]
        if matched_refs:
            strategy = "additional_reference_conditioning" if "additional_reference_conditioning" not in prior_strategies else "targeted_image_edit"
            is_local = strategy == "targeted_image_edit"
            ref_paths = ([current_best_path] if is_local and Path(current_best_path).is_file() else []) + [
                p for p in matched_refs + user_ref_paths if p != current_best_path
            ]
            ref_paths = list(dict.fromkeys(ref_paths))[:4]
            prompt = build_corrective_prompt(
                record, failed_assessments=failed, satisfied_assessments=satisfied, strategy=strategy, is_localized_edit=is_local
            )
            return PlannedCorrectionStep(
                strategy=strategy,
                is_localized_edit=is_local,
                model_id=model_id,
                workflow=workflow,
                revised_prompt=prompt,
                reference_paths=ref_paths,
                reference_changes=[f"Anchored orientation-matched reference role(s): {', '.join(sorted(set(needed_roles) & user_roles))}."],
                seed=next_seed,
                failed_criterion_ids=failed_ids,
                supporting_evidence=evidence,
                preserved_satisfied_criteria=preserved_ids,
                expected_improvement=f"Restore subject multi-view adherence using {', '.join(sorted(set(needed_roles) & user_roles))}.",
                generation_settings={"width": record.expected_output.width, "height": record.expected_output.height, "frames": 1, "seed": next_seed},
            )
        # Missing required orientation reference (e.g. rear view requested with no back_body ref) -> escalate cleanly!
        if "add_back_body_reference" in rec_types or "add_side_reference" in rec_types:
            return PlannedCorrectionStep(
                strategy="human_review_escalation",
                is_localized_edit=False,
                model_id=model_id,
                workflow=workflow,
                revised_prompt=record.original_prompt,
                reference_paths=user_ref_paths[:4],
                reference_changes=[],
                seed=base_seed,
                failed_criterion_ids=failed_ids,
                supporting_evidence=evidence,
                preserved_satisfied_criteria=preserved_ids,
                expected_improvement="User must supply missing back_body or side reference image for reliable correction.",
                generation_settings={},
                requires_human_review=True,
                stop_reason="missing_required_orientation_reference",
            )

    # 2a. Explicit-mask localized edit ONLY when an explicit user/harness mask file is supplied (RC-12).
    if (
        allow_targeted_edit
        and not prefer_full_regeneration
        and explicit_mask_path
        and Path(explicit_mask_path).expanduser().is_file()
        and "masked_localized_edit" not in prior_strategies
        and bool(current_best_path)
        and Path(current_best_path).expanduser().is_file()
    ):
        strategy = "masked_localized_edit"
        prov: Literal["", "user", "harness"] = mask_provenance or "user"
        ref_paths = [current_best_path] + [p for p in user_ref_paths if p != current_best_path]
        ref_paths = ref_paths[:4]
        prompt = build_corrective_prompt(
            record, failed_assessments=failed, satisfied_assessments=satisfied, strategy="targeted_image_edit", is_localized_edit=True
        )
        return PlannedCorrectionStep(
            strategy=strategy,
            is_localized_edit=True,
            model_id=model_id,
            workflow="builtin:flux2_masked_edit",
            revised_prompt=prompt,
            reference_paths=ref_paths,
            reference_changes=[f"Explicit {prov}-supplied mask applied via builtin:flux2_masked_edit on '{Path(current_best_path).name}'."],
            seed=next_seed,
            failed_criterion_ids=failed_ids,
            supporting_evidence=evidence,
            preserved_satisfied_criteria=preserved_ids,
            expected_improvement="Fix failed requirement(s) strictly inside explicit mask with zero outside-mask pixel drift.",
            generation_settings={"width": record.expected_output.width, "height": record.expected_output.height, "frames": 1, "seed": next_seed, "denoise": 0.72},
            mask_path=str(Path(explicit_mask_path).expanduser().resolve()),
            mask_provenance=prov,
        )

    # 2b. Prefer localized FLUX.2 ReferenceLatent edit FIRST when candidate exists and edit is allowed!
    can_edit = (
        allow_targeted_edit
        and not prefer_full_regeneration
        and "targeted_image_edit" not in prior_strategies
        and bool(current_best_path)
        and Path(current_best_path).expanduser().is_file()
        and (model_spec is None or model_spec.supports_edit)
    )
    if can_edit:
        strategy = "targeted_image_edit"
        ref_paths = [current_best_path] + [p for p in user_ref_paths if p != current_best_path]
        ref_paths = ref_paths[:4]
        prompt = build_corrective_prompt(
            record, failed_assessments=failed, satisfied_assessments=satisfied, strategy=strategy, is_localized_edit=True
        )
        return PlannedCorrectionStep(
            strategy=strategy,
            is_localized_edit=True,
            model_id=model_id,
            workflow=workflow,
            revised_prompt=prompt,
            reference_paths=ref_paths,
            reference_changes=[f"Conditioned FLUX.2 ReferenceLatent on prior candidate '{Path(current_best_path).name}' for localized editing."],
            seed=next_seed,
            failed_criterion_ids=failed_ids,
            supporting_evidence=evidence,
            preserved_satisfied_criteria=preserved_ids,
            expected_improvement="Fix failed requirement(s) via localized ReferenceLatent edit while preserving satisfied scene elements.",
            generation_settings={"width": record.expected_output.width, "height": record.expected_output.height, "frames": 1, "seed": next_seed},
        )

    # 3. Fallback / second attempt: explicit prompt refinement (full regeneration, honestly flagged is_localized_edit=False!).
    strategy = "prompt_refinement" if "prompt_refinement" not in prior_strategies else "seed_variation"
    prompt = build_corrective_prompt(
        record, failed_assessments=failed, satisfied_assessments=satisfied, strategy=strategy, is_localized_edit=False
    )
    return PlannedCorrectionStep(
        strategy=strategy,
        is_localized_edit=False,
        model_id=model_id,
        workflow=workflow,
        revised_prompt=prompt,
        reference_paths=user_ref_paths[:4],
        reference_changes=["Full regeneration from original user references (distinct from targeted editing)."],
        seed=next_seed,
        failed_criterion_ids=failed_ids,
        supporting_evidence=evidence,
        preserved_satisfied_criteria=preserved_ids,
        expected_improvement="Satisfy failed mandatory requirement(s) via refined layout/prompt while preserving original intent.",
        generation_settings={"width": record.expected_output.width, "height": record.expected_output.height, "frames": 1, "seed": next_seed},
    )


# --------------------------------------------------------------------------- 7-Factor Best-Candidate Retention (Section 7)


def score_candidate(
    inspection: VisualInspectionResult,
    record: GenerationVerificationRecord,
    *,
    baseline_passed_ids: set[str] | None = None,
    generation_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a candidate on all 7 Section 7 dimensions without blindly trusting VLM confidence."""
    mandatory = [a for a in inspection.criterion_assessments if a.mandatory]
    passed_ids = {a.criterion_id for a in mandatory if a.status == "passed"}
    failed_ids = {a.criterion_id for a in mandatory if a.status == "failed"}
    uncertain_ids = {a.criterion_id for a in mandatory if a.status in {"uncertain", "unverifiable", "pending"}}

    # 1. Mandatory requirement satisfaction
    mandatory_passed = len(passed_ids)
    mandatory_failed = len(failed_ids)

    # 2. Regression check: previously satisfied mandatory requirements lost + material defects
    lost = sorted((baseline_passed_ids or set()) - passed_ids)
    defect_count = len(inspection.detected_defects)

    # 3. Locked attribute preservation
    locked_ids = {c.criterion_id for c in record.mandatory_acceptance_criteria if c.locked}
    locked_failed = sorted(locked_ids & failed_ids)
    locked_preserved = len(locked_ids) - len(locked_failed)

    # 4. Reference consistency (average lightweight CV color-band similarity if refs exist)
    ref_sims = [
        float(c.get("color_band_similarity"))
        for c in (inspection.reference_comparisons or [])
        if isinstance(c, dict) and isinstance(c.get("color_band_similarity"), (int, float))
    ]
    ref_consistency = round(sum(ref_sims) / len(ref_sims), 3) if ref_sims else 1.0

    # 5. Visual quality proxy (sharpness + absence of structural defects)
    struct = inspection.structural_metrics or {}
    sharpness = float(struct.get("laplacian_sharpness") or 0.0)
    struct_defect_n = len(struct.get("structural_defects") or [])
    visual_quality = round(max(0.0, min(1.0, sharpness / 180.0)) - 0.35 * struct_defect_n, 3)

    # 6. Temporal consistency for video
    vid = inspection.video_summary or {}
    temporal_fails = sum(1 for s in (vid.get("suspicious_intervals") or []) if s.get("severity") == "fail")
    temporal_score = 1.0 if not vid else round(max(0.0, 1.0 - 0.4 * temporal_fails - (0.5 if vid.get("is_effectively_static") else 0.0)), 3)

    # Lexicographic priority tuple (higher is better) matching Section 7's exact 7 criteria!
    # RC-09: Confirmed mandatory passes come FIRST, followed by total unresolved (failed + uncertain),
    # so a timed-out/uncertain reinspection (0 passes, 0 fails, 1 uncertain) NEVER beats the original!
    unresolved_count = mandatory_failed + len(uncertain_ids)
    rank_tuple = (
        mandatory_passed,                     # 1a. Confirmed mandatory passes strictly first
        -unresolved_count,                    # 1b. Fewer total unresolved (failed + uncertain)
        -mandatory_failed,                    # 1c. Fewer confirmed mandatory failures
        -len(locked_failed),                  # 3. Never break locked attributes
        -len(lost),                           # 2a. Penalize regressions on previously satisfied criteria
        -defect_count,                        # 2b. Fewer material defects
        -len(uncertain_ids),                  # 1d. Fewer unresolved uncertainties
        round(ref_consistency, 2),            # 4. Reference consistency
        round(temporal_score, 2),             # 6. Temporal consistency
        round(visual_quality, 2),             # 5. Visual quality
        -round(generation_latency_ms / 1000.0, 1),  # 7. Lower latency tie-breaker
    )
    return {
        "mandatory_passed": mandatory_passed,
        "mandatory_failed": mandatory_failed,
        "mandatory_uncertain": len(uncertain_ids),
        "unresolved_count": unresolved_count,
        "passed_ids": sorted(passed_ids),
        "failed_ids": sorted(failed_ids),
        "uncertain_ids": sorted(uncertain_ids),
        "lost_previously_satisfied": lost,
        "locked_failed": locked_failed,
        "locked_preserved": locked_preserved,
        "defect_count": defect_count,
        "reference_consistency": ref_consistency,
        "visual_quality": visual_quality,
        "temporal_consistency": temporal_score,
        "generation_latency_ms": round(generation_latency_ms, 1),
        "rank_tuple": list(rank_tuple),
    }


def is_candidate_strict_improvement(
    candidate_score: dict[str, Any],
    best_score: dict[str, Any],
) -> tuple[bool, bool, str]:
    """Compare a new candidate against the current best.

    Returns ``(is_better, is_regression, explanation)``.
    Enforces Section 7 & RC-09:
    - A candidate that fixes one requirement but breaks a previously satisfied mandatory
      requirement or locked attribute is NEVER an unqualified improvement.
    - A candidate whose reinspection timed out or turned a failure into ``uncertain`` WITHOUT
      a confirmed gain in passed mandatory criteria NEVER replaces the previous best.
    """
    lost = candidate_score.get("lost_previously_satisfied") or []
    locked_failed = candidate_score.get("locked_failed") or []
    if locked_failed:
        return (
            False,
            True,
            f"Rejected regression: broke locked attribute(s) {', '.join(locked_failed)}.",
        )
    if lost and candidate_score["mandatory_passed"] <= best_score["mandatory_passed"]:
        return (
            False,
            True,
            f"Rejected regression: lost previously satisfied requirement(s) {', '.join(lost)} without net mandatory gain.",
        )
    if lost and candidate_score["mandatory_failed"] >= best_score["mandatory_failed"]:
        return (
            False,
            True,
            f"Rejected regression: introduced failure on {', '.join(lost)} (not an unqualified improvement).",
        )
    # RC-09: Never treat failure -> uncertain (e.g. VLM timeout/unavailable) as an improvement!
    cand_passed = int(candidate_score.get("mandatory_passed") or 0)
    best_passed = int(best_score.get("mandatory_passed") or 0)
    cand_unc = int(candidate_score.get("mandatory_uncertain") or 0)
    best_unc = int(best_score.get("mandatory_uncertain") or 0)
    if cand_passed <= best_passed and cand_unc > best_unc:
        return (
            False,
            False,
            "Reinspection increased uncertainty without confirmed mandatory pass gain; retaining previous best.",
        )
    if tuple(candidate_score["rank_tuple"]) > tuple(best_score["rank_tuple"]):
        if lost:
            return (
                False,
                True,
                f"Rejected partial trade-off: broke previously satisfied {', '.join(lost)}; retaining previous best.",
            )
        return (
            True,
            False,
            f"Improved mandatory satisfaction ({best_score['mandatory_passed']} -> {candidate_score['mandatory_passed']}, "
            f"failures {best_score['mandatory_failed']} -> {candidate_score['mandatory_failed']}).",
        )
    return (
        False,
        False,
        "Did not surpass current best candidate; retaining previous best.",
    )


# --------------------------------------------------------------------------- Bounded Correction Coordinator (Sections 6, 8, 9)

GenerateFn = Callable[[PlannedCorrectionStep], tuple[str, float]]
InspectFn = Callable[[str, GenerationVerificationRecord], VisualInspectionResult]


class CorrectionCoordinator:
    """Host-enforced bounded correction loop integrated with ResourcePlanner & VisualInspector."""

    def __init__(
        self,
        *,
        planner: ResourcePlanner | None = None,
        registry: ModelRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.planner = planner or ResourcePlanner(registry=registry)
        self.registry = registry

    def run_bounded_correction(
        self,
        *,
        record: GenerationVerificationRecord,
        original_output_id: str,
        original_output_path: str,
        initial_inspection: VisualInspectionResult,
        generate_fn: GenerateFn,
        inspect_fn: InspectFn,
        base_seed: int = 41001,
        snapshot_fn: Callable[[], ResourceSnapshot] | None = None,
        storage_check_fn: Callable[[], StorageCheck] | None = None,
        cancel_event: threading.Event | None = None,
        status_callback: Callable[[str], None] | None = None,
        prefer_full_regeneration: bool = False,
        current_model_id: str | None = None,
        explicit_mask_path: str | None = None,
        mask_provenance: Literal["", "user", "harness"] = "",
    ) -> CorrectionSessionReport:
        """Execute up to ``budget.limit_for_media(media_kind)`` validated corrections and retain the best candidate."""
        t_session = time.monotonic()
        media_kind = record.expected_output.media_kind
        budget: CorrectionBudget = record.correction_budget

        # Security guardrail: cloud escalation is NEVER enabled automatically!
        if budget.allow_cloud_escalation:
            budget = budget.model_copy(update={"allow_cloud_escalation": False})

        limit = budget.limit_for_media(media_kind)
        total_mandatory = sum(1 for c in record.mandatory_acceptance_criteria if c.mandatory)
        best_id = original_output_id
        best_path = original_output_path
        best_inspection = initial_inspection
        best_score = score_candidate(initial_inspection, record, generation_latency_ms=0.0)
        initial_passed = best_score["mandatory_passed"]
        best_passed_ids = set(best_score["passed_ids"])

        all_candidates = [original_output_path]
        attempts: list[CorrectionAttemptRecord] = []
        prior_strategies: list[str] = []
        stopped_reason = ""
        cancelled = False
        best_reason = "Original generation retained."

        # Check early exit before doing any work.
        if best_score["mandatory_failed"] == 0 and not initial_inspection.detected_defects:
            stopped_reason = (
                "all_verifiable_mandatory_satisfied"
                if best_score["mandatory_uncertain"] == 0
                else "no_failed_mandatory_only_uncertain"
            )
        elif limit <= 0:
            stopped_reason = "correction_budget_zero"

        active_model = current_model_id or record.selected_model_id

        for attempt_num in range(1, limit + 1):
            if stopped_reason:
                break
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                stopped_reason = "cancelled_by_user"
                break

            step = select_correction_step(
                record,
                best_inspection,
                current_best_path=best_path,
                attempt_number=attempt_num,
                prior_strategies=prior_strategies,
                base_seed=base_seed,
                allow_targeted_edit=budget.allow_targeted_edit,
                registry=self.registry,
                prefer_full_regeneration=prefer_full_regeneration,
                explicit_mask_path=explicit_mask_path,
                mask_provenance=mask_provenance,
            )

            # Unsupported or human-review escalation stops cleanly without wasting GPU!
            ok_strat, strat_err = validate_correction_strategy(step.strategy, media_kind=media_kind)
            if not ok_strat or step.requires_human_review or step.strategy == "human_review_escalation":
                stopped_reason = step.stop_reason or ("unsupported_strategy" if not ok_strat else "escalated_human_review")
                attempts.append(
                    CorrectionAttemptRecord(
                        attempt_number=attempt_num,
                        original_output_id=original_output_id,
                        original_output_path=original_output_path,
                        failed_criterion_ids=step.failed_criterion_ids,
                        supporting_evidence=step.supporting_evidence,
                        selected_strategy=step.strategy if ok_strat else "human_review_escalation",  # type: ignore[arg-type]
                        is_localized_edit=False,
                        original_prompt=record.original_prompt,
                        revised_prompt=step.revised_prompt,
                        reference_changes=step.reference_changes,
                        model_id=step.model_id,
                        workflow=step.workflow,
                        generation_settings=step.generation_settings,
                        preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                        expected_improvement=step.expected_improvement,
                        final_disposition="escalated_human_review",
                        notes=[strat_err if not ok_strat else stopped_reason],
                    )
                )
                break

            # ResourcePlanner admission + predictive storage check BEFORE EVERY correction attempt!
            snap = snapshot_fn() if snapshot_fn is not None else None
            s_check = storage_check_fn() if storage_check_fn is not None else None
            w_req = WorkloadRequest(
                model_id=step.model_id,
                workflow=step.workflow,
                width=int(step.generation_settings.get("width") or record.expected_output.width),
                height=int(step.generation_settings.get("height") or record.expected_output.height),
                frames=int(step.generation_settings.get("frames") or record.expected_output.frames),
                local_only=True,
                has_reference_assets=bool(step.reference_paths),
            )
            r_plan: ExecutionPlan = self.planner.plan_execution(
                w_req,
                current_model_id=active_model,
                snapshot=snap,
                storage_check=s_check,
            )
            if not r_plan.admitted:
                stopped_reason = f"resource_admission_blocked: {r_plan.admission_reason}"
                attempts.append(
                    CorrectionAttemptRecord(
                        attempt_number=attempt_num,
                        original_output_id=original_output_id,
                        original_output_path=original_output_path,
                        failed_criterion_ids=step.failed_criterion_ids,
                        supporting_evidence=step.supporting_evidence,
                        selected_strategy=step.strategy,  # type: ignore[arg-type]
                        is_localized_edit=step.is_localized_edit,
                        original_prompt=record.original_prompt,
                        revised_prompt=step.revised_prompt,
                        reference_changes=step.reference_changes,
                        model_id=step.model_id,
                        workflow=step.workflow,
                        generation_settings=step.generation_settings,
                        resource_requirements=r_plan.public(),
                        preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                        expected_improvement=step.expected_improvement,
                        final_disposition="blocked_resource_or_policy",
                        notes=[r_plan.admission_reason],
                    )
                )
                break

            prior_strategies.append(step.strategy)
            if status_callback:
                status_callback("Improving result")

            # Execute bounded corrective generation (preserving original output untouched!).
            cand_id = f"{original_output_id}_cor{attempt_num}_{secrets.token_hex(2)}"
            t_gen = time.monotonic()
            try:
                cand_path, reported_gen_ms = generate_fn(step)
                gen_ms = reported_gen_ms if reported_gen_ms > 0 else round((time.monotonic() - t_gen) * 1000.0, 1)
            except Exception as exc:
                attempts.append(
                    CorrectionAttemptRecord(
                        attempt_number=attempt_num,
                        original_output_id=original_output_id,
                        original_output_path=original_output_path,
                        candidate_output_id=cand_id,
                        failed_criterion_ids=step.failed_criterion_ids,
                        supporting_evidence=step.supporting_evidence,
                        selected_strategy=step.strategy,  # type: ignore[arg-type]
                        is_localized_edit=step.is_localized_edit,
                        original_prompt=record.original_prompt,
                        revised_prompt=step.revised_prompt,
                        reference_changes=step.reference_changes,
                        model_id=step.model_id,
                        workflow=step.workflow,
                        generation_settings=step.generation_settings,
                        resource_requirements=r_plan.public(),
                        preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                        expected_improvement=step.expected_improvement,
                        final_disposition="failed_execution",
                        generation_latency_ms=round((time.monotonic() - t_gen) * 1000.0, 1),
                        notes=[f"Corrective generation failed cleanly: {type(exc).__name__}: {exc}"],
                    )
                )
                active_model = None
                continue

            active_model = step.model_id
            if cand_path and cand_path not in all_candidates:
                all_candidates.append(cand_path)

            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                stopped_reason = "cancelled_by_user_after_generation"
                attempts.append(
                    CorrectionAttemptRecord(
                        attempt_number=attempt_num,
                        original_output_id=original_output_id,
                        original_output_path=original_output_path,
                        candidate_output_id=cand_id,
                        candidate_output_path=cand_path,
                        failed_criterion_ids=step.failed_criterion_ids,
                        supporting_evidence=step.supporting_evidence,
                        selected_strategy=step.strategy,  # type: ignore[arg-type]
                        is_localized_edit=step.is_localized_edit,
                        original_prompt=record.original_prompt,
                        revised_prompt=step.revised_prompt,
                        reference_changes=step.reference_changes,
                        model_id=step.model_id,
                        workflow=step.workflow,
                        generation_settings=step.generation_settings,
                        resource_requirements=r_plan.public(),
                        preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                        expected_improvement=step.expected_improvement,
                        final_disposition="cancelled",
                        generation_latency_ms=gen_ms,
                    )
                )
                break

            # Reinspect corrected output against ALL mandatory requirements in record!
            if status_callback:
                status_callback("Checking quality")
            t_ins = time.monotonic()
            try:
                cand_inspection = inspect_fn(cand_path, record)
            except Exception as exc:
                ins_ms = round((time.monotonic() - t_ins) * 1000.0, 1)
                attempts.append(
                    CorrectionAttemptRecord(
                        attempt_number=attempt_num,
                        original_output_id=original_output_id,
                        original_output_path=original_output_path,
                        candidate_output_id=cand_id,
                        candidate_output_path=cand_path,
                        failed_criterion_ids=step.failed_criterion_ids,
                        supporting_evidence=step.supporting_evidence,
                        selected_strategy=step.strategy,  # type: ignore[arg-type]
                        is_localized_edit=step.is_localized_edit,
                        original_prompt=record.original_prompt,
                        revised_prompt=step.revised_prompt,
                        reference_changes=step.reference_changes,
                        model_id=step.model_id,
                        workflow=step.workflow,
                        generation_settings=step.generation_settings,
                        resource_requirements=r_plan.public(),
                        preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                        expected_improvement=step.expected_improvement,
                        final_disposition="retained_previous_best",
                        generation_latency_ms=gen_ms,
                        inspection_latency_ms=ins_ms,
                        notes=[f"Reinspection failed cleanly ({type(exc).__name__}: {exc}); candidate preserved and previous best retained."],
                    )
                )
                continue
            ins_ms = round((time.monotonic() - t_ins) * 1000.0, 1)

            cand_score = score_candidate(
                cand_inspection,
                record,
                baseline_passed_ids=best_passed_ids,
                generation_latency_ms=gen_ms,
            )
            new_defects = [d for d in cand_inspection.detected_defects if d not in best_inspection.detected_defects]
            better, is_regression, cmp_reason = is_candidate_strict_improvement(cand_score, best_score)

            if better:
                disposition: Any = "selected_as_best"
                best_id = cand_id
                best_path = cand_path
                best_inspection = cand_inspection
                best_score = cand_score
                best_passed_ids = set(cand_score["passed_ids"])
                best_reason = cmp_reason
            elif is_regression:
                disposition = "rejected_regression"
            else:
                disposition = "retained_previous_best"

            attempts.append(
                CorrectionAttemptRecord(
                    attempt_number=attempt_num,
                    original_output_id=original_output_id,
                    original_output_path=original_output_path,
                    candidate_output_id=cand_id,
                    candidate_output_path=cand_path,
                    failed_criterion_ids=step.failed_criterion_ids,
                    supporting_evidence=step.supporting_evidence,
                    selected_strategy=step.strategy,  # type: ignore[arg-type]
                    is_localized_edit=step.is_localized_edit,
                    original_prompt=record.original_prompt,
                    revised_prompt=step.revised_prompt,
                    reference_changes=step.reference_changes,
                    model_id=step.model_id,
                    workflow=step.workflow,
                    generation_settings=step.generation_settings,
                    resource_requirements=r_plan.public(),
                    preserved_satisfied_criteria=step.preserved_satisfied_criteria,
                    expected_improvement=step.expected_improvement,
                    actual_inspection_result=cand_inspection,
                    satisfied_criteria_lost=list(cand_score["lost_previously_satisfied"]),
                    newly_introduced_defects=new_defects,
                    candidate_rank_score=cand_score,
                    final_disposition=disposition,
                    generation_latency_ms=gen_ms,
                    inspection_latency_ms=ins_ms,
                    notes=[cmp_reason],
                )
            )

            # Early termination if all verifiable mandatory requirements are now satisfied!
            if best_score["mandatory_failed"] == 0 and not best_inspection.detected_defects:
                stopped_reason = (
                    "all_verifiable_mandatory_satisfied"
                    if best_score["mandatory_uncertain"] == 0
                    else "remaining_criteria_uncertain"
                )
                break

        if not stopped_reason:
            stopped_reason = "correction_budget_exhausted"

        # Determine final honest UI status (NEVER imply full verification when mandatory criteria remain uncertain/failed!).
        unresolved: list[str] = list(best_inspection.limitations)
        for a in best_inspection.criterion_assessments:
            if a.mandatory and a.status == "failed":
                unresolved.append(f"Unresolved [{a.criterion_id}]: {a.failure_description or a.original_requirement}")
            elif a.mandatory and a.status in {"uncertain", "unverifiable", "pending"}:
                unresolved.append(f"Uncertain [{a.criterion_id}]: {a.confidence_limitations or a.original_requirement}")

        # RC-08: Route ALL final session statuses strictly through compute_inspection_status() so
        # review_recommended or human_review_required=True can NEVER be upgraded to requirements_verified!
        _best_dec, best_ui_st, best_ui_lbl, best_human_req = compute_inspection_status(
            best_inspection.criterion_assessments,
            detected_defects=best_inspection.detected_defects,
        )
        human_req = bool(
            best_human_req
            or best_inspection.human_review_required
            or best_score["mandatory_failed"] > 0
            or best_score["mandatory_uncertain"] > 0
        )
        if best_ui_st == "correction_recommended":
            ui_st: Any = "review_recommended"
            ui_lbl = UI_INSPECTION_LABELS["review_recommended"]
        elif best_ui_st == "requirements_verified" and not human_req and best_path != original_output_path:
            ui_st = "correction_complete"
            ui_lbl = UI_INSPECTION_LABELS["correction_complete"]
        elif best_ui_st == "requirements_verified" and human_req:
            ui_st = "review_recommended"
            ui_lbl = UI_INSPECTION_LABELS["review_recommended"]
        else:
            ui_st = best_ui_st
            ui_lbl = best_ui_lbl

        return CorrectionSessionReport(
            media_kind=media_kind,
            original_output_id=original_output_id,
            original_output_path=original_output_path,
            all_candidate_paths=all_candidates,
            best_candidate_id=best_id,
            best_candidate_path=best_path,
            best_candidate_is_original=(best_path == original_output_path),
            best_candidate_reason=best_reason,
            initial_mandatory_passed=initial_passed,
            final_mandatory_passed=best_score["mandatory_passed"],
            total_mandatory=total_mandatory,
            attempts=attempts,
            attempts_executed=sum(1 for a in attempts if a.final_disposition not in {"escalated_human_review", "blocked_resource_or_policy"}),
            budget_limit=limit,
            stopped_reason=stopped_reason,
            cancelled=cancelled,
            ui_status=ui_st,
            ui_status_label=ui_lbl,
            unresolved_limitations=list(dict.fromkeys(unresolved)),
            human_review_required=human_req,
            total_latency_ms=round((time.monotonic() - t_session) * 1000.0, 1),
        )


def plan_batched_inspection_and_correction(
    items: list[dict[str, Any]],
    *,
    warm_model_id: str | None = "flux2_9b",
) -> dict[str, Any]:
    """Schedule batch generation -> single unload -> sequential GPU VLM inspection -> warm batch correction.

    Minimizes FLUX/Wan eviction-reload cycles (~20s per cycle on RTX 4070 8 GB) compared to
    interleaving GPU VLM inspection after every single image.
    """
    n = max(len(items), 1)
    interleaved_evictions = n * 2  # unload FLUX + unload VLM per item
    batched_evictions = 2          # 1 unload after generation batch + 1 unload after VLM inspection batch
    estimated_reload_savings_s = max(0, (interleaved_evictions - batched_evictions) // 2) * 20
    return {
        "batch_size": len(items),
        "warm_model_id": warm_model_id,
        "phases": [
            {"phase": 1, "name": "cpu_deterministic_prefilter", "vram_eviction": False},
            {"phase": 2, "name": "single_comfy_unload_then_batch_gpu_vlm", "vram_eviction": True, "verify_vram_release": True},
            {"phase": 3, "name": "warm_model_batch_corrections", "vram_eviction": False, "retain_model": warm_model_id},
        ],
        "interleaved_evictions": interleaved_evictions,
        "batched_evictions": batched_evictions,
        "estimated_reload_savings_s": estimated_reload_savings_s,
    }
