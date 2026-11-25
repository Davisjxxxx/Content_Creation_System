"""Quality-First Capability-Aware Workflow Router, Reference Optimizer & Multi-Stage Planner (Sprint 6).

Coordinates:
- Interpreted ``StructuredPromptSpec`` (from ``intent_interpreter.py``)
- Component-aware ``ModelRegistry`` (quality-first defaults: FLUX.2 9B, Wan 2.2 14B)
- Intelligent reference deduplication, quality filtering, relevance selection & provisional pixel budget
- Single-stage vs multi-stage planning (e.g. FLUX.2 keyframe synthesis -> Wan 2.2 First-Last-Frame)
- ``ResourcePlanner`` admission & ``compile_prompt_for_model`` validation
- Machine-readable ``GenerationVerificationRecord`` generation for Sprint 7/8 visual QA & correction

Security & Reliability:
- Outputs strictly validated Pydantic execution plans (NEVER arbitrary Python or unverified workflows).
- Never allows local LLM suggestions to invent models, bypass ResourcePlanner, or enable cloud billing.
- Preserves user-supplied starter images verbatim (never silently rewrites them).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .inspection_contract import (
    AcceptanceCriterion,
    CorrectionBudget,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    ReferenceProvenanceRecord,
)
from .intent_interpreter import InterpretedRequest, interpret_natural_language
from .model_registry import DEFAULT_REGISTRY, ModelRegistry, ModelSpec
from .prompt_compilers import compile_prompt_for_model
from .reference_picker import validate_local_image
from .resource_monitor import ResourceSnapshot, StorageCheck
from .resource_planner import ExecutionPlan, ResourcePlanner, WorkloadRequest
from .subject_references import (
    SubjectReferenceItem,
    SubjectReferenceSet,
    select_relevant_references,
)

# Provisional adaptive reference pixel budgets (calibrated from live RTX 4070 8 GB scaling;
# explicitly flagged as provisional pending further quality-vs-latency human evaluation).
PROVISIONAL_REF_MEGAPIXELS: dict[int, float] = {
    1: 1.0,
    2: 0.65,
    3: 0.5,
    4: 0.5,
}


class ExecutionStage(BaseModel):
    """One validated, executable stage inside a GenerationPlan."""

    stage_index: int
    stage_id: str
    purpose: Literal[
        "still_generate",
        "still_multiref",
        "still_edit",
        "starter_keyframe",
        "end_keyframe",
        "video_i2v",
        "video_flf2v",
    ]
    model_id: str
    workflow: str
    compiled_positive: str
    compiled_negative: str = ""
    width: int
    height: int
    frames: int = 1
    reference_paths: list[str] = Field(default_factory=list)
    reference_megapixels: float | None = None
    pixel_budget_provisional: bool = True
    consumes_stage_output: list[str] = Field(default_factory=list)
    produces_asset_origin: Literal["final_output", "generated_intermediate"] = "final_output"
    resource_plan: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RoutedGenerationPlan(BaseModel):
    """Validated single-stage or multi-stage plan produced by the offline intelligence router."""

    ok: bool
    detected_intent: str
    primary_model_id: str
    primary_workflow: str
    is_multi_stage: bool
    stages: list[ExecutionStage] = Field(default_factory=list)
    reference_optimization: dict[str, Any] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    verification_record: GenerationVerificationRecord | None = None
    error: str = ""

    def public(self) -> dict[str, Any]:
        return self.model_dump()


def _provisional_mp(ref_count: int) -> float | None:
    if ref_count <= 0:
        return None
    return PROVISIONAL_REF_MEGAPIXELS.get(min(ref_count, 4), 0.5)


def optimize_subject_references(
    ref_set: SubjectReferenceSet | None,
    *,
    prompt: str,
    task: str,
    subject_id: str | None = None,
    max_supplemental: int = 4,
    workflow_supports_multiref: bool = True,
    check_files: bool = False,
) -> dict[str, Any]:
    """Deduplicate, quality-check, and select only the references actually needed for the request."""
    if ref_set is None or (ref_set.starter_image is None and not ref_set.references):
        return {
            "scenario": "general",
            "starter_image": None,
            "selected": [],
            "deduplicated": [],
            "low_quality_warnings": [],
            "omitted": [],
            "recommended_strategy": "no_references",
            "provisional_ref_megapixels": None,
            "workflow_limitations": [],
        }

    # 1. Deduplicate redundant references by SHA-256 fingerprint or resolved path (preserving locked first).
    deduped_items: list[SubjectReferenceItem] = []
    removed_dupes: list[dict[str, str]] = []
    low_quality_warnings: list[str] = []
    seen_keys: set[tuple[str, str]] = set()

    ordered_input = sorted(ref_set.references, key=lambda r: (0 if r.locked else 1))
    for item in ordered_input:
        fp = item.asset_fingerprint
        if check_files and Path(item.path).is_file():
            checked = validate_local_image(item.path, workflow="multi_reference")
            if checked.ok:
                fp = fp or checked.sha256
                if min(checked.width, checked.height) < 384:
                    low_quality_warnings.append(
                        f"Reference '{Path(item.path).name}' ({item.role}) is low-res ({checked.width}x{checked.height})."
                    )
        dedupe_key = (item.subject_id, fp or str(Path(item.path)))
        if dedupe_key in seen_keys and not item.locked:
            removed_dupes.append({"role": item.role, "path": item.path, "reason": "Duplicate asset fingerprint/path"})
            continue
        seen_keys.add(dedupe_key)
        deduped_items.append(item.model_copy(update={"asset_fingerprint": fp}))

    cleaned_set = ref_set.model_copy(update={"references": deduped_items})
    sel = select_relevant_references(
        cleaned_set,
        prompt=prompt,
        task=task,
        subject_id=subject_id,
        max_supplemental=max_supplemental,
        workflow_supports_multiref=workflow_supports_multiref,
    )
    selected = list(sel["selected"])
    ref_count = len(selected) + (1 if ref_set.starter_image is not None and task != "video_generate" else 0)
    return {
        **sel,
        "deduplicated": removed_dupes,
        "low_quality_warnings": low_quality_warnings,
        "provisional_ref_megapixels": _provisional_mp(ref_count),
    }


def _select_quality_first_model(
    *,
    is_video: bool,
    requested_model: str | None,
    registry: ModelRegistry,
    comfyui_dir: Path | None = None,
    allow_cloud: bool = False,
) -> tuple[ModelSpec | None, str]:
    """Quality-first model selection: FLUX.2 9B for stills, Wan 2.2 14B for video unless overridden/missing."""
    pref = str(requested_model or "").strip()
    if pref and pref != "auto":
        if pref.lower().startswith("runway") and not allow_cloud:
            return None, "Cloud model requested without explicit host cloud authorization."
        spec = registry.get(pref)
        if spec is None:
            return None, f"Requested model '{pref}' is not registered in Avatar V2's ModelRegistry."
        return spec, f"Honored explicit model preference '{spec.model_id}'."

    priority = ("wan22_a14b", "wan22_5b") if is_video else ("flux2_9b", "flux2_4b")
    for mid in priority:
        spec = registry.get(mid)
        if spec is None:
            continue
        if comfyui_dir is not None:
            missing = [c.filename for c in spec.components if not c.exists_in(comfyui_dir)]
            if missing:
                continue
        reason = (
            f"Quality-first default selected '{spec.model_id}' ({spec.label})."
            if mid == priority[0]
            else f"Primary quality model '{priority[0]}' unavailable on disk; fell back to '{spec.model_id}'."
        )
        return spec, reason

    fallback = registry.get(priority[0])
    return fallback, f"Defaulted to '{priority[0]}' (disk inventory unchecked)."


class WorkflowRouter:
    """Validates and builds deterministic single-stage or multi-stage GenerationPlans."""

    def __init__(
        self,
        *,
        registry: ModelRegistry = DEFAULT_REGISTRY,
        planner: ResourcePlanner | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or ResourcePlanner(registry=registry)

    def route(
        self,
        interpreted: InterpretedRequest,
        *,
        starter_image: str | None = None,
        subject_refs: SubjectReferenceSet | None = None,
        current_model_id: str | None = None,
        snapshot: ResourceSnapshot | None = None,
        storage_check: StorageCheck | None = None,
        comfyui_dir: Path | None = None,
        allow_cloud: bool = False,
        check_ref_files: bool = False,
    ) -> RoutedGenerationPlan:
        """Produce a validated execution plan + Sprint 7/8 GenerationVerificationRecord."""
        if not interpreted.ok:
            return RoutedGenerationPlan(
                ok=False,
                detected_intent=interpreted.detected_intent,
                primary_model_id="",
                primary_workflow="",
                is_multi_stage=False,
                error=interpreted.error or "Intent interpretation failed validation.",
                warnings=list(interpreted.warnings),
            )

        spec = interpreted.spec
        if not spec.constraints.local_only and not allow_cloud:
            return RoutedGenerationPlan(
                ok=False,
                detected_intent=interpreted.detected_intent,
                primary_model_id="",
                primary_workflow="",
                is_multi_stage=False,
                error="Local-only execution enforced: cloud inference requires explicit host authorization.",
            )

        # Check reference set conflicts before planning.
        if subject_refs is not None:
            ref_conflicts = [c for c in subject_refs.detect_conflicts(check_disk=check_ref_files) if c.severity == "error"]
            if ref_conflicts:
                return RoutedGenerationPlan(
                    ok=False,
                    detected_intent=interpreted.detected_intent,
                    primary_model_id="",
                    primary_workflow="",
                    is_multi_stage=False,
                    error=ref_conflicts[0].message,
                )

        is_video = spec.intent.task in {"video_generate", "video_scene"}
        rationale: list[str] = []
        warnings: list[str] = list(interpreted.warnings)

        # Pick quality-first primary model (FLUX.2 9B for stills; Wan 2.2 14B for video).
        primary_spec, model_reason = _select_quality_first_model(
            is_video=is_video,
            requested_model=spec.constraints.model_preference,
            registry=self.registry,
            comfyui_dir=comfyui_dir,
            allow_cloud=allow_cloud,
        )
        if primary_spec is None:
            return RoutedGenerationPlan(
                ok=False,
                detected_intent=interpreted.detected_intent,
                primary_model_id="",
                primary_workflow="",
                is_multi_stage=False,
                error=model_reason,
            )
        rationale.append(model_reason)

        # Optimize references intelligently.
        ref_opt = optimize_subject_references(
            subject_refs,
            prompt=spec.original_prompt,
            task=spec.intent.task,
            max_supplemental=4,
            workflow_supports_multiref=(not is_video),
            check_files=check_ref_files,
        )
        warnings.extend(ref_opt.get("low_quality_warnings") or [])
        selected_supp_paths = [str(r["path"]) for r in ref_opt.get("selected") or []]
        has_starter = bool(starter_image and str(starter_image).strip())
        has_rear_supp = any(r.get("role") in {"back_body", "rear_three_quarter"} for r in (ref_opt.get("selected") or []))

        stages: list[ExecutionStage] = []
        intermediate_records: list[ReferenceProvenanceRecord] = []

        # Helper to validate one stage via prompt compiler + ResourcePlanner.
        def _build_stage(
            *,
            idx: int,
            stage_id: str,
            purpose: Any,
            model_spec: ModelSpec,
            stage_spec: Any,
            ref_paths: list[str],
            prev_model: str | None,
            consumes: list[str] | None = None,
            origin: Literal["final_output", "generated_intermediate"] = "final_output",
        ) -> tuple[ExecutionStage | None, str]:
            compiled = compile_prompt_for_model(stage_spec, model_spec.model_id, registry=self.registry)
            if not compiled.ok:
                err_msg = (compiled.unsupported_capabilities or [c["message"] for c in compiled.conflicts if c.get("severity") == "error"])[0]
                return None, err_msg
            w = int(stage_spec.constraints.width or model_spec.default_width)
            h = int(stage_spec.constraints.height or model_spec.default_height)
            f = int(stage_spec.constraints.frames or model_spec.default_frames)
            w_req = WorkloadRequest(
                model_id=model_spec.model_id,
                workflow=model_spec.workflow,
                width=w,
                height=h,
                frames=f,
                local_only=not allow_cloud,
                has_reference_assets=bool(ref_paths),
            )
            r_plan: ExecutionPlan = self.planner.plan_execution(
                w_req,
                current_model_id=prev_model,
                snapshot=snapshot,
                storage_check=storage_check,
                comfyui_dir=comfyui_dir,
            )
            if not r_plan.admitted:
                return None, r_plan.admission_reason
            st = ExecutionStage(
                stage_index=idx,
                stage_id=stage_id,
                purpose=purpose,
                model_id=model_spec.model_id,
                workflow=model_spec.workflow,
                compiled_positive=compiled.positive,
                compiled_negative=compiled.negative,
                width=w,
                height=h,
                frames=f,
                reference_paths=list(ref_paths),
                reference_megapixels=_provisional_mp(len(ref_paths)) if model_spec.family == "flux2" and ref_paths else None,
                consumes_stage_output=list(consumes or []),
                produces_asset_origin=origin,
                resource_plan=r_plan.public(),
                warnings=list(compiled.warnings),
            )
            return st, ""

        # --- Case 1: Still image generation / editing / multi-reference ---
        if not is_video:
            combined_refs: list[str] = []
            if has_starter:
                combined_refs.append(str(starter_image))
            for p in selected_supp_paths:
                if p not in combined_refs:
                    combined_refs.append(p)
            combined_refs = combined_refs[:4]
            purpose: Any = (
                "still_edit"
                if spec.intent.task == "image_edit"
                else ("still_multiref" if combined_refs else "still_generate")
            )
            st, err = _build_stage(
                idx=1,
                stage_id="stage_1_still",
                purpose=purpose,
                model_spec=primary_spec,
                stage_spec=spec,
                ref_paths=combined_refs,
                prev_model=current_model_id,
            )
            if st is None:
                return RoutedGenerationPlan(
                    ok=False,
                    detected_intent=interpreted.detected_intent,
                    primary_model_id=primary_spec.model_id,
                    primary_workflow=primary_spec.workflow,
                    is_multi_stage=False,
                    error=err,
                )
            stages.append(st)
            warnings.extend(st.warnings)
            rationale.append(
                f"Single-stage still workflow ({purpose}) using {primary_spec.model_id} with {len(combined_refs)} reference(s)."
            )

        # --- Case 2: Video generation (single-stage I2V vs multi-stage FLUX keyframe + Wan I2V/FLF2V) ---
        else:
            needs_flf_turn = (
                interpreted.detected_intent == "flf_video_generate"
                or (ref_opt.get("recommended_strategy") == "flux_keyframe_plus_wan_flf2v" and has_rear_supp)
            )
            still_model, still_reason = _select_quality_first_model(
                is_video=False,
                requested_model="auto",
                registry=self.registry,
                comfyui_dir=comfyui_dir,
                allow_cloud=allow_cloud,
            )
            prev_m = current_model_id
            consumes_for_video: list[str] = []

            # Stage A: Synthesize starter image ONLY when the user did NOT supply a starter_image!
            # (NEVER silently alter an explicitly supplied user starter image!)
            if not has_starter:
                if still_model is None:
                    return RoutedGenerationPlan(
                        ok=False,
                        detected_intent=interpreted.detected_intent,
                        primary_model_id=primary_spec.model_id,
                        primary_workflow=primary_spec.workflow,
                        is_multi_stage=True,
                        error=f"Text-to-video requires a local FLUX.2 still model for starter keyframe synthesis: {still_reason}",
                    )
                starter_spec = spec.model_copy(deep=True)
                starter_spec.intent.task = "image_generate"
                starter_spec.motion.state = "automatic"
                starter_spec.camera_movement.state = "automatic"
                starter_spec.constraints = starter_spec.constraints.model_copy(update={"width": 896, "height": 1280, "frames": 1})
                st_start, err = _build_stage(
                    idx=len(stages) + 1,
                    stage_id="stage_starter_keyframe",
                    purpose="starter_keyframe",
                    model_spec=still_model,
                    stage_spec=starter_spec,
                    ref_paths=selected_supp_paths[:3],
                    prev_model=prev_m,
                    origin="generated_intermediate",
                )
                if st_start is None:
                    return RoutedGenerationPlan(
                        ok=False,
                        detected_intent=interpreted.detected_intent,
                        primary_model_id=primary_spec.model_id,
                        primary_workflow=primary_spec.workflow,
                        is_multi_stage=True,
                        error=err,
                    )
                stages.append(st_start)
                prev_m = still_model.model_id
                consumes_for_video.append(st_start.stage_id)
                intermediate_records.append(
                    ReferenceProvenanceRecord(
                        asset_id=st_start.stage_id,
                        role="start_frame",
                        path="<generated:stage_starter_keyframe>",
                        origin="generated_intermediate",
                    )
                )
                rationale.append("Multi-stage plan: synthesizing FLUX.2 starter keyframe (no starter image provided).")

            # Stage B: Synthesize matching rear end-keyframe for front-to-back turns using WanFirstLastFrameToVideo!
            if needs_flf_turn and still_model is not None:
                end_spec = spec.model_copy(deep=True)
                end_spec.intent.task = "image_generate"
                end_spec.motion.state = "automatic"
                end_spec.camera_movement.state = "automatic"
                end_spec.original_prompt = (
                    f"Matching rear end-keyframe in the exact same environment and lighting: {spec.original_prompt}"
                )
                end_spec.constraints = end_spec.constraints.model_copy(update={"width": 896, "height": 1280, "frames": 1})
                end_refs = ([str(starter_image)] if has_starter else []) + [
                    p for p in selected_supp_paths if p != str(starter_image)
                ]
                st_end, err = _build_stage(
                    idx=len(stages) + 1,
                    stage_id="stage_end_keyframe",
                    purpose="end_keyframe",
                    model_spec=still_model,
                    stage_spec=end_spec,
                    ref_paths=end_refs[:3],
                    prev_model=prev_m,
                    consumes=["stage_starter_keyframe"] if not has_starter else [],
                    origin="generated_intermediate",
                )
                if st_end is None:
                    warnings.append(f"Could not plan rear end-keyframe ({err}); falling back to single-frame I2V.")
                    needs_flf_turn = False
                else:
                    stages.append(st_end)
                    prev_m = still_model.model_id
                    consumes_for_video.append(st_end.stage_id)
                    intermediate_records.append(
                        ReferenceProvenanceRecord(
                            asset_id=st_end.stage_id,
                            role="end_frame",
                            path="<generated:stage_end_keyframe>",
                            origin="generated_intermediate",
                        )
                    )
                    rationale.append(
                        "Multi-stage turn plan: generating environment-matched rear end-keyframe with FLUX.2 "
                        "for WanFirstLastFrameToVideo so supplied rear references guide the turn."
                    )

            # Final Video Stage: Wan 2.2 I2V or WanFirstLastFrameToVideo.
            vid_refs = [str(starter_image)] if has_starter else []
            st_vid, err = _build_stage(
                idx=len(stages) + 1,
                stage_id="stage_video_final",
                purpose="video_flf2v" if needs_flf_turn else "video_i2v",
                model_spec=primary_spec,
                stage_spec=spec,
                ref_paths=vid_refs,
                prev_model=prev_m,
                consumes=consumes_for_video,
                origin="final_output",
            )
            if st_vid is None:
                return RoutedGenerationPlan(
                    ok=False,
                    detected_intent=interpreted.detected_intent,
                    primary_model_id=primary_spec.model_id,
                    primary_workflow=primary_spec.workflow,
                    is_multi_stage=len(stages) > 0,
                    error=err,
                )
            stages.append(st_vid)
            warnings.extend(st_vid.warnings)
            if not needs_flf_turn and selected_supp_paths and has_starter:
                warnings.extend(ref_opt.get("workflow_limitations") or [])

        # Build Sprint 7/8 GenerationVerificationRecord comparing against ORIGINAL user requirements.
        final_stage = stages[-1]
        user_ref_records = [
            ReferenceProvenanceRecord(
                asset_id=r.asset_id,
                role=r.role,
                path=r.path,
                subject_id=r.subject_id,
                origin="character_profile" if r.source_provenance == "character_profile" else "user_source",
                asset_fingerprint=r.asset_fingerprint,
                preserve_identity=r.preserve_identity,
                preserve_appearance=r.preserve_appearance,
                preserve_wardrobe=r.preserve_wardrobe,
            )
            for r in spec.reference_assets
        ]
        criteria = [
            AcceptanceCriterion(
                criterion_id=r.id,
                category=r.category,
                description=r.description,
                source="user_explicit",
                mandatory=True,
                verifiable_automatically=(r.category not in {"subjective_style", "emotion_nuance"}),
            )
            for r in spec.required_attributes
        ]
        unverifiable: list[str] = []
        if is_video and primary_spec.negative_prompt_strength == "ignored_cfg1" and spec.excluded_attributes:
            unverifiable.append(
                f"Negative exclusions ({', '.join(spec.excluded_attributes)}) are not enforced natively at CFG 1.0."
            )

        verification = GenerationVerificationRecord(
            original_prompt=spec.original_prompt,
            detected_intent=interpreted.detected_intent,
            compiled_prompt=final_stage.compiled_positive,
            compiled_negative=final_stage.compiled_negative,
            compiler_warnings=list(warnings),
            transformation_trace=interpreted.trace.public(),
            selected_model_id=primary_spec.model_id,
            selected_workflow=primary_spec.workflow,
            stages=[s.model_dump() for s in stages],
            reference_requirements=user_ref_records,
            intermediate_assets=intermediate_records,
            expected_output=ExpectedOutputCharacteristics(
                task=spec.intent.task,
                media_kind="video" if is_video else "image",
                width=final_stage.width,
                height=final_stage.height,
                frames=final_stage.frames,
                expected_orientation=interpreted.expected_orientation,
                expected_subject_count=sum(s.count for s in spec.subjects) if spec.subjects else None,
                expected_typography=list(interpreted.typography_targets),
                expected_exclusions=list(spec.excluded_attributes),
                expected_motion=spec.motion.value,
                expected_camera_movement=spec.camera_movement.value,
            ),
            mandatory_acceptance_criteria=criteria,
            known_unverifiable_requirements=unverifiable,
            execution_configuration={
                "local_only": not allow_cloud,
                "is_multi_stage": len(stages) > 1,
                "reference_megapixels": final_stage.reference_megapixels,
                "pixel_budget_provisional": True,
            },
            correction_budget=CorrectionBudget(max_retries=spec.constraints.max_retries, allow_cloud_escalation=False),
        )

        return RoutedGenerationPlan(
            ok=True,
            detected_intent=interpreted.detected_intent,
            primary_model_id=primary_spec.model_id,
            primary_workflow=primary_spec.workflow,
            is_multi_stage=len(stages) > 1,
            stages=stages,
            reference_optimization=ref_opt,
            rationale=rationale,
            warnings=list(dict.fromkeys(warnings)),
            verification_record=verification,
        )


def plan_intelligent_generation(
    prompt: str,
    *,
    requested_action: str | None = None,
    starter_image: str | None = None,
    subject_refs: SubjectReferenceSet | None = None,
    guided_spec: StructuredPromptSpec | None = None,
    input_mode: Literal["automatic", "guided", "script"] = "automatic",
    model_preference: str = "auto",
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    use_llm: bool | Literal["auto"] = False,
    snapshot: ResourceSnapshot | None = None,
    storage_check: StorageCheck | None = None,
    comfyui_dir: Path | None = None,
    registry: ModelRegistry = DEFAULT_REGISTRY,
    planner: ResourcePlanner | None = None,
    **llm_kwargs: Any,
) -> RoutedGenerationPlan:
    """One-call facade for Automatic, Guided, and Script modes (deterministic by default, optional local LLM)."""
    interpreted = interpret_natural_language(
        prompt,
        requested_action=requested_action,
        starter_image=starter_image,
        subject_refs=subject_refs,
        guided_spec=guided_spec,
        input_mode=input_mode,
        model_preference=model_preference,
        width=width,
        height=height,
        frames=frames,
        seed=seed,
        use_llm=use_llm,
        snapshot=snapshot,
        **llm_kwargs,
    )
    router = WorkflowRouter(registry=registry, planner=planner)
    return router.route(
        interpreted,
        starter_image=starter_image,
        subject_refs=subject_refs,
        snapshot=snapshot,
        storage_check=storage_check,
        comfyui_dir=comfyui_dir,
    )
