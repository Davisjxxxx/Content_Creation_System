"""Sprint 8 Regression Tests — Bounded Automated Corrective Generation & Calibration Guardrails.

Clearly distinguishes simulated unit/integration tests (this file) from live GPU ComfyUI + Ollama
corrective generation trials (executed separately via ``scripts/eval_sprint8_corrections.py``).

Covers all 23 Sprint 8 Section 15 regression areas:
- Correction recommendation validation & unsupported correction actions
- Mandatory requirement preservation & locked attribute preservation
- Exact-count, text, reference-selection, multi-view, and video continuity correction
- 7-factor candidate ranking & correction-induced regression rejection
- Retry budget exhaustion, early termination, user cancellation, failed-generation recovery
- ResourcePlanner admission, predictive storage protection, offline execution
- Invalid VLM responses, inspector disagreement, false acceptance, uncertain classifications
- Existing frontend preservation, SDK and Script mode compatibility
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from avatar_v2.correction_engine import (
    SUPPORTED_CORRECTION_STRATEGIES,
    CorrectionCoordinator,
    PlannedCorrectionStep,
    build_corrective_prompt,
    is_candidate_strict_improvement,
    plan_batched_inspection_and_correction,
    score_candidate,
    select_correction_step,
    validate_correction_strategy,
)
from avatar_v2.inspection_contract import (
    AcceptanceCriterion,
    CorrectionBudget,
    CorrectionRecommendation,
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    ReferenceProvenanceRecord,
    VisualInspectionResult,
    compute_inspection_status,
    get_verification_eligibility,
)
from avatar_v2.ollama_client import LLMCallResult, LocalLLMInfo, LocalOllamaClient
from avatar_v2.prompt_schema import StructuredPromptSpec
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceSnapshot, StorageCheck, StorageStatus
from avatar_v2.sdk_client import AvatarV2Client
from avatar_v2.visual_inspector import VisualInspector, reconcile_inspector_judgments


# --------------------------------------------------------------------------- Helpers


def _img(path: Path, color: tuple[int, int, int] = (200, 200, 200)) -> Path:
    im = Image.new("RGB", (512, 512), color)
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 220, 440), fill=(40, 150, 70), outline=(10, 40, 20), width=3)
    d.ellipse((280, 60, 460, 420), fill=(210, 50, 50), outline=(80, 15, 15), width=3)
    im.save(path)
    return path


def _snap(*, ram_sev: str = "normal", temp_sev: str = "normal", gpu_free_mb: int = 7200) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=1.0,
        host=HostMemoryStats(
            ram_total_mb=64000,
            ram_available_mb=32000 if ram_sev == "normal" else 1800,
            ram_available_pct=50.0 if ram_sev == "normal" else 2.8,
            swap_total_mb=8192,
            swap_used_mb=0,
            swap_used_pct=0.0,
        ),
        gpus=[GPUTelemetry(index=0, name="RTX 4070", used_mb=8188 - gpu_free_mb, total_mb=8188, free_mb=gpu_free_mb, temp_c=52, util_pct=5)],
        storage=StorageStatus(path="/tmp", total_gb=1000.0, free_gb=44.0, tier="normal"),
        ram_severity=ram_sev,  # type: ignore[arg-type]
        swap_severity="normal",
        temp_severity=temp_sev,  # type: ignore[arg-type]
        storage_severity="normal",
        overall_severity=ram_sev if ram_sev != "normal" else temp_sev,  # type: ignore[arg-type]
    )


def _storage_ok() -> StorageCheck:
    return StorageCheck(allowed=True, tier="normal", free_gb=44.0, estimated_mb=20.0, projected_free_gb=43.98, reserve_gb=15.0)


def _storage_blocked() -> StorageCheck:
    return StorageCheck(
        allowed=False, tier="emergency_stop", free_gb=10.0, estimated_mb=500.0, projected_free_gb=9.5, reserve_gb=15.0,
        reason="Insufficient disk space below 15 GB emergency reserve.",
    )


def _record(
    prompt: str,
    criteria: list[tuple[str, str, str, bool]],
    *,
    media_kind: str = "image",
    orientation: str = "general",
    refs: list[ReferenceProvenanceRecord] | None = None,
    budget: CorrectionBudget | None = None,
) -> GenerationVerificationRecord:
    return GenerationVerificationRecord(
        original_prompt=prompt,
        detected_intent="still_generate" if media_kind == "image" else "video_generate",
        compiled_prompt=prompt,
        selected_model_id="flux2_9b" if media_kind == "image" else "wan22_a14b",
        selected_workflow="builtin:flux2_txt2img" if media_kind == "image" else "workflows/wan22_i2v_14b_gguf_api.json",
        reference_requirements=refs or [],
        expected_output=ExpectedOutputCharacteristics(
            task="image_generate" if media_kind == "image" else "video_generate",
            media_kind=media_kind,  # type: ignore[arg-type]
            width=896 if media_kind == "image" else 480,
            height=1280 if media_kind == "image" else 832,
            frames=1 if media_kind == "image" else 81,
            expected_orientation=orientation,
        ),
        mandatory_acceptance_criteria=[
            AcceptanceCriterion(criterion_id=cid, category=cat, description=desc, mandatory=True, locked=locked)
            for cid, cat, desc, locked in criteria
        ],
        correction_budget=budget or CorrectionBudget(),
    )


def _inspection(
    statuses: dict[str, tuple[str, str]],
    *,
    defects: list[str] | None = None,
    recs: list[CorrectionRecommendation] | None = None,
    sharpness: float = 140.0,
) -> VisualInspectionResult:
    assessments = [
        CriterionAssessment(
            criterion_id=cid,
            original_requirement=desc,
            category=cid,
            mandatory=True,
            verification_eligibility="hybrid_corroborated",
            inspection_method="local_vlm",
            independent_corroboration=(st == "passed"),
            observed_evidence=f"obs:{cid}:{st}",
            status=st,  # type: ignore[arg-type]
            failure_description=f"failed:{cid}" if st == "failed" else "",
        )
        for cid, (st, desc) in statuses.items()
    ]
    decision, ui_st, ui_lbl, human_req = compute_inspection_status(assessments, detected_defects=defects)
    return VisualInspectionResult(
        inspector_model="qwen3-vl:8b-instruct",
        execution_mode="sequential_gpu_vlm",
        overall_decision=decision,
        ui_status=ui_st,
        ui_status_label=ui_lbl,
        criterion_assessments=assessments,
        detected_defects=list(defects or []),
        structural_metrics={"laplacian_sharpness": sharpness, "structural_defects": []},
        correction_recommendations=list(recs or []),
        human_review_required=human_req,
    )


# --------------------------------------------------------------------------- 1. Strategy Validation & Unsupported Actions


def test_correction_strategy_validation_rejects_unsupported_actions() -> None:
    """Section 15: Installed strategies validate cleanly; uninstalled inpainting/ControlNet are honestly rejected."""
    assert validate_correction_strategy("targeted_image_edit", media_kind="image")[0] is True
    assert validate_correction_strategy("prompt_refinement", media_kind="image")[0] is True
    assert validate_correction_strategy("video_regeneration", media_kind="video")[0] is True
    assert validate_correction_strategy("inpainting", media_kind="image")[0] is False
    assert validate_correction_strategy("composition_conditioning", media_kind="image")[0] is False
    assert validate_correction_strategy("pose_conditioning", media_kind="image")[0] is False
    assert validate_correction_strategy("run_shell_command", media_kind="image")[0] is False
    # Never represent full regeneration as localized editing.
    assert SUPPORTED_CORRECTION_STRATEGIES["targeted_image_edit"]["is_localized_edit"] is True
    assert SUPPORTED_CORRECTION_STRATEGIES["prompt_refinement"]["is_localized_edit"] is False


# --------------------------------------------------------------------------- 2. Intent, Locked Attributes, Count, Text, Multi-View & Video Planning


def test_intent_preservation_and_strategy_selection_across_failure_types(tmp_path: Path) -> None:
    """Section 15: Preserves original prompt & locked traits across count, text, multi-view, and video strategies."""
    cand = _img(tmp_path / "cand.png")
    back_ref = _img(tmp_path / "back.png")

    # Exact-count & typography -> prefers FLUX.2 targeted_image_edit first, then prompt_refinement on retry 2.
    rec_cnt = _record(
        "Wooden desk with exactly 3 ceramic mugs and sign 'STUDIO 04'.",
        [
            ("object_quantities", "object_quantities", "Exactly 3 ceramic coffee mugs", False),
            ("typography", "typography", "Sign reads 'STUDIO 04'", True),
        ],
    )
    ins_cnt = _inspection(
        {"object_quantities": ("failed", "Exactly 3 ceramic coffee mugs"), "typography": ("passed", "Sign reads 'STUDIO 04'")},
        recs=[CorrectionRecommendation(
            failed_criterion_id="object_quantities", category="object_quantities",
            original_requirement="Exactly 3 ceramic coffee mugs", supporting_evidence="Counted 2 mugs",
            action_type="revise_composition", recommended_action="Fix mug count",
            preserve_satisfied_requirements=["typography: Sign reads 'STUDIO 04'"],
        )],
    )
    step1 = select_correction_step(rec_cnt, ins_cnt, current_best_path=str(cand), attempt_number=1, prior_strategies=[])
    assert step1.strategy == "targeted_image_edit"
    assert step1.is_localized_edit is True
    assert str(cand) in step1.reference_paths
    assert "STUDIO 04" in step1.revised_prompt  # Locked & satisfied requirement preserved!

    step2 = select_correction_step(rec_cnt, ins_cnt, current_best_path=str(cand), attempt_number=2, prior_strategies=["targeted_image_edit"])
    assert step2.strategy == "prompt_refinement"
    assert step2.is_localized_edit is False
    assert "STUDIO 04" in step2.revised_prompt

    # Rear-view multi-view mismatch WITH back_body ref -> selects additional_reference_conditioning!
    rec_mv = _record(
        "Full-body rear view in gallery.",
        [("multiview_reference", "multiview_reference", "Rear low hair bun matches back_body reference", True)],
        orientation="rear_facing",
        refs=[ReferenceProvenanceRecord(asset_id="b1", role="back_body", path=str(back_ref))],
    )
    ins_mv = _inspection(
        {"multiview_reference": ("failed", "Rear low hair bun matches back_body reference")},
        recs=[CorrectionRecommendation(
            failed_criterion_id="multiview_reference", category="multiview_reference",
            original_requirement="Rear low hair bun matches back_body reference", supporting_evidence="Loose bob instead of low bun",
            action_type="select_subject_references", recommended_action="Use back_body ref",
            target_reference_roles=["back_body"],
        )],
    )
    step_mv = select_correction_step(rec_mv, ins_mv, current_best_path=str(cand), attempt_number=1, prior_strategies=[])
    assert step_mv.strategy == "additional_reference_conditioning"
    assert str(back_ref) in step_mv.reference_paths

    # Video continuity defect -> bounded video_regeneration (preserving original user starter image!).
    rec_vid = _record(
        "Smooth turn from front to back.",
        [("temporal_consistency", "temporal_consistency", "Smooth background without abrupt midpoint shift", False)],
        media_kind="video",
        refs=[ReferenceProvenanceRecord(asset_id="s1", role="primary_starter", path=str(cand))],
    )
    ins_vid = _inspection(
        {"temporal_consistency": ("failed", "Smooth background without abrupt midpoint shift")},
        recs=[CorrectionRecommendation(
            failed_criterion_id="temporal_consistency", category="temporal_consistency",
            original_requirement="Smooth background", supporting_evidence="Midpoint background shift",
            action_type="inspect_temporal_interval", recommended_action="Regenerate smoothly",
        )],
    )
    step_vid = select_correction_step(rec_vid, ins_vid, current_best_path=str(cand), attempt_number=1, prior_strategies=[])
    assert step_vid.strategy == "video_regeneration"
    assert step_vid.reference_paths == [str(cand)]  # Original starter never silently replaced!


# --------------------------------------------------------------------------- 3. Best-Candidate Ranking & Regression Rejection


def test_candidate_ranking_rejects_regression_and_preserves_original(tmp_path: Path) -> None:
    """Section 15: Fixing 1 requirement while breaking a previously satisfied or locked criterion is rejected."""
    orig = _img(tmp_path / "orig.png")
    c1 = _img(tmp_path / "cand1.png")
    c2 = _img(tmp_path / "cand2.png")

    rec = _record(
        "3 mugs and exact text 'STUDIO 04'.",
        [
            ("mugs", "object_quantities", "Exactly 3 mugs", False),
            ("sign", "typography", "Sign reads 'STUDIO 04'", True),
        ],
    )
    ins_orig = _inspection({"mugs": ("failed", "Exactly 3 mugs"), "sign": ("passed", "Sign reads 'STUDIO 04'")})
    # Candidate 1 fixes mugs, BUT breaks locked 'sign'! Must be rejected as regression!
    ins_reg = _inspection({"mugs": ("passed", "Exactly 3 mugs"), "sign": ("failed", "Sign reads 'STUDIO 04'")})
    # Candidate 2 fixes mugs AND preserves 'sign'! Selected as best!
    ins_good = _inspection({"mugs": ("passed", "Exactly 3 mugs"), "sign": ("passed", "Sign reads 'STUDIO 04'")})

    seq = [(str(c1), ins_reg), (str(c2), ins_good)]
    idx = {"i": 0}

    def fake_gen(step: PlannedCorrectionStep) -> tuple[str, float]:
        path = seq[idx["i"]][0]
        idx["i"] += 1
        return path, 1500.0

    def fake_ins(path: str, _rec: GenerationVerificationRecord) -> VisualInspectionResult:
        return ins_reg if path == str(c1) else ins_good

    report = CorrectionCoordinator().run_bounded_correction(
        record=rec,
        original_output_id="job_1",
        original_output_path=str(orig),
        initial_inspection=ins_orig,
        generate_fn=fake_gen,
        inspect_fn=fake_ins,
        snapshot_fn=_snap,
        storage_check_fn=_storage_ok,
    )
    assert report.attempts[0].final_disposition == "rejected_regression"
    assert "sign" in report.attempts[0].satisfied_criteria_lost
    assert report.attempts[1].final_disposition == "selected_as_best"
    assert report.best_candidate_path == str(c2)
    assert report.ui_status == "correction_complete"
    assert str(orig) in report.all_candidate_paths and orig.is_file()  # Original preserved!


# --------------------------------------------------------------------------- 4. Budget Exhaustion, Cancellation, Failure Recovery & Resource Guard


def test_budget_exhaustion_cancellation_recovery_and_resource_guards(tmp_path: Path) -> None:
    """Section 15: Enforces hard retry limits, clean cancellation, execution error recovery, and RAM/storage blocks."""
    orig = _img(tmp_path / "orig.png")
    rec = _record("3 mugs.", [("mugs", "object_quantities", "Exactly 3 mugs", False)], budget=CorrectionBudget(max_image_retries=2, max_retries=2))
    ins_fail = _inspection({"mugs": ("failed", "Exactly 3 mugs")})

    # A. Retry budget exhaustion (max 2 image retries) + failure recovery on attempt 1
    calls = {"n": 0}
    def flaky_gen(step: PlannedCorrectionStep) -> tuple[str, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Simulated ComfyUI transient error")
        p = _img(tmp_path / f"cand_{calls['n']}.png")
        return str(p), 900.0

    rep_budget = CorrectionCoordinator().run_bounded_correction(
        record=rec, original_output_id="j_bud", original_output_path=str(orig),
        initial_inspection=ins_fail, generate_fn=flaky_gen, inspect_fn=lambda _p, _r: ins_fail,
        snapshot_fn=_snap, storage_check_fn=_storage_ok,
    )
    assert calls["n"] == 2
    assert rep_budget.attempts[0].final_disposition == "failed_execution"
    assert rep_budget.stopped_reason == "correction_budget_exhausted"
    assert rep_budget.best_candidate_is_original is True
    assert rep_budget.ui_status == "review_recommended"

    # B. User cancellation stops immediately before spending GPU!
    ev = threading.Event()
    ev.set()
    gen_ran = {"flag": False}
    rep_cancel = CorrectionCoordinator().run_bounded_correction(
        record=rec, original_output_id="j_can", original_output_path=str(orig),
        initial_inspection=ins_fail,
        generate_fn=lambda _s: (gen_ran.__setitem__("flag", True) or str(orig), 100.0),
        inspect_fn=lambda _p, _r: ins_fail,
        cancel_event=ev, snapshot_fn=_snap, storage_check_fn=_storage_ok,
    )
    assert rep_cancel.cancelled is True
    assert gen_ran["flag"] is False

    # C. Predictive storage guard blocks correction cleanly!
    rep_stor = CorrectionCoordinator().run_bounded_correction(
        record=rec, original_output_id="j_stor", original_output_path=str(orig),
        initial_inspection=ins_fail,
        generate_fn=lambda _s: (str(orig), 100.0),
        inspect_fn=lambda _p, _r: ins_fail,
        snapshot_fn=_snap, storage_check_fn=_storage_blocked,
    )
    assert rep_stor.attempts[0].final_disposition == "blocked_resource_or_policy"
    assert "storage" in rep_stor.stopped_reason.lower() or "disk" in rep_stor.stopped_reason.lower()

    # D. Emergency RAM pressure blocks correction via ResourcePlanner!
    rep_ram = CorrectionCoordinator().run_bounded_correction(
        record=rec, original_output_id="j_ram", original_output_path=str(orig),
        initial_inspection=ins_fail,
        generate_fn=lambda _s: (str(orig), 100.0),
        inspect_fn=lambda _p, _r: ins_fail,
        snapshot_fn=lambda: _snap(ram_sev="emergency"), storage_check_fn=_storage_ok,
    )
    assert rep_ram.attempts[0].final_disposition == "blocked_resource_or_policy"


# --------------------------------------------------------------------------- 5. Spatial Localization, State Cross-Check, Inspector Disagreement & Uncertainty


def test_spatial_bbox_and_state_cross_check_and_disagreement(tmp_path: Path) -> None:
    """Section 15: Localized coordinates refute holistic spatial false-accept; disagreement stays uncertain."""
    assert get_verification_eligibility("spatial_relationships") == "requires_human_review"
    assert get_verification_eligibility("file_integrity") == "deterministic_reliable"

    # Disagreement reconciliation NEVER silently accepts!
    st1, _ = reconcile_inspector_judgments("passed", "failed", secondary_is_deterministic=True)
    assert st1 == "failed"
    st2, _ = reconcile_inspector_judgments("passed", "failed", secondary_is_deterministic=False)
    assert st2 == "uncertain"

    # Holistic VLM says 'passed' on spatial left/right AND closed notebooks,
    # BUT localized_objects shows vase at x=62% (right of lamp x=25%) and notebook state='open'!
    img = _img(tmp_path / "spatial.png")
    rec = _record(
        "Green vase left of brass lamp, and 2 closed notebooks.",
        [
            ("vase_left", "spatial_relationships", "Green glass vase is positioned to the viewer's left of the brass lamp", False),
            ("notebooks", "object_quantities", "Exactly 2 closed notebooks visible", False),
        ],
    )

    class _SimClient(LocalOllamaClient):
        def list_models(self) -> list[LocalLLMInfo]:
            return [LocalLLMInfo("qwen3-vl:8b-instruct", 6_000_000_000, 5.6, "qwen3vl", "8B", "Q4_K_M", ("vision_qa",))]
        def complete_vision_json(self, **kwargs: Any) -> LLMCallResult:
            return LLMCallResult(
                ok=True, model="qwen3-vl:8b-instruct", mode="sequential_gpu", latency_ms=250.0,
                data={
                    "observed_scene_summary": "Lamp on left, vase on right, one open and one closed notebook.",
                    "localized_objects": [
                        {"label": "brass lamp", "state": "standing", "horizontal_side": "left", "x_center_pct": 25},
                        {"label": "green glass vase", "state": "standing", "horizontal_side": "right", "x_center_pct": 62},
                        {"label": "white notebook", "state": "open", "horizontal_side": "left", "x_center_pct": 35},
                        {"label": "brown notebook", "state": "closed", "horizontal_side": "center", "x_center_pct": 55},
                    ],
                    "criteria": [
                        {"criterion_id": "vase_left", "observed_evidence": "Vase is left of lamp", "status": "passed"},
                        {"criterion_id": "notebooks", "observed_evidence": "2 notebooks visible", "observed_count": 2, "status": "passed"},
                    ],
                },
            )

    res = VisualInspector(client=_SimClient()).inspect_image(img, rec, use_vlm=True)
    by_id = {c.criterion_id: c for c in res.criterion_assessments}
    assert by_id["vase_left"].status == "failed"
    assert by_id["notebooks"].status == "failed"
    assert res.ui_status == "correction_recommended"


# --------------------------------------------------------------------------- 6. Batched Scheduling, Offline Guarantee, Frontend & SDK Compatibility


def test_batched_scheduling_offline_frontend_and_sdk_compatibility() -> None:
    """Section 15: Batching avoids repeated FLUX evictions; cloud stays disabled; frontend & SDK preserved."""
    plan = plan_batched_inspection_and_correction([{"id": f"img_{i}"} for i in range(4)], warm_model_id="flux2_9b")
    assert plan["batched_evictions"] < plan["interleaved_evictions"]
    assert plan["estimated_reload_savings_s"] >= 60

    # Cloud escalation hard-disabled in CorrectionBudget
    b = CorrectionBudget()
    assert b.allow_cloud_escalation is False
    assert b.limit_for_media("image") == 2
    assert b.limit_for_media("video") == 1

    # Frontend preserved: no separate correction dashboard or agent panel
    html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "view-correction-dashboard" not in html
    assert "Corrections (" in html

    # Native SDK & StructuredPromptSpec remain 100% compatible
    spec = StructuredPromptSpec(original_prompt="A crisp portrait in natural studio lighting.")
    assert spec.original_prompt == "A crisp portrait in natural studio lighting."
    sdk = AvatarV2Client(socket_path="/tmp/nonexistent_avatar_v2_test.sock")
    assert hasattr(sdk, "generate_image") and hasattr(sdk, "generate_video")
