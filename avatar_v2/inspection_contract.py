"""Machine-readable verification, visual inspection, and bounded correction interfaces (Sprints 6–8).

Preserves an authoritative record comparing generated outputs against the ORIGINAL user
requirements (not merely against a rewritten/compiled prompt), separates user-provided
source references from generated intermediate assets, classifies requirement-level
verification eligibility honestly, tracks every bounded correction attempt and candidate
ranking, and enforces host-side retry and local-only budgets.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

VERIFICATION_CONTRACT_VERSION = "1.3"

InspectionMethod = Literal[
    "pending",
    "deterministic",
    "structural_cv",
    "ocr",
    "reference_cv",
    "region_localization",
    "local_vlm",
    "video_temporal",
    "human",
]

VerificationEligibility = Literal[
    "deterministic_reliable",
    "hybrid_corroborated",
    "requires_human_review",
]

# Honest per-category verification eligibility derived from Sprint 7 live benchmarks
# (prevents weak categories such as subtle object states, photographic left/right spatial
# layout, fine hand anatomy, or micro-identity from being silently auto-accepted).
CATEGORY_VERIFICATION_ELIGIBILITY: dict[str, VerificationEligibility] = {
    "file_integrity": "deterministic_reliable",
    "dimensions": "deterministic_reliable",
    "aspect_ratio": "deterministic_reliable",
    "format": "deterministic_reliable",
    "video_duration": "deterministic_reliable",
    "frame_rate": "deterministic_reliable",
    "temporal_flicker": "deterministic_reliable",
    "video_static": "deterministic_reliable",
    "typography": "hybrid_corroborated",
    "colors": "hybrid_corroborated",
    "missing_or_unwanted": "hybrid_corroborated",
    "complex_nl": "hybrid_corroborated",
    "prompt_injection": "hybrid_corroborated",
    "multiview_reference": "hybrid_corroborated",
    "environment": "hybrid_corroborated",
    "subject_presence": "hybrid_corroborated",
    "object_quantities": "requires_human_review",
    "spatial_relationships": "requires_human_review",
    "composition": "requires_human_review",
    "clothing_appearance": "requires_human_review",
    "character_consistency": "requires_human_review",
    "reference_adherence": "requires_human_review",
    "photorealism": "requires_human_review",
    "fine_hand_anatomy": "requires_human_review",
    "facial_identity_subtle": "requires_human_review",
    "micro_detail_realism": "requires_human_review",
    "video_motion": "requires_human_review",
    "camera_movement": "requires_human_review",
    "temporal_consistency": "hybrid_corroborated",
}


def get_verification_eligibility(category: str) -> VerificationEligibility:
    """Return honest automated verification eligibility for a requirement category."""
    return CATEGORY_VERIFICATION_ELIGIBILITY.get(str(category or "").strip().lower(), "requires_human_review")


UIInspectionStatus = Literal[
    "generated",
    "generating",
    "checking_quality",
    "improving_result",
    "requirements_verified",
    "correction_complete",
    "some_uncertain",
    "correction_recommended",
    "review_recommended",
]

UI_INSPECTION_LABELS: dict[UIInspectionStatus, str] = {
    "generated": "Generated",
    "generating": "Generating",
    "checking_quality": "Checking quality",
    "improving_result": "Improving result",
    "requirements_verified": "Requirements verified",
    "correction_complete": "Correction complete",
    "some_uncertain": "Some requirements uncertain",
    "correction_recommended": "Correction recommended",
    "review_recommended": "Review recommended",
}

CorrectionActionType = Literal[
    "revise_composition",
    "targeted_edit",
    "text_correction",
    "select_subject_references",
    "add_back_body_reference",
    "add_side_reference",
    "revise_ending_keyframe",
    "inspect_temporal_interval",
    "revise_framing",
    "request_human_review",
]

CorrectionStrategyType = Literal[
    "prompt_refinement",
    "seed_variation",
    "targeted_image_edit",
    "masked_localized_edit",
    "inpainting",
    "reference_adjustment",
    "additional_reference_conditioning",
    "composition_conditioning",
    "pose_conditioning",
    "alternative_approved_workflow",
    "keyframe_correction",
    "video_regeneration",
    "human_review_escalation",
]


class AcceptanceCriterion(BaseModel):
    """One machine-readable verification criterion derived from the original user request."""

    criterion_id: str
    category: str
    description: str
    source: Literal["user_explicit", "guided_ui", "script", "reference_role", "inferred_default"] = "user_explicit"
    mandatory: bool = True
    verifiable_automatically: bool = True
    verification_hint: str = ""
    locked: bool = False


class ReferenceProvenanceRecord(BaseModel):
    """Distinguishes user-supplied source assets from generated intermediate assets."""

    asset_id: str
    role: str
    path: str
    subject_id: str = "primary"
    origin: Literal["user_source", "character_profile", "generated_intermediate"] = "user_source"
    asset_fingerprint: str = ""
    preserve_identity: bool = True
    preserve_appearance: bool = True
    preserve_wardrobe: bool = True


class ExpectedOutputCharacteristics(BaseModel):
    """Structured expectations that Sprint 7/8 visual/temporal QA inspects against."""

    task: str
    media_kind: Literal["image", "video"]
    width: int
    height: int
    frames: int = 1
    expected_orientation: str = "general"
    expected_subject_count: int | None = None
    expected_typography: list[str] = Field(default_factory=list)
    expected_exclusions: list[str] = Field(default_factory=list)
    expected_motion: str = ""
    expected_camera_movement: str = ""


class CriterionAssessment(BaseModel):
    """Requirement-level visual inspection result populated by Sprint 7/8 or human review.

    Note: ``confidence`` is auxiliary diagnostic metadata and must NEVER be treated as
    objective proof that a requirement passed. Unverifiable requirements remain ``uncertain``.
    """

    criterion_id: str
    original_requirement: str = ""
    category: str = ""
    mandatory: bool = True
    verification_eligibility: VerificationEligibility = "hybrid_corroborated"
    inspection_method: InspectionMethod = "pending"
    independent_corroboration: bool = False
    observed_evidence: str = ""
    status: Literal["pending", "passed", "failed", "uncertain", "unverifiable"] = "pending"
    assessor: Literal[
        "pending",
        "human",
        "deterministic",
        "structural_cv",
        "ocr",
        "reference_cv",
        "region_localization",
        "local_vlm",
        "video_temporal",
    ] = "pending"
    artifact_frame: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_limitations: str = ""
    failure_description: str = ""
    recommended_action: str = ""
    note: str = ""


class CorrectionRecommendation(BaseModel):
    """Structured Sprint 8 correction handoff identifying failed requirements & evidence.

    Strictly restricted to closed ``action_type`` values; visual inspection models can never
    emit arbitrary execution commands or invent unsupported workflows.
    """

    recommendation_id: str = Field(default_factory=lambda: f"cor_{secrets.token_hex(4)}")
    failed_criterion_id: str
    category: str
    original_requirement: str
    supporting_evidence: str
    artifact_frame: str = ""
    action_type: CorrectionActionType
    recommended_action: str
    preserve_satisfied_requirements: list[str] = Field(default_factory=list)
    target_reference_roles: list[str] = Field(default_factory=list)
    affected_time_intervals: list[str] = Field(default_factory=list)
    requires_human_review: bool = False


class VisualInspectionResult(BaseModel):
    """Contract interface populated by Sprint 7/8 local image/video quality inspection."""

    inspected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    inspection_version: str = VERIFICATION_CONTRACT_VERSION
    candidate_id: str = ""
    output_path: str = ""
    output_sha256: str = ""
    requirement_set_hash: str = ""
    reference_fingerprints: dict[str, str] = Field(default_factory=dict)
    inspector_model: str = ""
    execution_mode: str = "deterministic_only"
    overall_decision: Literal["pending", "pass", "warn", "needs_correction", "unavailable"] = "pending"
    ui_status: UIInspectionStatus = "generated"
    ui_status_label: str = "Generated"
    methods_used: list[str] = Field(default_factory=list)
    criterion_assessments: list[CriterionAssessment] = Field(default_factory=list)
    detected_defects: list[str] = Field(default_factory=list)
    deterministic_metrics: dict[str, Any] = Field(default_factory=dict)
    structural_metrics: dict[str, Any] = Field(default_factory=dict)
    reference_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    video_summary: dict[str, Any] = Field(default_factory=dict)
    correction_recommendations: list[CorrectionRecommendation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    prompt_injection_warnings: list[str] = Field(default_factory=list)
    subjective_quality_1_to_5: float | None = Field(default=None, ge=1.0, le=5.0)
    human_review_required: bool = True
    latency_ms: float = 0.0
    vram_released_verified: bool | None = None

    def compact_ui_summary(self) -> dict[str, Any]:
        """Minimal, non-intrusive status summary for the existing approved frontend."""
        passed = sum(1 for c in self.criterion_assessments if c.status == "passed")
        failed = sum(1 for c in self.criterion_assessments if c.status == "failed")
        uncertain = sum(
            1 for c in self.criterion_assessments if c.status in {"uncertain", "unverifiable", "pending"}
        )
        return {
            "candidate_id": self.candidate_id,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "requirement_set_hash": self.requirement_set_hash,
            "reference_fingerprints": dict(self.reference_fingerprints),
            "inspection_version": self.inspection_version,
            "inspected_at": self.inspected_at,
            "ui_status": self.ui_status,
            "ui_status_label": self.ui_status_label,
            "overall_decision": self.overall_decision,
            "inspector_model": self.inspector_model,
            "execution_mode": self.execution_mode,
            "passed_count": passed,
            "failed_count": failed,
            "uncertain_count": uncertain,
            "total_criteria": len(self.criterion_assessments),
            "recommendations_count": len(self.correction_recommendations),
            "human_review_required": self.human_review_required,
            "detected_defects": list(self.detected_defects[:5]),
            "limitations": list(self.limitations[:4]),
            "criteria": [
                {
                    "criterion_id": c.criterion_id,
                    "original_requirement": c.original_requirement,
                    "mandatory": c.mandatory,
                    "status": c.status,
                    "inspection_method": c.inspection_method,
                    "verification_eligibility": c.verification_eligibility,
                    "independent_corroboration": c.independent_corroboration,
                    "observed_evidence": c.observed_evidence[:240],
                    "artifact_frame": c.artifact_frame,
                    "failure_description": c.failure_description[:240],
                    "recommended_action": c.recommended_action[:240],
                }
                for c in self.criterion_assessments
            ],
            "recommendations": [
                {
                    "failed_criterion_id": r.failed_criterion_id,
                    "action_type": r.action_type,
                    "recommended_action": r.recommended_action[:260],
                    "supporting_evidence": r.supporting_evidence[:240],
                }
                for r in self.correction_recommendations[:6]
            ],
        }


def compute_requirement_set_hash(record: "GenerationVerificationRecord") -> str:
    """Stable SHA-256 digest binding an inspection result to the exact original requirements."""
    payload = {
        "original_prompt": record.original_prompt,
        "criteria": [
            {
                "id": c.criterion_id,
                "category": c.category,
                "description": c.description,
                "mandatory": c.mandatory,
                "locked": c.locked,
            }
            for c in record.mandatory_acceptance_criteria
        ],
        "expected": record.expected_output.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_inspection_status(
    assessments: list[CriterionAssessment],
    *,
    detected_defects: list[str] | None = None,
) -> tuple[Literal["pass", "warn", "needs_correction", "unavailable"], UIInspectionStatus, str, bool]:
    """Derive overall decision & concise UI status strictly without false certainty.

    Never returns ``requirements_verified`` when any mandatory requirement is failed,
    uncertain, unverifiable, pending, or lacks independent corroboration in a human-review category.
    """
    if not assessments:
        if detected_defects:
            return ("needs_correction", "correction_recommended", UI_INSPECTION_LABELS["correction_recommended"], True)
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)

    mandatory = [a for a in assessments if a.mandatory]
    mandatory_failed = [a for a in mandatory if a.status == "failed"]
    mandatory_uncertain = [a for a in mandatory if a.status in {"uncertain", "unverifiable", "pending"}]
    optional_failed = [a for a in assessments if not a.mandatory and a.status == "failed"]
    needs_human_corroboration = any(
        a.verification_eligibility == "requires_human_review" and not a.independent_corroboration
        for a in mandatory
        if a.status == "passed"
    )

    if mandatory_failed or detected_defects:
        return ("needs_correction", "correction_recommended", UI_INSPECTION_LABELS["correction_recommended"], True)
    if mandatory_uncertain:
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)
    if optional_failed:
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)
    if mandatory and all(a.status == "passed" for a in mandatory):
        if needs_human_corroboration:
            return (
                "warn",
                "review_recommended",
                UI_INSPECTION_LABELS["review_recommended"],
                True,
            )
        return (
            "pass",
            "requirements_verified",
            UI_INSPECTION_LABELS["requirements_verified"],
            False,
        )
    return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)


class CorrectionBudget(BaseModel):
    """Hard host-enforced bounded retry budget (Sprint 8, Section 8 — no unbounded loops)."""

    max_retries: int = Field(default=2, ge=0, le=3)
    max_image_retries: int = Field(default=2, ge=0, le=3)
    max_video_retries: int = Field(default=1, ge=0, le=2)
    retries_used: int = Field(default=0, ge=0)
    allow_seed_retry: bool = True
    allow_prompt_adjustment: bool = True
    allow_targeted_edit: bool = True
    allow_cloud_escalation: bool = False  # Always False without explicit user authorization!
    extended_session_authorized: bool = False

    def limit_for_media(self, media_kind: Literal["image", "video"]) -> int:
        """Tightest authorized remaining retry limit across global, per-media, and retries_used (RC-11)."""
        media_cap = self.max_video_retries if media_kind == "video" else self.max_image_retries
        effective_cap = min(max(0, self.max_retries), max(0, media_cap))
        return max(0, effective_cap - max(0, self.retries_used))

    @property
    def remaining(self) -> int:
        return max(0, self.max_retries - self.retries_used)


class CorrectionAttemptRecord(BaseModel):
    """Complete immutable provenance & outcome record for one bounded correction attempt (Section 6)."""

    attempt_id: str = Field(default_factory=lambda: f"att_{secrets.token_hex(4)}")
    attempt_number: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_output_id: str
    original_output_path: str
    candidate_output_id: str = ""
    candidate_output_path: str = ""
    failed_criterion_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    selected_strategy: CorrectionStrategyType
    is_localized_edit: bool = False
    original_prompt: str
    revised_prompt: str
    reference_changes: list[str] = Field(default_factory=list)
    model_id: str
    workflow: str
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    preserved_satisfied_criteria: list[str] = Field(default_factory=list)
    expected_improvement: str = ""
    actual_inspection_result: VisualInspectionResult | None = None
    satisfied_criteria_lost: list[str] = Field(default_factory=list)
    newly_introduced_defects: list[str] = Field(default_factory=list)
    candidate_rank_score: dict[str, Any] = Field(default_factory=dict)
    final_disposition: Literal[
        "selected_as_best",
        "retained_previous_best",
        "rejected_regression",
        "blocked_resource_or_policy",
        "cancelled",
        "failed_execution",
        "escalated_human_review",
    ]
    generation_latency_ms: float = 0.0
    inspection_latency_ms: float = 0.0
    notes: list[str] = Field(default_factory=list)


class CorrectionSessionReport(BaseModel):
    """Summary of a bounded Sprint 8 correction session preserving all candidates & provenance."""

    session_id: str = Field(default_factory=lambda: f"csess_{secrets.token_hex(5)}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    media_kind: Literal["image", "video"] = "image"
    original_output_id: str
    original_output_path: str
    all_candidate_paths: list[str] = Field(default_factory=list)
    best_candidate_id: str
    best_candidate_path: str
    best_candidate_is_original: bool = True
    best_candidate_reason: str = ""
    initial_mandatory_passed: int = 0
    final_mandatory_passed: int = 0
    total_mandatory: int = 0
    attempts: list[CorrectionAttemptRecord] = Field(default_factory=list)
    attempts_executed: int = 0
    budget_limit: int = 2
    stopped_reason: str = ""
    cancelled: bool = False
    ui_status: UIInspectionStatus = "review_recommended"
    ui_status_label: str = "Review recommended"
    unresolved_limitations: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    total_latency_ms: float = 0.0

    def compact_ui_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ui_status": self.ui_status,
            "ui_status_label": self.ui_status_label,
            "best_candidate_id": self.best_candidate_id,
            "best_candidate_path": self.best_candidate_path,
            "best_candidate_is_original": self.best_candidate_is_original,
            "best_candidate_reason": self.best_candidate_reason[:260],
            "initial_mandatory_passed": self.initial_mandatory_passed,
            "final_mandatory_passed": self.final_mandatory_passed,
            "total_mandatory": self.total_mandatory,
            "attempts_executed": self.attempts_executed,
            "budget_limit": self.budget_limit,
            "stopped_reason": self.stopped_reason,
            "cancelled": self.cancelled,
            "human_review_required": self.human_review_required,
            "candidate_count": len(self.all_candidate_paths),
            "unresolved_limitations": list(self.unresolved_limitations[:4]),
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "selected_strategy": a.selected_strategy,
                    "is_localized_edit": a.is_localized_edit,
                    "candidate_output_path": a.candidate_output_path,
                    "final_disposition": a.final_disposition,
                    "satisfied_criteria_lost": list(a.satisfied_criteria_lost),
                    "newly_introduced_defects": list(a.newly_introduced_defects[:3]),
                }
                for a in self.attempts
            ],
        }


class GenerationVerificationRecord(BaseModel):
    """Complete immutable provenance & acceptance record attached to planned/queued jobs."""

    record_version: str = VERIFICATION_CONTRACT_VERSION
    record_id: str = Field(default_factory=lambda: f"ver_{secrets.token_hex(6)}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_prompt: str
    detected_intent: str
    compiled_prompt: str
    compiled_negative: str = ""
    compiler_warnings: list[str] = Field(default_factory=list)
    transformation_trace: dict[str, Any] = Field(default_factory=dict)
    selected_model_id: str
    selected_workflow: str
    stages: list[dict[str, Any]] = Field(default_factory=list)
    reference_requirements: list[ReferenceProvenanceRecord] = Field(default_factory=list)
    intermediate_assets: list[ReferenceProvenanceRecord] = Field(default_factory=list)
    expected_output: ExpectedOutputCharacteristics
    mandatory_acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    known_unverifiable_requirements: list[str] = Field(default_factory=list)
    execution_configuration: dict[str, Any] = Field(default_factory=dict)
    inspection_result: VisualInspectionResult | None = None
    correction_budget: CorrectionBudget = Field(default_factory=CorrectionBudget)
    correction_session: CorrectionSessionReport | None = None

    def public(self) -> dict[str, Any]:
        return self.model_dump()
