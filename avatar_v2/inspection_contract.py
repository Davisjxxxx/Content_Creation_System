"""Machine-readable verification, visual inspection, and correction interfaces (Sprints 6–8).

Preserves an authoritative record comparing generated outputs against the ORIGINAL user
requirements (not merely against a rewritten/compiled prompt), separating user-provided
source references from generated intermediate assets, capturing requirement-level visual
inspection evidence, and preparing bounded Sprint 8 correction recommendations without
autonomous unbounded regeneration.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

VERIFICATION_CONTRACT_VERSION = "1.1"

InspectionMethod = Literal[
    "pending",
    "deterministic",
    "structural_cv",
    "ocr",
    "reference_cv",
    "local_vlm",
    "video_temporal",
    "human",
]

UIInspectionStatus = Literal[
    "generated",
    "checking_quality",
    "requirements_verified",
    "some_uncertain",
    "correction_recommended",
]

UI_INSPECTION_LABELS: dict[UIInspectionStatus, str] = {
    "generated": "Generated",
    "checking_quality": "Checking quality",
    "requirements_verified": "Requirements verified",
    "some_uncertain": "Some requirements uncertain",
    "correction_recommended": "Correction recommended",
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


class AcceptanceCriterion(BaseModel):
    """One machine-readable verification criterion derived from the original user request."""

    criterion_id: str
    category: str
    description: str
    source: Literal["user_explicit", "guided_ui", "script", "reference_role", "inferred_default"] = "user_explicit"
    mandatory: bool = True
    verifiable_automatically: bool = True
    verification_hint: str = ""


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
    """Structured expectations that Sprint 7 visual/temporal QA inspects against."""

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
    """Requirement-level visual inspection result populated by Sprint 7 or human review.

    Note: ``confidence`` is auxiliary diagnostic metadata and must NEVER be treated as
    objective proof that a requirement passed. Unverifiable requirements remain ``uncertain``.
    """

    criterion_id: str
    original_requirement: str = ""
    category: str = ""
    mandatory: bool = True
    inspection_method: InspectionMethod = "pending"
    observed_evidence: str = ""
    status: Literal["pending", "passed", "failed", "uncertain", "unverifiable"] = "pending"
    assessor: Literal[
        "pending",
        "human",
        "deterministic",
        "structural_cv",
        "ocr",
        "reference_cv",
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
    """Contract interface populated by Sprint 7 local image/video quality inspection."""

    inspected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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


def compute_inspection_status(
    assessments: list[CriterionAssessment],
    *,
    detected_defects: list[str] | None = None,
) -> tuple[Literal["pass", "warn", "needs_correction", "unavailable"], UIInspectionStatus, str, bool]:
    """Derive overall decision & concise UI status strictly without false certainty.

    Never returns ``requirements_verified`` when any mandatory requirement is failed,
    uncertain, unverifiable, or pending.
    """
    if not assessments:
        if detected_defects:
            return ("needs_correction", "correction_recommended", UI_INSPECTION_LABELS["correction_recommended"], True)
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)

    mandatory = [a for a in assessments if a.mandatory]
    mandatory_failed = [a for a in mandatory if a.status == "failed"]
    mandatory_uncertain = [a for a in mandatory if a.status in {"uncertain", "unverifiable", "pending"}]
    optional_failed = [a for a in assessments if not a.mandatory and a.status == "failed"]

    if mandatory_failed or detected_defects:
        return ("needs_correction", "correction_recommended", UI_INSPECTION_LABELS["correction_recommended"], True)
    if mandatory_uncertain:
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)
    if optional_failed:
        return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)
    if mandatory and all(a.status == "passed" for a in mandatory):
        return ("pass", "requirements_verified", UI_INSPECTION_LABELS["requirements_verified"], False)
    return ("warn", "some_uncertain", UI_INSPECTION_LABELS["some_uncertain"], True)


class CorrectionBudget(BaseModel):
    """Hard bounded retry budget interface prepared for Sprint 8 (no unbounded loops)."""

    max_retries: int = Field(default=2, ge=0, le=3)
    retries_used: int = Field(default=0, ge=0)
    allow_seed_retry: bool = True
    allow_prompt_adjustment: bool = True
    allow_cloud_escalation: bool = False  # Always False without explicit user consent!

    @property
    def remaining(self) -> int:
        return max(0, self.max_retries - self.retries_used)


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

    def public(self) -> dict[str, Any]:
        return self.model_dump()
