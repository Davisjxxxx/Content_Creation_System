"""Machine-readable verification and correction interfaces for Sprints 7 & 8 (Sprint 6, Section 13).

Preserves an authoritative record comparing generated outputs against the ORIGINAL user
requirements (not merely against a rewritten/compiled prompt), separating user-provided
source references from generated intermediate assets, and defining bounded correction budgets
without implementing unbounded retry loops during Sprint 6.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

VERIFICATION_CONTRACT_VERSION = "1.0"


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
    """Interface record populated by Sprint 7 visual inspection or human review."""

    criterion_id: str
    status: Literal["pending", "passed", "failed", "unverifiable"] = "pending"
    assessor: Literal["pending", "human", "deterministic", "local_vlm"] = "pending"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str = ""


class VisualInspectionResult(BaseModel):
    """Contract interface populated by Sprint 7 local image/video quality inspection."""

    inspected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    inspector_model: str = ""
    overall_decision: Literal["pending", "pass", "warn", "needs_correction", "unavailable"] = "pending"
    criterion_assessments: list[CriterionAssessment] = Field(default_factory=list)
    detected_defects: list[str] = Field(default_factory=list)
    subjective_quality_1_to_5: float | None = Field(default=None, ge=1.0, le=5.0)
    human_review_required: bool = True


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
