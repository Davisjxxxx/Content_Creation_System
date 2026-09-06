from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceMaturity(str, Enum):
    E5_ESTABLISHED = "E5_ESTABLISHED"
    E4_STRONG = "E4_STRONG"
    E3_CONTESTED = "E3_CONTESTED"
    E2_PLAUSIBLE = "E2_PLAUSIBLE"
    E1_SPECULATIVE = "E1_SPECULATIVE"
    E0_UNSUPPORTED = "E0_UNSUPPORTED"


class CandidateStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    RESEARCHING = "RESEARCHING"
    RED_TEAM = "RED_TEAM"
    READY_FOR_CERTIFICATION = "READY_FOR_CERTIFICATION"
    CERTIFIED = "CERTIFIED"
    CERTIFIED_WITH_QUALIFICATIONS = "CERTIFIED_WITH_QUALIFICATIONS"
    NEEDS_MORE_RESEARCH = "NEEDS_MORE_RESEARCH"
    HOLD = "HOLD"
    REJECTED = "REJECTED"
    SELECTED = "SELECTED"
    IN_PRODUCTION = "IN_PRODUCTION"
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"


class AgentRole(str, Enum):
    SOURCE_SCOUT = "SOURCE_SCOUT"
    CREATOR_WATCH = "CREATOR_WATCH"
    TREND_ANALYST = "TREND_ANALYST"
    EDGE_SCOUT = "EDGE_SCOUT"
    PRIMARY_RESEARCHER = "PRIMARY_RESEARCHER"
    CLAIM_EXTRACTOR = "CLAIM_EXTRACTOR"
    EVIDENCE_ANALYST = "EVIDENCE_ANALYST"
    EDGE_ADVERSARY = "EDGE_ADVERSARY"
    RED_TEAM = "RED_TEAM"
    STORY_ANALYST = "STORY_ANALYST"
    JASON_FIT = "JASON_FIT"
    CERTIFIER = "CERTIFIER"
    NARRATIVE_ARCHITECT = "NARRATIVE_ARCHITECT"
    SCRIPT_WRITER = "SCRIPT_WRITER"
    VISUAL_DIRECTOR = "VISUAL_DIRECTOR"
    FINAL_QA = "FINAL_QA"


class Platform(str, Enum):
    YOUTUBE = "YOUTUBE"
    YOUTUBE_SHORTS = "YOUTUBE_SHORTS"
    TIKTOK = "TIKTOK"
    FACEBOOK_REELS = "FACEBOOK_REELS"


@dataclass(slots=True)
class SourceRef:
    url: str
    title: str
    publisher: str = ""
    published_at: str | None = None
    accessed_at: str = field(default_factory=utc_now)
    source_tier: int = 3
    is_primary: bool = False
    notes: str = ""


@dataclass(slots=True)
class Claim:
    claim_id: str
    text: str
    sources: list[SourceRef] = field(default_factory=list)
    evidence_maturity: EvidenceMaturity = EvidenceMaturity.E1_SPECULATIVE
    confidence: float = 0.0
    edge_claim: bool = False
    competing_explanations: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    allowed_language: str = ""
    forbidden_language: list[str] = field(default_factory=list)
    replication_status: str = "unknown"
    last_verified_at: str = field(default_factory=utc_now)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.claim_id.strip():
            errors.append("claim_id is required")
        if not self.text.strip():
            errors.append("claim text is required")
        if not 0 <= self.confidence <= 1:
            errors.append("confidence must be between 0 and 1")
        return errors


@dataclass(slots=True)
class StoryScores:
    relevance: float = 0.0
    novelty: float = 0.0
    timeliness: float = 0.0
    evidence_quality: float = 0.0
    visual_potential: float = 0.0
    audience_curiosity: float = 0.0
    business_relevance: float = 0.0
    learning_value: float = 0.0
    edge_interest: float = 0.0
    saturation_penalty: float = 0.0
    composite: float = 0.0


@dataclass(slots=True)
class StoryCandidate:
    story_id: str
    title: str
    summary: str
    lane: str = "AI"
    sublane: str = "GENERAL_AI"
    status: CandidateStatus = CandidateStatus.DISCOVERED
    sources: list[SourceRef] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    scores: StoryScores = field(default_factory=StoryScores)
    tags: list[str] = field(default_factory=list)
    jason_fit_score: float | None = None
    selection_reason: str = ""
    rejection_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for claim in data["claims"]:
            maturity = claim.get("evidence_maturity")
            if isinstance(maturity, EvidenceMaturity):
                claim["evidence_maturity"] = maturity.value
        return data


@dataclass(slots=True)
class AgentResult:
    agent: AgentRole
    story_id: str
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""


@dataclass(slots=True)
class CertificationDecision:
    story_id: str
    decision: CandidateStatus
    rationale: str
    blocking_issues: list[str] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)
    required_disclosures: list[str] = field(default_factory=list)
    certified_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ProductionBrief:
    story_id: str
    master_angle: str
    target_duration_minutes: float
    platforms: list[Platform]
    required_claim_ids: list[str]
    visual_notes: list[str] = field(default_factory=list)
    disclosure_notes: list[str] = field(default_factory=list)
    approval_required: bool = True


@dataclass(slots=True)
class CostEvent:
    story_id: str
    subsystem: str
    provider: str
    model: str
    unit: str
    quantity: float
    cost_usd: float
    timestamp: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
