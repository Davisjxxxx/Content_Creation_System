from __future__ import annotations

from collections import Counter

from .models import CandidateStatus, CertificationDecision, EvidenceMaturity, StoryCandidate


MaturityRank = {
    EvidenceMaturity.E0_UNSUPPORTED: 0,
    EvidenceMaturity.E1_SPECULATIVE: 1,
    EvidenceMaturity.E2_PLAUSIBLE: 2,
    EvidenceMaturity.E3_CONTESTED: 3,
    EvidenceMaturity.E4_STRONG: 4,
    EvidenceMaturity.E5_ESTABLISHED: 5,
}


def certify_story(candidate: StoryCandidate, red_team_blockers: list[str] | None = None) -> CertificationDecision:
    blockers = list(red_team_blockers or [])
    qualifications: list[str] = []
    disclosures: list[str] = []

    if not candidate.sources:
        blockers.append("No sources attached to story")
    if not candidate.claims:
        blockers.append("No atomic claims available for certification")

    for claim in candidate.claims:
        blockers.extend(f"{claim.claim_id}: {e}" for e in claim.validate())
        if not claim.sources:
            blockers.append(f"{claim.claim_id}: no source mapping")
        if claim.evidence_maturity == EvidenceMaturity.E0_UNSUPPORTED:
            qualifications.append(f"{claim.claim_id}: unsupported claim must not be asserted as fact")
        if claim.edge_claim:
            disclosures.append(f"{claim.claim_id}: label as edge/speculative/contested as appropriate")
            if not claim.competing_explanations:
                blockers.append(f"{claim.claim_id}: edge claim missing competing explanations")

    if blockers:
        return CertificationDecision(
            story_id=candidate.story_id,
            decision=CandidateStatus.NEEDS_MORE_RESEARCH,
            rationale="Certification failed closed because blocking evidence or red-team issues remain.",
            blocking_issues=sorted(set(blockers)),
            qualifications=sorted(set(qualifications)),
            required_disclosures=sorted(set(disclosures)),
        )

    maturity_counts = Counter(c.evidence_maturity for c in candidate.claims)
    weak_count = sum(maturity_counts[m] for m in (EvidenceMaturity.E0_UNSUPPORTED, EvidenceMaturity.E1_SPECULATIVE, EvidenceMaturity.E2_PLAUSIBLE))
    if weak_count:
        return CertificationDecision(
            story_id=candidate.story_id,
            decision=CandidateStatus.CERTIFIED_WITH_QUALIFICATIONS,
            rationale="Story is publishable only if lower-maturity claims retain explicit qualifications and source-linked wording.",
            qualifications=sorted(set(qualifications + ["Preserve evidence maturity language in script and visuals"])),
            required_disclosures=sorted(set(disclosures)),
        )

    return CertificationDecision(
        story_id=candidate.story_id,
        decision=CandidateStatus.CERTIFIED,
        rationale="All material claims are source-mapped, evidence-ranked, and free of unresolved blocking issues.",
        required_disclosures=sorted(set(disclosures)),
    )
