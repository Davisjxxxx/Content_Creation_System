from __future__ import annotations

from dataclasses import dataclass

from .models import AvatarManifest, ShotSpec


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


class PolicyError(ValueError):
    pass


def evaluate_policy(avatar: AvatarManifest, shot: ShotSpec) -> PolicyDecision:
    """Allow adult-capable workflows while enforcing hard safety boundaries.

    The orchestrator does not impose a blanket adult-content filter. Adult nudity
    or sexual content requires every represented subject to be an age-verified,
    consenting adult or a synthetic adult persona with adult provenance.
    Unknown provenance fails closed for adult jobs.
    """
    if shot.content_class == "general":
        return PolicyDecision(True, "general-content job")

    for subject in avatar.subjects:
        if subject.kind == "unknown":
            return PolicyDecision(False, f"subject {subject.id} has unknown provenance")
        if not subject.age_verified_18_plus:
            return PolicyDecision(False, f"subject {subject.id} is not verified 18+")
        if not subject.consent_confirmed:
            return PolicyDecision(False, f"subject {subject.id} lacks consent confirmation")

    return PolicyDecision(True, "adult-capable job with verified adult consent/provenance")


def enforce_policy(avatar: AvatarManifest, shot: ShotSpec) -> None:
    decision = evaluate_policy(avatar, shot)
    if not decision.allowed:
        raise PolicyError(decision.reason)
