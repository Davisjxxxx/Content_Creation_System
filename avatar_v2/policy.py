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


def evaluate_cloud_policy(avatar: AvatarManifest, shot: ShotSpec) -> PolicyDecision:
    """Stricter gate for jobs that send prompts or media to a third-party cloud provider.

    Cloud jobs are limited to general content (adult classes stay on local renderers and
    are not permitted by the hosted provider's usage policy). Because a cloud job sends a
    person's likeness off this machine, every subject needs declared provenance: a
    synthetic persona, or a consenting adult whose consent is confirmed. Unknown fails closed.
    """
    if shot.content_class != "general":
        return PolicyDecision(
            False,
            "Cloud providers accept general content only; adult content classes stay on local renderers.",
        )
    for subject in avatar.subjects:
        if subject.kind == "synthetic":
            continue
        if subject.kind == "consenting_adult" and subject.consent_confirmed and subject.age_verified_18_plus:
            continue
        return PolicyDecision(
            False,
            f"Subject {subject.id} has no declared provenance for cloud upload. Mark the avatar as a synthetic "
            "persona, or as a consenting adult with confirmed consent, before using a cloud provider.",
        )
    return PolicyDecision(True, "general-content cloud job with declared subject provenance")


def enforce_policy(avatar: AvatarManifest, shot: ShotSpec) -> None:
    decision = evaluate_policy(avatar, shot)
    if not decision.allowed:
        raise PolicyError(decision.reason)
