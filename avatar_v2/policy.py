from __future__ import annotations

import re
from dataclasses import dataclass

from .models import AvatarManifest, ShotSpec

# People shown by this app are adults. The avatar model already requires an apparent age of 18+;
# these patterns catch prompt text that describes someone younger. They are deliberately narrow
# (explicit ages under 18 and unambiguous words), so "26-year-old", "minor asymmetry" or
# "baby hairs" are not flagged.
_SMALL_NUMBER = r"(?:[1-9]|1[0-7])"
_SMALL_WORD = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen)"
)
_MINOR_PATTERNS = (
    re.compile(rf"\b{_SMALL_NUMBER}\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b", re.IGNORECASE),
    re.compile(rf"\b{_SMALL_WORD}\s*-?\s*(?:years?|yrs?)\s*-?\s*old\b", re.IGNORECASE),
    re.compile(rf"\b(?:aged?)\s*{_SMALL_NUMBER}\b", re.IGNORECASE),
    re.compile(rf"\b{_SMALL_NUMBER}\s*(?:yo|y/o|y\.o\.?)(?=\W|$)", re.IGNORECASE),
    re.compile(
        r"\b(?:child|children|kid|kids|toddlers?|infants?|preteens?|pre-teens?|tweens?|schoolgirls?|schoolboys?|"
        r"juveniles?|adolescents?|underage|under-age|minors|an? minor|under\s*18|loli|lolita|shota|"
        r"middle[- ]schoolers?|elementary[- ]schoolers?|high[- ]schoolers?)\b",
        re.IGNORECASE,
    ),
)
# Extra wording that is refused for adult content classes (local rendering).
_ADULT_CONTENT_EXTRA = re.compile(
    r"\b(?:teens?|teenagers?|teenage|barely[- ]legal|jailbait|young[- ]looking|childlike|child-like)\b", re.IGNORECASE
)


def depicts_minor(text: str | None, *, adult_content: bool = False) -> str | None:
    """Return the phrase if ``text`` describes someone under 18, else None."""
    text = str(text or "")
    patterns = _MINOR_PATTERNS + ((_ADULT_CONTENT_EXTRA,) if adult_content else ())
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _described_text(shot: ShotSpec) -> str:
    return " ".join([shot.user_prompt or "", *(beat.description for beat in shot.action_beats)])


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
    consenting adult or a synthetic adult persona with adult provenance, and a
    prompt that describes anyone under 18 is refused. Unknown provenance fails
    closed for adult jobs.
    """
    if shot.content_class == "general":
        return PolicyDecision(True, "general-content job")

    phrase = depicts_minor(_described_text(shot), adult_content=True)
    if phrase:
        return PolicyDecision(
            False, f"Adult content can only show adults; the prompt describes someone under 18 (\"{phrase}\")."
        )
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
    phrase = depicts_minor(_described_text(shot))
    if phrase:
        return PolicyDecision(
            False,
            f"The presenter has to be an adult (18+). Remove \"{phrase}\" from the description; nothing was sent "
            "to Runway or charged.",
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
