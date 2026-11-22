"""Presenter-focused prompt compilation for length-limited cloud image and video models.

The local composer writes long, H3-native structured prompts. Hosted models enforce hard
limits (Gen-4 Image and Gen-4.5 allow 1,000 UTF-16 units), so this compiler builds a
prompt in priority order and drops the least important sections first. Sections that
mention a tagged reference (``@presenter``, ``@outfit``, ``@location``) are never dropped,
because an unmentioned tag leaves its reference image unused.
"""

from __future__ import annotations

from typing import Any

from .brand_instructor import REFERENCE_SLOTS
from .models import AvatarManifest, ShotSpec
from .providers.runway import utf16_len

# Consistent-variation presets are the presenter's reviewable reference slots. Each one binds
# the same approved identity reference and seed; its output becomes a pending candidate.
PRESENTER_VARIATIONS: tuple[dict[str, str], ...] = tuple(
    {"id": slot["id"], "label": slot["label"], "prompt": slot["prompt"]} for slot in REFERENCE_SLOTS
)
_VARIATIONS_BY_ID = {item["id"]: item for item in PRESENTER_VARIATIONS}

STILL_REALISM = (
    "Photorealistic photograph of a real-looking adult: natural skin texture with fine pores and subtle "
    "imperfections, accurate eyes with natural catchlights, true-to-life color, flattering soft key light, "
    "sharp focus on the eyes, gentle depth of field."
)
MOTION_REALISM = (
    "Natural blinking, subtle breathing, realistic micro-expressions and small head movements; the face and "
    "identity stay exactly the same in every frame; smooth, physically plausible motion with no morphing."
)
NO_TEXT = "No text, captions, logos or watermarks anywhere in the frame."


def variation_by_id(variation_id: str) -> dict[str, str]:
    try:
        return _VARIATIONS_BY_ID[variation_id]
    except KeyError as exc:
        raise ValueError(f"Unknown presenter variation '{variation_id}'.") from exc


def variations_payload() -> list[dict[str, str]]:
    return [dict(item) for item in PRESENTER_VARIATIONS]


def subject_phrase(avatar: AvatarManifest) -> str:
    """Describe the presenter's sex and apparent adult age without identity claims."""
    noun = {"female": "woman", "male": "man"}.get(avatar.sex, "person")
    possessive = {"female": "her", "male": "his"}.get(avatar.sex, "their")
    age = avatar.apparent_age_years
    if age is None:
        return f"an adult {noun}"
    if age < 20:
        return f"an adult {noun} aged {age}"
    decade = (age // 10) * 10
    position = age % 10
    stage = "early" if position <= 3 else "mid" if position <= 6 else "late"
    joiner = "-" if stage == "mid" else " "
    return f"a {noun} in {possessive} {stage}{joiner}{decade}s"


def _clean(text: str | None) -> str:
    return " ".join(str(text or "").split())


def _sentence(text: str) -> str:
    text = _clean(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _truncate_words(text: str, budget: int) -> str:
    if budget <= 1:
        return ""
    if utf16_len(text) <= budget:
        return text
    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join([*kept, word]) + "…"
        if utf16_len(candidate) > budget:
            break
        kept.append(word)
    return (" ".join(kept) + "…") if kept else ""


def compile_presenter_prompt(
    avatar: AvatarManifest,
    shot: ShotSpec,
    *,
    budget: int,
    identity_tag: str | None = None,
    outfit_tag: str | None = None,
    location_tag: str | None = None,
    variation: dict[str, str] | None = None,
    motion: bool = False,
    realism: str | None = None,
) -> tuple[str, list[str]]:
    """Return ``(prompt, dropped_section_names)`` within ``budget`` UTF-16 units.

    ``realism`` replaces the generic still-image realism block (for example a profile's own
    positive realism guards); motion prompts always use the motion realism block.
    """
    subject = subject_phrase(avatar)
    if identity_tag:
        lead = f"@{identity_tag}, {subject}: exactly the same face and identity as in the @{identity_tag} reference."
    else:
        lead = _sentence(subject[0].upper() + subject[1:])

    camera_bits = [f"Framing: {_clean(shot.camera.framing)}"]
    if shot.camera.lens_equivalent_mm:
        camera_bits.append(f"{shot.camera.lens_equivalent_mm}mm lens look")
    if motion and shot.camera.movement:
        camera_bits.append(f"camera: {_clean(shot.camera.movement)}")
    beats = " ".join(
        f"[{beat.start_s:g}-{beat.end_s:g}s] {_clean(beat.description)}" for beat in shot.action_beats
    )

    # (name, text, priority, droppable); lower priority numbers are kept longer.
    sections: list[tuple[str, str, int, bool]] = [
        ("identity", lead, 0, False),
        ("direction", _sentence(shot.user_prompt), 1, False),
    ]
    if variation:
        sections.append(("variation", _sentence(variation["prompt"]), 2, True))
    if motion and beats:
        sections.append(("timing", _sentence(beats), 2, True))
    if outfit_tag:
        sections.append(("outfit", f"Wearing the outfit shown in @{outfit_tag}.", 0, False))
    elif shot.wardrobe:
        sections.append(("outfit", _sentence(f"Wardrobe: {shot.wardrobe}"), 3, True))
    if location_tag:
        sections.append(("location", f"Setting: the location shown in @{location_tag}.", 0, False))
    elif shot.environment:
        sections.append(("location", _sentence(f"Setting: {shot.environment}"), 3, True))
    sections.append(("camera", _sentence("; ".join(camera_bits)), 4, True))
    if avatar.persistent_features:
        features = "; ".join(_clean(item) for item in avatar.persistent_features if _clean(item))
        if features:
            sections.append(("features", _sentence(f"Keep these identity details: {features}"), 3, True))
    # A profile's own realism guards encode its exclusions, so they outrank camera and feature lists.
    realism_priority = 2 if (realism and not motion) else 5
    sections.append(("realism", MOTION_REALISM if motion else (realism or STILL_REALISM), realism_priority, True))
    sections.append(("no_text", NO_TEXT, 6, True))

    def assemble(items: list[tuple[str, str, int, bool]]) -> str:
        return " ".join(text for _name, text, _priority, _droppable in items if text)

    kept = list(sections)
    dropped: list[str] = []
    while utf16_len(assemble(kept)) > budget:
        candidates = [item for item in kept if item[3]]
        if not candidates:
            break
        victim = max(candidates, key=lambda item: item[2])
        kept.remove(victim)
        dropped.append(victim[0])

    prompt = assemble(kept)
    if utf16_len(prompt) > budget:
        # Only non-droppable sections remain; shorten the free-text direction last.
        others = utf16_len(assemble([item for item in kept if item[0] != "direction"]))
        room = budget - others - 1
        kept = [
            (name, _truncate_words(text, room) if name == "direction" else text, priority, droppable)
            for name, text, priority, droppable in kept
        ]
        if "direction (shortened)" not in dropped:
            dropped.append("direction (shortened)")
        prompt = assemble(kept)
    return prompt, dropped


def presenter_receipt(prompt: str, dropped: list[str], budget: int) -> dict[str, Any]:
    return {"chars": utf16_len(prompt), "budget": budget, "dropped_sections": list(dropped)}
