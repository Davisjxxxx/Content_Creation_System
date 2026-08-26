from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .library import AvatarProfile

ReferenceGroup = Literal["identity", "body", "anatomy"]


@dataclass(frozen=True)
class CoverageSlot:
    id: str
    label: str
    group: ReferenceGroup
    role: str
    framing: str
    direction: str
    adult_only: bool = False
    required: bool = True


# A neutral, production-reference package. Intimate slots are deliberately
# clinical rather than sexual: they are identity/anatomy records, not scenes.
COVERAGE_SLOTS: tuple[CoverageSlot, ...] = (
    CoverageSlot("identity_front", "Face · front", "identity", "front", "head-and-shoulders", "straight-on eye-level view"),
    CoverageSlot("identity_three_quarter_left", "Face · 3/4 left", "identity", "three_quarter_left", "head-and-shoulders", "three-quarter view toward subject-left"),
    CoverageSlot("identity_three_quarter_right", "Face · 3/4 right", "identity", "three_quarter_right", "head-and-shoulders", "three-quarter view toward subject-right"),
    CoverageSlot("identity_profile_left", "Face · left profile", "identity", "profile_left", "head-and-shoulders", "true subject-left profile"),
    CoverageSlot("identity_profile_right", "Face · right profile", "identity", "profile_right", "head-and-shoulders", "true subject-right profile"),
    CoverageSlot("body_full_front", "Body · full front", "body", "full_front", "full body", "straight-on front view in a neutral anatomical pose"),
    CoverageSlot("body_full_rear", "Body · full rear", "body", "rear", "full body", "straight-on rear view in the same neutral pose"),
    CoverageSlot("body_full_left", "Body · full left", "body", "left", "full body", "true subject-left side view"),
    CoverageSlot("body_full_right", "Body · full right", "body", "right", "full body", "true subject-right side view"),
    CoverageSlot("body_top_front", "Angle · top/front", "body", "top_front", "full body", "elevated camera looking down from the front"),
    CoverageSlot("body_top_rear", "Angle · top/rear", "body", "top_rear", "full body", "elevated camera looking down from the rear"),
    CoverageSlot("body_low_front", "Angle · bottom/front", "body", "low_front", "full body", "low camera looking upward from the front without distortion"),
    CoverageSlot("body_low_rear", "Angle · bottom/rear", "body", "low_rear", "full body", "low camera looking upward from the rear without distortion"),
    CoverageSlot("anatomy_face", "Detail · face and skin", "anatomy", "face_skin", "close detail", "evenly lit face showing natural pores, freckles, moles, scars, and makeup-free skin"),
    CoverageSlot("anatomy_torso_front", "Detail · torso and navel", "anatomy", "torso_front", "torso detail", "straight-on torso view including chest, abdomen, navel, waist, and hip landmarks", adult_only=True),
    CoverageSlot("anatomy_torso_rear", "Detail · back", "anatomy", "torso_rear", "torso detail", "straight-on back view including shoulders, spine landmarks, waist, and scars or tattoos", adult_only=True),
    CoverageSlot("anatomy_pelvis_front", "Anatomy · pelvis front", "anatomy", "pelvis_front", "clinical anatomy detail", "neutral straight-on pelvis and external genital anatomy reference", adult_only=True),
    CoverageSlot("anatomy_pelvis_three_quarter", "Anatomy · pelvis 3/4", "anatomy", "pelvis_three_quarter", "clinical anatomy detail", "neutral three-quarter pelvis and external genital anatomy reference", adult_only=True),
    CoverageSlot("anatomy_pelvis_rear", "Anatomy · pelvis rear", "anatomy", "pelvis_rear", "clinical anatomy detail", "neutral rear pelvis, gluteal, cleft, and perineal landmarks", adult_only=True),
    CoverageSlot("anatomy_hands", "Detail · hands", "anatomy", "hands", "close detail", "both hands relaxed, palms and backs visible, correct fingers and distinguishing marks"),
    CoverageSlot("anatomy_feet", "Detail · feet", "anatomy", "feet", "close detail", "both feet in neutral stance, tops and soles visible, correct toes and distinguishing marks"),
)

_BY_ID = {slot.id: slot for slot in COVERAGE_SLOTS}

APPEARANCE_FIELDS = (
    "face",
    "body_type",
    "skin_details",
    "freckles_moles",
    "scars",
    "tattoos",
    "anatomy_details",
)


def designer_spec() -> dict[str, Any]:
    return {
        "slots": [asdict(slot) for slot in COVERAGE_SLOTS],
        "appearance_fields": list(APPEARANCE_FIELDS),
        "notice": (
            "Use only synthetic adults or age-verified consenting adults. Generated views are candidates: "
            "accept each one only after checking identity, body proportions, landmarks, and anatomy."
        ),
    }


def slot_by_id(slot_id: str) -> CoverageSlot:
    try:
        return _BY_ID[str(slot_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown avatar designer slot: {slot_id}") from exc


def slot_path(profile: AvatarProfile, slot: CoverageSlot) -> str:
    refs = {
        "identity": profile.identity_refs,
        "body": profile.body_refs,
        "anatomy": profile.anatomy_refs,
    }[slot.group]
    return str(refs.get(slot.role) or "")


def coverage_report(profile: AvatarProfile) -> dict[str, Any]:
    items = [
        {
            **asdict(slot),
            "path": slot_path(profile, slot),
            "filled": bool(slot_path(profile, slot)),
        }
        for slot in COVERAGE_SLOTS
    ]
    required = [item for item in items if item["required"]]
    filled = sum(bool(item["filled"]) for item in required)
    return {
        "filled": filled,
        "required": len(required),
        "percent": round((filled / len(required)) * 100) if required else 100,
        "missing": [item["id"] for item in required if not item["filled"]],
        "items": items,
    }


def build_designer_prompt(
    slot_id: str,
    appearance: dict[str, str] | None = None,
    apparent_age_years: int | None = None,
    render_mode: Literal["fl2va", "body_ref2va", "guided_ref2va"] = "fl2va",
    guide_labels: list[str] | None = None,
    component_labels: list[str] | None = None,
) -> str:
    slot = slot_by_id(slot_id)
    clean = {
        key: str(value).strip()
        for key, value in (appearance or {}).items()
        if key in APPEARANCE_FIELDS and str(value).strip()
    }
    details = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in clean.items())
    anatomy = (
        "This is a neutral clinical adult anatomy reference, not a sexual action. "
        if slot.adult_only
        else "This is a neutral identity and body reference. "
    )
    age_anchor = (
        f"The target is visibly a {apparent_age_years}-year-old adult. Preserve age-specific facial structure, "
        "skin texture, soft-tissue distribution, and body maturity consistently; do not make the subject look "
        "younger or older. "
        if apparent_age_years is not None
        else ""
    )
    guided = render_mode == "guided_ref2va"
    body_referenced = render_mode == "body_ref2va"
    source_direction = (
        "Create an H3 photorealistic avatar reference view using the selected image only as the target identity anchor. "
        "Additional anatomy images are guide-only clinical topology references, not depictions of the target person. "
        "Use them only for plausible landmarks, surface continuity, and soft-tissue structure; never copy their face, "
        "identity, age, ethnicity, skin tone, body shape, proportions, hair, marks, scars, tattoos, or clothing. "
        if guided
        else (
            "Create an H3 photorealistic avatar reference view using the primary image as the exact identity anchor. "
            "Additional body images can include same-avatar proportion references and explicitly selected appearance-component "
            "examples. Preserve the primary image for facial identity. Use same-avatar images for stable proportions and marks; "
            "use selected component examples only for their named morphology, never for face, identity, age, ethnicity, skin marks, or clothing. "
            if body_referenced
            else "Create an H3 photorealistic avatar reference turn from the exact selected opening image. "
        )
    )
    transition = (
        f"Establish one coherent {slot.framing}: {slot.direction}. Use restrained movement, stable balance, and "
        "consistent three-dimensional body structure. Hold the requested view nearly motionless for the final 30% so "
        "the final frame is a clean profile candidate. "
        if guided or body_referenced
        else (
            "For the first 15% of the clip, hold the exact source composition and identity. "
            f"From 15% to 70%, make one slow continuous camera-and-pose transition into a {slot.framing}: {slot.direction}. "
            "Use restrained movement, stable balance, and consistent three-dimensional body structure throughout the turn. "
            "By 70%, the requested view is fully established. Hold that target view nearly motionless for the final 30% so "
            "the final frame is a clean profile candidate. "
        )
    )
    guide_note = (
        " Guide-only references selected: " + "; ".join(str(label).strip() for label in (guide_labels or []) if str(label).strip()) + "."
        if guided and guide_labels
        else ""
    )
    component_note = (
        " Selected appearance components: " + "; ".join(str(label).strip() for label in component_labels if str(label).strip()) + "."
        if body_referenced and component_labels
        else ""
    )
    return (
        f"{source_direction}{anatomy}{age_anchor}{transition}Preserve the exact facial identity, adult age, ethnicity, body type, "
        "skeletal proportions, skin tone, hairline, and every visible distinguishing mark from the source image. "
        "Use a neutral reference-sheet presentation, 70 mm lens perspective, flat studio lighting, plain gray background, "
        "centered composition, and complete requested anatomy in frame. Keep left/right body landmarks consistent through "
        "the turn. Do not beautify, reshape, add, remove, mirror, duplicate, or relocate freckles, moles, scars, tattoos, "
        "navel, limbs, joints, or anatomy. No cuts, no sudden reframing, and no motion blur in the final hold."
        + (f" Recorded appearance manifest: {details}." if details else "")
        + guide_note
        + component_note
    )
