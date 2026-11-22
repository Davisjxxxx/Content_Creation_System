"""BRAND_INSTRUCTOR_PHOTOREAL_V1: the reusable fictional presenter profile.

The character is fictional and synthetic: she is not based on, and must not resemble, any
real person. Prompts describe the target look, but prompt text alone cannot guarantee
anatomical or temporal realism. Every generated reference is a candidate that needs manual
approval, and the realism QA suite plus human review decide whether output is usable.
"""

from __future__ import annotations

from typing import Any

PROFILE_ID = "brand_instructor_photoreal_v1"  # vault ids are lowercase
DISPLAY_NAME = "BRAND_INSTRUCTOR_PHOTOREAL_V1"
SPEC_VERSION = 1

CHARACTER_SPEC: dict[str, Any] = {
    "fictional": True,
    "subject_kind": "synthetic",
    "sex": "female",
    "apparent_age_years": 26,
    "persona": "attractive, intelligent, approachable, natural and professional",
    "hair": "natural brunette hair with realistic individual strands",
    "eyes": "expressive brown eyes",
    "face": "believable facial proportions with realistic, subtle facial asymmetry",
    "skin": "authentic skin texture with visible pores and fine natural variation",
    "makeup": "subtle, natural makeup",
    "lips": "natural lip shape",
    "wardrobe": "professional smart-casual clothing",
    "framing": "chest-up",
    "lighting": "soft natural studio lighting",
    "background": "neutral background",
    "gaze": "direct eye contact with the camera",
    "presence": "calm, confident presentation",
}

# Positive identity description, written so it also works for providers that have no
# negative-prompt field (Runway's image and video models in Avatar V2's registry).
IDENTITY_PROMPT = (
    "A fictional 26-year-old woman who looks attractive, intelligent, approachable, natural and professional. "
    "Natural brunette hair with realistic individual strands and a few soft flyaways; expressive brown eyes; "
    "believable facial proportions with subtle, realistic asymmetry; authentic unretouched skin texture with visible "
    "pores and fine natural variation; subtle natural makeup; natural lip shape. Professional smart-casual clothing. "
    "Chest-up framing, soft natural studio lighting, neutral background, direct eye contact with the camera, calm and "
    "confident presentation."
)

# Condensed form that leaves room for the @presenter binding, slot direction and realism guards
# inside 1,000-character prompt limits (Gen-4 Image, Gen-4.5).
IDENTITY_PROMPT_COMPACT = (
    "Fictional 26-year-old woman: attractive, intelligent, approachable, natural and professional. Natural brunette "
    "hair with realistic strands, expressive brown eyes, believable proportions with subtle natural asymmetry, "
    "authentic skin texture with visible pores, subtle makeup, natural lips, smart-casual professional clothing. Soft "
    "natural studio light, neutral background, direct eye contact, calm and confident."
)

# Studio "looks": alternative identity descriptions for designing the master photo. "natural" is
# the default because the brief asks for subtle makeup and real skin; "original" is the text above.
LOOKS: dict[str, dict[str, str]] = {
    "natural": {
        "label": "Natural",
        "hint": "Everyday look, very little makeup",
        "prompt": (
            "Fictional 26-year-old woman: approachable, intelligent and natural, attractive in an everyday way. "
            "Natural brunette hair without highlights, realistic strands, brown eyes, believable proportions with "
            "subtle asymmetry, matte skin with visible pores, minimal makeup and no eyeliner, natural brows and lip "
            "colour, smart-casual professional clothing. Soft natural studio light, neutral background, calm and "
            "confident."
        ),
    },
    "polished": {
        "label": "Polished",
        "hint": "Camera-ready, soft makeup",
        "prompt": (
            "Fictional 26-year-old woman: polished, intelligent, approachable and professional. Neatly styled natural "
            "brunette hair with realistic strands, brown eyes, believable proportions with subtle asymmetry, real skin "
            "texture with visible pores, soft professional makeup, natural lips, tailored smart-casual clothing. Soft "
            "natural studio light, neutral background, calm and confident."
        ),
    },
    "original": {
        "label": "Original",
        "hint": "The BRAND description as written",
        "prompt": IDENTITY_PROMPT_COMPACT,
    },
}
DEFAULT_LOOK = "natural"
LOOK_EXTRA_MAX = 160


def look_prompt(look: str, extra: str = "") -> str:
    """Identity description for a Studio look, plus the user's optional extra details."""
    base = LOOKS.get(str(look), LOOKS[DEFAULT_LOOK])["prompt"]
    extra = " ".join(str(extra or "").split())[:LOOK_EXTRA_MAX].strip()
    if not extra:
        return base
    return f"{base} {extra if extra[-1] in '.!?' else extra + '.'}"


# Positive restatement of the exclusions for providers without negative prompts.
REALISM_GUARDS = (
    "Skin keeps its real texture with a natural matte sheen rather than an airbrushed finish; the face keeps subtle "
    "natural asymmetry; features stay true to life in size and proportion; each eye shows one soft, natural catchlight "
    "from the key light."
)

# Used verbatim as a negative prompt by providers that support one (local SDXL/Juggernaut and Wan
# through ComfyUI). For Runway models the same intent is carried by REALISM_GUARDS instead.
EXCLUSIONS: tuple[str, ...] = (
    "excessive skin smoothing",
    "airbrushed skin",
    "artificial perfect symmetry",
    "glossy plastic skin",
    "waxy skin",
    "exaggerated features",
    "oversized eyes",
    "overfilled lips",
    "artificial eye reflections",
    "duplicated catchlights",
    "glassy doll eyes",
    "CGI look",
    "beauty filter",
    "over-sharpening halos",
    "extra fingers",
    "fused fingers",
    "text",
    "logo",
    "watermark",
)
NEGATIVE_PROMPT = ", ".join(EXCLUSIONS)

PERSISTENT_FEATURES: tuple[str, ...] = (
    "natural brunette hair with realistic individual strands",
    "expressive brown eyes",
    "subtle natural facial asymmetry",
    "authentic skin texture with visible pores",
    "subtle natural makeup",
    "natural lip shape",
)

# The eight reviewable reference types from the brief (left and right profiles are separate
# slots). The first five reuse the Avatar Designer's identity slots; the rest are presenter slots.
REFERENCE_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "id": "identity_front",
        "role": "front",
        "label": "Straight-on portrait (master identity)",
        "master": True,
        "prompt": "Straight-on chest-up portrait, head and shoulders square to the camera, relaxed neutral expression, "
        "direct eye contact.",
    },
    {
        "id": "identity_three_quarter_left",
        "role": "three_quarter_left",
        "label": "Three-quarter left",
        "prompt": "Chest-up three-quarter view with the face turned about 45 degrees toward her left, eyes to the camera, "
        "neutral expression.",
    },
    {
        "id": "identity_three_quarter_right",
        "role": "three_quarter_right",
        "label": "Three-quarter right",
        "prompt": "Chest-up three-quarter view with the face turned about 45 degrees toward her right, eyes to the camera, "
        "neutral expression.",
    },
    {
        "id": "identity_profile_left",
        "role": "profile_left",
        "label": "Left profile",
        "prompt": "True left profile, the left side of the face toward the camera, neutral expression, chest-up, same "
        "lighting and background.",
    },
    {
        "id": "identity_profile_right",
        "role": "profile_right",
        "label": "Right profile",
        "prompt": "True right profile, the right side of the face toward the camera, neutral expression, chest-up, same "
        "lighting and background.",
    },
    {
        "id": "presenter_expression_neutral",
        "role": "expression_neutral",
        "label": "Neutral expression",
        "prompt": "Chest-up, facing the camera with a calm neutral expression, lips relaxed and closed.",
    },
    {
        "id": "presenter_expression_smile",
        "role": "expression_smile",
        "label": "Natural smile",
        "prompt": "Chest-up, facing the camera with a natural, gentle smile that reaches the eyes; teeth only slightly visible.",
    },
    {
        "id": "presenter_expression_speaking",
        "role": "expression_speaking",
        "label": "Speaking expression",
        "prompt": "Chest-up, mid-sentence while explaining a point, mouth naturally open on a vowel, engaged eyes, relaxed brows.",
    },
    {
        "id": "presenter_upper_body",
        "role": "upper_body_instructional",
        "label": "Upper-body instructional framing",
        "prompt": "Waist-up instructional framing, an open-hand explanatory gesture at chest height with relaxed, natural "
        "fingers, facing the camera.",
    },
)
_SLOTS_BY_ID = {slot["id"]: slot for slot in REFERENCE_SLOTS}

# Voice identity is stored separately from visual identity and can be swapped without
# touching the face. The preset voice is left empty on purpose: audition and choose one.
VOICE_PROFILE: dict[str, Any] = {
    "id": "brand_instructor_voice_v1",
    "label": "BRAND Instructor narration voice",
    "independent_of_visual_identity": True,
    "source": "runway_tts",
    "model": "eleven_v4",
    "voice": {"type": "runway-preset", "presetId": ""},
    "design_prompt": (
        "Warm, clear, confident female presenter voice in her mid-twenties with a clear, neutral accent, a calm "
        "medium pace, friendly and professional delivery, natural breaths and no vocal fry."
    ),
    "speed": 0.95,
    "stability": 0.6,
    "similarity_boost": 0.75,
    "style": 0.15,
    "language_code": "en",
    "target_wpm": [140, 160],
    "pronunciations": {},
}

# Recommended provider settings. Paid generation still needs explicit approval per job.
RECOMMENDED: dict[str, Any] = {
    "master_candidates": {
        "task": "still",
        "model": "gen4_image",
        "ratio": "1080:1440",
        "seeds": [41001, 41002, 41003, 41004],
        "estimated_credits": 32,
        "note": "Text-only design (no references). Review all four and approve one as the master.",
    },
    "reference_slots": {
        "task": "variations",
        "model": "gen4_image",
        "ratio": "1080:1440",
        "seed": 41001,
        "estimated_credits_per_slot": 8,
        "note": "Each slot binds the approved master as @presenter; candidates wait for manual approval.",
    },
    "talking_head": {
        "task": "avatar_video",
        "model": "gwm1_avatars",
        "speech": "audio",
        "note": "Runway custom avatar created from the approved master; approved narration audio drives lip sync.",
    },
    "performance_transfer": {
        "task": "performance",
        "model": "act_two",
        "ratio": "720:1280",
        "body_control": True,
        "expression_intensity": 3,
        "output_audio": "voiceover",
        "note": "Driving footage audio is removed before upload; the approved voiceover is muxed locally.",
    },
    "broll_motion": {"task": "image_to_video", "model": "gen4.5", "ratio": "720:1280", "duration": 5},
}


def slot_by_id(slot_id: str) -> dict[str, Any]:
    try:
        return _SLOTS_BY_ID[str(slot_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown BRAND reference slot '{slot_id}'.") from exc


def profile_entry() -> dict[str, Any]:
    """Vault record for the presenter. Adult content classes stay locked (no consent flag)."""
    return {
        "avatar_id": PROFILE_ID,
        "display_name": DISPLAY_NAME,
        "subject_kind": "synthetic",
        "sex": "female",
        "apparent_age_years": 26,
        "age_verified_18_plus": True,
        "consent_confirmed": False,
        "nude_mode": "clothed",
        "persistent_features": list(PERSISTENT_FEATURES),
        "notes": (
            "Fictional synthetic presenter for BRAND; not based on any real person. The master identity and every "
            "reference slot require manual approval. Voice identity is configured separately."
        ),
    }


def spec_payload() -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "display_name": DISPLAY_NAME,
        "spec_version": SPEC_VERSION,
        "character": dict(CHARACTER_SPEC),
        "identity_prompt": IDENTITY_PROMPT,
        "realism_guards": REALISM_GUARDS,
        "exclusions": list(EXCLUSIONS),
        "negative_prompt": NEGATIVE_PROMPT,
        "persistent_features": list(PERSISTENT_FEATURES),
        "reference_slots": [dict(slot) for slot in REFERENCE_SLOTS],
        "voice_profile": dict(VOICE_PROFILE),
        "recommended": RECOMMENDED,
        "limits_notice": (
            "Prompt instructions alone cannot guarantee anatomical or temporal realism. Approve references manually and "
            "run the realism QA suite with human review before treating output as production-ready."
        ),
    }
