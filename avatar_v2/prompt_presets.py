from __future__ import annotations

from typing import Any, Final


ADULT_PROMPT_NOTICE: Final[str] = (
    "Adult presets are available only for age-verified, consenting adults or "
    "synthetic adult personas with explicit adult provenance."
)

ADULT_PROMPT_DEFAULTS: Final[dict[str, str]] = {
    "roles": "confident_performer",
    "environments": "private_suite",
    "actions": "flirtatious_pose_sequence",
    "toys": "none",
    "moves": "slow_hip_sway",
    "positions": "standing_three_quarter",
}

ADULT_PROMPT_PRESETS: Final[dict[str, tuple[dict[str, str], ...]]] = {
    "roles": (
        {"id": "confident_performer", "label": "Confident adult performer", "prompt": "a confident adult performer engaging the camera with clear agency"},
        {"id": "glamour_model", "label": "Editorial glamour model", "prompt": "an adult glamour model holding polished editorial presence"},
        {"id": "romantic_lead", "label": "Romantic lead", "prompt": "an adult romantic lead conveying warmth, trust, and mutual interest"},
        {"id": "playful_partner", "label": "Playful partner", "prompt": "a playful consenting adult partner with relaxed, responsive expression"},
        {"id": "assertive_lead", "label": "Assertive scene lead", "prompt": "an assertive consenting adult leading the scene with controlled body language"},
        {"id": "receptive_partner", "label": "Receptive partner", "prompt": "a receptive consenting adult communicating comfort and active participation"},
    ),
    "environments": (
        {"id": "private_suite", "label": "Private luxury suite", "prompt": "a private luxury suite with warm practical lighting and closed curtains"},
        {"id": "boudoir_studio", "label": "Boudoir studio", "prompt": "a controlled boudoir studio with softbox key light and a clean fabric backdrop"},
        {"id": "modern_bedroom", "label": "Modern bedroom", "prompt": "a private modern bedroom with soft bedside lighting and uncluttered surfaces"},
        {"id": "candlelit_loft", "label": "Candlelit loft", "prompt": "a private candlelit loft with warm pools of light and deep background falloff"},
        {"id": "neon_lounge", "label": "Neon lounge", "prompt": "a private neon lounge with magenta-blue rim light and reflective accents"},
        {"id": "secluded_spa", "label": "Secluded spa", "prompt": "a secluded spa room with diffused light, towels, and subtle steam"},
    ),
    "actions": (
        {"id": "flirtatious_pose_sequence", "label": "Flirtatious pose sequence", "prompt": "perform a slow flirtatious pose sequence while maintaining eye contact"},
        {"id": "sensual_dance", "label": "Sensual dance", "prompt": "perform a restrained sensual dance with deliberate transitions"},
        {"id": "wardrobe_reveal", "label": "Wardrobe reveal", "prompt": "stage a gradual wardrobe reveal with clear, intentional hand placement"},
        {"id": "camera_tease", "label": "Approach and tease", "prompt": "approach the camera, pause, and tease with controlled expression and timing"},
        {"id": "suggestive_touch", "label": "Suggestive touch", "prompt": "use a slow suggestive self-touch sequence with clearly visible, natural hand motion"},
        {"id": "relaxed_recline", "label": "Relaxed recline", "prompt": "transition into a relaxed recline and hold a composed gaze toward camera"},
    ),
    "toys": (
        {"id": "none", "label": "None", "prompt": "no intimacy prop"},
        {"id": "silk_blindfold", "label": "Silk blindfold", "prompt": "a silk blindfold used consensually as a visible scene prop"},
        {"id": "soft_restraints", "label": "Soft restraints", "prompt": "soft satin restraints used consensually with relaxed hands and no distress"},
        {"id": "massage_oil", "label": "Massage oil", "prompt": "a small bottle of massage oil used as a continuity prop"},
        {"id": "wand_massager", "label": "Wand massager", "prompt": "a handheld wand massager used consensually as a clearly visible prop"},
        {"id": "feather_teaser", "label": "Feather teaser", "prompt": "a feather teaser used consensually with gentle, deliberate motion"},
    ),
    "moves": (
        {"id": "slow_hip_sway", "label": "Slow hip sway", "prompt": "use a slow hip sway with realistic balance and continuous weight transfer"},
        {"id": "body_wave", "label": "Controlled body wave", "prompt": "perform a controlled full-body wave with natural joint timing"},
        {"id": "turn_look_back", "label": "Turn and look back", "prompt": "turn away, settle the hips, then look back over the shoulder"},
        {"id": "crawl_forward", "label": "Crawl toward camera", "prompt": "move slowly toward the camera on hands and knees with stable anatomy"},
        {"id": "kneel_to_stand", "label": "Kneel to stand", "prompt": "rise smoothly from kneeling to standing with believable leverage"},
        {"id": "recline_and_rise", "label": "Recline and rise", "prompt": "recline, pause, then rise with continuous contact and realistic weight support"},
    ),
    "positions": (
        {"id": "standing_three_quarter", "label": "Standing three-quarter", "prompt": "finish in a standing three-quarter pose with relaxed shoulders"},
        {"id": "seated_edge", "label": "Seated at edge", "prompt": "finish seated at the edge of the bed with both feet naturally supported"},
        {"id": "kneeling_upright", "label": "Kneeling upright", "prompt": "finish kneeling upright with balanced hips and an elongated posture"},
        {"id": "reclining_side", "label": "Reclining on side", "prompt": "finish reclining on one side with natural limb overlap"},
        {"id": "hands_and_knees", "label": "Hands and knees", "prompt": "finish on hands and knees with realistic points of contact and neutral balance"},
        {"id": "lying_back", "label": "Lying back", "prompt": "finish lying back with knees bent and anatomically natural spacing"},
    ),
}


def _custom_options(custom_options: Any, category: str) -> list[dict[str, object]]:
    if not isinstance(custom_options, dict):
        return []
    built_in_ids = {option["id"] for option in ADULT_PROMPT_PRESETS[category]}
    seen = set(built_in_ids)
    output: list[dict[str, object]] = []
    for raw in custom_options.get(category, []):
        if not isinstance(raw, dict):
            continue
        option_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        if not option_id or not label or not prompt or option_id in seen:
            continue
        seen.add(option_id)
        output.append({"id": option_id, "label": label, "prompt": prompt, "custom": True})
    return output


def adult_prompt_presets_payload(custom_options: Any = None) -> dict[str, object]:
    """Return JSON-safe preset data without exposing mutable module constants."""

    return {
        "notice": ADULT_PROMPT_NOTICE,
        "defaults": dict(ADULT_PROMPT_DEFAULTS),
        "categories": {
            category: [dict(option) for option in options] + _custom_options(custom_options, category)
            for category, options in ADULT_PROMPT_PRESETS.items()
        },
    }
