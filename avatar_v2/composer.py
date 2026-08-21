from __future__ import annotations

from uuid import uuid4

from .models import AvatarManifest, RenderJob, ShotSpec
from .policy import enforce_policy
from .router import choose_engine

DEFAULT_NEGATIVE = (
    "identity drift, face morphing, plastic skin, waxy skin, temporal flicker, "
    "warped anatomy, extra fingers, fused fingers, duplicate limbs, disappearing jewelry, "
    "changing clothing texture, background warping, rubbery motion, floating hair, "
    "synchronized fabric motion, inconsistent lighting, oversharpening"
)


def _time_plan(shot: ShotSpec) -> str:
    if not shot.action_beats:
        return "Maintain continuous natural human motion with realistic inertia and micro-adjustments."
    return " ".join(
        f"[{beat.start_s:0.3f}-{beat.end_s:0.3f}] {beat.description.strip()}"
        for beat in shot.action_beats
    )


def build_prompt(avatar: AvatarManifest, shot: ShotSpec) -> str:
    sections = [
        shot.user_prompt.strip(),
        "Preserve the exact same subject identity throughout the entire clip.",
        f"Camera: {shot.camera.framing}; movement: {shot.camera.movement}.",
        f"Temporal plan: {_time_plan(shot)}",
    ]
    if shot.camera.lens_equivalent_mm:
        sections.append(f"Camera lens equivalent: {shot.camera.lens_equivalent_mm}mm.")
    if shot.wardrobe:
        sections.append(f"Wardrobe continuity: {shot.wardrobe}.")
    if shot.environment:
        sections.append(f"Environment continuity: {shot.environment}.")
    if shot.wind.speed_mps > 0:
        sections.append(
            "Wind physics: "
            f"direction {shot.wind.direction}, approximately {shot.wind.speed_mps:g} m/s, "
            f"gust variation {shot.wind.gust_variation_pct}%. Hair roots remain comparatively "
            "constrained; loose strands and tips respond more strongly with inertial lag. "
            "Clothing responds according to material weight and attachment points rather than moving as one sheet."
        )
    if avatar.persistent_features:
        sections.append("Persistent identity anchors: " + "; ".join(avatar.persistent_features) + ".")
    sections.append(
        "Photorealistic skin microtexture, physically plausible body mechanics, natural blinking, "
        "breathing and posture corrections, coherent shadows and reflections, realistic motion blur."
    )
    return "\n".join(sections)


def _add_ref_pack(assets: dict[str, str | int | float | None], prefix: str, refs: list[str]) -> None:
    for index, path in enumerate(refs, start=1):
        assets[f"{prefix}_{index}"] = path


def build_job(avatar: AvatarManifest, shot: ShotSpec) -> RenderJob:
    enforce_policy(avatar, shot)
    selected = choose_engine(shot)
    init_image = shot.references.init_image or avatar.identity_refs[0]
    negative = DEFAULT_NEGATIVE
    if shot.negative_prompt:
        negative = f"{DEFAULT_NEGATIVE}, {shot.negative_prompt.strip()}"

    assets: dict[str, str | int | float | None] = {
        "PROMPT": build_prompt(avatar, shot),
        "NEGATIVE_PROMPT": negative,
        "INIT_IMAGE": init_image,
        "LAST_FRAME": shot.references.last_frame,
        "MOTION_VIDEO": shot.references.motion_video,
        "SCENE_IMAGE": shot.references.scene_image,
        "AUDIO": shot.references.audio,
        "WIDTH": shot.width,
        "HEIGHT": shot.height,
        "FPS": shot.fps,
        "FRAMES": shot.frames,
        "SEED": shot.seed,
    }
    _add_ref_pack(assets, "IDENTITY_REF", avatar.identity_refs)
    _add_ref_pack(assets, "BODY_REF", avatar.body_refs)
    _add_ref_pack(assets, "HAIR_REF", avatar.hair_refs)
    _add_ref_pack(assets, "WARDROBE_REF", avatar.wardrobe_refs)

    return RenderJob(
        job_id=f"{shot.shot_id}-{uuid4().hex[:8]}",
        avatar=avatar,
        shot=shot,
        selected_engine=selected,
        prompt=str(assets["PROMPT"]),
        negative_prompt=negative,
        asset_map=assets,
    )
