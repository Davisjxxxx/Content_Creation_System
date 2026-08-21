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


def _base_prompt(avatar: AvatarManifest, shot: ShotSpec) -> list[str]:
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
    return sections


def _unique(items: list[tuple[str | None, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for path, role in items:
        if not path or path in seen:
            continue
        seen.add(path)
        output.append((path, role))
    return output


def _h3_reference_sets(avatar: AvatarManifest, shot: ShotSpec) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    refs = shot.references
    primary = refs.init_image or avatar.identity_refs[0]

    image_candidates: list[tuple[str | None, str]] = [(primary, "primary identity anchor")]
    image_candidates += [(p, "additional identity anchor") for p in avatar.identity_refs]
    image_candidates += [(p, "body/proportion reference") for p in avatar.body_refs]
    image_candidates += [(p, "hair identity and material reference") for p in avatar.hair_refs]
    image_candidates += [(p, "wardrobe reference") for p in avatar.wardrobe_refs]
    image_candidates.append((refs.scene_image, "environment and lighting reference only"))
    image_candidates += [(p, "additional visual reference") for p in refs.extra_images]

    video_candidates: list[tuple[str | None, str]] = [(refs.motion_video, "motion and camera choreography only")]
    video_candidates += [(p, "additional motion/video reference") for p in refs.extra_videos]

    audio_candidates: list[tuple[str | None, str]] = [(refs.audio, "voice, timing, or sound reference")]
    audio_candidates += [(p, "additional audio reference") for p in refs.extra_audios]

    return (
        _unique(image_candidates)[:9],
        _unique(video_candidates)[:3],
        _unique(audio_candidates)[:3],
    )


def _h3_binding_text(images: list[tuple[str, str]], videos: list[tuple[str, str]], audios: list[tuple[str, str]]) -> str:
    bindings: list[str] = []
    for i, (_, role) in enumerate(images, start=1):
        bindings.append(f"<Picture {i}> = {role}")
    for i, (_, role) in enumerate(videos, start=1):
        bindings.append(f"<Video {i}> = {role}")
    for i, (_, role) in enumerate(audios, start=1):
        bindings.append(f"<Audio {i}> = {role}")
    if not bindings:
        return ""
    return (
        "Reference binding: "
        + "; ".join(bindings)
        + ". Respect each reference's assigned role; motion references must not replace the avatar identity."
    )


def build_prompt(avatar: AvatarManifest, shot: ShotSpec, selected_engine: str | None = None) -> str:
    selected = selected_engine or choose_engine(shot)
    sections = _base_prompt(avatar, shot)
    if selected == "h3_ref2va":
        images, videos, audios = _h3_reference_sets(avatar, shot)
        binding = _h3_binding_text(images, videos, audios)
        if binding:
            sections.insert(0, binding)
    return "\n".join(sections)


def _add_ref_pack(assets: dict[str, str | int | float | None], prefix: str, refs: list[str]) -> None:
    for index, path in enumerate(refs, start=1):
        assets[f"{prefix}_{index}"] = path


def _add_h3_refs(assets: dict[str, str | int | float | None], avatar: AvatarManifest, shot: ShotSpec) -> None:
    images, videos, audios = _h3_reference_sets(avatar, shot)
    for index, (path, _) in enumerate(images, start=1):
        assets[f"H3_PICTURE_{index}"] = path
    for index, (path, _) in enumerate(videos, start=1):
        assets[f"H3_VIDEO_{index}"] = path
    for index, (path, _) in enumerate(audios, start=1):
        assets[f"H3_AUDIO_{index}"] = path
    assets["H3_PICTURE_COUNT"] = len(images)
    assets["H3_VIDEO_COUNT"] = len(videos)
    assets["H3_AUDIO_COUNT"] = len(audios)


def build_job(avatar: AvatarManifest, shot: ShotSpec) -> RenderJob:
    enforce_policy(avatar, shot)
    selected = choose_engine(shot)
    init_image = shot.references.init_image or avatar.identity_refs[0]
    negative = DEFAULT_NEGATIVE
    if shot.negative_prompt:
        negative = f"{DEFAULT_NEGATIVE}, {shot.negative_prompt.strip()}"

    prompt = build_prompt(avatar, shot, selected)
    assets: dict[str, str | int | float | None] = {
        "PROMPT": prompt,
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
        "DURATION": shot.duration_s,
        "SEED": shot.seed,
    }
    _add_ref_pack(assets, "IDENTITY_REF", avatar.identity_refs)
    _add_ref_pack(assets, "BODY_REF", avatar.body_refs)
    _add_ref_pack(assets, "HAIR_REF", avatar.hair_refs)
    _add_ref_pack(assets, "WARDROBE_REF", avatar.wardrobe_refs)
    if selected == "h3_ref2va":
        _add_h3_refs(assets, avatar, shot)

    return RenderJob(
        job_id=f"{shot.shot_id}-{uuid4().hex[:8]}",
        avatar=avatar,
        shot=shot,
        selected_engine=selected,
        prompt=prompt,
        negative_prompt=negative,
        asset_map=assets,
    )
