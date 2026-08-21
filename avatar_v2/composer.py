from __future__ import annotations

from uuid import uuid4

from .h3_runtime import actual_duration_s
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


def _physical_direction(avatar: AvatarManifest, shot: ShotSpec) -> list[str]:
    sections = [
        "Preserve the exact same subject identity throughout the entire clip.",
        f"Camera framing: {shot.camera.framing}; camera movement: {shot.camera.movement}.",
        f"Temporal action plan: {_time_plan(shot)}",
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
            f"gust variation {shot.wind.gust_variation_pct}%. Hair roots stay comparatively constrained; "
            "loose strands and tips respond more strongly with inertial lag. Clothing responds according to "
            "material weight and attachment points rather than moving as one rigid sheet."
        )
    if avatar.persistent_features:
        sections.append("Persistent identity anchors: " + "; ".join(avatar.persistent_features) + ".")
    sections.append(
        "Use live-action photorealism with natural skin microtexture, physically plausible body mechanics, "
        "natural blinking, breathing and posture corrections, coherent shadows/reflections, and realistic motion blur."
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


def _h3_reference_sets(
    avatar: AvatarManifest,
    shot: ShotSpec,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    refs = shot.references
    primary = refs.init_image or avatar.identity_refs[0]

    image_candidates: list[tuple[str | None, str]] = [(primary, "primary identity anchor")]
    image_candidates += [(p, "additional identity anchor") for p in avatar.identity_refs]
    image_candidates += [(p, "body and proportion reference") for p in avatar.body_refs]
    image_candidates += [(p, "hair identity and material reference") for p in avatar.hair_refs]
    image_candidates += [(p, "wardrobe reference") for p in avatar.wardrobe_refs]
    image_candidates.append((refs.scene_image, "environment and lighting reference"))
    image_candidates += [(p, "additional visual reference") for p in refs.extra_images]

    video_candidates: list[tuple[str | None, str]] = [(refs.motion_video, "motion and camera choreography reference")]
    video_candidates += [(p, "additional motion or temporal reference") for p in refs.extra_videos]

    audio_candidates: list[tuple[str | None, str]] = [(refs.audio, "voice, timing, or sound reference")]
    audio_candidates += [(p, "additional audio reference") for p in refs.extra_audios]

    return (
        _unique(image_candidates)[:9],
        _unique(video_candidates)[:3],
        _unique(audio_candidates)[:3],
    )


def _soundscape(shot: ShotSpec, has_audio_ref: bool = False) -> str:
    sounds: list[str] = []
    if shot.environment:
        sounds.append(f"Natural ambience appropriate to {shot.environment}")
    if shot.wind.speed_mps > 0:
        sounds.append("soft wind interaction with hair and clothing")
    sounds.append("subtle footsteps, breathing, fabric movement, and room or outdoor ambience as physically appropriate")
    if has_audio_ref:
        sounds.append("referenced voice or sound characteristics remain consistent where requested")
    return ". ".join(sounds) + "."


def _base_timeline(avatar: AvatarManifest, shot: ShotSpec) -> str:
    direction = " ".join(_physical_direction(avatar, shot))
    return f"[Shot 1] {shot.user_prompt.strip()} {direction} Keep one coherent shot unless the user explicitly requests a cut."


def _h3_fl2va_prompt(avatar: AvatarManifest, shot: ShotSpec) -> str:
    refs = shot.references
    duration = actual_duration_s(shot.duration_s)
    alignment = ""
    if refs.init_image and refs.last_frame:
        alignment = (
            "How the reference pictures align with the target video — <Picture 1> aligns with 0.00 seconds; "
            f"<Picture 2> aligns with {duration:.2f} seconds."
        )
    elif refs.init_image:
        alignment = "For the target video, at 0.00 seconds, <Picture 1> is the fully referenced opening frame."
    elif refs.last_frame:
        alignment = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> aligns with {duration:.2f} seconds as the final frame."
        )

    parts = []
    if alignment:
        parts.append(alignment)
    parts.extend(
        [
            f"integrated_multimodal_description: {_base_timeline(avatar, shot)}",
            f"overall_soundscape: {_soundscape(shot)}",
            "non_diegetic_music: N/A unless explicitly requested by the user.",
        ]
    )
    return "\n\n".join(parts)


def _h3_ref2va_prompt(avatar: AvatarManifest, shot: ShotSpec) -> str:
    images, videos, audios = _h3_reference_sets(avatar, shot)

    subject_lines: list[str] = []
    identity_pictures = [i for i, (_, role) in enumerate(images, start=1) if "identity" in role]
    body_pictures = [i for i, (_, role) in enumerate(images, start=1) if "body" in role or "proportion" in role]
    hair_pictures = [i for i, (_, role) in enumerate(images, start=1) if "hair" in role]
    wardrobe_pictures = [i for i, (_, role) in enumerate(images, start=1) if "wardrobe" in role]
    scene_pictures = [i for i, (_, role) in enumerate(images, start=1) if "environment" in role]

    identity_sources = ", ".join(f"<Picture {i}>" for i in identity_pictures) or "the supplied identity references"
    subject_lines.append(
        f"<Subject 1> is the target synthetic adult person. Preserve appearance and identity from {identity_sources}; "
        "do not inherit facial identity, body identity, clothing identity, or demographic traits from motion-reference videos."
    )
    if body_pictures:
        subject_lines.append(
            "<Subject 1> uses " + ", ".join(f"<Picture {i}>" for i in body_pictures) + " for body proportions and pose-scale consistency."
        )
    if hair_pictures:
        subject_lines.append(
            "<Subject 1> uses " + ", ".join(f"<Picture {i}>" for i in hair_pictures) + " for hairline, length, density, texture, and material behavior."
        )
    if wardrobe_pictures:
        subject_lines.append(
            "<Subject 1> uses " + ", ".join(f"<Picture {i}>" for i in wardrobe_pictures) + " for wardrobe appearance and continuity."
        )
    if scene_pictures:
        subject_lines.append(
            "<Subject 2> is the target environment derived from " + ", ".join(f"<Picture {i}>" for i in scene_pictures) + ", including layout, lighting, and background appearance."
        )
    for i, (_, role) in enumerate(videos, start=1):
        subject_lines.append(f"<Video {i}> supplies {role}; transfer motion structure without replacing <Subject 1>'s identity.")
    for i, (_, role) in enumerate(audios, start=1):
        subject_lines.append(f"<Audio {i}> supplies {role} without changing visual identity.")

    task_types = ["reference generation"]
    if audios:
        task_types.append("audio reference")
    summary = (
        f"[{' + '.join(task_types)}] Generate one photorealistic target video centered on <Subject 1>. "
        "Identity and appearance references have priority over motion references; motion, camera rhythm, environment, and audio are transferred only in their assigned roles."
    )

    retention: list[str] = [
        "<Subject 1> (entire video): fully_preserved - preserve face, body identity, skin details, hair identity, and persistent appearance across all frames."
    ]
    if scene_pictures:
        retention.append("<Subject 2> (entire video): fully_preserved - retain the referenced environment, layout, lighting logic, and spatial continuity.")
    for i, (_, role) in enumerate(images, start=1):
        relation = "fully_preserved" if any(token in role for token in ("identity", "body", "hair", "wardrobe", "environment")) else "weak_reference"
        retention.append(f"<Picture {i}>: {relation} - use only for its assigned role: {role}.")
    for i, (_, role) in enumerate(videos, start=1):
        retention.append(f"<Video {i}>: attribute_transfer - transfer {role} to <Subject 1> without transferring source identity.")
    for i, (_, role) in enumerate(audios, start=1):
        retention.append(f"<Audio {i}>: reference - follow {role}; do not copy unrelated audio characteristics.")

    return "\n\n".join(
        [
            "subject_definitions:\n" + "\n".join(subject_lines),
            "summary: " + summary,
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description: " + _base_timeline(avatar, shot),
            "overall_soundscape: " + _soundscape(shot, has_audio_ref=bool(audios)),
            "non_diegetic_music: N/A unless explicitly requested by the user.",
        ]
    )


def build_prompt(avatar: AvatarManifest, shot: ShotSpec, selected_engine: str | None = None) -> str:
    selected = selected_engine or choose_engine(shot)
    if selected == "h3_ref2va":
        return _h3_ref2va_prompt(avatar, shot)
    if selected == "h3_fl2va":
        return _h3_fl2va_prompt(avatar, shot)
    return "\n".join([shot.user_prompt.strip(), *_physical_direction(avatar, shot)])


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
