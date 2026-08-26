from __future__ import annotations

from .models import ShotSpec


def choose_engine(shot: ShotSpec) -> str:
    if shot.engine_preference != "auto":
        return shot.engine_preference

    refs = shot.references
    has_omni_refs = any(
        [
            refs.motion_video,
            refs.scene_image,
            refs.audio,
            refs.extra_images,
            refs.extra_videos,
            refs.extra_audios,
        ]
    )
    if has_omni_refs:
        return "h3_ref2va"
    if refs.init_image or refs.last_frame:
        return "h3_fl2va"
    return "h3_fl2va"
