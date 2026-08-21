from __future__ import annotations

from .models import ShotSpec


def choose_engine(shot: ShotSpec) -> str:
    if shot.engine_preference != "auto":
        return shot.engine_preference

    refs = shot.references
    if refs.motion_video:
        return "wan22"
    if refs.last_frame:
        return "ltx25"
    if refs.scene_image and refs.audio:
        return "seedance"
    if shot.camera.movement not in {"static", "locked", "none"}:
        return "ltx25"
    return "wan22"
