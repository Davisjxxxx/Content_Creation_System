from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .h3_access import recommend_local_h3
from .h3_runtime import preset_by_id, valid_frame_count


@dataclass(frozen=True)
class RuntimeProfile:
    id: str
    label: str
    description: str
    preferred_preset: str
    steps: int
    cfg: float
    force_int8: bool = False
    prefer_full: bool = False
    fallback_engine: str = "wan22"


PROFILES: tuple[RuntimeProfile, ...] = (
    RuntimeProfile(
        "auto",
        "Auto",
        "Detect VRAM and installed H3 models; prefer INT8 on 12–16 GB cards.",
        "vertical_4070",
        20,
        1.0,
    ),
    RuntimeProfile(
        "quality",
        "H3 Quality",
        "Prefer a non-INT8 H3 checkpoint when one is installed and the GPU can support it.",
        "vertical_4070",
        24,
        1.0,
        prefer_full=True,
    ),
    RuntimeProfile(
        "int8_12gb",
        "H3 INT8 · 12 GB",
        "Force the pruned INT8 checkpoint. This is the primary RTX 4070 experiment profile.",
        "vertical_4070",
        20,
        1.0,
        force_int8=True,
    ),
    RuntimeProfile(
        "low_memory",
        "H3 Low Memory",
        "INT8 + smaller canvas + fewer steps. Useful for determining whether a workflow can fit before quality tuning.",
        "vertical_fast",
        12,
        1.0,
        force_int8=True,
    ),
    RuntimeProfile(
        "low_memory_quality",
        "H3 Low Memory · Quality",
        "INT8 at the 4070 canvas with a reduced step count; middle ground between fit and quality.",
        "vertical_4070",
        16,
        1.0,
        force_int8=True,
    ),
    RuntimeProfile(
        "wan_fallback",
        "Wan 2.2 Fallback",
        "Skip H3 and use the configured Wan 2.2 workflow.",
        "vertical_4070",
        20,
        1.0,
        fallback_engine="wan22",
    ),
)


def profile_by_id(profile_id: str) -> RuntimeProfile:
    return next((profile for profile in PROFILES if profile.id == profile_id), PROFILES[0])


def payload() -> list[dict[str, Any]]:
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "description": profile.description,
            "preferred_preset": profile.preferred_preset,
            "steps": profile.steps,
            "cfg": profile.cfg,
        }
        for profile in PROFILES
    ]


def _int8_model(models: list[str], mode: str) -> str | None:
    mode = mode.lower()
    return next(
        (
            name
            for name in models
            if "minimax_h3" in name.lower()
            and mode in name.lower()
            and ("pruned_int8" in name.lower() or "int8" in name.lower())
        ),
        None,
    )


def _full_model(models: list[str], mode: str) -> str | None:
    mode = mode.lower()
    return next(
        (
            name
            for name in models
            if "minimax_h3" in name.lower()
            and mode in name.lower()
            and "int8" not in name.lower()
        ),
        None,
    )


def resolve_profile(
    *,
    profile_id: str,
    engine: str,
    system_stats: dict[str, Any],
    diffusion_models: list[str],
    local_h3_authorized: bool,
    requested_preset: str | None,
    duration_s: float,
) -> dict[str, Any]:
    profile = profile_by_id(profile_id)
    if profile.id == "wan_fallback":
        return {
            "profile": profile.id,
            "engine": "wan22",
            "model": None,
            "preset": requested_preset or profile.preferred_preset,
            "steps": profile.steps,
            "cfg": profile.cfg,
            "reason": "Wan fallback selected explicitly.",
        }

    mode = "ref2va" if engine == "h3_ref2va" else "fl2va"
    recommendation = recommend_local_h3(
        mode=mode,
        system_stats=system_stats,
        diffusion_models=diffusion_models,
        local_h3_authorized=local_h3_authorized,
        fallback_route=profile.fallback_engine,
    )

    chosen = recommendation.model
    reason = recommendation.reason
    if local_h3_authorized:
        if profile.force_int8:
            forced = _int8_model(diffusion_models, mode)
            if forced:
                chosen = forced
                reason = "Profile forced the installed pruned INT8 H3 checkpoint."
            else:
                reason = "INT8 profile selected, but no matching INT8 H3 checkpoint is installed."
        elif profile.prefer_full:
            full = _full_model(diffusion_models, mode)
            if full:
                chosen = full
                reason = "Quality profile selected the installed non-INT8 H3 checkpoint."

    preset_id = requested_preset or profile.preferred_preset
    if profile.id == "low_memory":
        # Preserve orientation while forcing the smallest practical canvas.
        requested = preset_by_id(preset_id)
        preset_id = "vertical_fast" if requested.height >= requested.width else "landscape_fast"
    preset = preset_by_id(preset_id)

    return {
        "profile": profile.id,
        "engine": recommendation.route if recommendation.route != f"h3_{mode}" else engine,
        "model": chosen,
        "preset": preset.id,
        "width": preset.width,
        "height": preset.height,
        "fps": 24,
        "frames": valid_frame_count(duration_s),
        "steps": profile.steps,
        "cfg": profile.cfg,
        "reason": reason,
        "vram_gb": recommendation.vram_gb,
        "local_h3_allowed": recommendation.local_allowed,
    }
