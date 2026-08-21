from __future__ import annotations

from dataclasses import dataclass

H3_FPS = 24
CANVAS_MULTIPLE = 32
FRAME_BLOCK = 17
FRAME_OFFSET = 5
MAX_DURATION_S = 15.0
MAX_SHORT_EDGE = 768
MAX_LONG_EDGE = 1344


@dataclass(frozen=True)
class H3Preset:
    id: str
    label: str
    width: int
    height: int
    tier: str


PRESETS: tuple[H3Preset, ...] = (
    H3Preset("vertical_fast", "Vertical · Fast · 352×608", 352, 608, "fast"),
    H3Preset("vertical_4070", "Vertical · RTX 4070 · 480×864", 480, 864, "recommended"),
    H3Preset("vertical_max", "Vertical · Max · 768×1344", 768, 1344, "max"),
    H3Preset("landscape_fast", "Landscape · Fast · 608×352", 608, 352, "fast"),
    H3Preset("landscape_4070", "Landscape · RTX 4070 · 864×480", 864, 480, "recommended"),
    H3Preset("landscape_quality", "Landscape · Quality · 960×544", 960, 544, "quality"),
    H3Preset("landscape_max", "Landscape · Max · 1344×768", 1344, 768, "max"),
    H3Preset("square_fast", "Square · Fast · 512×512", 512, 512, "fast"),
    H3Preset("square_quality", "Square · Quality · 768×768", 768, 768, "quality"),
)


def snap_canvas(width: int, height: int) -> tuple[int, int]:
    width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return width, height


def valid_frame_count(duration_s: float) -> int:
    duration_s = max(1 / H3_FPS, min(float(duration_s), MAX_DURATION_S))
    requested = max(FRAME_OFFSET, round(duration_s * H3_FPS))
    return requested + (FRAME_OFFSET - (requested % FRAME_BLOCK)) % FRAME_BLOCK


def actual_duration_s(duration_s: float) -> float:
    return valid_frame_count(duration_s) / H3_FPS


def preset_by_id(preset_id: str) -> H3Preset:
    return next((preset for preset in PRESETS if preset.id == preset_id), PRESETS[1])


def validate_h3_canvas(width: int, height: int) -> list[str]:
    issues: list[str] = []
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        issues.append("H3 canvas dimensions should be multiples of 32.")
    short_edge, long_edge = sorted((width, height))
    if short_edge > MAX_SHORT_EDGE:
        issues.append(f"Short edge {short_edge}px exceeds the native {MAX_SHORT_EDGE}px target.")
    if long_edge > MAX_LONG_EDGE:
        issues.append(f"Long edge {long_edge}px exceeds the native {MAX_LONG_EDGE}px target.")
    return issues


def presets_payload() -> list[dict[str, str | int]]:
    return [
        {
            "id": preset.id,
            "label": preset.label,
            "width": preset.width,
            "height": preset.height,
            "tier": preset.tier,
        }
        for preset in PRESETS
    ]
