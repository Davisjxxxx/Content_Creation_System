from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INT8_TOKEN = "pruned_int8"


@dataclass(frozen=True)
class H3Recommendation:
    route: str
    model: str | None
    reason: str
    local_allowed: bool
    vram_gb: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "model": self.model,
            "reason": self.reason,
            "local_allowed": self.local_allowed,
            "vram_gb": self.vram_gb,
        }


def detect_vram_gb(system_stats: dict[str, Any]) -> float | None:
    devices = system_stats.get("devices") or system_stats.get("system", {}).get("devices") or []
    totals: list[float] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        raw = device.get("vram_total") or device.get("total_memory") or device.get("memory_total")
        if isinstance(raw, (int, float)) and raw > 0:
            totals.append(float(raw) / (1024 ** 3))
    return max(totals) if totals else None


def _find(models: list[str], *, mode: str, int8: bool | None = None) -> str | None:
    mode = mode.lower()
    candidates = [name for name in models if mode in name.lower() and "minimax_h3" in name.lower()]
    if int8 is True:
        candidates = [name for name in candidates if INT8_TOKEN in name.lower() or "int8" in name.lower()]
    elif int8 is False:
        candidates = [name for name in candidates if INT8_TOKEN not in name.lower() and "int8" not in name.lower()]
    return candidates[0] if candidates else None


def recommend_local_h3(
    *,
    mode: str,
    system_stats: dict[str, Any],
    diffusion_models: list[str],
    local_h3_authorized: bool,
    fallback_route: str = "wan22",
) -> H3Recommendation:
    """Recommend a local H3 variant without downloading or bypassing licensing.

    `local_h3_authorized` is an explicit user/admin assertion that their use of
    the open H3 weights is permitted in their territory or covered by a separate
    written MiniMax license. The app does not infer or bypass that requirement.
    """
    vram = detect_vram_gb(system_stats)
    if not local_h3_authorized:
        return H3Recommendation(
            route=fallback_route,
            model=None,
            reason="Local H3 open-weight use is not enabled. Use hosted H3 or the configured local fallback.",
            local_allowed=False,
            vram_gb=vram,
        )

    full = _find(diffusion_models, mode=mode, int8=False)
    int8 = _find(diffusion_models, mode=mode, int8=True)

    # Consumer 12 GB cards should prefer the pruned INT8 checkpoint when it is installed.
    if vram is not None and vram <= 16 and int8:
        return H3Recommendation(
            route=f"h3_{mode}",
            model=int8,
            reason=f"{vram:.1f} GB VRAM detected; selecting installed pruned INT8 H3 checkpoint.",
            local_allowed=True,
            vram_gb=vram,
        )

    if full:
        return H3Recommendation(
            route=f"h3_{mode}",
            model=full,
            reason="Full H3 checkpoint is installed and preferred for this device profile.",
            local_allowed=True,
            vram_gb=vram,
        )

    if int8:
        return H3Recommendation(
            route=f"h3_{mode}",
            model=int8,
            reason="Full checkpoint unavailable; selecting installed pruned INT8 H3 checkpoint.",
            local_allowed=True,
            vram_gb=vram,
        )

    return H3Recommendation(
        route=fallback_route,
        model=None,
        reason="No compatible local H3 diffusion checkpoint was discovered in ComfyUI.",
        local_allowed=True,
        vram_gb=vram,
    )
