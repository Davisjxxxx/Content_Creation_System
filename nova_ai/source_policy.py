from __future__ import annotations

from .models import SourceRef


TIER_LABELS = {
    1: "primary/high-authority",
    2: "specialist/reputable secondary",
    3: "creator/community/discovery signal",
    4: "unverified/viral/anonymous",
}


def source_weight(source: SourceRef) -> float:
    base = {1: 1.0, 2: 0.75, 3: 0.45, 4: 0.15}.get(source.source_tier, 0.1)
    if source.is_primary:
        base = min(1.0, base + 0.15)
    return base


def claim_source_quality(sources: list[SourceRef]) -> float:
    if not sources:
        return 0.0
    weights = sorted((source_weight(s) for s in sources), reverse=True)
    top = weights[0]
    corroboration = min(0.25, 0.05 * max(0, len(weights) - 1))
    return round(min(1.0, top + corroboration), 3)
