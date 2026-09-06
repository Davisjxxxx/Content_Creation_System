from __future__ import annotations

from .models import StoryScores


DEFAULT_WEIGHTS = {
    "relevance": 0.13,
    "novelty": 0.10,
    "timeliness": 0.10,
    "evidence_quality": 0.16,
    "visual_potential": 0.10,
    "audience_curiosity": 0.12,
    "business_relevance": 0.10,
    "learning_value": 0.10,
    "edge_interest": 0.09,
}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def calculate_story_score(scores: StoryScores, weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    positive = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        positive += _clamp(getattr(scores, key)) * weight
        total_weight += weight
    base = positive / total_weight if total_weight else 0.0
    penalty = _clamp(scores.saturation_penalty) * 0.15
    composite = _clamp(base - penalty)
    scores.composite = round(composite, 2)
    return scores.composite


def rank_candidates(candidates):
    for candidate in candidates:
        calculate_story_score(candidate.scores)
    return sorted(candidates, key=lambda c: c.scores.composite, reverse=True)
