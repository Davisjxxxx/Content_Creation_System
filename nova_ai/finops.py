from __future__ import annotations

from dataclasses import dataclass, field

from .models import CostEvent


@dataclass(slots=True)
class BudgetPolicy:
    target_monthly_usd: float = 250.0
    hard_monthly_usd: float = 500.0
    soft_story_usd: float = 40.0
    hard_story_usd: float = 60.0
    research_soft_usd: float = 5.0
    certification_soft_usd: float = 3.0
    image_soft_usd: float = 5.0
    video_soft_usd: float = 25.0
    presenter_soft_usd: float = 8.0
    voice_soft_usd: float = 3.0


@dataclass(slots=True)
class CostLedger:
    policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    events: list[CostEvent] = field(default_factory=list)

    def record(self, event: CostEvent) -> None:
        if event.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        projected = self.story_total(event.story_id) + event.cost_usd
        if projected > self.policy.hard_story_usd:
            raise RuntimeError(
                f"hard story budget exceeded for {event.story_id}: ${projected:.2f} > ${self.policy.hard_story_usd:.2f}"
            )
        self.events.append(event)

    def story_total(self, story_id: str) -> float:
        return round(sum(e.cost_usd for e in self.events if e.story_id == story_id), 4)

    def subsystem_total(self, story_id: str, subsystem: str) -> float:
        return round(sum(e.cost_usd for e in self.events if e.story_id == story_id and e.subsystem == subsystem), 4)

    def requires_human_budget_approval(self, story_id: str) -> bool:
        return self.story_total(story_id) > self.policy.soft_story_usd
