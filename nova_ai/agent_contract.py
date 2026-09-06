from __future__ import annotations

from dataclasses import dataclass

from .models import AgentRole


@dataclass(frozen=True, slots=True)
class AgentSpec:
    role: AgentRole
    purpose: str
    prompt_path: str
    required_inputs: tuple[str, ...]
    required_context: tuple[str, ...] = (
        "editorial_constitution.md",
        "evidence_policy.md",
        "ai_lane_scope.md",
    )
    required_output_keys: tuple[str, ...] = ()
    fail_closed: bool = True
    max_attempts: int = 2
    soft_budget_usd: float = 1.0
    hard_budget_usd: float = 3.0
    notes: str = ""

    def validate_output(self, output: dict) -> list[str]:
        missing = [key for key in self.required_output_keys if key not in output]
        return [f"missing required output key: {key}" for key in missing]
