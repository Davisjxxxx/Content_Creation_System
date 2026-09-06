from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agent_contract import AgentSpec
from .models import AgentResult, AgentRole
from .registry import get_agent_spec


DiscoveryExecutor = Callable[[AgentSpec, dict, dict], AgentResult]


@dataclass(slots=True)
class DiscoveryRun:
    results: list[AgentResult] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def success(self) -> bool:
        return not self.stop_reason and all(r.success for r in self.results)


class DiscoveryEngine:
    """Runs pre-story discovery. It never certifies truth; it produces candidate material for research."""

    def __init__(self, executor: DiscoveryExecutor):
        self.executor = executor

    def run(self, *, source_registry: dict, creator_registry: dict, lookback_window: str, context: dict | None = None) -> DiscoveryRun:
        ctx = dict(context or {})
        run = DiscoveryRun()
        jobs = (
            (AgentRole.SOURCE_SCOUT, {"source_registry": source_registry, "lookback_window": lookback_window}),
            (AgentRole.CREATOR_WATCH, {"creator_registry": creator_registry, "lookback_window": lookback_window}),
        )
        for role, inputs in jobs:
            spec = get_agent_spec(role)
            result = self.executor(spec, inputs, ctx)
            run.results.append(result)
            validation_errors = spec.validate_output(result.output) if result.success else []
            if validation_errors:
                result.success = False
                result.errors.extend(validation_errors)
            if not result.success and spec.fail_closed:
                run.stop_reason = "; ".join(result.errors) or f"{role.value} failed closed"
                return run
            if role == AgentRole.SOURCE_SCOUT:
                run.candidates.extend(result.output.get("candidates", []))
            elif role == AgentRole.CREATOR_WATCH:
                run.signals.extend(result.output.get("signals", []))
            ctx[role.value] = result.output
        return run
