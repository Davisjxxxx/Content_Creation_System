from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .agent_contract import AgentSpec
from .models import AgentResult, AgentRole, CandidateStatus, StoryCandidate
from .registry import get_agent_spec


DISCOVERY_SEQUENCE = (
    AgentRole.TREND_ANALYST,
    AgentRole.EDGE_SCOUT,
    AgentRole.PRIMARY_RESEARCHER,
    AgentRole.CLAIM_EXTRACTOR,
    AgentRole.EVIDENCE_ANALYST,
    AgentRole.EDGE_ADVERSARY,
    AgentRole.RED_TEAM,
    AgentRole.STORY_ANALYST,
    AgentRole.JASON_FIT,
    AgentRole.CERTIFIER,
)

PRODUCTION_SEQUENCE = (
    AgentRole.NARRATIVE_ARCHITECT,
    AgentRole.SCRIPT_WRITER,
    AgentRole.VISUAL_DIRECTOR,
    AgentRole.FINAL_QA,
)

AgentExecutor = Callable[[AgentSpec, StoryCandidate, dict], AgentResult]


@dataclass(slots=True)
class PipelineRun:
    story_id: str
    results: list[AgentResult] = field(default_factory=list)
    stopped_at: AgentRole | None = None
    stop_reason: str = ""

    @property
    def success(self) -> bool:
        return not self.stop_reason and all(r.success for r in self.results)


class NovaOrchestrator:
    """Deterministic controller. Agents provide judgments; orchestration owns state and stop gates."""

    def __init__(self, executor: AgentExecutor):
        self.executor = executor

    def run_sequence(self, candidate: StoryCandidate, sequence: tuple[AgentRole, ...], context: dict | None = None) -> PipelineRun:
        ctx = dict(context or {})
        run = PipelineRun(story_id=candidate.story_id)

        for role in sequence:
            spec = get_agent_spec(role)
            result = self.executor(spec, candidate, ctx)
            run.results.append(result)

            validation_errors = spec.validate_output(result.output) if result.success else []
            if validation_errors:
                result.success = False
                result.errors.extend(validation_errors)

            if not result.success and spec.fail_closed:
                run.stopped_at = role
                run.stop_reason = "; ".join(result.errors) or f"{role.value} failed closed"
                candidate.status = CandidateStatus.HOLD
                return run

            ctx[role.value] = result.output

        return run

    def research_and_certify(self, candidate: StoryCandidate, context: dict | None = None) -> PipelineRun:
        candidate.status = CandidateStatus.RESEARCHING
        run = self.run_sequence(candidate, DISCOVERY_SEQUENCE, context)
        if run.success:
            candidate.status = CandidateStatus.READY_FOR_CERTIFICATION
        return run

    def produce_selected(self, candidate: StoryCandidate, context: dict | None = None) -> PipelineRun:
        if candidate.status not in {
            CandidateStatus.SELECTED,
            CandidateStatus.CERTIFIED,
            CandidateStatus.CERTIFIED_WITH_QUALIFICATIONS,
        }:
            raise RuntimeError("production requires a selected or certified story")
        candidate.status = CandidateStatus.IN_PRODUCTION
        run = self.run_sequence(candidate, PRODUCTION_SEQUENCE, context)
        if run.success:
            candidate.status = CandidateStatus.READY_FOR_APPROVAL
        return run
