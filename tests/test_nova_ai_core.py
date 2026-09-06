import unittest

from nova_ai.certification import certify_story
from nova_ai.finops import BudgetPolicy, CostLedger
from nova_ai.models import (
    AgentResult,
    AgentRole,
    CandidateStatus,
    Claim,
    CostEvent,
    EvidenceMaturity,
    SourceRef,
    StoryCandidate,
    StoryScores,
)
from nova_ai.orchestrator import NovaOrchestrator
from nova_ai.registry import AGENT_SPECS
from nova_ai.scoring import calculate_story_score


class NovaCoreTests(unittest.TestCase):
    def test_registry_has_core_agents(self):
        self.assertIn(AgentRole.CERTIFIER, AGENT_SPECS)
        self.assertIn(AgentRole.EDGE_ADVERSARY, AGENT_SPECS)
        self.assertIn(AgentRole.FINAL_QA, AGENT_SPECS)

    def test_story_score_penalizes_saturation(self):
        base = StoryScores(relevance=90, novelty=90, timeliness=90, evidence_quality=90,
                           visual_potential=90, audience_curiosity=90, business_relevance=90,
                           learning_value=90, edge_interest=90, saturation_penalty=0)
        crowded = StoryScores(relevance=90, novelty=90, timeliness=90, evidence_quality=90,
                              visual_potential=90, audience_curiosity=90, business_relevance=90,
                              learning_value=90, edge_interest=90, saturation_penalty=100)
        self.assertGreater(calculate_story_score(base), calculate_story_score(crowded))

    def test_certification_fails_without_sources_and_claims(self):
        story = StoryCandidate(story_id="S1", title="x", summary="y")
        decision = certify_story(story)
        self.assertEqual(decision.decision, CandidateStatus.NEEDS_MORE_RESEARCH)
        self.assertTrue(decision.blocking_issues)

    def test_edge_claim_requires_competing_explanation(self):
        source = SourceRef(url="https://example.test", title="source", source_tier=1, is_primary=True)
        claim = Claim(claim_id="C1", text="test", sources=[source], evidence_maturity=EvidenceMaturity.E2_PLAUSIBLE,
                      confidence=0.6, edge_claim=True)
        story = StoryCandidate(story_id="S2", title="x", summary="y", sources=[source], claims=[claim])
        decision = certify_story(story)
        self.assertEqual(decision.decision, CandidateStatus.NEEDS_MORE_RESEARCH)
        self.assertTrue(any("competing explanations" in x for x in decision.blocking_issues))

    def test_qualified_edge_claim_can_certify_with_qualifications(self):
        source = SourceRef(url="https://example.test", title="source", source_tier=1, is_primary=True)
        claim = Claim(claim_id="C1", text="test", sources=[source], evidence_maturity=EvidenceMaturity.E2_PLAUSIBLE,
                      confidence=0.6, edge_claim=True, competing_explanations=["alternative"])
        story = StoryCandidate(story_id="S3", title="x", summary="y", sources=[source], claims=[claim])
        decision = certify_story(story)
        self.assertEqual(decision.decision, CandidateStatus.CERTIFIED_WITH_QUALIFICATIONS)

    def test_hard_budget_fails_closed(self):
        ledger = CostLedger(policy=BudgetPolicy(hard_story_usd=10))
        ledger.record(CostEvent(story_id="S", subsystem="research", provider="x", model="m", unit="call", quantity=1, cost_usd=7))
        with self.assertRaises(RuntimeError):
            ledger.record(CostEvent(story_id="S", subsystem="video", provider="x", model="m", unit="sec", quantity=1, cost_usd=4))

    def test_orchestrator_fails_closed_on_invalid_output(self):
        story = StoryCandidate(story_id="S4", title="x", summary="y")
        def executor(spec, candidate, context):
            return AgentResult(agent=spec.role, story_id=candidate.story_id, success=True, output={})
        run = NovaOrchestrator(executor).research_and_certify(story)
        self.assertFalse(run.success)
        self.assertIsNotNone(run.stopped_at)
        self.assertEqual(story.status, CandidateStatus.HOLD)


if __name__ == "__main__":
    unittest.main()
