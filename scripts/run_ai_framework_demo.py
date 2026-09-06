from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nova_ai.certification import certify_story
from nova_ai.models import Claim, EvidenceMaturity, SourceRef, StoryCandidate, StoryScores
from nova_ai.scoring import calculate_story_score


def main():
    source = SourceRef(url="https://example.test/primary", title="Example primary source", publisher="Example", source_tier=1, is_primary=True)
    claim = Claim(claim_id="CLM-DEMO-001", text="A demo claim exists only to prove framework data flow.", sources=[source], evidence_maturity=EvidenceMaturity.E4_STRONG, confidence=0.9)
    story = StoryCandidate(story_id="STORY-DEMO-001", title="Framework Demo", summary="No live model or publishing call is made.", sources=[source], claims=[claim], scores=StoryScores(relevance=90, novelty=70, timeliness=60, evidence_quality=90, visual_potential=60, audience_curiosity=75, business_relevance=80, learning_value=90, edge_interest=20, saturation_penalty=10))
    print("story_score=", calculate_story_score(story.scores))
    print("certification=", certify_story(story).decision.value)


if __name__ == "__main__":
    main()
