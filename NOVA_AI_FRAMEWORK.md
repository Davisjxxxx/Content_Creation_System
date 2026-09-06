# Nova AI Intelligence Engine

Status: framework branch, AI proof lane only.

## Purpose
Build the upstream intelligence and editorial brain for Nova Content System without altering the existing production-engine behavior. AI-01 is the only active content lane for the proof cycle. Other lanes remain architectural future work.

## v1 boundaries
In scope:
- AI discovery and creator-watch signals
- core and edge research
- source provenance
- atomic claim ledger
- evidence maturity scoring
- adversarial review
- story scoring
- creator-fit interface
- fail-closed certification
- narrative/script/visual agent contracts
- final QA contract
- provider abstraction
- FinOps ledger and hard story budget
- explicit human selection and approval gates
- future connectors for YouTube, YouTube Shorts, TikTok, and Facebook Reels

Out of scope for this branch:
- LinkedIn generator integration
- blog site or blog publishing
- voice clone artifacts/credentials
- live publishing credentials
- other editorial lanes
- automatic public publishing

## Control principle
Agents advise. Deterministic orchestration controls state, budgets, dependencies, and stop gates. Retrieved web content is untrusted data and cannot modify agent/system instructions.

## Research pipeline
SOURCE_SCOUT + CREATOR_WATCH -> TREND_ANALYST -> EDGE_SCOUT -> PRIMARY_RESEARCHER -> CLAIM_EXTRACTOR -> EVIDENCE_ANALYST -> EDGE_ADVERSARY -> RED_TEAM -> STORY_ANALYST -> JASON_FIT -> CERTIFIER

## Production pipeline after human selection
NARRATIVE_ARCHITECT -> SCRIPT_WRITER -> VISUAL_DIRECTOR -> existing Nova production engine -> FINAL_QA -> HUMAN APPROVAL -> platform connectors

## Evidence maturity
E5 established; E4 strong; E3 contested; E2 plausible; E1 speculative; E0 unsupported/contradicted.

## Why the agent prompts are files
Prompts are versioned contracts. They can be reviewed, diffed, benchmarked, adversarially tested, and swapped independently of the runtime provider.

## Provider strategy
The engine must not hardwire a specific LLM, search, TTS, avatar, image, video, or publishing vendor. Provider adapters implement narrow interfaces. Provider selection is a runtime/config concern and should be benchmarked against quality, cost, latency, and reliability.

## Human gates
1. Story selection
2. Budget override above soft threshold
3. Final approval
4. Publish action

## Next integration gate
Before merging this framework into the local Nova MVP, reconcile the remote branch with the full local repository. The GitHub remote did not contain the current local MVP when this framework branch was created.
