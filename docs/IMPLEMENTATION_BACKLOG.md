# Nova AI Implementation Backlog

The framework branch is intentionally provider-neutral and credential-free. After the full local Nova MVP is reconciled into GitHub, implement in this order.

## NIS-001 Repository reconciliation
- bring the active local production engine into GitHub
- preserve source-of-truth paths and existing E2E test
- rebase/transplant this framework
- run old + new tests together

## NIS-002 Durable persistence
- story, source, claim, certification, audit, cost, asset, and approval tables
- provenance/source ancestry graph representation
- migrations and rollback

## NIS-003 Secure source retrieval
- URL allowlist/scheme validation
- SSRF protection
- prompt-injection isolation
- content normalization
- source existence verification
- provenance logging

## NIS-004 Live discovery
- source registry
- RSS/API/web search adapters
- creator registry
- Source Scout + Creator Watch scheduling
- deduplication and freshness detection

## NIS-005 Model/provider runtime
- OpenAI/Anthropic/Google/local adapters as selected
- structured JSON validation
- retry policy
- provider diversity rules
- cost/latency telemetry

## NIS-006 Research and Claim Ledger
- primary-source dossier runtime
- atomic claim persistence
- evidence maturity engine
- independent-source grouping
- citation verification

## NIS-007 Adversarial certification
- Edge Adversary runtime
- Red Team runtime
- claim-language constraints
- certification expiry/reverification
- regression suite for hallucination/citation laundering

## NIS-008 Opportunity board
- story ranking
- creator-fit provisional model
- Select / Reject / Save / More Research controls
- capture explicit labels for future learning

## NIS-009 Persona and Builder's Lab evidence
- creator style profile ingestion
- approved voice/provider identity mapping outside Git
- project-evidence records for first-person claims
- prevent invented personal experience

## NIS-010 Narrative and script production
- narrative blueprint
- claim-to-script traceability
- long-form script
- independent short-form hooks
- script drift validator

## NIS-011 Asset/provenance engine
- evidence-first asset acquisition
- license/use-basis manifest
- graphics/diagram generation
- generated-media labeling
- provider-neutral image/video interfaces

## NIS-012 Computer vision and media QA
- scene detection
- blur/black-frame/duplicate checks
- semantic B-roll matching
- 16:9 to 9:16 smart crop
- caption-safe zones
- visual-evidence mismatch detection

## NIS-013 Voice/presenter
- approved voice clone integration
- presenter/avatar provider adapter if retained
- cost budget and fallbacks
- synthetic-media disclosure logic

## NIS-014 Production-engine integration
- connect storyboard/assets/voice to existing FFmpeg pipeline
- render manifest
- YouTube master + Shorts + TikTok + Facebook variants

## NIS-015 Approval control plane
- human final-review screen
- material claim-change summary
- provenance and warning display
- short-lived scoped approval tokens
- budget override tokens

## NIS-016 Distribution connectors
- YouTube API
- TikTok Content Posting API
- Facebook Reels publishing API
- dry run, idempotency, per-platform kill switch
- no autonomous publish without approval

## NIS-017 Analytics and learning
- normalized cross-platform metrics
- topic/hook/format performance
- selection-model training dataset
- exploration quota to avoid filter bubbles
- ROI attribution

## NIS-018 Desktop packaging
- integrate intelligence + production + approval into desktop shell
- local service lifecycle
- secure secrets/config UI
- health/status panel
- clean-machine Windows/Linux/macOS validation

## Certification stop gate
Do not enable live publishing until critical adversarial findings AR-02, AR-03, AR-08, AR-09, AR-10, and AR-12 have executable controls and security/QA signoff.
