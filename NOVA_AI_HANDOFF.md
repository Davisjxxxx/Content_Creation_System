# Nova AI Intelligence Handoff

## What is implemented in this framework branch
- AI-only lane definition and taxonomy
- evidence maturity model
- story/claim/source dataclasses
- agent contracts and registry
- 16 versioned agent prompts
- deterministic story scoring
- deterministic fail-closed certification helper
- provider interfaces
- FinOps hard budget ledger
- deterministic orchestration skeleton
- security/editorial/persona/platform constitutions
- adversarial review
- QA/certification checklist
- desktop integration plan
- cost/ROI control model
- unit tests and mock demo

## What is intentionally not implemented
- live LLM/search providers
- live discovery crawlers
- live database/provenance graph
- voice clone or avatar
- actual video generation
- computer-vision QA
- YouTube/TikTok/Facebook OAuth/upload
- blog
- LinkedIn integration
- desktop packaging

## Why
Those capabilities require either live credentials, reconciliation with the current local MVP, platform app registration, or security review. The remote repository did not contain the current local MVP at framework-build time, so modifying/duplicating that production engine from GitHub would have been unsafe.

## Required next gate
1. Push/reconcile the full local Nova MVP into GitHub.
2. Rebase or transplant `nova_ai/` and these docs onto the true active codebase.
3. Run framework tests plus existing `tests/test_pipeline_e2e.py`.
4. Resolve adversarial critical findings.
5. Perform security review.
6. QA/certify.
7. Only then enable live provider and publishing work.
