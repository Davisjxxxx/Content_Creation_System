# Nova AI Framework Adversarial Review

Date: 2026-09-06
Scope: AI-first intelligence framework, not the existing local production engine.

## Review posture
Assume the design will fail through hallucination, prompt injection, correlated agent errors, source laundering, cost runaway, over-automation, or false confidence.

## Critical findings

### AR-01 Correlated-model failure
Severity: HIGH
Multiple agents using the same model/provider can appear to independently agree while repeating the same model bias or hallucination.
Mitigation in framework: agent contracts separate roles; provider abstraction exists.
Required before certification: add provider-diversity policy for high-risk certification or require independent source-grounded deterministic checks. Never count model votes as independent evidence.

### AR-02 Prompt injection through research sources
Severity: CRITICAL
Web pages, papers, repository READMEs, transcripts, and social content can contain instructions intended to redirect agents.
Mitigation: Security Constitution states retrieved content is untrusted and cannot provide instructions.
Required implementation: tool layer must separate retrieved data from system/context channels, strip/label active instructions, restrict tool permissions per role, and log tool calls.

### AR-03 Citation fabrication
Severity: CRITICAL
An LLM can invent a plausible URL, DOI, benchmark, or quotation.
Mitigation: fail-closed evidence policy and source mapping.
Required implementation: source existence/retrieval validation must be deterministic before certification. Citations that cannot be fetched/verified block publication.

### AR-04 Source laundering
Severity: HIGH
Ten articles may all derive from one vendor release.
Mitigation: Primary Researcher and Red Team explicitly trace claims upstream.
Required implementation: provenance graph should track source ancestry and independent-corroboration groups.

### AR-05 Story-scoring bias toward hype
Severity: HIGH
Trend/curiosity scoring can systematically select exaggerated AI stories.
Mitigation: evidence quality has a high weight and certification is independent.
Required implementation: do not expose composite story score to certifier before evidence decision, preventing popularity anchoring.

### AR-06 Creator-fit self-reinforcement
Severity: MEDIUM/HIGH
A fit model can narrow topics into a filter bubble and learn quirks from too little data.
Mitigation: explicit labels only, uncertainty required.
Required implementation: exploration quota and calibration metrics; never let fit suppress high-value unfamiliar topics completely.

### AR-07 Edge-track sensationalism
Severity: HIGH
The Edge lane can become a mechanism for laundering weak speculation into authoritative content.
Mitigation: E0-E5 policy, competing explanations, Edge Adversary.
Required implementation: lower-maturity claim language must be programmatically carried into script constraints and checked for drift at Final QA.

### AR-08 Human approval theater
Severity: HIGH
A user can approve a polished asset without seeing what changed after certification.
Required implementation: approval screen must surface material claim changes, unresolved warnings, cost, synthetic-media disclosures, and provenance gaps, not just a video preview.

### AR-09 Cost explosion from generative video
Severity: HIGH
Rerolls and long synthetic sequences can dominate cost.
Mitigation: FinOps hard story cap.
Required implementation: reserve budget before generation; count failed/rerolled generations; require explicit override token above soft cap.

### AR-10 Publishing blast radius
Severity: CRITICAL
Compromised or mistaken publishing credentials can publish bad content across multiple platforms.
Mitigation: publishing separated from research and requires approval token.
Required implementation: platform-bound short-lived approval tokens, least-privilege OAuth, dry-run mode, idempotency keys, and per-platform kill switch.

### AR-11 Persona hallucination
Severity: HIGH
Style generation can invent personal experiences or opinions.
Mitigation: Persona Contract blocks unsupported first-person claims.
Required implementation: Builder's Lab first-person claims must reference a project-evidence record.

### AR-12 Copyright/provenance failure
Severity: HIGH
Automated asset retrieval can use copyrighted media without sufficient rights.
Mitigation: Visual Director prioritizes provenance and Final QA checks it.
Required implementation: asset manifest must contain license/use basis for every externally sourced asset; unknown rights are blocking.

## Architecture weaknesses intentionally not solved yet
- no durable database implementation
- no live source retrieval engine
- no live LLM provider implementation
- no provenance graph database
- no OAuth/publishing implementation
- no desktop UI integration
- no CV/media QA runtime
- no voice/avatar implementation
- no model benchmarking harness

These are not hidden. They are the next implementation gates.

## Adversarial verdict
CONDITIONAL PASS AS FRAMEWORK ONLY.
The architecture is coherent enough to proceed, but it must not be certified as production-ready or granted publishing credentials until AR-02, AR-03, AR-08, AR-09, AR-10, and AR-12 have executable controls and tests.
