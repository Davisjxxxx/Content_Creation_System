# Evidence Policy

## Evidence maturity
- E5_ESTABLISHED: multiple strong, independent lines of evidence or mature reproducible evidence.
- E4_STRONG: strong direct evidence with limited unresolved questions.
- E3_CONTESTED: substantive evidence exists but interpretation is genuinely disputed.
- E2_PLAUSIBLE: mechanistically plausible and supported by some observations, but insufficient evidence.
- E1_SPECULATIVE: coherent or interesting hypothesis with little direct support.
- E0_UNSUPPORTED: currently unsupported, materially contradicted, or stated beyond available evidence.

## Source tiers
Tier 1: original documentation, repositories, primary datasets, peer-reviewed research, official regulators, direct company/model release materials.
Tier 2: reputable specialist journalism, research institutions, technical analyses that expose their sources.
Tier 3: creators, social posts, newsletters, Reddit/community discussion. Useful for discovery and sentiment.
Tier 4: anonymous, copied, viral, unsourced, or content-farm material. Discovery only unless independently verified.

## Claim requirements
Every material factual claim must have:
- unique claim_id
- source mapping
- evidence maturity
- confidence 0-1
- replication status when relevant
- competing explanations for edge claims
- allowed wording
- forbidden wording when overstatement risk exists

## Freshness
AI changes quickly. Verify model names, availability, pricing, product behavior, benchmark claims, policies, leadership, and launch status at research time. Record the verification timestamp.

## Benchmark discipline
Distinguish:
- vendor-reported benchmark
- independent benchmark
- synthetic benchmark
- real-world task evaluation
- cherry-picked demo
Never imply one category proves another.

## Edge rule
Edge claims are publishable only when the content explicitly distinguishes evidence from hypothesis. The adversary agent must attempt to falsify the claim and identify mundane alternatives.

## Citation laundering rule
A secondary article citing another secondary article does not create independent corroboration. Trace consequential claims to the earliest available source.
