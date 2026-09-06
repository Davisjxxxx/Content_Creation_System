# Creator Persona Contract

## Purpose
This file defines how a creator-specific model may be used without confusing stylistic simulation with factual authority.

## Persona layers
1. Voice identity: literal synthetic voice configuration, stored outside source control as provider identifiers/secrets.
2. Language fingerprint: cadence, vocabulary, sentence patterns, humor, analogies, transitions.
3. Presenter model: preferred explanation structure, depth, pacing, curiosity, skepticism, and rhetorical patterns.
4. Editorial preference model: explicit topic selections, rejections, revisions, and ratings.

## Data rules
- Do not infer or store sensitive personal traits.
- Do not train on private material unless explicitly designated for that purpose.
- Store raw audio, voice-clone artifacts, provider tokens, and personal training corpora outside Git.
- Keep versioned, inspectable preference summaries in configuration.
- Human edits are higher-quality labels than passive engagement assumptions.

## Required controls
The persona engine may change style but must not:
- alter evidence maturity
- invent personal experiences
- claim the creator tested something unless an explicit Builder's Lab record proves it
- fabricate opinions or endorsements
- generate synthetic quotations attributed to the creator as prior statements

## Initial state
Until a validated creator profile is supplied, use a neutral explanatory style and mark Jason-fit scores as provisional.
