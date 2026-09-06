# PR Review Guide

This branch should remain isolated until the full local Nova MVP has been reconciled into GitHub.

## Reviewer focus
1. Does the AI-only scope match the intended proof lane?
2. Are agent responsibilities sufficiently separated?
3. Are evidence and edge-claim rules fail-closed enough?
4. Are source, claim, cost, state, and approval boundaries explicit?
5. Are any prompts capable of silently widening scope or asserting unsupported claims?
6. Are the adversarial findings complete enough to drive security implementation?
7. Does anything in this branch conflict with the active local production engine once reconciled?

## Do not approve for production based on this branch alone
The current branch is a framework and control-plane foundation. Live search/model/media/publishing providers, desktop integration, and the current full production engine are not all present in this remote branch yet.
