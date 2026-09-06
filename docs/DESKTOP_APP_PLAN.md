# Desktop App Integration Plan

## Goal
Package the existing Nova production engine and the AI Intelligence Engine into one desktop-operated system after framework certification.

## Recommended logical surfaces
1. Daily Opportunity Board
2. Story Dossier
3. Claim Ledger / Evidence Inspector
4. Select / Reject / Save controls
5. Production Workspace
6. Asset Provenance Inspector
7. Final Approval screen
8. Cost/FinOps panel
9. Provider health/configuration
10. Audit/history panel

## Required background services
- scheduler/discovery worker
- research/orchestration worker
- media/render worker
- local API/control plane
- provider adapter layer
- persistence layer
- analytics importer

## State model
DISCOVERED -> RESEARCHING -> RED_TEAM -> READY_FOR_CERTIFICATION -> CERTIFIED[_WITH_QUALIFICATIONS] -> SELECTED -> IN_PRODUCTION -> READY_FOR_APPROVAL -> APPROVED -> PUBLISHED

HOLD, NEEDS_MORE_RESEARCH, and REJECTED are explicit terminal/branch states.

## Desktop security
The UI must not itself hold provider secrets in page state. Privileged publishing and budget override actions should call a local control plane that validates session, state, and approval scope.
