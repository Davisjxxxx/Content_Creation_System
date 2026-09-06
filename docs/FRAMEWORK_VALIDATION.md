# Framework Validation

Local validation performed before GitHub push on 2026-09-06:

- `python -m unittest tests/test_nova_ai_core.py -v`
- 7 tests passed
- `python -m compileall -q nova_ai scripts`
- prompt registry completeness check: 16 registered agents, 0 missing prompt files
- mock framework demo: story score computed and deterministic certification returned `CERTIFIED`

## Important limitation
These results validate only the self-contained Nova AI framework assembled for this branch. They do not validate the full active local Nova production engine because that code was not present in the GitHub remote at framework-build time.

## Required combined validation after repository reconciliation
- run all Nova AI framework tests
- run existing production-engine E2E test `pytest -q tests/test_pipeline_e2e.py`
- verify installer behavior remains unchanged
- verify dashboard and production-state contracts remain intact
- run security regression suite before live credentials are introduced
