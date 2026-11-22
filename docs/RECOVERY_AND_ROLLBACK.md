# Recovery and preflight hardening — 2026-11-22

Follow-up to draft PR #5 (`ab936ff`). Restores desktop development, protects the legacy data, and
hardens the Runway integration ahead of its first paid test. No credits were bought, no API key was
set, no paid job was submitted, nothing was deleted, and no model was downloaded.

## 1. Legacy data

| Check | Result |
|---|---|
| Source → destination | `/home/jd/avatar_project/avatar_project_legacy` → `/home/jd/avatar_project_legacy`. The destination was absent; both are on the same filesystem (device 66307). |
| Move | Atomic rename (`mv -nT`, no clobber); no data copied. The inode is unchanged (59005672), all 164 entries are present, and no process was using the tree. |
| Compatibility symlinks | **35 of 35 resolve** (all 35 were broken before). |
| Git ignore | `.gitignore` now excludes `/avatar_project_legacy/`, `.venv*/`, and specific credential patterns (`.env.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.token`, `*_api_key.py`, `*api_key*.txt|json`, `credentials*.json`, `client_secret*.json`, `service-account*.json`, `.claude/settings.local.json`). Verified with `git check-ignore`; `avatar_v2/secret_store.py` and `.env.example` remain trackable. |
| Tracked / staged credentials | None. The only credential-like filename tracked is `avatar_v2/secret_store.py` (the key-storage code; no secret). |
| Git history | 98 commits across 12 local and remote refs scanned for key patterns (Google, OpenAI/Anthropic, GitHub, Hugging Face, AWS, Slack, private keys, hardcoded key assignments): **no matches** outside test fixtures. |
| `gemini_api_key.py` (legacy tree, untracked) | Contains one hardcoded string shaped like a Google API key (52-byte file, dated 2025-06-04); a compiled `.pyc` copy sits beside it. **No evidence of exposure**: it was never committed on any ref and never staged. The file was not modified, printed or uploaded. Recommendation: move the key to an environment variable or keyring. If the key is still active and the legacy tree was ever synced, shared or backed up elsewhere, rotate it in Google Cloud. |

## 2. Python 3.14 environment and packaging

| Step | Result |
|---|---|
| Rollback environment | `.venv` (Python 3.12) kept unchanged. Its tests pass (274), but it cannot start the GUI: system PyGObject is 3.14-only. |
| New environment | `.venv314`: Python 3.14.4, `--system-site-packages` (GTK 3 and WebKit2 4.1 via gi 3.56.2), `pip install -e '.[desktop,dev]'` (Pillow 11.3.0, pydantic 2.14.4, pywebview 5.4, pytest 8.4.2, PyInstaller 6.22.3). |
| Tests | **274 passed** on Python 3.14 and on the 3.12 rollback environment. |
| Source launch | The real GTK desktop window starts with `AVATAR_V2_UITEST=1` in an isolated profile. Results: no JS errors, all 7 views, all Runway and Presenter checks, and 23 of 23 non-ComfyUI API probes OK (the 4 ComfyUI probes fail with connection refused, as expected). |
| Launcher | `scripts/launch_avatar_v2.sh` now prefers `.venv314`, honors `AVATAR_V2_PYTHON` for rollback, and `AVATAR_V2_FROM_SOURCE=1` skips the package. The desktop shortcut is unchanged (it calls this script). Verified from source through the launcher. |
| Package rebuild | The previous build was preserved as `dist/AvatarV2-20260903-py312`. The new `dist/AvatarV2` (346 MB, bundles `libpython3.14`) contains all 12 new modules, the hardened Runway adapter, and a UI byte-identical to the source. Launched through the launcher, its self-test results match the source launch. |
| Previous package | **Preserved, but it no longer launches on this OS**, independently of this change. Its bundled WebKit typelibs reference `webkit_get_major_version`, which the upgraded system `libjavascriptcoregtk-4.1` no longer exports. |

### Rollback paths

| To roll back | Do this |
|---|---|
| The GUI to the previous code | `git worktree add ../avatar_v2_rollback ab936ff` (or `0c54e46`), then `cd ../avatar_v2_rollback && /home/jd/avatar_project/.venv314/bin/python -m avatar_v2.desktop`. Verified working for `ab936ff`. |
| The interpreter | `AVATAR_V2_PYTHON=/home/jd/avatar_project/.venv/bin/python scripts/launch_avatar_v2.sh` (CLI and tests only on 3.12). |
| The launcher | Revert `scripts/launch_avatar_v2.sh`, or force source with `AVATAR_V2_FROM_SOURCE=1`. |
| The package | `mv dist/AvatarV2 dist/AvatarV2-py314 && mv dist/AvatarV2-20260903-py312 dist/AvatarV2`. It only launches if the system WebKitGTK is downgraded, so prefer the source rollback above. |
| The Studio build (2026-11-22) | New builds are staged as `dist/AvatarV2.next` and switched in by the launcher on the next launch while the app is closed. The build it replaced is renamed `dist/AvatarV2-prev-<timestamp>`. To go back: `mv dist/AvatarV2 dist/AvatarV2-studio && mv dist/AvatarV2-prev-<timestamp> dist/AvatarV2`. |
| The Studio behaviour | `git revert` the Studio commit, or check out `2456bde` (the last commit with confirmations, upload consent and default caps). Old cap settings stay in `desktop.json` and are honoured again by that code. |
| The legacy location | `mv -nT /home/jd/avatar_project_legacy /home/jd/avatar_project/avatar_project_legacy`. Not recommended, since it breaks the 35 symlinks again. |

## 3. Runway submission and billing hardening

**Submission states.** Every cloud job durably records one of these states (atomic write plus fsync)
before moving on:

| State | Meaning | Billing |
|---|---|---|
| `preparing` | Media upload phase | Definitely none |
| `sending` | Recorded **before** the create request leaves | Unknown if interrupted |
| `accepted` | Task id durably saved | Tracked |
| `rejected` / `not_sent` | Definitive 4xx, or connect/pool failure | Definitely none |
| `uncertain` | Response lost (read timeout, dropped connection), 5xx, or success without a usable id | **Unknown** |

- **Never labelled unbilled when ambiguous.** Uncertain submissions (including an app crash after
  `sending`) are recorded as `UNCERTAIN` in the ledger. Their estimate counts against budgets until
  you resolve them. A crash before `sending` is the only case reported as "no generation was billed".
- **Durable task ids.** When the intent cannot be recorded, nothing is sent. When Runway's task id
  cannot be persisted, the task is **cancelled** and its charge kept unknown. If cancellation also
  fails, the task is finished in memory and the persistence failure is flagged on the receipt.
- **Conservative charges.** A charge is `actual` only when Runway reports `cost.credits`. Failed,
  cancelled, budget-cancelled, timed-out and cost-less tasks keep their estimate as **unconfirmed**
  spend.
- **No automatic resubmission.** Only HTTP 429 is retried, because it proves no task exists.
  Restarted uncertain jobs are retired to history, not re-queued. Lessons mark such segments
  `unresolved`. Any new paid job requires an explicit acknowledgement while unknown charges exist.
- **Resolution.** In Presenter → Providers & budget, each unknown charge can be marked
  **Not billed** or **Billed (credits)** after you check Runway usage, or reconciled from a task id
  in your Runway dashboard. Reconciling polls and downloads that task and never creates a new one.
- **Restore bug fixed (pre-existing).** Interrupted items used to stay in `queue/items/` and reappear
  on every launch. They are now recorded once and retired to history. The 10 stale Sep 3 items in
  `~/.avatar_v2/queue/items` will be retired as "interrupted" on the next real launch; they will not
  be executed.
- **JavaScript bridge.** pywebview now receives `JsBridge`, an explicit allowlist of the 87
  operations the UI calls. `DesktopAPI` internals (`queue`, `library` and their `Path` objects) are no
  longer reachable from page JavaScript. Four unused methods are not exposed: `provider_profile_delete`,
  `runway_avatar_delete`, `runway_avatar_status`, `runway_voices_list`. Tests keep the allowlist equal
  to the UI's calls and guard the pywebview exposure rules.

**Regression tests (`tests/test_runway_hardening.py`, 28):**
- lost responses (read timeout, dropped connection, 500/502, success without an id, non-JSON);
- definitive failures (connect errors, 401, 422, 429 retries);
- persistence failures (intent, task id, and cancel failure);
- actual versus unknown charges, and budget-cancel reservation;
- worker integration with the durable pre-send record;
- restart retirement and atomic writes;
- ledger resolution, budget exhaustion with acknowledgement, reconciliation without resubmission,
  and lesson safety;
- bridge exposure and UI sync.

## 4. Local rendering

Deferred as instructed. ComfyUI was not reinstalled, no H3/Wan/Qwen weights were downloaded, and
local workflow definitions and GPU cleanup controls are unchanged. See
[`STORAGE_RECOVERY_PLAN.md`](STORAGE_RECOVERY_PLAN.md).

## 5. Readiness for the first paid test

The adapter is ready for **one minimal, owner-approved master-image test**: Gen-4 Image at 1080×1440,
one candidate, about 8 credits. The owner sets the API key, enables Runway with upload consent,
confirms the per-job estimate, and resolves any unknown charge before submitting more work.

Remaining limitations:
- no live Runway call has been made, so request schemas, cost reporting and output URLs are verified
  only against the documented API with mocks;
- Runway exposes no idempotency key or task-list endpoint, so an uncertain submission can only be
  settled from Runway's usage or dashboard;
- local lanes need about 110 GB free plus licensed weights.
