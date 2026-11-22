# Avatar V2 capability audit — 2026-11-21

Phase 1 of the photorealistic presenter upgrade. This records what is implemented,
operational, experimental, unavailable, or misconfigured **on this machine today**,
from code inspection plus live checks. It is a snapshot; re-run the checks in
"Method" before relying on it later.

Branch audited: `avatar-v2-h3-desktop` @ `0c54e46` plus its uncommitted working tree
(carried unchanged onto `avatar-v2-presenter-upgrade`).

## Status legend

| Status | Meaning |
|---|---|
| OPERATIONAL | Verified working on this machine during this audit |
| IMPLEMENTED | Code and tests exist; cannot be exercised today because a dependency is absent |
| EXPERIMENTAL | Implemented but documented as not guaranteed (for example, may OOM on 8 GB) |
| UNAVAILABLE | The runtime, model weights or service it needs are not present on this machine |
| MISCONFIGURED | Configuration points at something wrong or missing |
| DECLARED ONLY | Named in types/docs, but no implementation exists |

## 1. Machine findings (highest impact first)

> **Recovery pass, 2026-11-22** (details in [`RECOVERY_AND_ROLLBACK.md`](RECOVERY_AND_ROLLBACK.md)):
>
> - Finding 3 is resolved: the legacy tree is back at `/home/jd/avatar_project_legacy`, and all 35
>   symlinks resolve.
> - Finding 2 is resolved for development: the new `.venv314` launches the GUI from source, and the
>   3.12 `.venv` is kept as a rollback.
> - Finding 7 is resolved: the package was rebuilt on Python 3.14. The previous package is preserved,
>   but it no longer launches on the upgraded OS.
> - Finding 1 is still deferred by decision; see [`STORAGE_RECOVERY_PLAN.md`](STORAGE_RECOVERY_PLAN.md).

1. **The local ComfyUI renderer stack is gone — UNAVAILABLE / MISCONFIGURED.**
   `/home/jd/AvatarForge` (ComfyUI 0.33, its venv, and every H3 / Wan / Qwen
   text-encoder / VAE / LoRA / ESRGAN weight) no longer exists. A full-filesystem search
   found no relocated copy, and it is not in the Trash. The last ComfyUI activity in
   `~/.avatar_v2/logs/comfyui.log` is an H3 render completing on 2026-09-03.
   `~/.avatar_v2/desktop.json` still points `comfyui_dir`, `comfyui_python` and
   `comfyui_output` at the deleted paths. **Every local render lane is down**:
   H3 Ref2VA/FL2VA, Wan 2.2, AnimateDiff, SDXL txt2img and the ESRGAN upscale lane.
   - Restoring needs a ComfyUI reinstall plus model re-acquisition. H3 weights are
     license-gated, and the deployment rules require explicit user authorization before
     any re-download. Nothing was downloaded during this audit.
   - Disk: 65 GB free of 915 GB (93% used). The previous H3 INT8 set alone was about
     60 GB (two 21 GB diffusion checkpoints, a ~15 GB text encoder, VAEs and LoRAs).
2. **The project venv was broken by an OS Python upgrade — repaired.** `/usr/bin/python3`
   is now 3.14. `.venv/bin/python3` was a symlink to `/usr/bin/python3`, so the venv ran
   3.14 and could not see its 3.12 packages (httpx, pydantic, pywebview, pytest).
   `pyvenv.cfg` still declares 3.12.3, so the symlink was re-pointed to
   `/usr/bin/python3.12`. Revert with `ln -sfn /usr/bin/python3 .venv/bin/python3`.
   **Remaining limitation:** the system PyGObject now ships only a 3.14 extension
   (`gi/_gi.cpython-314-x86_64-linux-gnu.so`). So the 3.12 venv runs the tests and CLI but
   cannot start the GTK webview: launching via the venv fails with "You must have either QT or
   GTK". The packaged `dist/AvatarV2` build bundles its own interpreter and is unaffected, but it
   predates the Phase 2 code. The durable fix is to rebuild `.venv` on Python 3.14 with
   `--system-site-packages` and reinstall `.[desktop,dev]`, then regenerate the PyInstaller
   build. Not done without user approval. A disposable 3.14 venv was used to run the UI
   self-test (see section 6).
3. **The legacy A1111 tree was moved into the repository working tree.**
   `/home/jd/avatar_project_legacy` (61 GB) now lives at `./avatar_project_legacy/`
   (untracked). Result: all 35 root-level compatibility symlinks
   (`webui`, `models`, `Wav2Lip`, `AnimateDiff`, …) are broken. `.git/info/exclude` only
   covers the old symlink names, not the new directory, so a broad `git add -A` would try
   to stage 61 GB, including a file named `gemini_api_key.py`. **Recommended:** move it
   back to `/home/jd/avatar_project_legacy`, or add `/avatar_project_legacy/` to
   `.git/info/exclude`. Not changed by this audit.
4. **GPU** — RTX 4070 Laptop GPU, **8188 MiB**, driver 580.178.04. `nvidia-smi` memory
   detection works, which is the GPU manager's query path. The repo docs and CLAUDE.md
   talk about a 12 GB-class card; this machine is 8 GB.
5. **Local vision service — OPERATIONAL.** Ollama answers at `127.0.0.1:11434` with
   `qwen3-vl:8b-instruct` and `qwen3-vl:2b-instruct` installed. That covers reference
   classification and candidate QA (service-level check; no classification was run).
6. **ffmpeg / ffprobe 8.0.1 — OPERATIONAL.** Used for QA metadata and last-frame chaining.
7. **Desktop build.** `dist/AvatarV2/AvatarV2` (2026-09-03 17:51) matches the source as of
   that date. `scripts/launch_avatar_v2.sh` prefers this build over the venv, so the
   desktop shortcut will not show any new code until the build is regenerated.

## 2. Component audit

| Component | Status | Evidence | Notes |
|---|---|---|---|
| Avatar Identity Vault (`library.py`) | OPERATIONAL (data layer) | `~/.avatar_v2/library/vault.json`: 34 profiles (4 `synthetic`, 30 `consenting_adult`); 86 of 100 reference paths resolve | The 4 synthetic benchmark profiles are intact (`/home/jd/Videos/ai benchmark images`). 14 reference paths in other profiles point to missing files. |
| Profile Designer (`avatar_designer.py`, `queue_avatar_designer`) | IMPLEMENTED; generation UNAVAILABLE | Coverage slots, slot prompts, candidate accept/reject | Candidate generation needs local H3. |
| Reference cache and local vision classification (`reference_cache.py`) | OPERATIONAL service; large uncommitted change set | Ollama up; +431/−39 lines uncommitted | Preserved, not committed. |
| Candidate QA (`candidate_qa.py`) | IMPLEMENTED, advisory; untracked | Local VLM comparison; `tests/test_candidate_qa.py` (untracked) | No embedding-based face-similarity metric. |
| Prompt compiler (`composer.py`) | OPERATIONAL (pure code) | H3-native Ref2VA/FL2VA structure, timed beats, physical direction, age/sex anchors | No per-model length budget. Prompts often exceed 1,500 chars; fine for H3, too long for cloud models capped at 1,000 UTF-16 units. |
| Prompt presets (`prompt_presets.py`) | OPERATIONAL | Adult preset catalog only | No presenter or educational presets. |
| Provider interface | MISCONFIGURED for multi-provider use | `ProviderConfig.provider: Literal["comfyui"]`; `RenderQueue._execute` hard-codes `ComfyUIProvider` | No provider abstraction, so a cloud adapter could not be queued. |
| Render router (`router.py`) | OPERATIONAL, minimal | `auto` → `h3_ref2va` when omni refs exist, else `h3_fl2va` | Ignores availability. `EngineName` declares `ltx25` and `seedance` with no implementation (DECLARED ONLY). |
| ComfyUI integration (`providers/comfyui.py`) | IMPLEMENTED; UNAVAILABLE here | Health, object_info, HTTP upload, queue, wait, `/free`, interrupt; tests pass against fakes | Service not installed. |
| H3 Ref2VA / FL2VA | IMPLEMENTED; previously verified (commit `288cef6`, 2026-08-23); UNAVAILABLE now | Builders in `workflows_builder.py` | Weights gone; `local_h3_authorized` is still `true` in settings. Guided/hybrid Ref2VA routes are EXPERIMENTAL on 8 GB per README. |
| Wan 2.2 (FLF2V, Fun-Control) | IMPLEMENTED; previously READY; UNAVAILABLE now | Builders and canvas/frame snapping | |
| AnimateDiff SD 1.5 / SDXL | IMPLEMENTED; previously READY; UNAVAILABLE now | Bundled API workflows | |
| Upscale lane (4x-UltraSharp) | IMPLEMENTED; UNAVAILABLE now | `build_upscale_workflow` | |
| Runtime profiles (`runtime_profiles.py`) | IMPLEMENTED | 6 profiles, tests | Meaningful only with local H3 installed. |
| GPU memory manager (`gpu_memory.py`) | PARTIALLY VERIFIED | `nvidia-smi` query detects 8188 MiB | `/free` pre/post cleanup and the forced preflight for `int8_12gb` / `low_memory` / `low_memory_quality` cannot be verified without ComfyUI; unit tests cover the logic. Never kills processes automatically. |
| Render queue (`render_queue.py`) | OPERATIONAL (pure code) | Persistence, restart recovery, 870 history records, cancel, chaining | Single-provider before this upgrade. |
| Desktop settings / persistence | MISCONFIGURED | `desktop.json` points at deleted ComfyUI paths | |
| File / library management (Results) | OPERATIONAL | `history()` lists records and marks missing outputs | |
| QA (`qa.py`) | OPERATIONAL, minimal | ffprobe metadata only | No perceptual, identity or lip-sync QA metrics. |
| Export | GAP | Finished outputs are copied to `preferred_output_dir` | No export presets (codec, loudness, captions) for course delivery. |
| Lip sync | UNAVAILABLE locally | Only a legacy `Wav2Lip` tree under `avatar_project_legacy/`, not integrated into Avatar V2 | |
| Tests | OPERATIONAL | `pytest`: 122 passed on the baseline tree; `compileall` clean | |
| README renderer status table | STALE | Says H3 is "BLOCKED on weights"; H3 was later verified; all local lanes are now unavailable | |

## 3. Uncommitted work preserved

Present before this upgrade and carried unchanged onto the new branch (not committed,
not modified except where Phase 2 adds clearly separated code):

- Modified: `README.md`, `avatar_v2/desktop.py`, `avatar_v2/library.py`,
  `avatar_v2/reference_cache.py`, `avatar_v2/render_queue.py`, `avatar_v2/ui/index.html`,
  `avatar_v2/workflows_builder.py`, `tests/test_avatar_designer.py`,
  `tests/test_backend_ext.py`, `tests/test_reference_cache.py`,
  `tests/test_render_queue_lifecycle.py` (11 files, +1679/−184).
- Untracked: `AGENTS.md`, `avatar_v2/candidate_qa.py`, `tests/test_candidate_qa.py`,
  `docs/AVATAR_STUDIO_PRODUCT_DESIGN.md`, `lora_training/`, `.serena/`,
  `avatar_project_legacy/` (see machine finding 3).

## 4. Gap analysis against the presenter mission

| Requirement | Today | Gap / plan |
|---|---|---|
| Photographic identity realism | Local lanes unavailable; legacy SDXL checkpoints exist only in the legacy tree | Phase 2 adds Runway `gen4_image` / `gen4_image_turbo` / `gemini_image3.1_flash` with tagged identity references |
| Facial consistency | Advisory VLM candidate QA (local) | Phase 2 binds the same tagged identity anchors on every still; an embedding-based face-similarity metric is still a gap |
| Natural motion | H3 / Wan locally (unavailable) | Phase 2 adds `gen4.5`, `gen4_turbo`, hosted `h3_max`, and `act_two` performance transfer |
| Realistic lip sync | None integrated | Phase 2 adds Runway custom avatars plus `gwm1_avatars` script- or audio-driven presenter video |
| Repeatable educational production | Queue, history, seeds, receipts | Script segmentation, caption/export presets and loudness normalization remain later-phase work |

## 5. Phase 2 outcome (Runway integration)

Added on branch `avatar-v2-presenter-upgrade` (see `docs/RUNWAY_PROVIDER.md`):

- **Provider seam:** `ProviderConfig.provider` now accepts `comfyui | runway`. `RenderQueue`
  picks the provider per item. Cloud providers declare `uses_local_gpu = False`; for those
  jobs the queue logs a `skipped` cleanup event instead of calling ComfyUI `/free`. Local jobs
  keep the forced preflight for `int8_12gb`, `low_memory` and `low_memory_quality`.
  `AVATAR_V2_GPU_*` controls are untouched.
- **Runway adapter (`providers/runway.py`):** a registry of 9 models with verified schemas,
  documented-limit validation, local cost estimates, inline-vs-ephemeral-upload media handling
  with EXIF stripping, a credential-isolated HTTP client, polling with backoff, cancel and
  timeout, output download, failure-code advice, and redacted receipts.
- **Presenter prompt compiler (`presenter.py`):** UTF-16-budgeted prompts, `@tag` identity
  binding, priority trimming, and 7 consistent-variation presets.
- **Cloud policy (`policy.evaluate_cloud_policy`):** general content only; declared synthetic
  or consented provenance required.
- **Credentials (`secret_store.py`):** env var, then OS keyring (optional), then a user-only file.
  Credential-like settings keys are stripped from `desktop.json` and from `get_state`.
- **Desktop and UI:** Runway renderer card and task panel (still, variations, image-to-video,
  talking presenter, performance transfer, upscale), Estimate → confirm → Queue flow, settings
  card (enable, consent, key, caps, defaults, balance check), cost line on queue items, and
  Runway checks in the built-in UI self-test.

### Phases 3-10

- **Reconciliation and accounting:** task ids are persisted at creation, and restarts resume
  polling instead of failing or re-billing. An append-only cost ledger backs daily and monthly budgets.
  Every output (local or cloud) gets a SHA-256, with a verify action.
- **Midjourney (Phase 3):** no official public API exists, and automated access is not permitted, so
  only manual import is implemented (`reference_import.py`). Imports are validated, hashed, copied
  unchanged with provenance, and registered as pending candidates. Midjourney's own documentation
  host refused automated fetches (HTTP 403), so the user should confirm their plan's
  commercial-use terms.
- **BRAND profile (Phases 4-5):** `brand_instructor.py`. Spec, identity prompts, positive realism
  guards, provider exclusions, 9 reference slots, a voice profile, and recommended settings
  (`docs/BRAND_INSTRUCTOR.md`). Accepted references are hash-bound; cloud jobs refuse a different
  primary identity; generated and imported candidates are never auto-accepted.
- **Animation paths (Phase 6):** Path A, the talking head (`gwm1_avatars`), requires approved
  narration. Path B, performance transfer (`act_two`, the only documented Runway performance-capture
  endpoint), strips the driving footage's audio before upload and mutes or remuxes the output with
  the approved voiceover. Local alternatives (H3 Ref2VA with a motion reference, Wan 2.2
  Fun-Control) remain available once ComfyUI is reinstalled.
- **Voice (Phase 7):** `voiceover.py` keeps voice profiles separate from visual identity. Covers
  authorized narration import with an approval step, Runway TTS (`eleven_v4/v3/multilingual_v2`),
  voices designed from text only (no cloning), pronunciation respelling, segmenting, and timing
  estimates.
- **Realism QA (Phase 8):** `realism_qa.py`. Pillow and ffmpeg proxies for skin detail versus the
  master, lighting continuity, flicker, identity drift and geometry stability (dHash), blink rate,
  lip-sync correlation and lag, A/V alignment, and freeze/black frames. Plus an 11-criterion human
  review and a production gate that requires a 20-30 s segment.
- **Automation (Phase 9):** a Presenter view with provider availability, budget, approval queue,
  voice and narration, manual import, QA review, lessons with changed-segment-only re-rendering, and
  hash verification. The Runway panel adds slot, seeded-candidate, voiceover, narration and
  saved-provider-profile controls, plus a one-click BRAND preset.

### Additional findings during implementation

- **pywebview exposure (pre-existing):** pywebview's `get_functions` recurses into every public,
  non-callable attribute of the API object. So `queue`, `library` and their `Path` attributes
  (for example `queue.work_dir.unlink`) are callable from page JavaScript. The page loads no remote
  content, so the risk is low. New stores are built on demand and the new queue attribute is
  private, so the surface did not grow. Hardening recommendation: make `queue` and `library`
  private attributes.
- **Patched pywebview (environment):** the installed `pywebview` in `.venv` has an `[AV2DEBUG]`
  print added to `webview/util.py`. Someone edited site-packages; it is harmless but unusual.
- **UI bug fixed (pre-existing):** `applyRuntime()` wrote to a non-existent `#rCanvas` element and
  threw on H3 jobs after queueing, so the success log and queue refresh never ran. It is now
  null-guarded.

## 6. Verification

- `pytest`: 246 passed (122 baseline + 124 new), `compileall` clean.
- In-app UI self-test (`AVATAR_V2_UITEST=1`), run with an isolated `HOME` under a disposable
  Python 3.14 venv: no JS errors, all 7 views active (including Presenter), all Runway and
  Presenter checks passed, and every new API probe returned OK. ComfyUI probes failed with
  "connection refused", as expected with ComfyUI absent.
- Commit separation: the user's pre-existing uncommitted work was committed separately. The
  versions of the four mixed Python and README files were reconstructed by reversing this session's
  edits and verified byte-exact. The Python files compile to bytecode identical to the
  `dist/AvatarV2` build (made seconds after the user's last edit on 2026-09-03), and every file's
  diff stat equals the stat recorded at the start of this session. `index.html` was taken verbatim
  from the build. That tree passes the original 122 tests.
- Not verified: any live Runway call (no API key or credits configured), any local render
  (ComfyUI absent), and a 20-30 s production segment (requires paid generation or local lanes).

## 7. Method

Commands used (all read-only except the venv symlink repair):
`git status/branch -a/log/ls-remote` (and a fetch of two unfamiliar remote branches, both
ruled out as app branches), `nvidia-smi --query-gpu/--query-compute-apps`,
`curl 127.0.0.1:8188/system_stats` (no answer), filesystem searches for ComfyUI,
`minimax_h3*` / `wan2*` files and large `.safetensors`, `df -h`, Ollama `/api/tags`,
`ffmpeg -version`, a vault reference-path existence check,
`.venv/bin/python -m compileall -q avatar_v2 scripts`, and `.venv/bin/python -m pytest -q`.
