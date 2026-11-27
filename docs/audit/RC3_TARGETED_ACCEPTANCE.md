# Avatar V2 RC3 — Targeted Acceptance & Adversarial Verification Report

- **Date:** 2026-11-27
- **Branch:** `rc3-astra-remediation`
- **Build Commit for `dist/AvatarV2.rc3`:** `4472e7522f13b684738327b6dd1c6694c563279c`
- **Baseline RC2 Commit (preserved unchanged):** `87ce485e7879f1ea085596dece66a95d9c60b9ff` (`rc1-astra-remediation`)
- **Original RC1 Commit (preserved unchanged):** `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` (`sprint-1-gpu-memory-foundation`)
- **Astra RC1 Audit Commit (preserved unchanged):** `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` (`audit/astra-rc1-independent`)
- **Astra RC2 Recheck Commit (preserved unchanged):** `c63fb9fb7d4598db2a122b16b4c3a03feb1a2316` (`audit/astra-rc2-recheck`)

---

## 1. Automated & Adversarial Regression Verification (Including Clean Isolated `HOME`)

| Verification Layer | Command / Scope | Result |
| --- | --- | --- |
| **Full Suite (Clean Isolated `HOME`)** | `HOME=/tmp/avatar_rc3_iso_home .venv/bin/pytest -q` | **469 / 469 passed (0 failed, 0 skipped)** — zero implicit `/home/jd/AvatarForge` dependency |
| **Gate 1 & 2 P0/P1 Adversarial Suite** (`RC-01`, `RC-02`, `RC-03`) | `tests/test_rc3_gate1_gate2_p0_ownership.py` | **Passed** — synthetic presenter-master inside `preferred_output_dir` preserved untouched (even with forged history); copied/moved/replaced/hard-linked/symlinked assets refused; exact 18189→28189 endpoint redirection sends 0 unauthorized POSTs to external spy server; restart `send_signal_if_owned()` uses `signalled_via_reopened_pidfd`; `JsBridge` rejects unselected external paths on `studio_presenter_add` and `studio_pick` |
| **Gate 3 Adversarial Suite** (`RC-10`, `RC-11`, `RC-16`, `RC-18`) | `tests/test_rc3_gate3_durability_budgets.py` | **Passed** — pre-ack crash snapshot restores as `recovery_required` with 0 replays; durability write failure refuses dispatch; budget write `OSError` at `max_items=1` fails closed and prevents double-charge across live + detached callers; non-terminal preview output (`status={}`) is never promoted by `ComfyUIProvider.wait()`; slow QA thread past 2s shutdown timeout retains `_dir_lock` blocking second writer; `desktop.json` containing `[]` quarantines without `TypeError` |
| **Gate 4 Adversarial Suite** (`RC-04`, `RC-06`, `RC-07`, `RC-09`, `RC-19`, `RC-22`) | `tests/test_rc3_gate4_refs_criteria_verdicts.py` | **Passed** — `studio_price`/`studio_go("move")` with missing or unknown engine refuses paid Runway fallback; Wan video refuses starter + locked `back_body` and omits unconditioned refs from `verification_record`; `"Exactly two cups and three bowls."` and `"Two ceramic cups. Keep the wooden box closed."` preserve all requirements; live Node.js execution of `ui/index.html` confirms uninspected alternate displays `"Not inspected"`, `asset_fingerprint` survives `charProfileLoad -> forgeSubjectRefsPayload`, and `stLoad()` 1→0 transition invokes `0` balance fetches; `.pins.json` survives manager restart and releases on `unpin_job` |
| **Gate 5 Adversarial Suite** (`RC-12`, `RC-13`, `RC-14`, `RC-15`, `RC-17`) | `tests/test_rc3_gate5_dispatch_receipts_handoff_sandbox.py` | **Passed** — Wan 5B correction selects `wan22_i2v_5b_turbo_gguf_api.json` + `WAN5B_MODEL` + `Wan2.2_VAE.safetensors`; resolved Wan seed `1428693541` recorded verbatim; FLUX still records `frames=1`; real `ffmpeg` 1s 160x96 8-frame MP4 probed as `width=160, height=96, frames=8, duration_s=1.0`; resident Ollama blocks ordinary ComfyUI launch; failed pre-VLM `/free` retains `comfy_models_warm=True`; `GenerationBroker(comfyui_dir=None)` advertises `models == []` and rejects submit; `os.setsid()` children in distinct PGIDs are tracked with aggregate `VmRSS` and cleanly terminated |
| **Frontend JS Syntax Check** | `node --check` on extracted `<script>` from `avatar_v2/ui/index.html` | **PASS (`UI_JS_SYNTAX_OK`)** |

---

## 2. Isolated RC3 Packaging, Bytecode Correspondence & Deterministic Tree Digest (`dist/AvatarV2.rc3`)

- **Build Command:**
```bash
.venv314/bin/pyinstaller --noconfirm --clean \
  --distpath /tmp/avatar_v2_rc3_build/dist \
  --workpath /tmp/avatar_v2_rc3_build/build \
  scripts/avatar_v2.spec
mv /tmp/avatar_v2_rc3_build/dist/AvatarV2 dist/AvatarV2.rc3
```
- **Build Environment:** Python `3.14.4` (`.venv314`), PyInstaller `6.22.3`, Linux `7.0.0-38-generic-x86_64-with-glibc2.43`
- **Executable SHA-256 (`dist/AvatarV2.rc3/AvatarV2`):** `cb3fad889b41b10856ff1dfacf9111b7589905943f89d554d33566d59cf4f48e`
- **Deterministic Onedir Tree Digest (`dist/AvatarV2.rc3/`, `42,708` entries):** `473c80d2a1d8f66b8142ba803435f0d8adc01ef40f35bda20f95fccef2fc8540`
- **PYZ Bytecode Correspondence (`CArchiveReader` → `PYZ.pyz`):**
  - **`58 / 58`** `avatar_v2.*` modules matched exact normalized bytecode (`co_argcount`, `co_posonlyargcount`, `co_kwonlyargcount`, `co_nlocals`, `co_flags`, `co_code`, recursive `co_consts`, `co_names`, `co_varnames`, `co_freevars`, `co_cellvars`) compiled from commit `4472e75` (**`0` mismatched**).
- **Bundled Data & Workflow Integrity (`7 / 7` exact SHA-256 matches):**
  - `avatar_v2/ui/index.html` — `263d817d92a3b82a2f821db206acb2f0bb05c4dc099f0660b30eaf5970165a38`
  - `avatar_v2/data/prompt_libraries.json` — `f5d07ab5a7364ff7e8c5771752f74135b36fb9203db6d993733d0b64c0dc3caa`
  - `avatar_v2/sdk_client.py` — `46a249b14cf213ee4da0efb8a72737c8991c9c0f644f13642c21ede55671b0a5`
  - `workflows/animatediff_sd15_api.json` — `3bcfa5560c10ead491c79b9350e38c3bf168f139cbbb900ce5d3048e02412d4d`
  - `workflows/animatediff_sdxl_api.json` — `3bcfa5560c10ead491c79b9350e38c3bf168f139cbbb900ce5d3048e02412d4d`
  - `workflows/wan22_i2v_14b_gguf_api.json` — `062c81a58b52fdb9402522262d9f08aa7d9e5e41cafee772280ed90f530207a0`
  - `workflows/wan22_i2v_5b_turbo_gguf_api.json` — `7f1cd4d5b8dd192e37ce9e62233ee30a4792b97a1e8ee021dc211fe4de00c1d4`

---

## 3. Packaged Desktop Self-Test (`AVATAR_V2_UITEST=1` from `/tmp` with Clean Isolated `HOME`)

Executed `/home/jd/avatar_project/dist/AvatarV2.rc3/AvatarV2` with `cwd=/tmp`, `HOME=/tmp/rc3_uitest_os_home`, `AVATAR_V2_HOME=/tmp/rc3_uitest_home`, and `AVATAR_V2_UITEST=1`:

- **Views (`4 / 4`):** `make=true`, `forge=true`, `library=true`, `settings=true`
- **Recipes (`7 / 7`):** `look`, `photos`, `voice`, `talk`, `act`, `move`, `scene` — all `shown=true`
- **Deterministic UI Invariants:** `tiles=7`, `forge_prompt=true`, `forge_models=true`, `guided_opens=true`, `subject_refs_roles=true`, `character_profiles_ok=true`, `script_panel_shown=true`, `script_sandbox_available=true`, `browse_local_present=true`, `automatic_restored=true`, `avatars_ok=true`, `state_ok=true`, `move_without_master_explains=true`, `library_ok=true`, `missing_image_refused=true`, `missing_video_refused=true`, `viewer_opens=true`, `viewer_missing_explained=true`, `viewer_enter_uses=true`, `viewer_next=true`, `viewer_escape=true`, `select_black_on_white=true`, `settings_limits=true`
- **Environment Observations:** `engine_state="not_installed"` (confirming isolated `HOME`), `avatars_tab="0 saved avatars"`, `all_tab="0 items"`
- **JavaScript Console Errors:** **`[]` (0 errors)**

---

## 4. Protected Assets, Branches & Hardware State Verification

| Protected Asset / Reference | Expected Pre-RC3 Value | Post-RC3 Verified Value | Status |
| --- | --- | --- | --- |
| Production `dist/AvatarV2/AvatarV2` | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` | **UNCHANGED** |
| Original RC1 `dist/AvatarV2.next/AvatarV2` | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` | **UNCHANGED** |
| Preserved RC2 `dist/AvatarV2.rc2/AvatarV2` | `5eb0a9a43a08f10d433dd5829320092af47584ea898aaa675430d74ab70cfd1c` | `5eb0a9a43a08f10d433dd5829320092af47584ea898aaa675430d74ab70cfd1c` | **UNCHANGED** |
| Owner Config `~/.avatar_v2/desktop.json` | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` | **UNCHANGED** |
| Branch `sprint-1-gpu-memory-foundation` | `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` | `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` | **UNCHANGED** |
| Branch `rc1-astra-remediation` | `87ce485e7879f1ea085596dece66a95d9c60b9ff` | `87ce485e7879f1ea085596dece66a95d9c60b9ff` | **UNCHANGED** |
| Branch `audit/astra-rc1-independent` | `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` | `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` | **UNCHANGED** |
| Branch `audit/astra-rc2-recheck` | `c63fb9fb7d4598db2a122b16b4c3a03feb1a2316` | `c63fb9fb7d4598db2a122b16b4c3a03feb1a2316` | **UNCHANGED** |
| Secondary NTFS Drive `/dev/nvme1n1` | Untouched | Untouched | **UNCHANGED** |
| Local GPU (`RTX 4070 Laptop GPU`) | Idle desktop baseline | Idle (`1101 MiB` used / `6681 MiB` free) | **CLEAN** |
