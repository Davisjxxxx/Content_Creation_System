# Avatar V2 RC4 — Adversarial Acceptance & Verification Evidence

**Remediation Branch:** `rc4-astra-remediation`  
**Build Source Commit:** `4dee3e57b5071cf3a61317688f8ff086dc903ff3` (including P0 commit `21cb0303fcb86b1e946b75925384e29b996de69d`)  
**Packaged Candidate:** `dist/AvatarV2.rc4/AvatarV2`  
**Build Environment:** `.venv314/bin/pyinstaller` (PyInstaller `6.22.3`, Python `3.14.4`, Linux `7.0.0-38-generic-x86_64`, glibc `2.43`)

---

## 1. Layer A — Implementation Boundary & Adversarial Regression Suites

### Clean-HOME Full Suite
Executed in a completely clean temporary `HOME` directory with zero dependency on `~/.avatar_v2` or owner media:

```bash
rm -rf /tmp/avatar_rc4_iso_home && mkdir -p /tmp/avatar_rc4_iso_home && \
HOME=/tmp/avatar_rc4_iso_home .venv/bin/pytest -q
```

- **Result:** **490 / 490 PASSED, 0 failed, 0 skipped** (exit code `0`).

### New RC4 Adversarial Suites Added
1. **`tests/test_rc4_p0_ownership_invariants.py`** (P0 `F01` & `F02`):
   - Live HTTP fixture server testing shared-renderer zero-POST guarantee across `GPUMemoryManager.maybe_cleanup()`, `ComfyUIProvider.free_memory()`, `ComfyUIProvider.interrupt()`, `DesktopAPI.gpu_free()`, and `DesktopAPI.shutdown_engine()`.
   - Live HTTP 307 redirect server redirecting `/free` to a healthy foreign fixture server -> refused with `RendererOwnershipError`, 0 POSTs received by target.
   - Real OS socket + `/proc` IPv4 vs IPv6 separation (`127.0.0.1` listener refuses `[::1]`; `[::1]` listener refuses `127.0.0.1`; `localhost` pins strictly to concrete IP literal).
   - Missing `boot_id` and unreadable `/proc/<pid>/fd` fail closed.
   - Config-flip race during `gpu_free()` / `shutdown_engine()` prevented by single immutable `ProviderConfig` snapshot.
   - Archive destination collision preserves pre-existing `presenter_master.png` byte-for-byte via `O_CREAT | O_EXCL | O_NOFOLLOW` (`copy_file_exclusive()`).
   - Real separate OS subprocess holding `fcntl.flock` verifies fail-closed `RegistryLockError` under lock contention; 10-thread concurrent `register_managed_asset()` preserves all 10 entries with zero lost updates.
   - Empty `job_id`, symlink/hard-link, unarchived external provider output auto-registration, and replaced-file deletion all refused.
2. **`tests/test_rc4_p1_durability_verification_metadata.py`** (P1/P2 `F03`–`F13`):
   - `F03`: Synchronous `forge_go(..., end_image=...)` authorization, staging, SHA-256 pinning, and Phase-2 tamper/missing refusal.
   - `F04`: Modal/auxiliary object-state extraction (`"There are 3 mugs on the table and the wooden box must remain closed."`) in compound sentences.
   - `F05`: `hybrid_corroborated` without `independent_corroboration` downgrades to `"warn"` / `"Some requirements uncertain"`; live SHA-256 revalidation on `history()` and `_studio_results()` invalidates replaced files to `"Not inspected"`.
   - `F06` / `F07` / `F08`: Persist-before-admit rollback on `add()`/`add_many()` `OSError`; post-submit timeout keeps `prompt_id` and transitions to `recovery_required`; ack persistence `OSError` raises & enters `recovery_required`; committed `done/complete` history dominates stale active checkpoints on restart; candidate persistence failure halts before reinspection; callable `_`-prefixed runtime hooks stripped from JSON serialization.
   - `F09`: `WanFirstLastFrameToVideo` and `WanFunControlToVideo` extract exact `480x832, 81` into `submitted_params` while keeping `requested_params` separate.
   - `F10`: Live HTTP 500 & timeout on `/api/ps` return `"unknown"` and block ComfyUI dispatch; real `GPUMemoryManager` release events match `_cleanup_verified_released()`.
   - `F11`: Local ComfyUI renders with `outputs=[]`, missing/empty files, or wrong media suffix (PNG for Wan video) raise `ComfyUIExecutionError`.
   - `F12`: Unreadable `/proc` returns `telemetry_ok=False`, stops broker admission, and terminates via `pidfd`.
   - `F13`: `_save_pins_locked()` rolls back & propagates `OSError`; `RenderQueue.cancel()` invokes `on_job_finished` to release pins.

---

## 2. Layer B — Packaged Black-Box Verification (`dist/AvatarV2.rc4`)

### 2.1 Build Command & Artifact Identity

```bash
rm -rf /tmp/avatar_v2_rc4_build dist/AvatarV2.rc4 && \
.venv314/bin/pyinstaller --noconfirm --clean \
  --distpath /tmp/avatar_v2_rc4_build/dist \
  --workpath /tmp/avatar_v2_rc4_build/build \
  scripts/avatar_v2.spec && \
mv /tmp/avatar_v2_rc4_build/dist/AvatarV2 dist/AvatarV2.rc4
```

- **Executable Path:** `dist/AvatarV2.rc4/AvatarV2`
- **Executable SHA-256:** `a63423b57203ea547d9fcde49e4bafec6e92030b47d69e0077f50a1169a6ed0b`
- **Deterministic Tree Digest:** `8cefc6da46adde6a16074ab70fb9b42747360752f436a8fa4bf838467c5343ac` (`42,708` entries)
- **Exact Tree Digest Recipe:** Sorted `os.walk` over `dist/AvatarV2.rc4`, hashing `f"{typ}\0{rel}\0{val}\n"` UTF-8 bytes into SHA-256 where `typ` is `"L"` (symlink target), `"D"` (`""`), or `"F"` (file SHA-256 hex digest).

### 2.2 PYZ Bytecode Correspondence (100% Exact Match)
Using `PyInstaller.archive.readers.CArchiveReader` on `dist/AvatarV2.rc4/AvatarV2` compiled against commit `4dee3e57b5071cf3a61317688f8ff086dc903ff3` under Python 3.14.4:
- **`avatar_v2.*` archived modules:** **`58 / 58` exact bytecode match** (`mismatched=[]`)
- **`desktop_entry` CArchive entry:** **exact bytecode match** (`desktop_entry_ok=True`)

### 2.3 Bundled Data & Workflow Assets (7 / 7 Exact SHA-256 Matches)

| Bundled Relative Path (`_internal/...`) | SHA-256 (Matches Repo Source Byte-for-Byte) |
|---|---|
| `avatar_v2/data/prompt_libraries.json` | `f5d07ab5a7364ff7e8c5771752f74135b36fb9203db6d993733d0b64c0dc3caa` |
| `avatar_v2/sdk_client.py` | `46a249b14cf213ee4da0efb8a72737c8991c9c0f644f13642c21ede55671b0a5` |
| `avatar_v2/ui/index.html` | `d8361504e2ee10fa7d413ad3a1b112e946c52fc1c3f6bae51b0b22aea2a48a9b` |
| `workflows/animatediff_sd15_api.json` | `3bcfa5560c10ead491c79b9350e38c3bf168f139cbbb900ce5d3048e02412d4d` |
| `workflows/animatediff_sdxl_api.json` | `3bcfa5560c10ead491c79b9350e38c3bf168f139cbbb900ce5d3048e02412d4d` |
| `workflows/wan22_i2v_14b_gguf_api.json` | `062c81a58b52fdb9402522262d9f08aa7d9e5e41cafee772280ed90f530207a0` |
| `workflows/wan22_i2v_5b_turbo_gguf_api.json` | `7f1cd4d5b8dd192e37ce9e62233ee30a4792b97a1e8ee021dc211fe4de00c1d4` |

### 2.4 Archived PYZ & Shipped HTML Black-Box Checks
- **Archived PYZ Execution (zero repo source on `sys.path`, loading exclusively from `dist/AvatarV2.rc4/AvatarV2` `PYZ.pyz`):** Verified `F01` (`missing_record_boot_id`), `F02` (`copy_file_exclusive` collision preservation + empty `job_id` refusal + locked `delete_managed_asset_if_owned`), `F04` (`wooden box must remain closed`), `F05` (`hybrid_corroborated` downgrade), `F09` (`WanFirstLastFrameToVideo` `480x832, 81`), `F10` (`fetch_ollama_ps_state` offline distinction), and `F12` (unreadable `/proc` fails closed) -> **`PACKAGED_PYZ_BLACKBOX_ALL_PASS`**.
- **Shipped HTML (`dist/AvatarV2.rc4/_internal/avatar_v2/ui/index.html`) executed in Node.js v24.18.0:** Verified `_resolveOutputInspection()` renders `"Not inspected"` (`cor: null`) for a stale per-output record even when job-level `r.correction.ui_status_label === "Requirements verified"` -> **`PACKAGED_INDEX_HTML_BLACKBOX_PASS`**.

### 2.5 Isolated Packaged Launch from `/tmp` (`AVATAR_V2_UITEST=1`)
Launched `/home/jd/avatar_project/scripts/launch_avatar_v2.sh` from `/tmp` with `AVATAR_V2_RC4=1`, `HOME=/tmp/rc4_uitest_os_home`, `AVATAR_V2_HOME=/tmp/rc4_uitest_home`, `AVATAR_V2_UITEST=1`:
- **Views rendered:** `4 / 4` (`make`, `forge`, `library`, `settings` all `true`)
- **Recipes rendered:** `7 / 7` (`look`, `photos`, `voice`, `talk`, `act`, `move`, `scene` all `shown: true`)
- **UI Checks:** `forge_prompt: true`, `forge_models: true`, `guided_opens: true`, `subject_refs_roles: true`, `character_profiles_ok: true`, `script_panel_shown: true`, `script_sandbox_available: true`, `browse_local_present: true`, `automatic_restored: true`, `avatars_ok: true`, `state_ok: true`, `library_ok: true`, `missing_image_refused: true`, `missing_video_refused: true`, `viewer_opens: true`, `select_black_on_white: true`, `settings_limits: true`
- **WebKit2GTK JS Errors:** `[]` (zero)

---

## 3. Protected Assets & Audit Provenance Integrity

All production, prior release candidates, owner configuration, GPU state, and audit branches were verified byte-for-byte unchanged after RC4 build and testing:

| Protected Asset / Git Ref | Verified SHA-256 / Commit Hash | Status |
|---|---|---|
| `dist/AvatarV2/AvatarV2` (Production) | `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893` | **UNCHANGED** |
| `dist/AvatarV2.next/AvatarV2` (RC1) | `2ee61d892b290e8df9b5303f55725dd56b748d788bf20552c8fb51a78ab23dee` | **UNCHANGED** |
| `dist/AvatarV2.rc2/AvatarV2` (RC2) | `5eb0a9a43a08f10d433dd5829320092af47584ea898aaa675430d74ab70cfd1c` | **UNCHANGED** |
| `dist/AvatarV2.rc3/AvatarV2` (RC3) | `cb3fad889b41b10856ff1dfacf9111b7589905943f89d554d33566d59cf4f48e` | **UNCHANGED** |
| `/home/jd/.avatar_v2/desktop.json` | `5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043` | **UNCHANGED** |
| `sprint-1-gpu-memory-foundation` | `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` | **UNCHANGED** |
| `rc1-astra-remediation` | `87ce485e7879f1ea085596dece66a95d9c60b9ff` | **UNCHANGED** |
| `rc3-astra-remediation` | `8811c63a6ee54f76c379f96e03c0c5195498b593` | **UNCHANGED** |
| `audit/astra-rc1-independent` | `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` | **UNCHANGED** |
| `audit/astra-rc2-recheck` | `c63fb9fb7d4598db2a122b16b4c3a03feb1a2316` | **UNCHANGED** |
| `audit/astra-rc3-recheck` | `9909031c4c3e9b39d0b4c29f6d0619be564d8a74` | **UNCHANGED** |
| GPU (`nvidia-smi`) | `RTX 4070 Laptop GPU, 1130 MiB / 8188 MiB, 5% util` (idle) | **CLEAN** |
