# Avatar V2 RC4 — Invariant Architecture & Security Boundary Design

**Remediation Branch:** `rc4-astra-remediation`  
**Base Commit (Exact RC3 Packaged Build):** `4472e7522f13b684738327b6dd1c6694c563279c`  
**P0 Invariant Commit (`F01`–`F02`):** `21cb0303fcb86b1e946b75925384e29b996de69d`  
**P1/P2 Invariant Commit (`F03`–`F13` + `AVATAR_V2_RC4=1`):** `4dee3e57b5071cf3a61317688f8ff086dc903ff3`  
**Independent Audit Baseline:** Astra RC3 Recheck @ `audit/astra-rc3-recheck` (`9909031c4c3e9b39d0b4c29f6d0619be564d8a74`), reconciling all 23 original RC1 findings (`5d53b2acf5c5b83d8c44de74f4babee6419cdff0`) and RC2 recheck (`c63fb9fb7d4598db2a122b16b4c3a03feb1a2316`).

---

## 1. Methodology Shift (RC3 → RC4)

RC1–RC3 remediations closed individual reproducers while leaving bypass paths, split-brain persistence states, and higher-layer checks that could be bypassed by alternate callers. RC4 enforces every safety and consistency property at the **lowest practical shared boundary** so that no caller—automatic, manual, restart/recovery, retry, studio, or script—can bypass the invariant.

---

## 2. Enforced Invariants

### Invariant 1 — Renderer Ownership & Global Control Authorization (`F01` / `RC-01`)

**Statement:**  
No process signal, global `POST /free`, global `POST /interrupt`, or engine shutdown/restart may reach a renderer unless the application proves exclusive ownership of that exact live process AND exact listener socket `(address_family, concrete_ip, port)` immediately before the operation, with HTTP redirects disabled and concrete IP literal pinning.

**Shared Enforcement Boundary:**
1. **`avatar_v2/ownership.py`**
   - `verify_process_ownership()`: Fails closed on Linux whenever `record.boot_id` OR live `/proc/sys/kernel/random/boot_id` is empty/missing (`missing_record_boot_id` / `missing_live_boot_id`), in addition to PID, `start_ticks`, executable realpath, and `ownership_class == "owned_private"`.
   - `pid_listening_endpoints()` & `authorize_owned_endpoint(record, base_url, require_listening=True) -> (ok, pinned_url, reason)`: Inspects `/proc/<pid>/fd` + `/proc/net/tcp` + `/proc/net/tcp6` (failing closed if `/proc` is unreadable). Enforces exact `(family, ip, port)` matching: an IPv4 listener (`127.0.0.1` / `0.0.0.0`) NEVER authorizes `[::1]`; an IPv6 listener (`::1` / `::`) NEVER authorizes `127.0.0.1`; `localhost` is resolved and pinned to the proven concrete IP literal (`http://127.0.0.1:<port>` or `http://[::1]:<port>`).
   - `send_signal_if_owned()` & `OwnedProcessHandle.send_signal()`: Requires Linux kernel `os.pidfd_open()` + `signal.pidfd_send_signal()` after live verification (fails closed if unavailable — no TOCTOU `os.kill(pid)` fallback).
2. **`avatar_v2/providers/comfyui.py`**
   - `ComfyUIProvider.__init__(..., ownership_authorizer=...)` raises `RendererOwnershipError(PermissionError)` if authorization fails or returns a non-pinned target.
   - `free_memory()` and global `interrupt(allow_global=True)` invoke `_authorize_global_target()`, pin the concrete IP URL, set `follow_redirects=False`, and explicitly reject any `3xx` redirect response (`RendererOwnershipError`).
   - Prompt-scoped `interrupt(prompt_id=...)` uses `follow_redirects=False` and never escalates to global interrupt.
3. **`avatar_v2/gpu_memory.py`**
   - `GPUMemoryManager.maybe_cleanup()` enforces `ownership_authorizer` before **every** `provider.free_memory()` call (automatic pre-render, post-render, model-switch, pre-retry, pre-VLM, and manual cleanup).
   - Preserves mandatory environment controls `AVATAR_V2_GPU_AUTOCLEAN`, `AVATAR_V2_GPU_THRESHOLD` (default 85%), and `AVATAR_V2_GPU_CHECK_INTERVAL`, and emits unified release-evidence fields (`cleanup_requested`, `cleanup_performed`, `ownership_denied`, `effective`, `vram_released_verified`).
4. **`avatar_v2/desktop.py` & `avatar_v2/render_queue.py`**
   - Every `ComfyUIProvider` instance (in `RenderQueue._make_provider`, `DesktopAPI.gpu_free()`, `DesktopAPI.shutdown_engine()`, `DesktopAPI._restart_comfyui_engine()`, and pre-VLM cleanup) is wired through `_build_comfy_provider()` with `_authorize_comfy_global_op` and snapshots **one immutable `ProviderConfig`** per operation (eliminating config-flip races).

**Entry Points Audited:** Automatic pre-render cleanup, automatic post-render cleanup, model-transition unload, failure recovery cleanup, pre-VLM unload, manual `gpu_free()`, `shutdown_engine()`, `_restart_comfyui_engine()`, queue cancellation (`interrupt`).

---

### Invariant 2 — Filesystem Ownership, Exclusive Acquisition & Locked Deletion (`F02` / `RC-02`)

**Statement:**  
Application file creation MUST NEVER overwrite a pre-existing file, symlink, or hard link. Registration MUST require non-empty provenance (`job_id`), a regular non-symlink single-link file (`S_ISREG`, `st_nlink == 1`) verified on an open `O_NOFOLLOW` descriptor, and atomic commit under a fail-closed cross-process lock. Deletion MUST verify ownership, `dir_fd`/`O_NOFOLLOW` `fstat`, descriptor SHA-256, final `lstat`, `os.unlink(..., dir_fd=dir_fd)`, and registry removal inside the same exclusive transaction lock.

**Shared Enforcement Boundary (`avatar_v2/ownership.py`):**
- `_RegistryTransactionLock` (`RegistryLockError(OSError)`): Combines per-path `threading.RLock` + `fcntl.flock(LOCK_EX | LOCK_NB)` with bounded retry; fails closed if the lock cannot be acquired (never mutates unlocked).
- `allocate_exclusive_path()`, `reserve_fresh_output_path()`, `copy_file_exclusive()`: Opens destination exclusively via `O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW` (allocating a suffixed sibling path on collision rather than overwriting), streams bytes, `fsync`s, verifies SHA-256, and unlinks partial files on failure.
- `ManagedAssetRecord.from_dict()` & `register_managed_asset()`: Rejects empty `job_id`, rejects symlinks/hard links (`st_nlink != 1`), hashes the open `O_NOFOLLOW` fd directly, and commits atomically under `_RegistryTransactionLock`.
- `delete_managed_asset_if_owned()`: Executes full ownership verification + `dir_fd`/`O_NOFOLLOW` verification + descriptor SHA-256 check + `os.unlink(name, dir_fd=dir_fd)` + registry removal atomically under `_RegistryTransactionLock`.

**Entry Points Hardened:** `render_queue.py` (`_archive_outputs`, `_extract_candidate_frame`, removal of unarchived external output auto-registration), `desktop.py` (`_execute_local_correction_step`), `desktop_studio.py` (`_studio_finish_scene`, `studio_delete`), `providers/comfyui.py` (`_copy_to_input`), `reference_import.py`, `voiceover.py`.

---

### Invariant 3 — Durable Queue State Machine & Expected Media Verification (`F06`, `F07`, `F08`, `F11` / `RC-10`, `RC-11`, `RC-16`, `RC-18`)

**Statement:**  
1. No job may be admitted into executable memory or wake the worker until its durable item record is persisted to disk (`add()` / `add_many()` roll back on `OSError`).
2. Once submitted (`prompt_id` set, `phase in POST_SUBMIT_PHASES`, or `submission_token` recorded), no timeout, transport error, or failure may clear `prompt_id` or automatically requeue the job; it transitions strictly to `phase="recovery_required"`.
3. Durable submission acknowledgement persistence failure raises and transitions to `phase="recovery_required"` rather than silently continuing.
4. Committed terminal `done/complete` history (`history/<id>.json`) always dominates stale active checkpoints (`items/<id>.json`) across restarts and retries (`_has_committed_terminal_history()` + monotonic `state_revision`), retiring stale active files idempotently without clobbering history.
5. Corrective generation persists initial inspection state BEFORE invoking `CorrectionCoordinator` and durably persists each candidate path + SHA-256 BEFORE reinspection begins (halting into `recovery_required` on persistence failure).
6. Local ComfyUI renders MUST produce at least one readable, non-empty media file matching expected workflow suffix (`IMAGE_SUFFIXES` for stills, `VIDEO_SUFFIXES` for Wan video) or raise `ComfyUIExecutionError` (never `done / complete / []`).

**Shared Enforcement Boundary:** `avatar_v2/render_queue.py` (`QueuedRender.state_revision`, `add`, `add_many`, `_record_local_submitted`, `_handle_failure`, `_restore`, `_persist_history`, `_execute`, `_item_to_dict`) and `avatar_v2/desktop.py` (`_run_visual_inspection`, `_persisting_gen_hook`).

---

### Invariant 4 — Live Inspection Evidence Validity & Truthful Corroboration (`F05` / `RC-08`, `RC-09`)

**Statement:**  
1. `compute_inspection_status()` (`avatar_v2/inspection_contract.py`) MUST NOT return `"pass"` / `"Requirements verified"` when any mandatory `requires_human_review` OR `hybrid_corroborated` criterion lacks `independent_corroboration=True` (unless assessed by `"human"` or `"deterministic"`).
2. Every reopened result (`DesktopAPI.history()` and `DesktopStudioMixin._studio_results()`) runs `_revalidate_item_inspections()`, computing live `sha256_file()` and comparing against the recorded `output_sha256` and `requirement_set_hash`. Any missing, replaced, or hash-mismatched output invalidates its per-output inspection AND job-level verdict to `"Not inspected"` (`inspected=False, stale=True`).
3. The shipped UI (`avatar_v2/ui/index.html`, `_resolveOutputInspection()`) never falls back to job-level `r.correction.ui_status_label` when a per-output record is marked `stale === true` or `inspected === false`.

---

### Invariant 5 — Deferred Reference Authorization, Mandatory Criteria & Durable Pin Admission (`F03`, `F04`, `F13` / `RC-03`, `RC-06`, `RC-07`, `RC-19`)

**Statement:**
1. **Deferred End-Picture Binding (`F03`):** `DesktopStudioMixin.forge_go()` authorizes, stages (`_stage_reference_if_external`), SHA-256 fingerprints, and pins `end_image` immediately during the user's JsBridge call, forwarding `staged_end_image` + `end_image_sha256` into `forge_scene`. `_forge_continue_scene()` requires the staged file to exist and match its SHA-256 (failing cleanly rather than silently dropping `LAST_FRAME`) and pins it to the spawned Wan scene job.
2. **Mandatory Criteria Extraction (`F04`):** `avatar_v2/intent_interpreter.py` expands `_OBJECT_STATE_RE` to cover modal/auxiliary phrases (`"must remain closed"`, `"should stay open"`, etc.) and inspects coordinated sub-clauses inside compound sentences so an earlier matched count never hides a mandatory object-state clause in the same sentence.
3. **Fail-Closed Pin Persistence & Release (`F13`):** `avatar_v2/reference_picker.py` (`_save_pins_locked()`) propagates `OSError` and rolls back in-memory pin mutations instead of swallowing write failures; `RenderQueue.cancel()` invokes `on_job_finished(item)` when cancelling queued jobs so reference pins are released on every terminal transition.

---

### Invariant 6 — Resolved Video Metadata Fidelity (`F09` / `RC-13`)

**Statement:**  
`render_queue._extract_resolved_workflow_params()` recognizes all Wan 2.2 conditioning nodes—`WanImageToVideo`, `WanFirstLastFrameToVideo`, and `WanFunControlToVideo`—extracting exact resolved `width`, `height`, and `frames` (`length`), while storing `requested_params` separately from `submitted_params` so FLF2V receipts never fall back to ShotSpec defaults (e.g., `144` frames instead of `81`).

---

### Invariant 7 — Bidirectional VRAM Handoff & Fail-Closed Sandbox Telemetry (`F10`, `F12` / `RC-14`, `RC-17`)

**Statement:**
1. **Bidirectional VRAM Handoff (`F10`):** `avatar_v2/ollama_client.py` (`fetch_ollama_ps_state()`) distinguishes `"offline"` (`ConnectError` / `ConnectionRefusedError`) from `"unknown"` (timeout, HTTP 5xx, malformed JSON). `RenderQueue._ensure_ollama_vram_released_before_comfy()` fails closed on `"unknown"` (never proceeding into ComfyUI on ambiguous Ollama GPU state). `DesktopAPI._cleanup_verified_released()` validates the production `GPUMemoryManager` event schema (`vram_released_verified` / `cleanup_performed` / `cleanup_requested + effective`).
2. **Fail-Closed Sandbox Telemetry & `pidfd` Termination (`F12`):** `avatar_v2/script_runner.py` (`_collect_descendant_tree_stats()`) returns `(telemetry_ok, pids, total_rss_kb, error)` and fails closed if `/proc` cannot be enumerated or live `VmRSS` cannot be read. `_watch_process()` stops broker admission and terminates the script run when telemetry is lost, using Linux `os.pidfd_open` + `signal.pidfd_send_signal` across verified descendants.

---

## 3. Explicit OS-User Trust Boundary

Avatar V2 runs as a single desktop user on Linux without privileged root daemons or kernel LSM confinement:
- **Enforced within the application boundary:** Ownership verification against stale/reused PIDs, IPv4/IPv6 localhost confusion, HTTP 3xx redirects, accidental overwrites of pre-existing/owner files, concurrent app/worker registry races (`fcntl.flock`), silent persistence loss, stale inspection claims, and unverified VRAM handoffs.
- **Outside the unprivileged application boundary:** A malicious process running as the **same OS UID** with arbitrary shell access can directly modify files in `~/.avatar_v2` or signal processes without going through Avatar V2. Script mode mitigates accidental/adversarial child escapes via process-tree `/proc` tracking, broker token gating, fail-closed telemetry, and `pidfd` termination, but is not a kernel-enforced VM or container sandbox against arbitrary same-UID kernel exploits.
