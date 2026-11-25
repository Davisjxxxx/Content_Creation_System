# Sprint 4 — Script Mode, Bubblewrap Isolation, Native SDK & Shared Local Reference Picker

**Date:** 2026-11-25  
**Status:** Software-Verified (CPU-only + Live Linux Bubblewrap OS Isolation Verified); Live Sprint 1–3 GPU Benchmarks Carried Forward (owner ComfyUI PID `91038` still rendering at 99% GPU util)

---

## 1. Delivered Modules & Architecture

Sprint 4 extends Avatar V2's shared execution spine without duplicating inference or GPU memory logic and without redesigning the approved frontend:

```text
Automatic / Guided / Script Mode (#view-forge)
  ├─ Shared Local Reference Picker (My Stuff + Native Browse Local Files)
  │    └─ avatar_v2/reference_picker.py (validation, access control, hash-deduplicated job staging)
  ├─ Static AST Inspector (avatar_v2/script_inspector.py — zero code execution)
  ├─ Static Structured Prompt Extractor (avatar_v2/prompt_extractor.py — literal AST only)
  └─ Controlled Script Execution:
       ├─ Mode A: Static Extracted Run (zero untrusted Python executed)
       └─ Mode B: Bubblewrap OS Sandbox (avatar_v2/script_runner.py + prlimit)
            └─ Standalone stdlib SDK (/sdk/avatar_v2_sdk.py -> avatar_v2/sdk_client.py)
                 └─ Authenticated Unix-socket IPC (--unshare-net compatible)
                      └─ Host GenerationBroker (avatar_v2/script_broker.py)
                           └─ StructuredPromptSpec -> Policy -> ModelRegistry -> PromptCompilers -> ResourcePlanner -> RenderQueue
```

## 2. Security & Isolation Verification Results (Live Host)

Verified directly on Linux (`bubblewrap 0.9.0`, `/usr/bin/bwrap`, `/usr/bin/prlimit`):
- **Fail-Closed Guarantee:** When `bwrap` is unavailable or fails its live namespace probe, `ScriptRunManager.start_sandboxed_run()` raises `SandboxUnavailableError` and refuses unrestricted execution.
- **Network Isolation (`--unshare-net`):** Live test confirmed host loopback TCP (`127.0.0.1`, protecting ComfyUI `:8188`) and external networks are unreachable from inside the sandbox, while authenticated Unix-domain-socket IPC to Avatar V2's `GenerationBroker` works reliably.
- **Credential, Filesystem & GPU Isolation (`--clearenv`, `--dev`, minimal `--ro-bind`):** Live test confirmed injected `RUNWAYML_API_SECRET` env vars, unmounted host files, and `/dev/nvidia0` are completely invisible inside the sandbox.
- **Complementary OS Limits & Cleanup:** `prlimit` enforces virtual address space (`--as`), CPU time (`--cpu`), open files (`--nofile`), and file size (`--fsize`); host wall-clock watchdog + process-group `SIGTERM`/`SIGKILL` cleanly terminates runaway `while True` loops.
- **Frontend Filesystem Guard:** `LocalReferenceManager` blocks arbitrary frontend-supplied paths unless explicitly returned by the native OS file picker or residing within trusted Avatar V2 directories.

## 3. Sprint 3 Carry-Forward Refinements Included

- **Warm FLUX.2 Tiled VAE (`avatar_v2/resource_planner.py`):** Avoids forcing `VAEDecodeTiled` on healthy warm same-model FLUX.2 stills (`<=1024×1280`) where free VRAM is naturally ~1–1.5 GB because weights are already resident; still enables tiled VAE on high-res stills, cold tight-VRAM starts, extended video frames, and Wan 2.2 5B 48-ch VAE (subject to live tiled-vs-untiled GPU measurement).
- **Wan 2.2 CFG/Step Compiler (`avatar_v2/prompt_compilers.py`):** Adapts prompt ordering and negative-prompt warnings cleanly between 4-step CFG 1.0 Lightning/Turbo (`wan22_a14b`, `wan22_5b`) and strong-CFG Wan variants (`wan22_funcontrol`).

## 4. Carried-Forward Live GPU & Desktop Verification Gates

All live GPU benchmarks remain unexecuted while ComfyUI `PID 91038` is actively running:
1. Original vs `--cache-ram 6 24` RAM/swap stability.
2. Cold vs warm FLUX.2 9B latency & model retention (`respect_retention=True`).
3. Wan 2.2 14B ↔ 5B shared `umt5_xxl` component reuse.
4. Tiled vs untiled VAE decoding measurements.
5. Attention compatibility / SageAttention isolated test.
6. 13-category prompt adherence baseline (`avatar_v2/adherence_benchmark.py`).
7. Repeated generation stability, resource recovery, isolated PyInstaller bundle activation (`dist/AvatarV2.next`), and live desktop UI interaction check.
