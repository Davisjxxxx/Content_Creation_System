# Avatar V2 — Script Import and Execution Integration Architecture

**Status:** Approved for Sprints 4–5 (2026-11-25)  
**Prerequisite Gates:** Sprint 2 Structured Prompt Schema (`PromptSpec`) and Sprint 3 Central `ResourcePlanner` must be established before Script mode implementation.

---

## 1. Three-Mode Interaction Model

Script mode replaces the previously proposed Manual mode while preserving the existing approved frontend:

1. **Automatic:** Direct natural-language prompt entry, optional reference assets, automatic prompt interpretation, model selection, workflow routing, and internal resource optimization.
2. **Guided:** Optional task-applicable creative attribute dropdowns, reusable scene recipes, attribute locking, controlled randomization, and model-specific prompt compilation.
3. **Script:** Local `.py` selection (+ optional JSON/YAML config, prompt library, profile, or reference directory), static AST validation (no execution on import), concise compatibility findings, Start / Pause / Resume / Stop, progress, and output review.

---

## 2. Static AST Inspection (`avatar_v2/script_inspector.py`)

- Parses scripts strictly via `ast.parse()` (**never** `importlib`, `exec`, or `eval` during inspection).
- Detects: imported dependencies (stdlib / SDK / approved venv / missing / local), referenced JSON/YAML configs, target backend (Native SDK / A1111 / ComfyUI / direct diffusers), model checkpoints, prompt syntax (e.g., A1111 `(token:weight)`), parameters/toggles, I/O paths, `while True` / continuous loops, batch loops, in-script `torch.cuda` / `nvidia-smi`, network calls, subprocess execution, filesystem mutations, and static bugs.
- Classifies scripts into four tiers:
  - **Tier A — Native Avatar V2 SDK Script** (`avatar_v2_sdk`)
  - **Tier B — Legacy A1111 HTTP Script** (`/sdapi/v1/txt2img`, `a1111_client`)
  - **Tier C — Declarative / Prompt-Library Script** (static prompt/recipe extraction)
  - **Tier D — Incompatible / Direct-Weights Script** (`diffusers.from_pretrained`, arbitrary training — blocked from bypassing `RenderQueue`)
- Static analysis is treated strictly as compatibility preflight, **never** as proof of safety.

---

## 3. Legacy Script Evaluation: `continuous_generator_juggernaut.py`

Static AST audit of `/home/jd/avatar_project_legacy/continuous_generator_juggernaut.py` (574 lines, never imported):
- **Backend:** Calls `generation.a1111_client.txt2img` / `health_check` against Automatic1111 (`http://127.0.0.1:7860`).
- **Config Dependencies:** `config/models.json` and `config/generation_profiles.json` relative to `PROJECT_ROOT`.
- **Non-Interchangeable Parameters:** SDXL checkpoint (`Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors`, not installed), `CLIP_stop_at_last_layers=2`, `DPM++ 2M SDE`, `steps=30-45`, `cfg_scale=4.5-5.5`, `enable_hr` (`4xnmkd-siax_200k`), `restore_faces=True`, and `(tag:1.4)` weighted syntax are **not** directly interchangeable with ComfyUI FLUX.2.
- **Uncontrolled Loop & CUDA Anti-Pattern:** Unbounded `while True:` with `time.sleep(8)` and no item/disk/RAM caps; runs `import torch; torch.cuda.empty_cache()` inside the client process (ineffective against a separate server process and wastes ~300–500 MB VRAM).
- **Static Bugs & Policy Constraints:** `FORCE_AGE = True` injects literal `"True"` into prompts; references undefined `ENHANCED_REALISM_DETAILS`; missing comma concatenates two poses; contains hardcoded legacy adult/youth-ambiguous defaults that **must never** be blindly migrated and must pass `avatar_v2/policy.py` before any generation.

### Approved Three-Path Compatibility Strategy (Sprint 4–5)
1. **Static Structured Prompt Extractor:** Extracts compatible scene recipes, attribute categories, and profiles into clean JSON/YAML without executing Python or importing hardcoded legacy defaults.
2. **Native Avatar V2 Python SDK (Preferred Long-Term):** Thin stdlib client (`avatar_v2_sdk`) communicating over a per-run authenticated Unix domain socket, delegating 100% of generation, CUDA management, QA, and cancellation to `RenderQueue`, `ResourcePlanner`, `GPUMemoryManager`, `policy.py`, and `candidate_qa.py`.
3. **Optional Local A1111 Compatibility Adapter (Opt-In Only):** Bounded, loopback/UDS-only shim implementing documented `/sdapi/v1/options`, `/sdapi/v1/txt2img`, and `/sdapi/v1/img2img` mappings routed through `policy.py`, `ResourcePlanner`, and `RenderQueue`. Explicitly rejects unsupported parameters/extensions (`ADetailer`, arbitrary `alwayson_scripts`, uninstalled checkpoints without explicit user-approved mapping), never auto-enables, never exposes to external networks, and never modifies original script files.

---

## 4. OS Isolation & Continuous Generation Guardrails

- **Linux Sandbox (`/usr/bin/bwrap` + `/usr/bin/prlimit` — verified installed):**
  - `bwrap --unshare-all --unshare-net --die-with-parent --new-session --clearenv`
  - Read-only bind mounts for approved Python venv, script directory, and explicitly selected reference directory; writable mount **only** to `~/.avatar_v2/script_runs/<run_id>/output` and ephemeral `/tmp`.
  - Complete hiding of `~/.avatar_v2/secrets/`, `desktop.json`, `~/.ssh`, and unrelated `$HOME` paths; `RUNWAYML_API_SECRET` never exposed.
  - **`CUDA_VISIBLE_DEVICES=""`** inside the script subprocess prevents imported scripts from creating rogue CUDA contexts or competing with ComfyUI for the 8 GB RTX 4070; all GPU work is brokered through `RenderQueue`.
  - Complementary `prlimit` CPU/RAM/process/FD caps + wall-clock timeout + controlled sandbox-escape security tests required before claiming isolation is effective.
- **Host-Enforced Continuous Generation Guardrails:**
  - Default limit: **10 items/run** (owner-configurable, including deliberate authorization for extended continuous runs).
  - Safe-boundary **Pause** (finishes active render cleanly, pauses before next item), **Resume**, and visible **Stop** (cancels active queue item + terminates sandbox group + preserves completed outputs).
  - Predictive storage guard (Normal `>=30 GB`, Warning `<30 GB`, Restricted `<20 GB`, Emergency Stop `<15 GB`), graduated RAM/swap/PSI/thermal guard, 3-failure circuit breaker, and **local-only default** (zero cloud credit spend without explicit host-enforced approval).
