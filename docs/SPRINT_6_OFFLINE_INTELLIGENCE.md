# Sprint 6 — Offline Local Intelligence, Capability-Aware Routing & Visual QA Contract

**Date:** 2026-11-25  
**Status:** Software-Verified (`384` passing tests, `0` failures) + Live 4-Model Local Ollama CPU vs Sequential-GPU Benchmark Completed (`~/.avatar_v2/benchmarks/sprint6_local_llm_eval.json`)

---

## 1. Local Language-Model Inventory (Verified Live via `ollama list` — Zero Downloads)

| Installed Model | Disk Size | Quant / Params | Family | Intended Role |
|---|---:|---|---|---|
| `llama3.1:8b` | `4.9 GB` | Q4_K_M / 8B | Llama | Primary structured-JSON candidate |
| `qwen3:8b` | `5.2 GB` | Q4_K_M / 8B | Qwen3 | Primary structured-JSON candidate (`think: false`) |
| `qwen3.5:9b` | `6.6 GB` | Q4_K_M / 9.7B | Qwen3.5 | High-capacity semantic candidate (`think: false`) |
| `ministral-3:8b` | `6.0 GB` | 8B | Mistral | Alternative instruction candidate |
| `llama3.2:3b` | `2.0 GB` | 3B | Llama | Fast lightweight CPU-only candidate |
| `qwen3-vl:8b-instruct` | `6.1 GB` | Q4_K_M / 8.8B | Qwen3-VL | Reserved for Sprint 7 local visual QA |
| `qwen3-vl:4b-instruct` / `2b` | `3.3 / 2.6 GB` | 4B / 2B | Qwen3-VL | Lightweight Sprint 7 VLM candidates |

No new models were downloaded (`44 GB` free NVMe).

---

## 2. Offline Intelligence Architecture

Lightweight deterministic-first Python orchestration reusing every Sprint 1–5 subsystem without introducing external agent frameworks or new UI panels:

```text
Automatic / Guided / Script User Prompt (+ optional starter & SubjectReferenceSet)
  -> Intent Interpreter (avatar_v2/intent_interpreter.py)
       ├─ Deterministic requirement & intent parser (11 intents, zero latency)
       └─ Optional untrusted local Ollama JSON assist (avatar_v2/ollama_client.py, loopback-only, CPU num_gpu=0 when ComfyUI warm)
            -> Strict merge (never overrides original_prompt or locked attributes; deterministic fallback on any failure)
  -> Quality-First Capability-Aware Workflow Router (avatar_v2/workflow_router.py)
       ├─ Quality-first defaults: FLUX.2 9B (stills), Wan 2.2 14B (video)
       ├─ Intelligent reference deduplication, quality warnings & provisional pixel budgets (1.0 / 0.65 / 0.5 MP)
       ├─ Single-stage vs multi-stage planning (FLUX starter -> Wan I2V; FLUX matched rear keyframe -> Wan FLF2V)
       └─ ResourcePlanner admission + PromptCompiler validation per stage
  -> Machine-Readable Verification Record (avatar_v2/inspection_contract.py)
       └─ Preserves original requirements, compiled prompt, user_source vs generated_intermediate assets,
          expected output characteristics, and bounded CorrectionBudget for Sprints 7 & 8
```

## 3. Frontend Preservation & Security Boundaries

- **Zero new primary UI controls:** No LLM selector dropdown, no agent configuration panel, no routing dashboard, and zero new `JS_API_METHODS` were added to `avatar_v2/ui/index.html`.
- **Untrusted LLM Isolation:** `LocalOllamaClient` enforces loopback-only endpoints (`127.0.0.1` / `localhost`). LLM suggestions are validated against `StructuredPromptSpec.detect_conflicts()`, `policy.depicts_minor`, and unsafe-directive filters (blocking cloud/Runway or shell injection), with deterministic fallback whenever Ollama is offline, slow, or returns malformed JSON.
- **Explicit Starter Preservation:** Multi-stage video plans never silently overwrite a user-supplied starter image; they distinguish `origin="user_source"` from `origin="generated_intermediate"` in `GenerationVerificationRecord`.

## 4. Live Controlled Local Ollama Benchmark Results (`scripts/eval_local_llm_intelligence.py`)

Evaluated on the idle workstation across 6 challenging semantic + adversarial prompts (exact counts, typography, spatial relations, editing, front-to-back video turns, and prompt-injection attempting Runway cloud / shell commands). Raw data: `~/.avatar_v2/benchmarks/sprint6_local_llm_eval.json`.

| Model | Mode | JSON Reliability | Intent Accuracy | Prompt Preservation | Safety Compliance | Mean Latency | Post-Call VRAM | Selection Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`qwen3:8b`** (`5.2 GB`) | **Sequential GPU** | **`1.00`** | **`1.00`** | **`1.00`** | **`1.00`** | **`4.22 s`** | `6,312 MB`* | **Primary default (cold GPU / sequential)** |
| **`qwen3:8b`** (`5.2 GB`) | **CPU (`num_gpu=0`)** | **`1.00`** | **`1.00`** | **`1.00`** | **`1.00`** | `24.20 s` | **`802 MB`** | **Primary 8B CPU fallback (0 MB VRAM)** |
| **`llama3.2:3b`** (`2.0 GB`) | **CPU (`num_gpu=0`)** | **`1.00`** | **`1.00`** | **`1.00`** | **`1.00`** | **`13.67 s`** | **`813 MB`** | **Fastest CPU-only option when ComfyUI is warm** |
| **`llama3.2:3b`** (`2.0 GB`) | Sequential GPU | `1.00` | `1.00` | `1.00` | `1.00` | `6.20 s` | `3,615 MB`* | Good secondary GPU fallback |
| **`llama3.1:8b`** (`4.9 GB`) | CPU (`num_gpu=0`) | `1.00` | `1.00` | `1.00` | `1.00` | `25.08 s` | `862 MB` | Strong runner-up |
| **`llama3.1:8b`** (`4.9 GB`) | Sequential GPU | `1.00` | `1.00` | `1.00` | `1.00` | `5.79 s` | `6,186 MB`* | Strong runner-up |
| `qwen3.5:9b` (`6.6 GB`) | CPU (`num_gpu=0`) | `0.833` | `1.00` | `1.00` | `1.00` | `37.64 s` | `802 MB` | **Deprioritized** (slowest + 1 CPU timeout) |
| `qwen3.5:9b` (`6.6 GB`) | Sequential GPU | `1.00` | `1.00` | `1.00` | `1.00` | `14.97 s` | `6,958 MB`* | **Deprioritized** (3.5× slower than `qwen3:8b`) |

*\*Note on GPU `post_vram`: Ollama unloads GPU weights asynchronously shortly after returning HTTP 200. When ComfyUI has warm FLUX.2 9B weights resident (`5.5 GB` VRAM), running an 8B LLM on GPU would evict FLUX.2 9B (costing `+20s` cold reload + `+4.2 GB` RSS), completely erasing any LLM GPU speedup. Therefore `choose_llm_execution_schedule()` defaults to **CPU-only (`num_gpu=0`) whenever ComfyUI is warm or active** (`0 MB` VRAM disruption), and only permits sequential GPU (`keep_alive=0`) when the GPU is already cold.*

**Empirical Model Selection Justification (Section 3):**
- Selected **`qwen3:8b`** as primary structured-intelligence model and **`llama3.2:3b`** as fast lightweight CPU option strictly on measured latency + 100% JSON/intent/safety reliability — explicitly deprioritizing newer/larger `qwen3.5:9b` based on measured data.
- **No model downloads required.**

## 5. Known Limitations & Next-Sprint Readiness

- Formal human scoring (`passed` / `subjective_quality_1_to_5`) of the immutable 13-category baseline and multi-view outputs remains a human evaluation step (intentionally never fabricated).
- Adaptive multi-reference pixel budgets (`1.0 / 0.65 / 0.5 MP`) are empirically calibrated provisional defaults for 8 GB VRAM; formal A/B human quality comparison at `0.5 MP` vs `1.0 MP` can be evaluated further as needed.
- Sprint 7 (Local VLM Visual Inspection using installed `qwen3-vl:8b-instruct` / `4b` / `2b`) and Sprint 8 (Bounded Corrective Regeneration) integrate directly against `GenerationVerificationRecord` (`avatar_v2/inspection_contract.py`).
