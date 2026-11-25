# Sprint 6 — Offline Local Intelligence, Capability-Aware Routing & Visual QA Contract

**Date:** 2026-11-25  
**Status:** Software-Verified (`384` passing tests, `0` failures); Live Local Ollama Benchmark (`eval_local_llm_intelligence.py`) queued immediately after the running Sprint 5 13-category Prompt Adherence Baseline (`task-641`) finishes so GPU/CPU measurements remain unconfounded.

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

## 4. Pending Live Measurements (Unconfounded Execution Queue)

1. Complete Sprint 5 13-category Prompt Adherence Baseline (`task-641`, running live on isolated `:8199`, `adh_01`–`adh_03` already succeeded).
2. Run `scripts/eval_local_llm_intelligence.py --mode both` on `llama3.2:3b`, `llama3.1:8b`, `qwen3:8b`, and `qwen3.5:9b` (CPU-only `num_gpu=0` vs sequential GPU `keep_alive=0s`) once `:8199` finishes so CPU/GPU timings are completely unconfounded, and record empirical model selection results here.
