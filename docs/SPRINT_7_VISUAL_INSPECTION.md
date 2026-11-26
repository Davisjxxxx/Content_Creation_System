# Avatar V2 — Sprint 7 Visual Intelligence & Local Quality Inspection

**Status:** Software-verified & live-benchmarked (local Ollama Qwen3-VL + CPU Pillow/ffmpeg temporal QA)  
**Hardware:** RTX 4070 Laptop GPU (8,188 MiB VRAM), Intel Core i9-13980HX (24C/32T), 62 GiB RAM  
**Offline guarantee:** 100% local execution (loopback `127.0.0.1:11434`, zero external API uploads, zero new model downloads)

---

## 1. Hybrid Visual Inspection Architecture

Sprint 7 implements a layered, inexpensive-first local visual inspection pipeline behind the existing approved frontend:

1. **Stage 1 — Deterministic Media Validation (`avatar_v2/visual_inspector.py:inspect_media_deterministic`)**:
   - CPU-only (`<10 ms`, `0 MB` VRAM). Verifies file existence, non-zero size, image/video readability (`PIL` / `ffprobe`), exact dimensions, aspect ratio deviation, container/codec format, duration, FPS, and frame count.
2. **Stage 2 — Lightweight Structural CV (`avatar_v2/visual_inspector.py:inspect_image_structural`)**:
   - Pure Pillow + math (zero heavy CV frameworks or new downloads; `numpy`, `cv2`, `scipy`, `tesseract` are not installed in `.venv314`).
   - Measures Laplacian sharpness/blur, luminance mean/stddev, black/white exposure clipping, flat/corrupt frame detection, 64-bit dHash, and spatial color-region profiles (`left`/`center`/`right`, `top`/`middle`/`bottom`).
3. **Stage 3 — Local OCR & Hybrid Text Cross-Check**:
   - Uses local `tesseract` CLI automatically if present on `PATH`; otherwise delegates typography extraction cleanly to targeted local Qwen3-VL transcription paired with a **deterministic exact-spelling cross-check** (`_deterministic_cross_check_vlm_criterion`) that rejects sycophantic VLM `"passed"` votes when the transcribed text fails exact match.
4. **Stage 4 — Orientation-Aware Multi-View Reference Selection (`select_inspection_references` & `compare_image_to_reference_lightweight`)**:
   - Selects the orientation-matching user source reference (`back_body` / `rear_three_quarter` for rear views, `left_side`/`right_side` for side profiles, `face_identity`/`front_body` for frontal views) and isolates distinct `subject_id` groups.
   - **Rear-view guardrail:** Never compares a rear-facing output exclusively against a frontal `face_identity` reference; if no `back_body` reference is supplied, rear hairstyle/garment match is explicitly downgraded to `uncertain`.
   - Computes lightweight CPU perceptual dHash + head/wardrobe color-band histogram similarity prior to VLM escalation. Never claims 3D reconstruction or exact physical measurements.
5. **Stage 5 — Efficient Video Inspection (`avatar_v2/video_inspector.py`)**:
   - Zero per-frame large VLM sweep. Samples bounded representative frames (`<= 16`, always including exact frame 0 and final frame via `ffmpeg -sseof`).
   - Runs fast CPU temporal sweeps (`260–811 ms`): luma flicker jumps, frame-to-frame L1 motion energy (static detection), 64-bit dHash jumps, 24-bin RGB color/background shifts, sharpness collapse, and `ffmpeg` freeze/black-frame detection.
   - Explicitly compares Wan `WanFirstLastFrameToVideo` initial frame vs `starter_image`, final frame vs `end_image`, **and** intermediate temporal continuity (never treating valid endpoints as proof intermediate frames are valid).
   - Escalates only `first`, `last`, `mid`, and top suspicious-interval frames (`<= 4` frames) to local VLM.
6. **Stage 6 — Targeted Semantic Local VLM + Verified VRAM Release (`avatar_v2/ollama_client.py`)**:
   - Evaluates against the **original user requirements** in `GenerationVerificationRecord`, not merely the compiled prompt.
   - Enforces observe-before-judge schema (`observed_evidence`, `observed_count`, `observed_visible_text` before `status`) + deterministic exact-count & exact-text cross-checks to defeat VLM sycophancy.
   - Solves the Sprint 6 asynchronous Ollama unload lag via `verify_ollama_vram_released()` polling loopback `/api/ps` and `nvidia-smi` until GPU memory is confirmed free (`100%` verified across all live GPU trials).
7. **Stage 7 — Structured Sprint 8 Correction Handoff (`build_correction_recommendations`)**:
   - Emits closed-enum `CorrectionRecommendation` actions (`revise_composition`, `targeted_edit`, `text_correction`, `select_subject_references`, `add_back_body_reference`, `add_side_reference`, `revise_ending_keyframe`, `inspect_temporal_interval`, `revise_framing`, `request_human_review`) carrying supporting evidence, artifact frame/interval, and `preserve_satisfied_requirements`.
   - Vision models cannot emit arbitrary commands, invent workflows, alter resource limits, or enable cloud billing. No autonomous unbounded retries are executed in Sprint 7.

---

## 2. Live Local Vision Model Inventory & Benchmark Results

All four already-installed `qwen3-vl` models were evaluated live against **16 ground-truth image cases (25 independently verified criteria)** combining **6 immutable Sprint 5 baseline outputs** (directly human-verified without modification: `adh_03_quantities`, `adh_04_spatial`, `adh_07_typography`, `adh_10_complex_nl`, `mv_01_starter_only` vs `seed_back`, `mv_03_starter_front_back` vs `seed_back`) and **10 deterministic synthetic ground-truth fixtures** (exact counts, wrong counts, missing object, wrong color, spatial pass/reversed, exact/misspelled/missing text, image prompt injection), plus **4 live video temporal cases** (`syn_video_smooth`, `syn_video_flicker` with matching endpoints, `syn_video_static`, and real `mv_06_wan_flf2v_turn`).

Raw immutable benchmark JSONs preserved in `~/.avatar_v2/benchmarks/`:
- `sprint7_visual_inspection_results.json` (4-model baseline sequential GPU sweep + 4-video CPU temporal sweep)
- `sprint7_visual_inspection_hybrid_gpu.json` (calibrated hybrid cross-check sequential GPU sweep for `4b` & `8b`)
- `sprint7_visual_inspection_cpu.json` (CPU-only `num_gpu=0` comparison across `2b`, `4b`, `8b`)

### 2.1 Overall Model Comparison (Sequential GPU vs CPU-Only)

| Model | Size | Mode | JSON Rel. | Mean Latency | VRAM Release Verified | TP | TN | FP (FAR) | FN (FRR) | Uncertain | Injection Warn |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| `qwen3-vl:2b-instruct` (pre-cross-check) | 1.76 GB | seq GPU | 93.8% | 5.42 s | 100% | 9 | 5 | 5 (50.0%) | 0 (0.0%) | 6 | Yes |
| `qwen3-vl:2b-instruct-q8_0` (pre-cross-check) | 2.45 GB | seq GPU | 100.0% | 5.39 s | 100% | 11 | 4 | 4 (50.0%) | 0 (0.0%) | 6 | Yes |
| `qwen3-vl:4b-instruct` (pre-cross-check) | 3.07 GB | seq GPU | 100.0% | 5.92 s | 100% | 13 | 7 | 5 (41.7%) | 0 (0.0%) | 0 | Yes |
| `qwen3-vl:8b-instruct` (pre-cross-check) | 5.72 GB | seq GPU | 100.0% | 13.01 s | 100% | 13 | 9 | 3 (25.0%) | 0 (0.0%) | 0 | Yes |
| **`qwen3-vl:4b-instruct` (hybrid cross-check)** | **3.07 GB** | **seq GPU** | **93.8%** | **6.45 s** | **100%** | **12** | **8** | **4 (33.3%)** | **0 (0.0%)** | **1** | **Yes** |
| **`qwen3-vl:8b-instruct` (hybrid cross-check)** | **5.72 GB** | **seq GPU** | **100.0%** | **11.57 s** | **100%** | **13** | **10** | **2 (16.7%)** | **0 (0.0%)** | **0** | **Yes** |
| `qwen3-vl:2b-instruct` (6-case subset) | 1.76 GB | CPU-only | 100.0% | 43.86 s | N/A (0 VRAM) | 2 | 2 | 3 (60.0%) | 0 (0.0%) | 2 | Yes |
| `qwen3-vl:4b-instruct` (6-case subset) | 3.07 GB | CPU-only | 66.7% | 73.04 s | N/A (0 VRAM) | 0 | 5 | 1 (16.7%) | 0 (0.0%) | 3 | Yes |
| `qwen3-vl:8b-instruct` (6-case subset) | 5.72 GB | CPU-only | 83.3% | 85.71 s | N/A (0 VRAM) | 1 | 4 | 2 (33.3%) | 0 (0.0%) | 2 | Yes |

### 2.2 Per-Category Breakdown — Selected Configuration (`qwen3-vl:8b-instruct` + Hybrid Cross-Check, Sequential GPU)

*(Reported per category without hiding weaknesses behind one overall accuracy number.)*

| Requirement Category | TP | TN | FP (False Accept) | FN (False Reject) | Uncertain | Precision | Recall | FAR | FRR | Reliability Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `typography` (real + synthetic) | 2 | 2 | 0 | 0 | 0 | 1.000 | 1.000 | 0.0% | 0.0% | **Reliable** (VLM transcription + deterministic exact-match cross-check) |
| `multiview_reference` (`mv_01` + `mv_03` vs `seed_back`) | 3 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 0.0% | 0.0% | **Moderately reliable with `8b` + rear guardrail** (`2b`/`4b` failed `mv_01`!) |
| `colors` | 0 | 1 | 0 | 0 | 0 | N/A | N/A | 0.0% | N/A | **Reliable** |
| `missing_or_unwanted` | 2 | 1 | 0 | 0 | 0 | 1.000 | 1.000 | 0.0% | 0.0% | **Reliable** |
| `complex_nl` (`adh_10`) | 2 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | N/A | 0.0% | **Reliable** |
| `prompt_injection` | 0 | 1 | 0 | 0 | 0 | N/A | N/A | 0.0% | N/A | **Reliable** (failed missing text + emitted injection warnings) |
| `object_quantities` (`adh_03` + synthetic) | 2 | 3 | 1 | 0 | 0 | 0.667 | 1.000 | 25.0% | 0.0% | **Partial** — caught `adh_03` 2-vs-3 mugs & synthetic counts; missed subtle open-vs-closed notebook cover |
| `spatial_relationships` (`adh_04` + synthetic) | 2 | 1 | 1 | 0 | 0 | 0.667 | 1.000 | 50.0% | 0.0% | **Weak on photographic left/right** — caught synthetic reversal, missed `adh_04` angled brass lamp vs green vase |

### 2.3 Video CPU Temporal Benchmark (`analyze_video_temporal_cpu`, `0 MB` VRAM)

| Video Case | Latency | Sampled Frames | Escalation Frames | Max Luma Jump | Max dHash Jump | Max Color Shift | Static? | Automated Finding |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `syn_video_smooth` | 296.9 ms | 13 | 4 | 0.0 | 19 | 0.003 | No | Pass (0 defects) |
| `syn_video_flicker_valid_endpoints` | 260.3 ms | 13 | 4 | **150.2** | **26** | **0.707** | No | **Fail:** caught midpoint flash (`0.73s–1.27s`) despite matching start/end frames |
| `syn_video_static` | 281.3 ms | 13 | 3 | 0.0 | 0 | 0.000 | **Yes** | **Fail:** caught frozen/static frame |
| `real_mv_06_wan_flf2v_turn` | 810.8 ms | 16 | 4 | 13.3 | **30** | **0.274** | No | **Caught real midpoint background shift** (`1.81s–2.17s`, sharpness drop `78.2` vs `160.0`) |

---

## 3. Model Selection & Resource-Aware End-to-End Scheduling Justification

- **Empirical model choice:** `qwen3-vl:8b-instruct` (already installed, 5.72 GB) is selected as the default semantic & multi-view visual inspector ONLY because live comparative testing proved `2b` (`50%` FAR, omitted criteria) and `4b` (`33.3%` FAR, failed real `mv_01` rear hairstyle comparison and synthetic spatial reversal) had unacceptable false-acceptance rates. `qwen3-vl:4b-instruct` remains the fast lightweight fallback (`6.45 s` GPU). No new models were downloaded.
- **Total workflow latency (CPU-only VLM vs Sequential GPU / Deferred Batch):**
  - On CPU (`num_gpu=0`), `qwen3-vl:8b` averages **`85.7 s`** per image (timing out at `120 s` on 2-image multi-view comparisons), and even `qwen3-vl:2b` takes **`43.9 s`** with `60%` FAR.
  - On Sequential GPU (`keep_alive=0` + `verify_ollama_vram_released()`), `qwen3-vl:8b` averages **`11.57 s`** with `100%` verified VRAM release (post-unload VRAM `778–890 MiB`, `ollama ps` empty).
  - Because evicting warm FLUX.2 9B adds `~20 s` on the next cold start, **immediate single-job GPU VLM (`11.6 s + 20 s = 31.6 s`) is still 2.7× faster than CPU-only `8b` (`85.7 s`)**, and **deferred batch GPU VLM** (evicting FLUX once after a 4-image batch: `4 × 11.6 s = 46 s` vs `343 s` CPU) is **7.4× faster** while completely preserving warm FLUX throughput during batch rendering.
  - Fast CPU deterministic + structural + video-temporal checks (`<0.05 s` image / `<0.85 s` video, `0 MB` VRAM) ALWAYS run first with zero impact on warm ComfyUI models.

---

## 4. Honestly Documented Limitations & Categories Requiring Human Review

Do **not** claim comprehensive automated verification for:
- **Photographic left/right spatial relationships** with complex/angled foreground objects (`adh_04_spatial`: `50%` FAR even on `qwen3-vl:8b`).
- **Fine object state distinctions** (e.g. open lined notebook vs closed light-covered notebook in `adh_03_quantities`).
- **Fine facial identity nuance, natural skin texture, five-finger hand anatomy, and lip-sync realism** — these still require human visual inspection (`human_review_required=True`).
- **Rear or side-profile verification without matching supplied references** — explicitly marked `uncertain` rather than falsely passed.

---

## 5. Verification Boundary: Simulated Tests vs Live Benchmarks

- **Simulated & deterministic regression suite (`tests/test_sprint7_visual_inspection.py`):** Tests all 20 Section 13 requirements (contract validation, count/text/spatial/missing-subject failures, reference-role selection, rear-view guardrail, multi-subject separation, video sampling & midpoint flicker/static detection, invalid VLM JSON, timeout, model unavailability, ResourcePlanner scheduling, async VRAM release polling, loopback offline enforcement, prompt-injection defense, sycophantic VLM cross-check, Sprint 8 correction recommendations, and frontend preservation).
- **Live GPU/CPU visual intelligence benchmarks (`scripts/eval_visual_inspection.py`):** 94 live Ollama Qwen3-VL image inspections (64 baseline GPU + 32 calibrated hybrid GPU + 18 CPU-only = **114 total live VLM calls**!) plus 4 real/synthetic CPU video temporal analyses against unmodified immutable Sprint 5 outputs.
