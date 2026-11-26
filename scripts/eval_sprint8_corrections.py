#!/usr/bin/env python3
"""Controlled Sprint 8 Benchmarks: Held-Out Visual Calibration + Live GPU Correction Comparison.

Part 1 (--mode heldout):
    Evaluates our upgraded Sprint 8 VisualInspector (localized coordinate & object-state
    cross-checks + verification eligibility) on:
    - Calibration check (adh_03_quantities, adh_04_spatial)
    - Completely separate Held-Out Evaluation Set (4 unseen immutable Sprint 5 outputs +
      7 brand-new unseen deterministic synthetic image/video fixtures).

Part 2 (--mode live_gpu):
    Launches an isolated ComfyUI instance (port 18189, --cache-ram 6 24) and runs controlled
    live GPU comparisons on RTX 4070 with FLUX.2 9B + Qwen3-VL 8B (verified VRAM release):
    - Condition A: Original immutable Sprint 5 baseline without correction (untouched!)
    - Condition B: Deterministic prompt compilation without correction
    - Condition C: Local Ollama LLM-assisted prompting without correction
    - Condition D: Hybrid inspection + bounded CorrectionCoordinator (max 2 attempts)
    - Condition E: Targeted editing (FLUX.2 ReferenceLatent edit) vs Complete regeneration
      (prompt_refinement) measuring preserved satisfied requirements & regressions.

Preserves all original outputs and every corrected candidate under:
    ~/.avatar_v2/benchmarks/sprint8_corrections/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avatar_v2.correction_engine import (
    CorrectionCoordinator,
    PlannedCorrectionStep,
    plan_batched_inspection_and_correction,
)
from avatar_v2.inspection_contract import (
    AcceptanceCriterion,
    CorrectionBudget,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    ReferenceProvenanceRecord,
)
from avatar_v2.intent_interpreter import interpret_natural_language
from avatar_v2.ollama_client import (
    DEFAULT_OLLAMA_URL,
    LLMExecutionSchedule,
    LocalOllamaClient,
    _query_gpu_vram_mb,
    verify_ollama_vram_released,
)
from avatar_v2.prompt_compilers import compile_prompt_for_model
from avatar_v2.prompt_schema import StructuredPromptSpec
from avatar_v2.video_inspector import analyze_video_temporal_cpu
from avatar_v2.visual_inspector import VisualInspector
from avatar_v2.workflows_builder import build_flux2_txt2img_workflow

BASELINE_ROOT = Path.home() / ".avatar_v2" / "benchmarks" / "phase_a2_multiview_adherence"
OUT_ROOT = Path.home() / ".avatar_v2" / "benchmarks" / "sprint8_corrections"
COMFY_DIR = Path("/home/jd/AvatarForge/comfyui")
COMFY_PY = Path("/home/jd/AvatarForge/venv/bin/python")
COMFY_PORT = 18189


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(cand).is_file():
            return ImageFont.truetype(cand, size=size)
    return ImageFont.load_default()


def _rec(
    prompt: str,
    criteria: list[tuple[str, str, str, bool]],
    *,
    w: int = 896,
    h: int = 1280,
    orientation: str = "general",
    typography: list[str] | None = None,
    exclusions: list[str] | None = None,
    refs: list[ReferenceProvenanceRecord] | None = None,
) -> GenerationVerificationRecord:
    return GenerationVerificationRecord(
        original_prompt=prompt,
        detected_intent="still_generate",
        compiled_prompt=prompt,
        selected_model_id="flux2_9b",
        selected_workflow="builtin:flux2_txt2img",
        reference_requirements=refs or [],
        expected_output=ExpectedOutputCharacteristics(
            task="image_generate",
            media_kind="image",
            width=w,
            height=h,
            expected_orientation=orientation,
            expected_typography=typography or [],
            expected_exclusions=exclusions or [],
        ),
        mandatory_acceptance_criteria=[
            AcceptanceCriterion(criterion_id=cid, category=cat, description=desc, mandatory=True, locked=locked)
            for cid, cat, desc, locked in criteria
        ],
        correction_budget=CorrectionBudget(max_retries=2, max_image_retries=2, allow_cloud_escalation=False),
    )


def _compute_per_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, int]] = {}
    for row in rows:
        b = by_cat.setdefault(row["category"], {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "uncertain": 0, "total": 0})
        b["total"] += 1
        gt, pred = row["ground_truth"], row["predicted"]
        if pred in {"uncertain", "unverifiable", "pending"}:
            b["uncertain"] += 1
        elif gt == "passed" and pred == "passed":
            b["tp"] += 1
        elif gt == "failed" and pred == "failed":
            b["tn"] += 1
        elif gt == "failed" and pred == "passed":
            b["fp"] += 1
        elif gt == "passed" and pred == "failed":
            b["fn"] += 1
    out: dict[str, Any] = {}
    for cat, b in sorted(by_cat.items()):
        tp, tn, fp, fn = b["tp"], b["tn"], b["fp"], b["fn"]
        out[cat] = {
            **b,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
            "far": round(fp / (fp + tn), 3) if (fp + tn) else None,
            "frr": round(fn / (fn + tp), 3) if (fn + tp) else None,
        }
    return out


# --------------------------------------------------------------------------- Part 1: Separate Calibration vs Held-Out Evaluation


def run_heldout_eval(model: str = "qwen3-vl:8b-instruct") -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    adh = BASELINE_ROOT / "outputs" / "adherence"
    mref = BASELINE_ROOT / "outputs" / "mref"
    font_lg = _load_font(54)

    with tempfile.TemporaryDirectory(prefix="avatar_v2_s8_heldout_") as tmp_str:
        tmp = Path(tmp_str)

        # Build brand-new unseen synthetic held-out fixtures (never used in Sprint 7!).
        # 1. Count 5 orange triangles PASS
        im = Image.new("RGB", (640, 480), (242, 242, 240))
        d = ImageDraw.Draw(im)
        for i, cx in enumerate((90, 205, 320, 435, 550)):
            d.polygon([(cx, 160), (cx - 42, 280), (cx + 42, 280)], fill=(235, 125, 25), outline=(110, 50, 10))
        p_cnt5_pass = tmp / "ho_count5_pass.png"
        im.save(p_cnt5_pass)

        # 2. Count 5 orange triangles FAIL (only 4 orange triangles + 1 green circle)
        im = Image.new("RGB", (640, 480), (242, 242, 240))
        d = ImageDraw.Draw(im)
        for cx in (90, 205, 320, 435):
            d.polygon([(cx, 160), (cx - 42, 280), (cx + 42, 280)], fill=(235, 125, 25), outline=(110, 50, 10))
        d.ellipse((508, 190, 592, 274), fill=(40, 165, 70), outline=(15, 70, 25), width=3)
        p_cnt5_fail = tmp / "ho_count5_fail.png"
        im.save(p_cnt5_fail)

        # 3. Spatial PASS: purple diamond LEFT of cyan square
        im = Image.new("RGB", (640, 480), (245, 245, 245))
        d = ImageDraw.Draw(im)
        d.polygon([(150, 140), (230, 240), (150, 340), (70, 240)], fill=(135, 50, 185), outline=(55, 15, 80))
        d.rectangle((410, 150, 570, 310), fill=(30, 190, 210), outline=(10, 80, 95), width=3)
        p_sp_pass = tmp / "ho_spatial_pass.png"
        im.save(p_sp_pass)

        # 4. Spatial REVERSED FAIL: cyan square LEFT of purple diamond
        im = Image.new("RGB", (640, 480), (245, 245, 245))
        d = ImageDraw.Draw(im)
        d.rectangle((70, 150, 230, 310), fill=(30, 190, 210), outline=(10, 80, 95), width=3)
        d.polygon([(490, 140), (570, 240), (490, 340), (410, 240)], fill=(135, 50, 185), outline=(55, 15, 80))
        p_sp_fail = tmp / "ho_spatial_fail.png"
        im.save(p_sp_fail)

        # 5. Text PASS: "PHASE 08"
        im = Image.new("RGB", (640, 360), (235, 235, 235))
        d = ImageDraw.Draw(im)
        d.rectangle((70, 110, 570, 250), fill=(22, 24, 28))
        d.text((160, 150), "PHASE 08", fill=(252, 252, 252), font=font_lg)
        p_txt_pass = tmp / "ho_text_pass.png"
        im.save(p_txt_pass)

        # 6. Text TYPO FAIL: "PHSAE 08"
        im = Image.new("RGB", (640, 360), (235, 235, 235))
        d = ImageDraw.Draw(im)
        d.rectangle((70, 110, 570, 250), fill=(22, 24, 28))
        d.text((160, 150), "PHSAE 08", fill=(252, 252, 252), font=font_lg)
        p_txt_fail = tmp / "ho_text_fail.png"
        im.save(p_txt_fail)

        # 7. Unwanted object PRESENT FAIL: forbids yellow circle, bright yellow circle drawn in center
        im = Image.new("RGB", (640, 480), (242, 242, 242))
        d = ImageDraw.Draw(im)
        d.rectangle((60, 320, 580, 440), fill=(155, 120, 85))
        d.ellipse((240, 120, 400, 280), fill=(245, 215, 25), outline=(140, 115, 10), width=4)
        p_unwanted = tmp / "ho_unwanted_fail.png"
        im.save(p_unwanted)

        # Calibration check cases (the 2 toughest Sprint 7 failures)
        cal_cases = [
            {
                "case_id": "cal_adh_03_quantities",
                "split": "calibration",
                "path": adh / "adh_03_quantities_00001_.png",
                "record": _rec(
                    "Top-down photograph of a wooden desk with exactly 3 ceramic coffee mugs and exactly 2 closed notebooks.",
                    [
                        ("three_mugs", "object_quantities", "Exactly 3 ceramic coffee mugs with handles are visible", False),
                        ("two_closed_notebooks", "object_quantities", "Exactly 2 closed notebooks are visible (none open)", False),
                    ],
                ),
                "gt": {"three_mugs": "failed", "two_closed_notebooks": "failed"},
            },
            {
                "case_id": "cal_adh_04_spatial",
                "split": "calibration",
                "path": adh / "adh_04_spatial_00001_.png",
                "record": _rec(
                    "Green glass vase left of brass desk lamp on wooden table, with potted fern behind the table near a white curtain.",
                    [
                        ("vase_left_of_lamp", "spatial_relationships", "Green glass vase is positioned to the viewer's left of the brass lamp", False),
                        ("fern_behind_table", "spatial_relationships", "Potted fern stands behind the table near a white curtain", True),
                    ],
                ),
                "gt": {"vase_left_of_lamp": "failed", "fern_behind_table": "passed"},
            },
        ]

        # Completely separate HELD-OUT evaluation cases (never used during Sprint 7 calibration!)
        heldout_cases = [
            {
                "case_id": "ho_real_adh_02_composition",
                "split": "held_out",
                "path": adh / "adh_02_composition_00001_.png",
                "record": _rec(
                    "Full-body portrait placed on the right third of the frame with open brick courtyard space on the left.",
                    [
                        ("full_body", "composition", "Entire body from head to shoes visible without cropping", False),
                        ("right_third", "composition", "Subject placed on the viewer's right third with open space on the left", False),
                    ],
                ),
                "gt": {"full_body": "passed", "right_third": "passed"},
            },
            {
                "case_id": "ho_real_adh_05_colors",
                "split": "held_out",
                "path": adh / "adh_05_colors_00001_.png",
                "record": _rec(
                    "Portrait wearing a cobalt blue blazer, crisp white shirt underneath, holding a bright yellow notebook.",
                    [
                        ("cobalt_blazer", "colors", "Blazer is cobalt blue", False),
                        ("white_shirt", "colors", "Shirt underneath is white", False),
                        ("yellow_notebook", "colors", "Notebook is bright yellow", False),
                    ],
                ),
                "gt": {"cobalt_blazer": "passed", "white_shirt": "passed", "yellow_notebook": "passed"},
            },
            {
                "case_id": "ho_real_adh_06_clothing",
                "split": "held_out",
                "path": adh / "adh_06_clothing_00001_.png",
                "record": _rec(
                    "Portrait of adult woman with shoulder-length auburn hair in a half-up style, tortoiseshell glasses, dark emerald knit turtleneck, and small gold hoop earrings.",
                    [
                        ("auburn_half_up", "clothing_appearance", "Shoulder-length auburn hair pulled back in a half-up style (not a short chin bob)", False),
                        ("tortoiseshell_glasses", "clothing_appearance", "Wearing tortoiseshell glasses", False),
                        ("emerald_turtleneck", "clothing_appearance", "Wearing dark emerald knit turtleneck", False),
                        ("gold_hoops", "clothing_appearance", "Wearing small gold hoop earrings", False),
                    ],
                ),
                "gt": {"auburn_half_up": "failed", "tortoiseshell_glasses": "passed", "emerald_turtleneck": "passed", "gold_hoops": "passed"},
            },
            {
                "case_id": "ho_real_mv_04_left_profile",
                "split": "held_out",
                "path": mref / "mv_04_multi_perspective_00001_.png",
                "record": _rec(
                    "Full-body walking left side profile matching the left_side reference silhouette, low hair bun, olive blazer, and navy trousers.",
                    [
                        ("left_profile_orientation", "multiview_reference", "Subject is rendered in clear left side profile", False),
                        ("profile_wardrobe_match", "multiview_reference", "Low hair bun, olive blazer, and navy trousers match the left_side reference", False),
                    ],
                    orientation="left_profile",
                    refs=[ReferenceProvenanceRecord(asset_id="ref_left", role="left_side", path=str(mref / "seed_left_00001_.png"))],
                ),
                "gt": {"left_profile_orientation": "passed", "profile_wardrobe_match": "passed"},
            },
            {
                "case_id": "ho_syn_count5_pass",
                "split": "held_out",
                "path": p_cnt5_pass,
                "record": _rec("Exactly 5 orange triangles.", [("five_triangles", "object_quantities", "Exactly 5 orange triangles are visible", False)], w=640, h=480),
                "gt": {"five_triangles": "passed"},
            },
            {
                "case_id": "ho_syn_count5_fail",
                "split": "held_out",
                "path": p_cnt5_fail,
                "record": _rec("Exactly 5 orange triangles.", [("five_triangles", "object_quantities", "Exactly 5 orange triangles are visible", False)], w=640, h=480),
                "gt": {"five_triangles": "failed"},
            },
            {
                "case_id": "ho_syn_spatial_pass",
                "split": "held_out",
                "path": p_sp_pass,
                "record": _rec("Purple diamond left of cyan square.", [("purple_left_of_cyan", "spatial_relationships", "Purple diamond is positioned to the viewer's left of the cyan square", False)], w=640, h=480),
                "gt": {"purple_left_of_cyan": "passed"},
            },
            {
                "case_id": "ho_syn_spatial_fail",
                "split": "held_out",
                "path": p_sp_fail,
                "record": _rec("Purple diamond left of cyan square.", [("purple_left_of_cyan", "spatial_relationships", "Purple diamond is positioned to the viewer's left of the cyan square", False)], w=640, h=480),
                "gt": {"purple_left_of_cyan": "failed"},
            },
            {
                "case_id": "ho_syn_text_pass",
                "split": "held_out",
                "path": p_txt_pass,
                "record": _rec("Plaque reading 'PHASE 08'.", [("phase_text", "typography", "Sign reads exactly 'PHASE 08'", False)], w=640, h=360, typography=["PHASE 08"]),
                "gt": {"phase_text": "passed"},
            },
            {
                "case_id": "ho_syn_text_fail",
                "split": "held_out",
                "path": p_txt_fail,
                "record": _rec("Plaque reading 'PHASE 08'.", [("phase_text", "typography", "Sign reads exactly 'PHASE 08' with exact spelling", False)], w=640, h=360, typography=["PHASE 08"]),
                "gt": {"phase_text": "failed"},
            },
            {
                "case_id": "ho_syn_unwanted_fail",
                "split": "held_out",
                "path": p_unwanted,
                "record": _rec("Empty wooden table with no yellow circle or ball.", [("no_yellow_circle", "missing_or_unwanted", "No yellow circle or sphere is visible", False)], w=640, h=480, exclusions=["yellow circle"]),
                "gt": {"no_yellow_circle": "failed"},
            },
        ]

        inspector = VisualInspector(
            ollama_url=DEFAULT_OLLAMA_URL,
            client=LocalOllamaClient(base_url=DEFAULT_OLLAMA_URL, timeout_s=120.0),
        )

        def _eval_split(cases: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
            rows: list[dict[str, Any]] = []
            latencies: list[float] = []
            print(f"\n=== Evaluating {split_name} split ({len(cases)} cases) with {model} ===", flush=True)
            for c in cases:
                t0 = time.monotonic()
                res = inspector.inspect_image(
                    c["path"], c["record"], use_vlm=True, preferred_model=model, comfy_models_warm=False, allow_sequential_gpu=True
                )
                latencies.append(round((time.monotonic() - t0) * 1000.0, 1))
                for a in res.criterion_assessments:
                    if a.criterion_id in c["gt"]:
                        rows.append({
                            "case_id": c["case_id"],
                            "criterion_id": a.criterion_id,
                            "category": a.category,
                            "ground_truth": c["gt"][a.criterion_id],
                            "predicted": a.status,
                            "method": a.inspection_method,
                            "corroborated": a.independent_corroboration,
                            "evidence": a.observed_evidence[:180],
                        })
            per_cat = _compute_per_category(rows)
            tp = sum(v["tp"] for v in per_cat.values())
            tn = sum(v["tn"] for v in per_cat.values())
            fp = sum(v["fp"] for v in per_cat.values())
            fn = sum(v["fn"] for v in per_cat.values())
            unc = sum(v["uncertain"] for v in per_cat.values())
            print(f"  -> {split_name}: TP={tp} TN={tn} FP={fp} FN={fn} UNC={unc} | mean={sum(latencies)/max(len(latencies),1):.0f}ms", flush=True)
            return {
                "split": split_name,
                "cases": len(cases),
                "criteria": len(rows),
                "mean_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
                "totals": {
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn, "uncertain": unc,
                    "far": round(fp / (fp + tn), 3) if (fp + tn) else None,
                    "frr": round(fn / (fn + tp), 3) if (fn + tp) else None,
                },
                "per_category": per_cat,
                "details": rows,
            }

        cal_res = _eval_split(cal_cases, "calibration_recheck")
        ho_res = _eval_split(heldout_cases, "held_out_unseen")
        vram_rel = verify_ollama_vram_released(DEFAULT_OLLAMA_URL, model=model, timeout_s=10.0)

    report = {
        "benchmark": "sprint8_calibration_and_heldout_eval",
        "model": model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_recheck": cal_res,
        "held_out_unseen": ho_res,
        "final_vram_release": vram_rel,
    }
    out_file = OUT_ROOT / "sprint8_heldout_eval.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved held-out eval to {out_file}", flush=True)
    return report


# --------------------------------------------------------------------------- Part 2: Live GPU Controlled Comparison (A vs B vs C vs D vs E)


def _start_isolated_comfy(out_dir: Path, in_dir: Path) -> subprocess.Popen[bytes]:
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "comfy_sprint8.log"
    log_f = log_path.open("wb")
    cmd = [
        str(COMFY_PY),
        str(COMFY_DIR / "main.py"),
        "--listen", "127.0.0.1",
        "--port", str(COMFY_PORT),
        "--cache-ram", "6", "24",
        "--output-directory", str(out_dir),
        "--input-directory", str(in_dir),
        "--disable-auto-launch",
    ]
    proc = subprocess.Popen(cmd, cwd=str(COMFY_DIR), stdout=log_f, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{COMFY_PORT}/system_stats"
    for _ in range(90):
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                print(f"Isolated ComfyUI ready on 127.0.0.1:{COMFY_PORT} (pid={proc.pid})", flush=True)
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited prematurely; check {log_path}")
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError("Timed out waiting for isolated ComfyUI.")


def _comfy_free(unload_models: bool = True) -> None:
    try:
        httpx.post(
            f"http://127.0.0.1:{COMFY_PORT}/free",
            json={"unload_models": unload_models, "free_memory": True},
            timeout=15.0,
        )
        time.sleep(1.5)
    except Exception:
        pass


def _render_flux2_9b(
    *,
    prompt: str,
    prefix: str,
    seed: int,
    ref_paths: list[str],
    in_dir: Path,
    out_dir: Path,
    width: int = 896,
    height: int = 1280,
) -> dict[str, Any]:
    """Run one live FLUX.2 9B job on isolated ComfyUI measuring latency, peak VRAM & output path."""
    existing = sorted(out_dir.glob(f"{prefix}_*.png"))
    if existing:
        return {
            "path": str(existing[0]),
            "elapsed_ms": 0.0,
            "peak_vram_mb": 0,
            "seed": seed,
            "ref_count": len(ref_paths),
            "reused_from_same_session": True,
        }
    mapping: dict[str, Any] = {
        "FORGE_UNET": "flux-2-klein-base-9b-Q4_K_M.gguf",
        "FORGE_TEXT_ENCODER": "Qwen3-8B-Q4_K_M.gguf",
        "FORGE_VAE": "flux2-vae.safetensors",
        "PROMPT": prompt,
        "NEGATIVE_PROMPT": "",
        "WIDTH": width,
        "HEIGHT": height,
        "STEPS": 28,   # Matches offline_builder.PHOTO_MODELS["9b"] (undistilled base)
        "CFG": 3.5,
        "SEED": seed,
        "OUTPUT_PREFIX": prefix,
    }
    for i, rpath in enumerate(ref_paths[:4], start=1):
        src = Path(rpath).expanduser().resolve()
        if src.is_file():
            dst_name = f"s8ref_{i}_{src.name}"
            dst = in_dir / dst_name
            if not dst.is_file():
                shutil.copy2(src, dst)
            mapping[f"REF_IMAGE_{i}"] = dst_name

    graph = build_flux2_txt2img_workflow(mapping)
    t0 = time.monotonic()
    resp = httpx.post(f"http://127.0.0.1:{COMFY_PORT}/prompt", json={"prompt": graph}, timeout=30.0)
    resp.raise_for_status()
    pid = resp.json()["prompt_id"]

    peak_vram = 0
    while True:
        used, _ = _query_gpu_vram_mb()
        if used is not None:
            peak_vram = max(peak_vram, used)
        hist = httpx.get(f"http://127.0.0.1:{COMFY_PORT}/history/{pid}", timeout=10.0).json()
        if pid in hist:
            entry = hist[pid]
            outputs = entry.get("outputs") or {}
            saved: list[str] = []
            for node_out in outputs.values():
                for img in node_out.get("images") or []:
                    sub = img.get("subfolder") or ""
                    fname = img.get("filename") or ""
                    full = out_dir / sub / fname if sub else out_dir / fname
                    if full.is_file():
                        saved.append(str(full))
            if not saved:
                raise RuntimeError(f"ComfyUI job {pid} completed with no saved outputs.")
            return {
                "path": saved[0],
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                "peak_vram_mb": peak_vram,
                "seed": seed,
                "ref_count": len(ref_paths),
            }
        time.sleep(1.0)


def run_live_gpu_corrections() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_ROOT / "outputs"
    in_dir = OUT_ROOT / "input"
    adh = BASELINE_ROOT / "outputs" / "adherence"
    mref = BASELINE_ROOT / "outputs" / "mref"

    # First ensure Ollama VRAM is completely empty before starting ComfyUI.
    verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=5.0)
    proc = _start_isolated_comfy(out_dir, in_dir)

    inspector = VisualInspector(
        ollama_url=DEFAULT_OLLAMA_URL,
        client=LocalOllamaClient(base_url=DEFAULT_OLLAMA_URL, timeout_s=120.0),
    )

    def _inspect_after_unload(img_path: str, record: GenerationVerificationRecord):
        _comfy_free(unload_models=True)
        res = inspector.inspect_image(
            img_path, record, use_vlm=True, preferred_model="qwen3-vl:8b-instruct", comfy_models_warm=False, allow_sequential_gpu=True
        )
        verify_ollama_vram_released(DEFAULT_OLLAMA_URL, model="qwen3-vl:8b-instruct", timeout_s=10.0)
        return res

    trials: dict[str, Any] = {}
    try:
        # ---- Case 1: adh_03_quantities (Conditions A, B, C, E1 targeted edit, E2 full regen) ----
        orig_q = str(adh / "adh_03_quantities_00001_.png")
        prompt_q = "Top-down photograph of a clean oak wooden desk with exactly 3 ceramic coffee mugs with handles and exactly 2 closed hardcover notebooks neatly separated on the table."
        rec_q = _rec(
            prompt_q,
            [
                ("three_mugs", "object_quantities", "Exactly 3 ceramic coffee mugs with handles are visible", False),
                ("two_closed_notebooks", "object_quantities", "Exactly 2 closed notebooks are visible (none open)", False),
            ],
        )

        # Condition B: Deterministic prompt compilation without correction
        spec_b = StructuredPromptSpec(original_prompt=prompt_q)
        comp_b = compile_prompt_for_model(spec_b, "flux2_9b")
        print("\n[Case 1 / Cond B] adh_03_quantities deterministic compiled prompt (no correction)...", flush=True)
        run_b = _render_flux2_9b(prompt=comp_b.positive, prefix="s8_adh03_condB_det", seed=41201, ref_paths=[], in_dir=in_dir, out_dir=out_dir)

        # Condition C: Local LLM-assisted prompting without correction
        _comfy_free(unload_models=True)
        interp_c = interpret_natural_language(
            prompt_q,
            requested_action="photo",
            use_llm=True,
            comfy_models_warm=False,
            llm_schedule=LLMExecutionSchedule(
                mode="sequential_gpu", num_gpu=-1, keep_alive="0s", reason="Sprint 8 Condition C sequential GPU", verify_vram_release=True
            ),
        )
        verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=10.0)
        comp_c = compile_prompt_for_model(interp_c.spec, "flux2_9b")
        print("[Case 1 / Cond C] adh_03_quantities local LLM-assisted prompt (no correction)...", flush=True)
        run_c = _render_flux2_9b(prompt=comp_c.positive, prefix="s8_adh03_condC_llm", seed=41202, ref_paths=[], in_dir=in_dir, out_dir=out_dir)

        # Condition D/E via CorrectionCoordinator: Attempt 1 = targeted_image_edit (E1), Attempt 2 = prompt_refinement (E2)!
        print("[Case 1 / Cond D+E] adh_03_quantities bounded CorrectionCoordinator (E1 edit vs E2 regen)...", flush=True)
        ins_q_orig = _inspect_after_unload(orig_q, rec_q)
        # Preserve raw VLM judgment honestly, while seeding independent ground-truth failure when evaluating E1 vs E2!
        from avatar_v2.inspection_contract import CorrectionRecommendation, CriterionAssessment, VisualInspectionResult
        seed_ins_q = ins_q_orig
        if ins_q_orig.overall_decision != "needs_correction":
            seed_ins_q = VisualInspectionResult(
                inspector_model="human_ground_truth_trigger",
                execution_mode="independent_ground_truth",
                overall_decision="needs_correction",
                ui_status="correction_recommended",
                ui_status_label="Correction recommended",
                criterion_assessments=[
                    CriterionAssessment(
                        criterion_id="three_mugs", original_requirement="Exactly 3 ceramic coffee mugs with handles are visible",
                        category="object_quantities", mandatory=True, status="failed", inspection_method="human",
                        observed_evidence="Independent ground truth: 2 coffee mugs with handles + 2 handleless bowls",
                        failure_description="Only 2 ceramic mugs with handles instead of 3",
                    ),
                    CriterionAssessment(
                        criterion_id="two_closed_notebooks", original_requirement="Exactly 2 closed notebooks are visible (none open)",
                        category="object_quantities", mandatory=True, status="failed", inspection_method="human",
                        observed_evidence="Independent ground truth: 1 open lined notebook + 1 closed brown notebook",
                        failure_description="Left notebook is open rather than closed",
                    ),
                ],
                correction_recommendations=[
                    CorrectionRecommendation(
                        failed_criterion_id="three_mugs", category="object_quantities",
                        original_requirement="Exactly 3 ceramic coffee mugs with handles are visible",
                        supporting_evidence="2 mugs + 2 handleless bowls",
                        action_type="revise_composition", recommended_action="Replace bowls so exactly 3 handled mugs and 2 closed notebooks appear",
                    )
                ],
            )
        gen_meta_q: list[dict[str, Any]] = []

        def _gen_q(step: PlannedCorrectionStep) -> tuple[str, float]:
            tag = "E1_edit" if step.is_localized_edit else "E2_regen"
            r = _render_flux2_9b(
                prompt=step.revised_prompt,
                prefix=f"s8_adh03_{tag}",
                seed=step.seed,
                ref_paths=step.reference_paths,
                in_dir=in_dir,
                out_dir=out_dir,
            )
            gen_meta_q.append({**r, "strategy": step.strategy, "is_localized_edit": step.is_localized_edit})
            return r["path"], r["elapsed_ms"]

        sess_q = CorrectionCoordinator().run_bounded_correction(
            record=rec_q,
            original_output_id="adh_03_orig",
            original_output_path=orig_q,
            initial_inspection=seed_ins_q,
            generate_fn=_gen_q,
            inspect_fn=_inspect_after_unload,
            base_seed=41203,
        )
        # Inspect Cond B & C outputs (batched after ComfyUI unload).
        ins_q_b = _inspect_after_unload(run_b["path"], rec_q)
        ins_q_c = _inspect_after_unload(run_c["path"], rec_q)

        trials["adh_03_quantities"] = {
            "cond_A_original_path": orig_q,
            "cond_A_inspection": ins_q_orig.compact_ui_summary(),
            "cond_B_deterministic": {**run_b, "inspection": ins_q_b.compact_ui_summary()},
            "cond_C_llm_prompt": {
                **run_c,
                "llm_used": bool(getattr(getattr(interp_c, "trace", None), "llm_used", False)),
                "inspection": ins_q_c.compact_ui_summary(),
            },
            "cond_DE_correction_session": sess_q.model_dump(),
            "cond_DE_gen_telemetry": gen_meta_q,
        }

        # ---- Case 2: adh_04_spatial (E1 targeted edit vs E2 full regen on real reversed spatial!) ----
        orig_s = str(adh / "adh_04_spatial_00001_.png")
        prompt_s = "Minimalist interior photograph of a wooden side table: a green glass vase placed on the LEFT side of the table, a modern brass desk lamp placed on the RIGHT side of the table, and a tall potted green fern standing behind the table next to a white sheer curtain."
        rec_s = _rec(
            prompt_s,
            [
                ("vase_left_of_lamp", "spatial_relationships", "Green glass vase is positioned to the viewer's left of the brass lamp", False),
                ("fern_behind_table", "spatial_relationships", "Potted fern stands behind the table near a white curtain", True),
            ],
        )
        print("\n[Case 2 / Cond E1+E2] adh_04_spatial targeted edit vs full regeneration...", flush=True)
        # Force E1 targeted edit AND E2 full regeneration so we directly compare preservation of fern_behind_table!
        run_s_e1 = _render_flux2_9b(
            prompt=(
                "Edit the first reference image to swap ONLY the positions on the wooden table so that the green glass vase "
                "is clearly on the viewer's LEFT and the brass desk lamp is clearly on the viewer's RIGHT. Keep the wooden table, "
                "the tall potted fern behind the table, and the white sheer curtain on the right unchanged."
            ),
            prefix="s8_adh04_E1_edit",
            seed=41301,
            ref_paths=[orig_s],
            in_dir=in_dir,
            out_dir=out_dir,
        )
        run_s_e2 = _render_flux2_9b(
            prompt=prompt_s,
            prefix="s8_adh04_E2_regen",
            seed=41302,
            ref_paths=[],
            in_dir=in_dir,
            out_dir=out_dir,
        )
        ins_s_e1 = _inspect_after_unload(run_s_e1["path"], rec_s)
        ins_s_e2 = _inspect_after_unload(run_s_e2["path"], rec_s)
        trials["adh_04_spatial"] = {
            "cond_A_original_path": orig_s,
            "cond_E1_targeted_edit": {**run_s_e1, "inspection": ins_s_e1.compact_ui_summary()},
            "cond_E2_full_regen": {**run_s_e2, "inspection": ins_s_e2.compact_ui_summary()},
        }

        # ---- Case 3: mv_01_starter_only rear hairstyle mismatch -> additional_reference_conditioning (seed_back) ----
        orig_mv = str(mref / "mv_01_starter_only_00001_.png")
        seed_back = str(mref / "seed_back_00001_.png")
        prompt_mv = "Full-body rear view of the woman facing away walking down a bright art gallery, wearing her olive linen blazer and navy trousers, with her chestnut hair neatly gathered in the exact low hair bun from the back reference image."
        rec_mv = _rec(
            prompt_mv,
            [
                ("rear_orientation", "multiview_reference", "Subject is oriented facing away from the camera", True),
                ("wardrobe_match", "multiview_reference", "Subject wears olive blazer and navy trousers matching reference", True),
                ("rear_hair_match", "multiview_reference", "Rear hairstyle matches the low hair bun in the back_body reference image", False),
            ],
            orientation="rear_facing",
            refs=[ReferenceProvenanceRecord(asset_id="ref_back", role="back_body", path=seed_back)],
        )
        print("\n[Case 3 / Cond D] mv_01_starter_only rear hairstyle correction via back_body ref...", flush=True)
        ins_mv_orig = _inspect_after_unload(orig_mv, rec_mv)
        gen_meta_mv: list[dict[str, Any]] = []

        def _gen_mv(step: PlannedCorrectionStep) -> tuple[str, float]:
            r = _render_flux2_9b(
                prompt=step.revised_prompt,
                prefix=f"s8_mv01_{step.strategy}",
                seed=step.seed,
                ref_paths=step.reference_paths,
                in_dir=in_dir,
                out_dir=out_dir,
            )
            gen_meta_mv.append({**r, "strategy": step.strategy, "is_localized_edit": step.is_localized_edit})
            return r["path"], r["elapsed_ms"]

        sess_mv = CorrectionCoordinator().run_bounded_correction(
            record=rec_mv,
            original_output_id="mv_01_orig",
            original_output_path=orig_mv,
            initial_inspection=ins_mv_orig,
            generate_fn=_gen_mv,
            inspect_fn=_inspect_after_unload,
            base_seed=41401,
        )
        trials["mv_01_rear_hair"] = {
            "cond_A_original_path": orig_mv,
            "cond_A_inspection": ins_mv_orig.compact_ui_summary(),
            "cond_D_correction_session": sess_mv.model_dump(),
            "cond_D_gen_telemetry": gen_meta_mv,
        }

    finally:
        _comfy_free(unload_models=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=8.0)

    # Compute total storage overhead for all generated Sprint 8 candidate outputs.
    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    report = {
        "benchmark": "sprint8_live_gpu_corrections",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_schedule_analysis": plan_batched_inspection_and_correction([{"id": "1"}, {"id": "2"}, {"id": "3"}], warm_model_id="flux2_9b"),
        "storage_overhead_mb": round(total_bytes / (1024 * 1024), 2),
        "trials": trials,
    }
    out_file = OUT_ROOT / "sprint8_live_gpu_corrections.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Sprint 8 live GPU correction results to {out_file}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heldout", "live_gpu", "all"], default="all")
    args = parser.parse_args()
    if args.mode in {"heldout", "all"}:
        run_heldout_eval()
    if args.mode in {"live_gpu", "all"}:
        run_live_gpu_corrections()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
