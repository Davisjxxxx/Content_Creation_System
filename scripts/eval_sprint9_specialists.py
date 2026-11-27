#!/usr/bin/env python3
"""Controlled Sprint 9 Live Benchmarks (Sections 5, 6, 7, 9, 14) — ZERO unauthorized downloads.

Modes:
  --mode blind_grid   : Live Qwen3-VL 8B blind 2D grid localization on photographic count/state/spatial cases
                        (compares directly against Sprint 8 single-pass VLM).
  --mode masked_edit  : Live FLUX.2 9B genuine masked localized edit (SetLatentNoiseMask + DifferentialDiffusion
                        + ImageCompositeMasked) measuring pixel preservation outside the edit mask.
  --mode wan_video    : Live Wan 2.2 A14B controlled comparison on identical starter frame:
                        Lightning 4-step (CFG 1.0) vs Lightning 6-step (CFG 1.0) vs Base Standard 10-step (CFG 3.5).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avatar_v2.inspection_contract import AcceptanceCriterion
from avatar_v2.ollama_client import (
    DEFAULT_OLLAMA_URL,
    _query_gpu_vram_mb,
    verify_ollama_vram_released,
)
from avatar_v2.specialist_vision import (
    generate_region_mask,
    measure_outside_mask_preservation,
    reconcile_criterion_with_detections,
    run_blind_object_localization,
)
from avatar_v2.video_inspector import analyze_video_temporal_cpu
from avatar_v2.workflows_builder import build_flux2_masked_edit_workflow, with_tiled_vae, without_wan_lightning

BASELINE_ROOT = Path.home() / ".avatar_v2" / "benchmarks" / "phase_a2_multiview_adherence"
S8_ROOT = Path.home() / ".avatar_v2" / "benchmarks" / "sprint8_corrections" / "outputs"
OUT_ROOT = Path.home() / ".avatar_v2" / "benchmarks" / "sprint9_specialists"
COMFY_DIR = Path("/home/jd/AvatarForge/comfyui")
COMFY_PY = Path("/home/jd/AvatarForge/venv/bin/python")
COMFY_PORT = 18189


def _start_isolated_comfy(out_dir: Path, in_dir: Path) -> subprocess.Popen[bytes]:
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "comfy_sprint9.log"
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
    for _ in range(90):
        try:
            if httpx.get(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=2.0).status_code == 200:
                print(f"Isolated ComfyUI ready on 127.0.0.1:{COMFY_PORT} (pid={proc.pid})", flush=True)
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early; check {log_path}")
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError("Timed out waiting for isolated ComfyUI.")


def _comfy_free() -> None:
    try:
        httpx.post(
            f"http://127.0.0.1:{COMFY_PORT}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=15.0,
        )
        time.sleep(1.5)
    except Exception:
        pass


def _run_graph(graph: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    resp = httpx.post(f"http://127.0.0.1:{COMFY_PORT}/prompt", json={"prompt": graph}, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"ComfyUI /prompt error {resp.status_code}: {resp.text[:500]}")
    pid = resp.json()["prompt_id"]
    peak_vram = 0
    while True:
        used, _ = _query_gpu_vram_mb()
        if used is not None:
            peak_vram = max(peak_vram, used)
        hist = httpx.get(f"http://127.0.0.1:{COMFY_PORT}/history/{pid}", timeout=10.0).json()
        if pid in hist:
            outputs = hist[pid].get("outputs") or {}
            files: list[str] = []
            for node_out in outputs.values():
                for key in ("images", "videos", "gifs"):
                    for entry in node_out.get(key) or []:
                        sub = entry.get("subfolder") or ""
                        fname = entry.get("filename") or ""
                        full = out_dir / sub / fname if sub else out_dir / fname
                        if full.is_file():
                            files.append(str(full))
            if not files:
                raise RuntimeError(f"ComfyUI job {pid} produced no output files: {outputs}")
            return {
                "path": files[0],
                "all_outputs": files,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "peak_vram_mb": peak_vram,
            }
        time.sleep(1.0)


# --------------------------------------------------------------------------- Mode 1: Blind 2D Grid Localization


def run_blind_grid_eval() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    adh = BASELINE_ROOT / "outputs" / "adherence"
    cases = [
        {
            "case_id": "photo_adh03_orig",
            "split": "calibration_photo",
            "path": adh / "adh_03_quantities_00001_.png",
            "targets": ["coffee mug with handle", "handleless bowl", "closed notebook", "open notebook"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="three_mugs", category="object_quantities", description="Exactly 3 ceramic coffee mugs with handles are visible"), "failed"),
                (AcceptanceCriterion(criterion_id="two_closed_notebooks", category="object_quantities", description="Exactly 2 closed notebooks are visible (none open)"), "failed"),
            ],
        },
        {
            "case_id": "photo_adh03_s8_E2_regen",
            "split": "heldout_photo",
            "path": S8_ROOT / "s8_adh03_E2_regen_00001_.png",
            "targets": ["coffee mug with handle", "handleless bowl", "closed notebook", "open notebook"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="three_mugs", category="object_quantities", description="Exactly 3 ceramic coffee mugs with handles are visible"), "passed"),
                (AcceptanceCriterion(criterion_id="two_closed_notebooks", category="object_quantities", description="Exactly 2 closed notebooks are visible (none open)"), "passed"),
            ],
        },
        {
            "case_id": "photo_adh03_s8_condB",
            "split": "heldout_photo",
            "path": S8_ROOT / "s8_adh03_condB_det_00001_.png",
            "targets": ["coffee mug with handle", "closed notebook"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="three_mugs", category="object_quantities", description="Exactly 3 ceramic coffee mugs with handles are visible"), "failed"),
                (AcceptanceCriterion(criterion_id="two_closed_notebooks", category="object_quantities", description="Exactly 2 closed notebooks are visible (none open)"), "failed"),
            ],
        },
        {
            "case_id": "photo_adh03_s8_condC",
            "split": "heldout_photo",
            "path": S8_ROOT / "s8_adh03_condC_llm_00001_.png",
            "targets": ["coffee mug with handle", "closed notebook"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="three_mugs", category="object_quantities", description="Exactly 3 ceramic coffee mugs with handles are visible"), "failed"),
                (AcceptanceCriterion(criterion_id="two_closed_notebooks", category="object_quantities", description="Exactly 2 closed notebooks are visible (none open)"), "passed"),
            ],
        },
        {
            "case_id": "photo_adh04_orig",
            "split": "calibration_photo",
            "path": adh / "adh_04_spatial_00001_.png",
            "targets": ["green glass vase", "brass lamp", "potted fern"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="vase_left_of_lamp", category="spatial_relationships", description="Green glass vase is positioned to the viewer's left of the brass lamp"), "failed"),
            ],
        },
        {
            "case_id": "photo_adh04_s8_E1_edit",
            "split": "heldout_photo",
            "path": S8_ROOT / "s8_adh04_E1_edit_00001_.png",
            "targets": ["green glass vase", "brass lamp", "potted fern"],
            "criteria": [
                (AcceptanceCriterion(criterion_id="vase_left_of_lamp", category="spatial_relationships", description="Green glass vase is positioned to the viewer's left of the brass lamp"), "passed"),
            ],
        },
    ]

    results: list[dict[str, Any]] = []
    tp = tn = fp = fn = unc = 0
    print("\n=== Sprint 9 Blind 2D Grid Localization (6 photographic cases, 10 criteria) ===", flush=True)
    for c in cases:
        if not c["path"].is_file():
            continue
        loc = run_blind_object_localization(c["path"], target_categories=c["targets"], use_grid_overlay=True)
        verdicts: list[dict[str, Any]] = []
        for crit, gt in c["criteria"]:
            rec = reconcile_criterion_with_detections(crit, loc.get("objects") or [])
            pred = rec["status"]
            if pred == "uncertain":
                unc += 1
            elif gt == "passed" and pred == "passed":
                tp += 1
            elif gt == "failed" and pred == "failed":
                tn += 1
            elif gt == "failed" and pred == "passed":
                fp += 1
            elif gt == "passed" and pred == "failed":
                fn += 1
            verdicts.append({"criterion_id": crit.criterion_id, "category": crit.category, "gt": gt, "pred": pred, "evidence": rec["evidence"]})
            print(f"  [{ 'OK' if pred == gt else ('UNC' if pred == 'uncertain' else 'MISS') }] {c['case_id']}/{crit.criterion_id}: gt={gt} pred={pred} | {rec['evidence']}", flush=True)
        results.append({
            "case_id": c["case_id"],
            "split": c["split"],
            "latency_ms": loc.get("latency_ms"),
            "object_count": len(loc.get("objects") or []),
            "objects": loc.get("objects") or [],
            "verdicts": verdicts,
        })

    vram_rel = verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=10.0)
    summary = {
        "benchmark": "sprint9_blind_grid_localization",
        "totals": {
            "tp": tp, "tn": tn, "fp_far": fp, "fn_frr": fn, "uncertain": unc,
            "far": round(fp / (fp + tn), 3) if (fp + tn) else None,
            "frr": round(fn / (fn + tp), 3) if (fn + tp) else None,
        },
        "vram_release": vram_rel,
        "cases": results,
    }
    out_p = OUT_ROOT / "sprint9_blind_grid_results.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Blind grid totals: {summary['totals']} -> saved to {out_p}", flush=True)
    return summary


# --------------------------------------------------------------------------- Mode 2: Live Masked Localized Editing


def run_masked_edit_eval() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_ROOT / "outputs"
    in_dir = OUT_ROOT / "input"
    adh = BASELINE_ROOT / "outputs" / "adherence"
    verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=5.0)
    proc = _start_isolated_comfy(out_dir, in_dir)

    trials: list[dict[str, Any]] = []
    try:
        # Test 1: Localized color edit ONLY on green vase in adh_04_spatial (cobalt blue vase; fern/lamp/curtain 100% locked!)
        src1 = adh / "adh_04_spatial_00001_.png"
        shutil.copy2(src1, in_dir / "s9_adh04_src.png")
        mask1 = in_dir / "s9_adh04_vase_mask.png"
        generate_region_mask(896, 1280, boxes_norm=[(21.0, 65.0, 48.0, 88.0)], pad_pct=2.0, feather_px=24, shape="ellipse", out_path=mask1)
        g1 = build_flux2_masked_edit_workflow({
            "SOURCE_IMAGE": "s9_adh04_src.png",
            "MASK_IMAGE": "s9_adh04_vase_mask.png",
            "PROMPT": "Interior studio photograph on a wooden table: replace the vase inside the masked area with a deep cobalt blue glass vase matching the exact soft daylight reflections, keep the brass lamp on the left, wooden table, tall potted fern, and white curtain completely unchanged.",
            "WIDTH": 896, "HEIGHT": 1280, "STEPS": 28, "CFG": 3.5, "SEED": 91001,
            "OUTPUT_PREFIX": "s9_masked_adh04_blue_vase",
        })
        print("\n[Masked Edit 1] adh_04 localized vase color change (green -> cobalt blue, outside mask locked)...", flush=True)
        r1 = _run_graph(g1, out_dir)
        pres1 = measure_outside_mask_preservation(src1, r1["path"], mask1)
        # Also measure Sprint 8 unmasked ReferenceLatent edit outside-mask drift on the exact same vase mask for comparison!
        s8_unmasked_pres = measure_outside_mask_preservation(src1, S8_ROOT / "s8_adh04_E1_edit_00001_.png", mask1)
        trials.append({
            "trial_id": "masked_color_edit_adh04",
            **r1,
            "mask_path": str(mask1),
            "preservation_masked_edit": pres1,
            "comparison_s8_unmasked_reflatent": s8_unmasked_pres,
        })
        print(f"  -> Done in {r1['elapsed_s']}s | peak VRAM={r1['peak_vram_mb']}MB | outside_mask_pct_changed={pres1['outside_mask_pct_changed']}% (vs S8 unmasked {s8_unmasked_pres['outside_mask_pct_changed']}%)", flush=True)

        # Test 2: Localized object replacement ONLY on left handleless bowl in adh_03_quantities -> handled coffee mug!
        src2 = adh / "adh_03_quantities_00001_.png"
        shutil.copy2(src2, in_dir / "s9_adh03_src.png")
        mask2 = in_dir / "s9_adh03_bowl_mask.png"
        generate_region_mask(896, 1280, boxes_norm=[(5.0, 20.0, 38.0, 44.0)], pad_pct=2.0, feather_px=24, shape="ellipse", out_path=mask2)
        g2 = build_flux2_masked_edit_workflow({
            "SOURCE_IMAGE": "s9_adh03_src.png",
            "MASK_IMAGE": "s9_adh03_bowl_mask.png",
            "PROMPT": "Overhead oak wooden desk photograph: replace the empty white bowl inside the masked area with a white ceramic coffee mug with a distinct curved handle on its left side filled with dark brewed coffee, matching the other two coffee mugs on the desk.",
            "WIDTH": 896, "HEIGHT": 1280, "STEPS": 28, "CFG": 3.5, "SEED": 91002,
            "OUTPUT_PREFIX": "s9_masked_adh03_bowl_to_mug",
        })
        print("[Masked Edit 2] adh_03 localized object replacement (left bowl -> handled coffee mug)...", flush=True)
        r2 = _run_graph(g2, out_dir)
        pres2 = measure_outside_mask_preservation(src2, r2["path"], mask2)
        trials.append({
            "trial_id": "masked_object_replace_adh03",
            **r2,
            "mask_path": str(mask2),
            "preservation_masked_edit": pres2,
        })
        print(f"  -> Done in {r2['elapsed_s']}s | peak VRAM={r2['peak_vram_mb']}MB | outside_mask_pct_changed={pres2['outside_mask_pct_changed']}%", flush=True)
    finally:
        _comfy_free()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    out_p = OUT_ROOT / "sprint9_masked_edit_results.json"
    out_p.write_text(json.dumps({"benchmark": "sprint9_masked_localized_edit", "trials": trials}, indent=2), encoding="utf-8")
    print(f"Saved masked edit results to {out_p}", flush=True)
    return {"trials": trials}


# --------------------------------------------------------------------------- Mode 3: Live Wan 2.2 A14B Lightning vs Standard Base


def run_wan_video_eval() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_ROOT / "outputs"
    in_dir = OUT_ROOT / "input"
    starter = BASELINE_ROOT / "outputs" / "mref" / "seed_front_00001_.png"
    verify_ollama_vram_released(DEFAULT_OLLAMA_URL, timeout_s=5.0)
    proc = _start_isolated_comfy(out_dir, in_dir)

    shutil.copy2(starter, in_dir / "s9_wan_starter.png")
    base_tpl = json.loads((REPO_ROOT / "workflows" / "wan22_i2v_14b_gguf_api.json").read_text(encoding="utf-8"))

    def _fill_wan(steps: int, cfg: float, prefix: str, use_lightning: bool) -> dict[str, Any]:
        raw = json.dumps(base_tpl)
        subs = {
            "${WAN_HIGH_MODEL}": "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
            "${WAN_LOW_MODEL}": "Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf",
            "${WAN_HIGH_LORA}": "wan2.2_i2v_lightning_4step_high.safetensors",
            "${WAN_LOW_LORA}": "wan2.2_i2v_lightning_4step_low.safetensors",
            "${WAN_TEXT_ENCODER}": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "${WAN_CLIP_DEVICE}": "cpu",
            "${PROMPT}": "The adult woman presenter looks warmly into the camera, smiles gently, smoothly raises her right hand to wave hello with five natural fingers, then lowers her hand calmly while keeping her olive blazer, low hair bun, and bright neutral studio background completely steady.",
            "${WAN_NEGATIVE}": "blurry, melted fingers, extra fingers, flickering background, abrupt jump, warped face, frozen still image",
            "${WAN_VAE}": "wan_2.1_vae.safetensors",
            "${INIT_IMAGE}": "s9_wan_starter.png",
            "${WAN_WIDTH}": "480",
            "${WAN_HEIGHT}": "832",
            "${WAN_LENGTH}": "49",
            "${WAN_SEED}": "92001",
            "${WAN_PREFIX}": prefix,
        }
        for k, v in subs.items():
            raw = raw.replace(f'"{k}"', v if k in {"${WAN_WIDTH}", "${WAN_HEIGHT}", "${WAN_LENGTH}", "${WAN_SEED}"} else json.dumps(v))
        graph = json.loads(raw)
        if not use_lightning:
            graph = without_wan_lightning(graph, steps=steps, cfg=cfg)
        else:
            half = max(1, steps // 2)
            graph["13"]["inputs"].update({"steps": steps, "cfg": cfg, "start_at_step": 0, "end_at_step": half})
            graph["14"]["inputs"].update({"steps": steps, "cfg": cfg, "start_at_step": half, "end_at_step": steps})
        return with_tiled_vae(graph)

    configs = [
        ("wan14b_lightning_4step_cfg1", 4, 1.0, True),
        ("wan14b_lightning_6step_cfg1", 6, 1.0, True),
        ("wan14b_base_10step_cfg3_5", 10, 3.5, False),
    ]
    trials: list[dict[str, Any]] = []
    try:
        for cid, steps, cfg, use_lt in configs:
            print(f"\n[Wan 2.2 Benchmark] {cid} (steps={steps}, cfg={cfg}, lightning={use_lt}, 49 frames @ 480x832)...", flush=True)
            g = _fill_wan(steps, cfg, f"s9_{cid}", use_lt)
            res = _run_graph(g, out_dir)
            temp = analyze_video_temporal_cpu(res["path"], starter_image=starter, max_frames=12)
            trials.append({
                "config_id": cid,
                "steps": steps,
                "cfg": cfg,
                "use_lightning": use_lt,
                "frames": 49,
                **res,
                "temporal_cpu": {
                    "max_luma_jump": temp.get("max_luma_jump"),
                    "max_dhash_jump": temp.get("max_dhash_jump"),
                    "max_color_shift": temp.get("max_color_shift"),
                    "median_motion_energy": temp.get("median_motion_energy"),
                    "is_effectively_static": temp.get("is_effectively_static"),
                    "median_sharpness": temp.get("median_sharpness"),
                    "temporal_defects": temp.get("temporal_defects"),
                    "start_endpoint_dhash": (temp.get("start_endpoint_comparison") or {}).get("dhash_distance"),
                },
            })
            print(
                f"  -> {res['elapsed_s']}s | peak VRAM={res['peak_vram_mb']}MB | "
                f"motion={temp.get('median_motion_energy')} | luma_jump={temp.get('max_luma_jump')} | "
                f"dhash_jump={temp.get('max_dhash_jump')} | defects={temp.get('temporal_defects')}",
                flush=True,
            )
            _comfy_free()
    finally:
        _comfy_free()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    out_p = OUT_ROOT / "sprint9_wan_video_results.json"
    out_p.write_text(json.dumps({"benchmark": "sprint9_wan_video_comparison", "trials": trials}, indent=2), encoding="utf-8")
    print(f"Saved Wan video comparison results to {out_p}", flush=True)
    return {"trials": trials}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["blind_grid", "masked_edit", "wan_video", "all"], default="all")
    args = parser.parse_args()
    if args.mode in {"blind_grid", "all"}:
        run_blind_grid_eval()
    if args.mode in {"masked_edit", "all"}:
        run_masked_edit_eval()
    if args.mode in {"wan_video", "all"}:
        run_wan_video_eval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
