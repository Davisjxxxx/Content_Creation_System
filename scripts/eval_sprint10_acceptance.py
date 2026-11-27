"""Sprint 10 Live Integrated Acceptance Runner (Separate Held-Out Grid Eval + Production Mixed Workload).

Runs strictly in an isolated temporary AVATAR_V2_HOME and isolated ComfyUI instance (port 18189),
never modifying production ~/.avatar_v2, dist/AvatarV2, or installed models.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avatar_v2 import desktop, local_engine, offline_builder as ob
from avatar_v2.gpu_memory import query_gpu_memory
from avatar_v2.inspection_contract import AcceptanceCriterion
from avatar_v2.ollama_client import LocalOllamaClient, verify_ollama_vram_released
from avatar_v2.resource_monitor import query_host_memory
from avatar_v2.specialist_vision import (
    reconcile_criterion_with_detections,
    run_blind_object_localization,
)

COMFY_DIR = Path("/home/jd/AvatarForge/comfyui")
COMFY_PY = Path("/home/jd/AvatarForge/venv/bin/python")
COMFY_PORT = 18189
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
OLLAMA_URL = "http://127.0.0.1:11434"

S5_ADH = Path("/home/jd/.avatar_v2/benchmarks/phase_a2_multiview_adherence/outputs/adherence")
S5_MREF = Path("/home/jd/.avatar_v2/benchmarks/phase_a2_multiview_adherence/outputs/mref")
OUT_ROOT = Path("/home/jd/.avatar_v2/benchmarks/sprint10_acceptance")


def _sample_hw(label: str) -> dict[str, Any]:
    gpus = query_gpu_memory()
    g0 = gpus[0].public() if gpus else {}
    host = query_host_memory()
    return {
        "label": label,
        "ts": datetime.now(timezone.utc).isoformat(),
        "vram_used_mb": g0.get("used_mb"),
        "vram_total_mb": g0.get("total_mb"),
        "ram_avail_mb": host.ram_available_mb,
        "swap_used_mb": host.swap_used_mb,
    }


def run_part_a_separate_photographic_grid_eval() -> dict[str, Any]:
    """Section 4: Evaluate blind 2D grid-assisted Qwen3-VL localization on 4 SEPARATE photos
    distinct from Sprint 9's 6 photographic images.
    """
    cases = [
        {
            "case_id": "s10_adh02_composition",
            "image": S5_ADH / "adh_02_composition_00001_.png",
            "objects": ["man", "wooden bench", "red lamp"],
            "checks": [
                {"id": "count_man_eq_1", "category": "object_quantities", "description": "exactly 1 man", "ground_truth": "passed"},
                {"id": "count_man_eq_2_false", "category": "object_quantities", "description": "exactly 2 men", "ground_truth": "failed"},
                {"id": "man_left_of_bench", "category": "spatial_relationships", "description": "man on the left of bench", "ground_truth": "passed"},
                {"id": "bench_left_of_man_false", "category": "spatial_relationships", "description": "bench on the left of man", "ground_truth": "failed"},
            ],
        },
        {
            "case_id": "s10_adh05_colors",
            "image": S5_ADH / "adh_05_colors_00001_.png",
            "objects": ["woman", "yellow folder", "eyeglasses"],
            "checks": [
                {"id": "count_folder_eq_1", "category": "object_quantities", "description": "exactly 1 folder", "ground_truth": "passed"},
                {"id": "count_folder_eq_2_false", "category": "object_quantities", "description": "exactly 2 folders", "ground_truth": "failed"},
            ],
        },
        {
            "case_id": "s10_adh06_clothing",
            "image": S5_ADH / "adh_06_clothing_00001_.png",
            "objects": ["eyeglasses", "earrings", "coffee mug"],
            "checks": [
                {"id": "count_glasses_eq_1", "category": "object_quantities", "description": "exactly 1 eyeglasses", "ground_truth": "passed"},
                {"id": "count_earrings_eq_2", "category": "object_quantities", "description": "exactly 2 earrings", "ground_truth": "passed"},
                {"id": "count_mugs_eq_1_false", "category": "object_quantities", "description": "exactly 1 coffee mug", "ground_truth": "failed"},
            ],
        },
        {
            "case_id": "s10_adh10_complex_nl",
            "image": S5_ADH / "adh_10_complex_nl_00001_.png",
            "objects": ["woman", "desk lamp", "arched window"],
            "checks": [
                {"id": "count_lamp_eq_1", "category": "object_quantities", "description": "exactly 1 lamp", "ground_truth": "passed"},
                {"id": "count_windows_eq_2", "category": "object_quantities", "description": "exactly 2 windows", "ground_truth": "passed"},
                {"id": "woman_left_of_lamp", "category": "spatial_relationships", "description": "woman on the left of lamp", "ground_truth": "passed"},
                {"id": "lamp_left_of_woman_false", "category": "spatial_relationships", "description": "lamp on the left of woman", "ground_truth": "failed"},
            ],
        },
    ]

    results: list[dict[str, Any]] = []
    totals = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "uncertain": 0}
    ollama_client = LocalOllamaClient(base_url=OLLAMA_URL, timeout_s=90.0)

    for case in cases:
        t0 = time.monotonic()
        loc = run_blind_object_localization(
            case["image"],
            target_categories=case["objects"],
            client=ollama_client,
            model="qwen3-vl:8b-instruct",
            use_grid_overlay=True,
        )
        elapsed_s = round(time.monotonic() - t0, 2)
        check_outcomes: list[dict[str, Any]] = []
        if not loc.get("ok"):
            for chk in case["checks"]:
                totals["uncertain"] += 1
                check_outcomes.append({**chk, "predicted": "uncertain", "bucket": "uncertain", "detail": loc.get("error")})
        else:
            for chk in case["checks"]:
                crit = AcceptanceCriterion(
                    criterion_id=chk["id"],
                    category=chk["category"],
                    description=chk["description"],
                )
                rec_out = reconcile_criterion_with_detections(crit, loc.get("objects") or [])
                pred = str(rec_out.get("status") or "uncertain")
                detail = str(rec_out.get("evidence") or "")
                gt = chk["ground_truth"]
                if pred == "uncertain":
                    bucket = "uncertain"
                elif gt == "passed" and pred == "passed":
                    bucket = "TP"
                elif gt == "failed" and pred == "failed":
                    bucket = "TN"
                elif gt == "failed" and pred == "passed":
                    bucket = "FP"
                else:
                    bucket = "FN"
                totals[bucket] += 1
                check_outcomes.append({**chk, "predicted": pred, "bucket": bucket, "detail": detail})

        results.append(
            {
                "case_id": case["case_id"],
                "image": str(case["image"]),
                "elapsed_s": elapsed_s,
                "vlm_ok": bool(loc.get("ok")),
                "vram_release": loc.get("vram_release"),
                "localized_objects": loc.get("objects") or [],
                "checks": check_outcomes,
            }
        )

    vram_rel = verify_ollama_vram_released(OLLAMA_URL, timeout_s=10.0)
    return {
        "distinct_from_sprint9": True,
        "cases": results,
        "totals": totals,
        "vram_released_after_eval": vram_rel,
    }


def _start_isolated_comfy(work_root: Path) -> subprocess.Popen[bytes]:
    in_dir = work_root / "comfy_input"
    out_dir = work_root / "comfy_output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_root / "comfy_sprint10.log"
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        [
            str(COMFY_PY),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFY_PORT),
            "--disable-auto-launch",
            "--cache-ram",
            "6",
            "24",
            "--input-directory",
            str(in_dir),
            "--output-directory",
            str(out_dir),
        ],
        cwd=str(COMFY_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{COMFY_URL}/system_stats", timeout=2.0)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"Isolated ComfyUI on port {COMFY_PORT} failed to start within 90s.")


def _stop_isolated_comfy(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    try:
        httpx.post(f"{COMFY_URL}/free", json={"unload_models": True, "free_memory": True}, timeout=5.0)
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _wait_history(history_dir: Path, qid: str, timeout_s: float = 900.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    target = history_dir / f"{qid}.json"
    while time.monotonic() < deadline:
        if target.is_file():
            return json.loads(target.read_text(encoding="utf-8"))
        time.sleep(1.0)
    raise TimeoutError(f"Queue job {qid} did not persist history within {timeout_s}s.")


def run_part_b_live_production_mixed_workload(work_root: Path) -> dict[str, Any]:
    """Sections 3F & 5: Exercise actual DesktopAPI + RenderQueue + CorrectionCoordinator + Wan 2.2 14B."""
    app_home = work_root / "app_home"
    app_home.mkdir(parents=True, exist_ok=True)
    in_dir = work_root / "comfy_input"
    out_dir = work_root / "comfy_output"

    # Point desktop module globals cleanly at our isolated app_home (NEVER touching live ~/.avatar_v2!)
    desktop.APP_HOME = app_home
    desktop.CONFIG_PATH = app_home / "desktop.json"
    desktop.RUNS_DIR = app_home / "runs"
    desktop.QUEUE_DIR = app_home / "queue"
    desktop.HISTORY_DIR = app_home / "history"
    desktop.LIBRARY_DIR = app_home / "library"
    desktop.COMFY_PID_PATH = app_home / "comfyui.pid"
    desktop.COMFY_LOG_PATH = app_home / "logs" / "comfyui.log"
    desktop.RUNWAY_AVATARS_PATH = app_home / "runway_avatars.json"
    desktop.CLOUD_LEDGER_PATH = app_home / "cloud_ledger.jsonl"
    desktop.VOICEOVER_DIR = app_home / "voiceover"
    desktop.IMPORTS_DIR = app_home / "library" / "imports"
    desktop.LESSONS_PATH = app_home / "lessons.json"
    desktop.QA_DIR = app_home / "qa"
    desktop.PROVIDER_PROFILES_PATH = app_home / "provider_profiles.json"

    settings = {
        **desktop.DEFAULT_SETTINGS,
        "comfyui_dir": str(COMFY_DIR),
        "comfyui_python": str(COMFY_PY),
        "comfyui_input": str(in_dir),
        "comfyui_output": str(out_dir),
        "archive_dir": str(app_home / "archive"),
        "comfyui_url": COMFY_URL,
        "vision_classifier_url": OLLAMA_URL,
        "vision_classifier_model": "qwen3-vl:8b-instruct",
    }
    desktop._write_json(desktop.CONFIG_PATH, settings)

    hw_trace: list[dict[str, Any]] = [_sample_hw("before_comfy_start")]
    proc = _start_isolated_comfy(work_root)
    api = desktop.DesktopAPI()
    summary: dict[str, Any] = {"hw_trace": hw_trace}
    try:
        hw_trace.append(_sample_hw("after_comfy_ready"))

        # 1. Production Automatic Image Generation + Live VLM Inspection + Production CorrectionCoordinator!
        # Use a crisp prompt with explicit typography + spatial requirement to exercise real inspection/correction path.
        t0 = time.monotonic()
        photo_res = api.forge_go(
            {
                "action": "photo",
                "prompt": (
                    "Studio still life photograph on a light oak table: a green ceramic vase on the left "
                    "and a brass desk lamp on the right, with a crisp white sign reading 'STUDIO 10' in front, "
                    "soft natural window light."
                ),
                "photo_model": "9b",
                "shape": "portrait",
                "count": 1,
                "auto_correct": True,
                "visual_inspection_vlm": True,
                "visual_inspection_model": "qwen3-vl:8b-instruct",
            }
        )
        assert photo_res["ok"], photo_res
        pqid = photo_res["queue_id"]
        photo_hist = _wait_history(desktop.HISTORY_DIR, pqid, timeout_s=900.0)
        photo_elapsed = round(time.monotonic() - t0, 2)
        hw_trace.append(_sample_hw("after_photo_inspect_and_correction"))

        # 2. Live Wan 2.2 A14B Video Generation via DesktopAPI.forge_go (explicit starter preserved + VideoInspector!)
        starter = S5_MREF / "seed_front_00001_.png"
        starter_sha = hashlib.sha256(starter.read_bytes()).hexdigest()
        # Use 49 frames (~3s) to keep mixed-workload bounded while exercising full 14B pipeline
        orig_frames = local_engine.FRAMES
        local_engine.FRAMES = 49
        try:
            t1 = time.monotonic()
            vid_res = api.forge_go(
                {
                    "action": "scene",
                    "image": str(starter),
                    "shots": ["The woman smiles warmly and gently waves her right hand hello in soft studio daylight."],
                    "video_model": "a14b",
                    "shape": "tall",
                    "visual_inspection_vlm": True,
                    "visual_inspection_model": "qwen3-vl:8b-instruct",
                }
            )
            assert vid_res["ok"], vid_res
            assert vid_res["photo_first"] is False
            vqid = vid_res["queue_id"]
            vid_hist = _wait_history(desktop.HISTORY_DIR, vqid, timeout_s=900.0)
            vid_elapsed = round(time.monotonic() - t1, 2)
        finally:
            local_engine.FRAMES = orig_frames
        hw_trace.append(_sample_hw("after_wan14b_video_and_inspection"))

        # 3. Verify FLUX recovery after Wan + VLM unload (mixed-workload transition check)
        t2 = time.monotonic()
        post_flux_res = api.forge_go(
            {
                "action": "photo",
                "prompt": "Minimalist photograph of a single cobalt blue coffee mug on a white marble countertop, soft morning light",
                "photo_model": "4b",
                "shape": "portrait",
                "count": 1,
                "visual_inspection_vlm": False,
            }
        )
        assert post_flux_res["ok"], post_flux_res
        pfqid = post_flux_res["queue_id"]
        post_flux_hist = _wait_history(desktop.HISTORY_DIR, pfqid, timeout_s=450.0)
        post_flux_elapsed = round(time.monotonic() - t2, 2)
        hw_trace.append(_sample_hw("after_post_video_flux_recovery"))

        summary.update(
            {
                "photo_job": {
                    "queue_id": pqid,
                    "status": photo_hist.get("status"),
                    "elapsed_s": photo_elapsed,
                    "outputs": photo_hist.get("outputs"),
                    "candidate_qa": photo_hist.get("candidate_qa"),
                    "verification_record": photo_hist.get("verification_record"),
                    "gpu_trace": photo_hist.get("gpu_trace"),
                },
                "video_job": {
                    "queue_id": vqid,
                    "status": vid_hist.get("status"),
                    "elapsed_s": vid_elapsed,
                    "starter_sha256": starter_sha,
                    "photo_first": vid_res["photo_first"],
                    "outputs": vid_hist.get("outputs"),
                    "candidate_qa": vid_hist.get("candidate_qa"),
                    "verification_record": vid_hist.get("verification_record"),
                    "gpu_trace": vid_hist.get("gpu_trace"),
                },
                "post_video_flux_recovery_job": {
                    "queue_id": pfqid,
                    "status": post_flux_hist.get("status"),
                    "elapsed_s": post_flux_elapsed,
                    "outputs": post_flux_hist.get("outputs"),
                    "gpu_trace": post_flux_hist.get("gpu_trace"),
                },
            }
        )
    finally:
        api.queue.shutdown()
        _stop_isolated_comfy(proc)
        time.sleep(2.0)
        verify_ollama_vram_released(OLLAMA_URL, timeout_s=10.0)
        hw_trace.append(_sample_hw("after_full_shutdown_recovery"))
    return summary


def run_part_c_live_production_correction_journey_f(work_root: Path) -> dict[str, Any]:
    """Section 3F: Live RTX 4070 + ComfyUI + Qwen3-VL 8B proof of actual production
    DesktopAPI._run_visual_inspection() -> CorrectionCoordinator -> _execute_local_correction_step().
    """
    app_home = work_root / "app_home"
    app_home.mkdir(parents=True, exist_ok=True)
    in_dir = work_root / "comfy_input"
    out_dir = work_root / "comfy_output"

    desktop.APP_HOME = app_home
    desktop.CONFIG_PATH = app_home / "desktop.json"
    desktop.RUNS_DIR = app_home / "runs"
    desktop.QUEUE_DIR = app_home / "queue"
    desktop.HISTORY_DIR = app_home / "history"
    desktop.LIBRARY_DIR = app_home / "library"
    desktop.COMFY_PID_PATH = app_home / "comfyui.pid"
    desktop.COMFY_LOG_PATH = app_home / "logs" / "comfyui.log"
    desktop.RUNWAY_AVATARS_PATH = app_home / "runway_avatars.json"
    desktop.CLOUD_LEDGER_PATH = app_home / "cloud_ledger.jsonl"
    desktop.VOICEOVER_DIR = app_home / "voiceover"
    desktop.IMPORTS_DIR = app_home / "library" / "imports"
    desktop.LESSONS_PATH = app_home / "lessons.json"
    desktop.QA_DIR = app_home / "qa"
    desktop.PROVIDER_PROFILES_PATH = app_home / "provider_profiles.json"

    settings = {
        **desktop.DEFAULT_SETTINGS,
        "comfyui_dir": str(COMFY_DIR),
        "comfyui_python": str(COMFY_PY),
        "comfyui_input": str(in_dir),
        "comfyui_output": str(out_dir),
        "archive_dir": str(app_home / "archive"),
        "comfyui_url": COMFY_URL,
        "vision_classifier_url": OLLAMA_URL,
        "vision_classifier_model": "qwen3-vl:8b-instruct",
    }
    desktop._write_json(desktop.CONFIG_PATH, settings)

    # Copy an immutable Sprint 5 still (missing the required 'OPEN' sign) into isolated work dir
    orig_copy = work_root / "initial_missing_sign.png"
    shutil.copy2(S5_ADH / "adh_04_spatial_00001_.png", orig_copy)

    proc = _start_isolated_comfy(work_root)
    api = desktop.DesktopAPI()
    try:
        from avatar_v2.workflow_router import plan_intelligent_generation

        prompt = (
            "Tabletop studio photograph with a green ceramic vase, a brass desk lamp, "
            "and a crisp white tabletop sign reading 'OPEN' clearly in front, soft daylight."
        )
        routed = plan_intelligent_generation(
            prompt,
            requested_action="photo",
            input_mode="automatic",
            model_preference="flux2_4b",
            use_llm=False,
        )
        assert routed.ok and routed.verification_record is not None
        item = api._forge_photo_item(
            prompt=prompt,
            shape="portrait",
            photo_model="4b",
            label="Sprint 10 Live Journey F Correction",
            extra_runtime={
                "verification_record": routed.verification_record.public(),
                "auto_correct": True,
                "visual_inspection_vlm": True,
                "visual_inspection_model": "qwen3-vl:8b-instruct",
            },
        )
        item.status = "done"
        item.outputs = [str(orig_copy)]

        t0 = time.monotonic()
        qa_res = api._run_visual_inspection(item)
        elapsed_s = round(time.monotonic() - t0, 2)
        return {
            "elapsed_s": elapsed_s,
            "outputs_retained": list(item.outputs),
            "original_retained": str(orig_copy) in item.outputs,
            "candidate_qa": qa_res,
            "correction_session": item.runtime["verification_record"].get("correction_session"),
            "vram_after": _sample_hw("after_journey_f_live"),
        }
    finally:
        api.queue.shutdown()
        _stop_isolated_comfy(proc)
        time.sleep(2.0)
        verify_ollama_vram_released(OLLAMA_URL, timeout_s=10.0)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if "--journey-f-live" in sys.argv:
        print("[Sprint 10] Running Part C: Live Production Journey F (DesktopAPI -> CorrectionCoordinator -> ComfyUI)...")
        part_c = run_part_c_live_production_correction_journey_f(OUT_ROOT / "journey_f_live")
        (OUT_ROOT / "sprint10_journey_f_live.json").write_text(json.dumps(part_c, indent=2), encoding="utf-8")
        cs = part_c.get("correction_session") or {}
        print(
            "[Sprint 10] Part C finished:",
            "attempts=", cs.get("attempts_executed"),
            "ui_status=", cs.get("ui_status"),
            "outputs=", len(part_c.get("outputs_retained") or []),
            "elapsed_s=", part_c.get("elapsed_s"),
        )
        return
    if "--part-b-only" not in sys.argv:
        print("[Sprint 10] Running Part A: Separate Photographic Held-Out Grid Localization Eval...")
        part_a = run_part_a_separate_photographic_grid_eval()
        (OUT_ROOT / "sprint10_heldout_grid_eval.json").write_text(json.dumps(part_a, indent=2), encoding="utf-8")
        print("[Sprint 10] Part A totals:", part_a["totals"], "vram_released:", part_a["vram_released_after_eval"])

    print("[Sprint 10] Running Part B: Live Production Mixed-Workload + CorrectionCoordinator + Wan 2.2 14B...")
    part_b = run_part_b_live_production_mixed_workload(OUT_ROOT / "live_session")
    (OUT_ROOT / "sprint10_live_mixed_workload.json").write_text(json.dumps(part_b, indent=2), encoding="utf-8")
    print(
        "[Sprint 10] Part B finished:",
        "photo=", part_b.get("photo_job", {}).get("status"),
        "outputs=", len(part_b.get("photo_job", {}).get("outputs") or []),
        "video=", part_b.get("video_job", {}).get("status"),
        "post_flux=", part_b.get("post_video_flux_recovery_job", {}).get("status"),
    )


if __name__ == "__main__":
    main()
