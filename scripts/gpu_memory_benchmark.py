#!/usr/bin/env python3
"""Reproducible GPU Memory & Generation Benchmark Harness for Avatar V2.

Safely establishes controlled baselines for FLUX.2 4B/9B and Wan 2.2 14B/5B workflows:
- Refuses to run live GPU trials if ComfyUI's queue is non-empty or another heavy GPU workload
  is active (use --dry-run to validate preflight & workflow resolution without touching the GPU).
- Captures per-trial: elapsed time, peak nvidia-smi VRAM, ComfyUI /system_stats torch pool,
  host RAM/swap/PSI/process RSS, GPU temperature & utilization, post-cleanup recovery, and
  optional tiled-VAE A/B comparison (one isolated variable at a time).
- Writes structured JSON + Markdown reports under ~/.avatar_v2/benchmarks/ (or --out-dir).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from avatar_v2 import local_engine, offline_builder
from avatar_v2.adherence_benchmark import default_adherence_suite
from avatar_v2.gpu_memory import GPUMemoryManager, query_gpu_processes
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from avatar_v2.providers.comfyui import ComfyUIProvider
from avatar_v2.resource_monitor import (
    ResourceGuard,
    check_storage_for_job,
    estimate_job_storage_bytes,
    query_gpu_telemetry,
)
from avatar_v2.workflows_builder import with_tiled_vae

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_OUT_DIR = Path.home() / ".avatar_v2" / "benchmarks"
ACTIVE_GPU_UTIL_GATE_PCT = 35


def read_proc_memory_breakdown(pid: int | None) -> dict[str, int]:
    """Read Linux RssAnon, RssFile, VmRSS, VmSwap, and Pss (MiB) for a process."""
    if not pid or pid <= 0:
        return {}
    out: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmPin", "VmLck"):
                if line.startswith(f"{key}:"):
                    out[f"{key.lower()}_mb"] = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def parse_comfyui_stage_timings(log_slice: str) -> dict[str, Any]:
    """Extract total execution time, denoising step seconds, and loaded/offloaded MB from a ComfyUI log slice."""
    timings: dict[str, Any] = {"step_durations_s": [], "loaded_mb": [], "offloaded_mb": []}
    m_total = re.search(r"Prompt executed in (\d+(?:\.\d+)?) seconds", log_slice)
    if m_total:
        timings["comfy_total_s"] = float(m_total.group(1))
    for m_step in re.finditer(r"(\d+(?:\.\d+)?)s/it\]", log_slice):
        timings["step_durations_s"].append(float(m_step.group(1)))
    for m_load in re.finditer(r"loaded (?:partially|completely);[^\n]*?(\d+(?:\.\d+)?) MB loaded(?:,\s*(\d+(?:\.\d+)?) MB offloaded)?", log_slice):
        timings["loaded_mb"].append(float(m_load.group(1)))
        if m_load.group(2) is not None:
            timings["offloaded_mb"].append(float(m_load.group(2)))
    if timings["step_durations_s"]:
        timings["mean_s_per_it"] = round(sum(timings["step_durations_s"]) / len(timings["step_durations_s"]), 2)
    return timings


def _comfy_pid() -> int | None:
    pid_path = Path.home() / ".avatar_v2" / "comfyui.pid"
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def check_gpu_idle_for_benchmark(base_url: str = DEFAULT_COMFY_URL) -> dict[str, Any]:
    """Verify ComfyUI is idle and no competing GPU compute workload is running."""
    reasons: list[str] = []
    comfy_stats: dict[str, Any] | None = None
    queue_busy = False
    try:
        with httpx.Client(timeout=5.0) as client:
            q = client.get(f"{base_url.rstrip('/')}/queue").json()
            running = len(q.get("queue_running") or [])
            pending = len(q.get("queue_pending") or [])
            queue_busy = bool(running or pending)
            if queue_busy:
                reasons.append(f"ComfyUI queue is busy (running={running}, pending={pending})")
            comfy_stats = client.get(f"{base_url.rstrip('/')}/system_stats").json()
    except Exception as exc:
        reasons.append(f"ComfyUI unreachable at {base_url}: {type(exc).__name__}: {exc}")

    gpus = query_gpu_telemetry(comfy_system_stats=comfy_stats)
    for gpu in gpus:
        if gpu.util_pct is not None and gpu.util_pct >= ACTIVE_GPU_UTIL_GATE_PCT:
            reasons.append(f"GPU {gpu.index} utilization is {gpu.util_pct}% (>= {ACTIVE_GPU_UTIL_GATE_PCT}% gate)")

    guard = ResourceGuard()
    snap = guard.evaluate(gpus=gpus, process_pid=_comfy_pid(), comfy_system_stats=comfy_stats)
    return {
        "idle": not reasons,
        "queue_busy": queue_busy,
        "reasons": reasons,
        "processes": query_gpu_processes(),
        "resource_snapshot": snap.public(),
    }


def build_scenario_workflow(
    scenario: str,
    *,
    seed: int,
    tiled_vae: bool = False,
    wan_clip_device: str = "default",
    init_image: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic ComfyUI API workflow for a named benchmark scenario."""
    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "bench",
            "display_name": "Benchmark",
            "subjects": [{"id": "bench", "kind": "synthetic"}],
            "identity_refs": [init_image] if init_image else [],
        }
    )
    provider = ComfyUIProvider(ProviderConfig(base_url=DEFAULT_COMFY_URL))

    if scenario in {"flux2_4b", "flux2_9b"}:
        model_id = "4b" if scenario == "flux2_4b" else "9b"
        spec = offline_builder.PHOTO_MODELS[model_id]
        width, height = (896, 1280)
        shot = ShotSpec.model_validate(
            {
                "shot_id": f"bench_{scenario}",
                "user_prompt": "Photorealistic studio portrait of a confident course instructor, natural soft lighting, 85mm lens",
                "width": width,
                "height": height,
                "seed": seed,
                "fps": 1,
                "duration_s": 1.0,
            }
        )
        asset_map: dict[str, Any] = {
            **offline_builder.image_placeholders(model_id),
            "PROMPT": shot.user_prompt,
            "NEGATIVE_PROMPT": offline_builder.PHOTO_NEGATIVE,
            "WIDTH": width,
            "HEIGHT": height,
            "STEPS": spec["steps"],
            "CFG": spec["cfg"],
            "SEED": seed,
            "SAMPLER_NAME": offline_builder.PHOTO_SAMPLER,
            "OUTPUT_PREFIX": f"avatar_v2/bench_{scenario}",
            "TILED_VAE": tiled_vae,
        }
        job = RenderJob(
            job_id=f"bench-{scenario}-{seed}",
            avatar=avatar,
            shot=shot,
            selected_engine="custom",
            prompt=shot.user_prompt,
            negative_prompt=offline_builder.PHOTO_NEGATIVE,
            asset_map=asset_map,
        )
        workflow = provider.resolve_workflow("builtin:flux2_txt2img", job, stage_assets=False)
        meta = {
            "scenario": scenario,
            "kind": "image",
            "model": spec["unet"],
            "width": width,
            "height": height,
            "frames": 1,
            "steps": spec["steps"],
            "cfg": spec["cfg"],
            "tiled_vae": tiled_vae,
        }
        return workflow, meta

    if scenario in {"wan22_14b", "wan22_5b"}:
        is_14b = scenario == "wan22_14b"
        width, height = local_engine.SHAPES["tall"]
        frames = local_engine.FRAMES
        wf_path = local_engine.WORKFLOW if is_14b else local_engine.WORKFLOW_5B
        placeholders = (
            local_engine.model_placeholders()
            if is_14b
            else {**local_engine.model_placeholders(), **local_engine.model_placeholders_5b(), "WAN_VAE": local_engine.MODELS_5B["WAN5B_VAE"][1]}
        )
        prompt = "She smiles warmly, nods gently, and gestures naturally while explaining a lesson."
        shot = ShotSpec.model_validate(
            {
                "shot_id": f"bench_{scenario}",
                "user_prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "fps": local_engine.FPS,
                "duration_s": frames / local_engine.FPS,
            }
        )
        asset_map = {
            **placeholders,
            "PROMPT": prompt,
            "WAN_NEGATIVE": local_engine.NEGATIVE,
            "WAN_WIDTH": width,
            "WAN_HEIGHT": height,
            "WAN_LENGTH": frames,
            "WAN_SEED": seed,
            "WAN_PREFIX": f"avatar_v2/bench_{scenario}",
            "WAN_CLIP_DEVICE": wan_clip_device,
            "INIT_IMAGE": init_image or "bench_start.png",
        }
        job = RenderJob(
            job_id=f"bench-{scenario}-{seed}",
            avatar=avatar,
            shot=shot,
            selected_engine="wan22",
            prompt=prompt,
            negative_prompt=local_engine.NEGATIVE,
            asset_map=asset_map,
        )
        workflow = provider.resolve_workflow(wf_path, job, stage_assets=False)
        if tiled_vae:
            workflow = with_tiled_vae(workflow)
        meta = {
            "scenario": scenario,
            "kind": "video",
            "model": local_engine.VIDEO_MODELS["a14b" if is_14b else "5b"]["label"],
            "width": width,
            "height": height,
            "frames": frames,
            "steps": 4,
            "cfg": 1.0,
            "tiled_vae": tiled_vae,
            "wan_clip_device": wan_clip_device,
        }
        return workflow, meta

    raise ValueError(f"Unknown scenario '{scenario}'. Choose from: flux2_4b, flux2_9b, wan22_14b, wan22_5b")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    idle_check = check_gpu_idle_for_benchmark(args.comfy_url)
    workflow, meta = build_scenario_workflow(
        args.scenario,
        seed=args.seed,
        tiled_vae=args.tiled_vae,
        wan_clip_device=args.wan_clip_device,
        init_image=args.init_image,
    )
    est_bytes = estimate_job_storage_bytes(
        kind=meta["kind"],
        width=meta["width"],
        height=meta["height"],
        frames=meta["frames"],
    )
    storage_gate = check_storage_for_job(out_dir, estimated_bytes=est_bytes * max(args.trials, 1))

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "meta": meta,
        "trials_requested": args.trials,
        "idle_check": idle_check,
        "comfy_memory_breakdown": read_proc_memory_breakdown(_comfy_pid()),
        "storage_gate": storage_gate.public(),
        "workflow_nodes": len(workflow),
        "has_tiled_vae_node": any(n.get("class_type") == "VAEDecodeTiled" for n in workflow.values()),
        "adherence_suite_cases": len(default_adherence_suite()),
        "trials": [],
    }

    if args.dry_run:
        return report

    if not idle_check["idle"]:
        raise RuntimeError(
            "Refusing to run live GPU benchmark while GPU/ComfyUI is not idle: "
            + "; ".join(idle_check["reasons"])
        )
    if not storage_gate.allowed:
        raise RuntimeError(f"Storage guard blocked benchmark: {storage_gate.reason}")
    if meta["kind"] == "video" and not args.init_image:
        raise RuntimeError("Video benchmark scenarios require --init-image pointing to a local PNG/JPG.")

    provider = ComfyUIProvider(ProviderConfig(base_url=args.comfy_url, timeout_s=1800))
    mem = GPUMemoryManager.from_environment()
    guard = ResourceGuard(out_dir)

    for idx in range(args.trials):
        trial_seed = args.seed + idx
        wf, _ = build_scenario_workflow(
            args.scenario,
            seed=trial_seed,
            tiled_vae=args.tiled_vae,
            wan_clip_device=args.wan_clip_device,
            init_image=args.init_image,
        )
        if meta["kind"] == "video" and args.init_image:
            staged = provider.upload_file(args.init_image, subfolder="avatar_v2/bench")
            for node in wf.values():
                if node.get("class_type") == "LoadImage":
                    node["inputs"]["image"] = staged

        pre_clean = mem.maybe_cleanup(
            provider,
            reason=f"bench-pre:{args.scenario}:{idx + 1}",
            force=not args.keep_loaded or idx == 0,
            unload_models=not args.keep_loaded or idx == 0,
        )
        before_snap = guard.evaluate(process_pid=_comfy_pid())
        mem.start_trace(f"bench-{args.scenario}-{idx + 1}")
        t0 = time.monotonic()
        error = ""
        outputs: list[str] = []
        try:
            prompt_id = provider.queue(wf)
            history = provider.wait(prompt_id)
            outputs = provider.output_files(history)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed_s = round(time.monotonic() - t0, 2)
        trace = mem.stop_trace()
        after_snap = guard.evaluate(process_pid=_comfy_pid())
        post_clean = mem.maybe_cleanup(
            provider,
            reason=f"bench-post:{args.scenario}:{idx + 1}",
            force=not args.keep_loaded or bool(error),
            unload_models=not args.keep_loaded or bool(error),
        )
        report["trials"].append(
            {
                "trial": idx + 1,
                "seed": trial_seed,
                "elapsed_s": elapsed_s,
                "error": error,
                "outputs": outputs,
                "gpu_trace": trace,
                "pre_clean": pre_clean,
                "post_clean": post_clean,
                "before_resources": before_snap.public(),
                "after_resources": after_snap.public(),
            }
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = out_dir / f"bench_{args.scenario}_{'tiled' if args.tiled_vae else 'std'}_{stamp}.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["saved_to"] = str(out_file)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avatar V2 reproducible GPU memory benchmark harness")
    parser.add_argument("--scenario", choices=("flux2_4b", "flux2_9b", "wan22_14b", "wan22_5b"), default="flux2_4b")
    parser.add_argument("--trials", type=int, default=3, help="Number of repeated trials (default: 3)")
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--tiled-vae", action="store_true", help="Replace VAEDecode with VAEDecodeTiled")
    parser.add_argument("--keep-loaded", action="store_true", help="Skip /free unload between repeated trials")
    parser.add_argument("--wan-clip-device", choices=("default", "cpu"), default="default")
    parser.add_argument("--init-image", default=None, help="Local start image for video scenarios")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--list-adherence", action="store_true", help="Print the 13-category prompt adherence benchmark suite as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Preflight check & workflow build without GPU execution")
    args = parser.parse_args(argv)
    if args.list_adherence:
        suite = [c.public() for c in default_adherence_suite()]
        print(json.dumps({"ok": True, "count": len(suite), "cases": suite}, indent=2))
        return 0
    try:
        report = run_benchmark(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"ok": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
