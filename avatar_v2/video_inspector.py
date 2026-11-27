"""Efficient Local Video Inspection Pipeline (Sprint 7, Section 6).

Avoids running a large vision model on every frame:
1. Deterministic container/stream validation via ``ffprobe`` (duration, fps, frames, resolution).
2. Representative frame extraction via ``ffmpeg`` guaranteeing exact first frame, exact last frame,
   and evenly sampled intermediate frames (capped at ``max_frames``).
3. Fast CPU temporal analysis via pure Pillow:
   - Luma jump / flicker detection
   - Motion energy (frame-to-frame L1 pixel difference; catches frozen/static video)
   - Abrupt scene / identity / background jump detection (dHash Hamming + 24-bin RGB histogram shift)
   - Sharpness / blur trajectory (Laplacian variance drops)
   - Basic motion direction & camera push-in proxies (horizontal energy centroid & center-vs-border contrast shift)
   - ``ffmpeg`` freeze & black-frame event detection
4. Explicit Wan First-and-Last-Frame comparison:
   - Requested starter image vs actual initial frame
   - Requested ending image vs actual final frame
   - Intermediate temporal continuity (NEVER treats valid endpoints as proof intermediate frames are valid!)
5. Targeted escalation: only first, last, midpoint, and top suspicious-interval frames (max 4 frames)
   are escalated to local Qwen3-VL when semantic video inspection is requested.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageChops, ImageOps, ImageStat

from .inspection_contract import (
    AcceptanceCriterion,
    CriterionAssessment,
    GenerationVerificationRecord,
    VisualInspectionResult,
    compute_inspection_status,
    compute_requirement_set_hash,
    get_verification_eligibility,
)
from .integrity import sha256_file
from .ollama_client import (
    DEFAULT_OLLAMA_URL,
    LLMExecutionSchedule,
    LocalOllamaClient,
    select_vision_model,
)
from .realism_qa import _dhash, _ffmpeg_events, _hamming, _laplacian_variance, _median
from .resource_monitor import ResourceSnapshot
from .resource_planner import ResourcePlanner
from .visual_inspector import (
    VLM_INSPECTION_SCHEMA,
    _VLM_INSPECTOR_SYSTEM_PROMPT,
    _color_histogram_24,
    _hist_l1_distance,
    build_correction_recommendations,
    compare_image_to_reference_lightweight,
    detect_image_prompt_injection,
    inspect_media_deterministic,
    sanitize_untrusted_text,
)

# Temporal thresholds calibrated for Wan 2.2 5s (16 fps, 81 frames) and synthetic fixtures.
VIDEO_TEMPORAL_THRESHOLDS: dict[str, float] = {
    "luma_jump_warn": 16.0,
    "luma_jump_fail": 28.0,
    "dhash_jump_warn": 15.0,
    "dhash_jump_fail": 22.0,
    "color_shift_warn": 0.24,
    "color_shift_fail": 0.36,
    "sharpness_drop_ratio_warn": 0.50,
    "static_motion_min": 0.35,
    "endpoint_color_dist_fail": 0.42,
    "endpoint_dhash_fail": 26.0,
}


def sample_representative_video_frames(
    video_path: str | Path,
    out_dir: Path,
    *,
    duration_s: float,
    max_frames: int = 16,
    width: int = 384,
) -> list[dict[str, Any]]:
    """Extract representative frames including exact start (t=0) and near-end frame cleanly via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for video inspection.")
    src = Path(video_path).expanduser().resolve()
    from .safe_output import private_output_directory
    out_dir = private_output_directory(out_dir)

    # Extract evenly spaced samples across the full duration, capping at max_frames.
    dur = max(0.2, float(duration_s or 1.0))
    target_fps = max(0.5, min(6.0, (max_frames - 1) / dur))
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        f"fps={target_fps:.4f},scale={width}:-2",
        "-frames:v",
        str(max_frames),
        str(out_dir / "sample_%04d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    sampled = sorted(out_dir.glob("sample_*.png"))

    # Guarantee an explicit final-frame extraction via sseof so endpoints are never missed!
    last_path = out_dir / "endpoint_last.png"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-sseof",
            "-0.12",
            "-i",
            str(src),
            "-vf",
            f"scale={width}:-2",
            "-update",
            "1",
            "-frames:v",
            "1",
            str(last_path),
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    entries: list[dict[str, Any]] = []
    n_samples = max(len(sampled), 1)
    for idx, fp in enumerate(sampled):
        ts = round((idx / max(n_samples - 1, 1)) * dur, 2) if n_samples > 1 else 0.0
        entries.append({"index": idx, "timestamp_s": ts, "path": fp, "role": "start" if idx == 0 else "sample"})
    if last_path.is_file() and last_path.stat().st_size > 0:
        if entries:
            # Replace or append as explicit ending frame if distinct from last sample.
            entries.append({"index": len(entries), "timestamp_s": round(dur, 2), "path": last_path, "role": "end"})
        else:
            entries.append({"index": 0, "timestamp_s": round(dur, 2), "path": last_path, "role": "end"})
    elif entries:
        entries[-1]["role"] = "end"
    return entries


def _horizontal_motion_centroid(diff_gray: Image.Image) -> float:
    """Return normalized horizontal centroid in [-1.0 (left), +1.0 (right)] of frame motion."""
    w, h = diff_gray.size
    small = diff_gray.resize((32, 18), Image.Resampling.BILINEAR)
    px = list(small.getdata())
    total_w = 0.0
    weighted_x = 0.0
    for y in range(18):
        row_off = y * 32
        for x in range(32):
            v = float(px[row_off + x])
            if v > 6.0:
                total_w += v
                weighted_x += v * ((x / 31.0) * 2.0 - 1.0)
    return round(weighted_x / total_w, 3) if total_w > 0 else 0.0


def analyze_video_temporal_cpu(
    video_path: str | Path,
    *,
    starter_image: str | Path | None = None,
    end_image: str | Path | None = None,
    max_frames: int = 16,
    thresholds: dict[str, float] | None = None,
    frame_dir: Path | None = None,
) -> dict[str, Any]:
    """Run fast CPU-only temporal & endpoint analysis and flag suspicious intervals for targeted escalation."""
    limits = {**VIDEO_TEMPORAL_THRESHOLDS, **(thresholds or {})}
    p = Path(video_path).expanduser().resolve()
    det = inspect_media_deterministic(p)
    if not det["readable"] or det["media_kind"] != "video":
        return {"ok": False, "deterministic": det, "error": "; ".join(det["issues"]) or "Video unreadable."}

    own_tmp = frame_dir is None
    tmp_ctx = tempfile.TemporaryDirectory(prefix="avatar_v2_vidqa_") if own_tmp else None
    target_dir = Path(tmp_ctx.name) if tmp_ctx is not None else Path(frame_dir)  # type: ignore[arg-type]

    try:
        frame_entries = sample_representative_video_frames(
            p, target_dir, duration_s=float(det.get("duration_s") or 1.0), max_frames=max_frames
        )
        if len(frame_entries) < 2:
            return {"ok": False, "deterministic": det, "error": "Too few frames extracted to evaluate temporal continuity."}

        lumas: list[float] = []
        sharpness_vals: list[float] = []
        dhashes: list[int] = []
        hists: list[list[float]] = []
        grays: list[Image.Image] = []
        center_sharpness: list[float] = []

        for entry in frame_entries:
            with Image.open(entry["path"]) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                gray = rgb.convert("L")
                grays.append(gray.copy())
                lumas.append(float(ImageStat.Stat(gray).mean[0]))
                sharpness_vals.append(_laplacian_variance(rgb))
                dhashes.append(_dhash(rgb))
                hists.append(_color_histogram_24(rgb))
                w, h = rgb.size
                center_crop = rgb.crop((int(0.25 * w), int(0.20 * h), int(0.75 * w), int(0.80 * h)))
                center_sharpness.append(_laplacian_variance(center_crop))

        luma_jumps: list[float] = []
        dhash_jumps: list[int] = []
        color_shifts: list[float] = []
        motion_energies: list[float] = []
        motion_centroids: list[float] = []
        suspicious_intervals: list[dict[str, Any]] = []

        sharp_med = _median(sharpness_vals) or 1.0

        _sev_rank = {"none": 0, "warn": 1, "fail": 2}
        def _bump_sev(cur: str, new: str) -> str:
            return new if _sev_rank[new] > _sev_rank[cur] else cur

        for i in range(len(frame_entries) - 1):
            t_a = frame_entries[i]["timestamp_s"]
            t_b = frame_entries[i + 1]["timestamp_s"]
            lj = abs(lumas[i + 1] - lumas[i])
            dj = _hamming(dhashes[i], dhashes[i + 1])
            cs = _hist_l1_distance(hists[i], hists[i + 1])
            diff_im = ImageChops.difference(grays[i], grays[i + 1])
            me = float(ImageStat.Stat(diff_im).mean[0])
            mc = _horizontal_motion_centroid(diff_im)

            luma_jumps.append(round(lj, 2))
            dhash_jumps.append(dj)
            color_shifts.append(cs)
            motion_energies.append(round(me, 2))
            motion_centroids.append(mc)

            reasons: list[str] = []
            severity = "none"
            if lj >= limits["luma_jump_fail"]:
                reasons.append(f"severe luma flicker/flash jump ({lj:.1f})")
                severity = _bump_sev(severity, "fail")
            elif lj >= limits["luma_jump_warn"]:
                reasons.append(f"noticeable luma jump ({lj:.1f})")
                severity = _bump_sev(severity, "warn")

            if dj >= limits["dhash_jump_fail"]:
                reasons.append(f"abrupt structural/scene jump (dHash={dj}/64)")
                severity = _bump_sev(severity, "fail")
            elif dj >= limits["dhash_jump_warn"]:
                reasons.append(f"elevated structural transition (dHash={dj}/64)")
                severity = _bump_sev(severity, "warn")

            if cs >= limits["color_shift_fail"]:
                reasons.append(f"abrupt color/wardrobe/background shift (L1={cs:.2f})")
                severity = _bump_sev(severity, "fail")
            elif cs >= limits["color_shift_warn"]:
                reasons.append(f"moderate color/background shift (L1={cs:.2f})")
                severity = _bump_sev(severity, "warn")

            if sharpness_vals[i + 1] < sharp_med * limits["sharpness_drop_ratio_warn"] and sharp_med > 25.0:
                reasons.append(
                    f"sharpness collapse ({sharpness_vals[i+1]:.1f} vs median {sharp_med:.1f})"
                )
                severity = _bump_sev(severity, "warn")

            if reasons:
                suspicious_intervals.append(
                    {
                        "interval": f"{t_a:.2f}s-{t_b:.2f}s",
                        "frame_indices": [i, i + 1],
                        "severity": severity,
                        "luma_jump": round(lj, 2),
                        "dhash_jump": dj,
                        "color_shift": cs,
                        "reasons": reasons,
                        "escalate_frame_path": str(frame_entries[i + 1]["path"]),
                    }
                )

        # Endpoint comparisons (Wan First-and-Last-Frame / I2V):
        # Explicitly verify start frame vs starter_image AND end frame vs end_image.
        start_comparison: dict[str, Any] | None = None
        end_comparison: dict[str, Any] | None = None
        endpoint_issues: list[str] = []

        if starter_image and Path(starter_image).expanduser().is_file():
            start_comparison = compare_image_to_reference_lightweight(
                starter_image, frame_entries[0]["path"], role="starter_frame"
            )
            if start_comparison.get("ok"):
                if (
                    start_comparison["full_color_distance"] > limits["endpoint_color_dist_fail"]
                    and start_comparison["dhash_distance"] > limits["endpoint_dhash_fail"]
                ):
                    endpoint_issues.append(
                        f"Initial video frame diverges strongly from requested starter image "
                        f"(dHash={start_comparison['dhash_distance']}, color_dist={start_comparison['full_color_distance']})."
                    )

        if end_image and Path(end_image).expanduser().is_file():
            end_comparison = compare_image_to_reference_lightweight(
                end_image, frame_entries[-1]["path"], role="end_keyframe"
            )
            if end_comparison.get("ok"):
                if (
                    end_comparison["full_color_distance"] > limits["endpoint_color_dist_fail"]
                    and end_comparison["dhash_distance"] > limits["endpoint_dhash_fail"]
                ):
                    endpoint_issues.append(
                        f"Final video frame diverges strongly from requested ending keyframe "
                        f"(dHash={end_comparison['dhash_distance']}, color_dist={end_comparison['full_color_distance']})."
                    )

        ffmpeg_events = _ffmpeg_events(p)
        median_motion = _median(motion_energies)
        is_effectively_static = median_motion < limits["static_motion_min"]

        # Select targeted escalation keyframes (never every frame!): start, end, midpoint, and top suspicious frame.
        escalation_indices: list[int] = [0]
        if len(frame_entries) > 2:
            escalation_indices.append(len(frame_entries) // 2)
        if suspicious_intervals:
            top_susp = sorted(
                suspicious_intervals,
                key=lambda item: (1 if item["severity"] == "fail" else 0, item["color_shift"] + item["luma_jump"] / 50.0),
                reverse=True,
            )[0]
            escalation_indices.append(int(top_susp["frame_indices"][1]))
        escalation_indices.append(len(frame_entries) - 1)
        unique_esc = list(dict.fromkeys(i for i in escalation_indices if 0 <= i < len(frame_entries)))[:4]

        escalation_frames = [
            {
                "index": idx,
                "timestamp_s": frame_entries[idx]["timestamp_s"],
                "role": frame_entries[idx]["role"],
                "path": str(frame_entries[idx]["path"]),
            }
            for idx in unique_esc
        ]

        # Intermediate continuity check: even when start & end endpoints pass, intermediate jumps MUST be flagged!
        intermediate_fail_intervals = [s for s in suspicious_intervals if s["severity"] == "fail"]
        temporal_defects: list[str] = list(endpoint_issues)
        if ffmpeg_events.get("black_segments", 0) > 0:
            temporal_defects.append(f"Detected {ffmpeg_events['black_segments']} black-frame segment(s).")
        if ffmpeg_events.get("freezes", 0) > 0:
            temporal_defects.append(f"Detected {ffmpeg_events['freezes']} frozen-frame segment(s).")
        if intermediate_fail_intervals:
            temporal_defects.append(
                f"Detected {len(intermediate_fail_intervals)} abrupt intermediate temporal discontinuity interval(s): "
                + ", ".join(f"{s['interval']} ({'; '.join(s['reasons'])})" for s in intermediate_fail_intervals[:3])
            )

        return {
            "ok": True,
            "deterministic": det,
            "sampled_frame_count": len(frame_entries),
            "max_luma_jump": max(luma_jumps) if luma_jumps else 0.0,
            "max_dhash_jump": max(dhash_jumps) if dhash_jumps else 0,
            "max_color_shift": max(color_shifts) if color_shifts else 0.0,
            "median_motion_energy": round(median_motion, 2),
            "is_effectively_static": is_effectively_static,
            "median_sharpness": round(sharp_med, 1),
            "ffmpeg_events": ffmpeg_events,
            "start_endpoint_comparison": start_comparison,
            "end_endpoint_comparison": end_comparison,
            "suspicious_intervals": suspicious_intervals,
            "escalation_frames": escalation_frames,
            "temporal_defects": temporal_defects,
            "endpoints_verified_only_note": (
                "Endpoint alignment (start/end frames) is evaluated separately from intermediate frame continuity; "
                "valid endpoints never prove intermediate frames are artifact-free."
            ),
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


class VideoInspector:
    """Hybrid local video inspector: CPU temporal/endpoint analysis + targeted keyframe VLM escalation."""

    def __init__(
        self,
        *,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        planner: ResourcePlanner | None = None,
        client: LocalOllamaClient | None = None,
        timeout_s: float = 55.0,
    ) -> None:
        self.ollama_url = ollama_url
        self.planner = planner or ResourcePlanner()
        self.client = client or LocalOllamaClient(base_url=ollama_url, timeout_s=timeout_s)

    def inspect_video(
        self,
        video_path: str | Path,
        record: GenerationVerificationRecord,
        *,
        starter_image: str | Path | None = None,
        end_image: str | Path | None = None,
        use_vlm: bool | Literal["auto"] = "auto",
        preferred_model: str | None = None,
        comfy_models_warm: bool = False,
        active_generation_running: bool = False,
        current_model_id: str | None = None,
        upcoming_model_id: str | None = None,
        snapshot: ResourceSnapshot | None = None,
        allow_sequential_gpu: bool = False,
        batch_queue_depth: int = 0,
        max_frames: int = 16,
    ) -> VisualInspectionResult:
        """Inspect a generated video efficiently (CPU temporal sweep + targeted keyframe VLM escalation)."""
        t0 = time.monotonic()
        p = Path(video_path).expanduser()
        inspected_sha = sha256_file(p) or ""
        inspected_contract = compute_requirement_set_hash(record)
        inspected_refs = {r.asset_id or r.role: sha256_file(r.path) or "" for r in record.reference_requirements}

        methods_used: list[str] = ["deterministic", "video_temporal"]
        limitations: list[str] = list(record.known_unverifiable_requirements)
        injection_warnings: list[str] = []

        # Auto-resolve starter / end images from verification record if not explicitly passed.
        if starter_image is None:
            for ref in record.reference_requirements:
                if ref.role in {"primary_starter", "start_frame"} and Path(ref.path).expanduser().is_file():
                    starter_image = ref.path
                    break
        if end_image is None:
            for ref in [*record.reference_requirements, *record.intermediate_assets]:
                if ref.role in {"end_frame", "end_keyframe"} and Path(ref.path).expanduser().is_file():
                    end_image = ref.path
                    break

        with tempfile.TemporaryDirectory(prefix="avatar_v2_videsc_") as tmp_str:
            temporal = analyze_video_temporal_cpu(
                p,
                starter_image=starter_image,
                end_image=end_image,
                max_frames=max_frames,
                frame_dir=Path(tmp_str),
            )
            det = temporal.get("deterministic") or inspect_media_deterministic(p, expected=record.expected_output)
            detected_defects: list[str] = list(det.get("issues") or [])
            if not temporal.get("ok"):
                detected_defects.append(str(temporal.get("error") or "Video temporal inspection failed."))
                assessments = [
                    CriterionAssessment(
                        criterion_id=c.criterion_id,
                        original_requirement=c.description,
                        category=c.category,
                        mandatory=c.mandatory,
                        inspection_method="deterministic",
                        assessor="deterministic",
                        observed_evidence="; ".join(detected_defects),
                        status="failed",
                        artifact_frame=str(p),
                        failure_description="; ".join(detected_defects),
                    )
                    for c in record.mandatory_acceptance_criteria
                ]
                recs = build_correction_recommendations(assessments, expected_orientation=record.expected_output.expected_orientation)
                decision, ui_status, ui_label, human_req = compute_inspection_status(assessments, detected_defects=detected_defects)
                return VisualInspectionResult(
                    execution_mode="cpu_deterministic_only",
                    overall_decision=decision,
                    ui_status=ui_status,
                    ui_status_label=ui_label,
                    methods_used=methods_used,
                    criterion_assessments=assessments,
                    detected_defects=detected_defects,
                    deterministic_metrics=det,
                    video_summary=temporal,
                    correction_recommendations=recs,
                    limitations=limitations,
                    human_review_required=human_req,
                    latency_ms=round((time.monotonic() - t0) * 1000.0, 1),
                )

            detected_defects.extend(temporal.get("temporal_defects") or [])
            suspicious_intervals = [s["interval"] for s in temporal.get("suspicious_intervals") or []]
            has_fail_temporal = any(s.get("severity") == "fail" for s in temporal.get("suspicious_intervals") or [])

            assessments_by_id: dict[str, CriterionAssessment] = {}
            vlm_criteria: list[AcceptanceCriterion] = []

            for c in record.mandatory_acceptance_criteria:
                cat = c.category.lower()
                desc_low = c.description.lower()
                if cat in {"file_integrity", "dimensions", "aspect_ratio", "video_duration", "frame_rate"}:
                    ok_det = len(det.get("issues") or []) == 0
                    assessments_by_id[c.criterion_id] = CriterionAssessment(
                        criterion_id=c.criterion_id,
                        original_requirement=c.description,
                        category=c.category,
                        mandatory=c.mandatory,
                        inspection_method="deterministic",
                        assessor="deterministic",
                        observed_evidence=f"{det.get('width')}x{det.get('height')} {det.get('fps')}fps {det.get('duration_s')}s",
                        status="passed" if ok_det else "failed",
                        artifact_frame=str(p),
                        failure_description="; ".join(det.get("issues") or []) if not ok_det else "",
                    )
                    continue

                # Temporal flicker / lighting stability / abrupt discontinuity check directly from CPU sweep.
                if (
                    cat in {"temporal_consistency", "temporal_flicker"}
                    and any(tok in desc_low for tok in ("flicker", "lighting", "abrupt", "flash", "freeze", "black"))
                ):
                    if has_fail_temporal or temporal["ffmpeg_events"].get("black_segments", 0) > 0:
                        assessments_by_id[c.criterion_id] = CriterionAssessment(
                            criterion_id=c.criterion_id,
                            original_requirement=c.description,
                            category=c.category,
                            mandatory=c.mandatory,
                            inspection_method="video_temporal",
                            assessor="video_temporal",
                            observed_evidence=(
                                f"max_luma_jump={temporal['max_luma_jump']}, max_dhash_jump={temporal['max_dhash_jump']}, "
                                f"max_color_shift={temporal['max_color_shift']}, suspicious={suspicious_intervals}"
                            ),
                            status="failed",
                            artifact_frame=suspicious_intervals[0] if suspicious_intervals else str(p),
                            failure_description="; ".join(temporal.get("temporal_defects") or ["Temporal jump detected."]),
                        )
                        continue
                    elif not suspicious_intervals:
                        assessments_by_id[c.criterion_id] = CriterionAssessment(
                            criterion_id=c.criterion_id,
                            original_requirement=c.description,
                            category=c.category,
                            mandatory=c.mandatory,
                            inspection_method="video_temporal",
                            assessor="video_temporal",
                            observed_evidence=(
                                f"Smooth across {temporal['sampled_frame_count']} sampled frames "
                                f"(max_luma_jump={temporal['max_luma_jump']}, max_color_shift={temporal['max_color_shift']})."
                            ),
                            status="passed",
                            artifact_frame=str(p),
                        )
                        continue

                # Static video fails explicit motion requirements immediately on CPU!
                if cat in {"video_motion", "camera_movement"} and temporal.get("is_effectively_static"):
                    assessments_by_id[c.criterion_id] = CriterionAssessment(
                        criterion_id=c.criterion_id,
                        original_requirement=c.description,
                        category=c.category,
                        mandatory=c.mandatory,
                        inspection_method="video_temporal",
                        assessor="video_temporal",
                        observed_evidence=f"Video appears static (median_motion_energy={temporal['median_motion_energy']}).",
                        status="failed",
                        artifact_frame=str(p),
                        failure_description="Requested motion/camera movement not present; video is nearly static.",
                    )
                    continue

                vlm_criteria.append(c)

            # Targeted keyframe VLM escalation (max 4 representative/suspicious frames, NEVER every frame!).
            need_vlm = bool(vlm_criteria) and use_vlm is not False
            sched_plan = self.planner.plan_visual_inspection(
                requires_vlm=need_vlm,
                active_generation_running=active_generation_running,
                comfy_models_warm=comfy_models_warm,
                current_model_id=current_model_id,
                upcoming_model_id=upcoming_model_id,
                snapshot=snapshot,
                allow_sequential_gpu=allow_sequential_gpu,
                batch_queue_depth=batch_queue_depth,
            )
            execution_mode = str(sched_plan["execution_mode"])
            inspector_model = ""
            vram_released_verified: bool | None = None

            if need_vlm and execution_mode in {"cpu_vlm", "sequential_gpu_vlm"}:
                available = self.client.list_models()
                chosen = select_vision_model(
                    available,
                    preferred=preferred_model,
                    lightweight_only=bool(sched_plan.get("prefer_lightweight_model")),
                )
                if chosen is None:
                    limitations.append("No local vision model available in Ollama; semantic video criteria remain uncertain.")
                    execution_mode = "cpu_deterministic_only"
                else:
                    inspector_model = chosen.name
                    ollama_sched = LLMExecutionSchedule(
                        mode="sequential_gpu" if execution_mode == "sequential_gpu_vlm" else "cpu_only",
                        num_gpu=int(sched_plan["num_gpu"]),
                        keep_alive="0s",
                        reason=str(sched_plan["reason"]),
                        verify_vram_release=bool(sched_plan.get("verify_vram_release")),
                        prefer_lightweight_model=bool(sched_plan.get("prefer_lightweight_model")),
                    )
                    esc_frames = temporal.get("escalation_frames") or []
                    img_paths: list[str | Path] = [f["path"] for f in esc_frames]
                    img_roles = [f"Video frame {f['role']} (t={f['timestamp_s']}s)" for f in esc_frames]
                    if starter_image and Path(starter_image).expanduser().is_file() and len(img_paths) < 5:
                        img_paths.append(starter_image)
                        img_roles.append("Requested Starter Image")
                    if end_image and Path(end_image).expanduser().is_file() and len(img_paths) < 6:
                        img_paths.append(end_image)
                        img_roles.append("Requested Ending Keyframe")

                    lines = [
                        f"Video frames provided in chronological order: {', '.join(f'Image {i+1} = {r}' for i, r in enumerate(img_roles))}.",
                        f"Original user request: {record.original_prompt}",
                        f"CPU temporal metrics: max_luma_jump={temporal['max_luma_jump']}, max_dhash_jump={temporal['max_dhash_jump']}, max_color_shift={temporal['max_color_shift']}, suspicious_intervals={suspicious_intervals}.",
                        "IMPORTANT: Even if first and last frames match, check intermediate frames for abrupt background swaps, clothing morphing, identity drift, or subject disappearance.",
                        "Evaluate each requirement independently:",
                    ]
                    for c in vlm_criteria:
                        lines.append(f"- [{c.criterion_id}] (category={c.category}, mandatory={c.mandatory}): {c.description}")

                    vlm_res = self.client.complete_vision_json(
                        model=chosen.name,
                        system_prompt=_VLM_INSPECTOR_SYSTEM_PROMPT,
                        user_prompt="\n".join(lines),
                        image_paths=img_paths,
                        schedule=ollama_sched,
                        json_schema=VLM_INSPECTION_SCHEMA,
                        max_tokens=800,
                        verify_unload=bool(sched_plan.get("verify_vram_release")),
                    )
                    if vlm_res.vram_release is not None:
                        vram_released_verified = bool(vlm_res.vram_release.get("released"))
                    if vlm_res.ok and isinstance(vlm_res.data, dict):
                        methods_used.append("local_vlm")
                        for vis_txt in vlm_res.data.get("observed_visible_text") or []:
                            injection_warnings.extend(detect_image_prompt_injection(str(vis_txt)))
                        by_cid = {
                            str(e.get("criterion_id") or "").strip(): e
                            for e in (vlm_res.data.get("criteria") or [])
                            if isinstance(e, dict)
                        }
                        for c in vlm_criteria:
                            entry = by_cid.get(c.criterion_id)
                            if not entry:
                                continue
                            raw_st = str(entry.get("status") or "uncertain").strip().lower()
                            st: Literal["passed", "failed", "uncertain"] = (
                                raw_st if raw_st in {"passed", "failed", "uncertain"} else "uncertain"
                            )
                            obs = sanitize_untrusted_text(str(entry.get("observed_evidence") or ""))
                            lim = sanitize_untrusted_text(str(entry.get("confidence_limitations") or ""))
                            fail_d = sanitize_untrusted_text(str(entry.get("failure_description") or ""))
                            injection_warnings.extend(detect_image_prompt_injection(obs))
                            elig = get_verification_eligibility(c.category)
                            assessments_by_id[c.criterion_id] = CriterionAssessment(
                                criterion_id=c.criterion_id,
                                original_requirement=c.description,
                                category=c.category,
                                mandatory=c.mandatory,
                                verification_eligibility=elig,
                                independent_corroboration=False,
                                inspection_method="local_vlm",
                                assessor="local_vlm",
                                observed_evidence=obs,
                                status=st,
                                artifact_frame=suspicious_intervals[0] if suspicious_intervals else f"t=0.0s..{det.get('duration_s', 0)}s",
                                confidence_limitations=lim,
                                failure_description=fail_d if st == "failed" else "",
                            )
                    else:
                        limitations.append(f"Targeted video VLM escalation failed ({vlm_res.error}); remaining criteria uncertain.")

            ordered: list[CriterionAssessment] = []
            for c in record.mandatory_acceptance_criteria:
                existing = assessments_by_id.get(c.criterion_id)
                if existing is None:
                    existing = CriterionAssessment(
                        criterion_id=c.criterion_id,
                        original_requirement=c.description,
                        category=c.category,
                        mandatory=c.mandatory,
                        inspection_method="video_temporal",
                        assessor="video_temporal",
                        observed_evidence=(
                            f"Sampled {temporal['sampled_frame_count']} frames; semantic motion/continuity check requires VLM or human review."
                        ),
                        status="uncertain",
                        artifact_frame=suspicious_intervals[0] if suspicious_intervals else str(p),
                        confidence_limitations="Keyframes alone cannot conclusively verify fine hand anatomy or subtle motion physics.",
                    )
                ordered.append(existing)

            if not ordered:
                ordered.append(
                    CriterionAssessment(
                        criterion_id="video_temporal_integrity",
                        original_requirement="Video plays cleanly without abrupt temporal discontinuities or endpoint divergence",
                        category="temporal_consistency",
                        mandatory=True,
                        inspection_method="video_temporal",
                        assessor="video_temporal",
                        observed_evidence=(
                            f"{temporal['sampled_frame_count']} frames sampled, max_luma_jump={temporal['max_luma_jump']}, "
                            f"max_color_shift={temporal['max_color_shift']}"
                        ),
                        status="failed" if detected_defects else "uncertain",
                        artifact_frame=suspicious_intervals[0] if suspicious_intervals else str(p),
                        confidence_limitations="" if detected_defects else "CPU temporal checks passed; semantic review recommended.",
                        failure_description="; ".join(detected_defects) if detected_defects else "",
                    )
                )

            # Strip temp paths from video_summary before returning cleanly.
            clean_summary = {
                k: v
                for k, v in temporal.items()
                if k not in {"deterministic"}
            }
            clean_summary["escalation_frames"] = [
                {k: v for k, v in f.items() if k != "path"}
                for f in (temporal.get("escalation_frames") or [])
            ]
            clean_summary["suspicious_intervals"] = [
                {k: v for k, v in s.items() if k != "escalate_frame_path"}
                for s in (temporal.get("suspicious_intervals") or [])
            ]

            supplied_roles = {r.role for r in record.reference_requirements if r.origin != "generated_intermediate"}
            recs = build_correction_recommendations(
                ordered,
                expected_orientation=record.expected_output.expected_orientation,
                supplied_roles=supplied_roles,
                suspicious_intervals=suspicious_intervals,
            )
            rec_by_cid = {r.failed_criterion_id: r.recommended_action for r in recs}
            final_assessments = [
                a.model_copy(update={"recommended_action": rec_by_cid[a.criterion_id]})
                if a.criterion_id in rec_by_cid and not a.recommended_action
                else a
                for a in ordered
            ]
            decision, ui_status, ui_label, human_req = compute_inspection_status(
                final_assessments, detected_defects=detected_defects
            )
            return VisualInspectionResult(
                candidate_id=p.stem,
                output_path=str(p),
                output_sha256=inspected_sha,
                requirement_set_hash=inspected_contract,
                reference_fingerprints=inspected_refs,
                inspector_model=inspector_model,
                execution_mode=execution_mode,
                overall_decision=decision,
                ui_status=ui_status,
                ui_status_label=ui_label,
                methods_used=list(dict.fromkeys(methods_used)),
                criterion_assessments=final_assessments,
                detected_defects=list(dict.fromkeys(detected_defects)),
                deterministic_metrics=det,
                video_summary=clean_summary,
                correction_recommendations=recs,
                limitations=list(dict.fromkeys(limitations)),
                prompt_injection_warnings=list(dict.fromkeys(injection_warnings)),
                human_review_required=human_req,
                latency_ms=round((time.monotonic() - t0) * 1000.0, 1),
                vram_released_verified=vram_released_verified,
            ).bind(record, p)
