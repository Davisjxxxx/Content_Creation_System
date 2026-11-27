"""Photorealism QA: repeatable automated measurements plus a mandatory human review.

The automated metrics are deliberately simple, dependency-light proxies (Pillow + ffmpeg):
they catch regressions and gross defects consistently, but they do not prove realism.
A segment is production-ready only when it is 20-30 s long, the automated checks pass, and a
named reviewer has rated every perceptual criterion. Region boxes assume the presenter's
chest-up, centred framing; pass ``face_box`` to override for other compositions.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .integrity import sha256_file

VERSION = 1
HUMAN_CRITERIA: tuple[tuple[str, str], ...] = (
    ("skin_detail", "Skin detail; no excessive smoothing or plastic sheen"),
    ("eyes_blinking", "Eye direction and natural blinking"),
    ("lip_sync", "Lip synchronization with the voice"),
    ("teeth_mouth", "Teeth and mouth free of artifacts"),
    ("facial_geometry", "Facial geometry stays stable"),
    ("hair", "Hair movement and consistency"),
    ("identity", "No identity drift across frames"),
    ("lighting", "Lighting continuity"),
    ("hands", "Hand and gesture quality"),
    ("av_timing", "Voice and visual timing"),
    ("defects", "No visible generation defects"),
)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "detail_ratio_warn": 0.75,
    "detail_ratio_fail": 0.55,
    "luma_jump_warn": 12.0,
    "luma_jump_fail": 25.0,
    "flicker_spike_warn": 4.0,
    "identity_hash_warn": 14.0,
    "identity_hash_fail": 22.0,
    "geometry_jitter_warn": 10.0,
    "lipsync_corr_warn": 0.30,
    "lipsync_corr_fail": 0.15,
    "lipsync_lag_ms_warn": 120.0,
    "av_duration_gap_warn": 0.25,
    "blink_rate_min": 6.0,
    "blink_rate_max": 45.0,
}
# Fractions of the frame (left, top, right, bottom) for a centred chest-up presenter.
FACE_BOX = (0.30, 0.06, 0.70, 0.56)
EYES_BOX = (0.34, 0.20, 0.66, 0.32)
MOUTH_BOX = (0.40, 0.36, 0.60, 0.50)
PRODUCTION_DURATION_S = (20.0, 30.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, timeout=timeout)


def probe(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required for realism QA; install ffmpeg.")
    proc = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=60)
    data = json.loads(proc.stdout or b"{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not video:
        raise RuntimeError("no video stream found")
    num, _, den = str(video.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": round(fps, 3),
        "video_codec": video.get("codec_name"),
        "duration_s": float(data.get("format", {}).get("duration") or video.get("duration") or 0.0),
        "has_audio": audio is not None,
        "audio_duration_s": float(audio.get("duration") or 0.0) if audio else 0.0,
    }


def _extract_frames(path: Path, folder: Path, fps: float, width: int) -> list[Path]:
    from .safe_output import private_output_directory
    folder = private_output_directory(folder)
    _run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:-2", str(folder / "f_%05d.png")])
    return sorted(folder.glob("f_*.png"))


def _audio_envelope(path: Path, fps: float) -> list[float]:
    proc = _run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
    samples = array("h")
    samples.frombytes(proc.stdout[: len(proc.stdout) // 2 * 2])
    window = max(1, int(16000 / fps))
    envelope = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if chunk:
            envelope.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
    return envelope


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))


def _laplacian_variance(image: Image.Image) -> float:
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    return float(ImageStat.Stat(edges).var[0])


def _dhash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _pearson(a: list[float], b: list[float]) -> float:
    count = min(len(a), len(b))
    if count < 3:
        return 0.0
    a, b = a[:count], b[:count]
    mean_a, mean_b = sum(a) / count, sum(b) / count
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    return cov / math.sqrt(var_a * var_b) if var_a and var_b else 0.0


def best_lag_correlation(signal: list[float], reference: list[float], max_lag: int) -> tuple[float, int]:
    """Highest Pearson correlation between ``signal`` and ``reference`` over integer lags."""
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = _pearson(signal[lag:], reference)
        else:
            corr = _pearson(signal, reference[-lag:])
        if corr > best[0]:
            best = (corr, lag)
    return best


def count_blinks(eye_detail: list[float], fps: float) -> int:
    """Short dips in eye-region detail (eyelids cover the high-contrast iris and lashes)."""
    if len(eye_detail) < 5:
        return 0
    baseline = _median(eye_detail)
    if baseline <= 0:
        return 0
    blinks, run = 0, 0
    max_run = max(1, int(fps * 0.5))
    for value in eye_detail:
        if value < baseline * 0.7:
            run += 1
        else:
            if 1 <= run <= max_run:
                blinks += 1
            run = 0
    return blinks


def _status(value: float, warn: float, fail: float | None = None, *, higher_is_better: bool = False) -> str:
    if higher_is_better:
        if fail is not None and value < fail:
            return "fail"
        return "warn" if value < warn else "pass"
    if fail is not None and value > fail:
        return "fail"
    return "warn" if value > warn else "pass"


def _ffmpeg_events(path: Path) -> dict[str, int]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-vf",
         "freezedetect=n=0.003:d=0.5,blackdetect=d=0.2:pix_th=0.10", "-an", "-f", "null", "-"],
        capture_output=True, timeout=600,
    )
    log = proc.stderr.decode("utf-8", "replace")
    return {"freezes": log.count("freeze_start"), "black_segments": log.count("black_start")}


def analyze(
    video: str | Path,
    *,
    master_image: str | Path | None = None,
    sample_fps: float = 10.0,
    face_box: tuple[float, float, float, float] = FACE_BOX,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the automated part of the suite and return a report awaiting human review."""
    path = Path(str(video)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    info = probe(path)
    metrics: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="avatar_v2_qa_") as tmp:
        frames = _extract_frames(path, Path(tmp), sample_fps, 384)
        if len(frames) < 3:
            raise RuntimeError("the video is too short to analyse")
        faces, eyes, mouths, lumas = [], [], [], []
        for frame_path in frames:
            with Image.open(frame_path) as frame:
                frame = frame.convert("RGB")
                faces.append(_crop(frame, face_box))
                eyes.append(_crop(frame, EYES_BOX))
                mouths.append(_crop(frame, MOUTH_BOX).convert("L"))
                lumas.append(ImageStat.Stat(frame.convert("L")).mean[0])

        detail = [_laplacian_variance(face) for face in faces]
        detail_median = _median(detail)
        if master_image and Path(str(master_image)).is_file():
            with Image.open(str(master_image)) as master:
                master = master.convert("RGB").resize((384, int(384 * master.height / master.width)))
                master_face = _crop(master, face_box)
                master_detail = _laplacian_variance(master_face)
                master_hash = _dhash(master_face)
            ratio = detail_median / master_detail if master_detail else 0.0
            metrics["skin_detail_vs_master"] = {
                "value": round(ratio, 3), "unit": "ratio",
                "status": _status(ratio, limits["detail_ratio_warn"], limits["detail_ratio_fail"], higher_is_better=True),
                "note": "Face-region detail relative to the master still; low values indicate smoothing.",
            }
        else:
            master_hash = None
            metrics["skin_detail_vs_master"] = {"value": None, "status": "n/a", "note": "No master still supplied."}
        metrics["face_detail"] = {"value": round(detail_median, 1), "unit": "laplacian variance", "status": "info"}

        jumps = [abs(b - a) for a, b in zip(lumas, lumas[1:])]
        max_jump = max(jumps) if jumps else 0.0
        metrics["lighting_continuity"] = {
            "value": round(max_jump, 2), "unit": "max luma jump (0-255)",
            "status": _status(max_jump, limits["luma_jump_warn"], limits["luma_jump_fail"]),
            "drift": round(max(lumas) - min(lumas), 2),
        }

        diffs = [ImageStat.Stat(ImageChops.difference(a.convert("L"), b.convert("L"))).mean[0] for a, b in zip(faces, faces[1:])]
        diff_median = _median(diffs) or 1e-6
        spikes = sum(1 for value in diffs if value > diff_median * limits["flicker_spike_warn"] and value > 4)
        metrics["temporal_flicker"] = {
            "value": spikes, "unit": "spike frames",
            "status": "pass" if spikes == 0 else "warn",
            "note": "Frame-to-frame changes far above the clip's median (pops, warps, cuts).",
        }

        hashes = [_dhash(face) for face in faces]
        anchor = master_hash if master_hash is not None else hashes[0]
        drift = [_hamming(anchor, value) for value in hashes]
        metrics["identity_drift"] = {
            "value": max(drift), "unit": "max dHash distance (0-64)",
            "status": _status(float(max(drift)), limits["identity_hash_warn"], limits["identity_hash_fail"]),
            "median": _median([float(d) for d in drift]),
            "anchor": "master still" if master_hash is not None else "first frame",
            "note": "Perceptual-hash proxy; confirm identity by human review.",
        }
        jitter = [_hamming(a, b) for a, b in zip(hashes, hashes[1:])]
        metrics["facial_geometry_stability"] = {
            "value": max(jitter) if jitter else 0, "unit": "max frame-to-frame dHash distance",
            "status": _status(float(max(jitter) if jitter else 0), limits["geometry_jitter_warn"]),
        }

        eye_detail = [_laplacian_variance(eye) for eye in eyes]
        blinks = count_blinks(eye_detail, sample_fps)
        minutes = max(info["duration_s"], len(frames) / sample_fps) / 60
        rate = blinks / minutes if minutes else 0.0
        metrics["blink_rate"] = {
            "value": round(rate, 1), "unit": "blinks per minute (heuristic)", "blinks": blinks,
            "status": "pass" if limits["blink_rate_min"] <= rate <= limits["blink_rate_max"] else "warn",
            "note": "Eyelid-closure heuristic; check eye direction and blink naturalness by eye.",
        }

        mouth_motion = [0.0] + [
            ImageStat.Stat(ImageChops.difference(a, b)).mean[0] for a, b in zip(mouths, mouths[1:])
        ]
        if info["has_audio"]:
            envelope = _audio_envelope(path, sample_fps)
            corr, lag = best_lag_correlation(mouth_motion, envelope, max_lag=int(sample_fps * 0.5))
            lag_ms = lag * 1000.0 / sample_fps
            status = _status(corr, limits["lipsync_corr_warn"], limits["lipsync_corr_fail"], higher_is_better=True)
            if status == "pass" and abs(lag_ms) > limits["lipsync_lag_ms_warn"]:
                status = "warn"
            metrics["lip_sync"] = {
                "value": round(corr, 3), "unit": "mouth-motion vs voice correlation",
                "lag_ms": round(lag_ms), "status": status,
                "note": "Positive lag means the mouth moves after the sound. Proxy only; judge sync by eye and ear.",
            }
            gap = abs(info["duration_s"] - info["audio_duration_s"]) if info["audio_duration_s"] else 0.0
            metrics["av_duration_alignment"] = {
                "value": round(gap, 3), "unit": "seconds",
                "status": "pass" if gap <= limits["av_duration_gap_warn"] else "warn",
            }
        else:
            metrics["lip_sync"] = {"value": None, "status": "n/a", "note": "No audio track."}
            metrics["av_duration_alignment"] = {"value": None, "status": "n/a", "note": "No audio track."}

    events = _ffmpeg_events(path)
    metrics["frozen_or_black_frames"] = {
        "value": events["freezes"] + events["black_segments"], "unit": "events", **events,
        "status": "pass" if not (events["freezes"] or events["black_segments"]) else "warn",
    }

    duration_ok = PRODUCTION_DURATION_S[0] <= info["duration_s"] <= PRODUCTION_DURATION_S[1]
    report = {
        "version": VERSION,
        "file": str(path),
        "sha256": sha256_file(path),
        "measured_at": _now(),
        "probe": info,
        "sample_fps": sample_fps,
        "face_box": list(face_box),
        "master_image": str(master_image) if master_image else None,
        "thresholds": limits,
        "metrics": metrics,
        "human_review": {
            "required": True,
            "complete": False,
            "reviewer": "",
            "criteria": [{"id": key, "label": label, "rating": None, "notes": ""} for key, label in HUMAN_CRITERIA],
        },
    }
    report["production_gate"] = production_gate(report, duration_ok=duration_ok)
    return report


def production_gate(report: dict[str, Any], *, duration_ok: bool | None = None) -> dict[str, Any]:
    if duration_ok is None:
        duration = float((report.get("probe") or {}).get("duration_s") or 0)
        duration_ok = PRODUCTION_DURATION_S[0] <= duration <= PRODUCTION_DURATION_S[1]
    failing = sorted(name for name, metric in (report.get("metrics") or {}).items() if metric.get("status") == "fail")
    review = report.get("human_review") or {}
    ratings = [item.get("rating") for item in review.get("criteria") or []]
    human_ok = bool(review.get("complete")) and all(isinstance(r, int) and r >= 4 for r in ratings)
    reasons = []
    if not duration_ok:
        reasons.append("Test an approximately 20-30 s instructional segment before calling the pipeline production-ready.")
    if failing:
        reasons.append("Automated checks failing: " + ", ".join(failing))
    if not human_ok:
        reasons.append("Human review incomplete or any criterion rated below 4 of 5.")
    return {
        "duration_ok": duration_ok,
        "automated_ok": not failing,
        "human_ok": human_ok,
        "ready": duration_ok and not failing and human_ok,
        "reasons": reasons,
    }


def apply_human_review(report: dict[str, Any], *, reviewer: str, ratings: dict[str, Any], notes: dict[str, str] | None = None) -> dict[str, Any]:
    """Record a reviewer's 1-5 ratings for every criterion and recompute the production gate."""
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise ValueError("Enter the reviewer's name.")
    criteria = []
    for key, label in HUMAN_CRITERIA:
        rating = ratings.get(key)
        if not isinstance(rating, int) or not 1 <= rating <= 5:
            raise ValueError(f"Rate '{label}' from 1 to 5.")
        criteria.append({"id": key, "label": label, "rating": rating, "notes": str((notes or {}).get(key, ""))[:500]})
    updated = dict(report)
    updated["human_review"] = {"required": True, "complete": True, "reviewer": reviewer[:120], "reviewed_at": _now(), "criteria": criteria}
    updated["production_gate"] = production_gate(updated)
    return updated
