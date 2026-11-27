from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from . import cloud_ledger
from .errors import classify_exception
from .gpu_memory import GPUMemoryManager
from .integrity import hash_files, sha256_file
from .models import ProviderConfig, RenderJob
from .ownership import DirectoryWriterLock, read_json_safe, register_managed_asset
from .providers import ComfyUIProvider, RunwayProvider
from .providers.comfyui import ComfyUICancelledError, ComfyUIExecutionError
from .resource_monitor import check_storage_for_job, estimate_job_storage_bytes
from .resource_planner import ResourcePlanner, WorkloadRequest

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
JobPhase = Literal[
    "queued",
    "preparing",
    "submitted",
    "rendering",
    "output_received",
    "inspecting",
    "correcting",
    "persisting",
    "complete",
    "cancelled",
    "failed",
    "recovery_required",
]
POST_SUBMIT_PHASES = {"submitted", "rendering", "output_received", "inspecting", "correcting", "persisting"}
MEMORY_SENSITIVE_PROFILES = {"int8_12gb", "low_memory", "low_memory_quality"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}


def _uses_local_gpu(provider: Any) -> bool:
    """Local renderers (ComfyUI) need VRAM management; cloud providers declare uses_local_gpu = False."""
    return bool(getattr(provider, "uses_local_gpu", True))


def _keeps_image_models_loaded(item: "QueuedRender") -> bool:
    """FLUX.2 photo jobs keep their models loaded between runs so the next image replays fast.

    Only the image path (builtin:flux2_txt2img) opts into this: the Wan video weights are far
    larger (8.75 GB) and keep their unload-after-render behavior.
    """
    return item.workflow == "builtin:flux2_txt2img"


def write_durable(path: Path, text: str) -> None:
    """Atomically replace ``path`` via unique temp file + fsync, so crashes/concurrent writes never corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass  # directory fsync is unsupported on some platforms
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _snapshot_submitted_params(item: "QueuedRender", job: RenderJob) -> dict[str, Any]:
    """Capture the actual effective graph parameters submitted to the renderer (RC-13)."""
    amap = dict(job.asset_map or {})
    shot = job.shot

    def _int_opt(*keys: str, default: int) -> int:
        for k in keys:
            val = amap.get(k)
            if val is not None and str(val).strip() != "":
                try:
                    return int(float(val))
                except (TypeError, ValueError):
                    continue
        return default

    seed = _int_opt("SEED", "WAN_SEED", default=int(shot.seed))
    width = _int_opt("WIDTH", "WAN_WIDTH", default=int(item.runtime.get("width") or shot.width))
    height = _int_opt("HEIGHT", "WAN_HEIGHT", default=int(item.runtime.get("height") or shot.height))
    frames = _int_opt("FRAMES", "WAN_LENGTH", default=int(item.runtime.get("frames") or shot.frames))
    steps = amap.get("STEPS") if amap.get("STEPS") is not None else item.runtime.get("steps")
    cfg = amap.get("CFG") if amap.get("CFG") is not None else item.runtime.get("cfg")
    model_file = (
        amap.get("FORGE_UNET")
        or amap.get("WAN_HIGH_MODEL")
        or amap.get("WAN_DIFFUSION_MODEL")
        or item.runtime.get("model")
    )
    wf_sha = ""
    if item.workflow and not str(item.workflow).startswith(("builtin:", "runway:")):
        wf_p = Path(str(item.workflow)).expanduser()
        if wf_p.is_file():
            wf_sha = sha256_file(wf_p) or ""
    ref_keys = ("INIT_IMAGE", "LAST_FRAME", "SOURCE_IMAGE", "MASK_IMAGE", "REF_IMAGE_1", "REF_IMAGE_2", "REF_IMAGE_3", "REF_IMAGE_4")
    staged_refs = {k: str(amap[k]) for k in ref_keys if amap.get(k)}
    return {
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "cfg": cfg,
        "model": str(model_file) if model_file is not None else None,
        "workflow": str(item.workflow or ""),
        "workflow_sha256": wf_sha,
        "staged_references": staged_refs,
    }


def _probe_output_media(outputs: list[str]) -> list[dict[str, Any]]:
    """Probe actual saved PNG/JPG/MP4 artifacts on disk via PIL / ffprobe (RC-13)."""
    from PIL import Image, ImageOps

    probed: list[dict[str, Any]] = []
    for raw in outputs:
        p = Path(str(raw)).expanduser()
        entry: dict[str, Any] = {"path": str(p), "exists": p.is_file()}
        if not p.is_file():
            probed.append(entry)
            continue
        try:
            entry["size_bytes"] = p.stat().st_size
            entry["sha256"] = sha256_file(p) or ""
        except OSError:
            pass
        suffix = p.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            entry["kind"] = "video"
            try:
                from .qa import ffprobe_video

                info = ffprobe_video(p)
                entry["width"] = int(info.get("width") or 0)
                entry["height"] = int(info.get("height") or 0)
                nb_frames = info.get("nb_frames")
                if nb_frames and str(nb_frames).isdigit():
                    entry["frames"] = int(nb_frames)
                dur = float(info.get("duration") or 0.0)
                entry["duration_s"] = round(dur, 3)
                rate = str(info.get("avg_frame_rate") or info.get("r_frame_rate") or "")
                num, _, den = rate.partition("/")
                if den and float(den):
                    entry["fps"] = round(float(num) / float(den), 3)
                elif rate:
                    entry["fps"] = round(float(rate), 3)
                if "frames" not in entry and entry.get("fps") and dur > 0:
                    entry["frames"] = int(round(float(entry["fps"]) * dur))
            except Exception:
                pass
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            entry["kind"] = "image"
            try:
                with Image.open(p) as im:
                    oriented = ImageOps.exif_transpose(im)
                    w, h = oriented.size
                    entry["width"] = int(w)
                    entry["height"] = int(h)
                    entry["frames"] = 1
                    entry["format"] = str(im.format or suffix.lstrip(".").upper())
            except Exception:
                pass
        else:
            entry["kind"] = "file"
        probed.append(entry)
    return probed


def _cloud_memory_event(reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "skipped": True,
        "cleanup_requested": False,
        "detail": "Cloud provider job: no local GPU work, so local VRAM cleanup is not applicable.",
    }


@dataclass
class QueuedRender:
    job: RenderJob
    workflow: str
    provider_config: ProviderConfig
    profile: str
    runtime: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    archive_dir: str | None = None
    chain_from_previous: bool = False
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: JobStatus = "queued"
    phase: JobPhase = "queued"
    error: str = ""
    error_category: str = ""
    error_message: str = ""
    prompt_id: str | None = None
    outputs: list[str] = field(default_factory=list)
    partial_outputs: list[str] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    gpu_trace: dict[str, Any] = field(default_factory=dict)
    candidate_qa: dict[str, Any] = field(default_factory=dict)
    cloud_receipt: dict[str, Any] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    submitted_params: dict[str, Any] = field(default_factory=dict)
    observed_media: list[dict[str, Any]] = field(default_factory=list)
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        shot = self.job.shot
        sub = self.submitted_params or {}
        obs0 = next((m for m in (self.observed_media or []) if m.get("exists") and m.get("width")), {})
        eff_seed = sub.get("seed") if sub.get("seed") is not None else shot.seed
        eff_w = obs0.get("width") or sub.get("width") or self.runtime.get("width") or shot.width
        eff_h = obs0.get("height") or sub.get("height") or self.runtime.get("height") or shot.height
        eff_frames = obs0.get("frames") or sub.get("frames") or self.runtime.get("frames") or shot.frames
        return {
            "id": self.id,
            "job_id": self.job.job_id,
            "label": self.label or shot.shot_id,
            "engine": self.job.selected_engine,
            "provider": self.provider_config.provider,
            "cloud_receipt": self.cloud_receipt,
            "cloud_estimate": self.runtime.get("estimate_credits") if self.provider_config.provider != "comfyui" else None,
            "output_hashes": self.output_hashes,
            "submitted_params": self.submitted_params,
            "observed_media": self.observed_media,
            "candidate_slot": self.runtime.get("candidate_slot"),
            "studio_action": self.runtime.get("studio_action"),
            "profile": self.profile,
            "seed": eff_seed,
            "status": self.status,
            "phase": self.phase,
            "error": self.error,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "prompt_id": self.prompt_id,
            "outputs": self.outputs,
            "partial_outputs": self.partial_outputs,
            "memory_events": self.memory_events,
            "gpu_trace": self.gpu_trace,
            "candidate_qa": self.candidate_qa,
            "verification_record": self.runtime.get("verification_record"),
            "chain_from_previous": self.chain_from_previous,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                ((self.finished_at or time.time()) - self.started_at)
                if self.started_at is not None
                else 0.0
            ),
            "avatar": self.job.avatar.display_name,
            "prompt": self.job.prompt[:400],
            "model": self.runtime.get("model") or sub.get("model"),
            "workflow": Path(self.workflow).name if self.workflow else "",
            "resolution": [int(eff_w), int(eff_h)],
            "frames": int(eff_frames),
            "duration_s": obs0.get("duration_s") or shot.duration_s,
            "steps": sub.get("steps") if sub.get("steps") is not None else self.runtime.get("steps"),
            "cfg": sub.get("cfg") if sub.get("cfg") is not None else self.runtime.get("cfg"),
        }


class RenderQueue:
    def __init__(
        self,
        work_dir: Path,
        memory_manager: GPUMemoryManager | None = None,
        history_dir: Path | None = None,
        candidate_qa_runner: Callable[[QueuedRender], dict[str, Any]] | None = None,
        ledger_path: Path | None = None,
        engine_restart: Callable[[], Any] | None = None,
        managed_assets_path: Path | None = None,
        on_job_finished: Callable[[QueuedRender], None] | None = None,
        *,
        acquire_lock: bool = True,
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._dir_lock: DirectoryWriterLock | None = None
        if acquire_lock:
            self._dir_lock = DirectoryWriterLock(self.work_dir / ".queue.lock")
            self._dir_lock.acquire()
        self._ledger_path = Path(ledger_path) if ledger_path else self.work_dir.parent / "cloud_ledger.jsonl"
        self._managed_assets_path = (
            Path(managed_assets_path) if managed_assets_path else self.work_dir.parent / "managed_assets.json"
        )
        self._on_job_finished = on_job_finished
        self.history_dir = Path(history_dir) if history_dir else self.work_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.memory_manager = memory_manager or GPUMemoryManager.from_environment()
        self.candidate_qa_runner = candidate_qa_runner
        # When a local render dies because the engine crashed or CUDA broke, the queue asks the app
        # to restart its own ComfyUI process once, then retries that shot. Never a process kill:
        # engine_restart is the app's idempotent engine_autostart, which only launches its own engine.
        self._engine_restart = engine_restart
        self._items: list[QueuedRender] = []
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_provider: ComfyUIProvider | None = None
        self._active_id: str | None = None
        self._last_local_workflow: str | None = None
        self._last_local_model_id: str | None = None
        self._last_local_failed: bool = False
        self.resource_planner = ResourcePlanner()
        self._restore()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="avatar-v2-render-queue")
        self._thread.start()

    def get_cancel_event(self, item_id: str) -> threading.Event:
        with self._lock:
            events = getattr(self, "_cancel_events", None)
            if events is None:
                events = {}
                self._cancel_events = events
            ev = events.get(item_id)
            if ev is None:
                ev = threading.Event()
                events[item_id] = ev
            return ev

    def add(self, item: QueuedRender) -> str:
        with self._lock:
            self._items.append(item)
            self._persist()
        self._wake.set()
        return item.id

    def add_many(self, items: list[QueuedRender]) -> list[str]:
        with self._lock:
            self._items.extend(items)
            self._persist()
        self._wake.set()
        return [item.id for item in items]

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [item.public() for item in self._items]

    def memory_status(self) -> dict[str, Any]:
        return self.memory_manager.status()

    def clear_finished(self) -> int:
        with self._lock:
            before = len(self._items)
            self._items = [item for item in self._items if item.status in {"queued", "running"}]
            removed = before - len(self._items)
            self._persist()
            return removed

    @staticmethod
    def _interrupt_provider(provider: Any, prompt_id: str | None) -> None:
        interrupt = getattr(provider, "interrupt", None)
        if not callable(interrupt):
            return
        try:
            sig = inspect.signature(interrupt)
            params = sig.parameters
            if "prompt_id" in params:
                kwargs: dict[str, Any] = {"prompt_id": prompt_id}
                if "allow_global" in params:
                    kwargs["allow_global"] = False
                interrupt(**kwargs)
                return
        except (TypeError, ValueError):
            pass
        interrupt()

    def cancel(self, item_id: str) -> bool:
        with self._lock:
            for item in self._items:
                if item.id != item_id:
                    continue
                if item.status == "queued":
                    self.get_cancel_event(item.id).set()
                    item.status = "cancelled"
                    item.phase = "cancelled"
                    item.finished_at = time.time()
                    self._persist()
                    self._persist_history(item)
                    return True
                if item.status == "running":
                    ev = self.get_cancel_event(item.id)
                    ev.set()
                    prev_phase = item.phase
                    item.runtime["cancelled_during_phase"] = prev_phase
                    if self._active_id == item.id and self._active_provider is not None:
                        try:
                            self._interrupt_provider(self._active_provider, item.prompt_id)
                        except Exception:
                            pass
                    item.status = "cancelled"
                    item.phase = "cancelled"
                    item.finished_at = time.time()
                    try:
                        self._persist()
                        self._persist_history(item)
                    except OSError:
                        pass
                    return True
                return False
        return False

    def _release_dir_lock(self) -> None:
        lock = getattr(self, "_dir_lock", None)
        if lock is not None:
            lock.release()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        self.memory_manager.stop_trace()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # RC-18: NEVER release the single-writer directory lock while the worker thread
        # is still alive (e.g. finishing QA/persistence); _worker's outer finally releases it.
        if thread is None or not thread.is_alive():
            self._release_dir_lock()

    def _persist(self) -> None:
        path = self.work_dir / "queue_state.json"
        write_durable(path, json.dumps([item.public() for item in self._items], indent=2))
        items_dir = self.work_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        for item in self._items:
            if item.status in {"queued", "running"}:
                write_durable(
                    items_dir / f"{item.id}.json",
                    json.dumps(self._item_to_dict(item), indent=2),
                )

    def _item_to_dict(self, item: QueuedRender) -> dict[str, Any]:
        return {
            "id": item.id,
            "job": item.job.model_dump_json(),
            "workflow": item.workflow,
            "provider_config": item.provider_config.model_dump(),
            "profile": item.profile,
            "runtime": item.runtime,
            "output_dir": item.output_dir,
            "archive_dir": item.archive_dir,
            "chain_from_previous": item.chain_from_previous,
            "label": item.label,
            "status": item.status,
            "phase": item.phase,
            "error": item.error,
            "error_category": item.error_category,
            "error_message": item.error_message,
            "prompt_id": item.prompt_id,
            "outputs": item.outputs,
            "partial_outputs": item.partial_outputs,
            "memory_events": item.memory_events,
            "gpu_trace": item.gpu_trace,
            "candidate_qa": item.candidate_qa,
            "cloud_receipt": item.cloud_receipt,
            "output_hashes": item.output_hashes,
            "submitted_params": item.submitted_params,
            "observed_media": item.observed_media,
            "queued_at": item.queued_at,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
        }

    def _item_from_dict(self, data: dict[str, Any]) -> QueuedRender:
        status = data.get("status", "queued")
        raw_phase = str(data.get("phase") or "")
        phase: str = raw_phase or ("queued" if status == "queued" else "preparing" if status == "running" else "complete" if status == "done" else "cancelled" if status == "cancelled" else "failed")
        provider = (data.get("provider_config") or {}).get("provider", "comfyui")
        submission = str(((data.get("cloud_receipt") or {}).get("submission") or {}).get("state") or "")
        if status == "running" and provider != "comfyui" and data.get("prompt_id"):
            # The cloud task was already created (and may be billing). Reconcile it by polling
            # the existing task on the next run instead of failing it or creating a duplicate.
            status = "queued"
            phase = "queued"
            receipt = dict(data.get("cloud_receipt") or {})
            receipt["reconcile_after_restart"] = True
            data["cloud_receipt"] = receipt
        elif status == "running" and provider != "comfyui" and submission != "preparing":
            # The app stopped while (or possibly after) sending the create request. Runway may have
            # accepted and billed it, so never call it unbilled and never resubmit automatically.
            status = "error"
            phase = "recovery_required"
            receipt = dict(data.get("cloud_receipt") or {})
            receipt.update({"status": "UNCERTAIN", "charge_status": "unknown", "credits_charged": None})
            receipt.setdefault("submission", {})["state"] = "uncertain"
            data["cloud_receipt"] = receipt
            data["error"] = data.get("error") or "The app stopped while submitting this job to Runway."
            data["error_category"] = "cloud_submission_uncertain"
            data["error_message"] = (
                "Runway may have accepted and billed this task before the app stopped. It was not resubmitted; check "
                "your Runway usage, then resolve it under Presenter → Providers & budget."
            )
            data["finished_at"] = data.get("finished_at") or time.time()
        elif status == "running":
            status = "error"
            if provider == "comfyui" and (data.get("prompt_id") or raw_phase in POST_SUBMIT_PHASES):
                # RC-10: Local job already crossed submission to ComfyUI. Never blindly resubmit!
                phase = "recovery_required"
                data["error"] = data.get("error") or (
                    "Local render was interrupted after submission to ComfyUI; inspect the previous output or retry explicitly."
                )
                data["error_category"] = "recovery_required"
                data["error_message"] = data["error"]
            else:
                phase = "failed"
                interrupted = "Render was interrupted by an application restart."
                if provider != "comfyui":
                    interrupted += (
                        " No request had reached Runway's task API, so no generation was billed; queue it again to retry."
                    )
                data["error"] = data.get("error") or interrupted
                data["error_category"] = data.get("error_category") or "render_failed"
                data["error_message"] = data.get("error_message") or data["error"]
            data["finished_at"] = data.get("finished_at") or time.time()
        return QueuedRender(
            id=data["id"],
            job=RenderJob.model_validate_json(data["job"]),
            workflow=data["workflow"],
            provider_config=ProviderConfig.model_validate(data["provider_config"]),
            profile=data.get("profile") or "auto",
            runtime=data.get("runtime") or {},
            output_dir=data.get("output_dir"),
            archive_dir=data.get("archive_dir"),
            chain_from_previous=bool(data.get("chain_from_previous")),
            label=data.get("label") or "",
            status=status,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            error=data.get("error") or "",
            error_category=data.get("error_category") or "",
            error_message=data.get("error_message") or "",
            prompt_id=data.get("prompt_id"),
            outputs=list(data.get("outputs") or []),
            partial_outputs=list(data.get("partial_outputs") or []),
            memory_events=list(data.get("memory_events") or []),
            gpu_trace=dict(data.get("gpu_trace") or {}),
            candidate_qa=dict(data.get("candidate_qa") or {}),
            cloud_receipt=dict(data.get("cloud_receipt") or {}),
            output_hashes=dict(data.get("output_hashes") or {}),
            submitted_params=dict(data.get("submitted_params") or {}),
            observed_media=list(data.get("observed_media") or []),
            queued_at=float(data.get("queued_at") or time.time()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )

    def _restore(self) -> None:
        items_dir = self.work_dir / "items"
        if not items_dir.is_dir():
            return
        for path in sorted(items_dir.glob("*.json")):
            data = read_json_safe(path, None, quarantine=True, expected_type=dict)
            if data is None:
                continue
            try:
                was_running = data.get("status") == "running"
                item = self._item_from_dict(data)
            except (KeyError, OSError, ValueError):
                continue
            self._items.append(item)
            if was_running and item.status == "error":
                if item.provider_config.provider == "comfyui" and item.phase != "recovery_required":
                    # Only pre-submission local renders (no prompt_id yet) safely requeue for one fresh attempt.
                    item.status = "queued"
                    item.phase = "queued"
                    item.error = ""
                    item.error_category = ""
                    item.error_message = ""
                    item.started_at = None
                    item.runtime["retry_after"] = time.time() + 120  # ComfyUI needs about a minute to boot
                    item.memory_events.append({"reason": "interrupted-by-restart-retry", "retry_after_s": 120})
                else:
                    # Post-submission local recovery_required OR interrupted cloud submission:
                    # record durably once and retire the active item file so it cannot blindly resubmit.
                    try:
                        self._record_cloud_final(item)
                        self._persist_history(item)
                    except OSError:
                        pass
        try:
            self._persist()  # keep queue_state.json in step with the restored items (scenes wait on them)
        except OSError:
            pass

    def _drop_item_file(self, item: QueuedRender) -> None:
        path = self.work_dir / "items" / f"{item.id}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _persist_history(self, item: QueuedRender) -> None:
        path = self.history_dir / f"{item.id}.json"
        write_durable(path, json.dumps(item.public(), indent=2))
        self._drop_item_file(item)

    def _next(self) -> QueuedRender | None:
        broken: set[str] = set()
        with self._lock:
            now = time.time()
            statuses = {item.id: item.status for item in self._items}
            for item in self._items:
                if item.status != "queued":
                    continue
                if float(item.runtime.get("retry_after") or 0) > now:
                    continue  # a shot awaiting its engine-restart retry
                chain = str(item.runtime.get("chain_from_id") or "")
                if chain and statuses.get(chain) not in {None, "done"}:
                    if statuses[chain] in {"error", "cancelled"}:
                        broken.add(chain)  # the previous shot died: stop the chain, never wait forever
                    continue  # wait for the previous shot (or its retry) before starting this one
                return item
        for failed_id in broken:
            self._cancel_downstream(failed_id)
        return None

    def _cancel_downstream(self, failed_id: str) -> int:
        """A chained scene stops at its first failure: cancel every queued shot after the failed one.

        Each cancellation is recorded durably (history record + retired item file) so an app
        restart cannot resurrect the shot as "queued" from a stale queue file.
        """
        cancelled: list[QueuedRender] = []
        with self._lock:
            pending = [failed_id]
            changed = True
            while changed:
                changed = False
                for item in self._items:
                    if item.status == "queued" and str(item.runtime.get("chain_from_id") or "") in pending:
                        item.status = "cancelled"
                        item.error = "Scene stopped: the previous shot failed."
                        item.finished_at = time.time()
                        pending.append(item.id)
                        cancelled.append(item)
                        changed = True
        for item in cancelled:
            try:
                self._persist_history(item)
            except OSError:
                pass
        if cancelled:
            self._persist()
        return len(cancelled)

    def _previous_output(self, current: QueuedRender) -> str | None:
        wanted = str(current.runtime.get("chain_from_id") or "")
        if wanted:
            # Explicit chain (scene shots): only that exact item's video, never a fallback to an older one.
            with self._lock:
                for item in self._items:
                    if item.id == wanted:
                        if item.status != "done":
                            return None
                        return next((o for o in item.outputs if Path(o).suffix.lower() in VIDEO_SUFFIXES and Path(o).is_file()), None)
            record_path = self.history_dir / f"{wanted}.json"  # it was cleared from the list: use its record
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if record.get("status") != "done":
                return None
            return next((o for o in record.get("outputs") or [] if Path(str(o)).suffix.lower() in VIDEO_SUFFIXES and Path(str(o)).is_file()), None)
        with self._lock:
            try:
                index = self._items.index(current)
            except ValueError:
                return None
            for item in reversed(self._items[:index]):
                if item.status == "done" and item.outputs:
                    for output in item.outputs:
                        if Path(output).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
                            return output
        return None

    def _extract_last_frame(self, video: str, target: Path) -> Path:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for last-frame chaining")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-0.08", "-i", video, "-frames:v", "1", str(target)],
            check=True,
            capture_output=True,
        )
        return target

    def _resolve_outputs(self, item: QueuedRender, outputs: list[str]) -> list[str]:
        root = item.output_dir or item.provider_config.output_dir
        root_resolved = Path(root).expanduser().resolve() if root else None
        resolved: list[str] = []
        for output in outputs:
            path = Path(output)
            if ".." in path.parts:
                raise ValueError(f"Refusing renderer output path with parent traversal: {output!r}")
            if not path.is_absolute() and root_resolved is not None:
                candidate = (root_resolved / path).resolve()
                if not candidate.is_relative_to(root_resolved):
                    raise ValueError(f"Renderer output path escapes configured output_dir: {output!r}")
                path = candidate
            resolved.append(str(path))
        return resolved

    def _register_managed_outputs(
        self,
        item: QueuedRender,
        paths: list[str],
        *,
        origin: Literal["generated_output", "managed_archive_copy"] = "generated_output",
    ) -> None:
        reg_path = getattr(self, "_managed_assets_path", None)
        if reg_path is None:
            return
        for out_p in paths:
            try:
                register_managed_asset(reg_path, out_p, origin=origin, job_id=item.id)
            except OSError:
                pass

    def _archive_outputs(self, item: QueuedRender) -> list[str]:
        target = Path(item.archive_dir).expanduser() / item.id
        target.mkdir(parents=True, exist_ok=True)
        archived: list[str] = []
        for output in item.outputs:
            src = Path(output)
            if not src.is_file():
                archived.append(output)
                continue
            try:
                dest = target / src.name
                shutil.copy2(src, dest)
                archived.append(str(dest))
                self._register_managed_outputs(item, [str(dest)], origin="managed_archive_copy")
            except OSError:
                archived.append(output)
        return archived

    def _extract_candidate_frame(self, item: QueuedRender) -> None:
        """Turn a stabilized H3 designer clip into an accept/reject still candidate."""
        video = next(
            (
                Path(output)
                for output in item.outputs
                if Path(output).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}
            ),
            None,
        )
        if video is None or not video.is_file():
            raise RuntimeError("H3 Avatar Designer completed without a readable video output")
        target = video.with_name(f"{video.stem}_candidate.png")
        frame = self._extract_last_frame(str(video), target)
        item.outputs.append(str(frame))
        self._register_managed_outputs(item, [str(frame)], origin="generated_output")

    def _persist_item_durable(self, item: QueuedRender) -> None:
        """Durably write one item's record. Raises on failure, so callers can stop safely."""
        write_durable(self.work_dir / "items" / f"{item.id}.json", json.dumps(self._item_to_dict(item), indent=2))
        try:
            self._persist()  # the summary file is best-effort; the item file above is the durable record
        except OSError:
            pass

    def _record_cloud_submission(self, item: QueuedRender, receipt: dict[str, Any]) -> None:
        """Durably note that a create request is about to be sent (raises, aborting the send, on failure)."""
        with self._lock:
            item.cloud_receipt = {**item.cloud_receipt, **dict(receipt)}
            self._persist_item_durable(item)

    def _record_cloud_task(self, item: QueuedRender, task_id: str, receipt: dict[str, Any]) -> None:
        """Durably persist a cloud task id the moment it exists (raises on failure; the provider then cancels)."""
        with self._lock:
            item.prompt_id = task_id
            item.cloud_receipt = {**item.cloud_receipt, **dict(receipt)}
            self._persist_item_durable(item)
        try:
            cloud_ledger.record(
                getattr(self, "_ledger_path", None),
                {
                    "task_id": task_id,
                    "queue_id": item.id,
                    "label": item.label,
                    "provider": item.provider_config.provider,
                    "model": receipt.get("model"),
                    "status": "CREATED",
                    "estimate_credits": (receipt.get("estimate") or {}).get("credits"),
                    "api_estimated_credits": receipt.get("api_estimated_credits"),
                },
            )
        except OSError as exc:  # the task id itself is already durable in the item record
            item.memory_events.append({"reason": "cloud-ledger-write-failed", "error": f"{type(exc).__name__}: {exc}"})

    def _record_cloud_final(self, item: QueuedRender) -> None:
        if item.provider_config.provider == "comfyui":
            return
        receipt = item.cloud_receipt or {}
        task_id = str(receipt.get("task_id") or item.prompt_id or "")
        submission = receipt.get("submission") or {}
        estimate = (receipt.get("estimate") or {}).get("credits")
        if estimate is None:
            estimate = item.runtime.get("estimate_credits")
        base = {
            "queue_id": item.id,
            "label": item.label,
            "provider": item.provider_config.provider,
            "model": receipt.get("model") or item.runtime.get("model"),
            "estimate_credits": estimate,
            "api_estimated_credits": receipt.get("api_estimated_credits"),
            "failure_code": receipt.get("failure_code"),
            "outputs": len(item.outputs),
        }
        if not task_id:
            if submission.get("state") not in {"sending", "uncertain"}:
                return  # nothing reached Runway's task API, so nothing could be billed
            cloud_ledger.record(
                getattr(self, "_ledger_path", None),
                {**base, "request_key": submission.get("request_key") or f"req:{item.id}", "status": "UNCERTAIN",
                 "charge_status": "unknown", "credits_charged": None},
            )
            return
        status = str(receipt.get("status") or "")
        if status not in cloud_ledger.FINAL_STATUSES:
            status = "SUCCEEDED" if item.status == "done" else "CANCELLED" if item.status == "cancelled" else "FAILED"
        charge = receipt.get("charge_status")
        if charge not in {"actual", "none"}:
            charge = "unknown"  # the estimate stays reserved until Runway's charge is known or resolved
        cloud_ledger.record(
            getattr(self, "_ledger_path", None),
            {**base, "task_id": task_id, "status": status, "charge_status": charge,
             "credits_charged": receipt.get("credits_charged") if charge == "actual" else (0 if charge == "none" else None)},
        )

    def _record_local_submitted(self, item: QueuedRender, prompt_id: str) -> None:
        """Persist local ComfyUI prompt_id durably BEFORE waiting so crashes can reconcile safely."""
        with self._lock:
            item.prompt_id = str(prompt_id)
            item.phase = "rendering"
            try:
                self._persist_item_durable(item)
            except OSError as exc:
                item.memory_events.append(
                    {"reason": "local-submit-persist-failed", "error": f"{type(exc).__name__}: {exc}"}
                )

    @staticmethod
    def _call_provider_render(
        provider: Any,
        workflow: str,
        job: RenderJob,
        *,
        on_resolved: Callable[[dict[str, Any]], None] | None = None,
        on_submitted: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        render_fn = provider.render
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(render_fn).parameters
            if "on_resolved" in params and on_resolved is not None:
                kwargs["on_resolved"] = on_resolved
            if "on_submitted" in params and on_submitted is not None:
                kwargs["on_submitted"] = on_submitted
            if "cancel_event" in params and cancel_event is not None:
                kwargs["cancel_event"] = cancel_event
        except (TypeError, ValueError):
            pass
        return render_fn(workflow, job, **kwargs)

    def _handle_failure(self, item: QueuedRender, exc: BaseException) -> None:
        """Record a render failure, then either recover once (local engine crash) or stop the chain."""
        if isinstance(exc, ComfyUICancelledError):
            partial_raw = list(getattr(exc, "partial_outputs", None) or [])
            with self._lock:
                if partial_raw:
                    try:
                        item.partial_outputs = self._resolve_outputs(item, partial_raw)
                    except Exception:
                        item.partial_outputs = partial_raw
                item.status = "cancelled"
                item.phase = "cancelled"
            return

        classified = classify_exception(exc)
        partial_raw = list(getattr(exc, "partial_outputs", None) or [])
        with self._lock:
            receipt = getattr(exc, "receipt", None)
            if isinstance(receipt, dict) and receipt:
                item.cloud_receipt = {**item.cloud_receipt, **dict(receipt)}
            if partial_raw:
                try:
                    item.partial_outputs = self._resolve_outputs(item, partial_raw)
                except Exception:
                    item.partial_outputs = partial_raw
            if item.status != "cancelled":
                item.status = "error"
                item.phase = "failed"
                item.error = f"{type(exc).__name__}: {exc}"
                item.error_category = classified.category
                item.error_message = classified.message
        # One automatic recovery for local renders when the engine died or CUDA broke:
        # clean the GPU through the safe memory manager, restart our own ComfyUI, retry once.
        if (
            item.provider_config.provider == "comfyui"
            and item.status == "error"
            and int(item.runtime.get("recover_attempts") or 0) < 1
            and item.error_category in {"comfyui_unavailable", "cuda_oom", "cuda_error_999", "timeout"}
        ):
            item.runtime["recover_attempts"] = int(item.runtime.get("recover_attempts") or 0) + 1
            item.memory_events.append({"reason": "engine-crash-recovery", "error": item.error})
            restart_error = None
            if self._engine_restart is not None:
                try:
                    self._engine_restart()
                except Exception as restart_exc:
                    restart_error = f"{type(restart_exc).__name__}: {restart_exc}"
            if restart_error:
                item.memory_events.append({"reason": "engine-restart-failed", "error": restart_error})
            # Wait for the restarted engine to actually answer before the retry can run, so the
            # single retry isn't wasted against a still-booting ComfyUI. (A healthy HTTP server
            # counts as ready; the render itself is re-submitted from scratch.)
            ready = False
            wait_s = float(getattr(self, "engine_ready_wait_s", 240.0))
            deadline = time.monotonic() + wait_s
            probe = self._make_provider(item.provider_config)
            while time.monotonic() < deadline and not self._stop.is_set():
                try:
                    probe.health()
                    ready = True
                    break
                except Exception:
                    time.sleep(2.0)
            item.runtime["retry_after"] = time.time() + (30.0 if ready else 45.0)
            item.memory_events.append({"reason": "engine-ready-wait", "ready": ready})
            cleanup = self.memory_manager.maybe_cleanup(
                self._active_provider,
                reason="pre-retry",
                force=True,
                unload_models=True,
            )
            item.memory_events.append(cleanup)
            with self._lock:
                item.status = "queued"
                item.phase = "queued"
                item.prompt_id = None
                item.error = None
                item.error_category = None
                item.error_message = None
                item.started_at = None
            try:
                self._persist()
            except OSError:
                pass
            self._wake.set()
        elif item.provider_config.provider == "comfyui" and item.status == "error":
            self._cancel_downstream(item.id)  # a scene stops at its first broken shot

    def _make_provider(self, config: ProviderConfig) -> Any:
        # Resolve the module-level provider names at call time so tests can substitute fakes.
        if config.provider == "runway":
            return RunwayProvider(config)
        return ComfyUIProvider(config)

    def _execute(self, item: QueuedRender) -> None:
        job = copy.deepcopy(item.job)
        if item.chain_from_previous:
            previous = self._previous_output(item)
            if not previous:
                error = RuntimeError("No successful previous video is available for chaining")
                error.error_category = "scene_chain_broken"  # type: ignore[attr-defined]
                error.error_message = (
                    "The previous shot's video is no longer available (it was interrupted or "
                    "deleted), so the scene stopped. Start the scene again."
                )
                raise error
            frame = self._extract_last_frame(previous, self.work_dir / "chains" / f"{item.id}.png")
            job.asset_map["INIT_IMAGE"] = str(frame)
            if job.selected_engine == "h3_ref2va" and "H3_PICTURE_1" in job.asset_map:
                job.asset_map["H3_PICTURE_1"] = str(frame)

        provider = self._make_provider(item.provider_config)
        self._active_provider = provider
        self._active_id = item.id

        if _uses_local_gpu(provider):
            planner: ResourcePlanner | None = getattr(self, "resource_planner", None)
            if planner is None:
                planner = ResourcePlanner()
                self.resource_planner = planner

            model_id = planner.registry.resolve_model_id(item.workflow, item.runtime, job.asset_map)
            upcoming_model_id = self._peek_next_local_model_id(exclude_id=item.id)
            storage_guard_on = getattr(self, "storage_guard_enabled", True)

            force_preclean = item.profile in MEMORY_SENSITIVE_PROFILES
            last_workflow = getattr(self, "_last_local_workflow", None)
            switched_workflow = bool(last_workflow and last_workflow != item.workflow)
            respect_retention = False

            if model_id is not None:
                req = WorkloadRequest(
                    model_id=model_id,
                    workflow=item.workflow,
                    width=item.job.shot.width,
                    height=item.job.shot.height,
                    frames=item.job.shot.frames,
                    local_only=True,
                    requires_vlm_qa=bool(item.runtime.get("candidate_qa")),
                    output_dir=item.output_dir or item.provider_config.output_dir or str(self.work_dir),
                    explicit_vram_mode=str(item.runtime.get("vram_mode") or "") or None,
                )
                dummy_storage = None
                if not storage_guard_on:
                    from .resource_monitor import StorageCheck

                    dummy_storage = StorageCheck(True, "normal", 100.0, 0.0, 100.0, 15.0)
                current_model_id = getattr(self, "_last_local_model_id", None)
                if current_model_id is None and last_workflow:
                    current_model_id = planner.registry.resolve_model_id(last_workflow)
                plan = planner.plan_execution(
                    req,
                    current_model_id=current_model_id,
                    upcoming_model_id=upcoming_model_id,
                    prior_failed_or_cancelled=bool(getattr(self, "_last_local_failed", False)),
                    storage_check=dummy_storage,
                )
                item.runtime["execution_plan"] = plan.public()
                if not plan.admitted:
                    item.memory_events.append({"reason": "resource-planner-blocked", "plan": plan.public()})
                    raise RuntimeError(plan.admission_reason)
                if plan.wan_clip_device == "cpu" and job.asset_map.get("WAN_CLIP_DEVICE") == "default":
                    job.asset_map["WAN_CLIP_DEVICE"] = "cpu"
                if plan.vae_strategy == "tiled" and str(item.workflow).startswith("builtin:"):
                    job.asset_map.setdefault("TILED_VAE", True)
                unload_models = plan.pre_unload_models or (switched_workflow and current_model_id is None)
                respect_retention = plan.respect_retention_above_threshold and not unload_models
                keep_loaded = not unload_models
                if switched_workflow and (plan.transition_kind == "cross_family" or current_model_id is None):
                    force_preclean = True
                reason = (
                    f"pre-render:{item.profile}"
                    + (
                        " · keep image models"
                        if _keeps_image_models_loaded(item) and keep_loaded
                        else " · model switch"
                        if switched_workflow and unload_models
                        else f" · {plan.retention_strategy}"
                        if keep_loaded
                        else ""
                    )
                )
            else:
                if storage_guard_on:
                    kind = "image" if _keeps_image_models_loaded(item) else "video"
                    est = estimate_job_storage_bytes(
                        kind=kind,
                        width=item.job.shot.width,
                        height=item.job.shot.height,
                        frames=item.job.shot.frames,
                    )
                    storage_target = item.output_dir or item.provider_config.output_dir or str(self.work_dir)
                    storage_check = check_storage_for_job(storage_target, estimated_bytes=est)
                    if not storage_check.allowed:
                        item.memory_events.append({"reason": "storage-guard-blocked", **storage_check.public()})
                        raise RuntimeError(storage_check.reason)
                keep_loaded = _keeps_image_models_loaded(item) and not switched_workflow
                unload_models = not keep_loaded
                reason = (
                    f"pre-render:{item.profile}"
                    + (" · keep image models" if keep_loaded else " · model switch" if switched_workflow else "")
                )

            cleanup_kwargs: dict[str, Any] = {
                "reason": reason,
                "force": force_preclean or switched_workflow,
                "unload_models": unload_models,
            }
            if isinstance(self.memory_manager, GPUMemoryManager):
                cleanup_kwargs["respect_retention"] = respect_retention
            item.memory_events.append(self.memory_manager.maybe_cleanup(provider, **cleanup_kwargs))
            self._last_local_workflow = item.workflow
            self._last_local_model_id = model_id
            self._last_local_failed = False

            self.memory_manager.start_trace(item.id)
            cancel_event = self.get_cancel_event(item.id)
            try:
                with self._lock:
                    item.submitted_params = _snapshot_submitted_params(item, job)
                    item.runtime["submitted_params"] = item.submitted_params
                    item.phase = "submitted"
                    item.runtime["submission_token"] = uuid.uuid4().hex
                    # RC-10: Persist submission transition durably BEFORE contacting the renderer
                    # so a crash after ComfyUI accepts the prompt (before on_submitted runs) NEVER
                    # replays accepted work on restart, and durability failure refuses dispatch.
                    self._persist_item_durable(item)
                result = self._call_provider_render(
                    provider,
                    item.workflow,
                    job,
                    on_submitted=lambda pid: self._record_local_submitted(item, pid),
                    cancel_event=cancel_event,
                )
            finally:
                item.gpu_trace = self.memory_manager.stop_trace()
        else:
            item.memory_events.append(_cloud_memory_event(f"pre-render:{item.profile}"))
            item.submitted_params = _snapshot_submitted_params(item, job)
            if item.prompt_id:
                job.asset_map["RUNWAY_RESUME_TASK_ID"] = item.prompt_id
            else:
                # Durable marker: until the provider reports "sending", nothing can have been billed.
                with self._lock:
                    item.cloud_receipt = {**item.cloud_receipt, "submission": {"state": "preparing"}}
                    self._persist_item_durable(item)
            if hasattr(provider, "on_submission_started"):
                provider.on_submission_started = lambda receipt: self._record_cloud_submission(item, receipt)
            if hasattr(provider, "on_task_created"):
                provider.on_task_created = lambda task_id, receipt: self._record_cloud_task(item, task_id, receipt)
            result = provider.render(item.workflow, job)

        with self._lock:
            item.prompt_id = result["prompt_id"]
            item.phase = "output_received"
        if result.get("cloud_receipt"):
            item.cloud_receipt = {**item.cloud_receipt, **dict(result["cloud_receipt"])}
        item.outputs = self._resolve_outputs(item, result["outputs"])
        if item.archive_dir:
            item.outputs = self._archive_outputs(item)
        else:
            self._register_managed_outputs(item, item.outputs, origin="generated_output")
        if item.runtime.get("extract_candidate_frame"):
            self._extract_candidate_frame(item)
        item.output_hashes = hash_files(item.outputs)
        item.observed_media = _probe_output_media(item.outputs)

    def _run_candidate_qa(self, item: QueuedRender, *, render_succeeded: bool = True) -> None:
        has_hook = bool(item.runtime.get("candidate_qa") or item.runtime.get("verification_record"))
        if not render_succeeded or item.status in {"error", "cancelled"} or not has_hook or self.candidate_qa_runner is None:
            return
        if self.get_cancel_event(item.id).is_set():
            return
        with self._lock:
            item.phase = "inspecting"
        try:
            item.candidate_qa = self.candidate_qa_runner(item)
        except Exception as exc:
            item.candidate_qa = {
                "ok": False,
                "decision": "unavailable",
                "advisory": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _peek_next_local_model_id(self, *, exclude_id: str | None = None) -> str | None:
        planner: ResourcePlanner | None = getattr(self, "resource_planner", None)
        if planner is None:
            return None
        with self._lock:
            for candidate in self._items:
                if candidate.id == exclude_id or candidate.status != "queued":
                    continue
                if candidate.provider_config.provider != "comfyui":
                    continue
                return planner.registry.resolve_model_id(
                    candidate.workflow, candidate.runtime, candidate.job.asset_map
                )
        return None

    def _post_cleanup(self, item: QueuedRender, provider: Any, *, render_succeeded: bool | None = None) -> dict[str, Any]:
        """One post-render memory event coordinated with ResourcePlanner."""
        effective_status = (
            "done"
            if (render_succeeded is True and item.status == "running")
            else item.status
        )
        is_cloud = (
            not _uses_local_gpu(provider)
            if provider is not None
            else item.provider_config.provider != "comfyui"
        )
        if is_cloud:
            return _cloud_memory_event(f"post-render:{effective_status}")
        force_postclean = effective_status in {"error", "cancelled"} or item.profile in {
            "low_memory",
            "low_memory_quality",
        }
        failed = effective_status in {"error", "cancelled"}
        self._last_local_failed = failed

        planner: ResourcePlanner | None = getattr(self, "resource_planner", None)
        respect_retention = False
        if planner is not None and not failed:
            completed_model = planner.registry.resolve_model_id(item.workflow, item.runtime, item.job.asset_map)
            next_model = self._peek_next_local_model_id(exclude_id=item.id)
            post_plan = planner.plan_post_render(
                completed_model_id=completed_model,
                next_queued_model_id=next_model,
                job_status=effective_status,
                requires_vlm_qa=bool(item.runtime.get("candidate_qa") or item.runtime.get("visual_inspection_vlm")),
            )
            keep_loaded = not post_plan["unload_models"]
            respect_retention = bool(post_plan["respect_retention"])
            reason = (
                f"post-render:{effective_status} · keep image models"
                if _keeps_image_models_loaded(item) and keep_loaded
                else post_plan["reason"]
            )
        else:
            keep_loaded = _keeps_image_models_loaded(item) and not failed
            reason = f"post-render:{effective_status}" + (" · keep image models" if keep_loaded else "")

        if not keep_loaded:
            self._last_local_workflow = None
            self._last_local_model_id = None

        cleanup_kwargs: dict[str, Any] = {
            "reason": reason,
            "force": force_postclean,
            "unload_models": not keep_loaded,
        }
        if isinstance(self.memory_manager, GPUMemoryManager):
            cleanup_kwargs["respect_retention"] = respect_retention
        return self.memory_manager.maybe_cleanup(provider, **cleanup_kwargs)

    def _worker(self) -> None:
        try:
            self._worker_loop()
        finally:
            self._release_dir_lock()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            item = self._next()
            if item is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            cancel_event = self.get_cancel_event(item.id)
            with self._lock:
                if item.status == "cancelled" or cancel_event.is_set():
                    continue
                item.status = "running"
                item.phase = "preparing"
                item.started_at = time.time()
                try:
                    self._persist()
                except OSError as exc:
                    item.memory_events.append(
                        {"reason": "queue-start-persist-failed", "error": f"{type(exc).__name__}: {exc}"}
                    )

            render_succeeded = False
            try:
                self._execute(item)
                render_succeeded = True
            except Exception as exc:  # surface exact renderer failures for tuning
                self._handle_failure(item, exc)
            finally:
                provider = self._active_provider
                try:
                    item.memory_events.append(
                        self._post_cleanup(item, provider, render_succeeded=render_succeeded)
                    )
                except Exception as exc:
                    item.memory_events.append(
                        {
                            "reason": "post-render-cleanup-exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                # RC-10: Keep item.status == "running" across QA & correction so cancellation still works
                # and UI never declares completion before inspection/correction & history persistence finish!
                self._run_candidate_qa(item, render_succeeded=render_succeeded)
                if item.outputs:
                    try:
                        item.output_hashes = hash_files(item.outputs)
                        item.observed_media = _probe_output_media(item.outputs)
                    except OSError:
                        pass

                with self._lock:
                    cancelled = item.status == "cancelled" or cancel_event.is_set()
                    if cancelled:
                        item.status = "cancelled"
                        item.phase = "cancelled"
                    elif render_succeeded and item.status == "running":
                        item.phase = "persisting"
                        item.status = "done"
                    elif item.status == "error" and item.phase not in {"failed", "recovery_required"}:
                        item.phase = "failed"

                try:
                    self._record_cloud_final(item)
                except OSError as exc:
                    item.memory_events.append({"reason": "cloud-ledger-write-failed", "error": f"{type(exc).__name__}: {exc}"})

                with self._lock:
                    item.finished_at = time.time()
                    self._active_provider = None
                    self._active_id = None
                    if item.status == "done":
                        item.phase = "complete"
                    persist_err: str | None = None
                    try:
                        self._persist_history(item)
                    except OSError as exc:
                        persist_err = f"{type(exc).__name__}: {exc}"
                        item.memory_events.append({"reason": "history-persist-failed", "error": persist_err})
                        if item.status == "done":
                            item.status = "error"
                            item.phase = "failed"
                            item.error = f"Failed to persist finished render record: {persist_err}"
                            item.error_category = "persistence_failed"
                            item.error_message = item.error
                    try:
                        self._persist()
                    except OSError as exc:
                        item.memory_events.append(
                            {"reason": "queue-persist-failed", "error": f"{type(exc).__name__}: {exc}"}
                        )
                    self._cancel_events.pop(item.id, None)
                on_finished = getattr(self, "_on_job_finished", None)
                if callable(on_finished):
                    try:
                        on_finished(item)
                    except Exception:
                        pass
