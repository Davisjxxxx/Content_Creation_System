from __future__ import annotations

import copy
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
from .integrity import hash_files
from .models import ProviderConfig, RenderJob
from .providers import ComfyUIProvider, RunwayProvider

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
MEMORY_SENSITIVE_PROFILES = {"int8_12gb", "low_memory", "low_memory_quality"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}


def _uses_local_gpu(provider: Any) -> bool:
    """Local renderers (ComfyUI) need VRAM management; cloud providers declare uses_local_gpu = False."""
    return bool(getattr(provider, "uses_local_gpu", True))


def write_durable(path: Path, text: str) -> None:
    """Atomically replace ``path`` and fsync it, so a crash leaves either the old or the new record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
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
    error: str = ""
    error_category: str = ""
    error_message: str = ""
    prompt_id: str | None = None
    outputs: list[str] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    gpu_trace: dict[str, Any] = field(default_factory=dict)
    candidate_qa: dict[str, Any] = field(default_factory=dict)
    cloud_receipt: dict[str, Any] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        shot = self.job.shot
        return {
            "id": self.id,
            "job_id": self.job.job_id,
            "label": self.label or shot.shot_id,
            "engine": self.job.selected_engine,
            "provider": self.provider_config.provider,
            "cloud_receipt": self.cloud_receipt,
            "cloud_estimate": self.runtime.get("estimate_credits") if self.provider_config.provider != "comfyui" else None,
            "output_hashes": self.output_hashes,
            "candidate_slot": self.runtime.get("candidate_slot"),
            "studio_action": self.runtime.get("studio_action"),
            "profile": self.profile,
            "seed": shot.seed,
            "status": self.status,
            "error": self.error,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "prompt_id": self.prompt_id,
            "outputs": self.outputs,
            "memory_events": self.memory_events,
            "gpu_trace": self.gpu_trace,
            "candidate_qa": self.candidate_qa,
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
            "model": self.runtime.get("model"),
            "workflow": Path(self.workflow).name if self.workflow else "",
            "resolution": [shot.width, shot.height],
            "frames": shot.frames,
            "duration_s": shot.duration_s,
            "steps": self.runtime.get("steps"),
            "cfg": self.runtime.get("cfg"),
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
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_path = Path(ledger_path) if ledger_path else self.work_dir.parent / "cloud_ledger.jsonl"
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
        self._active_provider: ComfyUIProvider | None = None
        self._active_id: str | None = None
        self._restore()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="avatar-v2-render-queue")
        self._thread.start()

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

    def cancel(self, item_id: str) -> bool:
        with self._lock:
            for item in self._items:
                if item.id != item_id:
                    continue
                if item.status == "queued":
                    item.status = "cancelled"
                    item.finished_at = time.time()
                    self._persist()
                    self._persist_history(item)
                    return True
                if item.status == "running" and self._active_id == item.id and self._active_provider:
                    try:
                        self._active_provider.interrupt()
                    except Exception:
                        pass
                    item.status = "cancelled"
                    item.finished_at = time.time()
                    self._persist()
                    self._persist_history(item)
                    return True
                return False
        return False

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        self.memory_manager.stop_trace()

    def _persist(self) -> None:
        path = self.work_dir / "queue_state.json"
        path.write_text(json.dumps([item.public() for item in self._items], indent=2), encoding="utf-8")
        items_dir = self.work_dir / "items"
        items_dir.mkdir(parents=True, exist_ok=True)
        for item in self._items:
            if item.status in {"queued", "running"}:
                (items_dir / f"{item.id}.json").write_text(
                    json.dumps(self._item_to_dict(item), indent=2), encoding="utf-8"
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
            "error": item.error,
            "error_category": item.error_category,
            "error_message": item.error_message,
            "prompt_id": item.prompt_id,
            "outputs": item.outputs,
            "memory_events": item.memory_events,
            "gpu_trace": item.gpu_trace,
            "candidate_qa": item.candidate_qa,
            "cloud_receipt": item.cloud_receipt,
            "output_hashes": item.output_hashes,
            "queued_at": item.queued_at,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
        }

    def _item_from_dict(self, data: dict[str, Any]) -> QueuedRender:
        status = data.get("status", "queued")
        provider = (data.get("provider_config") or {}).get("provider", "comfyui")
        submission = str(((data.get("cloud_receipt") or {}).get("submission") or {}).get("state") or "")
        if status == "running" and provider != "comfyui" and data.get("prompt_id"):
            # The cloud task was already created (and may be billing). Reconcile it by polling
            # the existing task on the next run instead of failing it or creating a duplicate.
            status = "queued"
            receipt = dict(data.get("cloud_receipt") or {})
            receipt["reconcile_after_restart"] = True
            data["cloud_receipt"] = receipt
        elif status == "running" and provider != "comfyui" and submission != "preparing":
            # The app stopped while (or possibly after) sending the create request. Runway may have
            # accepted and billed it, so never call it unbilled and never resubmit automatically.
            status = "error"
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
            error=data.get("error") or "",
            error_category=data.get("error_category") or "",
            error_message=data.get("error_message") or "",
            prompt_id=data.get("prompt_id"),
            outputs=list(data.get("outputs") or []),
            memory_events=list(data.get("memory_events") or []),
            gpu_trace=dict(data.get("gpu_trace") or {}),
            candidate_qa=dict(data.get("candidate_qa") or {}),
            cloud_receipt=dict(data.get("cloud_receipt") or {}),
            output_hashes=dict(data.get("output_hashes") or {}),
            queued_at=float(data.get("queued_at") or time.time()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )

    def _restore(self) -> None:
        items_dir = self.work_dir / "items"
        if not items_dir.is_dir():
            return
        for path in sorted(items_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                was_running = data.get("status") == "running"
                item = self._item_from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
            self._items.append(item)
            if was_running and item.status == "error":
                # Interrupted by a restart: record it once (an uncertain cloud submission keeps its
                # estimate reserved in the ledger) and retire the stale file so it cannot reappear.
                try:
                    self._record_cloud_final(item)
                    self._persist_history(item)
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
        path.write_text(json.dumps(item.public(), indent=2), encoding="utf-8")
        self._drop_item_file(item)

    def _next(self) -> QueuedRender | None:
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
                    continue  # wait for the previous shot (or its retry) before starting this one
                return item
        return None

    def _cancel_downstream(self, failed_id: str) -> int:
        """A chained scene stops at its first failure: cancel every queued shot after the failed one."""
        cancelled = 0
        with self._lock:
            pending = [failed_id]
            changed = True
            while changed:
                changed = False
                for item in self._items:
                    if item.status == "queued" and str(item.runtime.get("chain_from_id") or "") in pending:
                        item.status = "cancelled"
                        item.error = "Scene stopped: the previous shot failed."
                        pending.append(item.id)
                        cancelled += 1
                        changed = True
        if cancelled:
            self._persist()
        return cancelled

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
        resolved: list[str] = []
        for output in outputs:
            path = Path(output)
            if not path.is_absolute() and root:
                path = Path(root) / path
            resolved.append(str(path))
        return resolved

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

    def _handle_failure(self, item: QueuedRender, exc: BaseException) -> None:
        """Record a render failure, then either recover once (local engine crash) or stop the chain."""
        classified = classify_exception(exc)
        with self._lock:
            receipt = getattr(exc, "receipt", None)
            if isinstance(receipt, dict) and receipt:
                item.cloud_receipt = {**item.cloud_receipt, **dict(receipt)}
            if item.status != "cancelled":
                item.status = "error"
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
            item.runtime["retry_after"] = time.time() + 45  # let the restarted engine load its models
            item.memory_events.append({"reason": "engine-crash-recovery", "error": item.error})
            restart_error = None
            if self._engine_restart is not None:
                try:
                    self._engine_restart()
                except Exception as restart_exc:
                    restart_error = f"{type(restart_exc).__name__}: {restart_exc}"
            if restart_error:
                item.memory_events.append({"reason": "engine-restart-failed", "error": restart_error})
            cleanup = self.memory_manager.maybe_cleanup(
                self._active_provider,
                reason="pre-retry",
                force=True,
                unload_models=True,
            )
            item.memory_events.append(cleanup)
            with self._lock:
                item.status = "queued"
                item.error = None
                item.error_category = None
                item.error_message = None
                item.started_at = None
            self._persist()
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
                raise RuntimeError("No successful previous video is available for chaining")
            frame = self._extract_last_frame(previous, self.work_dir / "chains" / f"{item.id}.png")
            job.asset_map["INIT_IMAGE"] = str(frame)
            if job.selected_engine == "h3_ref2va" and "H3_PICTURE_1" in job.asset_map:
                job.asset_map["H3_PICTURE_1"] = str(frame)

        provider = self._make_provider(item.provider_config)
        self._active_provider = provider
        self._active_id = item.id

        if _uses_local_gpu(provider):
            force_preclean = item.profile in MEMORY_SENSITIVE_PROFILES
            item.memory_events.append(
                self.memory_manager.maybe_cleanup(
                    provider,
                    reason=f"pre-render:{item.profile}",
                    force=force_preclean,
                    unload_models=True,
                )
            )

            self.memory_manager.start_trace(item.id)
            try:
                result = provider.render(item.workflow, job)
            finally:
                item.gpu_trace = self.memory_manager.stop_trace()
        else:
            item.memory_events.append(_cloud_memory_event(f"pre-render:{item.profile}"))
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

        item.prompt_id = result["prompt_id"]
        if result.get("cloud_receipt"):
            item.cloud_receipt = {**item.cloud_receipt, **dict(result["cloud_receipt"])}
        item.outputs = self._resolve_outputs(item, result["outputs"])
        if item.archive_dir:
            item.outputs = self._archive_outputs(item)
        if item.runtime.get("extract_candidate_frame"):
            self._extract_candidate_frame(item)
        item.output_hashes = hash_files(item.outputs)

    def _run_candidate_qa(self, item: QueuedRender) -> None:
        if (
            item.status != "done"
            or not item.runtime.get("candidate_qa")
            or self.candidate_qa_runner is None
        ):
            return
        try:
            item.candidate_qa = self.candidate_qa_runner(item)
        except Exception as exc:
            item.candidate_qa = {
                "ok": False,
                "decision": "unavailable",
                "advisory": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._next()
            if item is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            with self._lock:
                item.status = "running"
                item.started_at = time.time()
                self._persist()

            try:
                self._execute(item)
                with self._lock:
                    if item.status != "cancelled":
                        item.status = "done"
            except Exception as exc:  # surface exact renderer failures for tuning
                self._handle_failure(item, exc)
            finally:
                provider = self._active_provider
                try:
                    is_cloud = (
                        not _uses_local_gpu(provider)
                        if provider is not None
                        else item.provider_config.provider != "comfyui"
                    )
                    if is_cloud:
                        item.memory_events.append(_cloud_memory_event(f"post-render:{item.status}"))
                    else:
                        force_postclean = item.status in {"error", "cancelled"} or item.profile in {
                            "low_memory",
                            "low_memory_quality",
                        }
                        item.memory_events.append(
                            self.memory_manager.maybe_cleanup(
                                provider,
                                reason=f"post-render:{item.status}",
                                force=force_postclean,
                                unload_models=True,
                            )
                        )
                except Exception as exc:
                    item.memory_events.append(
                        {
                            "reason": "post-render-cleanup-exception",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                self._run_candidate_qa(item)
                try:
                    self._record_cloud_final(item)
                except OSError as exc:
                    item.memory_events.append({"reason": "cloud-ledger-write-failed", "error": f"{type(exc).__name__}: {exc}"})

                with self._lock:
                    item.finished_at = time.time()
                    self._active_provider = None
                    self._active_id = None
                    self._persist()
                    self._persist_history(item)
