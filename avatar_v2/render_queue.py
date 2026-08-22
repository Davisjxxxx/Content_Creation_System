from __future__ import annotations

import copy
import json
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import classify_exception
from .gpu_memory import GPUMemoryManager
from .models import ProviderConfig, RenderJob
from .providers import ComfyUIProvider

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]
MEMORY_SENSITIVE_PROFILES = {"int8_12gb", "low_memory", "low_memory_quality"}


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
    ):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = Path(history_dir) if history_dir else self.work_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.memory_manager = memory_manager or GPUMemoryManager.from_environment()
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
            "queued_at": item.queued_at,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
        }

    def _item_from_dict(self, data: dict[str, Any]) -> QueuedRender:
        status = data.get("status", "queued")
        if status == "running":
            status = "error"
            data["error"] = data.get("error") or "Render was interrupted by an application restart."
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
                self._items.append(self._item_from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue

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
            return next((item for item in self._items if item.status == "queued"), None)

    def _previous_output(self, current: QueuedRender) -> str | None:
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

        provider = ComfyUIProvider(item.provider_config)
        self._active_provider = provider
        self._active_id = item.id

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

        item.prompt_id = result["prompt_id"]
        item.outputs = self._resolve_outputs(item, result["outputs"])
        if item.archive_dir:
            item.outputs = self._archive_outputs(item)

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
                classified = classify_exception(exc)
                with self._lock:
                    if item.status != "cancelled":
                        item.status = "error"
                        item.error = f"{type(exc).__name__}: {exc}"
                        item.error_category = classified.category
                        item.error_message = classified.message
            finally:
                provider = self._active_provider
                try:
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

                with self._lock:
                    item.finished_at = time.time()
                    self._active_provider = None
                    self._active_id = None
                    self._persist()
                    self._persist_history(item)
