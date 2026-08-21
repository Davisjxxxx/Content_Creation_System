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

from .models import ProviderConfig, RenderJob
from .providers import ComfyUIProvider

JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


@dataclass
class QueuedRender:
    job: RenderJob
    workflow: str
    provider_config: ProviderConfig
    profile: str
    output_dir: str | None = None
    chain_from_previous: bool = False
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: JobStatus = "queued"
    error: str = ""
    prompt_id: str | None = None
    outputs: list[str] = field(default_factory=list)
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job.job_id,
            "label": self.label or self.job.shot.shot_id,
            "engine": self.job.selected_engine,
            "profile": self.profile,
            "seed": self.job.shot.seed,
            "status": self.status,
            "error": self.error,
            "prompt_id": self.prompt_id,
            "outputs": self.outputs,
            "chain_from_previous": self.chain_from_previous,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                ((self.finished_at or time.time()) - self.started_at)
                if self.started_at is not None
                else 0.0
            ),
        }


class RenderQueue:
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._items: list[QueuedRender] = []
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._active_provider: ComfyUIProvider | None = None
        self._active_id: str | None = None
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
                    return True
                if item.status == "running" and self._active_id == item.id and self._active_provider:
                    try:
                        self._active_provider.interrupt()
                    except Exception:
                        pass
                    item.status = "cancelled"
                    item.finished_at = time.time()
                    self._persist()
                    return True
                return False
        return False

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    def _persist(self) -> None:
        path = self.work_dir / "queue_state.json"
        path.write_text(json.dumps([item.public() for item in self._items], indent=2), encoding="utf-8")

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
        result = provider.render(item.workflow, job)
        item.prompt_id = result["prompt_id"]
        item.outputs = self._resolve_outputs(item, result["outputs"])

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
                with self._lock:
                    if item.status != "cancelled":
                        item.status = "error"
                        item.error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    item.finished_at = time.time()
                    self._active_provider = None
                    self._active_id = None
                    self._persist()
