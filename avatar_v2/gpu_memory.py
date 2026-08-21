from __future__ import annotations

import gc
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class GPUSnapshot:
    index: int
    name: str
    used_mb: int
    total_mb: int

    @property
    def usage(self) -> float:
        return (self.used_mb / self.total_mb) if self.total_mb else 0.0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage"] = self.usage
        return data


def parse_nvidia_smi_memory(text: str) -> list[GPUSnapshot]:
    snapshots: list[GPUSnapshot] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            snapshots.append(
                GPUSnapshot(
                    index=int(parts[0]),
                    name=parts[1],
                    used_mb=int(float(parts[2])),
                    total_mb=int(float(parts[3])),
                )
            )
        except ValueError:
            continue
    return snapshots


def query_gpu_memory() -> list[GPUSnapshot]:
    """Return memory usage for every NVIDIA GPU visible to nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return parse_nvidia_smi_memory(result.stdout)


def query_gpu_processes() -> list[dict[str, Any]]:
    """List compute processes reported by nvidia-smi without mutating them."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    output: list[dict[str, Any]] = []
    for raw in result.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 3:
            continue
        try:
            output.append({"pid": int(parts[0]), "name": parts[1], "memory_mb": int(float(parts[2]))})
        except ValueError:
            continue
    return output


class GPUMemoryManager:
    """Observe GPU VRAM and request safe cleanup at renderer boundaries.

    This is the production-safe evolution of the user-provided gpu_cleaner.py.
    It deliberately does *not* kill arbitrary Python processes automatically.
    For ComfyUI, cache cleanup is requested through its `/free` endpoint because
    calling torch.cuda.empty_cache() in this desktop controller process cannot
    clear the allocator owned by the separate ComfyUI process.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        check_interval_s: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.threshold = min(max(float(threshold), 0.50), 0.99)
        self.check_interval_s = max(float(check_interval_s), 1.0)
        self.enabled = bool(enabled)
        self._trace_stop = threading.Event()
        self._trace_thread: threading.Thread | None = None
        self._trace_lock = threading.Lock()
        self._trace_label = ""
        self._trace_peak: dict[int, GPUSnapshot] = {}
        self._trace_samples = 0

    @classmethod
    def from_environment(cls) -> "GPUMemoryManager":
        enabled = os.getenv("AVATAR_V2_GPU_AUTOCLEAN", "1").strip().lower() not in {"0", "false", "no", "off"}
        try:
            threshold = float(os.getenv("AVATAR_V2_GPU_THRESHOLD", "0.85"))
        except ValueError:
            threshold = 0.85
        try:
            interval = float(os.getenv("AVATAR_V2_GPU_CHECK_INTERVAL", "5"))
        except ValueError:
            interval = 5.0
        return cls(threshold=threshold, check_interval_s=interval, enabled=enabled)

    def status(self) -> dict[str, Any]:
        snapshots = query_gpu_memory()
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "gpus": [item.public() for item in snapshots],
            "over_threshold": any(item.usage >= self.threshold for item in snapshots),
            "processes": query_gpu_processes(),
        }

    def start_trace(self, label: str) -> None:
        self.stop_trace()
        with self._trace_lock:
            self._trace_label = label
            self._trace_peak = {}
            self._trace_samples = 0
        self._trace_stop.clear()

        def worker() -> None:
            while not self._trace_stop.is_set():
                snapshots = query_gpu_memory()
                with self._trace_lock:
                    self._trace_samples += 1
                    for snap in snapshots:
                        current = self._trace_peak.get(snap.index)
                        if current is None or snap.used_mb > current.used_mb:
                            self._trace_peak[snap.index] = snap
                self._trace_stop.wait(self.check_interval_s)

        self._trace_thread = threading.Thread(target=worker, daemon=True, name="avatar-v2-gpu-trace")
        self._trace_thread.start()

    def stop_trace(self) -> dict[str, Any]:
        self._trace_stop.set()
        thread = self._trace_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(self.check_interval_s + 1.0, 2.0))
        self._trace_thread = None
        with self._trace_lock:
            report = {
                "label": self._trace_label,
                "samples": self._trace_samples,
                "peak": [self._trace_peak[index].public() for index in sorted(self._trace_peak)],
            }
            self._trace_label = ""
            self._trace_peak = {}
            self._trace_samples = 0
        return report

    def maybe_cleanup(
        self,
        provider: Any | None = None,
        *,
        reason: str,
        force: bool = False,
        unload_models: bool = True,
        wait_s: float = 8.0,
    ) -> dict[str, Any]:
        """Clean local Python garbage and, when needed, ask ComfyUI to free VRAM."""
        gc.collect()
        before = query_gpu_memory()
        should_clean = self.enabled and (force or any(item.usage >= self.threshold for item in before))
        requested = False
        error = ""

        if should_clean and provider is not None:
            try:
                provider.free_memory(unload_models=unload_models, free_memory=True)
                requested = True
            except Exception as exc:  # cleanup failure should be recorded, not mask a render result
                error = f"{type(exc).__name__}: {exc}"

        after = before
        if requested:
            deadline = time.monotonic() + max(wait_s, 0.0)
            while time.monotonic() < deadline:
                time.sleep(0.5)
                current = query_gpu_memory()
                if current:
                    after = current
                    if all(item.usage < self.threshold for item in current):
                        break

        return {
            "reason": reason,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "forced": force,
            "cleanup_requested": requested,
            "error": error,
            "before": [item.public() for item in before],
            "after": [item.public() for item in after],
        }

    def emergency_terminate(self, approved_pids: Iterable[int]) -> dict[str, Any]:
        """Terminate only explicitly approved PIDs. Never auto-select Python processes."""
        approved = {int(pid) for pid in approved_pids if int(pid) > 0 and int(pid) != os.getpid()}
        gpu_processes = {item["pid"]: item for item in query_gpu_processes()}
        terminated: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for pid in sorted(approved):
            if pid not in gpu_processes:
                errors.append({"pid": pid, "error": "PID is not currently reported as an NVIDIA compute process"})
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                terminated.append(gpu_processes[pid])
            except (OSError, PermissionError) as exc:
                errors.append({"pid": pid, "error": f"{type(exc).__name__}: {exc}"})

        return {"terminated": terminated, "errors": errors}
