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
        ownership_authorizer: Any = None,
    ) -> None:
        self.threshold = min(max(float(threshold), 0.50), 0.99)
        self.check_interval_s = max(float(check_interval_s), 1.0)
        self.enabled = bool(enabled)
        self._ownership_authorizer = ownership_authorizer
        self._trace_stop = threading.Event()
        self._trace_thread: threading.Thread | None = None
        self._trace_lock = threading.Lock()
        self._trace_label = ""
        self._trace_peak: dict[int, GPUSnapshot] = {}
        self._trace_samples = 0

    def set_ownership_authorizer(self, authorizer: Any) -> None:
        """Bind a centralized fail-closed ownership gate checked before every global /free (F01 / RC-01)."""
        self._ownership_authorizer = authorizer

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

    def update_settings(
        self,
        *,
        enabled: bool,
        threshold: float,
        check_interval_s: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = min(max(float(threshold), 0.50), 0.99)
        self.check_interval_s = max(float(check_interval_s), 1.0)

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

    def current_trace_peak(self) -> list[dict[str, Any]]:
        with self._trace_lock:
            return [self._trace_peak[index].public() for index in sorted(self._trace_peak)]

    def maybe_cleanup(
        self,
        provider: Any | None = None,
        *,
        reason: str,
        force: bool = False,
        unload_models: bool = True,
        respect_retention: bool = False,
        wait_s: float = 8.0,
    ) -> dict[str, Any]:
        """Clean local Python garbage and, when needed, ask ComfyUI to free VRAM."""
        gc.collect()
        before = query_gpu_memory()
        over_threshold = any(item.usage >= self.threshold for item in before)
        # Explicit/manual and memory-sensitive-profile cleanup must work even when
        # threshold-based automatic cleanup is disabled in Settings.
        should_clean = force or (self.enabled and over_threshold)
        # ComfyUI 0.46's /free handler defaults unset/false unload_models to free_memory=True,
        # which unloads all loaded models anyway. When a caller intentionally asks to retain
        # warm models (unload_models=False):
        # - If ResourcePlanner explicitly authorized retention (respect_retention=True), do NOT
        #   evict solely because total GPU VRAM exceeds the fixed 85% mark (FLUX.2 9B legitimately
        #   rests at ~85.5-90.4% total VRAM including desktop overhead).
        # - Otherwise (legacy callers without planner context), escalate to full unload at >=85%.
        effective_unload = unload_models or (over_threshold and not respect_retention)
        models_retained = bool(not effective_unload)
        requested = False
        ownership_denied = False
        error = ""

        if should_clean and provider is not None and effective_unload:
            # F01 / RC-01: Centralized global-cleanup ownership gate before ANY automatic/manual POST /free!
            authorizer = getattr(self, "_ownership_authorizer", None)
            if callable(authorizer):
                try:
                    auth_ok, auth_reason = authorizer(provider, "free_memory")
                except Exception as exc:
                    auth_ok, auth_reason = False, f"authorizer_error:{type(exc).__name__}:{exc}"
                if not auth_ok:
                    ownership_denied = True
                    error = f"ownership_denied:{auth_reason}"
            if not ownership_denied:
                try:
                    provider.free_memory(unload_models=True, free_memory=True)
                    requested = True
                except Exception as exc:  # cleanup failure should be recorded, not mask a render result
                    error = f"{type(exc).__name__}: {exc}"

        after = before
        if requested:
            deadline = time.monotonic() + max(wait_s, 0.0)
            before_total = sum(item.used_mb for item in before)
            previous_total = before_total
            saw_drop = False
            stable_samples = 0
            while time.monotonic() < deadline:
                time.sleep(0.5)
                current = query_gpu_memory()
                if current:
                    after = current
                    current_total = sum(item.used_mb for item in current)
                    if current_total <= before_total - 64:
                        saw_drop = True
                    if saw_drop and abs(current_total - previous_total) <= 16:
                        stable_samples += 1
                    else:
                        stable_samples = 0
                    previous_total = current_total
                    if force and saw_drop and stable_samples >= 1:
                        break
                    if not force and all(item.usage < self.threshold for item in current):
                        break

        released_mb = max(
            sum(item.used_mb for item in before) - sum(item.used_mb for item in after),
            0,
        )
        # F10: Unified release-evidence contract emitted by production GPUMemoryManager:
        # - cleanup_requested: HTTP /free was sent and succeeded
        # - cleanup_performed: cleanup actually executed without error or ownership denial
        # - effective: measured VRAM dropped >= 64 MiB (or telemetry unavailable on non-GPU test host while /free succeeded)
        vram_drop_verified = released_mb >= 64
        return {
            "reason": reason,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "forced": force,
            "unload_models": effective_unload,
            "models_retained": models_retained,
            "cleanup_requested": requested,
            "cleanup_performed": bool(requested and not error and not ownership_denied),
            "ownership_denied": ownership_denied,
            "error": error,
            "before": [item.public() for item in before],
            "after": [item.public() for item in after],
            "released_mb": released_mb,
            "effective": vram_drop_verified,
            "vram_released_verified": bool(vram_drop_verified if before else (requested and not error and not ownership_denied)),
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
