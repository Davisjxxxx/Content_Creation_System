"""Graduated host & GPU resource monitor and predictive storage guard.

Read-only observability and threshold evaluation for Avatar V2's local GPU engine.
Tracks:
- Available physical RAM, ComfyUI process RSS/PSS, swap utilization, swap-in/out activity,
  and Linux PSI (/proc/pressure/memory) so static swap occupancy is never mistaken for
  active memory thrashing.
- GPU memory (total / used / free), GPU temperature, GPU utilization, and optional ComfyUI
  /system_stats torch pool metrics (allocated/reserved vs total).
- Graduated storage tiers (Normal >= 30 GB, Warning < 30 GB, Restricted < 20 GB,
  Emergency Stop < 15 GB) with predictive per-job storage estimation.
- Hysteresis-backed state transitions to prevent pause/resume flapping.

This module never kills processes, never deletes files, and never calls CUDA directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Severity = Literal["normal", "warning", "intervention", "emergency"]
StorageTier = Literal["normal", "warning", "restricted", "emergency_stop"]

GIB = 1024 ** 3
MIB = 1024 ** 2

# Provisional graduated storage thresholds (bytes) — calibrated in Sprint 2.
STORAGE_NORMAL_MIN_GB = 30.0
STORAGE_WARNING_MIN_GB = 20.0
STORAGE_EMERGENCY_MIN_GB = 15.0

# Provisional graduated memory & thermal thresholds — calibrated in Sprint 2.
RAM_WARN_PCT = 20.0          # available physical RAM < 20%
RAM_INTERVENE_PCT = 10.0     # available physical RAM < 10%
RAM_EMERGENCY_PCT = 5.0      # available physical RAM < 5%
RAM_HYSTERESIS_PCT = 3.0

SWAP_WARN_PCT = 50.0
SWAP_INTERVENE_PCT = 75.0
SWAP_EMERGENCY_PCT = 90.0
SWAP_ACTIVE_PAGES_PER_S = 64.0  # swap-in/out rate indicating active paging
PSI_ACTIVE_AVG10 = 10.0         # Linux PSI some avg10 % indicating real contention

TEMP_WARN_C = 75
TEMP_INTERVENE_C = 80
TEMP_EMERGENCY_C = 84
TEMP_HYSTERESIS_C = 5

_SEVERITY_RANK: dict[Severity, int] = {
    "normal": 0,
    "warning": 1,
    "intervention": 2,
    "emergency": 3,
}


def max_severity(*levels: Severity) -> Severity:
    best: Severity = "normal"
    for level in levels:
        if _SEVERITY_RANK[level] > _SEVERITY_RANK[best]:
            best = level
    return best


@dataclass(frozen=True)
class HostMemoryStats:
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    ram_available_pct: float = 100.0
    swap_total_mb: int = 0
    swap_used_mb: int = 0
    swap_used_pct: float = 0.0
    pswpin: int | None = None
    pswpout: int | None = None
    psi_some_avg10: float | None = None
    psi_full_avg10: float | None = None
    process_pid: int | None = None
    process_rss_mb: int | None = None
    process_pss_mb: int | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUTelemetry:
    index: int
    name: str
    used_mb: int
    total_mb: int
    free_mb: int
    temp_c: int | None = None
    util_pct: int | None = None
    torch_reserved_mb: int | None = None
    torch_free_mb: int | None = None

    @property
    def usage_pct(self) -> float:
        return round((self.used_mb / self.total_mb) * 100.0, 1) if self.total_mb else 0.0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage_pct"] = self.usage_pct
        return data


@dataclass(frozen=True)
class StorageStatus:
    path: str
    total_gb: float
    free_gb: float
    tier: StorageTier

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageCheck:
    allowed: bool
    tier: StorageTier
    free_gb: float
    estimated_mb: float
    projected_free_gb: float
    reserve_gb: float
    reason: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: float
    host: HostMemoryStats
    gpus: list[GPUTelemetry]
    storage: StorageStatus
    swap_pages_per_s: float = 0.0
    active_memory_pressure: bool = False
    ram_severity: Severity = "normal"
    swap_severity: Severity = "normal"
    temp_severity: Severity = "normal"
    storage_severity: Severity = "normal"
    overall_severity: Severity = "normal"
    reasons: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "host": self.host.public(),
            "gpus": [gpu.public() for gpu in self.gpus],
            "storage": self.storage.public(),
            "swap_pages_per_s": round(self.swap_pages_per_s, 1),
            "active_memory_pressure": self.active_memory_pressure,
            "ram_severity": self.ram_severity,
            "swap_severity": self.swap_severity,
            "temp_severity": self.temp_severity,
            "storage_severity": self.storage_severity,
            "overall_severity": self.overall_severity,
            "reasons": list(self.reasons),
        }


def parse_proc_meminfo(text: str) -> dict[str, int]:
    """Parse /proc/meminfo into KiB integers keyed by field name."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return values


def parse_proc_vmstat_swap(text: str) -> tuple[int | None, int | None]:
    pswpin = pswpout = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        if parts[0] == "pswpin":
            try:
                pswpin = int(parts[1])
            except ValueError:
                pass
        elif parts[0] == "pswpout":
            try:
                pswpout = int(parts[1])
            except ValueError:
                pass
    return pswpin, pswpout


def parse_proc_psi_memory(text: str) -> tuple[float | None, float | None]:
    """Parse Linux PSI /proc/pressure/memory -> (some_avg10, full_avg10)."""
    some_avg10 = full_avg10 = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        for token in parts[1:]:
            if token.startswith("avg10="):
                try:
                    val = float(token.split("=", 1)[1])
                except ValueError:
                    continue
                if kind == "some":
                    some_avg10 = val
                elif kind == "full":
                    full_avg10 = val
    return some_avg10, full_avg10


def read_process_memory_mb(pid: int | None) -> tuple[int | None, int | None]:
    """Return (rss_mb, pss_mb) for a Linux process without touching it."""
    if not pid or pid <= 0:
        return None, None
    rss_mb = pss_mb = None
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_mb = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        rollup = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8")
        for line in rollup.splitlines():
            if line.startswith("Pss:"):
                pss_mb = int(line.split()[1]) // 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    return rss_mb, pss_mb


def query_host_memory(process_pid: int | None = None) -> HostMemoryStats:
    """Read Linux host RAM, swap, swap activity counters, PSI, and optional process RSS/PSS."""
    try:
        meminfo = parse_proc_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        meminfo = {}

    ram_total_kb = meminfo.get("MemTotal", 0)
    ram_avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    swap_total_kb = meminfo.get("SwapTotal", 0)
    swap_free_kb = meminfo.get("SwapFree", 0)
    swap_used_kb = max(swap_total_kb - swap_free_kb, 0)

    ram_avail_pct = round((ram_avail_kb / ram_total_kb) * 100.0, 2) if ram_total_kb > 0 else 100.0
    swap_used_pct = round((swap_used_kb / swap_total_kb) * 100.0, 2) if swap_total_kb > 0 else 0.0

    try:
        pswpin, pswpout = parse_proc_vmstat_swap(Path("/proc/vmstat").read_text(encoding="utf-8"))
    except OSError:
        pswpin, pswpout = None, None

    try:
        psi_some, psi_full = parse_proc_psi_memory(Path("/proc/pressure/memory").read_text(encoding="utf-8"))
    except OSError:
        psi_some, psi_full = None, None

    rss_mb, pss_mb = read_process_memory_mb(process_pid)

    return HostMemoryStats(
        ram_total_mb=ram_total_kb // 1024,
        ram_available_mb=ram_avail_kb // 1024,
        ram_available_pct=ram_avail_pct,
        swap_total_mb=swap_total_kb // 1024,
        swap_used_mb=swap_used_kb // 1024,
        swap_used_pct=swap_used_pct,
        pswpin=pswpin,
        pswpout=pswpout,
        psi_some_avg10=psi_some,
        psi_full_avg10=psi_full,
        process_pid=process_pid if rss_mb is not None else None,
        process_rss_mb=rss_mb,
        process_pss_mb=pss_mb,
    )


def parse_nvidia_smi_telemetry(text: str) -> list[GPUTelemetry]:
    gpus: list[GPUTelemetry] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            used = int(float(parts[2]))
            total = int(float(parts[3]))
            free = int(float(parts[4]))
            temp = int(float(parts[5])) if len(parts) > 5 and parts[5] not in {"", "N/A", "[N/A]"} else None
            util = int(float(parts[6])) if len(parts) > 6 and parts[6] not in {"", "N/A", "[N/A]"} else None
            gpus.append(GPUTelemetry(idx, name, used, total, free, temp_c=temp, util_pct=util))
        except ValueError:
            continue
    return gpus


def query_gpu_telemetry(comfy_system_stats: dict[str, Any] | None = None) -> list[GPUTelemetry]:
    """Query nvidia-smi VRAM, temperature, and utilization; optionally enrich with ComfyUI torch pool stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        gpus = parse_nvidia_smi_telemetry(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if not comfy_system_stats:
        return gpus

    by_index: dict[int, dict[str, Any]] = {}
    for dev in comfy_system_stats.get("devices") or []:
        if isinstance(dev, dict) and dev.get("type") == "cuda" and isinstance(dev.get("index"), int):
            by_index[int(dev["index"])] = dev

    enriched: list[GPUTelemetry] = []
    for gpu in gpus:
        dev = by_index.get(gpu.index)
        if dev:
            torch_total = dev.get("torch_vram_total")
            torch_free = dev.get("torch_vram_free")
            enriched.append(
                GPUTelemetry(
                    index=gpu.index,
                    name=gpu.name,
                    used_mb=gpu.used_mb,
                    total_mb=gpu.total_mb,
                    free_mb=gpu.free_mb,
                    temp_c=gpu.temp_c,
                    util_pct=gpu.util_pct,
                    torch_reserved_mb=(int(torch_total) // MIB) if isinstance(torch_total, (int, float)) else None,
                    torch_free_mb=(int(torch_free) // MIB) if isinstance(torch_free, (int, float)) else None,
                )
            )
        else:
            enriched.append(gpu)
    return enriched


def classify_storage_tier(free_gb: float) -> StorageTier:
    if free_gb < STORAGE_EMERGENCY_MIN_GB:
        return "emergency_stop"
    if free_gb < STORAGE_WARNING_MIN_GB:
        return "restricted"
    if free_gb < STORAGE_NORMAL_MIN_GB:
        return "warning"
    return "normal"


def query_storage_status(path: str | Path | None = None) -> StorageStatus:
    target = Path(path or Path.home()).expanduser()
    probe = target if target.exists() else target.parent
    if not probe.exists():
        probe = Path.home()
    usage = shutil.disk_usage(probe)
    total_gb = round(usage.total / GIB, 2)
    free_gb = round(usage.free / GIB, 2)
    return StorageStatus(
        path=str(probe),
        total_gb=total_gb,
        free_gb=free_gb,
        tier=classify_storage_tier(free_gb),
    )


def estimate_job_storage_bytes(
    *,
    kind: Literal["image", "video", "scene", "upscale"],
    width: int = 896,
    height: int = 1280,
    frames: int = 1,
    shots: int = 1,
    upscale: bool = False,
) -> int:
    """Predict total disk footprint (final output + temp frames + encode/mux scratch + safety buffer).

    Intentionally conservative so a job cannot breach the 15 GB protected reserve mid-run.
    """
    pixels = max(int(width), 64) * max(int(height), 64)
    shots = max(int(shots), 1)
    frames = max(int(frames), 1)

    if kind == "image":
        # PNG output + archive copy + thumbnail cache + ComfyUI staging scratch.
        raw_png = max(pixels * 3, 2 * MIB)
        total = int(raw_png * (3.0 if upscale else 1.5)) + 32 * MIB
        return total

    # Video / multi-shot scene / video upscale:
    # Encoded MP4 (~0.25-0.6 MB/s at 480p-720p) + intermediate frame/chain PNGs + ffmpeg mux copy.
    mp4_per_shot = max(int((pixels / (480 * 832)) * (frames / 81.0) * 32 * MIB), 16 * MIB)
    temp_frames_per_shot = max(int(pixels * 0.5), 2 * MIB) * (4 if not upscale else min(frames, 120))
    stitch_buffer = mp4_per_shot * shots if shots > 1 else 0
    if kind == "upscale" or upscale:
        mp4_per_shot = int(mp4_per_shot * 3.0)
    return (mp4_per_shot + temp_frames_per_shot) * shots + stitch_buffer + 128 * MIB


def check_storage_for_job(
    path: str | Path | None,
    *,
    estimated_bytes: int,
    reserve_gb: float = STORAGE_EMERGENCY_MIN_GB,
) -> StorageCheck:
    """Predictive storage gate: refuse starting a job that would breach the protected reserve."""
    status = query_storage_status(path)
    estimated_gb = max(estimated_bytes, 0) / GIB
    projected_free_gb = round(status.free_gb - estimated_gb, 3)
    estimated_mb = round(max(estimated_bytes, 0) / MIB, 1)

    if status.free_gb < reserve_gb:
        return StorageCheck(
            allowed=False,
            tier="emergency_stop",
            free_gb=status.free_gb,
            estimated_mb=estimated_mb,
            projected_free_gb=projected_free_gb,
            reserve_gb=reserve_gb,
            reason=(
                f"Storage emergency stop: {status.free_gb:.1f} GB free is below the "
                f"{reserve_gb:.0f} GB protected reserve."
            ),
        )
    if projected_free_gb < reserve_gb:
        return StorageCheck(
            allowed=False,
            tier=status.tier,
            free_gb=status.free_gb,
            estimated_mb=estimated_mb,
            projected_free_gb=projected_free_gb,
            reserve_gb=reserve_gb,
            reason=(
                f"Predicted job storage ({estimated_mb:.0f} MB) would leave {projected_free_gb:.1f} GB free, "
                f"below the {reserve_gb:.0f} GB protected reserve."
            ),
        )
    return StorageCheck(
        allowed=True,
        tier=status.tier,
        free_gb=status.free_gb,
        estimated_mb=estimated_mb,
        projected_free_gb=projected_free_gb,
        reserve_gb=reserve_gb,
    )


class ResourceGuard:
    """Stateful graduated monitor with hysteresis and swap-activity gating.

    Prevents pause/resume flapping by requiring RAM availability and GPU temperature to recover
    past explicit hysteresis bands before downgrading an elevated state.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path or Path.home()).expanduser()
        self._last_sample_ts: float | None = None
        self._last_swap_pages: int | None = None
        self._swap_history: deque[tuple[float, int]] = deque(maxlen=64)
        self._ram_state: Severity = "normal"
        self._temp_state: Severity = "normal"

    def _eval_ram(self, avail_pct: float) -> Severity:
        # Escalate immediately on breach.
        if avail_pct < RAM_EMERGENCY_PCT:
            self._ram_state = "emergency"
        elif avail_pct < RAM_INTERVENE_PCT:
            if _SEVERITY_RANK[self._ram_state] < _SEVERITY_RANK["intervention"]:
                self._ram_state = "intervention"
            elif self._ram_state == "emergency" and avail_pct >= RAM_EMERGENCY_PCT + RAM_HYSTERESIS_PCT:
                self._ram_state = "intervention"
        elif avail_pct < RAM_WARN_PCT:
            if _SEVERITY_RANK[self._ram_state] < _SEVERITY_RANK["warning"]:
                self._ram_state = "warning"
            elif self._ram_state in {"emergency", "intervention"} and avail_pct >= RAM_INTERVENE_PCT + RAM_HYSTERESIS_PCT:
                self._ram_state = "warning"
        else:
            if self._ram_state == "emergency" and avail_pct >= RAM_EMERGENCY_PCT + RAM_HYSTERESIS_PCT:
                self._ram_state = "intervention"
            if self._ram_state == "intervention" and avail_pct >= RAM_INTERVENE_PCT + RAM_HYSTERESIS_PCT:
                self._ram_state = "warning"
            if self._ram_state == "warning" and avail_pct >= RAM_WARN_PCT + RAM_HYSTERESIS_PCT:
                self._ram_state = "normal"
        return self._ram_state

    def _eval_temp(self, max_temp_c: int | None) -> Severity:
        if max_temp_c is None:
            return self._temp_state
        if max_temp_c >= TEMP_EMERGENCY_C:
            self._temp_state = "emergency"
        elif max_temp_c >= TEMP_INTERVENE_C:
            if _SEVERITY_RANK[self._temp_state] < _SEVERITY_RANK["intervention"]:
                self._temp_state = "intervention"
            elif self._temp_state == "emergency" and max_temp_c <= TEMP_EMERGENCY_C - TEMP_HYSTERESIS_C:
                self._temp_state = "intervention"
        elif max_temp_c >= TEMP_WARN_C:
            if _SEVERITY_RANK[self._temp_state] < _SEVERITY_RANK["warning"]:
                self._temp_state = "warning"
            elif self._temp_state in {"emergency", "intervention"} and max_temp_c <= TEMP_INTERVENE_C - TEMP_HYSTERESIS_C:
                self._temp_state = "warning"
        else:
            if self._temp_state == "emergency" and max_temp_c <= TEMP_EMERGENCY_C - TEMP_HYSTERESIS_C:
                self._temp_state = "intervention"
            if self._temp_state == "intervention" and max_temp_c <= TEMP_INTERVENE_C - TEMP_HYSTERESIS_C:
                self._temp_state = "warning"
            if self._temp_state == "warning" and max_temp_c <= TEMP_WARN_C - TEMP_HYSTERESIS_C:
                self._temp_state = "normal"
        return self._temp_state

    def _eval_swap(self, swap_pct: float, active_pressure: bool) -> Severity:
        # Never treat static swap occupancy alone as emergency memory pressure.
        if swap_pct <= SWAP_WARN_PCT:
            return "normal"
        if not active_pressure:
            return "warning"
        if swap_pct > SWAP_EMERGENCY_PCT:
            return "emergency"
        if swap_pct > SWAP_INTERVENE_PCT:
            return "intervention"
        return "warning"

    def evaluate(
        self,
        *,
        host: HostMemoryStats | None = None,
        gpus: list[GPUTelemetry] | None = None,
        storage: StorageStatus | None = None,
        now: float | None = None,
        process_pid: int | None = None,
        comfy_system_stats: dict[str, Any] | None = None,
    ) -> ResourceSnapshot:
        ts = time.monotonic() if now is None else float(now)
        host_stats = host if host is not None else query_host_memory(process_pid=process_pid)
        gpu_stats = gpus if gpus is not None else query_gpu_telemetry(comfy_system_stats=comfy_system_stats)
        storage_stats = storage if storage is not None else query_storage_status(self.storage_path)

        swap_rate = 0.0
        if host_stats.pswpin is not None and host_stats.pswpout is not None:
            total_pages = host_stats.pswpin + host_stats.pswpout
            # RC-14: Time-bucketed counter anchors (0.25s resolution over rolling 8s window).
            # Invariant across 10ms / 20ms / 100ms / 2s / jittered polling, and resets cleanly
            # on stale gaps (>30s) or kernel counter resets / backward clock jumps!
            if (
                self._last_sample_ts is not None
                and (ts < self._last_sample_ts or (ts - self._last_sample_ts) > 30.0)
            ) or (self._last_swap_pages is not None and total_pages < self._last_swap_pages):
                self._swap_history.clear()
                self._last_sample_ts = ts
                self._last_swap_pages = total_pages
                self._swap_history.append((ts, total_pages))
            else:
                anchor: tuple[float, int] | None = None
                for old_ts, old_pages in self._swap_history:
                    if 1.0 <= (ts - old_ts) <= 10.0 and total_pages >= old_pages:
                        anchor = (old_ts, old_pages)
                        break
                if anchor is not None:
                    dt = ts - anchor[0]
                    swap_rate = max(total_pages - anchor[1], 0) / dt
                elif (
                    self._last_sample_ts is not None
                    and self._last_swap_pages is not None
                    and 1.0 <= (ts - self._last_sample_ts) <= 30.0
                    and total_pages >= self._last_swap_pages
                ):
                    dt = ts - self._last_sample_ts
                    swap_rate = max(total_pages - self._last_swap_pages, 0) / dt

                # Record at most one anchor per 0.25s bucket so 10ms polling never evicts 1s+ history!
                if not self._swap_history or (ts - self._swap_history[-1][0]) >= 0.25:
                    self._swap_history.append((ts, total_pages))
                while len(self._swap_history) > 2 and (ts - self._swap_history[0][0]) > 8.0:
                    self._swap_history.popleft()
                self._last_sample_ts = ts
                self._last_swap_pages = total_pages

        psi_pressure = (host_stats.psi_some_avg10 or 0.0) >= PSI_ACTIVE_AVG10 or (host_stats.psi_full_avg10 or 0.0) >= 2.0
        active_pressure = swap_rate >= SWAP_ACTIVE_PAGES_PER_S or psi_pressure

        ram_sev = self._eval_ram(host_stats.ram_available_pct)
        swap_sev = self._eval_swap(host_stats.swap_used_pct, active_pressure)
        temps = [g.temp_c for g in gpu_stats if g.temp_c is not None]
        temp_sev = self._eval_temp(max(temps) if temps else None)
        storage_sev: Severity = {
            "normal": "normal",
            "warning": "warning",
            "restricted": "intervention",
            "emergency_stop": "emergency",
        }[storage_stats.tier]

        overall = max_severity(ram_sev, swap_sev, temp_sev, storage_sev)
        reasons: list[str] = []
        if ram_sev != "normal":
            reasons.append(f"RAM available {host_stats.ram_available_pct:.1f}% ({host_stats.ram_available_mb} MB) [{ram_sev}]")
        if swap_sev != "normal":
            activity = f"active paging {swap_rate:.0f} pg/s" if active_pressure else "static swap, no active thrashing"
            reasons.append(f"Swap used {host_stats.swap_used_pct:.1f}% ({activity}) [{swap_sev}]")
        if temp_sev != "normal" and temps:
            reasons.append(f"GPU temperature {max(temps)}°C [{temp_sev}]")
        if storage_sev != "normal":
            reasons.append(f"Free disk {storage_stats.free_gb:.1f} GB ({storage_stats.tier}) [{storage_sev}]")

        return ResourceSnapshot(
            timestamp=ts,
            host=host_stats,
            gpus=gpu_stats,
            storage=storage_stats,
            swap_pages_per_s=swap_rate,
            active_memory_pressure=active_pressure,
            ram_severity=ram_sev,
            swap_severity=swap_sev,
            temp_severity=temp_sev,
            storage_severity=storage_sev,
            overall_severity=overall,
            reasons=reasons,
        )
