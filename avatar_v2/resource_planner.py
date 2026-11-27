"""Centralized ResourcePlanner for Avatar V2 (Sprint 3, Sections 2 & 3).

Coordinates workload admission, model-family transitions, pressure-aware & upcoming-job-aware
model retention, offloading hints, tiled-VAE selection, and zero-contention VLM scheduling.

Architectural boundary:
- NEVER acts as a competing low-level inference memory manager.
- Delegates tensor placement, dynamic weight streaming, and intra-family selective eviction
  to ComfyUI's native ModelPatcher / comfy-aimdo / cudaMallocAsync.
- Keeps all resource decisions internal to Avatar V2 (zero frontend configuration knobs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .model_registry import DEFAULT_REGISTRY, ComponentTransition, ModelRegistry, ModelSpec
from .resource_monitor import (
    ResourceGuard,
    ResourceSnapshot,
    StorageCheck,
    check_storage_for_job,
    estimate_job_storage_bytes,
    measured_gpu_headroom,
    recent_gpu_headroom,
)

RetentionStrategy = Literal["retain_all", "comfy_native_selective", "full_evict"]
OffloadingStrategy = Literal["full_gpu", "cpu_text_encoder", "dynamic_vram_stream", "block_offload_cpu_te"]
VAEStrategy = Literal["standard", "tiled"]
VLMSchedule = Literal["cpu_only", "after_full_unload"]

# Still resolution threshold above which FLUX.2 switches to VAEDecodeTiled automatically.
TILED_STILL_PIXELS = 1024 * 1280
# Free VRAM floor below which video text encoders default to CPU and tiled VAE is activated.
TIGHT_VRAM_FREE_MB = 5200


@dataclass(frozen=True)
class WorkloadRequest:
    """Normalized description of an upcoming generation job for ResourcePlanner."""

    model_id: str
    workflow: str
    width: int = 896
    height: int = 1280
    frames: int = 1
    shots: int = 1
    upscale: bool = False
    local_only: bool = True
    has_reference_assets: bool = False
    requires_vlm_qa: bool = False
    output_dir: str | None = None
    explicit_vram_mode: str | None = None  # optional legacy hint ("balanced" | "cpu_offload")


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated internal execution plan produced by ResourcePlanner."""

    admitted: bool
    admission_reason: str
    model_id: str
    selected_workflow: str
    fallback_workflow: str | None
    transition_kind: str
    required_components: list[str]
    reusable_components: list[str]
    evict_components: list[str]
    retention_strategy: RetentionStrategy
    pre_unload_models: bool
    respect_retention_above_threshold: bool
    offloading_strategy: OffloadingStrategy
    wan_clip_device: Literal["default", "cpu"]
    vae_strategy: VAEStrategy
    vlm_schedule: VLMSchedule
    recovery_strategy: str
    storage_check: dict[str, Any]
    resource_severity: str
    rationale: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return asdict(self)


class ResourcePlanner:
    """Central deterministic planner replacing blind fixed-85% VRAM eviction."""

    def __init__(
        self,
        *,
        registry: ModelRegistry = DEFAULT_REGISTRY,
        guard: ResourceGuard | None = None,
    ) -> None:
        self.registry = registry
        self.guard = guard or ResourceGuard()

    def plan_execution(
        self,
        request: WorkloadRequest,
        *,
        current_model_id: str | None = None,
        upcoming_model_id: str | None = None,
        prior_failed_or_cancelled: bool = False,
        snapshot: ResourceSnapshot | None = None,
        storage_check: StorageCheck | None = None,
        active_generation_running: bool = False,
        comfyui_dir: str | Path | None = None,
    ) -> ExecutionPlan:
        """Produce a deterministic ExecutionPlan for an upcoming job using live or simulated telemetry."""
        rationale: list[str] = []
        spec = self.registry.get(request.model_id)
        if spec is None:
            return ExecutionPlan(
                admitted=False,
                admission_reason=f"Unknown model_id '{request.model_id}' in ModelRegistry.",
                model_id=request.model_id,
                selected_workflow=request.workflow,
                fallback_workflow=None,
                transition_kind="cold_start",
                required_components=[],
                reusable_components=[],
                evict_components=[],
                retention_strategy="full_evict",
                pre_unload_models=True,
                respect_retention_above_threshold=False,
                offloading_strategy="block_offload_cpu_te",
                wan_clip_device="cpu",
                vae_strategy="tiled",
                vlm_schedule="cpu_only",
                recovery_strategy="abort_unknown_model",
                storage_check={},
                resource_severity="emergency",
                rationale=[f"Model '{request.model_id}' is not registered."],
            )

        snap = snapshot if snapshot is not None else self.guard.evaluate()
        kind = "image" if spec.family == "flux2" else ("scene" if request.shots > 1 else "video")
        est_bytes = estimate_job_storage_bytes(
            kind=kind,  # type: ignore[arg-type]
            width=request.width,
            height=request.height,
            frames=request.frames,
            shots=request.shots,
            upscale=request.upscale,
        )
        s_check = (
            storage_check
            if storage_check is not None
            else check_storage_for_job(request.output_dir or self.guard.storage_path, estimated_bytes=est_bytes)
        )

        # Fallback workflow lookup from registry.
        fallback_spec = self.registry.get(spec.fallback_model_id) if spec.fallback_model_id else None
        fallback_wf = fallback_spec.workflow if fallback_spec else None

        # Component residency diff (strictly verifying file/loader/compat_group identity).
        transition: ComponentTransition = self.registry.diff_components(current_model_id, spec.model_id)

        # Admission gates: storage reserve, local-only policy, emergency RAM/thermal/swap pressure.
        admitted = True
        admission_reason = "ok"
        if not request.local_only and str(request.workflow).lower().startswith("runway"):
            admitted = False
            admission_reason = "Cloud provider requires explicit user authorization."
        elif not s_check.allowed:
            admitted = False
            admission_reason = s_check.reason
            rationale.append(f"Blocked by predictive storage guard: {s_check.reason}")
        elif snap.temp_severity == "emergency":
            admitted = False
            admission_reason = "GPU temperature reached emergency thermal pause threshold; wait for cooldown."
            rationale.append("Deferred admission due to emergency GPU thermal state.")
        elif snap.ram_severity == "emergency":
            admitted = False
            admission_reason = "Available physical RAM fell below 5% emergency reserve."
            rationale.append("Deferred admission due to emergency physical RAM state.")
        elif snap.swap_severity == "emergency" and snap.active_memory_pressure:
            admitted = False
            admission_reason = "Active swap thrashing detected above 90% swap utilization."
            rationale.append("Deferred admission due to active swap/PSI thrashing.")

        if comfyui_dir is not None:
            missing = [c.filename for c in spec.components if not c.exists_in(Path(comfyui_dir).expanduser())]
            if missing:
                admitted = False
                admission_reason = f"Missing required model file(s) for {spec.label}: {', '.join(missing)}"
                rationale.append(admission_reason)

        # Retention strategy: pressure-aware + component-aware + upcoming-job-aware
        # (replaces blind fixed-85% eviction that prematurely unloaded FLUX.2 9B at ~86-90% VRAM).
        pressure_elevated = (
            snap.ram_severity in {"intervention", "emergency"}
            or (snap.swap_severity in {"intervention", "emergency"} and snap.active_memory_pressure)
            or snap.temp_severity in {"intervention", "emergency"}
        )

        if prior_failed_or_cancelled:
            retention: RetentionStrategy = "full_evict"
            pre_unload = True
            respect_retention = False
            rationale.append("Prior job failed or was cancelled: forcing full /free eviction for clean recovery.")
        elif pressure_elevated:
            retention = "full_evict"
            pre_unload = True
            respect_retention = False
            rationale.append(f"System pressure elevated ({snap.overall_severity}): forcing clean /free eviction.")
        elif transition.transition_kind == "same_model":
            # Same exact model fingerprint & healthy pressure -> retain warm models even if VRAM >= 85%!
            retention = "retain_all"
            pre_unload = False
            respect_retention = True
            rationale.append(
                f"Consecutive same-model job ({spec.model_id}) under healthy pressure: retaining warm weights "
                "(bypassing fixed 85% VRAM eviction)."
            )
        elif transition.transition_kind == "shared_components":
            # Shared large supporting component (e.g. 6.4 GB UMT5 across Wan 14B <-> 5B):
            # delegate selective UNet/VAE eviction to ComfyUI's native ModelPatcher/comfy-aimdo.
            retention = "comfy_native_selective"
            pre_unload = False
            respect_retention = True
            rationale.append(
                f"Intra-family transition ({transition.from_model_id} -> {spec.model_id}) reuses "
                f"{', '.join(c.component_id for c in transition.reusable)} ({transition.reusable_mb} MB): "
                "delegating selective eviction to ComfyUI ModelPatcher."
            )
        elif transition.transition_kind == "cross_family":
            retention = "full_evict"
            pre_unload = True
            respect_retention = False
            rationale.append(
                f"Cross-family transition ({transition.from_model_id} -> {spec.model_id}): forcing full /free eviction."
            )
        else:  # cold_start
            retention = "full_evict" if spec.family == "wan22" else "retain_all"
            pre_unload = spec.family == "wan22"
            respect_retention = spec.family != "wan22"
            rationale.append(f"Cold start for {spec.model_id}: retention={retention}.")

        # Offloading & text-encoder placement strategy.
        gpu_free_mb = measured_gpu_headroom(snap.gpus) or 0
        if request.explicit_vram_mode == "cpu_offload":
            offloading: OffloadingStrategy = "block_offload_cpu_te" if spec.family == "wan22" else "cpu_text_encoder"
            wan_clip: Literal["default", "cpu"] = "cpu"
            rationale.append("Explicit CPU offload mode requested: placing text encoder on CPU.")
        elif spec.family == "flux2":
            if spec.model_id == "flux2_9b" or gpu_free_mb < TIGHT_VRAM_FREE_MB:
                offloading = "cpu_text_encoder"
                wan_clip = "cpu"
                rationale.append("FLUX.2 9B or tight VRAM: prefer CPU text encoder so UNet stays GPU-resident.")
            else:
                offloading = "full_gpu"
                wan_clip = "default"
                rationale.append("FLUX.2 4B fits comfortably in GPU memory.")
        else:
            # Wan 2.2 14B / 5B / Fun-Control on 8 GB VRAM.
            if gpu_free_mb < TIGHT_VRAM_FREE_MB or pressure_elevated or request.frames > 81:
                offloading = "block_offload_cpu_te"
                wan_clip = "cpu"
                rationale.append(
                    f"Tight VRAM headroom ({gpu_free_mb} MB free) or extended frames ({request.frames}): "
                    "placing UMT5 text encoder on CPU and streaming UNet blocks."
                )
            else:
                offloading = "dynamic_vram_stream"
                wan_clip = "default"
                rationale.append("Standard Wan 2.2 DynamicVRAM / block streaming path.")

        # VAE decoding strategy: tiled automatically for high-res stills, extended frames, Wan 5B 48-ch VAE,
        # or tight VRAM on non-warm starts. On a healthy warm same-model FLUX.2 still, free VRAM is naturally
        # low (~1-1.5 GB) specifically because the 9B weights are already resident; avoid unnecessary tiling
        # unless resolution exceeds TILED_STILL_PIXELS or system pressure is elevated (live benchmark gate).
        pixels = int(request.width) * int(request.height)
        warm_flux_still = (
            spec.family == "flux2"
            and transition.transition_kind == "same_model"
            and not pressure_elevated
            and pixels <= TILED_STILL_PIXELS
        )
        if (
            pixels > TILED_STILL_PIXELS
            or request.frames > 81
            or spec.model_id == "wan22_5b"
            or (gpu_free_mb < TIGHT_VRAM_FREE_MB and not warm_flux_still)
        ):
            vae_strategy: VAEStrategy = "tiled"
            rationale.append("Selected VAEDecodeTiled to bound VAE peak VRAM safely.")
        else:
            vae_strategy = "standard"
            if warm_flux_still and gpu_free_mb < TIGHT_VRAM_FREE_MB:
                rationale.append(
                    "Kept standard VAE on healthy warm FLUX.2 still (low free VRAM reflects resident weights; "
                    "subject to live tiled-vs-untiled benchmark verification)."
                )

        # Zero-contention local VLM / LLM scheduling:
        # If ComfyUI is actively running or another GPU job is immediately queued using the same model,
        # schedule Ollama on CPU (i9-13980HX 24C/32T); otherwise run sequentially after full /free.
        if active_generation_running or (upcoming_model_id and upcoming_model_id == spec.model_id):
            vlm_schedule: VLMSchedule = "cpu_only"
            rationale.append("VLM/LLM scheduled on CPU (num_gpu=0) to avoid evicting warm ComfyUI models.")
        else:
            vlm_schedule = "after_full_unload"

        recovery = (
            "abort_on_storage_or_policy"
            if not admitted
            else "step_down_tiled_cpu_te_and_retry_once"
        )

        return ExecutionPlan(
            admitted=admitted,
            admission_reason=admission_reason,
            model_id=spec.model_id,
            selected_workflow=request.workflow or spec.workflow,
            fallback_workflow=fallback_wf,
            transition_kind=transition.transition_kind,
            required_components=[c.component_id for c in transition.required],
            reusable_components=[c.component_id for c in transition.reusable],
            evict_components=[c.component_id for c in transition.evict],
            retention_strategy=retention,
            pre_unload_models=pre_unload,
            respect_retention_above_threshold=respect_retention,
            offloading_strategy=offloading,
            wan_clip_device=wan_clip,
            vae_strategy=vae_strategy,
            vlm_schedule=vlm_schedule,
            recovery_strategy=recovery,
            storage_check=s_check.public(),
            resource_severity=snap.overall_severity,
            rationale=rationale,
        )

    def plan_post_render(
        self,
        *,
        completed_model_id: str | None,
        next_queued_model_id: str | None,
        job_status: str,
        requires_vlm_qa: bool = False,
        snapshot: ResourceSnapshot | None = None,
    ) -> dict[str, Any]:
        """Decide post-render retention based on outcome, upcoming queue job, and pressure."""
        if job_status in {"error", "cancelled"}:
            return {
                "retention_strategy": "full_evict",
                "unload_models": True,
                "respect_retention": False,
                "reason": f"post-render:{job_status}",
            }
        if requires_vlm_qa:
            return {
                "retention_strategy": "full_evict",
                "unload_models": True,
                "respect_retention": False,
                "reason": "post-render:done · release for sequential VLM QA",
            }
        snap = snapshot if snapshot is not None else self.guard.evaluate()
        if snap.overall_severity == "emergency":
            return {
                "retention_strategy": "full_evict",
                "unload_models": True,
                "respect_retention": False,
                "reason": "post-render:done · emergency resource relief",
            }
        if completed_model_id and next_queued_model_id:
            transition = self.registry.diff_components(completed_model_id, next_queued_model_id)
            if transition.transition_kind in {"same_model", "shared_components"}:
                return {
                    "retention_strategy": "retain_all" if transition.transition_kind == "same_model" else "comfy_native_selective",
                    "unload_models": False,
                    "respect_retention": True,
                    "reason": f"post-render:done · retain for upcoming {next_queued_model_id}",
                }
        # Default for FLUX.2 still models: keep warm for quick interactive follow-up stills.
        spec = self.registry.get(completed_model_id or "")
        if spec is not None and spec.family == "flux2":
            return {
                "retention_strategy": "retain_all",
                "unload_models": False,
                "respect_retention": True,
                "reason": f"post-render:{job_status} · keep image models",
            }
        return {
            "retention_strategy": "full_evict",
            "unload_models": True,
            "respect_retention": False,
            "reason": f"post-render:{job_status}",
        }

    def plan_visual_inspection(
        self,
        *,
        requires_vlm: bool,
        active_generation_running: bool = False,
        comfy_models_warm: bool = False,
        current_model_id: str | None = None,
        upcoming_model_id: str | None = None,
        snapshot: ResourceSnapshot | None = None,
        allow_sequential_gpu: bool = False,
        batch_queue_depth: int = 0,
    ) -> dict[str, Any]:
        """Coordinate Sprint 7 visual inspection placement without evicting warm FLUX/Wan weights."""
        snap = snapshot if snapshot is not None else self.guard.evaluate()
        if not requires_vlm:
            return {
                "execution_mode": "cpu_deterministic_only",
                "num_gpu": 0,
                "prefer_lightweight_model": True,
                "verify_vram_release": False,
                "evict_comfy_models": False,
                "reason": "All requirements covered by fast CPU deterministic/structural inspection; VLM not needed.",
            }

        upcoming_same = bool(current_model_id and upcoming_model_id and current_model_id == upcoming_model_id)
        if snap.ram_severity == "emergency":
            return {
                "execution_mode": "cpu_deterministic_only",
                "num_gpu": 0,
                "prefer_lightweight_model": True,
                "verify_vram_release": False,
                "evict_comfy_models": False,
                "reason": "Emergency physical RAM pressure: skipping local VLM and running deterministic checks only.",
            }
        if (active_generation_running or comfy_models_warm or upcoming_same) and batch_queue_depth >= 3:
            return {
                "execution_mode": "deferred_batch",
                "num_gpu": 0,
                "prefer_lightweight_model": True,
                "verify_vram_release": False,
                "evict_comfy_models": False,
                "reason": (
                    "Multi-job batch queued with warm ComfyUI models: deferring VLM inspection so FLUX/Wan "
                    "throughput is preserved without cold reloads."
                ),
            }
        if active_generation_running or comfy_models_warm or upcoming_same or snap.temp_severity == "emergency":
            return {
                "execution_mode": "cpu_vlm",
                "num_gpu": 0,
                "prefer_lightweight_model": True,
                "verify_vram_release": False,
                "evict_comfy_models": False,
                "reason": (
                    "Warm/active ComfyUI models or GPU thermal limit: running lightweight local VLM on CPU "
                    "(num_gpu=0) to avoid evicting FLUX/Wan weights."
                ),
            }

        gpu_free_mb = recent_gpu_headroom(snap) or 0
        if allow_sequential_gpu and gpu_free_mb >= 4200:
            return {
                "execution_mode": "sequential_gpu_vlm",
                "num_gpu": -1,
                "prefer_lightweight_model": gpu_free_mb < 6200,
                "verify_vram_release": True,
                "evict_comfy_models": False,
                "reason": (
                    f"GPU cold with {gpu_free_mb} MB free VRAM: running sequential GPU VLM with "
                    "verified VRAM release before any subsequent generation."
                ),
            }
        return {
            "execution_mode": "cpu_vlm",
            "num_gpu": 0,
            "prefer_lightweight_model": True,
            "verify_vram_release": False,
            "evict_comfy_models": False,
            "reason": "Default zero-contention CPU-only local VLM inspection (num_gpu=0).",
        }

    def plan_specialist_execution(
        self,
        specialist_id: str,
        *,
        target_family: str = "flux2",
        active_generation_running: bool = False,
        comfy_models_warm: bool = False,
        snapshot: ResourceSnapshot | None = None,
        storage_path: str | Path = "/",
        explicit_acquisition_approval: bool = False,
    ) -> dict[str, Any]:
        """Integrate Sprint 9 specialist models with ResourcePlanner & protected storage reserve."""
        from .specialist_registry import DEFAULT_SPECIALIST_REGISTRY

        spec = DEFAULT_SPECIALIST_REGISTRY.get(specialist_id)
        if spec is None:
            return {"allowed": False, "execution_mode": "blocked", "reason": f"Unknown specialist '{specialist_id}'."}

        ok_compat, compat_err = DEFAULT_SPECIALIST_REGISTRY.check_compatibility(specialist_id, target_family)
        if not ok_compat:
            return {"allowed": False, "execution_mode": "blocked", "reason": compat_err}

        acq = DEFAULT_SPECIALIST_REGISTRY.evaluate_acquisition(
            specialist_id, storage_path=storage_path, explicit_approval=explicit_acquisition_approval
        )
        if not acq.get("allowed"):
            return {"allowed": False, "execution_mode": "deferred_acquisition", "reason": acq["reason"], "acquisition": acq}

        snap = snapshot if snapshot is not None else self.guard.evaluate()
        if snap.ram_severity == "emergency":
            return {"allowed": False, "execution_mode": "blocked", "reason": "Emergency physical RAM pressure; deferring specialist."}

        if spec.min_vram_mb == 0:
            return {
                "allowed": True,
                "execution_mode": "cpu_only",
                "verify_vram_release": False,
                "evict_comfy_models": False,
                "reason": f"Specialist '{spec.name}' runs strictly on CPU with 0 MB VRAM.",
            }

        if active_generation_running or comfy_models_warm:
            return {
                "allowed": True,
                "execution_mode": "deferred_batch_sequential_gpu",
                "verify_vram_release": True,
                "evict_comfy_models": False,
                "reason": (
                    f"Warm/active ComfyUI models present: deferring GPU specialist '{spec.name}' "
                    "to batch inspection window so FLUX/Wan weights are not evicted mid-batch."
                ),
            }

        gpu_free_mb = recent_gpu_headroom(snap) or 0
        if gpu_free_mb >= spec.min_vram_mb:
            return {
                "allowed": True,
                "execution_mode": "sequential_gpu",
                "verify_vram_release": True,
                "evict_comfy_models": False,
                "reason": f"GPU idle ({gpu_free_mb} MB free >= {spec.min_vram_mb} MB): running '{spec.name}' sequentially with verified VRAM release.",
            }
        return {
            "allowed": True,
            "execution_mode": "cpu_fallback",
            "verify_vram_release": False,
            "evict_comfy_models": False,
            "reason": f"Insufficient free VRAM ({gpu_free_mb} MB < {spec.min_vram_mb} MB); falling back to CPU.",
        }
