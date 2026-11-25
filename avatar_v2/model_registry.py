"""Component-aware Model Capability & Residency Registry (Sprint 3, Section 4).

Single source of truth for verified local model components, file identity, loader
compatibility, shared-component overlap, and model capabilities. Designed so
ResourcePlanner, PromptCompilers, and Sprint 4 Script Mode can query capabilities
without hardcoding model-family branches across the codebase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ComponentRole = Literal["diffusion", "text_encoder", "vae", "lora", "audio_vae"]
TransitionKind = Literal["cold_start", "same_model", "shared_components", "cross_family"]


@dataclass(frozen=True)
class ComponentSpec:
    """One verified model file and its loader / compatibility identity."""

    component_id: str
    role: ComponentRole
    subfolder: str
    filename: str
    loader: str
    format_type: Literal["gguf", "safetensors"]
    approx_mb: int
    compat_group: str

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        """Exact identity required before two workflows may be treated as sharing a component."""
        return (self.subfolder, self.filename, self.loader, self.compat_group)

    def exists_in(self, comfyui_dir: Path | None) -> bool:
        if comfyui_dir is None or not comfyui_dir.is_dir():
            return False
        return (comfyui_dir / "models" / self.subfolder / self.filename).is_file()

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    """Capability & component-dependency profile for one generation model."""

    model_id: str
    family: str
    label: str
    workflow: str
    tasks: tuple[str, ...]
    license: str
    quantization: str
    components: tuple[ComponentSpec, ...]
    supports_negative_prompt: bool = True
    negative_prompt_strength: Literal["strong", "light", "ignored_cfg1"] = "strong"
    supports_motion: bool = False
    supports_camera_movement: bool = False
    supports_reference_image: bool = False
    requires_start_image: bool = False
    supports_multi_ref: bool = False
    supports_edit: bool = False
    default_width: int = 896
    default_height: int = 1280
    default_frames: int = 1
    estimated_peak_vram_mb: int = 6000
    supports_cpu_text_encoder: bool = True
    fallback_model_id: str | None = None

    @property
    def total_weight_mb(self) -> int:
        return sum(c.approx_mb for c in self.components)

    @property
    def component_keys(self) -> dict[tuple[str, str, str, str], ComponentSpec]:
        return {c.identity_key: c for c in self.components}

    def public(self, comfyui_dir: Path | None = None) -> dict[str, Any]:
        missing = [c.filename for c in self.components if comfyui_dir is not None and not c.exists_in(comfyui_dir)]
        return {
            "model_id": self.model_id,
            "family": self.family,
            "label": self.label,
            "workflow": self.workflow,
            "tasks": list(self.tasks),
            "license": self.license,
            "quantization": self.quantization,
            "total_weight_mb": self.total_weight_mb,
            "estimated_peak_vram_mb": self.estimated_peak_vram_mb,
            "supports_negative_prompt": self.supports_negative_prompt,
            "negative_prompt_strength": self.negative_prompt_strength,
            "supports_motion": self.supports_motion,
            "supports_camera_movement": self.supports_camera_movement,
            "supports_reference_image": self.supports_reference_image,
            "requires_start_image": self.requires_start_image,
            "supports_multi_ref": self.supports_multi_ref,
            "supports_edit": self.supports_edit,
            "fallback_model_id": self.fallback_model_id,
            "installed": (len(missing) == 0) if comfyui_dir is not None else None,
            "missing_files": missing,
            "components": [c.public() for c in self.components],
        }


@dataclass(frozen=True)
class ComponentTransition:
    """Exact component reuse and eviction analysis between two models."""

    from_model_id: str | None
    to_model_id: str
    transition_kind: TransitionKind
    reusable: tuple[ComponentSpec, ...]
    evict: tuple[ComponentSpec, ...]
    required: tuple[ComponentSpec, ...]
    reusable_mb: int = 0
    evict_mb: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "from_model_id": self.from_model_id,
            "to_model_id": self.to_model_id,
            "transition_kind": self.transition_kind,
            "reusable": [c.component_id for c in self.reusable],
            "evict": [c.component_id for c in self.evict],
            "required": [c.component_id for c in self.required],
            "reusable_mb": self.reusable_mb,
            "evict_mb": self.evict_mb,
        }


# Verified component definitions matching installed files in /home/jd/AvatarForge/comfyui/models/.
_FLUX2_VAE = ComponentSpec("flux2_vae", "vae", "vae", "flux2-vae.safetensors", "VAELoader", "safetensors", 160, "flux2_vae_16ch")
_QWEN3_4B = ComponentSpec("qwen3_4b", "text_encoder", "text_encoders", "Qwen3-4B-Q5_K_M.gguf", "CLIPLoaderGGUF", "gguf", 2756, "flux2_te_7680")
_QWEN3_8B = ComponentSpec("qwen3_8b", "text_encoder", "text_encoders", "Qwen3-8B-Q4_K_M.gguf", "CLIPLoaderGGUF", "gguf", 4795, "flux2_te_12288")
_FLUX2_4B_UNET = ComponentSpec("flux2_4b_unet", "diffusion", "diffusion_models", "flux-2-klein-4b-Q5_K_M.gguf", "UnetLoaderGGUF", "gguf", 2931, "flux2_klein_4b")
_FLUX2_9B_UNET = ComponentSpec("flux2_9b_unet", "diffusion", "diffusion_models", "flux-2-klein-base-9b-Q4_K_M.gguf", "UnetLoaderGGUF", "gguf", 5635, "flux2_klein_9b")

_WAN_UMT5 = ComponentSpec("wan_umt5_fp8", "text_encoder", "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "CLIPLoader", "safetensors", 6424, "wan_umt5_xxl")
_WAN21_VAE = ComponentSpec("wan21_vae_16ch", "vae", "vae", "wan_2.1_vae.safetensors", "VAELoader", "safetensors", 242, "wan_vae_16ch")
_WAN22_VAE = ComponentSpec("wan22_vae_48ch", "vae", "vae", "Wan2.2_VAE.safetensors", "VAELoader", "safetensors", 1344, "wan_vae_48ch")
_WAN14B_HIGH = ComponentSpec("wan22_14b_high", "diffusion", "diffusion_models", "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf", "UnetLoaderGGUF", "gguf", 8343, "wan22_a14b_i2v")
_WAN14B_LOW = ComponentSpec("wan22_14b_low", "diffusion", "diffusion_models", "Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf", "UnetLoaderGGUF", "gguf", 8343, "wan22_a14b_i2v")
_WAN14B_LORA_HIGH = ComponentSpec("wan22_lightning_high", "lora", "loras", "wan2.2_i2v_lightning_4step_high.safetensors", "LoraLoaderModelOnly", "safetensors", 1170, "wan22_a14b_i2v")
_WAN14B_LORA_LOW = ComponentSpec("wan22_lightning_low", "lora", "loras", "wan2.2_i2v_lightning_4step_low.safetensors", "LoraLoaderModelOnly", "safetensors", 1170, "wan22_a14b_i2v")
_WAN5B_UNET = ComponentSpec("wan22_5b_turbo", "diffusion", "diffusion_models", "Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf", "UnetLoaderGGUF", "gguf", 3639, "wan22_5b_ti2v")
_WAN_FUNCONTROL_UNET = ComponentSpec("wan22_funcontrol_5b", "diffusion", "diffusion_models", "wan2.2_fun_control_5B_bf16.safetensors", "UNETLoader", "safetensors", 9800, "wan22_funcontrol")


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="flux2_4b",
        family="flux2",
        label="FLUX.2 klein 4B (Q5_K_M)",
        workflow="builtin:flux2_txt2img",
        tasks=("image_generate",),
        license="apache-2.0",
        quantization="GGUF Q5_K_M",
        components=(_FLUX2_4B_UNET, _QWEN3_4B, _FLUX2_VAE),
        supports_negative_prompt=True,
        negative_prompt_strength="light",
        default_width=896,
        default_height=1280,
        default_frames=1,
        estimated_peak_vram_mb=5650,
        supports_cpu_text_encoder=True,
    ),
    ModelSpec(
        model_id="flux2_9b",
        family="flux2",
        label="FLUX.2 klein 9B Base (Q4_K_M)",
        workflow="builtin:flux2_txt2img",
        tasks=("image_generate",),
        license="flux-non-commercial",
        quantization="GGUF Q4_K_M",
        components=(_FLUX2_9B_UNET, _QWEN3_8B, _FLUX2_VAE),
        supports_negative_prompt=True,
        negative_prompt_strength="strong",
        default_width=896,
        default_height=1280,
        default_frames=1,
        estimated_peak_vram_mb=7300,
        supports_cpu_text_encoder=True,
        fallback_model_id="flux2_4b",
    ),
    ModelSpec(
        model_id="wan22_a14b",
        family="wan22",
        label="Wan 2.2 A14B + Lightning (Q4_K_S)",
        workflow="workflows/wan22_i2v_14b_gguf_api.json",
        tasks=("video_generate", "video_scene"),
        license="apache-2.0",
        quantization="GGUF Q4_K_S",
        components=(_WAN14B_HIGH, _WAN14B_LOW, _WAN14B_LORA_HIGH, _WAN14B_LORA_LOW, _WAN_UMT5, _WAN21_VAE),
        supports_negative_prompt=True,
        negative_prompt_strength="ignored_cfg1",
        supports_motion=True,
        supports_camera_movement=True,
        supports_reference_image=True,
        requires_start_image=True,
        default_width=480,
        default_height=832,
        default_frames=81,
        estimated_peak_vram_mb=7650,
        supports_cpu_text_encoder=True,
        fallback_model_id="wan22_5b",
    ),
    ModelSpec(
        model_id="wan22_5b",
        family="wan22",
        label="Wan 2.2 TI2V 5B Turbo (Q5_K_M)",
        workflow="workflows/wan22_i2v_5b_turbo_gguf_api.json",
        tasks=("video_generate", "video_scene"),
        license="apache-2.0",
        quantization="GGUF Q5_K_M",
        components=(_WAN5B_UNET, _WAN_UMT5, _WAN22_VAE),
        supports_negative_prompt=True,
        negative_prompt_strength="ignored_cfg1",
        supports_motion=True,
        supports_camera_movement=True,
        supports_reference_image=True,
        requires_start_image=True,
        default_width=480,
        default_height=832,
        default_frames=81,
        estimated_peak_vram_mb=6600,
        supports_cpu_text_encoder=True,
    ),
    ModelSpec(
        model_id="wan22_funcontrol",
        family="wan22",
        label="Wan 2.2 Fun-Control 5B",
        workflow="builtin:wan22_funcontrol",
        tasks=("video_generate",),
        license="apache-2.0",
        quantization="BF16",
        components=(_WAN_FUNCONTROL_UNET, _WAN_UMT5, _WAN22_VAE),
        supports_negative_prompt=True,
        negative_prompt_strength="strong",
        supports_motion=True,
        supports_camera_movement=True,
        supports_reference_image=True,
        requires_start_image=True,
        default_width=832,
        default_height=480,
        default_frames=81,
        estimated_peak_vram_mb=7500,
        supports_cpu_text_encoder=True,
        fallback_model_id="wan22_5b",
    ),
)


class ModelRegistry:
    """Extensible registry of local generation models and component-level compatibility."""

    def __init__(self, models: tuple[ModelSpec, ...] = DEFAULT_MODELS) -> None:
        self._models: dict[str, ModelSpec] = {m.model_id: m for m in models}

    def register(self, spec: ModelSpec) -> None:
        self._models[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def list_models(self) -> list[ModelSpec]:
        return list(self._models.values())

    def resolve_model_id(self, workflow: str, runtime: dict[str, Any] | None = None, asset_map: dict[str, Any] | None = None) -> str | None:
        """Identify the canonical model_id for a queued workflow/runtime/asset_map."""
        rt = runtime or {}
        am = asset_map or {}
        explicit = str(rt.get("registry_model_id") or "").strip()
        if explicit in self._models:
            return explicit
        wf = str(workflow or "")
        wf_name = Path(wf).name if wf and not wf.startswith("builtin:") else wf
        if wf == "builtin:flux2_txt2img":
            unet = str(am.get("FORGE_UNET") or "").lower()
            model_lbl = str(rt.get("model") or "").lower()
            if "9b" in unet or "9b" in model_lbl:
                return "flux2_9b"
            return "flux2_4b"
        if wf_name == "wan22_i2v_14b_gguf_api.json":
            return "wan22_a14b"
        if wf_name == "wan22_i2v_5b_turbo_gguf_api.json":
            return "wan22_5b"
        if wf == "builtin:wan22_funcontrol":
            return "wan22_funcontrol"
        return None

    def diff_components(self, current_model_id: str | None, next_model_id: str) -> ComponentTransition:
        """Compute exact reusable vs evicted components between two model IDs.

        Only marks a component reusable when subfolder, filename, loader, AND compat_group match
        identically (prevents false sharing of Qwen3-4B vs Qwen3-8B or 16-ch vs 48-ch Wan VAEs).
        """
        nxt = self._models[next_model_id]
        if not current_model_id or current_model_id not in self._models:
            return ComponentTransition(
                from_model_id=current_model_id,
                to_model_id=next_model_id,
                transition_kind="cold_start",
                reusable=(),
                evict=(),
                required=nxt.components,
                reusable_mb=0,
                evict_mb=0,
            )
        cur = self._models[current_model_id]
        if cur.model_id == nxt.model_id:
            return ComponentTransition(
                from_model_id=cur.model_id,
                to_model_id=nxt.model_id,
                transition_kind="same_model",
                reusable=nxt.components,
                evict=(),
                required=(),
                reusable_mb=nxt.total_weight_mb,
                evict_mb=0,
            )

        cur_keys = cur.component_keys
        nxt_keys = nxt.component_keys
        reusable = tuple(spec for k, spec in nxt_keys.items() if k in cur_keys)
        evict = tuple(spec for k, spec in cur_keys.items() if k not in nxt_keys)
        required = tuple(spec for k, spec in nxt_keys.items() if k not in cur_keys)

        # Only treat intra-family transitions with a meaningfully large reusable component
        # (e.g., 6.4 GB UMT5 across Wan 14B <-> 5B) as shared_components; swapping 3-5.6 GB
        # FLUX.2 4B <-> 9B UNets shares only a tiny 160 MB VAE and is safer as a full eviction.
        reusable_mb = sum(c.approx_mb for c in reusable)
        if cur.family == nxt.family and reusable_mb >= 512:
            kind: TransitionKind = "shared_components"
        else:
            kind = "cross_family"
        return ComponentTransition(
            from_model_id=cur.model_id,
            to_model_id=nxt.model_id,
            transition_kind=kind,
            reusable=reusable,
            evict=evict,
            required=required,
            reusable_mb=reusable_mb,
            evict_mb=sum(c.approx_mb for c in evict),
        )

    def inventory(self, comfyui_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
        root = Path(comfyui_dir).expanduser() if comfyui_dir else None
        return {mid: spec.public(root) for mid, spec in self._models.items()}


DEFAULT_REGISTRY = ModelRegistry()
