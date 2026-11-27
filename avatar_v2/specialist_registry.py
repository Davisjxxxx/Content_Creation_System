"""Specialist Model, Storage Acquisition Gate & Larger-Model Research Registry (Sprint 9).

Extends Avatar V2's existing ModelRegistry and ResourcePlanner rather than creating a
competing subsystem. Classifies every specialist and larger-model candidate honestly by:
- Exact verified metadata (source, license, storage, VRAM, RAM, quantization, offloading)
- Local disk availability (zero unauthorized large downloads)
- Storage reserve compliance (30 GB Normal / 20 GB Warning / 15 GB Emergency Stop)
- ComfyUI / model-family compatibility (explicitly blocking incompatible SDXL ControlNets)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .resource_monitor import STORAGE_EMERGENCY_MIN_GB, STORAGE_NORMAL_MIN_GB, check_storage_for_job

AcquisitionClass = Literal[
    "already_available",
    "ready_for_isolated_evaluation",
    "requires_limited_acquisition",
    "requires_significant_acquisition_approval",
    "operationally_impractical",
    "incompatible",
    "unnecessary",
]

SpecialistRole = Literal[
    "object_localization",
    "segmentation_mask",
    "masked_editing",
    "pose_composition_control",
    "video_motion_control",
    "enhancement",
    "generation_expansion",
    "vision_language",
]


@dataclass(frozen=True)
class SpecialistModelSpec:
    """Metadata, compatibility, and acquisition classification for a Sprint 9 candidate."""

    specialist_id: str
    name: str
    role: SpecialistRole
    target_families: tuple[str, ...]
    official_source: str
    license: str
    storage_mb: int
    min_vram_mb: int
    ram_mb: int
    quantization: str
    offloading_support: str
    comfyui_compatible: bool
    expected_latency_s: str
    classification: AcquisitionClass
    enabled_by_default: bool = False
    requires_new_dependencies: tuple[str, ...] = field(default_factory=tuple)
    demonstrated_advantage: str = ""
    limitations: str = ""
    integration_complexity: Literal["low", "medium", "high"] = "low"
    verification_upgrade_eligible: bool = False

    def public(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SPECIALISTS: tuple[SpecialistModelSpec, ...] = (
    # --- Already available locally (ZERO downloads!) ---
    SpecialistModelSpec(
        specialist_id="qwen3_vl_8b_blind_grid",
        name="Qwen3-VL 8B Blind Inventory + Horizontal Grid Localizer",
        role="object_localization",
        target_families=("flux2", "wan22"),
        official_source="ollama:qwen3-vl:8b-instruct (already installed, 5.72 GB)",
        license="Apache-2.0",
        storage_mb=0,
        min_vram_mb=5800,
        ram_mb=7000,
        quantization="GGUF Q4_K_M",
        offloading_support="Sequential GPU with verified keep_alive=0 VRAM release; CPU fallback supported",
        comfyui_compatible=True,
        expected_latency_s="8-13s seq GPU",
        classification="already_available",
        enabled_by_default=False,  # Opt-in / triggered selectively for count & spatial criteria after calibration
        demonstrated_advantage="Removes directional/count leading bias from VLM prompt and reconciles counts, open/closed state, and x-positions deterministically in Python.",
        limitations="Still shares Qwen3-VL visual encoder; correlated errors remain possible on occluded/tiny objects.",
        integration_complexity="low",
        verification_upgrade_eligible=False,  # Upgraded only if held-out photographic benchmark proves <5% FAR
    ),
    SpecialistModelSpec(
        specialist_id="flux2_masked_latent_composite",
        name="FLUX.2 9B Masked Latent Edit + ImageCompositeMasked",
        role="masked_editing",
        target_families=("flux2",),
        official_source="ComfyUI 0.46 core (SetLatentNoiseMask + DifferentialDiffusion + ReferenceLatent + ImageCompositeMasked) + installed FLUX.2 9B",
        license="FLUX Non-Commercial (9B) / Apache-2.0 (4B & ComfyUI nodes)",
        storage_mb=0,
        min_vram_mb=6200,
        ram_mb=14000,
        quantization="GGUF Q4_K_M",
        offloading_support="CPU Qwen3-8B text encoder + GPU UNet",
        comfyui_compatible=True,
        expected_latency_s="240-415s (28 steps)",
        classification="already_available",
        enabled_by_default=True,
        demonstrated_advantage="Guarantees 0.00% pixel alteration outside feathered edit mask via ImageCompositeMasked while keeping full FLUX.2 9B fidelity inside the mask.",
        limitations="Not a dedicated 9-channel FLUX Fill/Inpaint weight; large geometric swaps spanning outside the mask still benefit from unmasked ReferenceLatent edit or full regen.",
        integration_complexity="low",
    ),
    SpecialistModelSpec(
        specialist_id="pillow_bbox_mask_generator",
        name="Pure-Pillow Bounding-Box & Feathered Region Mask Generator",
        role="segmentation_mask",
        target_families=("flux2",),
        official_source="Built-in avatar_v2/specialist_vision.py (Pillow 11.3)",
        license="MIT / HPND",
        storage_mb=0,
        min_vram_mb=0,
        ram_mb=32,
        quantization="CPU float/uint8",
        offloading_support="100% CPU-only",
        comfyui_compatible=True,
        expected_latency_s="<0.02s CPU",
        classification="already_available",
        enabled_by_default=True,
        demonstrated_advantage="Zero-VRAM deterministic mask synthesis from normalized detector boxes or named composition zones with smooth edge feathering.",
        limitations="Rectangular/elliptical masks only; does not produce pixel-tight hair/finger silhouettes like SAM 2.1.",
        integration_complexity="low",
    ),
    SpecialistModelSpec(
        specialist_id="wan22_a14b_base_cfg",
        name="Wan 2.2 A14B Standard Base Sampler (No Lightning LoRA, Active CFG)",
        role="video_motion_control",
        target_families=("wan22",),
        official_source="Installed Wan2.2-I2V-A14B-HighNoise/LowNoise-Q4_K_S.gguf",
        license="Apache-2.0",
        storage_mb=0,
        min_vram_mb=7600,
        ram_mb=35000,
        quantization="GGUF Q4_K_S",
        offloading_support="CPU UMT5 + DynamicVRAM block streaming + VAEDecodeTiled",
        comfyui_compatible=True,
        expected_latency_s="650-950s (12 steps, CFG 3.5)",
        classification="already_available",
        enabled_by_default=False,  # Experimental until live benchmark confirms net quality-vs-latency benefit
        demonstrated_advantage="Enables real CFG > 1.0 negative prompting and full base-model motion dynamics without 4-step Lightning distillation artifacts.",
        limitations="Significantly slower than 4-step Lightning LoRA (~2.5-3.5x latency).",
        integration_complexity="low",
    ),
    SpecialistModelSpec(
        specialist_id="cached_yolo_world_v2",
        name="YOLO-World v2 Open-Vocabulary Detector (Cached in ~/.cache/huggingface)",
        role="object_localization",
        target_families=("flux2", "wan22"),
        official_source="Bingsu/yolo-world-mirror (yolov8x-worldv2.pt already cached locally, 140 MB)",
        license="GPL-3.0 (Ultralytics runtime) / Apache-2.0 (YOLO-World weights)",
        storage_mb=140,
        min_vram_mb=1200,
        ram_mb=1500,
        quantization="FP16 / FP32",
        offloading_support="CPU or fast sequential GPU (<0.5s)",
        comfyui_compatible=True,
        expected_latency_s="0.2-1.5s",
        classification="ready_for_isolated_evaluation",
        enabled_by_default=False,
        requires_new_dependencies=("ultralytics",),
        demonstrated_advantage="Weights already cached on disk (140 MB) for open-vocabulary bounding boxes and independent object counting.",
        limitations="Requires installing ultralytics package (GPL-3.0) in venv; weak on fine open/closed object states without crop verification.",
        integration_complexity="medium",
    ),
    SpecialistModelSpec(
        specialist_id="cached_adetailer_yolo",
        name="ADetailer Face / Hand / Person-Seg YOLOv8 Bundle (Cached locally)",
        role="segmentation_mask",
        target_families=("flux2", "wan22"),
        official_source="Bingsu/adetailer (face/hand/person-seg .pt already cached locally, 63 MB)",
        license="GPL-3.0 (Ultralytics runtime)",
        storage_mb=63,
        min_vram_mb=800,
        ram_mb=1000,
        quantization="FP16 / FP32",
        offloading_support="CPU or fast sequential GPU",
        comfyui_compatible=True,
        expected_latency_s="0.1-0.8s",
        classification="ready_for_isolated_evaluation",
        enabled_by_default=False,
        requires_new_dependencies=("ultralytics",),
        demonstrated_advantage="Provides person segmentation masks, face boxes, and hand boxes from already-cached 63 MB weights.",
        limitations="Requires ultralytics runtime dependency; does not provide finger-level skeletal landmarks.",
        integration_complexity="medium",
    ),
    # --- Lightweight specialists requiring limited acquisition (deferred behind storage/approval gate) ---
    SpecialistModelSpec(
        specialist_id="grounding_dino_tiny",
        name="Grounding DINO Tiny (Swin-T)",
        role="object_localization",
        target_families=("flux2", "wan22"),
        official_source="https://huggingface.co/IDEA-Research/grounding-dino-tiny",
        license="Apache-2.0",
        storage_mb=694,
        min_vram_mb=1400,
        ram_mb=2200,
        quantization="FP16 / FP32 (native HuggingFace transformers — already installed in AvatarForge venv!)",
        offloading_support="CPU (~2-4s) or sequential GPU (<0.4s)",
        comfyui_compatible=True,
        expected_latency_s="0.4s GPU / 3.0s CPU",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        demonstrated_advantage="Strong open-vocabulary bounding boxes & independent count/left-right spatial verification using Apache-2.0 weights and already-installed transformers.",
        limitations="694 MB download; bounding boxes alone cannot verify subtle open-vs-closed states or 3D depth.",
        integration_complexity="low",
        verification_upgrade_eligible=True,
    ),
    SpecialistModelSpec(
        specialist_id="florence_2_base_ft",
        name="Microsoft Florence-2 Base-FT (230M)",
        role="object_localization",
        target_families=("flux2", "wan22"),
        official_source="https://huggingface.co/microsoft/Florence-2-base-ft",
        license="MIT",
        storage_mb=460,
        min_vram_mb=1200,
        ram_mb=1800,
        quantization="FP16 / FP32",
        offloading_support="CPU or sequential GPU",
        comfyui_compatible=True,
        expected_latency_s="0.5s GPU / 3.5s CPU",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        demonstrated_advantage="Compact MIT-licensed multi-task vision specialist supporting phrase grounding, object detection, region captions, and OCR.",
        limitations="Requires trust_remote_code or native transformers adapter verification; 460 MB download.",
        integration_complexity="medium",
        verification_upgrade_eligible=True,
    ),
    SpecialistModelSpec(
        specialist_id="sam2_1_hiera_tiny",
        name="Meta SAM 2.1 Hiera Tiny",
        role="segmentation_mask",
        target_families=("flux2", "wan22"),
        official_source="https://huggingface.co/facebook/sam2.1-hiera-tiny",
        license="Apache-2.0",
        storage_mb=156,
        min_vram_mb=1100,
        ram_mb=1500,
        quantization="FP16 / BF16",
        offloading_support="CPU or sequential GPU",
        comfyui_compatible=True,
        expected_latency_s="0.3s GPU / 2.5s CPU",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        demonstrated_advantage="Pixel-accurate object and multi-subject masks from detector bounding boxes for precision masked editing.",
        limitations="Requires box/point prompt from a detector or user; deferred while Pillow box masks handle coarse regions cleanly.",
        integration_complexity="medium",
    ),
    SpecialistModelSpec(
        specialist_id="depth_anything_v2_small",
        name="Depth Anything V2 Small (ViT-S, 24.8M)",
        role="pose_composition_control",
        target_families=("flux2", "wan22"),
        official_source="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
        license="Apache-2.0 (Note: Base/Large/Giant are CC-BY-NC-4.0)",
        storage_mb=99,
        min_vram_mb=600,
        ram_mb=900,
        quantization="FP16 / FP32",
        offloading_support="CPU or GPU",
        comfyui_compatible=True,
        expected_latency_s="0.15s GPU / 0.8s CPU",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        demonstrated_advantage="Lightweight relative depth maps (99 MB, Apache 2.0) supporting front/behind depth checks and future depth conditioning.",
        limitations="Relative depth only (not metric 3D); FLUX.2 Klein has no installed Depth ControlNet weights yet.",
        integration_complexity="low",
    ),
    SpecialistModelSpec(
        specialist_id="dwpose_onnx",
        name="DWPose Wholebody ONNX (Body + Hands + Face Landmarks)",
        role="pose_composition_control",
        target_families=("flux2", "wan22"),
        official_source="https://huggingface.co/yzd-v/DWPose",
        license="Apache-2.0",
        storage_mb=210,
        min_vram_mb=700,
        ram_mb=1000,
        quantization="ONNX FP16/FP32",
        offloading_support="CPU via onnxruntime or GPU",
        comfyui_compatible=True,
        expected_latency_s="0.1s GPU / 0.6s CPU",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        requires_new_dependencies=("onnxruntime",),
        demonstrated_advantage="Explicit 2D skeletal + hand keypoints for pose continuity checks and future pose-conditioned generation.",
        limitations="Requires onnxruntime; no FLUX.2 Klein or Wan pose ControlNet generation weights are installed yet.",
        integration_complexity="medium",
    ),
    SpecialistModelSpec(
        specialist_id="realesrgan_x2plus",
        name="Real-ESRGAN x2plus Conservative Upscaler",
        role="enhancement",
        target_families=("flux2", "wan22"),
        official_source="https://github.com/xinntao/Real-ESRGAN",
        license="BSD-3-Clause",
        storage_mb=67,
        min_vram_mb=900,
        ram_mb=1200,
        quantization="FP16 / FP32",
        offloading_support="Tiled GPU or CPU (ComfyUI ImageUpscaleWithModel already wired in workflows_builder.py)",
        comfyui_compatible=True,
        expected_latency_s="1-3s per still",
        classification="requires_limited_acquisition",
        enabled_by_default=False,
        demonstrated_advantage="Conservative 2x resolution upscaling without face-replacement hallucination (unlike GFPGAN/CodeFormer).",
        limitations="Increases pixel resolution only — never recovers factual missing detail; must remain opt-in and preserve originals.",
        integration_complexity="low",
    ),
    # --- Larger generation / editing / video models (Section 11 — significant approval / impractical / incompatible) ---
    SpecialistModelSpec(
        specialist_id="wan22_fun_control_5b",
        name="Wan 2.2 Fun-Control 5B",
        role="video_motion_control",
        target_families=("wan22",),
        official_source="https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control",
        license="Apache-2.0",
        storage_mb=9800,
        min_vram_mb=7500,
        ram_mb=24000,
        quantization="BF16 (9.8 GB) or GGUF Q5_K_M (~4.5 GB)",
        offloading_support="CPU UMT5 + DynamicVRAM block streaming; reuses existing UMT5 (6.3 GB) + Wan2.2 48ch VAE (1.4 GB)",
        comfyui_compatible=True,
        expected_latency_s="220-340s (81 frames)",
        classification="requires_significant_acquisition_approval",
        enabled_by_default=False,
        demonstrated_advantage="Adds Canny/Depth/Pose control-video guidance using ComfyUI's already-installed Wan22FunControlToVideo node and existing UMT5 + Wan2.2 VAE.",
        limitations="Requires ~5-10 GB additional disk space plus a pose/depth preprocessor; current NVMe has only 43 GB free (13 GB above Normal tier).",
        integration_complexity="medium",
    ),
    SpecialistModelSpec(
        specialist_id="qwen_image_edit_gguf_q4",
        name="Qwen-Image-Edit (20B DiT + 8.3B Encoder, GGUF Q4)",
        role="generation_expansion",
        target_families=("qwen_image_edit",),
        official_source="https://huggingface.co/Qwen/Qwen-Image-Edit",
        license="Apache-2.0",
        storage_mb=18500,
        min_vram_mb=12000,
        ram_mb=32000,
        quantization="GGUF Q4_0 / Q4_K_M (~12-13.2 GB DiT + ~5.5 GB TE)",
        offloading_support="Heavy CPU/RAM block offloading required on 8 GB VRAM",
        comfyui_compatible=True,
        expected_latency_s="450-900s on 8 GB VRAM (offloaded)",
        classification="requires_significant_acquisition_approval",
        enabled_by_default=False,
        demonstrated_advantage="Dedicated Apache-2.0 instruction-based image editing model with strong localized text and object edit fidelity.",
        limitations="18.5+ GB storage would push 43 GB free NVMe below the 30 GB Warning threshold; requires explicit storage cleanup/mount & owner approval.",
        integration_complexity="high",
    ),
    SpecialistModelSpec(
        specialist_id="flux2_dev_32b_gguf_q4",
        name="FLUX.2 Dev (32B DiT + 24B Text Encoder, GGUF Q4_K_S)",
        role="generation_expansion",
        target_families=("flux2",),
        official_source="https://huggingface.co/black-forest-labs/FLUX.2-dev",
        license="FLUX Non-Commercial License",
        storage_mb=35000,
        min_vram_mb=16000,
        ram_mb=48000,
        quantization="GGUF Q4_K_S (~19.3 GB DiT + ~15 GB TE)",
        offloading_support="Extreme RAM streaming / PCIe bottleneck on 8 GB VRAM",
        comfyui_compatible=True,
        expected_latency_s=">1200s per image on 8 GB VRAM",
        classification="operationally_impractical",
        enabled_by_default=False,
        demonstrated_advantage="Higher base parameter count than FLUX.2 Klein 9B.",
        limitations="35+ GB footprint would breach the 15 GB Emergency Stop storage reserve immediately; severe PCIe streaming slowdown on 8 GB RTX 4070.",
        integration_complexity="high",
    ),
    SpecialistModelSpec(
        specialist_id="sdxl_controlnet_pack",
        name="SDXL ControlNet (Canny / Depth / OpenPose)",
        role="pose_composition_control",
        target_families=("sdxl",),
        official_source="Legacy SDXL ecosystem",
        license="Various",
        storage_mb=2500,
        min_vram_mb=6000,
        ram_mb=8000,
        quantization="FP16",
        offloading_support="N/A",
        comfyui_compatible=False,
        expected_latency_s="N/A",
        classification="incompatible",
        enabled_by_default=False,
        demonstrated_advantage="None for Avatar V2.",
        limitations="Architecturally INCOMPATIBLE with FLUX.2 and Wan 2.2 DiT architectures; must never be wired into FLUX.2 or Wan graphs.",
        integration_complexity="high",
    ),
)


class SpecialistRegistry:
    """Registry & acquisition gate for Sprint 9 specialist and larger-model candidates."""

    def __init__(self, specs: tuple[SpecialistModelSpec, ...] = DEFAULT_SPECIALISTS) -> None:
        self._specs: dict[str, SpecialistModelSpec] = {s.specialist_id: s for s in specs}

    def get(self, specialist_id: str) -> SpecialistModelSpec | None:
        return self._specs.get(specialist_id)

    def list_all(self) -> list[SpecialistModelSpec]:
        return list(self._specs.values())

    def list_by_role(self, role: SpecialistRole) -> list[SpecialistModelSpec]:
        return [s for s in self._specs.values() if s.role == role]

    def check_compatibility(self, specialist_id: str, target_family: str) -> tuple[bool, str]:
        """Verify strict model-family compatibility (blocks SDXL ControlNets on FLUX.2/Wan)."""
        spec = self.get(specialist_id)
        if spec is None:
            return False, f"Unknown specialist '{specialist_id}'."
        if spec.classification == "incompatible" or not spec.comfyui_compatible:
            return False, f"Specialist '{spec.name}' is incompatible: {spec.limitations}"
        if target_family not in spec.target_families:
            return (
                False,
                f"Specialist '{spec.name}' targets {spec.target_families}, incompatible with '{target_family}'.",
            )
        return True, "ok"

    def evaluate_acquisition(
        self,
        specialist_id: str,
        *,
        storage_path: str | Path = "/",
        explicit_approval: bool = False,
        temp_overhead_multiplier: float = 1.25,
    ) -> dict[str, Any]:
        """Enforce protected storage reserve & explicit approval gate before any model download."""
        spec = self.get(specialist_id)
        if spec is None:
            return {"allowed": False, "reason": f"Unknown specialist '{specialist_id}'."}
        if spec.classification == "already_available":
            return {
                "allowed": True,
                "already_available": True,
                "required_mb": 0,
                "classification": spec.classification,
                "reason": "Already available locally (zero download required).",
            }
        if spec.classification in {"incompatible", "operationally_impractical", "unnecessary"}:
            return {
                "allowed": False,
                "already_available": False,
                "required_mb": spec.storage_mb,
                "classification": spec.classification,
                "reason": f"Blocked ({spec.classification}): {spec.limitations}",
            }

        total_bytes = int(spec.storage_mb * temp_overhead_multiplier * 1024 * 1024)
        s_check = check_storage_for_job(storage_path, estimated_bytes=total_bytes)
        if not s_check.allowed:
            return {
                "allowed": False,
                "already_available": False,
                "required_mb": round(total_bytes / (1024 * 1024), 1),
                "classification": spec.classification,
                "storage_check": s_check.public(),
                "reason": f"Blocked by protected storage reserve ({STORAGE_EMERGENCY_MIN_GB} GB emergency / {STORAGE_NORMAL_MIN_GB} GB normal): {s_check.reason}",
            }
        if not explicit_approval:
            return {
                "allowed": False,
                "already_available": False,
                "required_mb": round(total_bytes / (1024 * 1024), 1),
                "classification": spec.classification,
                "storage_check": s_check.public(),
                "reason": f"Model acquisition '{spec.name}' ({spec.storage_mb} MB) requires explicit user authorization.",
            }
        return {
            "allowed": True,
            "already_available": False,
            "required_mb": round(total_bytes / (1024 * 1024), 1),
            "classification": spec.classification,
            "storage_check": s_check.public(),
            "reason": "Authorized and within protected storage reserve.",
        }


DEFAULT_SPECIALIST_REGISTRY = SpecialistRegistry()
