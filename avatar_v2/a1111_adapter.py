"""Optional Local-Only A1111 Compatibility Adapter for Avatar V2 (Sprint 5, Phase J).

Design principles:
- Strictly opt-in; Native Avatar V2 Python SDK remains the recommended long-term architecture.
- Does NOT attempt to recreate the full Automatic1111 platform.
- Supports ONLY an explicitly documented subset of legacy endpoints:
  - ``POST /sdapi/v1/txt2img``
  - ``POST /sdapi/v1/img2img`` (with explicit authorized reference image)
  - ``GET /sdapi/v1/sd-models``
  - ``GET /sdapi/v1/progress``
- Routes every compatible request through ``StructuredPromptSpec``, ``policy.depicts_minor``,
  ``ModelRegistry``, ``compile_prompt_for_model``, ``ResourcePlanner``, and ``GenerationBroker`` /
  ``RenderQueue`` (local-only by default, host-enforced run limits).
- Explicitly REJECTS unsupported A1111 parameters (hires-fix, alwayson_scripts/ControlNet/ADetailer,
  refiner, tiling, restore_faces, unknown/SDXL checkpoints) rather than silently ignoring them or
  silently substituting incompatible models.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model_registry import DEFAULT_REGISTRY, ModelRegistry
from .prompt_compilers import strip_sdxl_weights
from .script_broker import GenerationBroker
from .sdk_client import AvatarV2Client, RunLimitReached, RunStopped, SDKError

ADAPTER_VERSION = "1.0"

SUPPORTED_ENDPOINTS: tuple[str, ...] = (
    "POST /sdapi/v1/txt2img",
    "POST /sdapi/v1/img2img",
    "GET /sdapi/v1/sd-models",
    "GET /sdapi/v1/progress",
)

SUPPORTED_TXT2IMG_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "seed",
        "steps",
        "cfg_scale",
        "batch_size",
        "n_iter",
        "sampler_name",
        "override_settings",
        "init_images",
        "denoising_strength",
    }
)

# Parameters that MUST be explicitly rejected rather than silently ignored or faked.
UNSUPPORTED_FIELDS_REASONS: dict[str, str] = {
    "enable_hr": "A1111 Hires.fix ('enable_hr') is not supported; request target dimensions or use Avatar V2 upscaling.",
    "hr_scale": "A1111 Hires.fix ('hr_scale') is not supported by the compatibility adapter.",
    "hr_upscaler": "A1111 Hires.fix ('hr_upscaler') is not supported by the compatibility adapter.",
    "alwayson_scripts": "A1111 extension hooks ('alwayson_scripts': ControlNet/ADetailer/etc.) are not supported; migrate to avatar_v2_sdk.",
    "script_name": "A1111 server-side scripts ('script_name') are not supported.",
    "script_args": "A1111 server-side script arguments ('script_args') are not supported.",
    "refiner_checkpoint": "SDXL refiner checkpoints ('refiner_checkpoint') are not supported on FLUX.2.",
    "restore_faces": "Legacy CodeFormer/GFPGAN 'restore_faces' is not supported.",
    "tiling": "Seamless texture 'tiling' is not supported by FLUX.2 workflows.",
    "styles": "A1111 styles.csv lookup ('styles') is not supported; use Avatar V2 StructuredPromptSpec or scene recipes.",
}

# Explicit opt-in checkpoint aliases mapped ONLY to approved local Avatar V2 models.
APPROVED_CHECKPOINT_ALIASES: dict[str, str] = {
    "flux2_4b": "flux2_4b",
    "flux-2-klein-4b": "flux2_4b",
    "flux-2-klein-4b-Q5_K_M.gguf": "flux2_4b",
    "flux2_9b": "flux2_9b",
    "flux-2-klein-9b": "flux2_9b",
    "flux-2-klein-base-9b-Q4_K_M.gguf": "flux2_9b",
}


@dataclass(frozen=True)
class AdapterMappingResult:
    ok: bool
    http_status: int
    body: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class A1111CompatibilityAdapter:
    """Opt-in local-only translator routing compatible A1111 calls through GenerationBroker."""

    def __init__(
        self,
        broker: GenerationBroker,
        *,
        default_model_id: str = "flux2_9b",
        registry: ModelRegistry = DEFAULT_REGISTRY,
    ) -> None:
        self.broker = broker
        self.default_model_id = default_model_id
        self.registry = registry
        self._client = AvatarV2Client(token=broker.token, transport=broker.handle_request)

    def handle_sd_models(self) -> AdapterMappingResult:
        """GET /sdapi/v1/sd-models — list approved local Avatar V2 image models only."""
        models = [
            {
                "title": m.model_id,
                "model_name": m.label,
                "filename": m.components[0].filename if m.components else m.model_id,
                "avatar_v2_family": m.family,
            }
            for m in self.registry.list_models()
            if m.family == "flux2"
        ]
        return AdapterMappingResult(ok=True, http_status=200, body={"models": models})

    def handle_progress(self) -> AdapterMappingResult:
        """GET /sdapi/v1/progress — report broker run progress cleanly."""
        snap = self.broker.snapshot_status()
        total = max(1, int(snap.get("max_items") or 1))
        done = int(snap.get("completed") or 0)
        return AdapterMappingResult(
            ok=True,
            http_status=200,
            body={
                "progress": round(min(1.0, done / total), 3),
                "state": {
                    "job_count": total,
                    "job_no": done,
                    "interrupted": snap.get("state") == "stopped",
                },
                "avatar_v2_run": snap,
            },
        )

    def _validate_payload(self, payload: dict[str, Any], *, allow_img2img: bool) -> tuple[str | None, str, list[str]]:
        """Validate A1111 request payload strictly; reject unsupported params and unknown checkpoints."""
        if not isinstance(payload, dict):
            return "Request payload must be a JSON object.", "", []

        # 1. Explicitly reject unsupported A1111 parameters (never silently ignore!).
        for key, reason in UNSUPPORTED_FIELDS_REASONS.items():
            if key in payload and payload[key] not in (None, False, "", [], {}):
                return reason, "", []

        unknown = sorted(set(payload.keys()) - SUPPORTED_TXT2IMG_FIELDS)
        if unknown:
            return (
                f"Unsupported A1111 parameter(s): {', '.join(unknown)}. "
                "Only documented fields are accepted; migrate to avatar_v2_sdk for full control.",
                "",
                [],
            )

        warnings: list[str] = []
        model_id = self.default_model_id

        # 2. Validate override_settings strictly (no silent substitution of SDXL/Juggernaut!).
        overrides = payload.get("override_settings")
        if overrides is not None:
            if not isinstance(overrides, dict):
                return "override_settings must be an object.", "", []
            unsupported_ov = [k for k in overrides if k != "sd_model_checkpoint"]
            if unsupported_ov:
                return f"Unsupported override_settings key(s): {', '.join(unsupported_ov)}.", "", []
            ckpt = str(overrides.get("sd_model_checkpoint") or "").strip()
            if ckpt:
                mapped = APPROVED_CHECKPOINT_ALIASES.get(ckpt)
                if mapped is None:
                    return (
                        f"Checkpoint '{ckpt}' is not installed or supported by Avatar V2. "
                        f"Approved local image models: {', '.join(sorted(set(APPROVED_CHECKPOINT_ALIASES.values())))}. "
                        "Refusing silent model substitution.",
                        "",
                        [],
                    )
                model_id = mapped

        # 3. Validate batch / n_iter bounds.
        batch_size = int(payload.get("batch_size") or 1)
        n_iter = int(payload.get("n_iter") or 1)
        if batch_size < 1 or n_iter < 1:
            return "batch_size and n_iter must be >= 1.", "", []
        if batch_size * n_iter > self.broker.max_items:
            return (
                f"Requested {batch_size * n_iter} images exceeds host-enforced run limit ({self.broker.max_items}).",
                "",
                [],
            )

        # 4. Validate img2img references cleanly.
        if allow_img2img:
            init_imgs = payload.get("init_images")
            if not isinstance(init_imgs, list) or not init_imgs:
                return "img2img requires at least one authorized reference image in 'init_images'.", "", []
            if len(init_imgs) > 4:
                return "img2img supports at most 4 reference images.", "", []

        if payload.get("sampler_name"):
            warnings.append(
                f"Legacy sampler_name '{payload['sampler_name']}' is mapped to Avatar V2's verified FLUX.2 sampler."
            )
        if payload.get("steps") is not None:
            warnings.append(
                "Custom A1111 'steps' is advisory; Avatar V2 enforces model-verified sampling defaults for FLUX.2."
            )

        return None, model_id, warnings

    def handle_txt2img(self, payload: dict[str, Any], *, wait: bool = True) -> AdapterMappingResult:
        """POST /sdapi/v1/txt2img — translate to StructuredPromptSpec and route via GenerationBroker."""
        err, model_id, warnings = self._validate_payload(payload, allow_img2img=False)
        if err:
            return AdapterMappingResult(ok=False, http_status=400, body={"error": err, "adapter_version": ADAPTER_VERSION})

        raw_prompt = str(payload.get("prompt") or "").strip()
        if not raw_prompt:
            return AdapterMappingResult(ok=False, http_status=400, body={"error": "Prompt cannot be empty."})

        clean_prompt, stripped = strip_sdxl_weights(raw_prompt)
        if stripped:
            warnings.append("Stripped legacy SDXL (token:weight) emphasis syntax for FLUX.2 Qwen3 encoder.")

        neg = str(payload.get("negative_prompt") or "").strip()
        width = int(payload.get("width") or 896)
        height = int(payload.get("height") or 1280)
        seed_raw = payload.get("seed")
        base_seed = None if seed_raw in (None, "", -1) else int(seed_raw)
        total = int(payload.get("batch_size") or 1) * int(payload.get("n_iter") or 1)

        images_b64: list[str] = []
        jobs_meta: list[dict[str, Any]] = []
        for idx in range(total):
            item_seed = (base_seed + idx) if base_seed is not None else None
            try:
                res = self._client.generate_image(
                    clean_prompt,
                    model_id=model_id,
                    width=width,
                    height=height,
                    seed=item_seed,
                    negative_prompt=neg,
                    label=f"A1111 adapter #{idx + 1} · {model_id}",
                    wait=wait,
                )
            except RunLimitReached as exc:
                return AdapterMappingResult(ok=False, http_status=429, body={"error": str(exc), "code": exc.code})
            except RunStopped as exc:
                return AdapterMappingResult(ok=False, http_status=409, body={"error": str(exc), "code": exc.code})
            except SDKError as exc:
                status = 403 if exc.code in {"policy_blocked", "cloud_blocked", "unauthorized_path"} else 400
                return AdapterMappingResult(ok=False, http_status=status, body={"error": str(exc), "code": exc.code})

            jobs_meta.append({"job_id": res.job_id, "status": res.status, "outputs": res.outputs})
            warnings.extend(w for w in res.warnings if w not in warnings)
            for out_path in res.outputs:
                p = Path(out_path)
                if p.is_file():
                    images_b64.append(base64.b64encode(p.read_bytes()).decode("ascii"))

        info = {
            "adapter_version": ADAPTER_VERSION,
            "model_id": model_id,
            "local_only": not self.broker.allow_cloud,
            "warnings": warnings,
            "jobs": jobs_meta,
        }
        return AdapterMappingResult(
            ok=True,
            http_status=200,
            body={"images": images_b64, "parameters": dict(payload), "info": json.dumps(info)},
            warnings=warnings,
        )

    def handle_img2img(self, payload: dict[str, Any], *, wait: bool = True) -> AdapterMappingResult:
        """POST /sdapi/v1/img2img — routes authorized reference images through GenerationBroker."""
        err, model_id, warnings = self._validate_payload(payload, allow_img2img=True)
        if err:
            return AdapterMappingResult(ok=False, http_status=400, body={"error": err, "adapter_version": ADAPTER_VERSION})

        raw_prompt = str(payload.get("prompt") or "").strip()
        if not raw_prompt:
            return AdapterMappingResult(ok=False, http_status=400, body={"error": "Prompt cannot be empty."})

        clean_prompt, stripped = strip_sdxl_weights(raw_prompt)
        if stripped:
            warnings.append("Stripped legacy SDXL (token:weight) emphasis syntax for FLUX.2 Qwen3 encoder.")

        refs = [str(x) for x in (payload.get("init_images") or []) if str(x or "").strip()]
        neg = str(payload.get("negative_prompt") or "").strip()
        width = int(payload.get("width") or 896)
        height = int(payload.get("height") or 1280)
        seed_raw = payload.get("seed")
        seed = None if seed_raw in (None, "", -1) else int(seed_raw)

        try:
            res = self._client.submit_prompt(
                clean_prompt,
                task="image_generate",
                model_id=model_id,
                width=width,
                height=height,
                frames=1,
                seed=seed,
                negative_prompt=neg,
                references=refs,
                label=f"A1111 img2img adapter · {model_id}",
                wait=wait,
            )
        except RunLimitReached as exc:
            return AdapterMappingResult(ok=False, http_status=429, body={"error": str(exc), "code": exc.code})
        except RunStopped as exc:
            return AdapterMappingResult(ok=False, http_status=409, body={"error": str(exc), "code": exc.code})
        except SDKError as exc:
            status = 403 if exc.code in {"policy_blocked", "cloud_blocked", "unauthorized_path"} else 400
            return AdapterMappingResult(ok=False, http_status=status, body={"error": str(exc), "code": exc.code})

        images_b64 = [
            base64.b64encode(Path(out_path).read_bytes()).decode("ascii")
            for out_path in res.outputs
            if Path(out_path).is_file()
        ]
        info = {
            "adapter_version": ADAPTER_VERSION,
            "model_id": model_id,
            "local_only": not self.broker.allow_cloud,
            "warnings": warnings + list(res.warnings),
            "jobs": [{"job_id": res.job_id, "status": res.status, "outputs": res.outputs}],
        }
        return AdapterMappingResult(
            ok=True,
            http_status=200,
            body={"images": images_b64, "parameters": dict(payload), "info": json.dumps(info)},
            warnings=warnings + list(res.warnings),
        )
