from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .composer import build_job
from .avatar_designer import build_designer_prompt, coverage_report, designer_spec, slot_by_id
from .h3_runtime import H3_FPS, preset_by_id, presets_payload, valid_frame_count
from .library import AnatomyGuide, AvatarProfile, LibraryStore
from .models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from .policy import evaluate_policy
from .prompt_presets import adult_prompt_presets_payload
from .reference_cache import scan_reference_cache
from .providers import ComfyUIProvider
from .render_queue import QueuedRender, RenderQueue
from .runtime_profiles import payload as runtime_profiles_payload
from .runtime_profiles import resolve_profile
from .workflows_builder import WAN_FPS, snap_wan_canvas, wan_frame_count

APP_HOME = Path.home() / ".avatar_v2"
CONFIG_PATH = APP_HOME / "desktop.json"
RUNS_DIR = APP_HOME / "runs"
QUEUE_DIR = APP_HOME / "queue"
HISTORY_DIR = APP_HOME / "history"
LIBRARY_DIR = APP_HOME / "library"
COMFY_PID_PATH = APP_HOME / "comfyui.pid"
COMFY_LOG_PATH = APP_HOME / "logs" / "comfyui.log"

BUNDLED_WORKFLOWS: dict[str, Path] = {
    "ad_sd15": Path(__file__).resolve().parent.parent / "workflows" / "animatediff_sd15_api.json",
    "ad_sdxl": Path(__file__).resolve().parent.parent / "workflows" / "animatediff_sdxl_api.json",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "comfyui_url": "http://127.0.0.1:8188",
    "comfyui_input": "",
    "comfyui_output": "",
    "comfyui_dir": "",
    "comfyui_python": "",
    "workflow_h3_ref2va": "",
    "workflow_h3_fl2va": "",
    "workflow_wan22": "",
    "workflow_custom": "",
    "workflow_ad_sd15": "",
    "workflow_ad_sdxl": "",
    "h3_text_encoder": "",
    "h3_vae": "",
    "h3_audio_vae": "",
    "wan_diffusion_model": "",
    "wan_text_encoder": "",
    "wan_vae": "",
    "wan_lora": "",
    "wan_lora_strength": 1.0,
    "upscale_model": "4x-UltraSharp.pth",
    "timeout_s": 1800,
    "local_h3_authorized": False,
    "h3_local_fallback": "wan22",
    "default_h3_preset": "vertical_4070",
    "default_runtime_profile": "auto",
    "gpu_autoclean": True,
    "gpu_threshold": 0.85,
    "gpu_check_interval_s": 5,
    "preferred_output_dir": "",
    "adult_prompt_custom": {},
}

MAX_DESIGNER_BODY_REFS = 8  # H3 accepts nine pictures; reserve one for primary identity.


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clean_paths(items: list[str] | None) -> list[str]:
    return [str(Path(item).expanduser()) for item in (items or []) if item]


def _run_id(prefix: str = "render") -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{now}"


def _reference_bindings(job: RenderJob, resolved: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize reference wiring without returning local file paths."""

    asset_map = job.asset_map
    image_paths: list[str | None] = [
        job.shot.references.init_image or (job.avatar.identity_refs[0] if job.avatar.identity_refs else None),
        *job.avatar.identity_refs,
    ]
    if job.shot.content_class != "general":
        image_paths.extend(guide.path for guide in job.avatar.anatomy_guides)
    image_paths.extend(job.avatar.body_refs)
    image_paths.extend(job.avatar.hair_refs)
    image_paths.extend(job.avatar.wardrobe_refs)
    image_paths.append(job.shot.references.scene_image)
    image_paths.extend(job.shot.references.extra_images)
    unique_image_sources = len(dict.fromkeys(str(path) for path in image_paths if path))
    bound_h3_pictures = int(asset_map.get("H3_PICTURE_COUNT") or 0)
    summary: dict[str, Any] = {
        "source_counts": {
            "identity": len(job.avatar.identity_refs),
            "body": len(job.avatar.body_refs),
            "hair": len(job.avatar.hair_refs),
            "wardrobe": len(job.avatar.wardrobe_refs),
            "anatomy_guides": len(job.avatar.anatomy_guides),
            "extra_images": len(job.shot.references.extra_images),
            "videos": int(bool(job.shot.references.motion_video)) + len(job.shot.references.extra_videos),
            "audios": int(bool(job.shot.references.audio)) + len(job.shot.references.extra_audios),
        },
        "primary_identity_bound": bool(asset_map.get("INIT_IMAGE")),
        "h3_pictures": bound_h3_pictures,
        "h3_picture_sources": unique_image_sources if job.selected_engine == "h3_ref2va" else 0,
        "h3_pictures_omitted": (
            max(unique_image_sources - bound_h3_pictures, 0)
            if job.selected_engine == "h3_ref2va"
            else 0
        ),
        "h3_videos": int(asset_map.get("H3_VIDEO_COUNT") or 0),
        "h3_audios": int(asset_map.get("H3_AUDIO_COUNT") or 0),
        "reference_fidelity": str(asset_map.get("H3_REF_IMAGE_SIZE") or "max"),
    }
    if resolved is not None:
        nodes = list(resolved.values())
        task = next(
            (node for node in nodes if node.get("class_type") in {"MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo"}),
            None,
        )
        inputs = (task or {}).get("inputs", {})
        summary["resolved"] = {
            "load_images": sum(node.get("class_type") == "LoadImage" for node in nodes),
            "load_videos": sum(node.get("class_type") == "LoadVideo" for node in nodes),
            "load_audios": sum(node.get("class_type") == "LoadAudio" for node in nodes),
            "task": (task or {}).get("class_type"),
            "picture_links": len(inputs.get("ref_images") or []),
            "video_links": len(inputs.get("ref_videos") or []),
            "audio_links": len(inputs.get("ref_audios") or []),
            "first_frame_bound": bool(inputs.get("first_frame")),
            "last_frame_bound": bool(inputs.get("last_frame")),
        }
    return summary


class DesktopAPI:
    def __init__(self) -> None:
        self._self_window = None
        APP_HOME.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (APP_HOME / "logs").mkdir(parents=True, exist_ok=True)
        self.library = LibraryStore(LIBRARY_DIR)
        self.queue = RenderQueue(QUEUE_DIR, history_dir=HISTORY_DIR)
        settings = self._settings()
        self.queue.memory_manager.update_settings(
            enabled=bool(settings.get("gpu_autoclean", True)),
            threshold=float(settings.get("gpu_threshold", 0.85)),
            check_interval_s=float(settings.get("gpu_check_interval_s", 5)),
        )

    def _settings(self) -> dict[str, Any]:
        settings = {**DEFAULT_SETTINGS, **_read_json(CONFIG_PATH, {})}
        for key, path in BUNDLED_WORKFLOWS.items():
            settings_key = f"workflow_{key}"
            if not settings.get(settings_key):
                settings[settings_key] = str(path)
        return settings

    def get_state(self) -> dict[str, Any]:
        settings = self._settings()
        return {
            "app": "Avatar V2",
            "version": "0.3.0",
            "platform": platform.platform(),
            "settings": settings,
            "app_home": str(APP_HOME),
            "runs_dir": str(RUNS_DIR),
            "history_dir": str(HISTORY_DIR),
            "library_dir": str(LIBRARY_DIR),
            "h3_presets": presets_payload(),
            "runtime_profiles": runtime_profiles_payload(),
            "adult_prompt_presets": adult_prompt_presets_payload(settings.get("adult_prompt_custom")),
            "avatar_designer": designer_spec(),
            "h3_license_notice": (
                "Local MiniMax H3 open-weight execution stays disabled until you explicitly confirm "
                "that your use is permitted by the current H3 license or a separate written MiniMax license."
            ),
            "engines": [
                {"id": "h3_ref2va", "name": "MiniMax H3 · Ref2VA", "tag": "MULTI-REFERENCE"},
                {"id": "h3_fl2va", "name": "MiniMax H3 · FL2VA", "tag": "FIRST / LAST"},
                {"id": "wan22", "name": "Wan 2.2", "tag": "LOCAL FALLBACK"},
                {"id": "custom", "name": "Custom ComfyUI", "tag": "WORKFLOW"},
            ],
            "motion_categories": [
                "walk_confident", "walk_casual", "turn_and_smile", "sit_down", "stand_up",
                "hair_adjustment", "jacket_adjustment", "phone_selfie_walk", "natural_laugh",
                "look_away_return", "camera_push", "tracking_walk", "custom",
            ],
        }

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        merged = {**self._settings(), **settings}
        _write_json(CONFIG_PATH, merged)
        self.queue.memory_manager.update_settings(
            enabled=bool(merged.get("gpu_autoclean", True)),
            threshold=float(merged.get("gpu_threshold", 0.85)),
            check_interval_s=float(merged.get("gpu_check_interval_s", 5)),
        )
        return {"ok": True, "settings": merged}

    def save_adult_prompt_option(self, category: str, label: str, prompt: str) -> dict[str, Any]:
        category = str(category or "").strip()
        label = str(label or "").strip()
        prompt = str(prompt or "").strip()
        valid_categories = set(adult_prompt_presets_payload()["categories"])
        if category not in valid_categories:
            return {"ok": False, "error": "Unknown adult prompt category."}
        if not label or not prompt:
            return {"ok": False, "error": "A label and prompt text are required."}
        if len(label) > 80 or len(prompt) > 600:
            return {"ok": False, "error": "Custom labels are limited to 80 characters and prompts to 600 characters."}

        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "option"
        settings = {**DEFAULT_SETTINGS, **_read_json(CONFIG_PATH, {})}
        customs = settings.get("adult_prompt_custom")
        if not isinstance(customs, dict):
            customs = {}
        category_options = customs.get(category)
        if not isinstance(category_options, list):
            category_options = []
        used_ids = {
            str(item.get("id"))
            for item in adult_prompt_presets_payload(customs)["categories"][category]
            if isinstance(item, dict)
        }
        option_id = f"custom_{slug}"
        suffix = 2
        while option_id in used_ids:
            option_id = f"custom_{slug}_{suffix}"
            suffix += 1
        item = {"id": option_id, "label": label, "prompt": prompt}
        category_options.append(item)
        customs = {**customs, category: category_options}
        settings["adult_prompt_custom"] = customs
        _write_json(CONFIG_PATH, settings)
        return {"ok": True, "item": {**item, "custom": True}, "presets": adult_prompt_presets_payload(customs)}

    def choose_files(self, kind: str, multiple: bool = True) -> list[str]:
        if self._self_window is None:
            return []
        import webview

        filters = {
            "image": ("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)",),
            "video": ("Video (*.mp4;*.mov;*.webm;*.mkv)",),
            "audio": ("Audio (*.wav;*.mp3;*.flac;*.m4a;*.ogg)",),
            "workflow": ("ComfyUI API JSON (*.json)",),
            "all": ("All files (*.*)",),
        }
        result = self._self_window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=multiple,
            file_types=filters.get(kind, ("All files (*.*)",)),
        )
        return [str(Path(p)) for p in (result or [])]

    def choose_folder(self) -> str:
        if self._self_window is None:
            return ""
        import webview

        result = self._self_window.create_file_dialog(webview.FOLDER_DIALOG)
        return str(Path(result[0])) if result else ""

    def avatar_designer_scan_cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, **scan_reference_cache(str(payload.get("cache_dir") or ""))}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _archive_dir(self) -> Path | None:
        preferred = str(self._settings().get("preferred_output_dir") or "").strip()
        if preferred:
            return Path(preferred).expanduser()
        return RUNS_DIR / "outputs"

    def _provider_config(self, payload: dict[str, Any]) -> ProviderConfig:
        settings = {**self._settings(), **payload.get("settings", {})}
        return ProviderConfig(
            base_url=settings.get("comfyui_url") or DEFAULT_SETTINGS["comfyui_url"],
            input_dir=settings.get("comfyui_input") or None,
            output_dir=settings.get("comfyui_output") or None,
            timeout_s=int(settings.get("timeout_s") or 1800),
        )

    def doctor(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        settings = {**self._settings(), **payload.get("settings", {})}
        config = self._provider_config(payload)
        provider = ComfyUIProvider(config)
        report: dict[str, Any] = {
            "ok": False,
            "comfyui": False,
            "h3_available": False,
            "url": config.base_url,
        }
        try:
            stats = provider.health()
            caps = provider.capabilities()
            report.update(caps)
            report.update({"ok": True, "comfyui": True, "system_stats": stats})
            for node_name, input_name, field in (
                ("CheckpointLoaderSimple", "ckpt_name", "checkpoints"),
                ("ADE_LoadAnimateDiffModel", "model_name", "motion_modules"),
                ("KSampler", "sampler_name", "samplers"),
                ("ADE_AnimateDiffLoaderWithContext", "beta_schedule", "beta_schedules"),
            ):
                try:
                    spec = provider.object_info(node_name)[node_name]["input"]["required"][input_name]
                    from .providers.comfyui import _combo_options

                    report[field] = _combo_options(spec)
                except (KeyError, TypeError, httpx.HTTPError):
                    report[field] = []
            diffusion_models = caps.get("models", {}).get("diffusion_models", [])
            for profile_id in ("auto", "quality", "int8_12gb", "low_memory", "low_memory_quality"):
                report.setdefault("profile_preview", {})[profile_id] = resolve_profile(
                    profile_id=profile_id,
                    engine="h3_ref2va",
                    system_stats=stats,
                    diffusion_models=diffusion_models,
                    local_h3_authorized=bool(settings.get("local_h3_authorized")),
                    requested_preset=settings.get("default_h3_preset"),
                    duration_s=6,
                )
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    def _models_from_payload(self, payload: dict[str, Any]) -> tuple[AvatarManifest, ShotSpec]:
        avatar_payload = payload.get("avatar", {})
        avatar = AvatarManifest.model_validate(
            {
                "avatar_id": avatar_payload.get("avatar_id") or "desktop-avatar",
                "display_name": avatar_payload.get("display_name") or "Desktop Avatar",
                "apparent_age_years": avatar_payload.get("apparent_age_years"),
                "subjects": [
                    {
                        "id": avatar_payload.get("avatar_id") or "desktop-avatar",
                        "kind": avatar_payload.get("subject_kind", "unknown"),
                        "age_verified_18_plus": bool(avatar_payload.get("age_verified_18_plus", False)),
                        "consent_confirmed": bool(avatar_payload.get("consent_confirmed", False)),
                    }
                ],
                "identity_refs": _clean_paths(avatar_payload.get("identity_refs")),
                "body_refs": _clean_paths(avatar_payload.get("body_refs")),
                "hair_refs": _clean_paths(avatar_payload.get("hair_refs")),
                "wardrobe_refs": _clean_paths(avatar_payload.get("wardrobe_refs")),
                "anatomy_guides": avatar_payload.get("anatomy_guides") or [],
                "persistent_features": avatar_payload.get("persistent_features") or [],
            }
        )

        shot_payload = payload.get("shot", {})
        references = shot_payload.get("references", {})
        engine = shot_payload.get("engine_preference", "h3_ref2va")
        primary_reference = references.get("init_image") or (avatar.identity_refs[0] if avatar.identity_refs else None)
        if engine == "h3_ref2va" and payload.get("reference_mode") == "exact_primary" and primary_reference:
            engine = "h3_fl2va"
        preset = preset_by_id(shot_payload.get("h3_preset") or self._settings()["default_h3_preset"])
        if str(engine).startswith("h3_"):
            width, height, fps = preset.width, preset.height, H3_FPS
            duration = min(float(shot_payload.get("duration_s", 6)), 15.0)
        else:
            width = int(shot_payload.get("width", 720))
            height = int(shot_payload.get("height", 1280))
            fps = int(shot_payload.get("fps", 24))
            duration = float(shot_payload.get("duration_s", 6))

        shot = ShotSpec.model_validate(
            {
                "shot_id": shot_payload.get("shot_id") or "desktop-shot",
                "content_class": shot_payload.get("content_class", "general"),
                "user_prompt": shot_payload.get("user_prompt") or "Natural photorealistic human motion.",
                "duration_s": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "seed": int(shot_payload.get("seed", 41001)),
                "engine_preference": engine,
                "local_only": True,
                "action_beats": shot_payload.get("action_beats") or [],
                "camera": shot_payload.get("camera") or {},
                "wind": shot_payload.get("wind") or {},
                "wardrobe": shot_payload.get("wardrobe") or None,
                "environment": shot_payload.get("environment") or None,
                "negative_prompt": shot_payload.get("negative_prompt") or None,
                "references": {
                    "init_image": references.get("init_image") or None,
                    "last_frame": references.get("last_frame") or None,
                    "motion_video": references.get("motion_video") or None,
                    "scene_image": references.get("scene_image") or None,
                    "audio": references.get("audio") or None,
                    "extra_images": _clean_paths(references.get("extra_images")),
                    "extra_videos": _clean_paths(references.get("extra_videos")),
                    "extra_audios": _clean_paths(references.get("extra_audios")),
                },
            }
        )
        return avatar, shot

    def _workflow_for(self, payload: dict[str, Any], engine: str) -> str:
        explicit = payload.get("workflow")
        if explicit:
            return str(explicit)
        settings = {**self._settings(), **payload.get("settings", {})}
        key = {
            "h3_ref2va": "workflow_h3_ref2va",
            "h3_fl2va": "workflow_h3_fl2va",
            "wan22": "workflow_wan22",
            "custom": "workflow_custom",
        }.get(engine)
        return str(settings.get(key) or "") if key else ""

    def _resolve_runtime(self, payload: dict[str, Any], avatar: AvatarManifest, shot: ShotSpec) -> tuple[Any, dict[str, Any], str]:
        settings = {**self._settings(), **payload.get("settings", {})}
        profile_id = payload.get("runtime_profile") or settings.get("default_runtime_profile", "auto")
        requested_preset = payload.get("h3_preset") or shot.model_dump().get("h3_preset") or settings.get("default_h3_preset")

        if not shot.engine_preference.startswith("h3_") or profile_id == "wan_fallback":
            advanced = payload.get("advanced") or {}
            runtime = {
                "profile": profile_id,
                "engine": "wan22" if profile_id == "wan_fallback" else shot.engine_preference,
                "model": None,
                "preset": requested_preset,
                "steps": int(advanced.get("STEPS", 20)),
                "cfg": float(advanced.get("CFG", 1.0)),
                "reason": "Non-H3 renderer selected." if profile_id != "wan_fallback" else "Wan fallback selected.",
            }
        else:
            provider = ComfyUIProvider(self._provider_config(payload))
            stats = provider.health()
            caps = provider.capabilities()
            runtime = resolve_profile(
                profile_id=profile_id,
                engine=shot.engine_preference,
                system_stats=stats,
                diffusion_models=caps.get("models", {}).get("diffusion_models", []),
                local_h3_authorized=bool(settings.get("local_h3_authorized")),
                requested_preset=requested_preset,
                duration_s=shot.duration_s,
            )

        effective_engine = runtime.get("engine") or shot.engine_preference
        if effective_engine != shot.engine_preference:
            shot = shot.model_copy(update={"engine_preference": effective_engine})
        if str(effective_engine).startswith("h3_"):
            preset = preset_by_id(str(runtime.get("preset") or settings.get("default_h3_preset")))
            shot = shot.model_copy(update={"width": preset.width, "height": preset.height, "fps": H3_FPS})

        job = build_job(avatar, shot)
        job.asset_map["RUNTIME_PROFILE"] = str(runtime.get("profile") or profile_id)
        job.asset_map["STEPS"] = int(runtime.get("steps") or 20)
        job.asset_map["CFG"] = float(runtime.get("cfg") or 1.0)
        for key, value in (payload.get("advanced") or {}).items():
            job.asset_map[str(key)] = value
        if job.selected_engine.startswith("h3_"):
            job.asset_map["FRAMES"] = valid_frame_count(shot.duration_s)
            if runtime.get("model"):
                job.asset_map["H3_DIFFUSION_MODEL"] = str(runtime["model"])
            job.asset_map.setdefault("SAMPLER_NAME", "res_multistep")
            job.asset_map.setdefault("SCHEDULER", "simple")
            job.asset_map.setdefault("WEIGHT_DTYPE", "default")
            job.asset_map.setdefault("H3_TEXT_ENCODER", settings.get("h3_text_encoder") or "")
            job.asset_map.setdefault("H3_VAE", settings.get("h3_vae") or "")
            job.asset_map.setdefault("H3_AUDIO_VAE", settings.get("h3_audio_vae") or "")
        elif job.selected_engine == "wan22":
            job.asset_map.setdefault("SAMPLER_NAME", "uni_pc")
            job.asset_map.setdefault("SCHEDULER", "simple")
            job.asset_map.setdefault("WEIGHT_DTYPE", "default")
            job.asset_map.setdefault("WAN_DIFFUSION_MODEL", settings.get("wan_diffusion_model") or "")
            job.asset_map.setdefault("WAN_TEXT_ENCODER", settings.get("wan_text_encoder") or "")
            job.asset_map.setdefault("WAN_VAE", settings.get("wan_vae") or "")
            job.asset_map.setdefault("WAN_LORA", settings.get("wan_lora") or "")
            job.asset_map.setdefault("WAN_LORA_STRENGTH", float(settings.get("wan_lora_strength") or 1.0))
            job.asset_map["FPS"] = WAN_FPS
            job.asset_map["FRAMES"] = wan_frame_count(shot.duration_s)
            wan_width, wan_height = snap_wan_canvas(shot.width, shot.height)
            job.asset_map["WIDTH"] = wan_width
            job.asset_map["HEIGHT"] = wan_height
            job = job.model_copy(update={
                "shot": shot.model_copy(update={"fps": WAN_FPS, "width": wan_width, "height": wan_height})
            })
        job.asset_map.setdefault("OUTPUT_PREFIX", f"avatar_v2_{shot.shot_id[:24]}")
        workflow = self._workflow_for(payload, job.selected_engine)
        if not workflow and job.selected_engine in {"h3_ref2va", "h3_fl2va", "wan22", "upscale"}:
            if job.selected_engine == "wan22":
                mode = str((payload.get("advanced") or {}).get("WAN_MODE", "funcontrol"))
                workflow = "builtin:wan22_funcontrol" if mode != "flf2v" else "builtin:wan22"
            else:
                workflow = f"builtin:{job.selected_engine}"
        return job, runtime, workflow

    def compose(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            decision = evaluate_policy(avatar, shot)
            if not decision.allowed:
                return {"ok": False, "error": decision.reason}
            job, runtime, workflow = self._resolve_runtime(payload, avatar, shot)
            run_dir = RUNS_DIR / _run_id("plan")
            run_dir.mkdir(parents=True, exist_ok=True)
            job_path = run_dir / "job.json"
            job_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            _write_json(run_dir / "runtime.json", runtime)
            return {
                "ok": True,
                "job_id": job.job_id,
                "job_path": str(job_path),
                "engine": job.selected_engine,
                "runtime": runtime,
                "workflow_ready": bool(workflow),
                "prompt": job.prompt,
                "asset_map": job.asset_map,
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def dry_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            job, runtime, workflow = self._resolve_runtime(payload, avatar, shot)
            if not workflow:
                return {"ok": False, "error": f"No API-format workflow configured for {job.selected_engine}."}
            requested = str((payload.get("shot") or {}).get("engine_preference") or "")
            if requested.startswith("h3_") and not runtime.get("local_h3_allowed", True):
                return {
                    "ok": False,
                    "error": "Local H3 is not authorized in settings; choose Wan fallback or enable H3 only if your use is licensed.",
                }
            provider = ComfyUIProvider(self._provider_config(payload))
            resolved = provider.resolve_workflow(workflow, job, stage_assets=False)
            run_dir = RUNS_DIR / _run_id("dryrun")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
            _write_json(run_dir / "runtime.json", runtime)
            resolved_path = run_dir / "resolved_workflow.json"
            _write_json(resolved_path, resolved)
            return {
                "ok": True,
                "engine": job.selected_engine,
                "runtime": runtime,
                "resolved_workflow": str(resolved_path),
                "reference_bindings": _reference_bindings(job, resolved),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def queue_render(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            decision = evaluate_policy(avatar, shot)
            if not decision.allowed:
                return {"ok": False, "error": decision.reason}
            job, runtime, workflow = self._resolve_runtime(payload, avatar, shot)
            if not workflow:
                return {"ok": False, "error": f"No API-format workflow configured for {job.selected_engine}."}
            requested = str((payload.get("shot") or {}).get("engine_preference") or "")
            if requested.startswith("h3_") and not runtime.get("local_h3_allowed", True):
                return {
                    "ok": False,
                    "error": "Local H3 is not authorized in settings; choose Wan fallback or enable H3 only if your use is licensed.",
                }
            item = QueuedRender(
                job=job,
                workflow=workflow,
                provider_config=self._provider_config(payload),
                profile=str(runtime.get("profile") or "auto"),
                runtime=runtime,
                output_dir=payload.get("output_dir"),
                archive_dir=str(self._archive_dir()),
                chain_from_previous=bool(payload.get("chain_from_previous")),
                label=str(payload.get("label") or shot.shot_id),
            )
            item_id = self.queue.add(item)
            return {
                "ok": True,
                "queue_id": item_id,
                "runtime": runtime,
                "engine": job.selected_engine,
                "reference_bindings": _reference_bindings(job),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def queue_profile_sweep(self, payload: dict[str, Any], profile_ids: list[str] | None = None) -> dict[str, Any]:
        profile_ids = profile_ids or ["auto", "int8_12gb", "low_memory", "low_memory_quality"]
        created: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        base_seed = int(payload.get("shot", {}).get("seed", 41001))
        for index, profile_id in enumerate(profile_ids):
            candidate = json.loads(json.dumps(payload))
            candidate["runtime_profile"] = profile_id
            candidate.setdefault("shot", {})["seed"] = base_seed + index
            candidate["label"] = f"{candidate.get('label') or 'profile-test'} · {profile_id}"
            result = self.queue_render(candidate)
            if result.get("ok"):
                created.append({"profile": profile_id, **result})
            else:
                errors.append({"profile": profile_id, "error": str(result.get("error"))})
        return {"ok": bool(created), "created": created, "errors": errors}

    def queue_state(self) -> dict[str, Any]:
        return {"ok": True, "items": self.queue.list()}

    def cancel_queue_item(self, item_id: str) -> dict[str, Any]:
        return {"ok": self.queue.cancel(item_id)}

    def clear_finished(self) -> dict[str, Any]:
        return {"ok": True, "removed": self.queue.clear_finished()}

    def ui_test_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Record results from the in-app UI self-test (AVATAR_V2_UITEST=1)."""
        path = APP_HOME / "ui_test_report.json"
        try:
            existing = _read_json(path, [])
            existing = existing if isinstance(existing, list) else []
            existing.append({**report, "reported_at": datetime.now(timezone.utc).isoformat()})
            _write_json(path, existing)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def open_path(self, path: str) -> dict[str, Any]:
        try:
            target = str(Path(path).expanduser())
            system = platform.system()
            if system == "Windows":
                os.startfile(target)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # Avatar Vault / Motion Library / Scene Library
    # ------------------------------------------------------------------

    def vault_list(self) -> dict[str, Any]:
        return {"ok": True, "avatars": self.library.avatars()}

    def vault_save(self, avatar: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "avatar": self.library.put_avatar(avatar)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def vault_delete(self, avatar_id: str) -> dict[str, Any]:
        return {"ok": self.library.delete_avatar(avatar_id)}

    def vault_apply(self, avatar_id: str) -> dict[str, Any]:
        payload = self.library.profile_to_avatar_payload(avatar_id)
        if not payload:
            return {"ok": False, "error": f"avatar '{avatar_id}' not found"}
        return {"ok": True, "avatar": payload}

    def anatomy_guides_list(self) -> dict[str, Any]:
        return {"ok": True, "guides": self.library.anatomy_guides()}

    def anatomy_guides_save(self, guide: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = self.library.put_anatomy_guide(guide)
            return {"ok": True, "guide": saved}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def anatomy_guides_delete(self, guide_id: str) -> dict[str, Any]:
        return {"ok": self.library.delete_anatomy_guide(guide_id)}

    def appearance_components_list(self) -> dict[str, Any]:
        return {"ok": True, "components": self.library.appearance_components()}

    def appearance_components_save(self, component: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "component": self.library.put_appearance_component(component)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def appearance_components_delete(self, option_id: str) -> dict[str, Any]:
        return {"ok": self.library.delete_appearance_component(option_id)}

    def _selected_designer_guides(self, profile: AvatarProfile, slot: Any | None = None) -> list[AnatomyGuide]:
        guides = [
            AnatomyGuide.model_validate(entry)
            for entry in self.library.selected_anatomy_guides(profile.anatomy_guide_ids)
        ]
        if slot is None:
            return guides
        if slot.group == "body":
            return [guide for guide in guides if guide.region == "full_body"]
        if slot.group != "anatomy":
            return []
        if slot.role.startswith("pelvis"):
            return [guide for guide in guides if guide.region.startswith("pelvis")]
        if slot.role.startswith("torso"):
            return [guide for guide in guides if guide.region in {slot.role, "full_body"}]
        return [guide for guide in guides if guide.region == slot.role]

    @staticmethod
    def _designer_body_refs(
        profile: AvatarProfile,
        slot: Any | None = None,
        component_paths: list[str] | None = None,
    ) -> list[str]:
        refs: list[str] = []
        prioritized = []
        include_profile_body = slot is None or slot.group != "identity"
        if include_profile_body and slot is not None and slot.group == "body" and profile.body_refs.get(slot.role):
            prioritized.append(profile.body_refs[slot.role])
        prioritized.extend(component_paths or [])
        if include_profile_body:
            prioritized.extend(profile.body_refs.values())
        for value in prioritized:
            path = str(value or "").strip()
            if path and path not in refs:
                refs.append(path)
        return refs[:MAX_DESIGNER_BODY_REFS]

    def avatar_designer_coverage(self, avatar: dict[str, Any]) -> dict[str, Any]:
        try:
            profile = AvatarProfile.model_validate(avatar)
            return {"ok": True, "coverage": coverage_report(profile)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def avatar_designer_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            profile = AvatarProfile.model_validate(payload.get("avatar") or {})
            slot = slot_by_id(str(payload.get("slot_id") or ""))
            requested_mode = str(payload.get("render_mode") or "fl2va")
            all_components = self.library.selected_appearance_components(profile.component_option_ids) if profile.component_option_ids else []
            components = [
                component
                for component in all_components
                if (
                    slot.group == "identity"
                    and component["category"] in {"face_shape", "skin"}
                )
                or (
                    slot.group != "identity"
                    and component["category"] != "face_shape"
                )
            ]
            component_paths = [path for component in components for path in component.get("paths") or []]
            component_labels = [f"{component['label']} ({component['category'].replace('_', ' ')})" for component in components]
            saved_body_ref_count = len({str(value).strip() for value in profile.body_refs.values() if str(value).strip()})
            body_refs = self._designer_body_refs(profile, slot, component_paths) if requested_mode == "body_ref2va" else []
            if requested_mode == "body_ref2va" and not (saved_body_ref_count or all_components):
                return {"ok": False, "error": "Add at least one saved body / proportions reference or ready appearance component before using Ref2VA."}
            selected_guides = self._selected_designer_guides(profile) if requested_mode == "guided_ref2va" else []
            if requested_mode == "guided_ref2va" and not selected_guides:
                return {"ok": False, "error": "Select at least one validated anatomy guide before using guided Ref2VA."}
            guides = self._selected_designer_guides(profile, slot) if selected_guides else []
            guided = bool(guides)
            body_referenced = bool(body_refs)
            if (slot.adult_only or guided or body_referenced) and not (
                profile.subject_kind != "unknown"
                and profile.age_verified_18_plus
                and profile.consent_confirmed
            ):
                return {
                    "ok": False,
                    "error": "Adult anatomy slots require a synthetic or consenting-adult profile with verified 18+ age and confirmed consent.",
                }
            prompt = build_designer_prompt(
                slot.id,
                profile.appearance_manifest,
                profile.apparent_age_years,
                "guided_ref2va" if guided else "body_ref2va" if body_referenced else "fl2va",
                [f"{guide.label} ({guide.region.replace('_', ' ')})" for guide in guides],
                component_labels,
            )
            return {
                "ok": True,
                "slot": slot.id,
                "prompt": prompt,
                "engine": "h3_ref2va" if guided or body_referenced else "h3_fl2va",
                "guide_count": len(guides),
                "body_ref_count": len(body_refs) if body_referenced else 0,
                "saved_body_ref_count": saved_body_ref_count if requested_mode == "body_ref2va" else 0,
                "component_count": len(components) if body_referenced else 0,
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def avatar_designer_renderer_status(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        try:
            provider = ComfyUIProvider(self._provider_config(payload))
            caps = provider.capabilities()
            settings = {**self._settings(), **payload.get("settings", {})}
            h3_models = caps.get("h3_models") or {}
            diffusion_models = [str(item).lower() for item in h3_models.get("diffusion_models") or []]
            common_model_ready = all(h3_models.get(kind) for kind in ("text_encoders", "vae"))
            fl2va_model_ready = common_model_ready and any("fl2va" in item for item in diffusion_models)
            ref2va_model_ready = common_model_ready and any("ref2va" in item for item in diffusion_models)
            authorized = bool(settings.get("local_h3_authorized"))
            ffmpeg_ready = bool(shutil.which("ffmpeg"))
            base_ready = bool(caps.get("h3_available") and authorized and ffmpeg_ready)
            ready = bool(base_ready and fl2va_model_ready)
            guided_ready = bool(base_ready and fl2va_model_ready and ref2va_model_ready)
            missing: list[str] = []
            if not caps.get("h3_available"):
                missing.append("H3 ComfyUI nodes")
            if not fl2va_model_ready:
                missing.append("H3 FL2VA model, encoder, or VAE")
            if not authorized:
                missing.append("H3 local license authorization")
            if not ffmpeg_ready:
                missing.append("ffmpeg candidate-frame extraction")
            return {
                "ok": True,
                "ready": ready,
                "guided_ready": guided_ready,
                "engine": "h3_fl2va",
                "h3_available": bool(caps.get("h3_available")),
                "h3_models": h3_models,
                "authorized": authorized,
                "ffmpeg_ready": ffmpeg_ready,
                "fl2va_model_ready": fl2va_model_ready,
                "ref2va_model_ready": ref2va_model_ready,
                "missing": missing,
                "message": (
                    "MiniMax H3 FL2VA profile generation and final-frame extraction are ready."
                    if ready
                    else "H3 profile generation needs: " + ", ".join(missing)
                ),
            }
        except Exception as exc:
            return {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}

    def queue_avatar_designer(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            profile = AvatarProfile.model_validate(payload.get("avatar") or {})
            slot = slot_by_id(str(payload.get("slot_id") or ""))
            requested_mode = str(payload.get("render_mode") or "fl2va")
            all_components = self.library.selected_appearance_components(profile.component_option_ids) if profile.component_option_ids else []
            components = [
                component
                for component in all_components
                if (
                    slot.group == "identity"
                    and component["category"] in {"face_shape", "skin"}
                )
                or (
                    slot.group != "identity"
                    and component["category"] != "face_shape"
                )
            ]
            component_paths = [path for component in components for path in component.get("paths") or []]
            component_labels = [f"{component['label']} ({component['category'].replace('_', ' ')})" for component in components]
            saved_body_ref_count = len({str(value).strip() for value in profile.body_refs.values() if str(value).strip()})
            body_refs = self._designer_body_refs(profile, slot, component_paths) if requested_mode == "body_ref2va" else []
            if requested_mode == "body_ref2va" and not (saved_body_ref_count or all_components):
                return {"ok": False, "error": "Add at least one saved body / proportions reference or ready appearance component before using Ref2VA."}
            selected_guides = self._selected_designer_guides(profile) if requested_mode == "guided_ref2va" else []
            if requested_mode == "guided_ref2va" and not selected_guides:
                return {"ok": False, "error": "Select at least one validated anatomy guide before using guided Ref2VA."}
            guides = self._selected_designer_guides(profile, slot) if selected_guides else []
            guided = bool(guides)
            body_referenced = bool(body_refs)
            if (slot.adult_only or guided or body_referenced) and not (
                profile.subject_kind != "unknown"
                and profile.age_verified_18_plus
                and profile.consent_confirmed
            ):
                return {
                    "ok": False,
                    "error": "Adult anatomy slots require a synthetic or consenting-adult profile with verified 18+ age and confirmed consent.",
                }
            source = str(payload.get("source_image") or "").strip()
            if not source:
                source = next((path for path in [*profile.source_refs, *profile.ordered_identity] if path), "")
            if not source or not Path(source).expanduser().is_file():
                return {"ok": False, "error": "Select an existing primary source image before generating a candidate."}
            missing_guides = [guide.label for guide in guides if not Path(guide.path).expanduser().is_file()]
            if missing_guides:
                return {
                    "ok": False,
                    "error": "Selected anatomy guide files are missing: " + ", ".join(missing_guides),
                }
            missing_body_refs = [path for path in body_refs if not Path(path).expanduser().is_file()]
            if missing_body_refs:
                return {
                    "ok": False,
                    "error": "Saved body reference files are missing: " + ", ".join(Path(path).name for path in missing_body_refs),
                }

            status = self.avatar_designer_renderer_status(payload)
            ref2va = guided or body_referenced
            route_ready = status.get("guided_ready") if ref2va else status.get("ready")
            if not route_ready:
                if ref2va:
                    return {
                        "ok": False,
                        "error": "Guided profile generation requires both H3 FL2VA and Ref2VA models, authorization, and ffmpeg. Ref2VA remains experimental on 8 GB GPUs.",
                    }
                return {"ok": False, "error": status.get("message") or status.get("error") or "MiniMax H3 is not ready."}

            prompt = build_designer_prompt(
                slot.id,
                profile.appearance_manifest,
                profile.apparent_age_years,
                "guided_ref2va" if guided else "body_ref2va" if body_referenced else "fl2va",
                [f"{guide.label} ({guide.region.replace('_', ' ')})" for guide in guides],
                component_labels,
            )
            appearance_features = [
                f"{key.replace('_', ' ')}: {value}"
                for key, value in profile.appearance_manifest.items()
                if str(value).strip()
            ]
            requested_preset = str(payload.get("h3_preset") or self._settings().get("default_h3_preset") or "vertical_4070")
            render_payload = {
                "runtime_profile": "low_memory" if ref2va else str(payload.get("runtime_profile") or "low_memory_quality"),
                "h3_preset": "vertical_fast" if ref2va else requested_preset,
                "reference_mode": "multi_match" if ref2va else "exact_primary",
                "settings": payload.get("settings") or {},
                "advanced": {"H3_REF_IMAGE_SIZE": "match"} if ref2va else {},
                "avatar": {
                    "avatar_id": profile.avatar_id,
                    "display_name": profile.display_name,
                    "apparent_age_years": profile.apparent_age_years,
                    "subject_kind": profile.subject_kind,
                    "age_verified_18_plus": profile.age_verified_18_plus,
                    "consent_confirmed": profile.consent_confirmed,
                    "identity_refs": [source],
                    "body_refs": body_refs if body_referenced else [],
                    "anatomy_guides": [
                        {
                            "guide_id": guide.guide_id,
                            "label": guide.label,
                            "path": guide.path,
                            "region": guide.region,
                        }
                        for guide in guides
                    ],
                    "persistent_features": [*profile.persistent_features, *appearance_features],
                },
                "shot": {
                    "shot_id": f"designer-{slot.id}",
                    "content_class": "adult_nudity" if slot.adult_only or guided else "general",
                    "user_prompt": prompt,
                    "duration_s": 4.0,
                    "seed": int(payload.get("seed") or 41001),
                    "engine_preference": "h3_ref2va" if ref2va else "h3_fl2va",
                    "h3_preset": "vertical_fast" if ref2va else requested_preset,
                    "camera": {"framing": slot.framing, "movement": "one slow continuous transition, then locked final hold", "lens_equivalent_mm": 70},
                    "environment": "neutral gray reference studio with flat even lighting",
                    "references": {} if ref2va else {"init_image": source},
                },
            }
            avatar, shot = self._models_from_payload(render_payload)
            decision = evaluate_policy(avatar, shot)
            if not decision.allowed:
                return {"ok": False, "error": decision.reason}
            job, runtime, workflow = self._resolve_runtime(render_payload, avatar, shot)
            expected_engine = "h3_ref2va" if ref2va else "h3_fl2va"
            if job.selected_engine != expected_engine:
                return {"ok": False, "error": f"Avatar Designer requires {expected_engine}, but runtime resolved {job.selected_engine}."}
            if not runtime.get("local_h3_allowed", True):
                return {"ok": False, "error": "Local H3 is not authorized in Settings."}
            if not workflow:
                return {"ok": False, "error": f"No {expected_engine} workflow is available."}
            runtime = {
                **runtime,
                "extract_candidate_frame": True,
                "designer_slot": slot.id,
                "designer_render_mode": "guided_ref2va" if guided else "body_ref2va" if body_referenced else "fl2va",
                "guide_count": len(guides),
                "body_ref_count": len(body_refs) if body_referenced else 0,
                "saved_body_ref_count": saved_body_ref_count if requested_mode == "body_ref2va" else 0,
                "component_count": len(components) if body_referenced else 0,
            }
            job.asset_map["OUTPUT_PREFIX"] = f"avatar_designer_h3_{profile.avatar_id}_{slot.role}"
            item = QueuedRender(
                job=job,
                workflow=workflow,
                provider_config=self._provider_config(render_payload),
                profile=str(runtime.get("profile") or "low_memory_quality"),
                runtime=runtime,
                output_dir=payload.get("output_dir"),
                archive_dir=str(self._archive_dir()) if self._archive_dir() else None,
                label=f"Avatar Designer H3 · {profile.display_name} · {slot.label}",
            )
            queue_id = self.queue.add(item)
            return {
                "ok": True,
                "queue_id": queue_id,
                "slot": slot.id,
                "prompt": prompt,
                "engine": job.selected_engine,
                "guide_count": len(guides),
                "body_ref_count": len(body_refs) if body_referenced else 0,
                "saved_body_ref_count": saved_body_ref_count if requested_mode == "body_ref2va" else 0,
                "component_count": len(components) if body_referenced else 0,
                "runtime": runtime,
                "reference_bindings": _reference_bindings(job),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def motions_list(self) -> dict[str, Any]:
        return {"ok": True, "motions": self.library.motions()}

    def motions_save(self, motion: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "motion": self.library.put_motion(motion)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def motions_delete(self, motion_id: str) -> dict[str, Any]:
        return {"ok": self.library.delete_motion(motion_id)}

    def scenes_list(self) -> dict[str, Any]:
        return {"ok": True, "scenes": self.library.scenes()}

    def scenes_save(self, scene: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "scene": self.library.put_scene(scene)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def scenes_delete(self, scene_id: str) -> dict[str, Any]:
        return {"ok": self.library.delete_scene(scene_id)}

    # ------------------------------------------------------------------
    # GPU panel
    # ------------------------------------------------------------------

    def gpu_status(self) -> dict[str, Any]:
        status = self.queue.memory_status()
        active: dict[str, Any] = {}
        for item in self.queue.list():
            if item["status"] == "running":
                active = {
                    "queue_id": item["id"],
                    "label": item["label"],
                    "profile": item["profile"],
                    "engine": item["engine"],
                }
        peak: list[dict[str, Any]] = []
        if active:
            peak = self.queue.memory_manager.current_trace_peak()
        return {"ok": True, "manager": status, "active_render": active, "peak": peak}

    def gpu_free(self) -> dict[str, Any]:
        try:
            provider = ComfyUIProvider(self._provider_config({}))
            queue_status = provider.queue_status()
            if queue_status["busy"]:
                return {
                    "ok": False,
                    "error": (
                        "GPU memory can only be freed while ComfyUI is idle. "
                        f"Wait for {queue_status['running']} running and {queue_status['pending']} pending render(s), "
                        "or cancel them first."
                    ),
                }
            event = self.queue.memory_manager.maybe_cleanup(
                provider, reason="manual-gpu-free", force=True, unload_models=True
            )
            return {"ok": True, "event": event}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def gpu_details(self) -> dict[str, Any]:
        return {"ok": True, **self.queue.memory_status()}

    def gpu_terminate_pids(self, pids: list[int]) -> dict[str, Any]:
        """Emergency termination. Only explicitly supplied PIDs, never auto-selected."""
        try:
            result = self.queue.memory_manager.emergency_terminate(pids or [])
            return {"ok": True, **result}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------
    # ComfyUI service management
    # ------------------------------------------------------------------

    def comfyui_status(self) -> dict[str, Any]:
        settings = self._settings()
        ready = False
        try:
            ComfyUIProvider(self._provider_config({})).health()
            ready = True
        except Exception:
            pass
        pid = None
        try:
            pid = int(COMFY_PID_PATH.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            pass
        alive = bool(pid) and Path(f"/proc/{pid}").exists()
        if ready:
            state = "READY"
        elif alive:
            state = "STARTING"
        elif pid:
            state = "ERROR"
        else:
            state = "STOPPED"
        return {
            "ok": True,
            "state": state,
            "ready": ready,
            "pid": pid if alive else None,
            "managed": bool(pid),
            "comfyui_dir": settings.get("comfyui_dir") or "",
            "comfyui_python": settings.get("comfyui_python") or "",
        }

    def comfyui_start(self) -> dict[str, Any]:
        settings = self._settings()
        current = self.comfyui_status()
        if current["ready"]:
            return {"ok": True, "state": "READY", "message": "ComfyUI was already running."}
        comfyui_dir = Path(str(settings.get("comfyui_dir") or "")).expanduser()
        if not comfyui_dir.is_dir() or not (comfyui_dir / "main.py").is_file():
            return {
                "ok": False,
                "error": "comfyui_dir is not set in Settings or does not contain main.py.",
            }
        python = str(settings.get("comfyui_python") or "python3")
        url = str(settings.get("comfyui_url") or DEFAULT_SETTINGS["comfyui_url"])
        port = url.rsplit(":", 1)[-1] if ":" in url else "8188"
        try:
            log = open(COMFY_LOG_PATH, "ab")
            proc = subprocess.Popen(
                [python, "main.py", "--listen", "127.0.0.1", "--port", str(port)],
                cwd=str(comfyui_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        COMFY_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        return {"ok": True, "state": "STARTING", "pid": proc.pid}

    def comfyui_stop(self) -> dict[str, Any]:
        current = self.comfyui_status()
        if not current["ready"] and not current["pid"]:
            return {"ok": True, "state": "STOPPED", "message": "ComfyUI is not running."}
        pid = current["pid"]
        if not pid:
            return {
                "ok": False,
                "error": (
                    "ComfyUI is running but was not started by Avatar V2. "
                    "Stop it from its own terminal or use GPU Details for an explicit PID action."
                ),
            }
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
            time.sleep(0.5)
        if Path(f"/proc/{pid}").exists():
            return {"ok": False, "error": f"ComfyUI pid {pid} did not stop after SIGTERM."}
        try:
            COMFY_PID_PATH.unlink()
        except FileNotFoundError:
            pass
        return {"ok": True, "state": "STOPPED", "pid": pid}

    # ------------------------------------------------------------------
    # History and best-of-N
    # ------------------------------------------------------------------

    def history(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
            try:
                entries.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return {"ok": True, "items": entries}

    def open_output_dir(self) -> dict[str, Any]:
        preferred = str(self._settings().get("preferred_output_dir") or "")
        target = preferred or str(Path.home())
        return self.open_path(target)

    def queue_upscale(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue an upscale job for a finished render's MP4 output."""
        try:
            video = str((payload.get("video") or "").strip())
            if not video:
                return {"ok": False, "error": "video path is required"}
            source = Path(video).expanduser()
            if not source.is_file():
                return {"ok": False, "error": f"video not found: {source}"}
            fps = 16
            try:
                from .qa import ffprobe_video

                probe = ffprobe_video(source)
                num, _, den = str(probe.get("avg_frame_rate") or "16/1").partition("/")
                if den and float(den):
                    fps = float(num) / float(den)
                else:
                    fps = float(probe.get("avg_frame_rate") or 16)
            except Exception:
                pass
            upscale_payload = {
                "runtime_profile": "auto",
                "label": str(payload.get("label") or f"upscale · {source.stem}"),
                "avatar": {
                    "avatar_id": "upscale", "display_name": "Upscale",
                    "subject_kind": "synthetic", "age_verified_18_plus": True, "consent_confirmed": True,
                    "identity_refs": [], "body_refs": [], "hair_refs": [], "wardrobe_refs": [],
                    "persistent_features": [],
                },
                "shot": {
                    "shot_id": "upscale", "content_class": "general",
                    "user_prompt": "Upscale pass",
                    "duration_s": 2, "seed": 41001, "engine_preference": "upscale",
                    "width": 832, "height": 480, "fps": int(round(fps)),
                    "camera": {}, "wind": {},
                    "references": {"init_image": None, "last_frame": None, "motion_video": None,
                                   "scene_image": None, "audio": None,
                                   "extra_images": [], "extra_videos": [], "extra_audios": []},
                },
                "advanced": {
                    "UPSCALE_MODEL": str(payload.get("scale_model")
                                         or self._settings().get("upscale_model") or "4x-UltraSharp.pth"),
                    "OUTPUT_PREFIX": f"upscale_{source.stem[:40]}",
                    "SOURCE_VIDEO": str(source),
                },
            }
            return self.queue_render(upscale_payload)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def queue_best_of(self, payload: dict[str, Any], count: int = 4) -> dict[str, Any]:
        count = max(2, min(int(count), 8))
        created: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        base_seed = int(payload.get("shot", {}).get("seed", 41001))
        for index in range(count):
            candidate = json.loads(json.dumps(payload))
            candidate.setdefault("shot", {})["seed"] = base_seed + index
            candidate["label"] = f"{candidate.get('label') or 'best-of-n'} · #{index + 1}"
            result = self.queue_render(candidate)
            if result.get("ok"):
                created.append({"index": index + 1, **result})
            else:
                errors.append({"index": index + 1, "error": str(result.get("error"))})
        return {"ok": bool(created), "created": created, "errors": errors}


def main() -> None:
    import webview

    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    if not ui_path.exists():
        raise FileNotFoundError(f"Desktop UI not found: {ui_path}")

    ui_url = str(ui_path)
    if os.environ.get("AVATAR_V2_UITEST") == "1":
        ui_url += "?uitest=1"

    api = DesktopAPI()
    window = webview.create_window(
        "Avatar V2",
        ui_url,
        js_api=api,
        width=1480,
        height=940,
        min_size=(1180, 760),
        background_color="#050507",
        confirm_close=False,
    )
    api._self_window = window
    webview.start(debug=False)
    api.queue.shutdown()


if __name__ == "__main__":
    main()
