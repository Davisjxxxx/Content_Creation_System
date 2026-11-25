from __future__ import annotations

import json
import base64
import io
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
from PIL import Image, ImageOps

from .candidate_qa import compare_candidate_to_reference
from .composer import build_job
from . import brand_instructor
from .desktop_presenter import PresenterMixin
from .desktop_studio import StudioMixin
from .integrity import sha256_file
from .voiceover import apply_pronunciations
from .avatar_designer import build_designer_prompt, coverage_report, designer_spec, slot_by_id
from .h3_runtime import H3_FPS, preset_by_id, presets_payload, valid_frame_count
from .library import AnatomyGuide, AvatarCandidateRecord, AvatarProfile, LibraryStore
from .models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from .policy import PolicyError, evaluate_cloud_policy, evaluate_policy
from .presenter import compile_presenter_prompt, presenter_receipt, variation_by_id, variations_payload
from .prompt_presets import adult_prompt_presets_payload
from .reference_cache import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, scan_reference_cache
from .providers import ComfyUIProvider
from .providers import runway as runway_api
from .render_queue import QueuedRender, RenderQueue
from .secret_store import SecretStoreError, clear_secret, save_secret, secret_status
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
RUNWAY_AVATARS_PATH = APP_HOME / "runway_avatars.json"
CLOUD_LEDGER_PATH = APP_HOME / "cloud_ledger.jsonl"
VOICEOVER_DIR = APP_HOME / "voiceover"
IMPORTS_DIR = LIBRARY_DIR / "imports"
LESSONS_PATH = APP_HOME / "lessons.json"
QA_DIR = APP_HOME / "qa"
PROVIDER_PROFILES_PATH = APP_HOME / "provider_profiles.json"
# Where scripts/setup_local_engine.sh installs ComfyUI + its venv; used when settings leave them empty.
ENGINE_HOME = Path.home() / "AvatarForge"

# Settings keys that look like credentials are never persisted to desktop.json or sent to the UI.
SECRET_SETTING_PATTERN = re.compile(r"(api_?key|secret|token|password)", re.IGNORECASE)
DEFAULT_PRESENTER_PERSONALITY = (
    "You are {name}, a warm, clear and encouraging course instructor. You explain one idea at a time in "
    "plain language, check for understanding, and keep a calm, professional tone."
)

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
    "comfyui_args": "--disable-auto-launch",
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
    "last_image_dir": "",
    "last_script_dir": "",
    "adult_prompt_custom": {},
    "vision_classifier_url": "http://127.0.0.1:11434",
    "vision_classifier_model": "qwen3-vl:8b-instruct",
    # Runway cloud provider: ready as soon as an API key is saved. Spending limits are optional
    # and off by default (0 = no limit); they only apply when the owner sets one.
    "runway_default_image_model": "gen4_image",
    "runway_default_video_model": "gen4.5",
    "runway_job_limit": 0,
    "runway_batch_limit": 0,
    "runway_daily_limit": 0,
    "runway_monthly_limit": 0,
    "runway_poll_interval_s": 6,
    "runway_timeout_s": 1800,
}

_RUNWAY_NUMERIC_SETTINGS = (
    ("runway_job_limit", 0, 100000),
    ("runway_batch_limit", 0, 1000000),
    ("runway_poll_interval_s", 5, 60),
    ("runway_timeout_s", 60, 7200),
    ("runway_daily_limit", 0, 1000000),
    ("runway_monthly_limit", 0, 10000000),
)


def _proc_alive(pid: int | None) -> bool:
    """True only for a live process (a zombie's /proc entry does not count)."""
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            state = handle.read().split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return False


def _sanitize_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Drop credential-like keys and coerce Runway settings to documented values."""
    clean = {
        str(key): value
        for key, value in (settings or {}).items()
        if not SECRET_SETTING_PATTERN.search(str(key))
    }
    for key, low, high in _RUNWAY_NUMERIC_SETTINGS:
        if key in clean:
            try:
                clean[key] = int(min(max(float(clean[key]), low), high))
            except (TypeError, ValueError):
                clean.pop(key)
    for key, task in (
        ("runway_default_image_model", "text_to_image"),
        ("runway_default_video_model", "image_to_video"),
    ):
        if key in clean:
            model = runway_api.MODELS.get(str(clean[key]))
            if model is None or model.task != task:
                clean.pop(key)
    return clean

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


class DesktopAPI(StudioMixin, PresenterMixin):
    def __init__(self) -> None:
        self._self_window = None
        APP_HOME.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        (APP_HOME / "logs").mkdir(parents=True, exist_ok=True)
        self.library = LibraryStore(LIBRARY_DIR)
        self.queue = RenderQueue(
            QUEUE_DIR,
            history_dir=HISTORY_DIR,
            candidate_qa_runner=self._candidate_qa,
            ledger_path=CLOUD_LEDGER_PATH,
            engine_restart=self._engine_restart_hook,
        )
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
        # A local engine installed by scripts/setup_local_engine.sh works without touching settings.
        if not settings.get("comfyui_dir") and (ENGINE_HOME / "comfyui" / "main.py").is_file():
            settings["comfyui_dir"] = str(ENGINE_HOME / "comfyui")
        if not settings.get("comfyui_python") and (ENGINE_HOME / "venv" / "bin" / "python").is_file():
            settings["comfyui_python"] = str(ENGINE_HOME / "venv" / "bin" / "python")
        if not settings.get("comfyui_output") and settings.get("comfyui_dir"):
            settings["comfyui_output"] = str(Path(str(settings["comfyui_dir"])) / "output")
        return settings

    def get_state(self) -> dict[str, Any]:
        settings = {
            key: value for key, value in self._settings().items() if not SECRET_SETTING_PATTERN.search(key)
        }
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
                {"id": "runway", "name": "Runway · Cloud", "tag": "CLOUD · OPT-IN"},
            ],
            "runway": self._runway_public_status(settings),
            "presenter": self._presenter_state(),
            "motion_categories": [
                "walk_confident", "walk_casual", "turn_and_smile", "sit_down", "stand_up",
                "hair_adjustment", "jacket_adjustment", "phone_selfie_walk", "natural_laugh",
                "look_away_return", "camera_push", "tracking_walk", "custom",
            ],
        }

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        merged = _sanitize_settings({**self._settings(), **_sanitize_settings(settings)})
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

    def choose_files(self, kind: str, multiple: bool = True, directory: str = "") -> list[str]:
        if self._self_window is None:
            return []
        import webview

        filters = {
            "image": (
                "Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff;*.PNG;*.JPG;*.JPEG;*.WEBP;*.BMP;*.TIF;*.TIFF)",
                "All files (*.*)",
            ),
            "video": ("Video (*.mp4;*.mov;*.webm;*.mkv)",),
            "audio": ("Audio (*.wav;*.mp3;*.flac;*.m4a;*.ogg)",),
            "workflow": ("ComfyUI API JSON (*.json)",),
            "python": ("Python scripts (*.py)", "All files (*.*)"),
            "config": ("Config & prompt libraries (*.json;*.yaml;*.yml;*.toml)", "All files (*.*)"),
            "all": ("All files (*.*)",),
        }
        kwargs: dict[str, Any] = {
            "allow_multiple": multiple,
            "file_types": filters.get(kind, ("All files (*.*)",)),
        }
        if directory and Path(directory).expanduser().is_dir():
            kwargs["directory"] = str(Path(directory).expanduser())
        result = self._self_window.create_file_dialog(webview.OPEN_DIALOG, **kwargs)
        return [str(Path(p)) for p in (result or [])]

    def choose_folder(self) -> str:
        if self._self_window is None:
            return ""
        import webview

        result = self._self_window.create_file_dialog(webview.FOLDER_DIALOG)
        return str(Path(result[0])) if result else ""

    def avatar_designer_scan_cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            queue = getattr(self, "queue", None)
            active = [
                item
                for item in (queue.list() if queue is not None else [])
                if item.get("status") in {"queued", "running"}
            ]
            if active:
                return {
                    "ok": False,
                    "error": "Wait for the render queue to finish before local AI image analysis so the 8 GB GPU is not shared with H3.",
                }
            settings = self._settings() if hasattr(self, "library") else DEFAULT_SETTINGS
            image_paths = payload.get("image_paths")
            if image_paths is not None and not isinstance(image_paths, list):
                return {"ok": False, "error": "image_paths must be a list of local image files"}
            return {
                "ok": True,
                **scan_reference_cache(
                    str(payload.get("cache_dir") or ""),
                    selected_files=image_paths,
                    use_visual=True,
                    vision_base_url=str(settings.get("vision_classifier_url") or "http://127.0.0.1:11434"),
                    vision_model=str(
                        payload.get("vision_model")
                        or settings.get("vision_classifier_model")
                        or "qwen3-vl:8b-instruct"
                    ),
                    single_subject_confirmed=bool(payload.get("single_subject_confirmed", False)),
                ),
            }
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
                "sex": avatar_payload.get("sex") or "",
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
        avatar_slug = re.sub(r"[^a-z0-9]+", "_", avatar.display_name.lower()).strip("_") or "avatar"
        job.asset_map.setdefault("OUTPUT_PREFIX", f"{avatar_slug}_{shot.shot_id[:32]}")
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
        if str((payload.get("shot") or {}).get("engine_preference") or "") == "runway":
            return {
                "ok": False,
                "error": "Runway is a cloud provider with its own consent and cost checks: use Estimate, "
                "then Queue on Runway.",
            }
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
            target_path = Path(path).expanduser().resolve()
            if not target_path.exists():
                return {"ok": False, "error": f"File or folder no longer exists: {target_path}"}
            target = str(target_path)
            system = platform.system()
            if system == "Windows":
                os.startfile(target)  # type: ignore[attr-defined]
            elif system == "Darwin":
                result = subprocess.run(
                    ["open", target], capture_output=True, text=True, timeout=10, check=False
                )
                if result.returncode:
                    detail = (result.stderr or result.stdout or "open failed").strip()
                    return {"ok": False, "error": detail}
            else:
                if target_path.suffix.lower() in VIDEO_EXTENSIONS and self._launch_linux_video(target):
                    return {"ok": True}
                result = subprocess.run(
                    ["xdg-open", target], capture_output=True, text=True, timeout=10, check=False
                )
                if result.returncode:
                    detail = (result.stderr or result.stdout or "xdg-open failed").strip()
                    return {"ok": False, "error": detail}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _launch_linux_video(target: str) -> bool:
        """Open videos in a player that stays open at the end so they can be replayed and reviewed."""
        players = (
            ("mpv", ("--force-window=yes", "--keep-open=yes")),
            ("vlc", ("--started-from-file", "--play-and-pause")),
            ("ffplay", ("-seek_interval", "1")),
        )
        for command, options in players:
            executable = shutil.which(command)
            if not executable:
                continue
            try:
                subprocess.Popen(
                    [executable, *options, target],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True
            except OSError:
                continue
        return False

    def reference_thumbnail(self, path: str) -> dict[str, Any]:
        """Return a small local-only preview for the reference gallery."""
        try:
            source = Path(path).expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
                return {"ok": False, "error": "Reference must be a supported local image file."}
            with Image.open(source) as opened:
                preview = ImageOps.exif_transpose(opened).convert("RGB")
                preview.thumbnail((320, 320), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG", quality=78, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return {"ok": True, "data_url": f"data:image/jpeg;base64,{encoded}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _candidate_qa(self, item: QueuedRender) -> dict[str, Any]:
        """Compare a finished designer still after H3 has been unloaded."""
        if item.provider_config.provider == "runway":
            return self._runway_postprocess(item)
        if item.runtime.get("forge_scene"):
            # A prompt-only scene: the start photo just rendered, so its shots go next.
            return self._forge_continue_scene(item)
        if item.runtime.get("scene_shots"):
            return self._studio_finish_scene(item)
        if item.runtime.get("studio_voiceover"):
            return self._studio_mux_voiceover(item)
        source = Path(str(item.runtime.get("candidate_qa_reference") or "")).expanduser()
        candidate = next(
            (
                Path(output)
                for output in reversed(item.outputs)
                if Path(output).name.endswith("_candidate.png")
            ),
            None,
        )
        if candidate is None:
            raise ValueError("designer render has no extracted candidate still")
        settings = self._settings()
        try:
            result = compare_candidate_to_reference(
                source,
                candidate,
                slot_id=str(item.runtime.get("candidate_qa_slot") or "unknown"),
                slot_group=str(item.runtime.get("candidate_qa_group") or "body"),
                base_url=str(settings.get("vision_classifier_url") or DEFAULT_SETTINGS["vision_classifier_url"]),
                model=str(
                    item.runtime.get("candidate_qa_model")
                    or settings.get("vision_classifier_model")
                    or DEFAULT_SETTINGS["vision_classifier_model"]
                ),
            )
        except Exception as exc:
            result = {
                "ok": False,
                "decision": "unavailable",
                "same_subject": "uncertain",
                "overall_score": None,
                "scores": {},
                "reasons": ["Local candidate comparison did not complete; manual review is required."],
                "advisory": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result.update(
            {
                "avatar_id": str(item.runtime.get("candidate_qa_avatar_id") or ""),
                "slot_id": str(item.runtime.get("candidate_qa_slot") or ""),
                "group": str(item.runtime.get("candidate_qa_group") or "body"),
                "source_path": str(source.resolve()),
                "candidate_path": str(candidate.resolve()),
            }
        )
        avatar_id = result["avatar_id"]
        if avatar_id and self.library.get_avatar(avatar_id):
            self.library.record_candidate(
                avatar_id,
                {
                    "path": result["candidate_path"],
                    "source_path": result["source_path"],
                    "slot_id": result["slot_id"],
                    "group": result["group"],
                    "status": "pending",
                    "qa_decision": result["decision"],
                    "scores": result["scores"],
                    "reasons": result["reasons"],
                },
            )
            result["manifest_recorded"] = True
        else:
            result["manifest_recorded"] = False
        return result

    # ------------------------------------------------------------------
    # Avatar Vault / Motion Library / Scene Library
    # ------------------------------------------------------------------

    def vault_list(self) -> dict[str, Any]:
        return {"ok": True, "avatars": self.library.avatars()}

    def vault_save(self, avatar: dict[str, Any]) -> dict[str, Any]:
        try:
            saved = self.library.put_avatar(avatar)
            return {
                "ok": True,
                "avatar": saved,
                "manifest_path": str(self.library.manifest_path(saved["avatar_id"])),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def avatar_manifest_open(self, avatar_id: str) -> dict[str, Any]:
        try:
            path = self.library.manifest_path(avatar_id)
            if not path.is_file():
                return {"ok": False, "error": "Save the avatar once to create its structured manifest."}
            return self.open_path(str(path))
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def avatar_candidate_decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Explicitly accept a face/body candidate into a profile or reject it."""
        try:
            avatar_id = str(payload.get("avatar_id") or "").strip()
            slot = slot_by_id(str(payload.get("slot_id") or ""))
            decision = str(payload.get("decision") or "").strip()
            if decision not in {"accepted", "rejected"}:
                return {"ok": False, "error": "decision must be accepted or rejected"}
            candidate_path = Path(str(payload.get("path") or "")).expanduser().resolve()
            if not candidate_path.is_file() or candidate_path.suffix.lower() not in IMAGE_EXTENSIONS:
                return {"ok": False, "error": "candidate must be a readable local still image"}
            entry = self.library.get_avatar(avatar_id)
            if not entry:
                return {"ok": False, "error": f"avatar '{avatar_id}' not found"}
            profile = AvatarProfile.model_validate(entry)
            if decision == "accepted" and slot.group == "anatomy":
                return {
                    "ok": False,
                    "error": "Generated anatomy cannot be accepted as source evidence. Add a verified exact source image to that anatomy slot.",
                }
            candidates = []
            matched = False
            for item in profile.build_manifest.candidates:
                if Path(item.path).expanduser() == candidate_path:
                    candidates.append(item.model_copy(update={"status": decision}))
                    matched = True
                else:
                    candidates.append(item)
            if not matched:
                source_path = str(payload.get("source_path") or candidate_path)
                candidate = {
                    "path": str(candidate_path),
                    "source_path": source_path,
                    "slot_id": slot.id,
                    "group": slot.group,
                    "status": decision,
                    "qa_decision": str(payload.get("qa_decision") or "unavailable"),
                }
                candidates.append(AvatarCandidateRecord.model_validate(candidate))
            updates: dict[str, Any] = {
                "build_manifest": profile.build_manifest.model_copy(update={"candidates": candidates})
            }
            if decision == "accepted":
                key = "identity_refs" if slot.group == "identity" else "body_refs"
                refs = dict(getattr(profile, key))
                refs[slot.role] = str(candidate_path)
                updates[key] = refs
                if slot.group == "identity":
                    # Bind the accepted file by content so later jobs can prove they use this exact identity.
                    hashes = dict(profile.reference_hashes)
                    hashes[slot.role] = sha256_file(candidate_path) or ""
                    updates["reference_hashes"] = hashes
            saved = self.library.put_avatar(profile.model_copy(update=updates).model_dump())
            queue_id = str(payload.get("queue_id") or "").strip()
            if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", queue_id):
                history_path = HISTORY_DIR / f"{queue_id}.json"
                history = _read_json(history_path, {})
                if isinstance(history, dict) and history:
                    qa = dict(history.get("candidate_qa") or {})
                    qa["profile_status"] = decision
                    history["candidate_qa"] = qa
                    _write_json(history_path, history)
            return {
                "ok": True,
                "avatar": saved,
                "manifest_path": str(self.library.manifest_path(avatar_id)),
            }
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
        *,
        full_reference_pool: bool = False,
        full_component_paths: list[str] | None = None,
        primary_source: str = "",
        cycle_index: int = 0,
        cycle_total: int = 1,
    ) -> list[str]:
        refs: list[str] = []
        prioritized: list[str] = []
        include_profile_body = slot is None or slot.group != "identity"
        if slot is not None:
            if full_reference_pool and slot.group == "identity" and profile.identity_refs.get(slot.role):
                prioritized.append(profile.identity_refs[slot.role])
            elif slot.group == "body" and profile.body_refs.get(slot.role):
                prioritized.append(profile.body_refs[slot.role])
            elif full_reference_pool and slot.group == "anatomy" and profile.anatomy_refs.get(slot.role):
                prioritized.append(profile.anatomy_refs[slot.role])
        if not full_reference_pool:
            prioritized.extend(component_paths or [])
        if full_reference_pool:
            pool_candidates = [
                *profile.source_refs,
                *profile.ordered_identity,
                *profile.body_refs.values(),
                *profile.anatomy_refs.values(),
                *profile.hair_refs.values(),
                *profile.wardrobe_refs.values(),
                *(full_component_paths or []),
            ]
            pool: list[str] = []
            for value in pool_candidates:
                path = str(value or "").strip()
                if path and path != primary_source and path not in prioritized and path not in pool:
                    pool.append(path)
            if pool:
                total = max(1, cycle_total)
                start = (max(0, cycle_index) * len(pool)) // total
                prioritized.extend([*pool[start:], *pool[:start]])
        elif include_profile_body:
            prioritized.extend(profile.body_refs.values())
        for value in prioritized:
            path = str(value or "").strip()
            if path and path != primary_source and path not in refs:
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
            coverage_mode = str(payload.get("coverage_variant") or profile.nude_mode)
            if coverage_mode == "both":
                coverage_mode = "nude"
            if coverage_mode not in {"nude", "clothed"}:
                return {"ok": False, "error": "Profile coverage must be nude, clothed, or both."}
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
            if (coverage_mode == "nude" or slot.adult_only or guided or body_referenced) and not (
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
                sex=profile.sex,
                coverage_mode=coverage_mode,
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
            coverage_mode = str(payload.get("coverage_variant") or profile.nude_mode)
            if coverage_mode == "both":
                coverage_mode = "nude"
            if coverage_mode not in {"nude", "clothed"}:
                return {"ok": False, "error": "Profile coverage must be nude, clothed, or both."}
            validation_run = bool(payload.get("validation_run"))
            full_reference_pool = validation_run and bool(payload.get("full_reference_pool"))
            if slot.group == "anatomy" and (validation_run or requested_mode == "fl2va"):
                return {
                    "ok": False,
                    "error": (
                        "Exact FL2VA anatomy regeneration is disabled because temporal pose conversion can invent or "
                        "deform anatomy. Keep the verified source image as profile evidence, or explicitly choose an "
                        "experimental guided/body-reference route."
                    ),
                }
            try:
                cycle_index = max(0, int(payload.get("reference_cycle_index") or 0))
                cycle_total = max(1, int(payload.get("reference_cycle_total") or 1))
            except (TypeError, ValueError):
                cycle_index, cycle_total = 0, 1
            source = str(payload.get("source_image") or "").strip()
            if not source:
                source = next((path for path in [*profile.source_refs, *profile.ordered_identity] if path), "")
            if not source or not Path(source).expanduser().is_file():
                return {"ok": False, "error": "Select an existing primary source image before generating a candidate."}
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
            full_component_paths = [path for component in all_components for path in component.get("paths") or []]
            component_labels = [f"{component['label']} ({component['category'].replace('_', ' ')})" for component in components]
            saved_body_ref_count = len({str(value).strip() for value in profile.body_refs.values() if str(value).strip()})
            # A validation run may cycle through many exact source images, but it
            # must not silently change FL2VA into Ref2VA.  Ref2VA is an explicit,
            # experimental route; exact-primary validation keeps the selected
            # source as the FL2VA first frame.
            use_reference_pool = requested_mode == "body_ref2va"
            body_refs = self._designer_body_refs(
                profile,
                slot,
                component_paths,
                full_reference_pool=full_reference_pool,
                full_component_paths=full_component_paths,
                primary_source=source,
                cycle_index=cycle_index,
                cycle_total=cycle_total,
            ) if use_reference_pool else []
            effective_mode = requested_mode
            if effective_mode == "body_ref2va" and not body_refs and (
                full_reference_pool or not (saved_body_ref_count or all_components)
            ):
                return {"ok": False, "error": "Add at least one saved body / proportions reference or ready appearance component before using Ref2VA."}
            selected_guides = self._selected_designer_guides(profile) if effective_mode == "guided_ref2va" else []
            if effective_mode == "guided_ref2va" and not selected_guides:
                return {"ok": False, "error": "Select at least one validated anatomy guide before using guided Ref2VA."}
            guides = self._selected_designer_guides(profile, slot) if selected_guides else []
            guided = bool(guides)
            body_referenced = bool(body_refs)
            if (coverage_mode == "nude" or slot.adult_only or guided or body_referenced) and not (
                profile.subject_kind != "unknown"
                and profile.age_verified_18_plus
                and profile.consent_confirmed
            ):
                return {
                    "ok": False,
                    "error": "Adult anatomy slots require a synthetic or consenting-adult profile with verified 18+ age and confirmed consent.",
                }
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
                sex=profile.sex,
                coverage_mode=coverage_mode,
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
                    "sex": profile.sex,
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
                    "content_class": "adult_nudity" if coverage_mode == "nude" or slot.adult_only or guided else "general",
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
                "saved_body_ref_count": saved_body_ref_count if effective_mode == "body_ref2va" else 0,
                "component_count": len(components) if body_referenced else 0,
                "profile_coverage": coverage_mode,
                "validation_run": validation_run,
                "full_reference_pool": full_reference_pool,
                "reference_cycle_index": cycle_index if full_reference_pool else 0,
                "reference_cycle_total": cycle_total if full_reference_pool else 0,
                "candidate_qa": True,
                "candidate_qa_reference": str(Path(source).expanduser().resolve()),
                "candidate_qa_avatar_id": profile.avatar_id,
                "candidate_qa_slot": slot.id,
                "candidate_qa_group": slot.group,
                "candidate_qa_model": str(
                    payload.get("vision_model")
                    or self._settings().get("vision_classifier_model")
                    or DEFAULT_SETTINGS["vision_classifier_model"]
                ),
            }
            output_kind = "validation" if validation_run else "designer"
            job.asset_map["OUTPUT_PREFIX"] = f"avatar_{output_kind}_h3_{profile.avatar_id}_{slot.role}"
            item = QueuedRender(
                job=job,
                workflow=workflow,
                provider_config=self._provider_config(render_payload),
                profile=str(runtime.get("profile") or "low_memory_quality"),
                runtime=runtime,
                output_dir=payload.get("output_dir"),
                archive_dir=str(self._archive_dir()) if self._archive_dir() else None,
                label=f"Avatar {'Validation' if validation_run else 'Designer'} H3 · {profile.display_name} · {slot.label}",
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
                "saved_body_ref_count": saved_body_ref_count if effective_mode == "body_ref2va" else 0,
                "component_count": len(components) if body_referenced else 0,
                "validation_run": validation_run,
                "full_reference_pool": full_reference_pool,
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
        alive = _proc_alive(pid)
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

    @staticmethod
    def _default_comfyui_cache_ram_args(ram_total_mb: int | None = None) -> list[str]:
        """Bound ComfyUI's RAM pressure cache so host desktop RAM does not thrash into swap.

        Override via ``AVATAR_V2_COMFY_CACHE_RAM="6 24"`` (or ``"off"`` to omit).
        """
        override = os.getenv("AVATAR_V2_COMFY_CACHE_RAM", "").strip()
        if override.lower() in {"0", "off", "none", "false"}:
            return []
        if override:
            tokens = [tok for tok in override.split() if re.fullmatch(r"\d+(?:\.\d+)?", tok)]
            if 1 <= len(tokens) <= 2:
                return ["--cache-ram", *tokens]
        if ram_total_mb is None:
            try:
                from .resource_monitor import query_host_memory

                ram_total_mb = query_host_memory().ram_total_mb
            except Exception:
                ram_total_mb = 0
        if ram_total_mb >= 48_000:
            return ["--cache-ram", "6", "24"]
        if ram_total_mb >= 24_000:
            return ["--cache-ram", "4", "12"]
        if ram_total_mb >= 12_000:
            return ["--cache-ram", "2", "6"]
        return []

    @classmethod
    def _build_comfyui_launch_args(
        cls,
        settings: dict[str, Any],
        port: str,
        *,
        ram_total_mb: int | None = None,
    ) -> list[str]:
        """Build safe ComfyUI CLI args, preserving multi-token flags and bounded RAM cache."""
        raw_tokens = str(settings.get("comfyui_args") or "").split()
        extra: list[str] = []
        expect_value_for: str | None = None
        # Flags that accept one or two scalar arguments (numbers or simple identifiers).
        _VALUE_FLAGS = {
            "--cache-ram": 2,
            "--cache-lru": 1,
            "--reserve-vram": 1,
            "--async-offload": 1,
            "--preview-method": 1,
            "--cuda-device": 1,
        }
        remaining_vals = 0
        for tok in raw_tokens:
            if tok.startswith("--"):
                if not re.fullmatch(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", tok):
                    remaining_vals = 0
                    expect_value_for = None
                    continue
                extra.append(tok)
                expect_value_for = tok
                remaining_vals = _VALUE_FLAGS.get(tok, 0)
            elif remaining_vals > 0 and expect_value_for and re.fullmatch(r"[a-zA-Z0-9._-]+", tok):
                extra.append(tok)
                remaining_vals -= 1
            else:
                remaining_vals = 0
                expect_value_for = None

        cache_flags = {"--cache-ram", "--cache-classic", "--cache-lru", "--cache-none"}
        if not any(tok in cache_flags for tok in extra):
            extra.extend(cls._default_comfyui_cache_ram_args(ram_total_mb=ram_total_mb))
        return ["main.py", "--listen", "127.0.0.1", "--port", str(port), *extra]

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
        launch_args = self._build_comfyui_launch_args(settings, port)
        try:
            COMFY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log = open(COMFY_LOG_PATH, "ab")
            proc = subprocess.Popen(
                [python, *launch_args],
                cwd=str(comfyui_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        COMFY_PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        return {"ok": True, "state": "STARTING", "pid": proc.pid}

    def _engine_restart_hook(self) -> None:
        """The queue's recovery hook: start our own ComfyUI, and raise when it cannot start
        so the failure lands in the render's memory events instead of being silently lost."""
        result = self.engine_autostart()
        if not result.get("ok") and result.get("error"):
            raise RuntimeError(str(result["error"]))

    def comfyui_stop(self, wait_s: float = 90.0, force: bool = False) -> dict[str, Any]:
        """Ask the ComfyUI that Avatar V2 started to exit.

        A loaded engine can take a minute to release its models, so a slow exit is reported as
        "still shutting down", not as an error. With ``force=True`` (used at app shutdown), the
        app then kills its *own* engine process if it hasn't exited in time, so the GPU is free
        for other programs. Force-kill is scoped to the pid this app started; it never touches
        any other process.
        """
        current = self.comfyui_status()
        if not current["ready"] and not current["pid"]:
            try:  # a stale pid file would otherwise look like a crashed engine on next start
                COMFY_PID_PATH.unlink()
            except FileNotFoundError:
                pass
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
        deadline = time.monotonic() + max(float(wait_s), 0.0)
        while time.monotonic() < deadline and _proc_alive(pid):
            time.sleep(0.5)
        if _proc_alive(pid):
            if force:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                time.sleep(0.5)
                try:
                    COMFY_PID_PATH.unlink()
                except FileNotFoundError:
                    pass
                if not _proc_alive(pid):
                    return {"ok": True, "state": "STOPPED", "pid": pid,
                            "message": "ComfyUI was force-stopped at shutdown so the GPU is free."}
            return {"ok": True, "state": "STOPPING", "pid": pid,
                    "message": "ComfyUI is still shutting down (releasing its models); it finishes on its own."}
        try:
            COMFY_PID_PATH.unlink()
        except FileNotFoundError:
            pass
        return {"ok": True, "state": "STOPPED", "pid": pid}

    def shutdown_engine(self) -> None:
        """App close-down: stop queue work and make sure the GPU is fully released.

        Order matters: the queue stops picking new work, ComfyUI is asked to unload its models
        (fast VRAM release through ``POST /free``), and the app's own engine process is stopped —
        force-killed if it hasn't exited within the grace period — so closing the app always
        frees the GPU for videos, games, or other GPU work.
        """
        try:
            mgr = getattr(self, "_script_run_manager", None)
            if mgr is not None:
                mgr.stop()
        except Exception:
            pass
        try:
            self._close_media_server()
        except Exception:
            pass
        try:
            self.queue.shutdown()
        except Exception:
            pass
        try:
            ComfyUIProvider(self._provider_config({})).free_memory(unload_models=True, free_memory=True)
        except Exception:
            pass
        try:
            self.comfyui_stop(wait_s=10.0, force=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # History and best-of-N
    # ------------------------------------------------------------------

    def history(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for path in HISTORY_DIR.glob("*.json"):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(entry, dict):
                    continue
                entry["history_modified_at"] = path.stat().st_mtime
                entry["history_file"] = path.name
                outputs = entry.get("outputs")
                if isinstance(outputs, list):
                    entry["output_exists"] = {
                        output: Path(output).expanduser().is_file()
                        for output in outputs
                        if isinstance(output, str)
                    }
                entries.append(entry)
            except (json.JSONDecodeError, OSError):
                continue
        entries.sort(
            key=lambda item: float(
                item.get("history_modified_at")
                or item.get("finished_at")
                or item.get("started_at")
                or 0
            ),
            reverse=True,
        )
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

    # ------------------------------------------------------------------ Runway (cloud)
    #
    # Runway is ready as soon as an API key is saved. Jobs queue on one click; the cost is shown
    # before you click and recorded afterwards. Spending limits apply only when the owner sets one
    # (read from the *persisted* settings, never from a UI payload). The app never resubmits a
    # paid job on its own.

    @staticmethod
    def _runway_limit(settings: dict[str, Any], key: str) -> int:
        try:
            return max(int(settings.get(key) or 0), 0)
        except (TypeError, ValueError):
            return 0

    def _runway_public_status(self, settings: dict[str, Any]) -> dict[str, Any]:
        key = secret_status("runway")
        return {
            "ready": bool(key.get("configured")),
            "key": key,
            "models": runway_api.models_payload(),
            "variations": variations_payload(),
            "voices": list(runway_api.VOICE_PRESETS),
            "avatar_presets": list(runway_api.AVATAR_PRESETS),
            "limits": {
                "job": self._runway_limit(settings, "runway_job_limit"),
                "batch": self._runway_limit(settings, "runway_batch_limit"),
                "daily": self._runway_limit(settings, "runway_daily_limit"),
                "monthly": self._runway_limit(settings, "runway_monthly_limit"),
            },
            "default_personality": DEFAULT_PRESENTER_PERSONALITY,
            "api_version": runway_api.RUNWAY_API_VERSION,
            "docs_verified": runway_api.DOCS_VERIFIED,
            "credit_usd": runway_api.CREDIT_USD,
        }

    def runway_status(self) -> dict[str, Any]:
        return {"ok": True, **self._runway_public_status(self._settings())}

    def runway_save_key(self, api_key: str) -> dict[str, Any]:
        try:
            stored_in = save_secret("runway", api_key)
        except SecretStoreError as exc:
            return {"ok": False, "error": str(exc), "key": secret_status("runway")}
        return {"ok": True, "stored_in": stored_in, "key": secret_status("runway")}

    def runway_clear_key(self) -> dict[str, Any]:
        removed = clear_secret("runway")
        status = secret_status("runway")
        note = (
            "The RUNWAYML_API_SECRET environment variable still supplies a key; unset it to disable Runway access."
            if status.get("source") == "environment"
            else ""
        )
        return {"ok": True, "removed": removed, "key": status, "note": note}

    @staticmethod
    def _runway_gate(settings: dict[str, Any] | None = None, *, uploads: bool = False) -> str | None:
        """The only thing a Runway call needs is an API key; there is no enable or consent switch."""
        if not secret_status("runway")["configured"]:
            return (
                "Add your Runway API key first: Studio → Connect Runway, or run scripts/set_runway_key.sh "
                "(or set RUNWAYML_API_SECRET)."
            )
        return None

    def runway_check(self) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        try:
            with runway_api.client_from_store(timeout_s=30) as client:
                organization = client.organization()
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc), "category": exc.error_category}
        tier = organization.get("tier") or {}
        return {
            "ok": True,
            "credit_balance": organization.get("creditBalance"),
            "max_monthly_credit_spend": tier.get("maxMonthlyCreditSpend"),
            "models_with_limits": sorted((tier.get("models") or {}).keys())[:80],
        }

    @staticmethod
    def _runway_media_facts(spec: dict[str, Any], asset_map: dict[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        media = spec.get("media") or {}
        for key, fact in (("reference_video", "reference_video_s"), ("speech_audio", "speech_audio_s")):
            entry = media.get(key)
            if entry and asset_map.get(entry.get("asset", "")):
                facts[fact] = runway_api.media_duration_s(str(asset_map[entry["asset"]]))
        image = media.get("image")
        if image and asset_map.get(image.get("asset", "")):
            try:
                with Image.open(str(asset_map[image["asset"]])) as handle:
                    facts["image_size"] = handle.size
            except OSError:
                pass
        return facts

    @staticmethod
    def _aspect_warning(path: Any, ratio: str | None) -> str | None:
        if not path or not ratio:
            return None
        try:
            with Image.open(str(path)) as handle:
                width, height = handle.size
        except OSError:
            return None
        target_w, target_h = runway_api.ratio_dims(ratio)
        source, target = width / height, target_w / target_h
        if abs(source - target) / target > 0.05:
            return (
                f"The first frame is {width}x{height} but the output ratio is {ratio}; Runway will center-crop it. "
                "Match the still's orientation to avoid losing the head or hands."
            )
        return None

    def _runway_build(self, payload: dict[str, Any], *, variation_id: str | None = None) -> dict[str, Any]:
        """Turn a UI payload into a validated Runway job, spec, estimate and upload receipt (no network)."""
        settings = self._settings()
        options = dict(payload.get("runway") or {})
        task = str(options.get("task") or "still")
        if task == "variations":
            task = "still"
        endpoint_task = runway_api.TASK_ENDPOINT_TASK.get(task)
        if endpoint_task is None:
            raise runway_api.RunwayValidationError(f"Unknown Runway task '{task}'.")
        default_model = {
            "text_to_image": settings.get("runway_default_image_model") or "gen4_image",
            "image_to_video": settings.get("runway_default_video_model") or "gen4.5",
            "character_performance": "act_two",
            "avatar_video": "gwm1_avatars",
            "image_upscale": "magnific_precision_upscaler_v2",
            "text_to_speech": "eleven_v4",
        }[endpoint_task]
        model = runway_api.model_spec(str(options.get("model") or default_model))
        if model.task != endpoint_task:
            raise runway_api.RunwayValidationError(f"{model.label} cannot run the '{task}' task.")

        candidate = json.loads(json.dumps(payload))
        candidate.setdefault("shot", {})["engine_preference"] = "runway"
        avatar, shot = self._models_from_payload(candidate)

        is_video = endpoint_task in {"image_to_video", "character_performance", "avatar_video"}
        ratio = options.get("ratio") or (model.ratios[0] if model.ratios else None)
        width, height = runway_api.ratio_dims(ratio) if ratio else (shot.width, shot.height)
        duration = int(options.get("duration") or model.default_duration) if model.duration_range else None
        shot = shot.model_copy(
            update={
                "engine_preference": "runway",
                "local_only": False,
                "width": max(256, width),
                "height": max(256, height),
                "duration_s": float(min(duration or 1, 60)) if is_video else 1.0,
                "fps": 24 if is_video else 1,
            }
        )
        decision = evaluate_cloud_policy(avatar, shot)
        if not decision.allowed:
            raise PolicyError(decision.reason)
        job = build_job(avatar, shot)
        lock = self._identity_lock(avatar)
        local_media: dict[str, Any] = {}
        post: dict[str, Any] = {}

        params: dict[str, Any] = {}
        media: dict[str, Any] = {}
        prompt, dropped = "", []
        warnings: list[str] = []
        seed = shot.seed if model.supports_seed else None

        if endpoint_task == "text_to_image":
            references: list[dict[str, Any]] = []
            identity_tag = outfit_tag = location_tag = None
            human_subject = "human" if model.max_human_refs else None
            if avatar.identity_refs:
                identity_tag = "presenter"
                drift = self._identity_drift_warning(lock, job.asset_map.get("IDENTITY_REF_1"), role="primary identity reference")
                if drift:
                    warnings.append(drift)
                references.append(
                    {"asset": "IDENTITY_REF_1", "tag": identity_tag, "subject": human_subject, "role": "primary identity"}
                )
                for index in range(2, min(len(avatar.identity_refs), model.max_human_refs) + 1):
                    references.append({"asset": f"IDENTITY_REF_{index}", "subject": "human", "role": f"identity angle {index}"})
            else:
                warnings.append(
                    "No identity reference is set, so this is pure text-to-image. Use it to design a new fictional "
                    "presenter, then set the approved still as the primary identity for consistent variations."
                )
            object_subject = "object" if model.max_human_refs else None
            if avatar.wardrobe_refs and len(references) < model.max_refs:
                outfit_tag = "outfit"
                references.append({"asset": "WARDROBE_REF_1", "tag": outfit_tag, "subject": object_subject, "role": "wardrobe"})
            if shot.references.scene_image and len(references) < model.max_refs:
                location_tag = "location"
                references.append({"asset": "SCENE_IMAGE", "tag": location_tag, "subject": object_subject, "role": "location"})
            if len(references) < model.min_refs:
                raise runway_api.RunwayValidationError(
                    f"{model.label} needs at least {model.min_refs} reference image; set the primary identity first."
                )
            chosen = variation_id or options.get("variation") or options.get("slot")
            variation = variation_by_id(str(chosen)) if chosen else None
            if chosen and lock.get("profile_id"):
                # Slot outputs become pending candidates in the vault profile; never auto-accepted.
                post.update(
                    {
                        "candidate_slot": str(chosen),
                        "candidate_avatar_id": lock["profile_id"],
                        "candidate_reference": str(job.asset_map.get("IDENTITY_REF_1") or ""),
                    }
                )
            prompt, dropped = compile_presenter_prompt(
                avatar,
                shot,
                budget=model.prompt_max,
                identity_tag=identity_tag,
                outfit_tag=outfit_tag,
                location_tag=location_tag,
                variation=variation,
                realism=brand_instructor.REALISM_GUARDS if avatar.avatar_id == brand_instructor.PROFILE_ID else None,
            )
            params = {"prompt": prompt, "ratio": ratio, "seed": seed, "output_count": int(options.get("output_count") or 1)}
            media = {"references": [{k: v for k, v in ref.items() if v is not None} for ref in references]}
        elif endpoint_task == "image_to_video":
            if not job.asset_map.get("INIT_IMAGE"):
                raise runway_api.RunwayValidationError(
                    "Set the approved presenter still as the primary identity or first frame."
                )
            prompt, dropped = compile_presenter_prompt(avatar, shot, budget=model.prompt_max, motion=True)
            params = {"prompt": prompt, "duration": duration, "seed": seed}
            if model.ratios:
                params["ratio"] = ratio
                aspect = self._aspect_warning(job.asset_map.get("INIT_IMAGE"), ratio)
                if aspect:
                    warnings.append(aspect)
            if model.resolutions:
                params["resolution"] = options.get("resolution") or model.resolutions[0]
            if model.id == "h3_max":
                params["prompt_expansion"] = options.get("prompt_expansion") or "disabled"
            media = {"first_frame": {"asset": "INIT_IMAGE", "role": "first frame"}}
            approved = set((lock.get("approved") or {}).values())
            if approved and sha256_file(job.asset_map.get("INIT_IMAGE") or "") not in approved:
                warnings.append("The first frame is not an approved reference of this profile, so identity may drift.")
            if job.asset_map.get("LAST_FRAME"):
                if model.supports_last_frame:
                    media["last_frame"] = {"asset": "LAST_FRAME", "role": "last frame", "optional": True}
                else:
                    warnings.append(f"{model.label} ignores the last frame; MiniMax H3 Max accepts first + last keyframes.")
            if options.get("output_audio"):
                params["output_audio"] = str(options["output_audio"])
                if params["output_audio"] == "voiceover":
                    self._attach_voiceover(options, job, local_media)
        elif endpoint_task == "character_performance":
            if not job.asset_map.get("INIT_IMAGE"):
                raise runway_api.RunwayValidationError("Set the presenter still (primary identity) as the Act-Two character.")
            if not job.asset_map.get("MOTION_VIDEO"):
                raise runway_api.RunwayValidationError("Select a 3-30 s performance video in the Motion video slot.")
            drift = self._identity_drift_warning(lock, job.asset_map.get("INIT_IMAGE"), role="Act-Two character image", any_approved=True)
            if drift:
                warnings.append(drift)
            output_audio = str(options.get("output_audio") or ("voiceover" if options.get("voiceover_id") else "mute"))
            params = {
                "ratio": ratio,
                "seed": seed,
                "body_control": bool(options.get("body_control", True)),
                "expression_intensity": int(options.get("expression_intensity") or 3),
                "output_audio": output_audio,
            }
            media = {
                "character": {"asset": "INIT_IMAGE", "type": "image", "role": "presenter (character)"},
                "reference_video": {"asset": "MOTION_VIDEO", "role": "performance reference (audio removed)"},
            }
            if output_audio == "voiceover":
                self._attach_voiceover(options, job, local_media)
            warnings.append(
                "The performance video's audio track is removed before upload, so the presenter never receives "
                "the driving performer's voice."
            )
        elif endpoint_task == "avatar_video":
            avatar_id = str(options.get("avatar_id") or "").strip()
            preset = str(options.get("avatar_preset") or "").strip()
            if avatar_id:
                avatar_ref = {"type": "custom", "avatarId": avatar_id}
            elif preset:
                avatar_ref = {"type": "runway-preset", "presetId": preset}
            else:
                raise runway_api.RunwayValidationError(
                    "Choose your Runway presenter avatar (create one from the primary identity still first)."
                )
            if options.get("speech") == "audio":
                speech: dict[str, Any] = {"type": "audio"}
                job.asset_map["RUNWAY_NARRATION"] = self._narration_file(options, job)
                media = {"speech_audio": {"asset": "RUNWAY_NARRATION", "role": "narration audio"}}
            else:
                speech = {"type": "text", "text": str(options.get("script") or "").strip()}
                voice = str(options.get("voice") or "").strip()
                if voice:
                    speech["voice"] = {"type": "preset", "presetId": voice}
            params = {"avatar": avatar_ref, "speech": speech}
            if options.get("output_audio"):
                params["output_audio"] = str(options["output_audio"])
                if params["output_audio"] == "voiceover":
                    self._attach_voiceover(options, job, local_media)
            source = (self._runway_avatar_index().get(avatar_id) or {}) if avatar_id else {}
            master = str(lock.get("master_sha256") or "")
            if avatar_id and master and not master.startswith(str(source.get("reference_sha256") or "-")):
                warnings.append(
                    "This Runway avatar was not created from the approved master still (or predates it); "
                    "recreate it from the master to keep the identity locked."
                )
        elif endpoint_task == "text_to_speech":
            profile = self._voiceovers().get_profile(str(options.get("voice_profile_id") or "")) or {}
            profile_voice = profile.get("voice") or {}
            script = apply_pronunciations(str(options.get("script") or "").strip(), profile.get("pronunciations"))
            custom_id = str(options.get("tts_voice_id") or (profile_voice.get("id") if profile_voice.get("type") == "custom" else "") or "")
            preset_id = str(options.get("tts_voice") or profile_voice.get("presetId") or "")
            voice_ref = {"type": "custom", "id": custom_id} if custom_id else {"type": "runway-preset", "presetId": preset_id}
            prompt = script
            params = {"prompt": script, "voice": voice_ref, "seed": seed}
            for key in ("speed", "stability", "similarity_boost", "style"):
                value = options.get(key, profile.get(key))
                if value is not None and key in model.extras:
                    params[key] = float(value)
            language = options.get("language_code") or profile.get("language_code")
            if language and "language_code" in model.extras:
                params["language_code"] = str(language)
            for key in ("previous_text", "next_text"):
                if options.get(key) and "context" in model.extras:
                    params[key] = str(options[key])
            post.update({"register_voiceover": True, "voice_profile_id": str(profile.get("id") or ""), "script": script})
        elif endpoint_task == "image_upscale":
            source = str(options.get("image") or job.asset_map.get("INIT_IMAGE") or "")
            if not source:
                raise runway_api.RunwayValidationError("Select the still to upscale (primary identity or first frame).")
            job.asset_map["RUNWAY_UPSCALE_SOURCE"] = source
            params = {"scale_factor": int(options.get("scale_factor") or 2), "flavor": options.get("flavor") or "photo"}
            media = {"image": {"asset": "RUNWAY_UPSCALE_SOURCE", "role": "still to upscale"}}

        params = {key: value for key, value in params.items() if value is not None}
        spec: dict[str, Any] = {"version": 1, "model": model.id, "task": task, "params": params, "media": media}
        if local_media:
            spec["local"] = local_media
        spec["identity"] = {key: lock.get(key) for key in ("profile_id", "master_sha256", "primary_sha256")}
        runway_api.validate_spec(spec, job.asset_map)
        facts = self._runway_media_facts(spec, job.asset_map)
        reference_seconds = facts.get("reference_video_s")
        if reference_seconds is not None and not 3.0 <= reference_seconds <= 30.0:
            raise runway_api.RunwayValidationError(
                f"The performance video is {reference_seconds:.1f} s; Act-Two needs 3-30 s."
            )
        estimate = runway_api.estimate_credits(model.id, params, facts)
        cap = self._runway_limit(settings, "runway_job_limit")  # 0 = no limit (the default)
        spec["estimate"] = estimate
        spec["max_credits"] = cap or None

        spoken = str((params.get("speech") or {}).get("text") or "")
        job.prompt = prompt or spoken or f"Runway {model.label}"
        job.asset_map["PROMPT"] = job.prompt
        job.asset_map["RUNWAY_SPEC"] = json.dumps(spec)
        runtime = {
            "provider": "runway",
            "profile": "cloud_runway",
            "engine": "runway",
            "model": model.id,
            "task": task,
            "estimate_credits": estimate["credits"],
            "reason": f"Runway cloud job · {model.label}",
        }
        if post:
            runtime.update(post)
            runtime["candidate_qa"] = True  # routes the finished job through _runway_postprocess
        if payload.get("studio_action"):
            runtime["studio_action"] = str(payload["studio_action"])[:32]
        return {
            "job": job,
            "spec": spec,
            "model": model,
            "estimate": estimate,
            "cap": cap,
            "uploads": runway_api.describe_uploads(spec, job.asset_map),
            "warnings": warnings,
            "prompt": prompt,
            "prompt_receipt": presenter_receipt(prompt, dropped, model.prompt_max) if model.prompt_max else None,
            "runtime": runtime,
            "settings": settings,
        }

    def _runway_builds(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """One build, a batch of slot variations, or N seeded candidates of the same still."""
        options = payload.get("runway") or {}
        if options.get("task") == "variations":
            variation_ids = [str(item) for item in options.get("variations") or [] if str(item).strip()]
            if not variation_ids:
                raise runway_api.RunwayValidationError("Choose at least one presenter variation.")
            if len(variation_ids) > len(variations_payload()):
                raise runway_api.RunwayValidationError("Too many variations selected.")
            return [
                {"key": vid, "label": variation_by_id(vid)["label"], "build": self._runway_build(payload, variation_id=vid)}
                for vid in variation_ids
            ]
        count = int(options.get("candidates") or 1)
        if count > 1:
            if options.get("task", "still") != "still" or count > 8:
                raise runway_api.RunwayValidationError("Seeded candidate batches are for stills, at most 8 at a time.")
            base_seed = int((payload.get("shot") or {}).get("seed", 41001))
            builds = []
            for index in range(count):
                candidate = json.loads(json.dumps(payload))
                candidate.setdefault("shot", {})["seed"] = base_seed + index
                builds.append({"key": None, "label": f"candidate · seed {base_seed + index}", "build": self._runway_build(candidate)})
            return builds
        return [{"key": None, "label": None, "build": self._runway_build(payload)}]

    def _runway_limit_error(self, builds: list[dict[str, Any]], settings: dict[str, Any], total: float) -> str | None:
        """Only limits the owner set themselves can stop a job (all are off by default)."""
        job_limit = self._runway_limit(settings, "runway_job_limit")
        if job_limit:
            for entry in builds:
                credits = entry["build"]["estimate"]["credits"]
                if credits > job_limit:
                    return (
                        f"This job is about {credits:g} credits, above the per-job limit you set ({job_limit}). "
                        "Change or clear it under Advanced → Runway."
                    )
        batch_limit = self._runway_limit(settings, "runway_batch_limit")
        if batch_limit and len(builds) > 1 and total > batch_limit:
            return (
                f"This batch is about {total:g} credits, above the batch limit you set ({batch_limit}). "
                "Change or clear it under Advanced → Runway."
            )
        budget = self._budget_status(settings, total)
        return budget["error"]

    def runway_estimate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Local preview of a Runway job: model, cost, uploads, compiled prompt. No network."""
        try:
            builds = self._runway_builds(payload)
        except (runway_api.RunwayError, PolicyError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        first = builds[0]["build"]
        settings = first["settings"]
        total = sum(entry["build"]["estimate"]["credits"] for entry in builds)
        batch = len(builds) > 1 or bool(builds[0]["key"])
        estimate = dict(first["estimate"])
        if batch:
            estimate.update(
                {
                    "credits": total,
                    "usd": round(total * runway_api.CREDIT_USD, 2),
                    "basis": f"{len(builds)} jobs x {first['estimate']['basis']}",
                }
            )
        uploads: list[dict[str, Any]] = []
        for entry in builds:
            for item in entry["build"]["uploads"]:
                if item not in uploads:
                    uploads.append(item)
        limit_error = self._runway_limit_error(builds, settings, total)
        return {
            "ok": True,
            "model": first["model"].public(),
            "task": (payload.get("runway") or {}).get("task") or "still",
            "estimate": estimate,
            "breakdown": [
                {"label": entry["label"], "credits": entry["build"]["estimate"]["credits"]}
                for entry in builds
                if entry["label"]
            ],
            # Limits are optional; 0 means none is set.
            "cap": self._runway_limit(settings, "runway_batch_limit" if batch else "runway_job_limit"),
            "within_cap": limit_error is None,
            "limit_error": limit_error,
            "budget": self._budget_status(settings, total),
            "unresolved": self._unresolved_summary(),
            "uploads": uploads,
            "warnings": first["warnings"],
            "prompt": first["prompt"],
            "prompt_receipt": first["prompt_receipt"],
            "identity": first["spec"].get("identity"),
            "gate": self._runway_gate(settings),
        }

    def _runway_queue_item(self, build: dict[str, Any], payload: dict[str, Any], label: str) -> QueuedRender:
        settings = build["settings"]
        config = ProviderConfig(
            provider="runway",
            base_url=runway_api.RUNWAY_API_BASE,
            output_dir=str(RUNS_DIR / "runway"),
            timeout_s=int(settings.get("runway_timeout_s") or 1800),
            poll_interval_s=float(settings.get("runway_poll_interval_s") or 6),
        )
        return QueuedRender(
            job=build["job"],
            workflow=f"runway:{build['spec']['task']}:{build['model'].id}",
            provider_config=config,
            profile="cloud_runway",
            runtime=build["runtime"],
            archive_dir=str(self._archive_dir()),
            chain_from_previous=bool(payload.get("chain_from_previous")),
            label=label,
        )

    def queue_runway(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue a Runway job, slot-variation batch, or seeded candidate batch on one click."""
        try:
            builds = self._runway_builds(payload)
        except (runway_api.RunwayError, PolicyError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        settings = builds[0]["build"]["settings"]
        missing_key = self._runway_gate(settings)
        if missing_key:
            return {"ok": False, "error": missing_key}
        total = sum(entry["build"]["estimate"]["credits"] for entry in builds)
        limit_error = self._runway_limit_error(builds, settings, total)
        if limit_error:
            return {"ok": False, "error": limit_error, "budget": self._budget_status(settings, total)}
        budget = self._budget_status(settings, total)
        base_label = str(payload.get("label") or "").strip()
        items = []
        for entry in builds:
            build = entry["build"]
            label = base_label or f"runway · {build['model'].label}"
            if entry["label"]:
                label = f"{base_label or ('presenter variation' if entry['key'] else 'runway still')} · {entry['label']}"
            items.append(self._runway_queue_item(build, payload, label))
        queue_ids = self.queue.add_many(items)
        return {
            "ok": True,
            "queue_id": queue_ids[0],
            "queued": [
                {"queue_id": queue_id, "variation": entry["key"], "label": entry["label"]}
                for queue_id, entry in zip(queue_ids, builds)
            ],
            "total_credits": total,
            "budget": budget,
            "unresolved": self._unresolved_summary(),
            "runtime": builds[0]["build"]["runtime"],
            "warnings": builds[0]["build"]["warnings"],
        }

    def _runway_avatar_index(self) -> dict[str, Any]:
        data = _read_json(RUNWAY_AVATARS_PATH, {})
        return data if isinstance(data, dict) else {}

    def runway_avatars_list(self) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        try:
            with runway_api.client_from_store(timeout_s=30) as client:
                data = client.list_avatars(limit=100)
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        index = self._runway_avatar_index()
        items = []
        for entry in data.get("data") or []:
            voice = entry.get("voice") or {}
            avatar_id = str(entry.get("id") or "")
            items.append(
                {
                    "id": avatar_id,
                    "name": entry.get("name"),
                    "status": entry.get("status"),
                    "voice": voice.get("presetId") or voice.get("name"),
                    "created_at": entry.get("createdAt"),
                    "failure_reason": entry.get("failureReason"),
                    "vault_avatar_id": (index.get(avatar_id) or {}).get("vault_avatar_id"),
                }
            )
        return {"ok": True, "items": items, "has_more": bool(data.get("hasMore"))}

    def runway_avatar_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Upload the approved presenter still and create a Runway custom avatar for lip-synced videos."""
        blocked = self._runway_gate()
        if blocked:
            return {"ok": False, "error": blocked}
        try:
            avatar, shot = self._models_from_payload(
                {
                    "avatar": payload.get("avatar") or {},
                    "shot": {"engine_preference": "runway", "content_class": "general", "user_prompt": "presenter avatar"},
                }
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        decision = evaluate_cloud_policy(avatar, shot)
        if not decision.allowed:
            return {"ok": False, "error": decision.reason}
        reference = str(payload.get("reference_image") or (avatar.identity_refs[0] if avatar.identity_refs else "")).strip()
        if not reference:
            return {"ok": False, "error": "Set the approved presenter still as the primary identity first."}
        name = (str(payload.get("name") or avatar.display_name or "Presenter").strip() or "Presenter")[:50]
        voice = str(payload.get("voice") or "").strip() or runway_api.DEFAULT_AVATAR_VOICE
        if voice not in runway_api.VOICE_PRESETS:
            return {"ok": False, "error": f"'{voice}' is not one of Runway's avatar voices: {', '.join(runway_api.VOICE_PRESETS)}."}
        personality = (
            str(payload.get("personality") or "").strip() or DEFAULT_PRESENTER_PERSONALITY.format(name=name)
        )[:10000]
        body: dict[str, Any] = {
            "name": name,
            "personality": personality,
            "voice": {"type": "runway-live-preset", "presetId": voice},
            "imageProcessing": "none" if payload.get("image_processing") == "none" else "optimize",
        }
        start_script = str(payload.get("start_script") or "").strip()
        if start_script:
            body["startScript"] = start_script[:2000]
        try:
            with runway_api.client_from_store(timeout_s=120) as client:
                prepared = runway_api.prepare_media(
                    reference, kind="image", role="avatar reference image", uploader=client.upload
                )
                body["referenceImage"] = prepared.uri
                created = client.create_avatar(body)
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        avatar_id = str(created.get("id") or "")
        if avatar_id:
            index = self._runway_avatar_index()
            index[avatar_id] = {
                "vault_avatar_id": avatar.avatar_id,
                "name": name,
                "voice": voice,
                "reference_file": Path(reference).name,
                "reference_sha256": prepared.sha256[:16],
                "source_sha256": sha256_file(reference),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(RUNWAY_AVATARS_PATH, index)
        return {
            "ok": True,
            "avatar": {"id": avatar_id, "name": created.get("name"), "status": created.get("status")},
            "upload": prepared.public(),
        }

    def runway_avatar_status(self, avatar_id: str) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        if not runway_api._UUID_RE.match(str(avatar_id or "")):
            return {"ok": False, "error": "Runway avatar ids are UUIDs."}
        try:
            with runway_api.client_from_store(timeout_s=30) as client:
                data = client.get_avatar(str(avatar_id))
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "avatar": {
                "id": data.get("id"),
                "name": data.get("name"),
                "status": data.get("status"),
                "failure_reason": data.get("failureReason"),
            },
        }

    def runway_avatar_delete(self, avatar_id: str) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        if not runway_api._UUID_RE.match(str(avatar_id or "")):
            return {"ok": False, "error": "Runway avatar ids are UUIDs."}
        try:
            with runway_api.client_from_store(timeout_s=30) as client:
                client.delete_avatar(str(avatar_id))
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        index = self._runway_avatar_index()
        if index.pop(str(avatar_id), None) is not None:
            _write_json(RUNWAY_AVATARS_PATH, index)
        return {"ok": True, "deleted": str(avatar_id)}

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
    try:  # start the local GPU engine (ComfyUI) in the background when it is installed
        api.engine_autostart()
    except Exception:
        pass
    # Expose only the allowlisted operations, never the API object's internals (queue, library, paths).
    from .js_bridge import JsBridge

    window = webview.create_window(
        "Avatar V2",
        ui_url,
        js_api=JsBridge(api),
        width=1480,
        height=940,
        min_size=(1180, 760),
        background_color="#050507",
        confirm_close=False,
    )
    api._self_window = window
    webview.start(debug=False)
    api.shutdown_engine()  # close-down: free the GPU (stop queue work, /free, stop the engine process)


if __name__ == "__main__":
    main()
