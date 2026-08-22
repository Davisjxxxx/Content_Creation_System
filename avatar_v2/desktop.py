from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .composer import build_job
from .h3_runtime import H3_FPS, preset_by_id, presets_payload, valid_frame_count
from .library import LibraryStore
from .models import AvatarManifest, ProviderConfig, ShotSpec
from .policy import evaluate_policy
from .providers import ComfyUIProvider
from .render_queue import QueuedRender, RenderQueue
from .runtime_profiles import payload as runtime_profiles_payload
from .runtime_profiles import resolve_profile
from .workflows_builder import WAN_FPS, wan_frame_count

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
    "timeout_s": 1800,
    "local_h3_authorized": False,
    "h3_local_fallback": "wan22",
    "default_h3_preset": "vertical_4070",
    "default_runtime_profile": "auto",
    "gpu_autoclean": True,
    "gpu_threshold": 0.85,
    "gpu_check_interval_s": 5,
    "preferred_output_dir": "",
}


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
        return {
            "app": "Avatar V2",
            "version": "0.3.0",
            "platform": platform.platform(),
            "settings": self._settings(),
            "app_home": str(APP_HOME),
            "runs_dir": str(RUNS_DIR),
            "history_dir": str(HISTORY_DIR),
            "library_dir": str(LIBRARY_DIR),
            "h3_presets": presets_payload(),
            "runtime_profiles": runtime_profiles_payload(),
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
                    options = spec[0] if isinstance(spec, list) and spec else []
                    report[field] = [str(item) for item in options]
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
                "subjects": [
                    {
                        "id": avatar_payload.get("avatar_id") or "desktop-avatar",
                        "kind": avatar_payload.get("subject_kind", "synthetic"),
                        "age_verified_18_plus": bool(avatar_payload.get("age_verified_18_plus", True)),
                        "consent_confirmed": bool(avatar_payload.get("consent_confirmed", True)),
                    }
                ],
                "identity_refs": _clean_paths(avatar_payload.get("identity_refs")),
                "body_refs": _clean_paths(avatar_payload.get("body_refs")),
                "hair_refs": _clean_paths(avatar_payload.get("hair_refs")),
                "wardrobe_refs": _clean_paths(avatar_payload.get("wardrobe_refs")),
                "persistent_features": avatar_payload.get("persistent_features") or [],
            }
        )

        shot_payload = payload.get("shot", {})
        references = shot_payload.get("references", {})
        engine = shot_payload.get("engine_preference", "h3_ref2va")
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
            job = job.model_copy(update={"shot": shot.model_copy(update={"fps": WAN_FPS})})
        job.asset_map.setdefault("OUTPUT_PREFIX", f"avatar_v2_{shot.shot_id[:24]}")
        workflow = self._workflow_for(payload, job.selected_engine)
        if not workflow and job.selected_engine in {"h3_ref2va", "h3_fl2va", "wan22"}:
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
            return {"ok": True, "engine": job.selected_engine, "runtime": runtime, "resolved_workflow": str(resolved_path)}
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
            return {"ok": True, "queue_id": item_id, "runtime": runtime, "engine": job.selected_engine}
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
