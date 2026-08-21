from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .composer import build_job
from .h3_runtime import H3_FPS, preset_by_id, presets_payload, valid_frame_count
from .models import AvatarManifest, ProviderConfig, ShotSpec
from .policy import evaluate_policy
from .providers import ComfyUIProvider
from .render_queue import QueuedRender, RenderQueue
from .runtime_profiles import payload as runtime_profiles_payload
from .runtime_profiles import resolve_profile

APP_HOME = Path.home() / ".avatar_v2"
CONFIG_PATH = APP_HOME / "desktop.json"
RUNS_DIR = APP_HOME / "runs"
QUEUE_DIR = APP_HOME / "queue"

DEFAULT_SETTINGS: dict[str, Any] = {
    "comfyui_url": "http://127.0.0.1:8188",
    "comfyui_input": "",
    "comfyui_output": "",
    "workflow_h3_ref2va": "",
    "workflow_h3_fl2va": "",
    "workflow_wan22": "",
    "timeout_s": 1800,
    "local_h3_authorized": False,
    "h3_local_fallback": "wan22",
    "default_h3_preset": "vertical_4070",
    "default_runtime_profile": "auto",
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
        self.window = None
        APP_HOME.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.queue = RenderQueue(QUEUE_DIR)

    def _settings(self) -> dict[str, Any]:
        return {**DEFAULT_SETTINGS, **_read_json(CONFIG_PATH, {})}

    def get_state(self) -> dict[str, Any]:
        return {
            "app": "Avatar V2",
            "version": "0.3.0",
            "platform": platform.platform(),
            "settings": self._settings(),
            "app_home": str(APP_HOME),
            "runs_dir": str(RUNS_DIR),
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
        }

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        merged = {**self._settings(), **settings}
        _write_json(CONFIG_PATH, merged)
        return {"ok": True, "settings": merged}

    def choose_files(self, kind: str, multiple: bool = True) -> list[str]:
        if self.window is None:
            return []
        import webview

        filters = {
            "image": ("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)",),
            "video": ("Video (*.mp4;*.mov;*.webm;*.mkv)",),
            "audio": ("Audio (*.wav;*.mp3;*.flac;*.m4a;*.ogg)",),
            "workflow": ("ComfyUI API JSON (*.json)",),
        }
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=multiple,
            file_types=filters.get(kind, ("All files (*.*)",)),
        )
        return [str(Path(p)) for p in (result or [])]

    def choose_folder(self) -> str:
        if self.window is None:
            return ""
        import webview

        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        return str(Path(result[0])) if result else ""

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
        }.get(engine)
        return str(settings.get(key) or "") if key else ""

    def _resolve_runtime(self, payload: dict[str, Any], avatar: AvatarManifest, shot: ShotSpec) -> tuple[Any, dict[str, Any], str]:
        settings = {**self._settings(), **payload.get("settings", {})}
        profile_id = payload.get("runtime_profile") or settings.get("default_runtime_profile", "auto")
        requested_preset = payload.get("h3_preset") or shot.model_dump().get("h3_preset") or settings.get("default_h3_preset")

        if not shot.engine_preference.startswith("h3_") or profile_id == "wan_fallback":
            runtime = {
                "profile": profile_id,
                "engine": "wan22" if profile_id == "wan_fallback" else shot.engine_preference,
                "model": None,
                "preset": requested_preset,
                "steps": 20,
                "cfg": 1.0,
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
        if job.selected_engine.startswith("h3_"):
            job.asset_map["FRAMES"] = valid_frame_count(shot.duration_s)
            if runtime.get("model"):
                job.asset_map["H3_DIFFUSION_MODEL"] = str(runtime["model"])
        workflow = self._workflow_for(payload, job.selected_engine)
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
            if job.selected_engine.startswith("h3_") and not runtime.get("local_h3_allowed", True):
                return {
                    "ok": False,
                    "error": "Local H3 is not authorized in settings; choose Wan fallback or enable H3 only if your use is licensed.",
                }
            item = QueuedRender(
                job=job,
                workflow=workflow,
                provider_config=self._provider_config(payload),
                profile=str(runtime.get("profile") or "auto"),
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


def main() -> None:
    import webview

    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    if not ui_path.exists():
        raise FileNotFoundError(f"Desktop UI not found: {ui_path}")

    api = DesktopAPI()
    window = webview.create_window(
        "Avatar V2",
        str(ui_path),
        js_api=api,
        width=1480,
        height=940,
        min_size=(1180, 760),
        background_color="#050507",
        confirm_close=False,
    )
    api.window = window
    webview.start(debug=False)
    api.queue.shutdown()


if __name__ == "__main__":
    main()
