from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .composer import build_job
from .models import AvatarManifest, ProviderConfig, ShotSpec
from .policy import evaluate_policy
from .providers import ComfyUIProvider

APP_HOME = Path.home() / ".avatar_v2"
CONFIG_PATH = APP_HOME / "desktop.json"
RUNS_DIR = APP_HOME / "runs"

DEFAULT_SETTINGS: dict[str, Any] = {
    "comfyui_url": "http://127.0.0.1:8188",
    "comfyui_input": "",
    "comfyui_output": "",
    "workflow_h3_ref2va": "",
    "workflow_h3_fl2va": "",
    "workflow_wan22": "",
    "timeout_s": 1800,
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
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{now}"


class DesktopAPI:
    def __init__(self) -> None:
        self.window = None
        APP_HOME.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def _settings(self) -> dict[str, Any]:
        saved = _read_json(CONFIG_PATH, {})
        return {**DEFAULT_SETTINGS, **saved}

    def get_state(self) -> dict[str, Any]:
        return {
            "app": "Avatar V2",
            "version": "0.3.0",
            "platform": platform.platform(),
            "settings": self._settings(),
            "app_home": str(APP_HOME),
            "runs_dir": str(RUNS_DIR),
            "engines": [
                {"id": "h3_ref2va", "name": "MiniMax H3 · Ref2VA", "tag": "MULTI-REFERENCE"},
                {"id": "h3_fl2va", "name": "MiniMax H3 · FL2VA", "tag": "FIRST / LAST"},
                {"id": "wan22", "name": "Wan 2.2", "tag": "SPECIALIST"},
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
            "yaml": ("YAML (*.yaml;*.yml)",),
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
        if not result:
            return ""
        return str(Path(result[0]))

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
            report["comfyui"] = True
            report["system_stats"] = stats
            caps = provider.capabilities()
            report.update(caps)
            report["ok"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    def _models_from_payload(self, payload: dict[str, Any]) -> tuple[AvatarManifest, ShotSpec]:
        avatar_payload = payload.get("avatar", {})
        subject_kind = avatar_payload.get("subject_kind", "synthetic")
        avatar = AvatarManifest.model_validate(
            {
                "avatar_id": avatar_payload.get("avatar_id") or "desktop-avatar",
                "display_name": avatar_payload.get("display_name") or "Desktop Avatar",
                "subjects": [
                    {
                        "id": avatar_payload.get("avatar_id") or "desktop-avatar",
                        "kind": subject_kind,
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
        shot = ShotSpec.model_validate(
            {
                "shot_id": shot_payload.get("shot_id") or "desktop-shot",
                "content_class": shot_payload.get("content_class", "general"),
                "user_prompt": shot_payload.get("user_prompt") or "Natural photorealistic human motion.",
                "duration_s": float(shot_payload.get("duration_s", 6)),
                "fps": int(shot_payload.get("fps", 24)),
                "width": int(shot_payload.get("width", 720)),
                "height": int(shot_payload.get("height", 1280)),
                "seed": int(shot_payload.get("seed", 41001)),
                "engine_preference": shot_payload.get("engine_preference", "h3_ref2va"),
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
        if key and settings.get(key):
            return str(settings[key])
        return ""

    def compose(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            decision = evaluate_policy(avatar, shot)
            if not decision.allowed:
                return {"ok": False, "error": decision.reason, "policy": decision.reason}
            job = build_job(avatar, shot)
            run_dir = RUNS_DIR / _run_id("plan")
            run_dir.mkdir(parents=True, exist_ok=True)
            job_path = run_dir / "job.json"
            job_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            return {
                "ok": True,
                "job_id": job.job_id,
                "job_path": str(job_path),
                "run_dir": str(run_dir),
                "engine": job.selected_engine,
                "prompt": job.prompt,
                "asset_map": job.asset_map,
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def dry_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            job = build_job(avatar, shot)
            workflow = self._workflow_for(payload, job.selected_engine)
            if not workflow:
                return {"ok": False, "error": "Select an API-format ComfyUI workflow first."}
            provider = ComfyUIProvider(self._provider_config(payload))
            resolved = provider.resolve_workflow(workflow, job, stage_assets=False)
            run_dir = RUNS_DIR / _run_id("dryrun")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
            resolved_path = run_dir / "resolved_workflow.json"
            _write_json(resolved_path, resolved)
            return {
                "ok": True,
                "engine": job.selected_engine,
                "prompt": job.prompt,
                "resolved_workflow": str(resolved_path),
                "run_dir": str(run_dir),
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            avatar, shot = self._models_from_payload(payload)
            job = build_job(avatar, shot)
            workflow = self._workflow_for(payload, job.selected_engine)
            if not workflow:
                return {"ok": False, "error": "Select an API-format ComfyUI workflow first."}
            config = self._provider_config(payload)
            provider = ComfyUIProvider(config)
            result = provider.render(workflow, job)
            run_dir = RUNS_DIR / _run_id("render")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
            _write_json(run_dir / "result.json", {"prompt_id": result["prompt_id"], "outputs": result["outputs"]})

            outputs: list[str] = []
            for output in result["outputs"]:
                path = Path(output)
                if not path.is_absolute() and config.output_dir:
                    path = Path(config.output_dir) / path
                outputs.append(str(path))
            return {
                "ok": True,
                "engine": job.selected_engine,
                "prompt_id": result["prompt_id"],
                "outputs": outputs,
                "run_dir": str(run_dir),
            }
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


if __name__ == "__main__":
    main()
