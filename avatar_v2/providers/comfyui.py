from __future__ import annotations

import json
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from ..models import ProviderConfig, RenderJob


def replace_placeholders(value: Any, mapping: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: replace_placeholders(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_placeholders(v, mapping) for v in value]
    if not isinstance(value, str):
        return value

    for key, replacement in mapping.items():
        token = "${" + key + "}"
        if value == token:
            return replacement
        if token in value and replacement is not None:
            value = value.replace(token, str(replacement))
    return value


def _is_asset_key(key: str) -> bool:
    fixed = {"INIT_IMAGE", "LAST_FRAME", "MOTION_VIDEO", "SCENE_IMAGE", "AUDIO"}
    dynamic_prefixes = (
        "IDENTITY_REF_",
        "BODY_REF_",
        "HAIR_REF_",
        "WARDROBE_REF_",
        "H3_PICTURE_",
        "H3_VIDEO_",
        "H3_AUDIO_",
    )
    return key in fixed or key.startswith(dynamic_prefixes)


def _combo_options(spec: Any) -> list[str]:
    if not isinstance(spec, list) or not spec:
        return []
    head = spec[0]
    if isinstance(head, list):
        return [str(item) for item in head]
    if len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options", [])
        return [str(item) for item in options]
    return []


class ComfyUIProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()

    def object_info(self, node_name: str | None = None) -> dict[str, Any]:
        suffix = f"/{node_name}" if node_name else ""
        with httpx.Client(timeout=20) as client:
            response = client.get(f"{self.base_url}/object_info{suffix}")
            response.raise_for_status()
            return response.json()

    def list_models(self) -> dict[str, list[str]]:
        result = {"diffusion_models": [], "text_encoders": [], "vae": []}
        queries = (
            ("UNETLoader", "unet_name", "diffusion_models"),
            ("CLIPLoader", "clip_name", "text_encoders"),
            ("VAELoader", "vae_name", "vae"),
        )
        for node_name, input_name, category in queries:
            try:
                info = self.object_info(node_name)
                spec = info[node_name]["input"]["required"][input_name]
                result[category] = _combo_options(spec)
            except (KeyError, httpx.HTTPError):
                continue
        return result

    def capabilities(self) -> dict[str, Any]:
        info = self.object_info()
        h3_nodes = sorted(
            name for name in info.keys()
            if "minimaxh3" in name.lower() or "minimax h3" in name.lower()
        )
        models = self.list_models()
        h3_diffusion = [name for name in models["diffusion_models"] if "minimax_h3" in name.lower()]
        h3_encoders = [name for name in models["text_encoders"] if "minimax" in name.lower() or "qwen3vl" in name.lower()]
        h3_vaes = [name for name in models["vae"] if "minimax_h3" in name.lower()]
        return {
            "h3_available": bool(h3_nodes),
            "h3_nodes": h3_nodes[:50],
            "node_count": len(info),
            "models": models,
            "h3_models": {
                "diffusion_models": h3_diffusion,
                "text_encoders": h3_encoders,
                "vae": h3_vaes,
            },
        }

    def upload_file(self, path: str | Path, *, subfolder: str = "avatar_v2") -> str:
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(src)
        mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
        with src.open("rb") as handle, httpx.Client(timeout=max(60, self.config.timeout_s)) as client:
            response = client.post(
                f"{self.base_url}/upload/image",
                data={"type": "input", "subfolder": subfolder, "overwrite": "true"},
                files={"image": (src.name, handle, mime)},
            )
            response.raise_for_status()
            data = response.json()
        name = data.get("name") or src.name
        returned_subfolder = data.get("subfolder") or subfolder
        return str(Path(returned_subfolder) / name) if returned_subfolder else str(name)

    def _copy_to_input(self, src: Path, job_id: str, key: str) -> str:
        if not self.config.input_dir:
            raise RuntimeError("input_dir not configured")
        input_dir = Path(self.config.input_dir).expanduser().resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"avatar_v2_{job_id}_{key.lower()}_{src.name}"
        dest = input_dir / dest_name
        if src != dest:
            shutil.copy2(src, dest)
        return dest_name

    def _stage_assets(self, job: RenderJob) -> dict[str, Any]:
        mapping = dict(job.asset_map)
        staged_by_path: dict[str, str] = {}
        for key, raw in list(mapping.items()):
            if not _is_asset_key(key) or not raw:
                continue
            src = Path(str(raw)).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(f"{key} asset does not exist: {src}")
            path_key = str(src)
            if path_key in staged_by_path:
                mapping[key] = staged_by_path[path_key]
                continue

            if self.config.input_dir:
                staged = self._copy_to_input(src, job.job_id, key)
            else:
                staged = self.upload_file(src)
            staged_by_path[path_key] = staged
            mapping[key] = staged
        return mapping

    def resolve_workflow(
        self,
        workflow_path: str | Path,
        job: RenderJob,
        *,
        stage_assets: bool = True,
    ) -> dict[str, Any]:
        workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        mapping = self._stage_assets(job) if stage_assets else dict(job.asset_map)
        return replace_placeholders(workflow, mapping)

    def queue(self, workflow: dict[str, Any]) -> str:
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    def interrupt(self) -> None:
        with httpx.Client(timeout=10) as client:
            response = client.post(f"{self.base_url}/interrupt", json={})
            response.raise_for_status()

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask ComfyUI to unload models and release allocator/cache memory when idle."""
        with httpx.Client(timeout=10) as client:
            response = client.post(
                f"{self.base_url}/free",
                json={"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
            )
            response.raise_for_status()

    def wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_s
        with httpx.Client(timeout=30) as client:
            while time.monotonic() < deadline:
                response = client.get(f"{self.base_url}/history/{prompt_id}")
                response.raise_for_status()
                data = response.json()
                if prompt_id in data:
                    item = data[prompt_id]
                    status = item.get("status", {})
                    if status.get("completed") is True or item.get("outputs"):
                        return item
                time.sleep(1.5)
        raise TimeoutError(f"ComfyUI job {prompt_id} exceeded {self.config.timeout_s}s")

    @staticmethod
    def output_files(history_item: dict[str, Any]) -> list[str]:
        found: list[str] = []
        for node_output in history_item.get("outputs", {}).values():
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_output.get(key, []) or []:
                    filename = item.get("filename") if isinstance(item, dict) else None
                    subfolder = item.get("subfolder", "") if isinstance(item, dict) else ""
                    if filename:
                        found.append(str(Path(subfolder) / filename))
        return found

    def render(self, workflow_path: str | Path, job: RenderJob) -> dict[str, Any]:
        workflow = self.resolve_workflow(workflow_path, job, stage_assets=True)
        prompt_id = self.queue(workflow)
        history = self.wait(prompt_id)
        outputs = self.output_files(history)
        return {"prompt_id": prompt_id, "outputs": outputs, "history": history}
