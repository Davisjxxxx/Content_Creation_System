from __future__ import annotations

import json
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


class ComfyUIProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()

    def object_info(self) -> dict[str, Any]:
        with httpx.Client(timeout=20) as client:
            response = client.get(f"{self.base_url}/object_info")
            response.raise_for_status()
            return response.json()

    def capabilities(self) -> dict[str, Any]:
        info = self.object_info()
        h3_nodes = sorted(
            name for name in info.keys()
            if "minimaxh3" in name.lower() or "minimax h3" in name.lower()
        )
        return {
            "h3_available": bool(h3_nodes),
            "h3_nodes": h3_nodes[:50],
            "node_count": len(info),
        }

    def _stage_assets(self, job: RenderJob) -> dict[str, Any]:
        mapping = dict(job.asset_map)
        if not self.config.input_dir:
            return mapping

        input_dir = Path(self.config.input_dir).expanduser().resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        for key, raw in list(mapping.items()):
            if not _is_asset_key(key) or not raw:
                continue
            src = Path(str(raw)).expanduser().resolve()
            if not src.exists():
                raise FileNotFoundError(f"{key} asset does not exist: {src}")
            dest_name = f"avatar_v2_{job.job_id}_{key.lower()}_{src.name}"
            dest = input_dir / dest_name
            if src != dest:
                shutil.copy2(src, dest)
            mapping[key] = dest_name
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
