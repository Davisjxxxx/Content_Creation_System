from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from ..models import ProviderConfig, RenderJob
from ..workflows_builder import BUILTIN_BUILDERS, with_tiled_vae, with_wan_end_frame


class ComfyUIExecutionError(RuntimeError):
    """A ComfyUI workflow executed but one of its nodes failed."""

    def __init__(
        self,
        message: str,
        *,
        partial_outputs: list[str] | None = None,
        history_item: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_outputs: list[str] = list(partial_outputs or [])
        self.history_item: dict[str, Any] | None = history_item


class ComfyUICancelledError(RuntimeError):
    """A ComfyUI job was cancelled or interrupted before completion."""

    def __init__(self, message: str = "ComfyUI job cancelled", *, partial_outputs: list[str] | None = None) -> None:
        super().__init__(message)
        self.partial_outputs: list[str] = list(partial_outputs or [])


class ComfyUINotReadyError(RuntimeError):
    """ComfyUI is unreachable at the configured base URL."""


class RendererOwnershipError(PermissionError):
    """Raised when a global renderer operation (/free or global /interrupt) lacks proven ownership (F01 / RC-01)."""


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


_DYNAMIC_ASSET = re.compile(r"^(REF_IMAGE|IDENTITY_REF|BODY_REF|HAIR_REF|WARDROBE_REF|H3_PICTURE|H3_VIDEO|H3_AUDIO)_\d+$")


def _is_asset_key(key: str) -> bool:
    fixed = {
        "INIT_IMAGE",
        "LAST_FRAME",
        "MOTION_VIDEO",
        "SCENE_IMAGE",
        "AUDIO",
        "SOURCE_VIDEO",
        "SOURCE_IMAGE",
        "MASK_IMAGE",
    }
    return key in fixed or bool(_DYNAMIC_ASSET.match(key))


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
    def __init__(
        self,
        config: ProviderConfig,
        *,
        ownership_authorizer: Callable[[str, str], tuple[bool, str, str]] | None = None,
    ):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._ownership_authorizer = ownership_authorizer

    def _authorize_global_target(self, operation: str) -> str:
        """Authorize a global renderer operation and return the concrete pinned base URL (F01 / RC-01)."""
        if self._ownership_authorizer is None:
            return self.base_url
        ok, pinned_url, reason = self._ownership_authorizer(self.base_url, operation)
        if not ok or not str(pinned_url or "").strip():
            raise RendererOwnershipError(
                f"Refusing {operation} on unverified/unowned renderer {self.base_url} ({reason})."
            )
        return str(pinned_url).rstrip("/")

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
        from ..ownership import copy_file_exclusive

        if not self.config.input_dir:
            raise RuntimeError("input_dir not configured")
        input_dir = Path(self.config.input_dir).expanduser().resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"avatar_v2_{job_id}_{key.lower()}_{src.name}"
        dest = input_dir / dest_name
        if src == dest:
            return dest_name
        created = copy_file_exclusive(src, input_dir, dest_name)
        return created.name

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
                # Separate each logical reference on the ComfyUI server. Distinct
                # local files frequently share names such as body.png; a shared
                # overwrite subfolder would silently replace the earlier image.
                safe_job_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", job.job_id)
                safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key.lower())
                staged = self.upload_file(src, subfolder=f"avatar_v2/{safe_job_id}/{safe_key}")
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
        raw = str(workflow_path)
        if raw.startswith("builtin:"):
            engine = raw.split(":", 1)[1]
            builder = BUILTIN_BUILDERS.get(engine)
            if builder is None:
                raise RuntimeError(f"No builtin workflow builder for engine '{engine}'.")
            mapping = self._stage_assets(job) if stage_assets else dict(job.asset_map)
            return builder(mapping)
        workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        mapping = self._stage_assets(job) if stage_assets else dict(job.asset_map)
        resolved = replace_placeholders(workflow, mapping)
        if mapping.get("LAST_FRAME"):
            resolved = with_wan_end_frame(resolved, str(mapping["LAST_FRAME"]))
        if mapping.get("TILED_VAE"):
            resolved = with_tiled_vae(resolved)
        return resolved

    def queue(self, workflow: dict[str, Any]) -> str:
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{self.base_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    def interrupt(self, prompt_id: str | None = None, *, allow_global: bool = False) -> bool:
        """Cancel a specific ComfyUI prompt without disrupting unrelated work on a shared renderer.

        - If ``prompt_id`` is provided, removes it from ComfyUI's pending queue first; only sends
          a prompt-scoped ``POST /interrupt`` if ``prompt_id`` is currently running.
        - Refuses unscoped global ``POST /interrupt`` unless ``allow_global=True`` AND ownership is proven (F01).
        - Disables HTTP redirects on all ownership-sensitive cancellation requests.
        """
        pid = str(prompt_id or "").strip()
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            if pid:
                try:
                    q_resp = client.get(f"{self.base_url}/queue")
                    q_resp.raise_for_status()
                    q_data = q_resp.json()
                except Exception:
                    q_data = {}
                pending = q_data.get("queue_pending") or []
                running = q_data.get("queue_running") or []
                pending_ids = {str(entry[1]) for entry in pending if isinstance(entry, list) and len(entry) >= 2}
                running_ids = {str(entry[1]) for entry in running if isinstance(entry, list) and len(entry) >= 2}
                if pid in pending_ids:
                    resp = client.post(f"{self.base_url}/queue", json={"delete": [pid]})
                    if 300 <= getattr(resp, "status_code", 200) < 400:
                        raise RendererOwnershipError(f"Refusing HTTP redirect ({resp.status_code}) on /queue delete")
                    resp.raise_for_status()
                    return True
                if pid in running_ids:
                    resp = client.post(f"{self.base_url}/interrupt", json={"prompt_id": pid})
                    if 300 <= getattr(resp, "status_code", 200) < 400:
                        raise RendererOwnershipError(f"Refusing HTTP redirect ({resp.status_code}) on /interrupt")
                    resp.raise_for_status()
                    return True
                if not allow_global:
                    return False
            if not allow_global:
                return False
            target_url = self._authorize_global_target("global_interrupt")
            response = client.post(f"{target_url}/interrupt", json={})
            if 300 <= getattr(response, "status_code", 200) < 400:
                raise RendererOwnershipError(f"Refusing HTTP redirect ({response.status_code}) on global /interrupt")
            response.raise_for_status()
            return True

    def queue_status(self) -> dict[str, Any]:
        """Return queue occupancy so cache cleanup can stay idle-only."""
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            response = client.get(f"{self.base_url}/queue")
            if 300 <= getattr(response, "status_code", 200) < 400:
                raise RendererOwnershipError(f"Refusing HTTP redirect ({response.status_code}) on /queue")
            response.raise_for_status()
            data = response.json()
        running = len(data.get("queue_running") or [])
        pending = len(data.get("queue_pending") or [])
        return {"running": running, "pending": pending, "busy": bool(running or pending)}

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask an owned ComfyUI endpoint to unload models and release allocator/cache memory (F01 / RC-01).

        Enforces centralized ownership authorization, pins the concrete verified IP-literal endpoint,
        and disables HTTP redirects so transport behavior cannot redirect cleanup to an unrelated service.
        """
        target_url = self._authorize_global_target("free_memory")
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            response = client.post(
                f"{target_url}/free",
                json={"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
            )
            if 300 <= getattr(response, "status_code", 200) < 400:
                raise RendererOwnershipError(f"Refusing HTTP redirect ({response.status_code}) on global /free")
            response.raise_for_status()

    def get_history_item(self, prompt_id: str) -> dict[str, Any] | None:
        """Fetch a single ComfyUI history record if present, or None if not yet recorded."""
        with httpx.Client(timeout=15) as client:
            response = client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            data = response.json()
        item = data.get(prompt_id) if isinstance(data, dict) else None
        return item if isinstance(item, dict) else None

    @staticmethod
    def _extract_execution_error(item: dict[str, Any]) -> str | None:
        status = item.get("status") or {}
        if not isinstance(status, dict):
            return None
        messages = status.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, list) and len(message) >= 2 and message[0] == "execution_error":
                detail = message[1] if isinstance(message[1], dict) else {}
                return (
                    f"ComfyUI node {detail.get('node_id')} "
                    f"({detail.get('node_type')}) failed: "
                    f"{str(detail.get('exception_message', 'unknown error')).strip()}"
                )
        if status.get("status_str") == "error":
            return "ComfyUI execution failed without a node-level message"
        return None

    @staticmethod
    def _is_interrupted(item: dict[str, Any]) -> bool:
        status = item.get("status") or {}
        if not isinstance(status, dict):
            return False
        if status.get("status_str") == "interrupted":
            return True
        for message in status.get("messages") or []:
            if isinstance(message, list) and message and message[0] == "execution_interrupted":
                return True
        return False

    def wait(self, prompt_id: str, *, cancel_event: threading.Event | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.timeout_s
        with httpx.Client(timeout=30) as client:
            while time.monotonic() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    try:
                        self.interrupt(prompt_id, allow_global=False)
                    except Exception:
                        pass
                    raise ComfyUICancelledError(f"ComfyUI job {prompt_id} was cancelled")
                response = client.get(f"{self.base_url}/history/{prompt_id}")
                response.raise_for_status()
                data = response.json()
                if prompt_id in data:
                    item = data[prompt_id]
                    # RC-16: Always evaluate execution_error / interrupted BEFORE outputs, so partial
                    # artifacts from earlier nodes are NEVER promoted as a completed render!
                    err_msg = self._extract_execution_error(item)
                    if err_msg is not None:
                        raise ComfyUIExecutionError(
                            err_msg,
                            partial_outputs=self.output_files(item),
                            history_item=item,
                        )
                    if self._is_interrupted(item):
                        raise ComfyUICancelledError(
                            f"ComfyUI job {prompt_id} was interrupted",
                            partial_outputs=self.output_files(item),
                        )
                    status = item.get("status") if isinstance(item.get("status"), dict) else {}
                    status_str = str(status.get("status_str") or "").strip().lower()
                    # RC-16: Require explicit terminal completion from ComfyUI; never promote
                    # intermediate/preview outputs while status has no terminal marker.
                    if status.get("completed") is True or status_str == "success":
                        return item
                if cancel_event is not None:
                    if cancel_event.wait(timeout=1.5):
                        try:
                            self.interrupt(prompt_id, allow_global=False)
                        except Exception:
                            pass
                        raise ComfyUICancelledError(f"ComfyUI job {prompt_id} was cancelled")
                else:
                    time.sleep(1.5)
        raise TimeoutError(f"ComfyUI job {prompt_id} exceeded {self.config.timeout_s}s")

    @staticmethod
    def output_files(history_item: dict[str, Any]) -> list[str]:
        found: list[str] = []
        for node_output in (history_item.get("outputs") or {}).values():
            if not isinstance(node_output, dict):
                continue
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_output.get(key, []) or []:
                    filename = item.get("filename") if isinstance(item, dict) else None
                    subfolder = item.get("subfolder", "") if isinstance(item, dict) else ""
                    if not filename:
                        continue
                    rel = Path(str(subfolder)) / str(filename)
                    # RC-16: Refuse absolute paths or parent-directory traversal from renderer output metadata.
                    if rel.is_absolute() or ".." in rel.parts or str(filename).startswith(("/", "\\")):
                        raise ValueError(f"Refusing unsafe ComfyUI output path: {rel!s}")
                    found.append(str(rel))
        return found

    def render(
        self,
        workflow_path: str | Path,
        job: RenderJob,
        *,
        on_resolved: Callable[[dict[str, Any]], None] | None = None,
        on_submitted: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise ComfyUICancelledError("ComfyUI job was cancelled before submission")
        workflow = self.resolve_workflow(workflow_path, job, stage_assets=True)
        if on_resolved is not None:
            on_resolved(workflow)
        if cancel_event is not None and cancel_event.is_set():
            raise ComfyUICancelledError("ComfyUI job was cancelled before submission")
        prompt_id = self.queue(workflow)
        if on_submitted is not None:
            on_submitted(prompt_id)
        if cancel_event is not None:
            history = self.wait(prompt_id, cancel_event=cancel_event)
        else:
            history = self.wait(prompt_id)
        outputs = self.output_files(history)
        return {"prompt_id": prompt_id, "outputs": outputs, "history": history}
