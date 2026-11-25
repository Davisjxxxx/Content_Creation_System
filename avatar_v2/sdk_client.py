"""Standalone Native Avatar V2 Python SDK Client (Sprint 4, Section 6).

This module intentionally depends ONLY on the Python standard library so it can be mounted
read-only into an isolated Bubblewrap sandbox at ``/sdk/avatar_v2_sdk.py`` without exposing
host application internals, credentials, or direct ComfyUI HTTP access.

Contract version: ``1.0``
Transport: Authenticated JSON-lines over a host-provided Unix domain socket (works with
``--unshare-net`` network isolation) OR an in-process broker callable for unit tests.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SDK_CONTRACT_VERSION = "1.0"


class SDKError(RuntimeError):
    """Raised when the Avatar V2 host broker rejects or fails an SDK request."""

    def __init__(self, message: str, *, code: str = "sdk_error", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class RunLimitReached(SDKError):
    """Raised when the host-enforced generation limit for the script run is reached."""


class RunStopped(SDKError):
    """Raised when the user or host stopped/cancelled the active script run."""


@dataclass
class GenerationResult:
    """Normalized result returned to scripts by the Avatar V2 SDK."""

    job_id: str
    status: str
    model_id: str = ""
    outputs: list[str] = field(default_factory=list)
    compiled_prompt: str = ""
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Reserved for Sprint 7-8 local quality inspection and bounded corrective generation.
    quality_inspection: dict[str, Any] | None = None
    correction_hint: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationResult":
        return cls(
            job_id=str(data.get("job_id") or ""),
            status=str(data.get("status") or "unknown"),
            model_id=str(data.get("model_id") or ""),
            outputs=list(data.get("outputs") or []),
            compiled_prompt=str(data.get("compiled_prompt") or ""),
            warnings=list(data.get("warnings") or []),
            error=str(data.get("error") or ""),
            metadata=dict(data.get("metadata") or {}),
            quality_inspection=data.get("quality_inspection"),
            correction_hint=data.get("correction_hint"),
        )


class AvatarV2Client:
    """Lightweight Native Avatar V2 Python SDK client.

    All requests are brokered to the host Avatar V2 process where schema validation,
    capability checks, policy enforcement, prompt compilation, ResourcePlanner admission,
    and RenderQueue scheduling occur centrally.
    """

    contract_version: str = SDK_CONTRACT_VERSION

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        token: str | None = None,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._socket_path = socket_path or os.environ.get("AVATAR_V2_BROKER_SOCK") or ""
        self._token = token if token is not None else (os.environ.get("AVATAR_V2_BROKER_TOKEN") or "")
        self._transport = transport
        self._timeout_s = max(1.0, float(timeout_s))
        if self._transport is None and not self._socket_path:
            raise SDKError(
                "AvatarV2Client requires an active Avatar V2 broker connection "
                "(AVATAR_V2_BROKER_SOCK is not set).",
                code="no_broker",
            )

    def _call(self, op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        envelope = {
            "contract_version": SDK_CONTRACT_VERSION,
            "token": self._token,
            "op": op,
            "payload": dict(payload or {}),
        }
        if self._transport is not None:
            response = self._transport(envelope)
        else:
            response = self._call_unix_socket(envelope)

        if not isinstance(response, dict):
            raise SDKError("Malformed response from Avatar V2 broker.", code="bad_response")
        if not response.get("ok"):
            code = str(response.get("code") or "broker_rejected")
            msg = str(response.get("error") or "Request rejected by Avatar V2 host.")
            if code == "run_limit_reached":
                raise RunLimitReached(msg, code=code, details=response)
            if code == "run_stopped":
                raise RunStopped(msg, code=code, details=response)
            raise SDKError(msg, code=code, details=response)
        return response

    def _call_unix_socket(self, envelope: dict[str, Any]) -> dict[str, Any]:
        raw = (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self._timeout_s)
                sock.connect(self._socket_path)
                sock.sendall(raw)
                chunks: list[bytes] = []
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
        except OSError as exc:
            raise SDKError(f"Could not communicate with Avatar V2 broker: {exc}", code="broker_io") from exc
        line = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
        if not line:
            raise SDKError("Empty response from Avatar V2 broker.", code="broker_eof")
        return json.loads(line)

    # ------------------------------------------------------------------ discovery & status

    def capabilities(self) -> dict[str, Any]:
        """Discover approved local models, supported tasks, and current run limits."""
        return self._call("capabilities")

    def run_status(self) -> dict[str, Any]:
        """Inspect host-enforced run counters, pause/stop state, and remaining budget."""
        return self._call("run_status")

    def list_reference_assets(self) -> list[dict[str, Any]]:
        """List reference assets authorized and mounted for this script run."""
        res = self._call("list_references")
        return list(res.get("references") or [])

    # ------------------------------------------------------------------ submission

    def submit_spec(
        self,
        spec: Any,
        *,
        model_id: str | None = None,
        label: str = "",
        wait: bool = False,
        poll_interval_s: float = 0.25,
        timeout_s: float = 1800.0,
    ) -> GenerationResult:
        """Submit a canonical StructuredPromptSpec (model instance or dict) through Avatar V2."""
        if hasattr(spec, "model_dump"):
            spec_dict = spec.model_dump()
        elif isinstance(spec, dict):
            spec_dict = dict(spec)
        else:
            raise SDKError("spec must be a StructuredPromptSpec or dict.", code="invalid_spec")
        res = self._call(
            "submit",
            {
                "spec": spec_dict,
                "model_id": model_id,
                "label": label,
            },
        )
        result = GenerationResult.from_dict(res.get("job") or res)
        if wait and result.job_id:
            return self.wait_for_job(result.job_id, poll_interval_s=poll_interval_s, timeout_s=timeout_s)
        return result

    def submit_prompt(
        self,
        prompt: str,
        *,
        task: str = "image_generate",
        model_id: str | None = None,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        seed: int | None = None,
        references: list[str] | None = None,
        subject_references: list[dict[str, Any]] | None = None,
        attributes: dict[str, str] | None = None,
        label: str = "",
        wait: bool = False,
        poll_interval_s: float = 0.25,
        timeout_s: float = 1800.0,
    ) -> GenerationResult:
        """Submit a natural-language prompt (with optional references/subject_references/attributes) via StructuredPromptSpec."""
        constraints: dict[str, Any] = {"local_only": True}
        if model_id:
            constraints["model_preference"] = str(model_id)
        if width is not None:
            constraints["width"] = int(width)
        if height is not None:
            constraints["height"] = int(height)
        if frames is not None:
            constraints["frames"] = int(frames)
        elif task in {"video_generate", "video_scene"}:
            constraints["frames"] = 81
        if seed is not None:
            constraints["seed"] = int(seed)

        ref_items: list[dict[str, Any]] = [
            {
                "asset_id": f"ref_{idx}",
                "path": str(ref),
                "role": "start_frame" if task in {"video_generate", "video_scene"} else "identity",
            }
            for idx, ref in enumerate(references or [])
            if str(ref or "").strip()
        ]
        for s_idx, sref in enumerate(subject_references or []):
            if not isinstance(sref, dict) or not str(sref.get("path") or "").strip():
                continue
            ref_items.append(
                {
                    "asset_id": str(sref.get("asset_id") or sref.get("ref_id") or f"sref_{s_idx}"),
                    "path": str(sref["path"]),
                    "role": str(sref.get("role") or "front_body"),
                    "subject_id": str(sref.get("subject_id") or "primary"),
                    "view_orientation": str(sref.get("view_orientation") or ""),
                    "source_provenance": str(sref.get("source_provenance") or "script"),
                    "asset_fingerprint": str(sref.get("asset_fingerprint") or ""),
                    "preserve_identity": bool(sref.get("preserve_identity", True)),
                    "preserve_appearance": bool(sref.get("preserve_appearance", True)),
                    "preserve_wardrobe": bool(sref.get("preserve_wardrobe", True)),
                    "locked_attributes": list(sref.get("locked_attributes") or []),
                    "locked": bool(sref.get("locked", False)),
                }
            )

        spec_dict: dict[str, Any] = {
            "original_prompt": str(prompt or ""),
            "intent": {
                "task": task,
                "input_mode": "script",
            },
            "excluded_attributes": [
                part.strip() for part in str(negative_prompt or "").split(",") if part.strip()
            ],
            "reference_assets": ref_items,
            "constraints": constraints,
        }
        for key, val in (attributes or {}).items():
            if val:
                spec_dict[key] = {"value": str(val), "state": "specified", "provenance": "script"}
        return self.submit_spec(
            spec_dict,
            model_id=model_id,
            label=label,
            wait=wait,
            poll_interval_s=poll_interval_s,
            timeout_s=timeout_s,
        )

    def generate_image(
        self,
        prompt: str,
        *,
        model_id: str = "flux2_4b",
        width: int = 896,
        height: int = 1280,
        seed: int | None = None,
        negative_prompt: str = "",
        references: list[str] | None = None,
        subject_references: list[dict[str, Any]] | None = None,
        attributes: dict[str, str] | None = None,
        label: str = "",
        wait: bool = True,
    ) -> GenerationResult:
        return self.submit_prompt(
            prompt,
            task="image_generate",
            model_id=model_id,
            width=width,
            height=height,
            frames=1,
            seed=seed,
            negative_prompt=negative_prompt,
            references=references,
            subject_references=subject_references,
            attributes=attributes,
            label=label,
            wait=wait,
        )

    def edit_image(
        self,
        prompt: str,
        *,
        reference_image: str,
        model_id: str = "flux2_9b",
        seed: int | None = None,
        label: str = "",
        wait: bool = True,
    ) -> GenerationResult:
        return self.submit_prompt(
            prompt,
            task="image_edit",
            model_id=model_id,
            seed=seed,
            references=[reference_image],
            label=label,
            wait=wait,
        )

    def generate_video(
        self,
        prompt: str,
        *,
        start_image: str,
        model_id: str = "wan22_a14b",
        width: int = 480,
        height: int = 832,
        frames: int = 81,
        seed: int | None = None,
        negative_prompt: str = "",
        subject_references: list[dict[str, Any]] | None = None,
        attributes: dict[str, str] | None = None,
        label: str = "",
        wait: bool = True,
    ) -> GenerationResult:
        return self.submit_prompt(
            prompt,
            task="video_generate",
            model_id=model_id,
            width=width,
            height=height,
            frames=frames,
            seed=seed,
            negative_prompt=negative_prompt,
            references=[start_image],
            subject_references=subject_references,
            attributes=attributes,
            label=label,
            wait=wait,
        )

    def submit_batch(
        self,
        specs: Iterable[Any],
        *,
        model_id: str | None = None,
        wait_each: bool = True,
    ) -> list[GenerationResult]:
        """Submit a sequence of specs sequentially while obeying host limits and pause/stop controls."""
        results: list[GenerationResult] = []
        for item in specs:
            if isinstance(item, str):
                res = self.submit_prompt(item, model_id=model_id, wait=wait_each)
            else:
                res = self.submit_spec(item, model_id=model_id, wait=wait_each)
            results.append(res)
        return results

    def continuous(
        self,
        items: Iterable[Any],
        *,
        model_id: str | None = None,
        max_items: int | None = None,
        wait_each: bool = True,
    ) -> Iterator[GenerationResult]:
        """Yield generation results safely until items end, max_items is reached, or host stops/caps the run."""
        produced = 0
        for item in items:
            if max_items is not None and produced >= max_items:
                break
            try:
                if isinstance(item, str):
                    res = self.submit_prompt(item, model_id=model_id, wait=wait_each)
                else:
                    res = self.submit_spec(item, model_id=model_id, wait=wait_each)
            except (RunLimitReached, RunStopped):
                break
            produced += 1
            yield res

    # ------------------------------------------------------------------ job control

    def job_status(self, job_id: str) -> GenerationResult:
        res = self._call("job_status", {"job_id": str(job_id)})
        return GenerationResult.from_dict(res.get("job") or res)

    def job_result(self, job_id: str) -> GenerationResult:
        return self.job_status(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._call("cancel_job", {"job_id": str(job_id)})

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval_s: float = 0.25,
        timeout_s: float = 1800.0,
    ) -> GenerationResult:
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        interval = max(0.05, float(poll_interval_s))
        while True:
            result = self.job_status(job_id)
            if result.status in {"done", "error", "cancelled"}:
                return result
            if time.monotonic() >= deadline:
                raise SDKError(f"Timed out waiting for job '{job_id}'.", code="timeout")
            time.sleep(interval)


def connect(**kwargs: Any) -> AvatarV2Client:
    """Convenience entry point for native Avatar V2 scripts: ``import avatar_v2_sdk; client = avatar_v2_sdk.connect()``."""
    return AvatarV2Client(**kwargs)
