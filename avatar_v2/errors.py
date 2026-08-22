from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClassifiedError:
    category: str
    message: str
    detail: str


_CUDA_OOM = re.compile(r"out of memory|cuda oom", re.IGNORECASE)
_CUDA_999 = re.compile(r"cuda error 999|unknown error|cuInit|cudaGetDeviceCount", re.IGNORECASE)
_MISSING_NODE = re.compile(r"node .* not found|invalid prompt|no node|node id .* does not exist", re.IGNORECASE)
_MISSING_MODEL = re.compile(r"model.*not found|no such file|does not exist|failed finding central directory|PytorchStreamReader|safetensors", re.IGNORECASE)
_DISK_FULL = re.compile(r"no space left on device", re.IGNORECASE)
_PERMISSION = re.compile(r"permission denied|errno 13", re.IGNORECASE)
_H3_DIMS = re.compile(r"dimension|multiple of 32|canvas", re.IGNORECASE)
_H3_FRAMES = re.compile(r"frame count|17k|frame grid|valid frame", re.IGNORECASE)
_SWAP_OOM = re.compile(r"cannot allocate memory|killed", re.IGNORECASE)


def classify_exception(exc: BaseException) -> ClassifiedError:
    name = type(exc).__name__
    text = f"{name}: {exc}"

    if isinstance(exc, TimeoutError):
        return ClassifiedError("timeout", "The render exceeded its configured timeout.", text)
    if isinstance(exc, ConnectionError) or "connect" in name.lower():
        return ClassifiedError(
            "comfyui_unavailable",
            "ComfyUI is not reachable at the configured URL. Start it or check Settings.",
            text,
        )
    if _CUDA_OOM.search(text):
        return ClassifiedError(
            "cuda_oom",
            "CUDA out of memory. Use a smaller canvas, fewer frames, or a low-memory runtime profile.",
            text,
        )
    if _CUDA_999.search(text):
        return ClassifiedError(
            "cuda_error_999",
            "CUDA error 999 / unknown error detected. Consult docs/CUDA_ERROR_999_QUICK_REFERENCE.txt.",
            text,
        )
    if _MISSING_NODE.search(text):
        return ClassifiedError(
            "missing_comfyui_node",
            "The workflow references a ComfyUI node that is not installed in this ComfyUI build.",
            text,
        )
    if _MISSING_MODEL.search(text):
        return ClassifiedError(
            "missing_model",
            "A required model file is missing or corrupt on disk.",
            text,
        )
    if _DISK_FULL.search(text):
        return ClassifiedError("disk_full", "The output disk is full. Free space before rendering.", text)
    if _PERMISSION.search(text):
        return ClassifiedError("permission_error", "A file or folder permission was denied.", text)
    if _H3_FRAMES.search(text):
        return ClassifiedError(
            "invalid_h3_frames",
            "The requested H3 frame count is not valid; the app snaps to the 17k+5 grid.",
            text,
        )
    if _H3_DIMS.search(text):
        return ClassifiedError(
            "invalid_h3_dims",
            "The requested H3 canvas is invalid; the app snaps dimensions to multiples of 32.",
            text,
        )
    if _SWAP_OOM.search(text):
        return ClassifiedError(
            "system_oom",
            "System memory or swap was exhausted during the render.",
            text,
        )
    if isinstance(exc, PermissionError):
        return ClassifiedError("permission_error", "A file or folder permission was denied.", text)
    if isinstance(exc, FileNotFoundError):
        return ClassifiedError("missing_file", "A referenced file does not exist.", text)
    return ClassifiedError("render_failed", "The render failed. See the retained detail below.", text)


def enrich_item_error(item: Any) -> None:
    """Attach a human-readable diagnosis to a queue item while keeping the raw error."""
    if not getattr(item, "error", ""):
        return
    classified = classify_exception(RuntimeError(item.error))
    item.error_category = classified.category
    item.error_message = classified.message
