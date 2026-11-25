#!/usr/bin/env python3
"""Read-only compatibility probe for ComfyUI attention backends (PyTorch SDPA, comfy-kitchen, SageAttention).

Never installs packages or modifies the production ComfyUI environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _nvcc_version() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        for candidate in sorted(Path("/usr/local").glob("cuda*/bin/nvcc"), reverse=True):
            if candidate.is_file():
                nvcc = str(candidate)
                break
    if not nvcc:
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release\s+(\d+\.\d+)", out)
    return f"{match.group(1)} ({nvcc})" if match else nvcc


def probe_env(python_bin: Path) -> dict[str, Any]:
    snippet = r"""
import importlib.metadata, json, sys, torch
out = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "packages": {},
    "comfy_kitchen": {},
}
for pkg in ("triton", "sageattention", "flash-attn", "xformers", "comfy-kitchen", "comfy-aimdo", "gguf"):
    try:
        out["packages"][pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        out["packages"][pkg] = None
try:
    import comfy_kitchen
    out["comfy_kitchen"] = {
        "int8_attention_is_available": bool(comfy_kitchen.int8_attention_is_available()),
        "sol_attn_is_available": bool(getattr(comfy_kitchen, "sol_attn_is_available", lambda: False)()),
    }
except Exception as exc:
    out["comfy_kitchen"] = {"error": str(exc)}
print(json.dumps(out))
"""
    res = subprocess.run([str(python_bin), "-c", snippet], capture_output=True, text=True, timeout=30, check=True)
    data = json.loads(res.stdout)
    nvcc = _nvcc_version()
    data["host_nvcc"] = nvcc
    torch_cuda = str(data.get("cuda_version") or "")
    nvcc_ver = (nvcc or "").split(" ", 1)[0]
    data["nvcc_matches_torch_cuda"] = bool(nvcc_ver and torch_cuda and nvcc_ver == torch_cuda)
    data["recommendations"] = {
        "pytorch_sdpa": "active_baseline",
        "comfy_kitchen_int8": (
            "ready_installed"
            if data.get("comfy_kitchen", {}).get("int8_attention_is_available")
            else "unavailable"
        ),
        "sageattention_1_triton": (
            "compatible_isolated_test_candidate (pure-python + triton wheel, no nvcc needed)"
            if data["packages"].get("triton")
            else "requires_triton"
        ),
        "sageattention_2_cuda": (
            "ready_to_build"
            if data["nvcc_matches_torch_cuda"]
            else f"blocked_source_build (host nvcc={nvcc_ver or 'missing'} != torch cuda={torch_cuda}; needs prebuilt cu130 wheel or CUDA {torch_cuda} toolkit)"
        ),
    }
    return data


def main() -> int:
    default_py = Path.home() / "AvatarForge" / "venv" / "bin" / "python"
    python_bin = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_py
    if not python_bin.is_file():
        print(json.dumps({"ok": False, "error": f"Python environment not found: {python_bin}"}))
        return 1
    report = probe_env(python_bin)
    print(json.dumps({"ok": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
