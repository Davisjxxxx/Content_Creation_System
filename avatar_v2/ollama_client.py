"""Local-only Ollama inventory, resource-aware scheduling, and untrusted JSON client (Sprint 6).

Enforces:
- Strictly loopback-only communication (127.0.0.1 / localhost); refuses remote endpoints.
- Integrates with ResourcePlanner so local LLM inference defaults to CPU-only (``num_gpu=0``)
  whenever warm ComfyUI models are resident or GPU generation is active, avoiding VRAM eviction.
- Treats every LLM output as untrusted structured data with strict JSON parsing and clean
  fallback on timeout, unavailability, or malformed responses.
- Never auto-downloads models.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .resource_monitor import ResourceSnapshot

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Preference order for fast, reliable structured-JSON instruction models already installed locally.
PREFERRED_TEXT_MODELS: tuple[str, ...] = (
    "llama3.1:8b",
    "qwen3:8b",
    "qwen3.5:9b",
    "ministral-3:8b",
    "mistral:7b-instruct-q4_K_M",
    "llama3.2:3b",
)

_THINKING_FAMILIES = frozenset({"qwen3", "qwen35", "deepseek-r1"})
_VISION_HINTS = ("vl", "vision", "llava", "minicpm-v")
_EMBED_HINTS = ("embed", "nomic-embed", "bge")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass(frozen=True)
class LocalLLMInfo:
    name: str
    size_bytes: int
    size_gb: float
    family: str
    parameter_size: str
    quantization: str
    roles: tuple[str, ...] = field(default_factory=tuple)

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMExecutionSchedule:
    mode: Literal["cpu_only", "sequential_gpu", "disabled"]
    num_gpu: int
    keep_alive: str
    reason: str


@dataclass(frozen=True)
class LLMCallResult:
    ok: bool
    model: str
    mode: str
    latency_ms: float
    data: dict[str, Any] | None = None
    raw_text: str = ""
    error: str = ""


def _ensure_loopback_url(base_url: str) -> str:
    url = str(base_url or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _LOOPBACK_HOSTS:
        raise ValueError(f"Offline intelligence only permits loopback Ollama URLs; refused '{url}'.")
    return url


def _classify_roles(name: str, family: str) -> tuple[str, ...]:
    low = name.lower()
    fam = family.lower()
    if any(tok in low for tok in _EMBED_HINTS):
        return ("embedding",)
    if any(tok in low for tok in _VISION_HINTS) or "vl" in fam:
        return ("vision_qa", "text_chat")
    if "coder" in low:
        return ("coding", "structured_json")
    if "r1" in low:
        return ("reasoning",)
    if "3b" in low or "2b" in low:
        return ("structured_json", "lightweight_cpu")
    return ("structured_json", "prompt_intelligence")


def inspect_local_ollama_models(base_url: str = DEFAULT_OLLAMA_URL, *, timeout_s: float = 3.0) -> list[LocalLLMInfo]:
    """Query locally installed Ollama models without loading or downloading anything."""
    url = _ensure_loopback_url(base_url)
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{url}/api/tags")
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        return []

    out: list[LocalLLMInfo] = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "").strip()
        if not name:
            continue
        size_bytes = int(entry.get("size") or 0)
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        family = str(details.get("family") or "")
        param_size = str(details.get("parameter_size") or "")
        quant = str(details.get("quantization_level") or "")
        out.append(
            LocalLLMInfo(
                name=name,
                size_bytes=size_bytes,
                size_gb=round(size_bytes / (1024**3), 2),
                family=family,
                parameter_size=param_size,
                quantization=quant,
                roles=_classify_roles(name, family),
            )
        )
    return out


def select_intelligence_model(
    available: list[LocalLLMInfo],
    *,
    preferred: str | None = None,
    lightweight_only: bool = False,
) -> LocalLLMInfo | None:
    """Pick the best already-installed local text model for structured intent interpretation."""
    by_name = {m.name: m for m in available}
    if preferred and preferred in by_name:
        return by_name[preferred]
    order = ("llama3.2:3b", *PREFERRED_TEXT_MODELS) if lightweight_only else PREFERRED_TEXT_MODELS
    for candidate in order:
        if candidate in by_name:
            return by_name[candidate]
    for model in available:
        if "structured_json" in model.roles or "prompt_intelligence" in model.roles:
            return model
    return None


def choose_llm_execution_schedule(
    *,
    comfy_models_warm: bool = False,
    active_generation_running: bool = False,
    snapshot: ResourceSnapshot | None = None,
    allow_sequential_gpu: bool = False,
) -> LLMExecutionSchedule:
    """Coordinate local LLM placement with ResourcePlanner to avoid evicting warm ComfyUI models."""
    if snapshot is not None:
        if snapshot.ram_severity == "emergency":
            return LLMExecutionSchedule(
                mode="disabled",
                num_gpu=0,
                keep_alive="0s",
                reason="Disabled local LLM due to emergency physical RAM pressure; using deterministic fallback.",
            )
        if snapshot.temp_severity == "emergency":
            return LLMExecutionSchedule(
                mode="cpu_only",
                num_gpu=0,
                keep_alive="0s",
                reason="GPU in thermal cooldown; scheduling local LLM on CPU only.",
            )

    if active_generation_running or comfy_models_warm:
        return LLMExecutionSchedule(
            mode="cpu_only",
            num_gpu=0,
            keep_alive="0s",
            reason="Warm/active ComfyUI models present: running local LLM on CPU (num_gpu=0) to prevent VRAM eviction.",
        )

    if allow_sequential_gpu:
        gpu_free_mb = min((g.free_mb for g in snapshot.gpus), default=6500) if snapshot else 6500
        if gpu_free_mb >= 5800:
            return LLMExecutionSchedule(
                mode="sequential_gpu",
                num_gpu=-1,
                keep_alive="0s",
                reason="GPU cold with >=5.8 GB free VRAM: using sequential GPU LLM with immediate keep_alive=0s unload.",
            )

    return LLMExecutionSchedule(
        mode="cpu_only",
        num_gpu=0,
        keep_alive="0s",
        reason="Default zero-contention CPU-only local LLM scheduling (keep_alive=0s).",
    )


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Safely extract a single JSON object from raw LLM text; returns None if malformed."""
    text = str(raw_text or "").strip()
    if not text:
        return None
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    # Try direct parse first, then outermost {...} slice.
    for candidate in (text, _slice_outer_braces(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _slice_outer_braces(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return ""


class LocalOllamaClient:
    """Lightweight loopback-only Ollama JSON client with strict resource & timeout controls."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, *, timeout_s: float = 25.0) -> None:
        self.base_url = _ensure_loopback_url(base_url)
        self.timeout_s = max(2.0, float(timeout_s))

    def list_models(self) -> list[LocalLLMInfo]:
        return inspect_local_ollama_models(self.base_url, timeout_s=min(3.0, self.timeout_s))

    def complete_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schedule: LLMExecutionSchedule | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        sched = schedule or choose_llm_execution_schedule()
        if sched.mode == "disabled":
            return LLMCallResult(ok=False, model=model, mode=sched.mode, latency_ms=0.0, error=sched.reason)

        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": "json",
            "keep_alive": sched.keep_alive,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "num_gpu": int(sched.num_gpu),
            },
        }
        low_model = model.lower()
        if any(low_model.startswith(fam) for fam in _THINKING_FAMILIES):
            body["think"] = False

        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=body)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            return LLMCallResult(
                ok=False,
                model=model,
                mode=sched.mode,
                latency_ms=round((time.monotonic() - t0) * 1000.0, 1),
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
        content = str(((payload.get("message") or {}).get("content")) or "")
        parsed = extract_json_object(content)
        if parsed is None:
            return LLMCallResult(
                ok=False,
                model=model,
                mode=sched.mode,
                latency_ms=latency_ms,
                raw_text=content[:600],
                error="Local LLM returned non-JSON or malformed JSON output.",
            )
        return LLMCallResult(
            ok=True,
            model=model,
            mode=sched.mode,
            latency_ms=latency_ms,
            data=parsed,
            raw_text=content[:1200],
        )
