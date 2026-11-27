"""Local-only Ollama inventory, resource-aware scheduling, and untrusted JSON/VLM client (Sprints 6 & 7).

Enforces:
- Strictly loopback-only communication (127.0.0.1 / localhost); refuses remote endpoints.
- Integrates with ResourcePlanner so local LLM/VLM inference defaults to CPU-only (``num_gpu=0``)
  whenever warm ComfyUI models are resident or GPU generation is active, avoiding VRAM eviction.
- Verifies actual GPU VRAM release after ``keep_alive=0`` sequential GPU calls instead of assuming
  asynchronous Ollama unloading completed immediately when HTTP 200 returned.
- Treats every LLM/VLM output and any text visible inside images as untrusted data with strict
  JSON parsing and clean fallback on timeout, unavailability, or malformed responses.
- Never auto-downloads models and never uploads local images or references externally.
"""

from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from .resource_monitor import ResourceSnapshot, recent_gpu_headroom

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Empirically ordered installed local text models (Sprint 6 live benchmark on RTX 4070 + i9-13980HX):
# - qwen3:8b: 100% JSON/intent/safety, fastest 8B (4.2s GPU / 24.2s CPU)
# - llama3.1:8b: 100% JSON/intent/safety (5.8s GPU / 25.1s CPU)
# - llama3.2:3b: 100% JSON/intent/safety, fastest lightweight CPU (13.7s CPU / 0 MB VRAM)
# - qwen3.5:9b deprioritized (37.6s CPU with 83.3% JSON reliability, 15.0s GPU)
PREFERRED_TEXT_MODELS: tuple[str, ...] = (
    "qwen3:8b",
    "llama3.1:8b",
    "llama3.2:3b",
    "ministral-3:8b",
    "mistral:7b-instruct-q4_K_M",
    "qwen3.5:9b",
)

# Empirically ordered installed local vision models (Sprint 7 live benchmark on RTX 4070 + i9-13980HX):
# - qwen3-vl:8b-instruct: 100% JSON, 11.57s sequential GPU, 0% FRR, 16.7% FAR; ONLY installed model
#   to catch real multi-view rear-hair mismatch (mv_01) + synthetic spatial reversal + adh_03 mug count.
# - qwen3-vl:4b-instruct: fast 6.45s sequential GPU tier (33.3% FAR; misses mv_01 rear hair & spatial reversal).
# - qwen3-vl:2b-instruct / q8_0: fast (5.4s GPU / 43.9s CPU) but 50% FAR and frequent omitted criteria.
PREFERRED_VISION_MODELS: tuple[str, ...] = (
    "qwen3-vl:8b-instruct",
    "qwen3-vl:4b-instruct",
    "qwen3-vl:2b-instruct-q8_0",
    "qwen3-vl:2b-instruct",
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
)

LIGHTWEIGHT_VISION_MODELS: tuple[str, ...] = (
    "qwen3-vl:4b-instruct",
    "qwen3-vl:2b-instruct-q8_0",
    "qwen3-vl:2b-instruct",
    "qwen2.5vl:3b",
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
    mode: Literal["cpu_only", "sequential_gpu", "deferred_batch", "disabled"]
    num_gpu: int
    keep_alive: str
    reason: str
    verify_vram_release: bool = False
    prefer_lightweight_model: bool = False


@dataclass(frozen=True)
class LLMCallResult:
    ok: bool
    model: str
    mode: str
    latency_ms: float
    data: dict[str, Any] | None = None
    raw_text: str = ""
    error: str = ""
    vram_release: dict[str, Any] | None = None


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
        if "2b" in low or "3b" in low or "4b" in low:
            return ("vision_qa", "lightweight_vision", "text_chat")
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


def _complete_residency(models: Any) -> bool:
    return isinstance(models, list) and all(
        isinstance(model, dict) and isinstance(model.get("name") or model.get("model"), str)
        and bool((model.get("name") or model.get("model")).strip())
        and type(model.get("size_vram")) is int and model["size_vram"] >= 0
        for model in models
    )


def fetch_ollama_ps_state(
    base_url: str = DEFAULT_OLLAMA_URL,
    *,
    timeout_s: float = 3.0,
) -> tuple[Literal["online", "offline", "unknown"], list[dict[str, Any]], str]:
    """Query loopback ``/api/ps`` distinguishing genuinely offline daemon from unknown/timed-out residency (F10 / RC-14).

    - ``"offline"``: explicit TCP connection refused / ConnectError (Ollama daemon is not listening).
    - ``"unknown"``: timeout, HTTP error, or malformed JSON (service may hold VRAM; callers MUST fail closed!).
    - ``"online"``: valid JSON response with parsed ``models`` list.
    """
    url = _ensure_loopback_url(base_url)
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{url}/api/ps")
            age = getattr(resp, "headers", {}).get("age")
            if age is not None and int(age) != 0:
                return "unknown", [], "Stale cached Ollama residency sample"
            resp.raise_for_status()
            payload = resp.json()
            if time.monotonic() - started > timeout_s:
                return "unknown", [], "Ollama residency sample missed its deadline"
    except (httpx.ConnectError, ConnectionRefusedError) as exc:
        import errno
        chain: BaseException | None = exc
        while chain is not None:
            if isinstance(chain, OSError) and chain.errno == errno.ECONNREFUSED:
                return "offline", [], "Connection refused"
            chain = chain.__cause__ or chain.__context__
        return "unknown", [], str(exc)
    except Exception as exc:
        return "unknown", [], f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return "unknown", [], "Malformed /api/ps payload"
    models = payload.get("models")
    if not _complete_residency(models):
        return "unknown", [], "Incomplete Ollama residency telemetry"
    return "online", models, ""


def _fetch_running_ollama_models_status(
    base_url: str = DEFAULT_OLLAMA_URL,
    *,
    timeout_s: float = 3.0,
) -> tuple[bool, list[dict[str, Any]], str]:
    """Query loopback ``/api/ps`` distinguishing unreachable/error from reachable-empty (RC-14)."""
    state, models, err = fetch_ollama_ps_state(base_url, timeout_s=timeout_s)
    return (state == "online"), models, err


def list_running_ollama_models(base_url: str = DEFAULT_OLLAMA_URL, *, timeout_s: float = 3.0) -> list[dict[str, Any]]:
    """Query currently loaded local Ollama models via loopback ``/api/ps``."""
    ok, models, _err = _fetch_running_ollama_models_status(base_url, timeout_s=timeout_s)
    return models if ok else []


def _query_gpu_vram_mb() -> tuple[int | None, int | None]:
    """Read (used_mb, free_mb) from nvidia-smi without modifying any process."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None, None
        first = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        return int(float(parts[0])), int(float(parts[1]))
    except Exception:
        return None, None


def verify_ollama_vram_released(
    base_url: str = DEFAULT_OLLAMA_URL,
    *,
    model: str | None = None,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.3,
    min_free_vram_mb: int | None = None,
    ps_fetcher: Any = None,
    vram_fetcher: Any = None,
) -> dict[str, Any]:
    """Wait until Ollama has ACTUALLY unloaded GPU weights instead of assuming HTTP 200 means released.

    Fail-closed telemetry (RC-14):
    - Returns explicit ``status`` in ``{"released", "resident", "unknown"}``.
    - If Ollama ``/api/ps`` is unreachable/errors OR required GPU telemetry (``min_free_vram_mb``)
      cannot be read, returns ``released=False, status="unknown"`` rather than falsely claiming released!
    """
    url = _ensure_loopback_url(base_url)
    fetch_vram = vram_fetcher or _query_gpu_vram_mb

    t0 = time.monotonic()
    deadline = t0 + max(0.1, float(timeout_s))
    polls = 0
    last_active: list[dict[str, Any]] = []
    last_used: int | None = None
    last_free: int | None = None
    last_err = ""

    while True:
        polls += 1
        if ps_fetcher is not None:
            try:
                raw_ps = ps_fetcher()
                if raw_ps is None:
                    ps_ok, running, ps_err = False, [], "ps_fetcher returned None"
                else:
                    ps_ok, running, ps_err = True, raw_ps, ""
            except Exception as exc:
                ps_ok, running, ps_err = False, [], f"{type(exc).__name__}: {exc}"
        else:
            ps_ok, running, ps_err = _fetch_running_ollama_models_status(url, timeout_s=2.0)

        if ps_ok and not _complete_residency(running):
            ps_ok, ps_err = False, "Incomplete Ollama residency telemetry"
        used_mb, free_mb = fetch_vram()
        last_used, last_free = used_mb, free_mb

        if not ps_ok:
            last_err = ps_err or "Ollama /api/ps unavailable"
            if time.monotonic() >= deadline:
                return {
                    "released": False,
                    "status": "unknown",
                    "error": last_err,
                    "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                    "polls": polls,
                    "active_vram_models": last_active,
                    "gpu_used_mb": last_used,
                    "gpu_free_mb": last_free,
                }
            time.sleep(max(0.05, float(poll_interval_s)))
            continue

        vram_holders = [
            {
                "name": str(m.get("name") or m.get("model") or ""),
                "size_vram": int(m.get("size_vram") or 0),
            }
            for m in running
            if int(m.get("size_vram") or 0) > 0
            and (model is None or str(m.get("name") or m.get("model") or "") == model)
        ]
        last_active = vram_holders

        ps_clear = len(vram_holders) == 0
        if min_free_vram_mb is not None and free_mb is None:
            # Required VRAM threshold cannot be verified without GPU telemetry -> unknown!
            if time.monotonic() >= deadline:
                return {
                    "released": False,
                    "status": "unknown",
                    "error": "GPU free VRAM telemetry unavailable while min_free_vram_mb threshold is enforced.",
                    "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                    "polls": polls,
                    "active_vram_models": last_active,
                    "gpu_used_mb": last_used,
                    "gpu_free_mb": last_free,
                }
            time.sleep(max(0.05, float(poll_interval_s)))
            continue

        vram_clear = min_free_vram_mb is None or (free_mb is not None and free_mb >= min_free_vram_mb)
        if ps_clear and vram_clear:
            return {
                "released": True,
                "status": "released",
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                "polls": polls,
                "active_vram_models": vram_holders,
                "gpu_used_mb": used_mb,
                "gpu_free_mb": free_mb,
            }
        if time.monotonic() >= deadline:
            return {
                "released": False,
                "status": "resident" if not ps_clear else "unknown",
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                "polls": polls,
                "active_vram_models": last_active,
                "gpu_used_mb": last_used,
                "gpu_free_mb": last_free,
            }
        time.sleep(max(0.05, float(poll_interval_s)))


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


def select_vision_model(
    available: list[LocalLLMInfo],
    *,
    preferred: str | None = None,
    lightweight_only: bool = False,
) -> LocalLLMInfo | None:
    """Pick the simplest effective already-installed local vision model (never auto-downloads)."""
    by_name = {m.name: m for m in available}
    if preferred and preferred in by_name and "vision_qa" in by_name[preferred].roles:
        return by_name[preferred]
    order = LIGHTWEIGHT_VISION_MODELS if lightweight_only else PREFERRED_VISION_MODELS
    for candidate in order:
        if candidate in by_name:
            return by_name[candidate]
    for model in available:
        if "vision_qa" in model.roles:
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
                prefer_lightweight_model=True,
            )

    if active_generation_running or comfy_models_warm:
        return LLMExecutionSchedule(
            mode="cpu_only",
            num_gpu=0,
            keep_alive="0s",
            reason="Warm/active ComfyUI models present: running local LLM on CPU (num_gpu=0) to prevent VRAM eviction.",
            prefer_lightweight_model=True,
        )

    if allow_sequential_gpu:
        gpu_free_mb = recent_gpu_headroom(snapshot)
        if gpu_free_mb is not None and gpu_free_mb >= 5800:
            return LLMExecutionSchedule(
                mode="sequential_gpu",
                num_gpu=-1,
                keep_alive="0s",
                reason="GPU cold with >=5.8 GB free VRAM: using sequential GPU LLM with verified keep_alive=0s unload.",
                verify_vram_release=True,
            )

    return LLMExecutionSchedule(
        mode="cpu_only",
        num_gpu=0,
        keep_alive="0s",
        reason="Default zero-contention CPU-only local LLM scheduling (keep_alive=0s).",
    )


def choose_vision_execution_schedule(
    *,
    comfy_models_warm: bool = False,
    active_generation_running: bool = False,
    upcoming_same_model: bool = False,
    snapshot: ResourceSnapshot | None = None,
    allow_sequential_gpu: bool = False,
    min_gpu_free_mb: int = 4200,
    batch_queue_depth: int = 0,
) -> LLMExecutionSchedule:
    """Choose resource-aware local VLM schedule avoiding warm FLUX/Wan VRAM eviction."""
    if snapshot is not None:
        if snapshot.ram_severity == "emergency":
            return LLMExecutionSchedule(
                mode="disabled",
                num_gpu=0,
                keep_alive="0s",
                reason="Emergency physical RAM pressure: deterministic visual checks only (VLM disabled).",
            )
        if snapshot.ram_severity == "intervention" and (active_generation_running or batch_queue_depth > 1):
            return LLMExecutionSchedule(
                mode="deferred_batch",
                num_gpu=0,
                keep_alive="0s",
                reason="Elevated RAM pressure during queued batch: deferring VLM inspection until batch completes.",
                prefer_lightweight_model=True,
            )
        if snapshot.temp_severity == "emergency":
            return LLMExecutionSchedule(
                mode="cpu_only",
                num_gpu=0,
                keep_alive="0s",
                reason="GPU in thermal cooldown: scheduling lightweight local VLM on CPU only (num_gpu=0).",
                prefer_lightweight_model=True,
            )

    if active_generation_running or comfy_models_warm or upcoming_same_model:
        if batch_queue_depth >= 3:
            return LLMExecutionSchedule(
                mode="deferred_batch",
                num_gpu=0,
                keep_alive="0s",
                reason="Multi-item batch active with warm ComfyUI weights: deferring VLM inspection to protect throughput.",
                prefer_lightweight_model=True,
            )
        return LLMExecutionSchedule(
            mode="cpu_only",
            num_gpu=0,
            keep_alive="0s",
            reason="Warm/active ComfyUI models present: running lightweight VLM on CPU (num_gpu=0) to avoid VRAM eviction.",
            prefer_lightweight_model=True,
        )

    if allow_sequential_gpu:
        gpu_free_mb = recent_gpu_headroom(snapshot)
        if gpu_free_mb is not None and gpu_free_mb >= max(1, min_gpu_free_mb):
            return LLMExecutionSchedule(
                mode="sequential_gpu",
                num_gpu=-1,
                keep_alive="0s",
                reason=(
                    f"GPU cold with {gpu_free_mb} MB free VRAM (>= {min_gpu_free_mb} MB): "
                    "running sequential GPU VLM with verified VRAM release."
                ),
                verify_vram_release=True,
            )

    return LLMExecutionSchedule(
        mode="cpu_only",
        num_gpu=0,
        keep_alive="0s",
        reason="Zero-contention CPU-only local VLM inspection (keep_alive=0s).",
        prefer_lightweight_model=True,
    )


def encode_image_for_vlm(path: str | Path, *, max_side: int = 1024) -> str:
    """Encode a local image to base64 JPEG in-memory without modifying the source file."""
    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Local image not found for visual inspection: {src}")
    with Image.open(src) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        if max(rgb.size) > max_side:
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Safely extract a single JSON object from raw LLM/VLM text; returns None if malformed."""
    text = str(raw_text or "").strip()
    if not text:
        return None
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
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
    """Lightweight loopback-only Ollama JSON/VLM client with strict resource & timeout controls."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, *, timeout_s: float = 35.0) -> None:
        self.base_url = _ensure_loopback_url(base_url)
        self.timeout_s = max(2.0, float(timeout_s))

    def list_models(self) -> list[LocalLLMInfo]:
        return inspect_local_ollama_models(self.base_url, timeout_s=min(3.0, self.timeout_s))

    def list_running_models(self) -> list[dict[str, Any]]:
        return list_running_ollama_models(self.base_url, timeout_s=min(3.0, self.timeout_s))

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
        if sched.mode in {"disabled", "deferred_batch"}:
            return LLMCallResult(ok=False, model=model, mode=sched.mode, latency_ms=0.0, error=sched.reason)

        keep_alive_val: int | str = 0 if str(sched.keep_alive) in {"0", "0s"} else sched.keep_alive
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": "json",
            "keep_alive": keep_alive_val,
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
        if any(low_model.startswith(fam) for fam in _THINKING_FAMILIES) and "vl" not in low_model:
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

        vram_rel = None
        if sched.mode == "sequential_gpu" and sched.verify_vram_release:
            vram_rel = verify_ollama_vram_released(self.base_url, model=model, timeout_s=10.0)

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
                vram_release=vram_rel,
            )
        return LLMCallResult(
            ok=True,
            model=model,
            mode=sched.mode,
            latency_ms=latency_ms,
            data=parsed,
            raw_text=content[:1200],
            vram_release=vram_rel,
        )

    def complete_vision_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path],
        schedule: LLMExecutionSchedule | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 700,
        temperature: float = 0.0,
        max_image_side: int = 1024,
        verify_unload: bool = True,
        unload_timeout_s: float = 12.0,
    ) -> LLMCallResult:
        """Run strictly local-only multimodal visual inspection with verified VRAM release."""
        sched = schedule or choose_vision_execution_schedule()
        if sched.mode in {"disabled", "deferred_batch"}:
            return LLMCallResult(ok=False, model=model, mode=sched.mode, latency_ms=0.0, error=sched.reason)
        if not image_paths:
            return LLMCallResult(
                ok=False,
                model=model,
                mode=sched.mode,
                latency_ms=0.0,
                error="Visual inspection requires at least one readable local image.",
            )

        try:
            encoded_images = [encode_image_for_vlm(p, max_side=max_image_side) for p in image_paths]
        except Exception as exc:
            return LLMCallResult(
                ok=False,
                model=model,
                mode=sched.mode,
                latency_ms=0.0,
                error=f"Failed to read local image for VLM: {type(exc).__name__}: {exc}",
            )

        keep_alive_val: int | str = 0 if str(sched.keep_alive) in {"0", "0s"} else sched.keep_alive
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "format": json_schema if json_schema is not None else "json",
            "keep_alive": keep_alive_val,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": encoded_images},
            ],
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "num_gpu": int(sched.num_gpu),
                "num_ctx": 4096,
            },
        }
        low_model = model.lower()
        if low_model.startswith("qwen3") or low_model.startswith("qwen3.5"):
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

        vram_rel = None
        if verify_unload and sched.mode == "sequential_gpu":
            vram_rel = verify_ollama_vram_released(self.base_url, model=model, timeout_s=unload_timeout_s)

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
                error="Local VLM returned non-JSON or malformed JSON output.",
                vram_release=vram_rel,
            )
        return LLMCallResult(
            ok=True,
            model=model,
            mode=sched.mode,
            latency_ms=latency_ms,
            data=parsed,
            raw_text=content[:1500],
            vram_release=vram_rel,
        )
