"""Sprint 1 regression tests: /free keep-loaded fix, ComfyUI RAM bounds, graduated
resource monitoring, predictive storage guard, VAEDecodeTiled schema, and benchmark preflight."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avatar_v2 import desktop, gpu_memory, render_queue
from avatar_v2.composer import build_job
from avatar_v2.gpu_memory import GPUMemoryManager, GPUSnapshot
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.render_queue import QueuedRender, RenderQueue
from avatar_v2.resource_monitor import (
    GPUTelemetry,
    HostMemoryStats,
    ResourceGuard,
    StorageStatus,
    check_storage_for_job,
    classify_storage_tier,
    estimate_job_storage_bytes,
    parse_proc_meminfo,
    parse_proc_psi_memory,
    parse_proc_vmstat_swap,
)
from avatar_v2.workflows_builder import build_flux2_txt2img_workflow, with_tiled_vae


def _sample_item(tmp_path: Path, workflow: str = "builtin:flux2_txt2img", status: str = "queued") -> QueuedRender:
    avatar = AvatarManifest.model_validate(
        {"avatar_id": "a", "display_name": "A", "subjects": [{"id": "a", "kind": "synthetic"}]}
    )
    shot = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "portrait", "width": 896, "height": 1280})
    return QueuedRender(
        job=build_job(avatar, shot),
        workflow=workflow,
        provider_config=ProviderConfig(base_url="http://127.0.0.1:8188"),
        profile="low_memory",
        output_dir=str(tmp_path),
        status=status,  # type: ignore[arg-type]
    )


def test_gpu_memory_manager_retains_models_below_threshold_and_escalates_above(monkeypatch):
    calls: list[dict[str, bool]] = []

    class FakeProvider:
        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
            calls.append({"unload_models": unload_models, "free_memory": free_memory})

    mgr = GPUMemoryManager(threshold=0.85, enabled=True)

    # Below 85%: unload_models=False skips ComfyUI /free so warm FLUX.2 models actually stay resident.
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: [GPUSnapshot(0, "RTX 4070", 4500, 8188)])
    ev_keep = mgr.maybe_cleanup(FakeProvider(), reason="pre-render:low_memory", force=True, unload_models=False)
    assert calls == []
    assert ev_keep["cleanup_requested"] is False
    assert ev_keep["models_retained"] is True
    assert ev_keep["unload_models"] is False

    # At/above 85%: safety valve overrides unload_models=False and triggers full /free unload.
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: [GPUSnapshot(0, "RTX 4070", 7200, 8188)])
    ev_high = mgr.maybe_cleanup(FakeProvider(), reason="pre-render:low_memory", force=True, unload_models=False, wait_s=0.0)
    assert calls == [{"unload_models": True, "free_memory": True}]
    assert ev_high["cleanup_requested"] is True
    assert ev_high["models_retained"] is False
    assert ev_high["unload_models"] is True


def test_render_queue_unloads_on_workflow_switch_and_error(tmp_path: Path, monkeypatch):
    import threading

    queue = RenderQueue.__new__(RenderQueue)
    queue.work_dir = tmp_path / "q"
    queue.work_dir.mkdir()
    queue.history_dir = tmp_path / "h"
    queue.history_dir.mkdir()
    queue._items = []
    queue._lock = threading.RLock()
    queue._wake = threading.Event()
    queue._stop = threading.Event()
    queue._active_provider = None
    queue._active_id = None
    queue._last_local_workflow = "workflows/wan22_i2v_14b_gguf_api.json"
    queue.candidate_qa_runner = None
    queue._engine_restart = None
    queue.storage_guard_enabled = False

    recorded = []

    class FakeMem:
        def maybe_cleanup(self, provider, **kwargs):
            recorded.append(kwargs)
            return kwargs

        def start_trace(self, _label):
            pass

        def stop_trace(self):
            return {}

    still_png = tmp_path / "still.png"
    still_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    class FakeProv:
        def __init__(self, _cfg):
            pass

        def render(self, _wf, _job):
            return {"prompt_id": "p1", "outputs": [str(still_png)]}

    queue.memory_manager = FakeMem()
    monkeypatch.setattr(render_queue, "ComfyUIProvider", FakeProv)

    # Switching from Wan video -> FLUX.2 image must unload Wan weights first.
    img1 = _sample_item(tmp_path, workflow="builtin:flux2_txt2img")
    queue._execute(img1)
    assert recorded[-1]["unload_models"] is True
    assert "model switch" in recorded[-1]["reason"]

    # Second consecutive FLUX.2 image keeps models loaded.
    img2 = _sample_item(tmp_path, workflow="builtin:flux2_txt2img")
    queue._execute(img2)
    assert recorded[-1]["unload_models"] is False
    assert "keep image models" in recorded[-1]["reason"]

    # Error on an image job must force unload_models=True in post-cleanup.
    img2.status = "error"
    post = queue._post_cleanup(img2, FakeProv(img2.provider_config))
    assert post["unload_models"] is True
    assert queue._last_local_workflow is None


def test_comfyui_launch_args_preserve_multi_token_flags_and_bound_ram(monkeypatch):
    monkeypatch.delenv("AVATAR_V2_COMFY_CACHE_RAM", raising=False)
    args_64g = desktop.DesktopAPI._build_comfyui_launch_args(
        {"comfyui_args": "--disable-auto-launch --reserve-vram 1.5"},
        "8188",
        ram_total_mb=63911,
    )
    assert args_64g == [
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch", "--reserve-vram", "1.5", "--cache-ram", "6", "24",
    ]

    # Explicit user --cache-ram is preserved with both numeric tokens and not duplicated.
    args_custom = desktop.DesktopAPI._build_comfyui_launch_args(
        {"comfyui_args": "--disable-auto-launch --cache-ram 4 16 ; rm -rf /"},
        "8188",
        ram_total_mb=63911,
    )
    assert args_custom == [
        "main.py", "--listen", "127.0.0.1", "--port", "8188",
        "--disable-auto-launch", "--cache-ram", "4", "16",
    ]


def test_resource_monitor_graduated_thresholds_swap_gating_and_hysteresis():
    meminfo = parse_proc_meminfo("MemTotal: 65536000 kB\nMemAvailable: 12000000 kB\nSwapTotal: 8388608 kB\nSwapFree: 8192 kB\n")
    assert meminfo["MemTotal"] == 65536000 and meminfo["SwapFree"] == 8192
    assert parse_proc_vmstat_swap("pswpin 100\npswpout 250\n") == (100, 250)
    assert parse_proc_psi_memory("some avg10=12.50 avg60=4.00\nfull avg10=3.10 avg60=1.00\n") == (12.5, 3.1)

    guard = ResourceGuard()
    ok_storage = StorageStatus("/home/jd", 915.0, 48.0, "normal")

    # High static swap (99%) WITHOUT active swap activity or PSI stall stays warning (not emergency).
    s1 = guard.evaluate(
        host=HostMemoryStats(64000, 16000, 25.0, 8192, 8150, 99.5, pswpin=1000, pswpout=2000, psi_some_avg10=0.0, psi_full_avg10=0.0),
        gpus=[GPUTelemetry(0, "RTX 4070", 4000, 8188, 4188, temp_c=68, util_pct=90)],
        storage=ok_storage,
        now=100.0,
    )
    assert s1.swap_severity == "warning" and s1.active_memory_pressure is False and s1.overall_severity == "warning"

    # Active paging burst (>64 pages/s) with >90% swap escalates to emergency.
    s2 = guard.evaluate(
        host=HostMemoryStats(64000, 16000, 25.0, 8192, 8150, 99.5, pswpin=1400, pswpout=2400, psi_some_avg10=15.0, psi_full_avg10=3.0),
        gpus=[GPUTelemetry(0, "RTX 4070", 4000, 8188, 4188, temp_c=85, util_pct=100)],
        storage=ok_storage,
        now=102.0,
    )
    assert s2.swap_severity == "emergency" and s2.active_memory_pressure is True
    assert s2.temp_severity == "emergency"

    # Thermal hysteresis: cooling to 81°C stays at emergency until <= 79°C (84 - 5).
    s3 = guard.evaluate(
        host=HostMemoryStats(64000, 20000, 31.2, 8192, 1000, 12.2, pswpin=1400, pswpout=2400, psi_some_avg10=0.0, psi_full_avg10=0.0),
        gpus=[GPUTelemetry(0, "RTX 4070", 2000, 8188, 6188, temp_c=81, util_pct=50)],
        storage=ok_storage,
        now=110.0,
    )
    assert s3.temp_severity == "emergency"

    s4 = guard.evaluate(
        host=HostMemoryStats(64000, 20000, 31.2, 8192, 1000, 12.2, pswpin=1400, pswpout=2400, psi_some_avg10=0.0, psi_full_avg10=0.0),
        gpus=[GPUTelemetry(0, "RTX 4070", 1200, 8188, 6988, temp_c=68, util_pct=10)],
        storage=ok_storage,
        now=120.0,
    )
    assert s4.temp_severity == "normal" and s4.overall_severity == "normal"


def test_predictive_storage_tiers_and_reserve_block(tmp_path: Path, monkeypatch):
    assert classify_storage_tier(45.0) == "normal"
    assert classify_storage_tier(25.0) == "warning"
    assert classify_storage_tier(18.0) == "restricted"
    assert classify_storage_tier(14.9) == "emergency_stop"

    from avatar_v2 import resource_monitor

    # Simulate 15.1 GB free: small image allowed, multi-shot video blocked from crossing 15 GB reserve.
    monkeypatch.setattr(
        resource_monitor,
        "query_storage_status",
        lambda _p=None: StorageStatus(str(tmp_path), 500.0, 15.1, "restricted"),
    )
    img_check = check_storage_for_job(tmp_path, estimated_bytes=estimate_job_storage_bytes(kind="image", width=896, height=1280))
    assert img_check.allowed is True

    scene_bytes = estimate_job_storage_bytes(kind="scene", width=480, height=832, frames=81, shots=6)
    scene_check = check_storage_for_job(tmp_path, estimated_bytes=scene_bytes)
    assert scene_check.allowed is False
    assert "protected reserve" in scene_check.reason


def test_tiled_vae_schema_on_flux2_and_wan_workflows():
    # Baseline 896x1280 FLUX.2 keeps standard VAEDecode unless TILED_VAE=True.
    base_graph = build_flux2_txt2img_workflow({"WIDTH": 896, "HEIGHT": 1280})
    assert any(n["class_type"] == "VAEDecode" for n in base_graph.values())
    assert not any(n["class_type"] == "VAEDecodeTiled" for n in base_graph.values())

    # High-res FLUX.2 (>1024x1280) automatically upgrades to schema-valid VAEDecodeTiled.
    hires_graph = build_flux2_txt2img_workflow({"WIDTH": 1280, "HEIGHT": 1920})
    tiled_nodes = [n for n in hires_graph.values() if n["class_type"] == "VAEDecodeTiled"]
    assert len(tiled_nodes) == 1
    assert set(tiled_nodes[0]["inputs"]) == {"samples", "vae", "tile_size", "overlap", "temporal_size", "temporal_overlap"}

    # Wan 2.2 14B and 5B workflow JSONs stay untiled by default and transform cleanly with with_tiled_vae().
    root = Path(__file__).resolve().parents[1] / "workflows"
    for name in ("wan22_i2v_14b_gguf_api.json", "wan22_i2v_5b_turbo_gguf_api.json"):
        raw = json.loads((root / name).read_text(encoding="utf-8"))
        assert any(n["class_type"] == "VAEDecode" for n in raw.values())
        upgraded = with_tiled_vae(raw)
        assert not any(n["class_type"] == "VAEDecode" for n in upgraded.values())
        tiled = [n for n in upgraded.values() if n["class_type"] == "VAEDecodeTiled"]
        assert len(tiled) == 1 and tiled[0]["inputs"]["overlap"] == 64 and tiled[0]["inputs"]["temporal_overlap"] == 8
