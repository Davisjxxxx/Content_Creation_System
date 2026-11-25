"""Sprint 3 CPU-only tests: ModelRegistry, ResourcePlanner (simulated telemetry),
FLUX.2 & Wan 2.2 prompt compilers, data-driven prompt libraries, Guided mode integration,
and frontend progressive-disclosure preservation.

NOTE: All resource-planning tests here use explicit simulated telemetry fixtures and are
CPU-only software verification tests (distinct from live GPU hardware benchmarks).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from avatar_v2 import desktop, gpu_memory, local_engine, offline_builder as ob, render_queue
from avatar_v2.composer import build_job
from avatar_v2.gpu_memory import GPUMemoryManager, GPUSnapshot
from avatar_v2.model_registry import DEFAULT_REGISTRY, ModelRegistry
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.prompt_compilers import compile_prompt_for_model
from avatar_v2.prompt_library import (
    REQUIRED_CATEGORIES,
    build_spec_from_guided_payload,
    guided_ui_payload,
    load_prompt_library,
    validate_prompt_library,
)
from avatar_v2.prompt_schema import PromptAttribute, ReferenceAssetItem, RequirementItem, StructuredPromptSpec
from avatar_v2.render_queue import QueuedRender, RenderQueue
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceGuard, StorageCheck, StorageStatus
from avatar_v2.resource_planner import ResourcePlanner, WorkloadRequest


def _sim_snapshot(
    *,
    vram_used_mb: int = 7200,
    vram_total_mb: int = 8188,
    temp_c: int = 68,
    ram_avail_pct: float = 30.0,
    swap_pct: float = 20.0,
    psi_some: float = 0.0,
    free_disk_gb: float = 50.0,
):
    guard = ResourceGuard()
    host = HostMemoryStats(
        ram_total_mb=64000,
        ram_available_mb=int(64000 * ram_avail_pct / 100.0),
        ram_available_pct=ram_avail_pct,
        swap_total_mb=8192,
        swap_used_mb=int(8192 * swap_pct / 100.0),
        swap_used_pct=swap_pct,
        pswpin=100,
        pswpout=200,
        psi_some_avg10=psi_some,
        psi_full_avg10=0.0 if psi_some < 10 else 3.0,
    )
    gpus = [GPUTelemetry(0, "RTX 4070 Laptop GPU", vram_used_mb, vram_total_mb, max(vram_total_mb - vram_used_mb, 0), temp_c=temp_c, util_pct=40)]
    storage = StorageStatus("/sim", 915.0, free_disk_gb, "normal" if free_disk_gb >= 30 else ("warning" if free_disk_gb >= 20 else ("restricted" if free_disk_gb >= 15 else "emergency_stop")))
    return guard.evaluate(host=host, gpus=gpus, storage=storage, now=100.0)


# ------------------------------------------------------------------ 1. ModelRegistry


def test_model_registry_component_compatibility_and_incompatibility():
    reg = ModelRegistry()
    # Same model -> all components reusable.
    same = reg.diff_components("flux2_9b", "flux2_9b")
    assert same.transition_kind == "same_model" and not same.evict and not same.required

    # Wan 14B -> Wan 5B shares 6.4 GB UMT5 text encoder, but NOT 16-ch vs 48-ch VAEs or UNets!
    wan_shift = reg.diff_components("wan22_a14b", "wan22_5b")
    assert wan_shift.transition_kind == "shared_components"
    assert [c.component_id for c in wan_shift.reusable] == ["wan_umt5_fp8"]
    assert "wan21_vae_16ch" in [c.component_id for c in wan_shift.evict]
    assert "wan22_vae_48ch" in [c.component_id for c in wan_shift.required]

    # FLUX.2 4B -> FLUX.2 9B does NOT falsely treat Qwen3-4B (7680) and Qwen3-8B (12288) as compatible.
    flux_shift = reg.diff_components("flux2_4b", "flux2_9b")
    assert flux_shift.transition_kind == "cross_family"  # only tiny 160 MB VAE shared -> full evict safer
    assert [c.component_id for c in flux_shift.reusable] == ["flux2_vae"]
    assert "qwen3_4b" in [c.component_id for c in flux_shift.evict]
    assert "qwen3_8b" in [c.component_id for c in flux_shift.required]

    # FLUX.2 9B -> Wan 2.2 14B has zero shared components.
    cross = reg.diff_components("flux2_9b", "wan22_a14b")
    assert cross.transition_kind == "cross_family" and cross.reusable == ()


# ------------------------------------------------------------------ 2. ResourcePlanner (Simulated Telemetry)


def test_resource_planner_replaces_fixed_85_eviction_for_consecutive_flux2_9b(monkeypatch):
    planner = ResourcePlanner()
    # Simulated FLUX.2 9B resting at 89% total GPU VRAM (7,287 / 8,188 MiB) under healthy RAM & temp.
    snap = _sim_snapshot(vram_used_mb=7287, temp_c=66, ram_avail_pct=32.0, swap_pct=15.0)
    ok_storage = StorageCheck(True, "normal", 48.0, 40.0, 47.96, 15.0)

    plan = planner.plan_execution(
        WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img", width=896, height=1280, frames=1),
        current_model_id="flux2_9b",
        upcoming_model_id="flux2_9b",
        snapshot=snap,
        storage_check=ok_storage,
    )
    assert plan.admitted is True
    assert plan.retention_strategy == "retain_all"
    assert plan.pre_unload_models is False
    assert plan.respect_retention_above_threshold is True
    assert plan.vlm_schedule == "cpu_only"  # avoids evicting warm 9B weights!

    # Verify GPUMemoryManager honors respect_retention=True even at 89% VRAM!
    calls = []

    class FakeProv:
        def free_memory(self, **kw):
            calls.append(kw)

    mgr = GPUMemoryManager(threshold=0.85, enabled=True)
    monkeypatch.setattr(gpu_memory, "query_gpu_memory", lambda: [GPUSnapshot(0, "RTX 4070", 7287, 8188)])
    ev = mgr.maybe_cleanup(
        FakeProv(),
        reason="pre-render:low_memory · retain_all",
        force=True,
        unload_models=False,
        respect_retention=True,
    )
    assert calls == [] and ev["models_retained"] is True and ev["cleanup_requested"] is False


def test_resource_planner_transitions_pressure_storage_and_recovery():
    planner = ResourcePlanner()
    ok_storage = StorageCheck(True, "normal", 48.0, 200.0, 47.8, 15.0)
    healthy = _sim_snapshot(vram_used_mb=2000, temp_c=60, ram_avail_pct=35.0, swap_pct=20.0)

    # Shared-component transition (Wan 14B -> Wan 5B) delegates selective eviction to ComfyUI & enables tiled VAE for 5B.
    p_shared = planner.plan_execution(
        WorkloadRequest(model_id="wan22_5b", workflow="workflows/wan22_i2v_5b_turbo_gguf_api.json", width=480, height=832, frames=81),
        current_model_id="wan22_a14b",
        snapshot=healthy,
        storage_check=ok_storage,
    )
    assert p_shared.admitted is True
    assert p_shared.retention_strategy == "comfy_native_selective"
    assert p_shared.pre_unload_models is False
    assert p_shared.vae_strategy == "tiled"
    assert "wan_umt5_fp8" in p_shared.reusable_components

    # Cross-family transition (FLUX.2 9B -> Wan 14B) forces full eviction.
    p_cross = planner.plan_execution(
        WorkloadRequest(model_id="wan22_a14b", workflow="workflows/wan22_i2v_14b_gguf_api.json", width=480, height=832, frames=81),
        current_model_id="flux2_9b",
        snapshot=healthy,
        storage_check=ok_storage,
    )
    assert p_cross.retention_strategy == "full_evict" and p_cross.pre_unload_models is True
    assert p_cross.fallback_workflow == "workflows/wan22_i2v_5b_turbo_gguf_api.json"

    # Prior failure forces clean eviction even on same model.
    p_fail = planner.plan_execution(
        WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img"),
        current_model_id="flux2_9b",
        prior_failed_or_cancelled=True,
        snapshot=healthy,
        storage_check=ok_storage,
    )
    assert p_fail.retention_strategy == "full_evict" and p_fail.pre_unload_models is True

    # Thermal emergency (85°C) defers admission.
    hot = _sim_snapshot(vram_used_mb=2000, temp_c=85, ram_avail_pct=30.0)
    p_hot = planner.plan_execution(
        WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img"),
        snapshot=hot,
        storage_check=ok_storage,
    )
    assert p_hot.admitted is False and "temperature" in p_hot.admission_reason.lower()

    # Storage reserve breach blocks admission.
    low_disk = StorageCheck(False, "emergency_stop", 14.5, 50.0, 14.45, 15.0, "Storage emergency stop: 14.5 GB free")
    p_disk = planner.plan_execution(
        WorkloadRequest(model_id="flux2_4b", workflow="builtin:flux2_txt2img"),
        snapshot=healthy,
        storage_check=low_disk,
    )
    assert p_disk.admitted is False and "emergency stop" in p_disk.admission_reason.lower()

    # Post-render with VLM QA forces full unload so Ollama doesn't share VRAM.
    post_qa = planner.plan_post_render(
        completed_model_id="flux2_9b",
        next_queued_model_id="flux2_9b",
        job_status="done",
        requires_vlm_qa=True,
        snapshot=healthy,
    )
    assert post_qa["unload_models"] is True


# ------------------------------------------------------------------ 3. Model-Specific Prompt Compilers


def test_flux2_and_wan22_compilers_strip_sdxl_weights_and_flag_unsupported_capabilities():
    # FLUX.2 9B strips SDXL (tag:1.4) weights and warns rather than propagating SDXL syntax.
    img_spec = StructuredPromptSpec.from_guided(
        original_prompt="(photorealistic:1.4) portrait of an adult instructor in a sunlit loft",
        attributes={"lighting": "soft window light", "composition": "medium portrait"},
        locked_fields={"lighting"},
        excluded=["watermark", "cartoon"],
    )
    c9 = compile_prompt_for_model(img_spec, "flux2_9b")
    assert c9.ok is True
    assert ":1.4" not in c9.positive
    assert c9.positive.startswith("photorealistic portrait of an adult instructor")
    assert "soft window light" in c9.positive
    assert c9.negative == "watermark, cartoon"
    assert any("SDXL" in w for w in c9.warnings)
    assert c9.provenance["lighting"]["state"] == "locked"

    # FLUX.2 9B MUST NOT silently drop a mandatory motion attribute!
    bad_motion_for_still = img_spec.model_copy(
        update={"motion": PromptAttribute(value="waves hello to camera", state="locked", provenance="user_explicit")}
    )
    c_bad = compile_prompt_for_model(bad_motion_for_still, "flux2_9b")
    assert c_bad.ok is False and any("motion" in u.lower() for u in c_bad.unsupported_capabilities)

    # Wan 2.2 14B compiles motion-first and notes CFG 1.0 negative limitation.
    vid_spec = StructuredPromptSpec(
        original_prompt="The presenter greets the viewer from a studio desk",
        intent={"task": "video_generate", "input_mode": "guided"},
        motion=PromptAttribute(value="smiles warmly and waves hello", state="specified", provenance="guided_ui"),
        camera_movement=PromptAttribute(value="slow forward push-in", state="locked", provenance="guided_ui"),
        reference_assets=[ReferenceAssetItem(asset_id="s", role="start_frame", path="/tmp/s.png")],
        required_attributes=[RequirementItem(id="wave", category="video_motion", description="waves hello")],
        excluded_attributes=["flicker", "extra fingers"],
        constraints={"width": 480, "height": 832, "fps": 16, "duration_s": 5.0, "frames": 81},
    )
    cw = compile_prompt_for_model(vid_spec, "wan22_a14b")
    assert cw.ok is True
    assert "smiles warmly and waves hello" in cw.positive
    assert "slow forward push-in" in cw.positive
    assert cw.positive.index("smiles warmly") < cw.positive.index("slow forward push-in")
    assert any("CFG 1.0" in w for w in cw.warnings)


# ------------------------------------------------------------------ 4. Prompt Libraries & Guided Mode Integration


def test_structured_prompt_library_validation_and_guided_reproducibility():
    lib = load_prompt_library()
    assert set(REQUIRED_CATEGORIES).issubset(set(lib["categories"]))
    assert validate_prompt_library(lib) == []

    # Reproducible randomization with locked attributes preserved.
    guided_payload = {
        "attributes": {"lighting": "window_natural", "composition": "medium_portrait"},
        "locked": ["lighting"],
        "randomized": ["environment", "atmosphere"],
        "seed": 777,
    }
    s1 = build_spec_from_guided_payload(prompt="Adult instructor portrait", task="photo", guided=guided_payload)
    s2 = build_spec_from_guided_payload(prompt="Adult instructor portrait", task="photo", guided=guided_payload)
    assert s1.environment.value == s2.environment.value and s1.environment.value != ""
    assert s1.lighting.state == "locked" and "window" in s1.lighting.value.lower()


def test_guided_mode_backend_and_minimal_ui_integration(tmp_path: Path, monkeypatch):
    # Verify UI has collapsed-by-default Guided section and preserves natural-language default without technical knobs.
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert '<details class="card" id="forgeGuided"' in html
    assert '<details class="card" id="forgeGuided" open' not in html  # collapsed by default!
    assert 'id="forgeGuidedGrid"' in html and 'id="forgeGuidedRecipe"' in html
    assert "forgeRenderGuided()" in html

    # Verify desktop_studio forge_state / forge_price / forge_go accept optional guided payload and preserve Automatic mode.
    comfy = tmp_path / "comfy"
    (comfy / "main.py").parent.mkdir(parents=True)
    (comfy / "main.py").write_text("# mock")
    (comfy / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True)
    for sub, fname in [
        ("diffusion_models", ob.PHOTO_MODELS["4b"]["unet"]),
        ("text_encoders", ob.PHOTO_MODELS["4b"]["text_encoder"]),
        ("vae", ob.SHARED_MODELS["FORGE_VAE"][1]),
    ]:
        p = comfy / "models" / sub / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ok")

    monkeypatch.setattr(desktop, "APP_HOME", tmp_path / "home")
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "home" / "desktop.json")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", tmp_path / "home" / "library")
    monkeypatch.setattr(desktop, "QUEUE_DIR", tmp_path / "home" / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", tmp_path / "home" / "history")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", tmp_path / "home" / "cloud_ledger.jsonl")
    api = desktop.DesktopAPI()
    try:
        api.save_settings({"comfyui_dir": str(comfy), "comfyui_python": str(tmp_path / "py")})
        state = api.forge_state()
        assert state["ok"] is True and "guided" in state
        assert len(state["guided"]["categories"]) >= 8 and len(state["guided"]["recipes"]) >= 3

        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "ready", "message": "ready"})
        queued_items = []
        monkeypatch.setattr(api.queue, "add_many", lambda items: [queued_items.append(i) or i.id for i in items])

        # Automatic mode (no guided key) stays unchanged.
        auto_res = api.forge_go({"action": "photo", "prompt": "Adult presenter portrait", "photo_model": "4b", "count": 1})
        assert auto_res["ok"] is True and queued_items[-1].job.prompt == "Adult presenter portrait"

        # Guided mode compiles structured attributes into natural FLUX.2 prose.
        guided_res = api.forge_go(
            {
                "action": "photo",
                "prompt": "Adult presenter portrait",
                "photo_model": "4b",
                "count": 1,
                "guided": {
                    "attributes": {"lighting": "window_natural", "environment": "sunlit_loft"},
                    "locked": ["lighting"],
                    "randomized": [],
                    "seed": 41001,
                },
            }
        )
        assert guided_res["ok"] is True
        compiled_prompt = queued_items[-1].job.prompt
        assert compiled_prompt.startswith("Adult presenter portrait")
        assert "window" in compiled_prompt.lower() and "loft" in compiled_prompt.lower()
    finally:
        api.queue.shutdown()
