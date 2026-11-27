"""Adversarial RC3 Gate 5 tests: model-matched video correction workflow dispatch,
exact resolved-graph receipts & ffprobe video telemetry, bidirectional ComfyUI<->Ollama VRAM handoff,
fail-closed broker capabilities when comfyui_dir is missing, and recursive process-tree/RSS enforcement.

Directly reproduces Astra RC2 independent recheck findings:
- RC-12: Wan 5B correction paired model_id='wan22_5b' with 14B workflow
- RC-13: Wan resolved noise_seed=1428693541 reported as 41001; FLUX still frames=144; 1s 8-frame MP4 ffprobe 0.0s
- RC-14: Ordinary RenderQueue admitted ComfyUI while Ollama held VRAM; failed pre-VLM /free cleared comfy_warm=False
- RC-15: GenerationBroker(comfyui_dir=None) advertised all 4 models as installed
- RC-17: Subprocess tree with os.setsid() escaped single-PGID process count & aggregate RSS limits
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import avatar_v2.desktop as desktop
import avatar_v2.local_engine as local_engine
import avatar_v2.ollama_client as ollama_client
import avatar_v2.render_queue as rq
import avatar_v2.script_broker as sb
import avatar_v2.script_runner as sr
from avatar_v2.composer import build_job
from avatar_v2.correction_engine import select_correction_step
from avatar_v2.gpu_memory import GPUMemoryManager
from avatar_v2.inspection_contract import (
    CorrectionRecommendation,
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    VisualInspectionResult,
)
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec


def _make_job(job_id: str = "rc3-g5-1", *, seed: int = 41001, frames: int = 144) -> RenderJob:
    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "av-1",
            "display_name": "Test",
            "subjects": [{"id": "av-1", "kind": "synthetic"}],
            "identity_refs": [],
        }
    )
    shot = ShotSpec.model_validate(
        {
            "shot_id": "sh-1",
            "user_prompt": "A woman smiling warmly in soft studio light",
            "seed": seed,
            "duration_seconds": max(1.0, frames / 24.0),
        }
    )
    job = build_job(avatar, shot)
    job.job_id = job_id
    return job


# ---------------------------------------------------------------------------
# RC-12: Wan 5B correction MUST dispatch wan22_i2v_5b_turbo_gguf_api.json + 5B placeholders
# ---------------------------------------------------------------------------


def test_rc12_wan5b_correction_selects_5b_workflow_and_placeholders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-12 counterexample:
    A Wan 2.2 5B job failing visual inspection MUST plan and dispatch wan22_i2v_5b_turbo_gguf_api.json
    with Wan2.2_VAE.safetensors and WAN5B_MODEL — NEVER the 14B workflow!
    """
    record = GenerationVerificationRecord(
        verification_id="ver-rc12-5b",
        detected_intent="image_to_video",
        original_prompt="She walks steadily forward in a sunlit studio.",
        compiled_prompt="She walks steadily forward in a sunlit studio.",
        selected_model_id="wan22_5b",
        selected_workflow=str(local_engine.WORKFLOW_5B),
        expected_output=ExpectedOutputCharacteristics(task="video_scene", media_kind="video", width=480, height=832, frames=81),
    )
    inspection = VisualInspectionResult(
        output_path=str(tmp_path / "orig.mp4"),
        overall_decision="needs_correction",
        criterion_assessments=[
            CriterionAssessment(
                criterion_id="req_motion",
                category="video_temporal_consistency",
                original_requirement="Smooth steady walk without flicker",
                mandatory=True,
                status="failed",
                confidence=0.9,
                observed_evidence="Noticeable mid-clip temporal flicker",
                failure_description="Mid-clip temporal flicker",
            )
        ],
        correction_recommendations=[
            CorrectionRecommendation(
                failed_criterion_id="req_motion",
                category="video_temporal_consistency",
                original_requirement="Smooth steady walk without flicker",
                supporting_evidence="Noticeable mid-clip temporal flicker",
                action_type="inspect_temporal_interval",
                recommended_action="Regenerate video cleanly",
            )
        ],
    )

    step = select_correction_step(
        record,
        inspection,
        current_best_path=str(tmp_path / "orig.mp4"),
        attempt_number=1,
        prior_strategies=[],
        base_seed=41001,
    )
    assert step.model_id == "wan22_5b"
    assert Path(step.workflow).name == local_engine.WORKFLOW_5B.name
    assert "14b" not in Path(step.workflow).name.lower()

    # Also verify _execute_local_correction_step passes 5B workflow + 5B placeholders to ComfyUIProvider!
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")

    captured: dict[str, Any] = {}
    out_mp4 = tmp_path / "cor_5b.mp4"
    out_mp4.write_bytes(b"fake-mp4")

    class CaptureComfy:
        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            captured["workflow"] = workflow
            captured["asset_map"] = dict(job.asset_map)
            return {"prompt_id": "cor-5b-pid", "outputs": [str(out_mp4)]}

    monkeypatch.setattr(desktop, "ComfyUIProvider", CaptureComfy)
    api = desktop.DesktopAPI()
    try:
        item = rq.QueuedRender(
            id="job-5b-orig",
            job=_make_job("job-5b-orig"),
            workflow=str(local_engine.WORKFLOW_5B),
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="low_memory",
        )
        out_path, _ms = api._execute_local_correction_step(item, step)
        assert Path(out_path).is_file()
        assert Path(captured["workflow"]).name == local_engine.WORKFLOW_5B.name
        assert captured["asset_map"]["WAN5B_MODEL"] == local_engine.MODELS_5B["WAN5B_MODEL"][1]
        assert captured["asset_map"]["WAN_VAE"] == local_engine.MODELS_5B["WAN5B_VAE"][1]
    finally:
        api.queue.shutdown()

    # Unknown video model MUST raise ValueError (fail closed, never fall back to 14B)
    with pytest.raises(ValueError, match="Unsupported local Wan video model"):
        local_engine.workflow_for_video_model("unknown_video_99b")


# ---------------------------------------------------------------------------
# RC-13: Resolved graph seed/frames accuracy & exact ffprobe video telemetry
# ---------------------------------------------------------------------------


def test_rc13_resolved_graph_seed_still_frames_and_real_mp4_ffprobe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces ALL THREE Astra RC-13 counterexamples:
    1. Resolved Wan graph with RandomNoise.noise_seed=1428693541 vs ShotSpec.seed=41001 -> receipt records 1428693541!
    2. Single FLUX.2 still -> receipt records frames=1 (NEVER 144!).
    3. Real ffmpeg-created 1s 160x96 8-frame MP4 -> observed_media reports width=160, height=96, frames=8, duration_s=1.0!
    """
    # Part 1: Real ffmpeg 1s 160x96 8-frame MP4 probe
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        mp4 = tmp_path / "test_1s_8f.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x96:d=1:r=8",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
            ],
            check=True,
            capture_output=True,
        )
        probed = rq._probe_output_media([str(mp4)])
        assert len(probed) == 1
        assert probed[0]["width"] == 160
        assert probed[0]["height"] == 96
        assert probed[0]["frames"] == 8
        assert abs(float(probed[0]["duration_s"]) - 1.0) <= 0.15

    # Part 2 & 3: Resolved graph extraction for Wan seed=1428693541 AND FLUX still frames=1
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    out_png = tmp_path / "still.png"
    Image.new("RGB", (512, 768), color=(80, 120, 160)).save(out_png, format="PNG")
    out_mp4 = tmp_path / "wan_out.mp4"
    out_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    class ResolvedGraphProvider:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, *, on_resolved=None, on_submitted=None, cancel_event=None) -> dict[str, Any]:
            if workflow == "builtin:flux2_txt2img":
                resolved = {
                    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b-fp8.safetensors"}},
                    "5": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 512, "height": 768}},
                    "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 999888777}},
                }
                outs = [str(out_png)]
            else:
                resolved = {
                    "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf"}},
                    "12": {"class_type": "WanImageToVideo", "inputs": {"width": 480, "height": 832, "length": 81}},
                    "13": {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "enable", "noise_seed": 1428693541, "steps": 4, "cfg": 1.0}},
                }
                outs = [str(out_mp4)]
            if on_resolved is not None:
                on_resolved(resolved)
            if on_submitted is not None:
                on_submitted("pid-rc13")
            return {"prompt_id": "pid-rc13", "outputs": outs}

    monkeypatch.setattr(rq, "ComfyUIProvider", ResolvedGraphProvider)
    # Bypass Ollama HTTP check in unit test
    monkeypatch.setattr(ollama_client, "fetch_ollama_ps_state", lambda *a, **k: ("offline", [], "offline"))

    q = rq.RenderQueue(work_dir=work_dir, history_dir=history_dir, memory_manager=GPUMemoryManager(enabled=False))
    try:
        wan_job = _make_job("job-wan-seed", seed=41001, frames=144)
        q.add(
            rq.QueuedRender(
                id="job-wan-seed",
                job=wan_job,
                workflow=str(local_engine.WORKFLOW),
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        flux_job = _make_job("job-flux-still", seed=41001, frames=144)
        q.add(
            rq.QueuedRender(
                id="job-flux-still",
                job=flux_job,
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            items = {x["id"]: x for x in q.list()}
            if all(items[k]["status"] == "done" for k in ("job-wan-seed", "job-flux-still")):
                break
            time.sleep(0.05)

        wan_rec = json.loads((history_dir / "job-wan-seed.json").read_text(encoding="utf-8"))
        flux_rec = json.loads((history_dir / "job-flux-still.json").read_text(encoding="utf-8"))
        assert wan_rec["seed"] == 1428693541, f"Expected resolved Wan seed 1428693541, got {wan_rec['seed']}"
        assert wan_rec["submitted_params"]["seed"] == 1428693541
        assert flux_rec["submitted_params"]["frames"] == 1, f"FLUX still reported frames={flux_rec['submitted_params']['frames']}!"
        assert flux_rec["frames"] == 1
        assert flux_rec["seed"] == 999888777
    finally:
        q.shutdown()


# ---------------------------------------------------------------------------
# RC-14: Bidirectional ComfyUI <-> Ollama VRAM handoff & verified pre-VLM unload
# ---------------------------------------------------------------------------


def test_rc14_render_queue_blocks_comfyui_when_ollama_holds_vram(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's RC-14 counterexample A:
    Ordinary RenderQueue execution MUST check Ollama GPU residency before starting ComfyUI,
    and refuse dispatch if Ollama remains GPU-resident!
    """
    dispatched: list[str] = []

    class SpyComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            dispatched.append(job.job_id)
            return {"prompt_id": "should-not-dispatch", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", SpyComfy)
    monkeypatch.setattr(
        ollama_client,
        "fetch_ollama_ps_state",
        lambda *a, **k: ("online", [{"name": "qwen3-vl:8b", "size_vram": 5_500_000_000}], ""),
    )
    monkeypatch.setattr(
        ollama_client,
        "verify_ollama_vram_released",
        lambda *a, **k: {"released": False, "status": "resident", "active_vram_models": [{"name": "qwen3-vl:8b", "size_vram": 5_500_000_000}]},
    )

    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        q.add(
            rq.QueuedRender(
                id="job-ollama-blocked",
                job=_make_job("job-ollama-blocked"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == "job-ollama-blocked"), None)
            if cur and cur["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.05)
        cur = next(x for x in q.list() if x["id"] == "job-ollama-blocked")
        assert cur["status"] == "error"
        assert "ollama" in cur["error"].lower() and "vram" in cur["error"].lower()
        assert dispatched == [], "ComfyUI was dispatched while Ollama still held GPU VRAM!"
    finally:
        q.shutdown()


def test_rc14_failed_pre_vlm_unload_keeps_comfy_warm_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's RC-14 counterexample B:
    When pre-VLM maybe_cleanup() returns an error / cleanup_performed=False,
    DesktopAPI MUST keep comfy_models_warm=True and NOT clear _last_local_model_id!
    """
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")

    img = tmp_path / "out.png"
    Image.new("RGB", (256, 256), color=(100, 120, 140)).save(img, format="PNG")
    observed_warm: list[bool] = []

    class SpyVisualInspector:
        def __init__(self, **kwargs) -> None:
            pass

        def inspect_image(self, target, record, *, comfy_models_warm: bool = False, **kwargs):
            observed_warm.append(comfy_models_warm)
            return VisualInspectionResult(
                output_path=str(target),
                overall_decision="pass",
            )

    import avatar_v2.visual_inspector as vi

    monkeypatch.setattr(vi, "VisualInspector", SpyVisualInspector)

    api = desktop.DesktopAPI()
    try:
        api.queue._last_local_model_id = "flux2_9b"
        monkeypatch.setattr(
            api.queue.memory_manager,
            "maybe_cleanup",
            lambda *a, **k: {"reason": "pre-vlm-unload", "cleanup_performed": False, "error": "connection refused"},
        )
        ver = GenerationVerificationRecord(
            verification_id="ver-rc14",
            detected_intent="image_generate",
            original_prompt="Portrait",
            compiled_prompt="Portrait",
            selected_model_id="flux2_9b",
            selected_workflow="builtin:flux2_txt2img",
            expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=256, height=256, frames=1),
        )
        item = rq.QueuedRender(
            id="job-rc14-vlm",
            job=_make_job("job-rc14-vlm"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="auto",
            outputs=[str(img)],
            runtime={"verification_record": ver.public(), "visual_inspection_vlm": True},
        )
        api._run_visual_inspection(item)
        assert observed_warm == [True], f"Expected comfy_models_warm=True after failed /free, got {observed_warm}"
        assert api.queue._last_local_model_id == "flux2_9b", "_last_local_model_id was cleared despite failed /free!"
    finally:
        api.queue.shutdown()


# ---------------------------------------------------------------------------
# RC-15: GenerationBroker fails closed when comfyui_dir is None or missing
# ---------------------------------------------------------------------------


def test_rc15_broker_fails_closed_when_comfyui_dir_missing(tmp_path: Path) -> None:
    """Reproduces Astra's exact RC-15 counterexample:
    GenerationBroker(comfyui_dir=None) MUST advertise zero installed models and reject submission!
    """
    broker = sb.GenerationBroker(
        run_id="run-rc15",
        submit_fn=lambda ctx: "should-not-submit",
        status_fn=lambda jid: {"status": "done"},
        comfyui_dir=None,
    )
    caps = broker.handle_request(
        {
            "token": broker.token,
            "contract_version": sb.SDK_CONTRACT_VERSION,
            "op": "capabilities",
            "payload": {},
        }
    )
    assert caps["ok"] is True
    assert caps["models"] == [], f"Broker advertised models without comfyui_dir: {caps['models']}"

    sub = broker.handle_request(
        {
            "token": broker.token,
            "contract_version": sb.SDK_CONTRACT_VERSION,
            "op": "submit",
            "payload": {
                "model_id": "flux2_4b",
                "spec": {"original_prompt": "A ceramic vase on a table"},
            },
        }
    )
    assert sub["ok"] is False
    assert sub["code"] in {"unsupported_model", "model_unavailable", "missing_model"}


# ---------------------------------------------------------------------------
# RC-17: Recursive descendant tree process count (across os.setsid) & aggregate RSS
# ---------------------------------------------------------------------------


def test_rc17_descendant_tree_tracks_setsid_children_and_aggregate_rss() -> None:
    """Reproduces Astra's exact RC-17 counterexample:
    A child process that calls os.setsid() and spawns grandchildren MUST still be counted
    in _collect_descendant_tree_stats() with non-zero aggregate VmRSS, and terminated cleanly!
    """
    parent_script = (
        "import os, subprocess, sys, time; "
        "c1 = subprocess.Popen([sys.executable, '-c', 'import os, time; os.setsid(); time.sleep(30)']); "
        "c2 = subprocess.Popen([sys.executable, '-c', 'import os, time; os.setsid(); time.sleep(30)']); "
        "print(f'{c1.pid},{c2.pid}', flush=True); "
        "time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        c1_pid, c2_pid = [int(x) for x in line.split(",")]
        # Give children a moment to call os.setsid()
        time.sleep(0.25)

        pgid = os.getpgid(proc.pid)
        # Verify c1 and c2 are in DIFFERENT process groups from parent!
        assert os.getpgid(c1_pid) != pgid
        assert os.getpgid(c2_pid) != pgid

        tel_ok, pids, total_rss_kb, tel_err = sr.ScriptRunManager._collect_descendant_tree_stats(proc.pid, pgid)
        assert tel_ok is True, f"Telemetry failed unexpectedly: {tel_err}"
        assert {proc.pid, c1_pid, c2_pid}.issubset(pids), (
            f"setsid() descendants escaped process tree tracking! expected {proc.pid, c1_pid, c2_pid}, got {pids}"
        )
        assert total_rss_kb > 1000, f"Expected aggregate VmRSS > 1000 KiB, got {total_rss_kb}"

        # Terminate entire descendant tree including setsid() children!
        sr.ScriptRunManager._terminate_process_group(proc, pgid=pgid)
        proc.wait(timeout=5.0)
        time.sleep(0.2)
        _tel_ok2, rem_pids, _rss2, _err2 = sr.ScriptRunManager._collect_descendant_tree_stats(proc.pid, pgid)
        assert c1_pid not in rem_pids and c2_pid not in rem_pids
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)
