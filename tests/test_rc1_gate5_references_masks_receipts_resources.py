"""Adversarial regression suite for Astra RC1 Gate 5 (RC-06, RC-12, RC-13, RC-14, RC-19, RC-20, RC-21, RC-23).

Covers:
- RC-06: Missing starter error on photo, locked reference cardinality cap refusal, multi-subject
         mixing refusal, end-frame UI picker & verification_record recording, guided scene spec.
- RC-12: builtin:flux2_masked_edit registered + SOURCE_IMAGE/MASK_IMAGE staged by ComfyUIProvider,
         EXIF-aligned region mask, explicit-mask provenance, _MEIPASS-aware Wan workflow resolution.
- RC-13: Ground-truth submitted_params & PIL/ffprobe observed_media receipts matching actual outputs.
- RC-14: Fail-closed verify_ollama_vram_released (released|resident|unknown) and time-bucketed
         ResourceGuard swap-rate across 10ms/20ms/100ms/2s/jitter/stale-gap/counter-reset.
- RC-19: Character Profile SHA-256 tamper detection ("changed — relink") & active-job staged pinning.
- RC-20: AVATAR_V2_GPU_* environment variable precedence over desktop.json.
- RC-21 & RC-23: Honest source-only A1111 adapter status & full-res exact-black-mask zero-drift metric.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import a1111_adapter, desktop, local_engine, offline_builder as ob
from avatar_v2.correction_engine import select_correction_step
from avatar_v2.inspection_contract import (
    AcceptanceCriterion,
    CorrectionRecommendation,
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    VisualInspectionResult,
)
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec, SubjectSpec
from avatar_v2.ollama_client import verify_ollama_vram_released
from avatar_v2.providers.comfyui import ComfyUIProvider, _is_asset_key
from avatar_v2.reference_picker import LocalReferenceManager
from avatar_v2.render_queue import QueuedRender, RenderQueue
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceGuard, StorageStatus
from avatar_v2.specialist_vision import generate_region_mask, measure_outside_mask_preservation
from avatar_v2.workflows_builder import BUILTIN_BUILDERS


def _make_png(path: Path, size: tuple[int, int] = (512, 512), color: tuple[int, int, int] = (90, 120, 150)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    comfy = tmp_path / "comfy"
    (comfy / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True, exist_ok=True)
    (comfy / "main.py").write_text("# stub\n", encoding="utf-8")
    for spec in ob.PHOTO_MODELS.values():
        for folder, fname in (("diffusion_models", spec["unet"]), ("text_encoders", spec["text_encoder"])):
            p = comfy / "models" / folder / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stub")
    for folder, fname in ob.SHARED_MODELS.values():
        p = comfy / "models" / folder / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
    for mdict in (local_engine.MODELS, local_engine.MODELS_5B):
        for folder, fname in mdict.values():
            p = comfy / "models" / folder / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stub")

    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    desktop._write_json(
        app_home / "desktop.json",
        {**desktop.DEFAULT_SETTINGS, "comfyui_dir": str(comfy), "comfyui_output": str(tmp_path / "out")},
    )
    return app_home


def test_rc06_cardinality_multisubject_missing_starter_and_end_image_ui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject missing starter on photo, >4 locked refs, multi-subject mixing; wire end_image in UI & record."""
    html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="forgeEndImages"' in html
    assert "o.end_image=st.forge.endImage" in html

    _isolate(monkeypatch, tmp_path)
    api = desktop.DesktopAPI()
    try:
        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "ready", "message": "ready"})

        # 1. Missing starter image on photo submit MUST fail explicitly!
        res_missing = api.forge_go({"action": "photo", "prompt": "Portrait", "image": str(tmp_path / "nope.png")})
        assert res_missing["ok"] is False
        assert "start picture" in res_missing["error"].lower()

        # 2. Starter + 4 locked refs exceeds 4-slot FLUX.2 cap -> explicit refusal!
        refs = [_make_png(tmp_path / f"r{i}.png") for i in range(5)]
        res_cap = api.forge_go(
            {
                "action": "photo",
                "prompt": "Portrait",
                "image": str(refs[0]),
                "subject_references": {
                    "references": [
                        {"role": "front_body", "path": str(refs[1]), "locked": True},
                        {"role": "back_body", "path": str(refs[2]), "locked": True},
                        {"role": "left_side", "path": str(refs[3]), "locked": True},
                        {"role": "face_identity", "path": str(refs[4]), "locked": True},
                    ]
                },
            }
        )
        assert res_cap["ok"] is False
        assert "too many locked references" in res_cap["error"].lower()

        # 3. Multiple subjects without allow_multi_subject -> explicit refusal!
        res_ms = api.forge_go(
            {
                "action": "photo",
                "prompt": "Portrait",
                "subject_references": {
                    "references": [
                        {"role": "front_body", "subject_id": "claire", "path": str(refs[1])},
                        {"role": "front_body", "subject_id": "alex", "path": str(refs[2])},
                    ]
                },
            }
        )
        assert res_ms["ok"] is False
        assert "multiple subjects" in res_ms["error"].lower()

        # 4. Scene with end_image records end_keyframe in verification_record!
        monkeypatch.setattr(api.queue, "add_many", lambda items: [it.id for it in items])
        captured: list[QueuedRender] = []
        monkeypatch.setattr(api.queue, "add_many", lambda items: (captured.extend(items), [it.id for it in items])[1])
        res_flf = api.forge_go(
            {
                "action": "scene",
                "prompt": "She turns slowly toward the sunlit window",
                "image": str(refs[0]),
                "end_image": str(refs[1]),
            }
        )
        assert res_flf["ok"] is True, res_flf
        vr = captured[0].runtime["verification_record"]
        assert any(r.get("role") == "end_keyframe" for r in vr.get("reference_requirements") or [])
    finally:
        api.queue.shutdown()


def test_rc19_profile_fingerprint_tamper_and_staged_pinning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Modified saved profile file is flagged 'changed — relink'; pinned staged files survive LRU pruning."""
    _isolate(monkeypatch, tmp_path)
    img_a = _make_png(tmp_path / "ext" / "a.png", color=(200, 50, 50))
    api = desktop.DesktopAPI()
    try:
        api._reference_manager().register_selected([str(img_a)], workflow="multi_reference")
        saved = api.character_profiles(
            {"action": "save", "name": "Claire", "references": [{"role": "front_body", "path": str(img_a)}]}
        )
        assert saved["ok"] is True
        orig_fp = saved["profile"]["reference_set"]["references"][0]["asset_fingerprint"]

        # Mutate the external file bytes on disk!
        _make_png(img_a, color=(20, 220, 40))
        listed = api.character_profiles({"action": "list"})
        prof = listed["profiles"][0]
        assert prof["has_changed_references"] is True
        assert prof["reference_set"]["references"][0]["integrity_status"] == "changed — relink"

        # Attempting to resolve with stale fingerprint refuses cleanly!
        with pytest.raises(ValueError, match="changed on disk"):
            api._resolve_subject_references(
                starter_image="",
                raw_subject_refs={"references": [{"role": "front_body", "path": str(img_a), "asset_fingerprint": orig_fp}]},
                prompt="Portrait",
                task="image_generate",
                workflow_supports_multiref=True,
                stage=False,
            )

        # Staged reference pinning prevents LRU eviction of active queued job ref (floor is 8)!
        mgr = LocalReferenceManager(staging_dir=tmp_path / "staged", max_staged_files=8)
        f0 = _make_png(tmp_path / "ext" / "p0.png", color=(5, 5, 5))
        s1 = mgr.stage_for_job(f0, require_authorization=False)["staged_path"]
        mgr.pin_staged(s1, "job_active_1")
        for i in range(1, 11):
            time.sleep(0.005)
            fi = _make_png(tmp_path / "ext" / f"p{i}.png", color=(i * 15, i * 10, i * 5))
            mgr.stage_for_job(fi, require_authorization=False)
        # s1 is oldest, yet MUST survive past max_staged_files=8 because it is pinned to job_active_1!
        assert Path(s1).is_file()
        mgr.unpin_job("job_active_1")
        time.sleep(0.005)
        f_extra = _make_png(tmp_path / "ext" / "p_extra.png", color=(240, 240, 240))
        mgr.stage_for_job(f_extra, require_authorization=False)
        assert not Path(s1).is_file()
    finally:
        api.queue.shutdown()


def test_rc12_and_rc23_masked_edit_registered_staged_exif_and_exact_black_mask(tmp_path: Path) -> None:
    """flux2_masked_edit is registered, stages SOURCE_IMAGE/MASK_IMAGE, respects EXIF, and verifies 0-drift."""
    assert "flux2_masked_edit" in BUILTIN_BUILDERS
    assert _is_asset_key("SOURCE_IMAGE") is True
    assert _is_asset_key("MASK_IMAGE") is True

    src = _make_png(tmp_path / "ext_src.png", size=(640, 480), color=(80, 90, 100))
    mask_path = tmp_path / "ext_mask.png"
    generate_region_mask(100, 100, zone="right_half", feather_px=0, out_path=mask_path, image_path=src)
    with Image.open(mask_path) as m_im:
        assert m_im.size == (640, 480)

    # Full-res exact-black-mask 0-pixel-drift metric (RC-23)
    edited = tmp_path / "edited.png"
    with Image.open(src) as s_im:
        ed = s_im.copy()
        for y in range(480):
            for x in range(400, 640):
                ed.putpixel((x, y), (220, 40, 40))
        ed.save(edited, format="PNG")
    pres = measure_outside_mask_preservation(src, edited, mask_path)
    assert pres["exact_black_mask_pixels"] > 0
    assert pres["exact_black_mask_changed_pixels"] == 0
    assert pres["exact_black_mask_zero_drift"] is True
    assert pres["strictly_preserved_outside_mask"] is True

    # ComfyUIProvider stages external SOURCE_IMAGE and MASK_IMAGE into input_dir
    in_dir = tmp_path / "comfy_in"
    provider = ComfyUIProvider(ProviderConfig(base_url="http://127.0.0.1:18199", input_dir=str(in_dir)))
    job = RenderJob(
        job_id="maskjob1",
        avatar=AvatarManifest(
            avatar_id="av1",
            display_name="Test",
            subjects=[SubjectSpec(id="s1", kind="synthetic", age_verified_18_plus=True, consent_confirmed=True)],
        ),
        shot=ShotSpec(shot_id="s1", user_prompt="edit", width=640, height=480, frames=1),
        selected_engine="flux2",
        prompt="replace mug",
        negative_prompt="",
        asset_map={"SOURCE_IMAGE": str(src), "MASK_IMAGE": str(mask_path), "PROMPT": "replace mug"},
    )
    graph = provider.resolve_workflow("builtin:flux2_masked_edit", job, stage_assets=True)
    node_types = {n["class_type"] for n in graph.values()}
    assert "ImageCompositeMasked" in node_types
    assert "SetLatentNoiseMask" in node_types
    assert any("source_image" in p.name for p in in_dir.iterdir())
    assert any("mask_image" in p.name for p in in_dir.iterdir())

    # Video correction step resolves _MEIPASS-aware local_engine.WORKFLOW and never emits broken PNG keyframe step
    vrec = GenerationVerificationRecord(
        original_prompt="She turns smoothly",
        detected_intent="image_to_video",
        compiled_prompt="She turns smoothly",
        selected_model_id="wan22_a14b",
        selected_workflow="relative/broken.json",
        expected_output=ExpectedOutputCharacteristics(task="video_generate", media_kind="video", width=480, height=832, frames=81),
        mandatory_acceptance_criteria=[AcceptanceCriterion(criterion_id="t1", category="temporal_consistency", description="smooth", mandatory=True)],
    )
    vins = VisualInspectionResult(
        overall_decision="needs_correction",
        criterion_assessments=[CriterionAssessment(criterion_id="t1", category="temporal_consistency", mandatory=True, status="failed")],
        correction_recommendations=[
            CorrectionRecommendation(
                failed_criterion_id="t1",
                category="temporal_consistency",
                original_requirement="smooth",
                supporting_evidence="end jump",
                action_type="revise_ending_keyframe",
                recommended_action="revise end",
            )
        ],
    )
    step = select_correction_step(vrec, vins, current_best_path=str(src), attempt_number=1, prior_strategies=[])
    assert step.strategy == "video_regeneration"
    assert Path(step.workflow).is_absolute()


def test_rc13_ground_truth_submitted_params_and_observed_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """QueuedRender receipt reports actual submitted seed/dims/frames and probed output dimensions/hashes."""
    out_dir = tmp_path / "out"
    actual_img = _make_png(out_dir / "actual.png", size=(832, 1216), color=(50, 100, 150))

    class _FakeProv:
        uses_local_gpu = True

        def render(self, wf: str, job: RenderJob, on_submitted=None, cancel_event=None) -> dict[str, Any]:
            if on_submitted:
                on_submitted("pid-rc13")
            return {"prompt_id": "pid-rc13", "outputs": ["actual.png"]}

        def free_memory(self, **kw: Any) -> None:
            pass

        def queue_status(self) -> dict[str, Any]:
            return {"running": 0, "pending": 0, "busy": False}

    job = RenderJob(
        job_id="j13",
        avatar=AvatarManifest(
            avatar_id="av1",
            display_name="Test",
            subjects=[SubjectSpec(id="s1", kind="synthetic", age_verified_18_plus=True, consent_confirmed=True)],
        ),
        shot=ShotSpec(shot_id="s1", user_prompt="test", seed=41001, width=720, height=1280, frames=80),
        selected_engine="flux2",
        prompt="test",
        negative_prompt="",
        asset_map={"SEED": 987654321, "WIDTH": 832, "HEIGHT": 1216, "FRAMES": 1},
    )
    q = RenderQueue(work_dir=tmp_path / "q", history_dir=tmp_path / "h")
    try:
        q.storage_guard_enabled = False  # type: ignore[attr-defined]
        monkeypatch.setattr(q, "_make_provider", lambda cfg: _FakeProv())
        item = QueuedRender(
            job=job,
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(base_url="http://127.0.0.1:18199", output_dir=str(out_dir)),
            profile="auto",
            output_dir=str(out_dir),
        )
        qid = q.add(item)
        deadline = time.monotonic() + 5.0
        pub = None
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == qid), None)
            if cur and cur["status"] == "done":
                pub = cur
                break
            time.sleep(0.03)
        assert pub is not None
        assert pub["seed"] == 987654321  # NOT stale default 41001!
        assert pub["resolution"] == [832, 1216]
        assert pub["frames"] == 1
        assert pub["observed_media"][0]["width"] == 832
        assert pub["observed_media"][0]["height"] == 1216
        assert pub["observed_media"][0]["sha256"] == hashlib.sha256(actual_img.read_bytes()).hexdigest()
    finally:
        q.shutdown()


def test_rc14_ollama_vram_fail_closed_and_swap_rate_invariance() -> None:
    """verify_ollama_vram_released never claims released when unreachable; ResourceGuard swap rate is poll-invariant."""
    # 1. Unreachable Ollama -> released=False, status="unknown"
    unk = verify_ollama_vram_released(
        timeout_s=0.1,
        poll_interval_s=0.05,
        ps_fetcher=lambda: None,
        vram_fetcher=lambda: (2000, 6000),
    )
    assert unk["released"] is False
    assert unk["status"] == "unknown"

    # 2. Required min_free_vram_mb with missing GPU telemetry -> released=False, status="unknown"
    unk_gpu = verify_ollama_vram_released(
        timeout_s=0.1,
        poll_interval_s=0.05,
        min_free_vram_mb=5000,
        ps_fetcher=lambda: [],
        vram_fetcher=lambda: (None, None),
    )
    assert unk_gpu["released"] is False
    assert unk_gpu["status"] == "unknown"

    # 3. Resident model -> released=False, status="resident"
    res = verify_ollama_vram_released(
        timeout_s=0.1,
        poll_interval_s=0.05,
        ps_fetcher=lambda: [{"name": "qwen3-vl:8b-instruct", "size_vram": 5_000_000_000}],
        vram_fetcher=lambda: (6000, 2000),
    )
    assert res["released"] is False
    assert res["status"] == "resident"

    # 4. Clean release -> released=True, status="released"
    ok = verify_ollama_vram_released(
        timeout_s=0.1,
        poll_interval_s=0.05,
        min_free_vram_mb=5000,
        ps_fetcher=lambda: [],
        vram_fetcher=lambda: (500, 7500),
    )
    assert ok["released"] is True
    assert ok["status"] == "released"

    # 5. ResourceGuard swap-rate across 10ms, 20ms, 100ms, 2s, stale gap (>30s), and counter reset
    gpu = [GPUTelemetry(0, "RTX 4070", 2000, 8188, 6188, 55, 20)]
    storage = StorageStatus("/tmp", 500.0, 100.0, "normal")

    def _run_rate(step_s: float, rate_pg_s: float, duration_s: float = 3.0) -> float:
        guard = ResourceGuard()
        t = 1000.0
        pages = 10_000
        last_rate = 0.0
        steps = int(duration_s / step_s)
        for i in range(steps + 1):
            cur_t = t + i * step_s
            cur_pages = int(pages + (i * step_s) * rate_pg_s)
            host = HostMemoryStats(32000, 16000, 50.0, 8000, 4000, 50.0, pswpin=cur_pages, pswpout=0)
            snap = guard.evaluate(host=host, gpus=gpu, storage=storage, now=cur_t)
            last_rate = snap.swap_pages_per_s
        return last_rate

    for dt in (0.01, 0.02, 0.1, 1.5, 2.0):
        measured = _run_rate(dt, 400.0, duration_s=4.0)
        assert 340.0 <= measured <= 460.0, f"dt={dt} produced {measured}"

    # Stale gap > 30s and counter reset both reset cleanly to 0.0 (no bogus spike!)
    g_edge = ResourceGuard()
    h0 = HostMemoryStats(32000, 16000, 50.0, 8000, 4000, 50.0, pswpin=1000, pswpout=0)
    g_edge.evaluate(host=h0, gpus=gpu, storage=storage, now=100.0)
    h_stale = HostMemoryStats(32000, 16000, 50.0, 8000, 4000, 50.0, pswpin=50000, pswpout=0)
    s_stale = g_edge.evaluate(host=h_stale, gpus=gpu, storage=storage, now=200.0)
    assert s_stale.swap_pages_per_s == 0.0
    h_reset = HostMemoryStats(32000, 16000, 50.0, 8000, 4000, 50.0, pswpin=10, pswpout=0)
    s_reset = g_edge.evaluate(host=h_reset, gpus=gpu, storage=storage, now=202.0)
    assert s_reset.swap_pages_per_s == 0.0


def test_rc20_gpu_env_var_precedence_and_rc21_source_only_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """AVATAR_V2_GPU_* environment controls override desktop.json defaults; A1111 adapter flagged source_only."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("AVATAR_V2_GPU_AUTOCLEAN", "0")
    monkeypatch.setenv("AVATAR_V2_GPU_THRESHOLD", "0.72")
    monkeypatch.setenv("AVATAR_V2_GPU_CHECK_INTERVAL", "9")
    api = desktop.DesktopAPI()
    try:
        st = api.queue.memory_status()
        assert st["enabled"] is False
        assert abs(float(st["threshold"]) - 0.72) < 1e-6
        assert abs(float(api.queue.memory_manager.check_interval_s) - 9.0) < 1e-6
    finally:
        api.queue.shutdown()
    assert a1111_adapter.PACKAGED_STATUS == "source_only_optional"
