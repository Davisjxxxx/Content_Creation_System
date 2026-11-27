"""Sprint 10 Integrated Acceptance Test Suite (Journeys A–G, Packaging, Fault Injection, Sandbox & Boundaries)."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop, local_engine, offline_builder as ob, render_queue
from avatar_v2.a1111_adapter import A1111CompatibilityAdapter
from avatar_v2.correction_engine import CorrectionCoordinator, build_corrective_prompt
from avatar_v2.inspection_contract import (
    AcceptanceCriterion,
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    VisualInspectionResult,
)
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec, SubjectSpec
from avatar_v2.ollama_client import LLMCallResult, LocalLLMInfo, LocalOllamaClient
from avatar_v2.prompt_extractor import extract_prompt_data
from avatar_v2.resource_monitor import (
    GPUTelemetry,
    HostMemoryStats,
    ResourceGuard,
    StorageCheck,
    StorageStatus,
)
from avatar_v2.resource_planner import ResourcePlanner, WorkloadRequest
from avatar_v2.script_broker import BrokerSubmitContext, GenerationBroker
from avatar_v2.script_inspector import inspect_script
from avatar_v2.script_runner import (
    SandboxUnavailableError,
    ScriptRunManager,
    probe_sandbox_support,
    resolve_script_python,
)
from avatar_v2.subject_references import SubjectReferenceSet
from avatar_v2.visual_inspector import VisualInspector


# --------------------------------------------------------------------------- Helpers


def _make_png(path: Path, size: tuple[int, int] = (512, 512), color: tuple[int, int, int] = (80, 120, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _make_fake_comfy_install(root: Path) -> Path:
    comfy = root / "comfyui"
    comfy.mkdir(parents=True, exist_ok=True)
    (comfy / "main.py").write_text("# stub comfyui\n", encoding="utf-8")
    (comfy / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True, exist_ok=True)
    for spec in ob.PHOTO_MODELS.values():
        for folder, fname in (
            ("diffusion_models", spec["unet"]),
            ("text_encoders", spec["text_encoder"]),
        ):
            p = comfy / "models" / folder / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stub")
    for folder, fname in ob.SHARED_MODELS.values():
        p = comfy / "models" / folder / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
    for model_dict in (local_engine.MODELS, local_engine.MODELS_5B):
        for folder, fname in model_dict.values():
            p = comfy / "models" / folder / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stub")
    return comfy


def _isolate_desktop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    app_home = tmp_path / "avatar_home"
    app_home.mkdir(parents=True, exist_ok=True)
    comfy = _make_fake_comfy_install(tmp_path)
    out_dir = tmp_path / "comfy_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = tmp_path / "comfy_in"
    in_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "COMFY_PID_PATH", app_home / "comfyui.pid")
    monkeypatch.setattr(desktop, "COMFY_LOG_PATH", app_home / "logs" / "comfyui.log")
    monkeypatch.setattr(desktop, "RUNWAY_AVATARS_PATH", app_home / "runway_avatars.json")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    monkeypatch.setattr(desktop, "VOICEOVER_DIR", app_home / "voiceover")
    monkeypatch.setattr(desktop, "IMPORTS_DIR", app_home / "library" / "imports")
    monkeypatch.setattr(desktop, "LESSONS_PATH", app_home / "lessons.json")
    monkeypatch.setattr(desktop, "QA_DIR", app_home / "qa")
    monkeypatch.setattr(desktop, "PROVIDER_PROFILES_PATH", app_home / "provider_profiles.json")
    from avatar_v2 import gpu_memory as _gm

    monkeypatch.setattr(_gm, "query_gpu_memory", lambda: [])
    _orig_cleanup = _gm.GPUMemoryManager.maybe_cleanup
    monkeypatch.setattr(
        _gm.GPUMemoryManager,
        "maybe_cleanup",
        lambda self, provider=None, **kw: _orig_cleanup(self, provider, **{**kw, "wait_s": 0.0}),
    )

    settings = {
        **desktop.DEFAULT_SETTINGS,
        "comfyui_dir": str(comfy),
        "comfyui_python": sys.executable,
        "comfyui_input": str(in_dir),
        "comfyui_output": str(out_dir),
        "archive_dir": str(app_home / "archive"),
        "comfyui_url": "http://127.0.0.1:18199",
    }
    desktop._write_json(app_home / "desktop.json", settings)
    return app_home, out_dir


def _wait_queue_done(api: desktop.DesktopAPI, queue_id: str, timeout_s: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hist = desktop.HISTORY_DIR / f"{queue_id}.json"
        if hist.is_file():
            return json.loads(hist.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"Queue item {queue_id} did not persist history within {timeout_s}s; queue={api.queue.list()}")


# --------------------------------------------------------------------------- Journey A: Automatic Image Generation


def test_journey_a_automatic_generation_history_and_binding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_home, out_dir = _isolate_desktop(monkeypatch, tmp_path)
    out_img = _make_png(out_dir / "auto_cafe.png", (896, 1280), (140, 95, 70))

    class StubComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            self.cfg = cfg

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
            return {"ok": True}

        def render(self, workflow: str, job: RenderJob) -> dict[str, Any]:
            return {"prompt_id": "p-auto-1", "outputs": [str(out_img)]}

    monkeypatch.setattr(desktop, "ComfyUIProvider", StubComfy)
    monkeypatch.setattr(render_queue, "ComfyUIProvider", StubComfy)
    monkeypatch.setenv("AVATAR_V2_STORAGE_GUARD", "0")

    api = desktop.DesktopAPI()
    try:
        res = api.forge_go(
            {
                "action": "photo",
                "prompt": "A wooden table with a red ceramic vase on the left and a brass lamp on the right",
                "photo_model": "9b",
                "shape": "portrait",
                "count": 1,
            }
        )
        assert res["ok"] is True, res
        qid = res["queue_id"]
        done = _wait_queue_done(api, qid)
        assert done["status"] == "done", done
        assert done["outputs"]

        # Reopen from durable history and verify original requirements, compiled prompt, model settings & inspection
        hist_path = app_home / "history" / f"{qid}.json"
        assert hist_path.is_file()
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        vr = hist.get("verification_record")
        assert isinstance(vr, dict)
        assert "red ceramic vase on the left" in vr["original_prompt"]
        assert vr["selected_model_id"] == "flux2_9b"
        assert vr["compiled_prompt"]
        assert hist["steps"] == ob.PHOTO_MODELS["9b"]["steps"]
        assert hist["cfg"] == ob.PHOTO_MODELS["9b"]["cfg"]
        assert isinstance(vr.get("inspection_result"), dict)
        assert hist["candidate_qa"]["ok"] is True
        # Studio results surfaces history cleanly for reopen
        results = api._studio_results(limit=10)
        assert any(r["id"] == qid and r["inspection"] is not None for r in results)
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- Journey B: Guided Generation (Lock + Randomize)


def test_journey_b_guided_lock_randomize_survives_compilation_routing_and_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home, out_dir = _isolate_desktop(monkeypatch, tmp_path)
    out_img = _make_png(out_dir / "guided_portrait.png", (896, 1280), (90, 110, 140))

    class StubComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
            return {"ok": True}

        def render(self, workflow: str, job: RenderJob) -> dict[str, Any]:
            return {"prompt_id": "p-guided-1", "outputs": [str(out_img)]}

    monkeypatch.setattr(desktop, "ComfyUIProvider", StubComfy)
    monkeypatch.setattr(render_queue, "ComfyUIProvider", StubComfy)
    monkeypatch.setenv("AVATAR_V2_STORAGE_GUARD", "0")

    api = desktop.DesktopAPI()
    try:
        res = api.forge_go(
            {
                "action": "photo",
                "prompt": "Portrait of an adult woman architect holding blueprints",
                "photo_model": "9b",
                "guided": {
                    "attributes": {
                        "lighting": "golden hour warm directional sunlight",
                        "wardrobe": "tailored navy wool blazer",
                    },
                    "locked": ["lighting", "wardrobe"],
                    "randomized": ["environment"],
                    "random_seed": 4242,
                },
            }
        )
        assert res["ok"] is True, res
        qid = res["queue_id"]
        done = _wait_queue_done(api, qid)
        assert done["status"] == "done", done

        hist = json.loads((app_home / "history" / f"{qid}.json").read_text(encoding="utf-8"))
        vr = hist["verification_record"]
        assert "golden hour warm directional sunlight" in vr["compiled_prompt"]
        assert "tailored navy wool blazer" in vr["compiled_prompt"]
        locked_criteria = [c for c in vr["mandatory_acceptance_criteria"] if c.get("locked")]
        assert len(locked_criteria) >= 2
        assert any("golden hour warm directional sunlight" in c["description"] for c in locked_criteria)
        assert any("tailored navy wool blazer" in c["description"] for c in locked_criteria)

        # Verify CorrectionCoordinator preserves locked criteria in corrective prompt
        record = GenerationVerificationRecord.model_validate(vr)
        cor_prompt = build_corrective_prompt(
            record,
            failed_assessments=[
                CriterionAssessment(
                    criterion_id="req_hand",
                    category="subject",
                    original_requirement="holding blueprints",
                    status="failed",
                )
            ],
            satisfied_assessments=[],
            strategy="prompt_refinement",
            is_localized_edit=False,
        )
        assert "golden hour warm directional sunlight" in cor_prompt
        assert "tailored navy wool blazer" in cor_prompt
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- Journey C: Local Images, Profiles, Unicode, Staging Cache, Multi-Subject


def test_journey_c_local_images_profiles_unicode_and_ephemeral_staging_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home, _out_dir = _isolate_desktop(monkeypatch, tmp_path)

    user_dir = tmp_path / "My Photos — café résumé 日本"
    front = _make_png(user_dir / "claire front café.png", (640, 800), (180, 120, 100))
    back = _make_png(user_dir / "claire back view.png", (640, 800), (120, 140, 180))
    side = _make_png(user_dir / "claire left side.png", (640, 800), (110, 170, 120))
    face = _make_png(user_dir / "claire face close.png", (512, 512), (200, 150, 130))
    other_subj = _make_png(user_dir / "alex front.png", (640, 800), (70, 70, 190))

    orig_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (front, back, side, face, other_subj)}

    api = desktop.DesktopAPI()
    try:
        # 1. Dialog cancellation returns cleanly without mutating state
        monkeypatch.setattr(api, "choose_files", lambda *a, **kw: [])
        cancelled = api.pick_local_images({"workflow": "multi_reference", "multiple": True})
        assert cancelled["ok"] is True and cancelled["cancelled"] is True

        # 2. Browse local files with spaces + Unicode
        monkeypatch.setattr(api, "choose_files", lambda *a, **kw: [str(front), str(back), str(side), str(face), str(other_subj)])
        picked = api.pick_local_images({"workflow": "multi_reference", "multiple": True})
        assert picked["ok"] is True and len(picked["items"]) == 5

        # 3. Save Character Profile with roles + multi-subject separation
        saved = api.character_profiles(
            {
                "action": "save",
                "name": "Claire Studio — Café",
                "references": [
                    {"role": "front_body", "subject_id": "claire", "path": str(front)},
                    {"role": "back_body", "subject_id": "claire", "path": str(back)},
                    {"role": "left_side", "subject_id": "claire", "path": str(side)},
                    {"role": "face_identity", "subject_id": "claire", "path": str(face)},
                    {"role": "front_body", "subject_id": "alex", "path": str(other_subj)},
                ],
            }
        )
        assert saved["ok"] is True, saved
        pid = saved["profile"]["profile_id"]

        # Verify multi-subject separation (no accidental merging of claire + alex)
        rset = SubjectReferenceSet.model_validate(saved["profile"]["reference_set"])
        assert rset.subject_ids() == ["alex", "claire"]
        assert len(rset.by_subject()["claire"]) == 4
        assert len(rset.by_subject()["alex"]) == 1

        # 4. Stage references into ephemeral staged_refs cache, then DELETE staged_refs cache!
        staged_dir = app_home / "staged_refs"
        rset.stage_all(api._reference_manager(), require_authorization=False)
        assert staged_dir.is_dir() and any(staged_dir.iterdir())
        for f in staged_dir.glob("*"):
            f.unlink()

        # 5. Simulate restart: shut down first instance cleanly, then start second instance.
        api.queue.shutdown()
        api2 = desktop.DesktopAPI()
        try:
            listed = api2.character_profiles({"action": "list"})
            assert listed["ok"] is True
            prof = next(p for p in listed["profiles"] if p["profile_id"] == pid)
            ref_paths = [r["path"] for r in prof["reference_set"]["references"]]
            assert str(front.resolve()) in ref_paths
            assert all("staged_refs" not in p for p in ref_paths)

            # Originals remain 100% unchanged
            for p, h in orig_hashes.items():
                assert hashlib.sha256(p.read_bytes()).hexdigest() == h

            # 6. Missing/moved source file produces explicit validation error without crash
            moved = user_dir / "moved_back.png"
            back.rename(moved)
            bad = api2.pick_local_images({"action": "validate", "path": str(back), "workflow": "multi_reference"})
            assert bad["ok"] is False
            assert bad["error"]
            moved.rename(back)
            deleted = api2.character_profiles({"action": "delete", "profile_id": pid})
            assert deleted["ok"] is True
            assert all(p["profile_id"] != pid for p in deleted["profiles"])
        finally:
            api2.queue.shutdown()
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- Journey D: Video Generation + Starter Preservation + FLF2V


def test_journey_d_video_starter_never_replaced_and_flf2v_wired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home, out_dir = _isolate_desktop(monkeypatch, tmp_path)
    starter = _make_png(tmp_path / "starter_front.png", (640, 832), (100, 150, 200))
    back_ref = _make_png(tmp_path / "back_ref.png", (640, 832), (50, 90, 130))
    end_img = _make_png(tmp_path / "end_frame.png", (640, 832), (160, 110, 80))

    captured_jobs: list[RenderJob] = []

    class StubComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
            return {"ok": True}

        def render(self, workflow: str, job: RenderJob) -> dict[str, Any]:
            captured_jobs.append(job)
            vid = out_dir / "wan_out.mp4"
            vid.write_bytes(b"fake-mp4-bytes")
            return {"prompt_id": "p-vid-1", "outputs": [str(vid)]}

    monkeypatch.setattr(desktop, "ComfyUIProvider", StubComfy)
    monkeypatch.setattr(render_queue, "ComfyUIProvider", StubComfy)
    monkeypatch.setenv("AVATAR_V2_STORAGE_GUARD", "0")

    api = desktop.DesktopAPI()
    try:
        res = api.forge_go(
            {
                "action": "scene",
                "image": str(starter),
                "end_image": str(end_img),
                "shots": ["The subject turns slowly toward the window in soft daylight"],
                "video_model": "a14b",
                "shape": "tall",
                "subject_references": {
                    "references": [{"role": "back_body", "path": str(back_ref)}],
                },
            }
        )
        assert res["ok"] is True, res
        assert res["photo_first"] is False
        qid = res["queue_id"]
        done = _wait_queue_done(api, qid)
        assert done["status"] == "done", done
        assert len(captured_jobs) == 1
        amap = captured_jobs[0].asset_map
        assert hashlib.sha256(Path(amap["INIT_IMAGE"]).read_bytes()).hexdigest() == hashlib.sha256(starter.read_bytes()).hexdigest()
        assert "LAST_FRAME" in amap
        assert hashlib.sha256(Path(amap["LAST_FRAME"]).read_bytes()).hexdigest() == hashlib.sha256(end_img.read_bytes()).hexdigest()
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- Journey E: Script Mode, Shared Budget, Frozen Python, Sandbox


def test_journey_e_frozen_python_resolution_never_returns_frozen_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/dist/AvatarV2.next/AvatarV2")
    monkeypatch.delenv("AVATAR_V2_SCRIPT_PYTHON", raising=False)
    resolved = resolve_script_python()
    assert resolved != "/opt/dist/AvatarV2.next/AvatarV2"
    assert Path(resolved).is_file()
    assert "python" in Path(resolved).name.lower()


def test_journey_e_script_shared_budget_and_no_auto_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_desktop(monkeypatch, tmp_path)
    marker = tmp_path / "should_not_exist.txt"
    script_file = tmp_path / "native_batch.py"
    script_file.write_text(
        "import avatar_v2_sdk as av2\n"
        f"open({str(marker)!r}, 'w').write('RAN')\n"
        "def run():\n"
        "    av2.connect().generate_image('A cobalt mug on a cedar desk')\n",
        encoding="utf-8",
    )

    api = desktop.DesktopAPI()
    try:
        monkeypatch.setattr(api, "choose_files", lambda *a, **kw: [str(script_file)])
        sel = api.script_select()
        assert sel["ok"] is True
        assert sel["executed"] is False
        assert not marker.exists()

        # Shared host-enforced budget across original + corrective attempts
        broker = GenerationBroker(
            run_id="srun_budget_test",
            submit_fn=lambda ctx: f"job_{ctx.item_index}",
            status_fn=lambda jid: {"status": "done", "outputs": []},
            max_items=2,
        )
        ok1, _ = broker.charge_corrective_attempt("job_1", note="orig_sim")
        ok2, _ = broker.charge_corrective_attempt("job_1", note="retry_1")
        ok3, reason3 = broker.charge_corrective_attempt("job_1", note="retry_2")
        assert ok1 is True and ok2 is True and ok3 is False
        assert "budget exhausted" in reason3.lower()

        # Static extractor + A1111 adapter + legacy script explanation
        data_py = tmp_path / "prompt_library_legacy.py"
        data_py.write_text('LIGHTING = ["soft studio key light", "warm golden hour backlight"]\n', encoding="utf-8")
        extracted = extract_prompt_data(data_py)
        assert extracted.ok
        assert extracted.categories or extracted.candidate_pools or extracted.mapped_fields

        adapter = A1111CompatibilityAdapter(
            GenerationBroker(
                run_id="srun_a1111_test",
                submit_fn=lambda ctx: f"a1111_{ctx.item_index}",
                status_fn=lambda jid: {"status": "done", "outputs": []},
                max_items=2,
                storage_check_override=StorageCheck(True, "normal", 80.0, 10.0, 79.9, 15.0),
            )
        )
        models_res = adapter.handle_sd_models()
        assert models_res.ok and models_res.http_status == 200
        bad_hr = adapter.handle_txt2img({"prompt": "Studio portrait", "enable_hr": True})
        assert not bad_hr.ok and bad_hr.http_status == 400
        ok_a1111 = adapter.handle_txt2img(
            {"prompt": "Studio portrait of a violinist", "negative_prompt": "blurry", "width": 896, "height": 1152, "seed": 77}
        )
        assert ok_a1111.ok and ok_a1111.http_status == 200

        legacy_script = tmp_path / "unsupported_sdwebui.py"
        legacy_script.write_text(
            "import requests\n"
            "requests.post('http://127.0.0.1:7860/sdapi/v1/txt2img', json={'prompt': 'test', 'sd_model_checkpoint': 'x'})\n",
            encoding="utf-8",
        )
        rep = inspect_script(legacy_script)
        assert rep.ok and rep.classification in {"requires_review", "unsupported", "potentially_compatible_legacy"}
        assert rep.recommended_actions or rep.warnings or rep.summary
    finally:
        api.queue.shutdown()


def test_journey_e_live_bwrap_sandbox_isolation_and_fail_closed(tmp_path: Path) -> None:
    # 1. Fail-closed when Bubblewrap is missing
    mgr_broken = ScriptRunManager(
        runs_dir=tmp_path / "runs_broken",
        submit_fn=lambda ctx: "never",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        bwrap_override="/nonexistent/bwrap_missing",
    )
    ok_script = tmp_path / "safe_sdk.py"
    ok_script.write_text(
        "import avatar_v2_sdk as av2\n"
        "av2.connect().generate_image('A minimalist ceramic teapot on white linen')\n",
        encoding="utf-8",
    )
    with pytest.raises(SandboxUnavailableError):
        mgr_broken.start_sandboxed_run(script_path=ok_script, max_items=2)

    # 2. Live Bubblewrap boundary test
    probe = probe_sandbox_support()
    if not probe.available:
        pytest.skip(f"Bubblewrap unavailable on host: {probe.error}")

    probe_script = tmp_path / "boundary_probe.py"
    probe_script.write_text(
        "import socket, threading, avatar_v2_sdk as av2\n"
        "from pathlib import Path\n"
        "assert not Path('/dev/nvidia0').exists(), 'GPU device leaked into sandbox!'\n"
        "assert not Path.home().joinpath('.ssh').exists(), 'Host SSH dir leaked!'\n"
        "net_blocked = False\n"
        "try:\n"
        "    s = socket.create_connection(('1.1.1.1', 53), timeout=1.0)\n"
        "    s.close()\n"
        "except OSError:\n"
        "    net_blocked = True\n"
        "assert net_blocked, 'External network was reachable inside sandbox!'\n"
        "client = av2.connect()\n"
        "results = []\n"
        "def worker(i):\n"
        "    try:\n"
        "        r = client.generate_image(f'Studio glass bottle variant {i}', wait=False)\n"
        "        results.append(('ok', r.job_id))\n"
        "    except Exception as e:\n"
        "        results.append(('err', str(e)))\n"
        "threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]\n"
        "for t in threads: t.start()\n"
        "for t in threads: t.join()\n"
        "ok_count = sum(1 for k, _ in results if k == 'ok')\n"
        "assert ok_count == 2, f'Expected exactly 2 accepted jobs, got {ok_count}: {results}'\n"
        "print('BOUNDARY_OK')\n",
        encoding="utf-8",
    )

    submitted: list[BrokerSubmitContext] = []
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs_live",
        submit_fn=lambda ctx: (submitted.append(ctx) or f"jid_{len(submitted)}"),
        status_fn=lambda jid: {"status": "done", "outputs": []},
        storage_check_override=StorageCheck(True, "normal", 80.0, 10.0, 79.9, 15.0),
    )
    assert mgr.select_script(probe_script)["ok"] is True

    # TOCTOU mutation detection
    orig_text = probe_script.read_text(encoding="utf-8")
    probe_script.write_text(orig_text + "\n# tampered\n", encoding="utf-8")
    mutated = mgr.start_sandboxed_run(max_items=2)
    assert mutated["ok"] is False and mutated.get("code") == "script_mutated"

    # Restore & execute cleanly in Bubblewrap
    probe_script.write_text(orig_text, encoding="utf-8")
    assert mgr.select_script(probe_script)["ok"] is True
    started = mgr.start_sandboxed_run(max_items=2, max_concurrent=2, allow_review_classification=True)
    assert started["ok"] is True, started

    deadline = time.monotonic() + 15.0
    final_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        final_status = mgr.status()
        rec = final_status.get("record") or {}
        if rec.get("status") in {"completed", "error", "stopped"}:
            break
        time.sleep(0.1)

    rec = final_status["record"]
    assert rec["status"] == "completed", f"Sandbox run failed: {rec}"
    assert "BOUNDARY_OK" in (rec.get("stdout_tail") or "")
    assert len(submitted) == 2


# --------------------------------------------------------------------------- Journey F: Production DesktopAPI -> CorrectionCoordinator Flow


def test_journey_f_production_desktop_correction_coordinator_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home, out_dir = _isolate_desktop(monkeypatch, tmp_path)
    orig_img = _make_png(out_dir / "orig_failed.png", (896, 1280), (100, 100, 100))
    fixed_img = _make_png(out_dir / "cor_fixed.png", (896, 1280), (140, 160, 190))

    render_calls: list[dict[str, Any]] = []

    class StubComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
            return {"ok": True}

        def render(self, workflow: str, job: RenderJob) -> dict[str, Any]:
            render_calls.append({"workflow": workflow, "prompt": job.asset_map.get("PROMPT"), "seed": job.asset_map.get("SEED")})
            target = orig_img if len(render_calls) == 1 else fixed_img
            return {"prompt_id": f"p-cor-{len(render_calls)}", "outputs": [str(target)]}

    monkeypatch.setattr(desktop, "ComfyUIProvider", StubComfy)
    monkeypatch.setattr(render_queue, "ComfyUIProvider", StubComfy)
    monkeypatch.setenv("AVATAR_V2_STORAGE_GUARD", "0")
    monkeypatch.setattr("avatar_v2.ollama_client.verify_ollama_vram_released", lambda *a, **kw: {"released": True})

    def _fake_inspect(self: VisualInspector, path: Path, record: GenerationVerificationRecord, **kw: Any) -> VisualInspectionResult:
        is_fixed = Path(path).name == fixed_img.name
        cids = [c.criterion_id for c in record.mandatory_acceptance_criteria] or ["req_spatial"]
        assessments = [
            CriterionAssessment(
                criterion_id=cid,
                category="spatial_relationships",
                original_requirement="red vase on the left and brass lamp on the right",
                status="passed" if is_fixed else "failed",
                confidence=0.92 if is_fixed else 0.90,
                observed_evidence="Verified left/right placement" if is_fixed else "Vase and lamp reversed",
                assessor="local_vlm",
                inspection_method="local_vlm",
            )
            for cid in cids
        ]
        return VisualInspectionResult(
            inspector_model="qwen3-vl:8b-instruct",
            execution_mode="hybrid",
            overall_decision="pass" if is_fixed else "needs_correction",
            ui_status="requirements_verified" if is_fixed else "correction_recommended",
            criterion_assessments=assessments,
        )

    monkeypatch.setattr(VisualInspector, "inspect_image", _fake_inspect)

    api = desktop.DesktopAPI()
    try:
        res = api.forge_go(
            {
                "action": "photo",
                "prompt": "A table with a red vase on the left and a brass lamp on the right",
                "photo_model": "9b",
                "auto_correct": True,
                "visual_inspection_vlm": True,
            }
        )
        assert res["ok"] is True, res
        qid = res["queue_id"]
        done = _wait_queue_done(api, qid)
        assert done["status"] == "done", done
        assert len(render_calls) == 2
        assert len(done["outputs"]) == 2
        vr = done["verification_record"]
        cs = vr["correction_session"]
        assert cs["attempts_executed"] == 1
        assert cs["ui_status"] in {"requirements_verified", "correction_complete", "some_uncertain"}
        assert done["candidate_qa"]["correction"]["attempts_executed"] == 1
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- Journey G: Persistence & Interrupted Job Recovery


def test_journey_g_restart_recovery_without_duplication_or_false_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home, _out_dir = _isolate_desktop(monkeypatch, tmp_path)
    queue_dir = app_home / "queue"
    (queue_dir / "items").mkdir(parents=True, exist_ok=True)

    avatar = AvatarManifest(
        avatar_id="av1",
        display_name="Test",
        subjects=[
            SubjectSpec(
                id="subj1",
                kind="synthetic",
                age_verified_18_plus=True,
                consent_confirmed=True,
            )
        ],
    )
    shot = ShotSpec(shot_id="s1", user_prompt="A quiet mountain lake at sunrise", engine_preference="custom")
    job = RenderJob(
        job_id="j_interrupt_1",
        avatar=avatar,
        shot=shot,
        selected_engine="custom",
        prompt=shot.user_prompt,
        negative_prompt="",
        asset_map={},
    )
    interrupted = render_queue.QueuedRender(
        id="q_int_001",
        job=job,
        workflow="builtin:flux2_txt2img",
        provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:18199"),
        profile="low_memory",
        status="running",
        started_at=time.time() - 30.0,
    )
    q1 = render_queue.RenderQueue(work_dir=queue_dir, history_dir=app_home / "history")
    try:
        q1._persist_item_durable(interrupted)
    finally:
        q1.shutdown()

    q2 = render_queue.RenderQueue(work_dir=queue_dir, history_dir=app_home / "history")
    try:
        items = q2.list()
        matching = [it for it in items if it["id"] == "q_int_001"]
        assert len(matching) == 1
        restored = matching[0]
        assert restored["status"] == "queued"
        assert any(ev.get("reason") == "interrupted-by-restart-retry" for ev in restored["memory_events"])
        assert not (app_home / "history" / "q_int_001.json").exists()
    finally:
        q2.shutdown()


# --------------------------------------------------------------------------- Section 5: Polling Invariance, Fault Injections & Honest UI Status


def test_section5_swap_rate_invariant_across_rapid_and_slow_polling() -> None:
    gpu = [GPUTelemetry(index=0, name="RTX 4070", used_mb=2000, total_mb=8188, free_mb=6188, temp_c=55, util_pct=40)]
    storage = StorageStatus(path="/", total_gb=1000.0, free_gb=60.0, tier="normal")

    def _run_sim(interval_s: float) -> float:
        guard = ResourceGuard(storage_path="/")
        rates: list[float] = []
        t = 100.0
        steps = int(4.0 / interval_s)
        for _ in range(steps + 1):
            pages = int(round((t - 100.0) * 100.0))
            host = HostMemoryStats(
                ram_total_mb=64000,
                ram_available_mb=20000,
                ram_available_pct=31.2,
                swap_total_mb=8192,
                swap_used_mb=6500,
                swap_used_pct=79.3,
                pswpin=pages,
                pswpout=0,
            )
            snap = guard.evaluate(host=host, gpus=gpu, storage=storage, now=t)
            if t - 100.0 >= 1.5:
                rates.append(snap.swap_pages_per_s)
            t += interval_s
        return rates[-1]

    rapid_rate = _run_sim(0.1)
    slow_rate = _run_sim(2.0)
    assert 95.0 <= rapid_rate <= 105.0, f"Rapid polling distorted swap rate: {rapid_rate}"
    assert 95.0 <= slow_rate <= 105.0, f"Slow polling distorted swap rate: {slow_rate}"

    guard_glitch = ResourceGuard(storage_path="/")
    h0 = HostMemoryStats(ram_total_mb=64000, ram_available_mb=32000, ram_available_pct=50.0, swap_total_mb=8192, swap_used_mb=5000, swap_used_pct=61.0, pswpin=100, pswpout=0)
    h1 = HostMemoryStats(ram_total_mb=64000, ram_available_mb=32000, ram_available_pct=50.0, swap_total_mb=8192, swap_used_mb=5000, swap_used_pct=61.0, pswpin=101, pswpout=0)
    guard_glitch.evaluate(host=h0, gpus=gpu, storage=storage, now=10.0)
    snap_glitch = guard_glitch.evaluate(host=h1, gpus=gpu, storage=storage, now=10.1)
    assert snap_glitch.swap_pages_per_s < 10.0
    assert snap_glitch.active_memory_pressure is False


def test_section5_fault_injections_planner_vlm_timeout_and_honest_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    planner = ResourcePlanner()
    req = WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img", width=896, height=1280)
    low_storage = StorageCheck(False, "emergency_stop", 11.2, 20.0, 11.18, 15.0, "Free disk below 15 GB emergency reserve.")
    plan_disk = planner.plan_execution(req, storage_check=low_storage)
    assert plan_disk.admitted is False
    assert "15 GB" in plan_disk.admission_reason

    hot_snap = ResourceGuard().evaluate(
        host=HostMemoryStats(ram_total_mb=64000, ram_available_mb=1500, ram_available_pct=2.3, swap_total_mb=8192, swap_used_mb=7800, swap_used_pct=95.2, psi_some_avg10=30.0),
        gpus=[GPUTelemetry(0, "RTX 4070", 7800, 8188, 388, temp_c=86, util_pct=99)],
        storage=StorageStatus("/", 1000.0, 50.0, "normal"),
        now=50.0,
    )
    plan_mem = planner.plan_execution(req, snapshot=hot_snap, storage_check=StorageCheck(True, "normal", 50.0, 10.0, 49.99, 15.0))
    assert plan_mem.admitted is False

    # Invalid VLM JSON & timeout gracefully degrades to uncertain/advisory (NEVER falsely claims "Requirements verified"!)
    img = _make_png(tmp_path / "sample.png", (896, 1280))
    rec = GenerationVerificationRecord(
        original_prompt="Three glass apples on a slate table",
        detected_intent="image_generate",
        compiled_prompt="Three glass apples on a slate table",
        selected_model_id="flux2_9b",
        selected_workflow="builtin:flux2_txt2img",
        expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=896, height=1280),
        mandatory_acceptance_criteria=[
            AcceptanceCriterion(criterion_id="qty_apples", category="quantity", description="Exactly 3 glass apples", mandatory=True)
        ],
    )

    from avatar_v2 import visual_inspector as vi_mod

    fake_vlm = LocalLLMInfo(
        name="qwen3-vl:8b-instruct",
        size_bytes=6_000_000_000,
        size_gb=5.6,
        family="qwen3vl",
        parameter_size="8B",
        quantization="Q4_K_M",
        roles=("vision",),
    )
    monkeypatch.setattr(vi_mod, "inspect_local_ollama_models", lambda *a, **kw: [fake_vlm])
    monkeypatch.setattr(
        LocalOllamaClient,
        "complete_vision_json",
        lambda self, **kw: LLMCallResult(ok=False, model="qwen3-vl:8b-instruct", mode="gpu", latency_ms=12.0, error="invalid JSON from VLM"),
    )
    inspector = VisualInspector(ollama_url="http://127.0.0.1:11434")
    res_bad_json = inspector.inspect_image(img, rec, use_vlm=True, comfy_models_warm=False, allow_sequential_gpu=True)
    ui_bad = res_bad_json.compact_ui_summary()
    assert ui_bad["ui_status"] != "requirements_verified"
    assert ui_bad["ui_status_label"] != "Requirements verified"

    monkeypatch.setattr(
        LocalOllamaClient,
        "complete_vision_json",
        lambda self, **kw: LLMCallResult(ok=False, model="qwen3-vl:8b-instruct", mode="gpu", latency_ms=15.0, error="ReadTimeout"),
    )
    res_timeout = inspector.inspect_image(img, rec, use_vlm=True, comfy_models_warm=False, allow_sequential_gpu=True)
    ui_to = res_timeout.compact_ui_summary()
    assert ui_to["ui_status"] != "requirements_verified"
    assert ui_to["ui_status_label"] != "Requirements verified"

    # Correction cancellation mid-loop stops cleanly
    cancel_ev = threading.Event()
    cancel_ev.set()
    coord = CorrectionCoordinator(planner=planner)
    fail_insp = VisualInspectionResult(
        inspector_model="qwen3-vl:8b-instruct",
        execution_mode="hybrid",
        overall_decision="needs_correction",
        ui_status="correction_recommended",
        criterion_assessments=[
            CriterionAssessment(
                criterion_id="qty_apples",
                category="object_quantities",
                original_requirement="Exactly 3 glass apples",
                status="failed",
                confidence=0.85,
                assessor="local_vlm",
                inspection_method="local_vlm",
            )
        ],
    )
    sess = coord.run_bounded_correction(
        record=rec,
        original_output_id="orig_1",
        original_output_path=str(img),
        initial_inspection=fail_insp,
        generate_fn=lambda step: (_ for _ in ()).throw(AssertionError("Must not generate when already cancelled!")),
        inspect_fn=lambda p, r: fail_insp,
        cancel_event=cancel_ev,
    )
    assert sess.attempts_executed == 0
    assert sess.ui_status != "requirements_verified"

    # Defect #10 guard: uncorroborated requires_human_review category MUST NOT display "Requirements verified"!
    from avatar_v2.inspection_contract import compute_inspection_status

    uncorroborated_spatial = [
        CriterionAssessment(
            criterion_id="sp_1",
            category="spatial_relationships",
            original_requirement="green vase on the left of brass lamp",
            mandatory=True,
            verification_eligibility="requires_human_review",
            independent_corroboration=False,
            status="passed",
            assessor="local_vlm",
            inspection_method="local_vlm",
        )
    ]
    dec, ui_st, ui_lbl, hum = compute_inspection_status(uncorroborated_spatial)
    assert ui_st == "review_recommended"
    assert ui_lbl != "Requirements verified"
    assert hum is True
