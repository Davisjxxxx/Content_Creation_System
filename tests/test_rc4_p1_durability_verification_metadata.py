"""Adversarial RC4 P1/P2 tests covering Astra RC3 findings F03–F13.

Exercises Layer A (boundary invariants) and Layer B (black-box public API, real Node.js execution
of shipped ui/index.html, storage fault injection, live HTTP/process fixtures, and restart recovery).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
from PIL import Image
import pytest

import avatar_v2.desktop as desktop
import avatar_v2.local_engine as local_engine
import avatar_v2.offline_builder as ob
import avatar_v2.ollama_client as ollama_client
import avatar_v2.render_queue as rq
import avatar_v2.script_broker as sb
import avatar_v2.script_runner as sr
from avatar_v2.composer import build_job
from avatar_v2.gpu_memory import GPUMemoryManager
from avatar_v2.inspection_contract import (
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    VisualInspectionResult,
    compute_inspection_status,
    compute_requirement_set_hash,
)
from avatar_v2.intent_interpreter import interpret_natural_language
from avatar_v2.js_bridge import JsBridge
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from avatar_v2.providers.comfyui import ComfyUIProvider
from avatar_v2.reference_picker import LocalReferenceManager


def _make_png(path: Path, color: tuple[int, int, int] = (100, 140, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=color).save(path, format="PNG")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_job(job_id: str = "rc4-p1-1", *, seed: int = 41001) -> RenderJob:
    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "av-1",
            "display_name": "Test",
            "subjects": [{"id": "av-1", "kind": "synthetic"}],
            "identity_refs": [],
        }
    )
    shot = ShotSpec.model_validate(
        {"shot_id": "sh-1", "user_prompt": "A woman smiling warmly in studio light", "seed": seed}
    )
    job = build_job(avatar, shot)
    job.job_id = job_id
    return job


def _isolated_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[desktop.DesktopAPI, Path]:
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    return desktop.DesktopAPI(), app_home


def _stub_local_models_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local_engine,
        "inspect",
        lambda settings: {
            "installed": True,
            "missing_models": [],
            "comfyui_dir": "/tmp/fake_comfy",
            "workflow_present": True,
            "video_models": {
                "a14b": {"id": "a14b", "label": "Wan 2.2 A14B", "minutes": 6, "missing_models": [], "installed": True},
                "5b": {"id": "5b", "label": "Wan 2.2 5B", "minutes": 5, "missing_models": [], "installed": True},
            },
        },
    )
    monkeypatch.setattr(
        local_engine,
        "video_model_info",
        lambda settings: {
            "a14b": {"id": "a14b", "label": "Wan 2.2 A14B", "minutes": 6, "missing_models": [], "installed": True},
            "5b": {"id": "5b", "label": "Wan 2.2 5B", "minutes": 5, "missing_models": [], "installed": True},
        },
    )
    monkeypatch.setattr(
        ob,
        "inspect",
        lambda settings: {
            "installed": True,
            "models": {
                "4b": {"id": "4b", "label": "FLUX.2 4B", "missing_models": [], "installed": True},
                "9b": {"id": "9b", "label": "FLUX.2 9B", "missing_models": [], "installed": True},
            },
        },
    )


# ===========================================================================
# F03 (RC-03, RC-06): Deferred photo-first scene end_image auth, staging & refusal
# ===========================================================================


def test_f03_photo_first_scene_stages_end_image_at_submit_and_refuses_unselected_bridge_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra F03:
    1. Via JsBridge, an unselected external end_image MUST be refused at forge_go submit time!
    2. When an authorized end_image is moved/deleted BEFORE continuation runs, the staged copy
       survives and LAST_FRAME is still passed to Wan (NEVER silently dropped!).
    3. If the staged end_image itself disappears before continuation, continuation REFUSES instead
       of silently dropping LAST_FRAME!
    """
    _stub_local_models_installed(monkeypatch)
    api, app_home = _isolated_api(monkeypatch, tmp_path)
    try:
        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "ready", "message": "ready"})
        unselected_end = _make_png(tmp_path / "unselected_end.png", (200, 10, 10))
        bridge = JsBridge(api)

        # 1. Unselected external end_image rejected through JsBridge at submit time!
        res_unauth = bridge.forge_go(
            {
                "action": "scene",
                "prompt": "She walks across the sunlit studio.",
                "end_image": str(unselected_end),
            }
        )
        assert res_unauth["ok"] is False
        assert "not selected" in res_unauth["error"].lower() or "browse" in res_unauth["error"].lower()

        # 2. Authorized end_image staged & pinned at submit time; moving original preserves LAST_FRAME!
        auth_end = _make_png(tmp_path / "authorized_end.png", (30, 190, 60))
        api._reference_manager().register_selected([auth_end], workflow="image_to_video")
        captured_items: list[rq.QueuedRender] = []
        monkeypatch.setattr(api.queue, "add_many", lambda items: (captured_items.extend(items), [it.id for it in items])[1])

        res_ok = bridge.forge_go(
            {
                "action": "scene",
                "prompt": "She walks across the sunlit studio.",
                "end_image": str(auth_end),
            }
        )
        assert res_ok["ok"] is True, res_ok
        photo_item = captured_items[0]
        # Original end_image moves away BEFORE photo finishes!
        moved_end = tmp_path / "moved_elsewhere.png"
        auth_end.rename(moved_end)

        # Photo renders a starter frame; now continuation runs!
        start_photo = _make_png(tmp_path / "rendered_start.png", (80, 90, 210))
        photo_item.outputs = [str(start_photo)]
        captured_items.clear()
        cont = api._forge_continue_scene(photo_item)
        assert cont["ok"] is True, cont
        assert len(captured_items) == 1
        assert captured_items[0].job.asset_map.get("LAST_FRAME"), "LAST_FRAME was dropped after original moved!"
        assert Path(captured_items[0].job.asset_map["LAST_FRAME"]).is_file()

        # 3. If the staged end_image itself is removed before continuation -> explicit refusal (NEVER silent drop!)
        staged_p = Path(photo_item.runtime["forge_scene"]["staged_end_image"])
        staged_p.unlink()
        captured_items.clear()
        cont_missing = api._forge_continue_scene(photo_item)
        assert cont_missing["ok"] is False
        assert "end picture" in cont_missing["scene"].lower()
        assert captured_items == []
    finally:
        api.queue.shutdown()


# ===========================================================================
# F04 (RC-07): Single sentence combining count + mandatory object state
# ===========================================================================


def test_f04_single_sentence_count_and_object_state_both_preserved() -> None:
    """Reproduces Astra's exact F04 counterexample verbatim plus adjacent compound clauses."""
    res = interpret_natural_language(
        "Exactly two cups and the wooden box must remain closed.",
        requested_action="photo",
        use_llm=False,
    )
    descs = [r["description"].lower() for r in res.mandatory_requirements]
    assert any("2 cups" in d for d in descs), f"Missing cup count: {descs}"
    assert any("wooden box" in d and "closed" in d for d in descs), (
        f"Closed wooden box state hidden by count in same sentence: {descs}"
    )

    # Adjacent compound clause with 'while' and non-listed verb
    res2 = interpret_natural_language(
        "Three blue lanterns on the desk while the brass clock stays unlit.",
        requested_action="photo",
        use_llm=False,
    )
    descs2 = [r["description"].lower() for r in res2.mandatory_requirements]
    assert any("3 blue lanterns" in d for d in descs2), descs2
    assert any("brass clock" in d and "unlit" in d for d in descs2), descs2


# ===========================================================================
# F05 (RC-08, RC-09): Mandatory hybrid corroboration & live byte-hash revalidation
# ===========================================================================


def test_f05_hybrid_corroborated_without_independent_evidence_cannot_verify() -> None:
    """Reproduces Astra F05 part A: mandatory hybrid_corroborated with independent_corroboration=False
    MUST NOT return requirements_verified.
    """
    a = CriterionAssessment(
        criterion_id="crit_1",
        category="complex_nl",
        original_requirement="Two ceramic cups and a closed wooden box",
        mandatory=True,
        verification_eligibility="hybrid_corroborated",
        inspection_method="local_vlm",
        independent_corroboration=False,
        status="passed",
        assessor="local_vlm",
    )
    decision, ui_status, label, needs_human = compute_inspection_status([a])
    assert ui_status != "requirements_verified"
    assert label != "Requirements verified"
    assert needs_human is True

    # Once corroborated, it CAN verify
    a_corr = a.model_copy(update={"independent_corroboration": True})
    decision2, ui_status2, label2, needs_human2 = compute_inspection_status([a_corr])
    assert ui_status2 == "requirements_verified" and label2 == "Requirements verified" and needs_human2 is False


def test_f05_replaced_output_file_and_changed_requirement_set_invalidate_verdict_in_api_and_ui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra F05 part B end-to-end:
    1. Save history entry with verified output_sha256 + requirement_set_hash.
    2. Reopen via DesktopAPI._studio_results() & history() -> valid 'Requirements verified'.
    3. Replace the image file at the EXACT SAME PATH with different bytes -> MUST invalidate to 'Not inspected'!
    4. Pass reopened result to actual shipped ui/index.html JS in Node.js -> displays 'Not inspected' (NEVER 'Requirements verified'!).
    """
    api, app_home = _isolated_api(monkeypatch, tmp_path)
    try:
        out_img = _make_png(app_home / "archive" / "job-f05" / "still.png", (10, 120, 200))
        orig_sha = _sha256(out_img)
        vr = GenerationVerificationRecord(
            verification_id="ver-f05",
            detected_intent="image_generate",
            original_prompt="Two ceramic cups",
            compiled_prompt="Two ceramic cups",
            selected_model_id="flux2_9b",
            selected_workflow="builtin:flux2_txt2img",
            expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=256, height=256, frames=1),
        )
        from avatar_v2.inspection_contract import AcceptanceCriterion
        vr.mandatory_acceptance_criteria = [
            AcceptanceCriterion(criterion_id="count", category="object_quantities", description="Two cups")
        ]
        req_hash = compute_requirement_set_hash(vr)
        vr.inspection_result = VisualInspectionResult(
            output_path=str(out_img), output_sha256=orig_sha, requirement_set_hash=req_hash,
            criterion_assessments=[CriterionAssessment(
                criterion_id="count", category="object_quantities", mandatory=True, status="passed",
                assessor="human", inspection_method="human", verification_eligibility="requires_human_review",
                observed_evidence="Synthetic positive-control assessment for two cups",
            )],
        )
        hist_record = {
            "id": "job-f05",
            "label": "Test photo",
            "provider": "comfyui",
            "studio_action": "forge_photo",
            "status": "done",
            "phase": "complete",
            "finished_at": time.time(),
            "outputs": [str(out_img)],
            "output_hashes": {str(out_img): orig_sha},
            "verification_record": vr.public(),
            "candidate_qa": {
                "ok": True,
                "decision": "pass",
                "inspection": {
                    "ui_status": "requirements_verified",
                    "ui_status_label": "Requirements verified",
                    "output_sha256": orig_sha,
                    "requirement_set_hash": req_hash,
                    "inspected": True,
                },
                "output_inspections": {
                    str(out_img): {
                        "output_path": str(out_img),
                        "output_sha256": orig_sha,
                        "requirement_set_hash": req_hash,
                        "inspected": True,
                        "ui_status": "requirements_verified",
                        "ui_status_label": "Requirements verified",
                        "is_best": True,
                    }
                },
                "correction": {"ui_status": "requirements_verified", "ui_status_label": "Requirements verified"},
            },
        }
        hist_path = app_home / "history" / "job-f05.json"
        hist_path.write_text(json.dumps(hist_record, indent=2), encoding="utf-8")

        # Healthy before replacement
        before = api._studio_results(limit=10)[0]
        assert before["output_inspections"][str(out_img)]["ui_status_label"] == "Requirements verified"

        # Now REPLACE the picture at the exact same path with different bytes!
        _make_png(out_img, (240, 30, 30))
        assert _sha256(out_img) != orig_sha

        after = api._studio_results(limit=10)[0]
        per = after["output_inspections"][str(out_img)]
        assert per["inspected"] is False and per["stale"] is True
        assert per["ui_status_label"] == "Not inspected"
        assert after["inspection"]["ui_status_label"] == "Not inspected"
        assert after["correction"] is None

        # Also verify history() revalidates!
        hist_after = api.history()["items"][0]
        assert hist_after["candidate_qa"]["output_inspections"][str(out_img)]["ui_status_label"] == "Not inspected"

        # Execute ACTUAL shipped ui/index.html _resolveOutputInspection() in Node.js on reopened result!
        html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
        js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.DOTALL))
        harness = f"""
const vm = require('vm');
const makeEl = () => ({{ value:'', checked:true, disabled:false, innerHTML:'', textContent:'', style:{{}}, dataset:{{}}, classList:{{contains:()=>false,add:()=>{{}},remove:()=>{{}}}}, options:[], appendChild:()=>{{}}, querySelector:()=>null, querySelectorAll:()=>[], addEventListener:()=>{{}} }});
const sandbox = {{ console, Map, Set, Array, Object, Date, Math, JSON, Promise, Number, String, Boolean, RegExp, setTimeout:()=>0, clearTimeout:()=>{{}}, setInterval:()=>0, IntersectionObserver:class{{observe(){{}}unobserve(){{}}disconnect(){{}}}}, document:{{getElementById:()=>makeEl(),querySelector:()=>null,querySelectorAll:()=>[],createElement:()=>makeEl(),addEventListener:()=>{{}}}}, window:{{addEventListener:()=>{{}},pywebview:{{api:{{}}}}}} }};
vm.createContext(sandbox);
const safeSrc = {json.dumps(js)}.replace(/window\\.addEventListener\\('pywebviewready'[\\s\\S]*$/, '');
const rObj = {json.dumps(after)};
const outPath = {json.dumps(str(out_img))};
vm.runInContext(safeSrc + `
  const res = _resolveOutputInspection(${{JSON.stringify(rObj)}}, ${{JSON.stringify(outPath)}});
  console.log(JSON.stringify(res));
`, sandbox);
"""
        proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=True)
        ui_res = json.loads(proc.stdout.strip().splitlines()[-1])
        assert ui_res["statusLbl"] == "Not inspected", ui_res
        assert ui_res["cor"] is None

        # Also verify requirement-set hash mismatch invalidates even when file bytes match!
        _make_png(out_img, (10, 120, 200))  # restore original bytes
        assert _sha256(out_img) == orig_sha
        hist_record["candidate_qa"]["output_inspections"][str(out_img)]["requirement_set_hash"] = "0000000000000000"
        hist_path.write_text(json.dumps(hist_record, indent=2), encoding="utf-8")
        after_req = api._studio_results(limit=10)[0]
        assert after_req["output_inspections"][str(out_img)]["stale_reason"] == "requirement_set_changed"
        assert after_req["output_inspections"][str(out_img)]["ui_status_label"] == "Not inspected"
    finally:
        api.queue.shutdown()


# ===========================================================================
# F06, F07, F08, F11 (RC-10, RC-11, RC-16, RC-18): Authoritative Durable Queue State Machine
# ===========================================================================


def test_f06_rejected_queue_admission_never_executes_and_timeout_never_replays_acknowledged_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces BOTH Astra F06 counterexamples:
    1. One-shot storage OSError during RenderQueue.add() raises to caller AND worker NEVER executes item!
    2. Timeout in _handle_failure for a rendering item with prompt_id + submission_token MUST NOT
       requeue, MUST NOT clear prompt_id, and MUST enter recovery_required!
    """
    rendered: list[str] = []

    class SpyComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig, **kwargs) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            rendered.append(job.job_id)
            return {"prompt_id": "should-never-run", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", SpyComfy)
    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        # 1. One-shot storage error during add()
        real_write = rq.write_durable
        calls = {"n": 0}

        def _fail_first(path: Path, text: str) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("simulated one-shot disk error on enqueue")
            return real_write(path, text)

        monkeypatch.setattr(rq, "write_durable", _fail_first)
        with pytest.raises(OSError):
            q.add(
                rq.QueuedRender(
                    id="job-rejected-add",
                    job=_make_job("job-rejected-add"),
                    workflow="builtin:flux2_txt2img",
                    provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                    profile="auto",
                )
            )
        monkeypatch.setattr(rq, "write_durable", real_write)
        time.sleep(0.3)
        assert q.list() == [], "Rejected job remained in memory!"
        assert rendered == [], f"Worker executed a rejected job: {rendered}"

        # 2. Timeout after prompt acknowledgement MUST NOT clear prompt_id or requeue!
        ack_item = rq.QueuedRender(
            id="job-ack-timeout",
            job=_make_job("job-ack-timeout"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="auto",
            status="running",
            phase="rendering",
            prompt_id="comfy-ack-999",
            runtime={"submission_token": "tok-abc"},
        )
        q._handle_failure(ack_item, TimeoutError("ComfyUI job comfy-ack-999 exceeded 30s"))
        assert ack_item.status == "error"
        assert ack_item.phase == "recovery_required"
        assert ack_item.prompt_id == "comfy-ack-999", "Timeout erased acknowledged prompt_id!"
        assert int(ack_item.runtime.get("recover_attempts") or 0) == 0
    finally:
        q.shutdown()


def test_f07_restart_never_overwrites_committed_done_history_with_stale_active_checkpoint(
    tmp_path: Path,
) -> None:
    """Reproduces Astra's exact F07 counterexample:
    Committed history/<id>.json has status='done', phase='complete', outputs=[file].
    Stale active items/<id>.json still exists on disk (crash before checkpoint retirement).
    Reopening RenderQueue MUST keep the committed done history intact and retire the stale checkpoint!
    """
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    (work_dir / "items").mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    out_img = _make_png(tmp_path / "done_out.png")

    committed_hist = {
        "id": "job-f07-done",
        "job_id": "job-f07-done",
        "status": "done",
        "phase": "complete",
        "outputs": [str(out_img)],
        "state_revision": 5,
    }
    hist_path = history_dir / "job-f07-done.json"
    hist_path.write_text(json.dumps(committed_hist, indent=2), encoding="utf-8")

    # Stale active checkpoint left behind before unlink
    stale_item = {
        "id": "job-f07-done",
        "job": _make_job("job-f07-done").model_dump_json(),
        "workflow": "builtin:flux2_txt2img",
        "provider_config": ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188").model_dump(),
        "profile": "auto",
        "status": "running",
        "phase": "submitted",
        "outputs": [],
        "state_revision": 2,
    }
    stale_path = work_dir / "items" / "job-f07-done.json"
    stale_path.write_text(json.dumps(stale_item, indent=2), encoding="utf-8")

    q = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        after_hist = json.loads(hist_path.read_text(encoding="utf-8"))
        assert after_hist["status"] == "done"
        assert after_hist["phase"] == "complete"
        assert after_hist["outputs"] == [str(out_img)]
        assert not stale_path.exists(), "Stale checkpoint was not retired idempotently!"
    finally:
        q.shutdown()


def test_f08_ack_persist_failure_halts_into_recovery_and_candidate_persisted_before_reinspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces BOTH Astra F08 counterexamples:
    1. Failure persisting acknowledged prompt_id raises instead of silently diverging disk vs memory.
    2. Corrective candidate path & initial inspection result are durably written to items/<id>.json
       BEFORE reinspection begins!
    """
    api, app_home = _isolated_api(monkeypatch, tmp_path)
    try:
        # Part 1: Failed prompt_id acknowledgement write raises and sets recovery_required
        it1 = rq.QueuedRender(
            id="job-f08-ack",
            job=_make_job("job-f08-ack"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="auto",
            status="running",
            phase="submitted",
        )
        monkeypatch.setattr(api.queue, "_persist_item_durable", lambda item: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(RuntimeError, match="persist acknowledged"):
            api.queue._record_local_submitted(it1, "pid-ack-555")
        assert it1.phase == "recovery_required"
        assert it1.prompt_id == "pid-ack-555"
    finally:
        api.queue.shutdown()

    # Part 2: Inspect durable checkpoint on disk AT THE MOMENT reinspection starts!
    api2, app_home2 = _isolated_api(monkeypatch, tmp_path / "run2")
    try:
        orig_img = _make_png(tmp_path / "orig.png", (90, 90, 90))
        cand_img = _make_png(tmp_path / "cand.png", (200, 180, 50))
        vr = GenerationVerificationRecord(
            verification_id="ver-f08",
            detected_intent="image_generate",
            original_prompt="A red mug on a table",
            compiled_prompt="A red mug on a table",
            selected_model_id="flux2_9b",
            selected_workflow="builtin:flux2_txt2img",
            expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=256, height=256, frames=1),
        )
        fail_ins = VisualInspectionResult(
            output_path=str(orig_img),
            overall_decision="needs_correction",
            criterion_assessments=[
                CriterionAssessment(
                    criterion_id="c1",
                    category="colors",
                    original_requirement="Red mug",
                    mandatory=True,
                    status="failed",
                    observed_evidence="Mug appears gray",
                )
            ],
        )
        pass_ins = VisualInspectionResult(
            output_path=str(cand_img),
            overall_decision="pass",
            criterion_assessments=[
                CriterionAssessment(
                    criterion_id="c1",
                    category="colors",
                    original_requirement="Red mug",
                    mandatory=True,
                    status="passed",
                    independent_corroboration=True,
                )
            ],
        )
        disk_snapshot_at_reinspect: dict[str, Any] = {}
        call_n = {"n": 0}

        class SpyInspector:
            def __init__(self, **kwargs) -> None:
                pass

            def inspect_image(self, target, record, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    return fail_ins
                # Reinspection call! Read items/job-f08-cor.json RIGHT NOW from disk!
                ckpt_p = app_home2 / "queue" / "items" / "job-f08-cor.json"
                disk_snapshot_at_reinspect.update(json.loads(ckpt_p.read_text(encoding="utf-8")))
                return pass_ins

        import avatar_v2.visual_inspector as vi

        monkeypatch.setattr(vi, "VisualInspector", SpyInspector)
        it2 = rq.QueuedRender(
            id="job-f08-cor",
            job=_make_job("job-f08-cor"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="auto",
            status="running",
            phase="inspecting",
            outputs=[str(orig_img)],
            runtime={
                "verification_record": vr.public(),
                "visual_inspection_vlm": False,
                "auto_correct": True,
                "_corrective_generate_hook": lambda step: (str(cand_img), 10.0),
            },
        )
        api2._run_visual_inspection(it2)
        assert str(cand_img) in disk_snapshot_at_reinspect.get("outputs", []), (
            f"Candidate path was NOT durable on disk before reinspection began: {disk_snapshot_at_reinspect}"
        )
        assert isinstance(
            (disk_snapshot_at_reinspect.get("runtime") or {}).get("verification_record", {}).get("inspection_result"),
            dict,
        ), "Original inspection result was NOT durable on disk before reinspection began!"
    finally:
        api2.queue.shutdown()


def test_f11_completed_provider_with_zero_outputs_fails_instead_of_done_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra F11: ComfyUI returning completed with empty outputs=[] MUST NOT produce done/complete!"""

    class EmptyComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig, **kwargs) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, *, on_submitted=None, **kwargs) -> dict[str, Any]:
            if on_submitted:
                on_submitted("pid-empty-out")
            return {"prompt_id": "pid-empty-out", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", EmptyComfy)
    monkeypatch.setattr(ollama_client, "fetch_ollama_ps_state", lambda *a, **k: ("offline", [], "offline"))

    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        q.add(
            rq.QueuedRender(
                id="job-f11-empty",
                job=_make_job("job-f11-empty"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == "job-f11-empty"), None)
            if cur and cur["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.05)
        cur = next(x for x in q.list() if x["id"] == "job-f11-empty")
        assert cur["status"] == "error", f"Empty-output render improperly marked {cur['status']}"
        assert cur["phase"] != "complete"
    finally:
        q.shutdown()


# ===========================================================================
# F09 (RC-13): WanFirstLastFrameToVideo resolved-graph geometry extraction
# ===========================================================================


def test_f09_wan_first_last_frame_graph_extracts_exact_480x832_81_frames() -> None:
    """Reproduces Astra's exact F09 counterexample:
    WanFirstLastFrameToVideo node requesting 480x832, 81 frames MUST extract width=480, height=832, frames=81
    (NOT fallback 720x1280, 80 frames!).
    """
    resolved_flf = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf"}},
        "12": {
            "class_type": "WanFirstLastFrameToVideo",
            "inputs": {"width": 480, "height": 832, "length": 81, "batch_size": 1},
        },
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {"add_noise": "enable", "noise_seed": 777123456, "steps": 4, "cfg": 1.0},
        },
    }
    extracted = rq._extract_resolved_workflow_params(resolved_flf)
    assert extracted["width"] == 480
    assert extracted["height"] == 832
    assert extracted["frames"] == 81
    assert extracted["seed"] == 777123456


# ===========================================================================
# F10 (RC-14): Unknown Ollama state fails closed; unified GPUMemoryManager event contract
# ===========================================================================


def test_f10_timed_out_ollama_status_blocks_comfyui_and_production_cleanup_event_clears_warm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces BOTH sides of Astra F10:
    1. Timed-out Ollama /api/ps returns status='unknown' and BLOCKS ComfyUI dispatch!
    2. Production GPUMemoryManager.maybe_cleanup() emits cleanup_performed + effective + vram_released_verified,
       which DesktopAPI recognizes cleanly!
    """
    dispatched: list[str] = []

    class SpyComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig, **kwargs) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            dispatched.append(job.job_id)
            return {"prompt_id": "nope", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", SpyComfy)
    monkeypatch.setattr(
        ollama_client,
        "fetch_ollama_ps_state",
        lambda *a, **k: ("unknown", [], "ReadTimeout: timed out"),
    )

    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        q.add(
            rq.QueuedRender(
                id="job-f10-timeout",
                job=_make_job("job-f10-timeout"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == "job-f10-timeout"), None)
            if cur and cur["status"] in {"done", "error"}:
                break
            time.sleep(0.05)
        cur = next(x for x in q.list() if x["id"] == "job-f10-timeout")
        assert cur["status"] == "error"
        assert "unknown" in cur["error"].lower()
        assert dispatched == [], "ComfyUI was dispatched despite unknown/timed-out Ollama status!"
    finally:
        q.shutdown()

    # Part 2: Production GPUMemoryManager event contract emits cleanup_performed=True + effective=True
    class OkFreeProvider:
        base_url = "http://127.0.0.1:18189"

        def free_memory(self, **kwargs) -> None:
            pass

    from avatar_v2.gpu_memory import GPUSnapshot

    snaps = iter(
        [
            [GPUSnapshot(0, "RTX 4070", 6000, 8192)],
            [GPUSnapshot(0, "RTX 4070", 1200, 8192)],
            [GPUSnapshot(0, "RTX 4070", 1200, 8192)],
        ]
    )
    monkeypatch.setattr("avatar_v2.gpu_memory.query_gpu_memory", lambda: next(snaps, [GPUSnapshot(0, "RTX 4070", 1200, 8192)]))
    monkeypatch.setattr("avatar_v2.gpu_memory.time.sleep", lambda _: None)
    mgr = GPUMemoryManager(enabled=True)
    ev = mgr.maybe_cleanup(OkFreeProvider(), reason="pre-vlm-unload", force=True, unload_models=True, wait_s=2.0)
    assert ev["cleanup_requested"] is True
    assert ev["cleanup_performed"] is True
    assert ev["effective"] is True
    assert ev["vram_released_verified"] is True


# ===========================================================================
# F12 (RC-17): Unavailable process telemetry fails closed and stops admission
# ===========================================================================


def test_f12_unavailable_proc_telemetry_fails_closed_and_stops_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra F12: When /proc inventory is unreadable, collector returns telemetry_ok=False
    and _watch_process stops broker admission and terminates the run fail-closed!
    """
    monkeypatch.setattr(os, "listdir", lambda p: (_ for _ in ()).throw(PermissionError("proc denied")) if p == "/proc" else [])
    ok, pids, rss, err = sr.ScriptRunManager._collect_descendant_tree_stats(os.getpid(), os.getpgrp())
    assert ok is False
    assert "proc_list_unreadable" in err

    # Verify _watch_process stops broker & terminates child when telemetry is lost!
    monkeypatch.undo()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    broker = sb.GenerationBroker(
        run_id="run-f12-tel",
        submit_fn=lambda ctx: "never",
        status_fn=lambda jid: {"status": "done"},
        max_items=2,
    )
    rec = sr.ScriptRunRecord(
        run_id="run-f12-tel",
        script_path=str(tmp_path / "s.py"),
        script_name="s.py",
        script_sha256="a" * 64,
        classification="native_avatar_v2",
        execution_mode="sandbox",
        sandbox_mode="test",
        max_items=2,
        extended_authorized=False,
        started_at="2026-11-27T00:00:00Z",
        status="running",
    )
    mgr = sr.ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=lambda ctx: "never",
        status_fn=lambda jid: {"status": "done"},
    )
    monkeypatch.setattr(
        sr.ScriptRunManager,
        "_collect_descendant_tree_stats",
        staticmethod(lambda root_pid, root_pgid: (False, set(), 0, "proc_list_unreadable")),
    )
    temp_root = tmp_path / "tmp_run"
    temp_root.mkdir()
    try:
        mgr._watch_process(proc, broker, rec, temp_root, wall_timeout_s=10.0, max_processes=8, memory_limit_mb=512)
        assert rec.status == "error"
        assert "telemetry" in rec.error.lower()
        assert broker.state == "stopped"
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ===========================================================================
# F13 (RC-19): Failed pin persistence refuses admission; queued cancellation releases pins
# ===========================================================================


def test_f13_failed_pin_persistence_refuses_admission_and_queued_cancel_releases_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces BOTH Astra F13 counterexamples:
    1. When replacing .pins.json raises OSError, pin_staged raises (rolling back memory) and forge_go refuses admission!
    2. Cancelling a QUEUED job releases its pin from .pins.json!
    """
    staging_dir = tmp_path / "staged"
    mgr = LocalReferenceManager(staging_dir=staging_dir, max_staged_files=8)
    img = _make_png(tmp_path / "ref.png", (70, 130, 190))
    staged = mgr.stage_for_job(img, workflow="multi_reference", require_authorization=False)
    staged_path = Path(staged["staged_path"])

    # 1. os.replace failure on .pins.json MUST raise OSError and leave no in-memory or on-disk pin
    orig_replace = os.replace

    def _boom_replace(src: Any, dst: Any) -> None:
        if str(dst).endswith(".pins.json"):
            raise OSError("simulated atomic replace failure on .pins.json")
        return orig_replace(src, dst)

    monkeypatch.setattr(os, "replace", _boom_replace)
    with pytest.raises(OSError):
        mgr.pin_staged(staged_path, "job-pin-fail")
    assert mgr._pinned == {}, "In-memory pin was not rolled back after failed .pins.json write!"
    monkeypatch.setattr(os, "replace", orig_replace)

    # 2. Queued job cancellation invokes on_job_finished -> releases pin from .pins.json!
    mgr.pin_staged(staged_path, "job-queued-cancel")
    assert "job-queued-cancel" in json.loads((staging_dir / ".pins.json").read_text(encoding="utf-8"))[str(staged_path.resolve())]

    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
        on_job_finished=lambda item: mgr.unpin_job(item.id),
    )
    try:
        # Hold retry_after in the future so job stays queued cleanly until cancel()
        q.add(
            rq.QueuedRender(
                id="job-queued-cancel",
                job=_make_job("job-queued-cancel"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
                runtime={"retry_after": time.time() + 3600},
            )
        )
        assert q.cancel("job-queued-cancel") is True
        pins_after = json.loads((staging_dir / ".pins.json").read_text(encoding="utf-8"))
        assert pins_after == {}, f"Queued cancellation leaked staged pin: {pins_after}"
    finally:
        q.shutdown()
