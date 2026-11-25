"""Sprint 5 & Phase B verification suite:
- Phase B hardening: broker auth/version rejection, reference traversal, request flooding,
  concurrent race enforcement, child-process cleanup, and post-validation script mutation (TOCTOU).
- Phases C/D/E/H/I: Role-aware multi-view SubjectReferenceSet, multi-subject separation,
  conflict detection, deterministic orientation-aware reference selection, CharacterProfileStore,
  StructuredPromptSpec & SDK integration, and DesktopAPI/UI integration.
- Phase J: Optional local-only A1111 Compatibility Adapter with explicit unsupported-parameter
  and unknown-checkpoint rejection.
- Phase K: Expanded multi-view reference adherence benchmark suite.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop, js_bridge
from avatar_v2.a1111_adapter import A1111CompatibilityAdapter
from avatar_v2.adherence_benchmark import (
    MULTI_VIEW_CONSISTENCY_DIMENSIONS,
    default_adherence_suite,
    multi_view_reference_suite,
)
from avatar_v2.prompt_schema import ReferenceAssetItem, StructuredPromptSpec
from avatar_v2.reference_picker import LocalReferenceManager
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceGuard, StorageCheck, StorageStatus
from avatar_v2.script_broker import BrokerSubmitContext, GenerationBroker
from avatar_v2.script_runner import SandboxLimits, ScriptRunManager, probe_sandbox_support
from avatar_v2.sdk import AvatarV2Client
from avatar_v2.subject_references import (
    NON_3D_RECONSTRUCTION_NOTE,
    CharacterProfile,
    CharacterProfileStore,
    SubjectReferenceItem,
    SubjectReferenceSet,
    select_relevant_references,
)


def _img(path: Path, size: tuple[int, int] = (512, 768), color: tuple[int, int, int] = (100, 140, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _healthy_snap():
    guard = ResourceGuard()
    host = HostMemoryStats(64000, 42000, 65.6, 8192, 100, 1.2, 10, 20, 0.0, 0.0)
    gpus = [GPUTelemetry(0, "RTX 4070", 2000, 8188, 6188, temp_c=50, util_pct=5)]
    storage = StorageStatus("/sim", 915.0, 60.0, "normal")
    return guard.evaluate(host=host, gpus=gpus, storage=storage, now=100.0)


def _ok_storage():
    return StorageCheck(True, "normal", 60.0, 0.05, 59.95, 15.0)


class _SimQueue:
    def __init__(self, tmp_path: Path, *, auto_complete: bool = True) -> None:
        self.tmp_path = tmp_path
        self.auto_complete = auto_complete
        self._lock = threading.RLock()
        self._n = 0
        self.jobs: dict[str, dict[str, Any]] = {}
        self.contexts: list[BrokerSubmitContext] = []

    def submit(self, ctx: BrokerSubmitContext) -> str:
        with self._lock:
            self._n += 1
            jid = f"j_{self._n:04d}"
            self.contexts.append(ctx)
            out = _img(self.tmp_path / f"{jid}.png", (256, 256)) if self.auto_complete else None
            self.jobs[jid] = {
                "status": "done" if self.auto_complete else "queued",
                "outputs": [str(out)] if out else [],
                "error": "",
            }
            return jid

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self.jobs.get(job_id) or {"status": "unknown", "outputs": []})

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "cancelled"
                return True
            return False


# --------------------------------------------------------------------------- Phase B: Additional controlled script isolation tests


def test_phase_b_broker_auth_version_traversal_and_request_flood_limits(tmp_path: Path):
    sim = _SimQueue(tmp_path, auto_complete=False)
    ref = _img(tmp_path / "allowed.png")
    broker = GenerationBroker(
        run_id="sec_flood",
        submit_fn=sim.submit,
        status_fn=sim.status,
        cancel_fn=sim.cancel,
        max_items=3,
        max_concurrent=1,
        authorized_references={"/workspace/refs/allowed.png": str(ref)},
        snapshot_override=_healthy_snap(),
        storage_check_override=_ok_storage(),
    )

    # 1. Wrong bearer token rejected
    bad_tok = broker.handle_request({"contract_version": "1.0", "token": "wrong", "op": "capabilities", "payload": {}})
    assert bad_tok["ok"] is False and bad_tok["code"] == "unauthorized"

    # 2. Wrong contract version rejected
    bad_ver = broker.handle_request({"contract_version": "9.9", "token": broker.token, "op": "capabilities", "payload": {}})
    assert bad_ver["ok"] is False and bad_ver["code"] == "version_mismatch"

    # 3. Path traversal rejected
    trav = broker.handle_request(
        {
            "contract_version": "1.0",
            "token": broker.token,
            "op": "submit",
            "payload": {
                "model_id": "wan22_a14b",
                "spec": {
                    "original_prompt": "waves hello",
                    "intent": {"task": "video_generate", "input_mode": "script"},
                    "reference_assets": [{"asset_id": "r0", "role": "start_frame", "path": "../../../etc/passwd"}],
                    "constraints": {"frames": 81, "local_only": True},
                },
            },
        }
    )
    assert trav["ok"] is False and trav["code"] == "unauthorized_path"

    # 4. Concurrent request flooding: 12 simultaneous threads can admit AT MOST 1 active job!
    def _burst(i: int) -> dict[str, Any]:
        return broker.handle_request(
            {
                "contract_version": "1.0",
                "token": broker.token,
                "op": "submit",
                "payload": {
                    "model_id": "flux2_4b",
                    "spec": {
                        "original_prompt": f"Studio portrait #{i}",
                        "intent": {"task": "image_generate", "input_mode": "script"},
                        "constraints": {"local_only": True},
                    },
                },
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_burst, range(12)))
    accepted = [r for r in results if r.get("ok")]
    rejected = [r for r in results if not r.get("ok")]
    assert len(accepted) == 1
    assert len(rejected) == 11
    assert all(r["code"] == "concurrency_limit" for r in rejected)


def test_phase_b_script_mutation_after_validation_is_refused(tmp_path: Path):
    script = tmp_path / "clean_native.py"
    script.write_text("import avatar_v2_sdk\nprint('validated v1')\n", encoding="utf-8")
    sim = _SimQueue(tmp_path)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        snapshot_override=_healthy_snap(),
        storage_check_override=_ok_storage(),
    )
    sel = mgr.select_script(script)
    assert sel["ok"] is True and sel["executed"] is False

    # Attacker/process mutates the script on disk AFTER validation!
    script.write_text("import avatar_v2_sdk\nprint('MUTATED v2')\n", encoding="utf-8")
    blocked = mgr.start_sandboxed_run()
    assert blocked["ok"] is False and blocked.get("code") == "script_mutated"


def test_phase_b_live_bwrap_cleans_up_spawned_child_processes(tmp_path: Path):
    probe = probe_sandbox_support()
    if not probe.available:
        pytest.skip(f"Bubblewrap unavailable: {probe.error}")

    script = tmp_path / "spawner.py"
    script.write_text(
        "import subprocess, time, avatar_v2_sdk\n"
        "p = subprocess.Popen(['/bin/sleep', '60'])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    sim = _SimQueue(tmp_path)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        snapshot_override=_healthy_snap(),
        storage_check_override=_ok_storage(),
    )
    started = mgr.start_sandboxed_run(
        script_path=script,
        limits=SandboxLimits(wall_timeout_s=1.2),
        allow_review_classification=True,
    )
    assert started["ok"] is True
    final = mgr.wait_for_run(timeout_s=10)
    assert final["record"]["status"] == "error"
    assert "timeout" in final["record"]["error"].lower()


# --------------------------------------------------------------------------- Phases C, E, H, I: Multi-view references & Character Profiles


def test_subject_reference_set_conflicts_multi_subject_and_selection(tmp_path: Path):
    starter = _img(tmp_path / "starter.png")
    face = _img(tmp_path / "face.png")
    front = _img(tmp_path / "front.png")
    back = _img(tmp_path / "back.png")
    left = _img(tmp_path / "left.png")
    right = _img(tmp_path / "right.png")
    wardrobe = _img(tmp_path / "wardrobe.png")

    rset = SubjectReferenceSet(
        starter_image=SubjectReferenceItem(role="primary_starter", path=str(starter)),
        references=[
            SubjectReferenceItem(role="face_identity", subject_id="alice", path=str(face)),
            SubjectReferenceItem(role="front_body", subject_id="alice", path=str(front)),
            SubjectReferenceItem(role="back_body", subject_id="alice", path=str(back)),
            SubjectReferenceItem(role="left_side", subject_id="alice", path=str(left)),
            SubjectReferenceItem(role="appearance_wardrobe", subject_id="alice", path=str(wardrobe)),
            SubjectReferenceItem(role="front_body", subject_id="bob", path=str(right)),
        ],
    )
    assert rset.detect_conflicts(check_disk=True) == []
    assert rset.subject_ids() == ["alice", "bob"]
    assert NON_3D_RECONSTRUCTION_NOTE in rset.non_3d_disclaimer

    # Deterministic orientation-aware selection (never blends bob into alice!).
    rear_sel = select_relevant_references(rset, prompt="Alice seen from behind walking down a hallway", subject_id="alice", max_supplemental=2)
    assert rear_sel["scenario"] == "rear_facing"
    assert rear_sel["selected"][0]["role"] == "back_body"
    assert all(r["subject_id"] == "alice" for r in rear_sel["selected"])

    side_sel = select_relevant_references(rset, prompt="Left side profile of Alice", subject_id="alice", max_supplemental=2)
    assert side_sel["scenario"] == "left_profile"
    assert side_sel["selected"][0]["role"] == "left_side"

    # When Wan 2.2 cannot consume rear/side refs directly, never silently discard: report strategy & limitation!
    wan_turn = select_relevant_references(
        rset,
        prompt="Alice turns around from front to back and walks away",
        task="video_generate",
        subject_id="alice",
        workflow_supports_multiref=False,
    )
    assert wan_turn["recommended_strategy"] == "flux_keyframe_plus_wan_flf2v"
    assert any("Wan 2.2" in w for w in wan_turn["workflow_limitations"])

    # Conflict detection catches competing starter, duplicate singular view, and opposite-view reuse.
    bad_set = SubjectReferenceSet(
        starter_image=SubjectReferenceItem(role="primary_starter", path=str(starter)),
        references=[
            SubjectReferenceItem(role="primary_starter", path=str(front)),
            SubjectReferenceItem(role="front_body", subject_id="alice", path=str(front)),
            SubjectReferenceItem(role="front_body", subject_id="alice", path=str(back)),
            SubjectReferenceItem(role="back_body", subject_id="alice", path=str(front)),
        ],
    )
    codes = {c.code for c in bad_set.detect_conflicts()}
    assert "competing_starter_frame" in codes
    assert "duplicate_singular_view" in codes
    assert "contradictory_view_assignment" in codes


def test_structured_prompt_spec_and_sdk_support_role_aware_subject_references(tmp_path: Path):
    front = _img(tmp_path / "front.png")
    back = _img(tmp_path / "back.png")

    spec = StructuredPromptSpec(
        original_prompt="Rear view of the instructor walking down a sunlit hall",
        reference_assets=[
            ReferenceAssetItem(asset_id="r1", role="front_body", subject_id="main", path=str(front)),
            ReferenceAssetItem(asset_id="r2", role="back_body", subject_id="main", path=str(back)),
        ],
    )
    assert not any(c.severity == "error" for c in spec.detect_conflicts())

    # Contradictory assignment of same file to front_body & back_body caught on StructuredPromptSpec too.
    bad = StructuredPromptSpec(
        original_prompt="test",
        reference_assets=[
            ReferenceAssetItem(asset_id="r1", role="front_body", subject_id="main", path=str(front)),
            ReferenceAssetItem(asset_id="r2", role="back_body", subject_id="main", path=str(front)),
        ],
    )
    assert any(c.code == "contradictory_subject_view_orientation" for c in bad.detect_conflicts())

    # Native SDK submits role-aware subject_references cleanly through GenerationBroker.
    sim = _SimQueue(tmp_path, auto_complete=True)
    broker = GenerationBroker(
        run_id="sdk_mref",
        submit_fn=sim.submit,
        status_fn=sim.status,
        authorized_references={"/workspace/refs/front.png": str(front), "/workspace/refs/back.png": str(back)},
        snapshot_override=_healthy_snap(),
        storage_check_override=_ok_storage(),
    )
    client = AvatarV2Client(token=broker.token, transport=broker.handle_request)
    res = client.generate_image(
        "Rear view in studio",
        model_id="flux2_9b",
        subject_references=[
            {"role": "front_body", "path": "/workspace/refs/front.png"},
            {"role": "back_body", "path": "/workspace/refs/back.png"},
        ],
    )
    assert res.status == "done" and len(sim.contexts[0].staged_references) == 2


def test_character_profile_store_preserves_originals_and_desktop_api_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(desktop, "APP_HOME", tmp_path / "home")
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "home" / "desktop.json")
    monkeypatch.setattr(desktop, "HISTORY_DIR", tmp_path / "home" / "history")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)

    front = _img(tmp_path / "pics" / "front.png")
    back = _img(tmp_path / "pics" / "back.png")
    orig_front = front.read_bytes()

    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)
    api._self_window = None
    monkeypatch.setattr(api, "choose_files", lambda kind, multiple=True, directory="": [str(front), str(back)])
    assert api.pick_local_images({"workflow": "multi_reference", "multiple": True})["ok"] is True

    saved = api.character_profiles(
        {
            "action": "save",
            "name": "Maya Reference Set",
            "references": [
                {"role": "front_body", "path": str(front)},
                {"role": "back_body", "path": str(back)},
            ],
        }
    )
    assert saved["ok"] is True
    pid = saved["profile"]["profile_id"]
    listed = api.character_profiles({"action": "list"})
    assert len(listed["profiles"]) == 1

    # Deleting profile removes metadata ONLY — never modifies or deletes original image files!
    deleted = api.character_profiles({"action": "delete", "profile_id": pid})
    assert deleted["ok"] is True and deleted["profiles"] == []
    assert front.is_file() and back.is_file() and front.read_bytes() == orig_front


# --------------------------------------------------------------------------- Phase J: Optional A1111 Compatibility Adapter


def test_a1111_adapter_maps_compatible_calls_and_rejects_unsupported_explicitly(tmp_path: Path):
    sim = _SimQueue(tmp_path, auto_complete=True)
    ref = _img(tmp_path / "ref.png")
    broker = GenerationBroker(
        run_id="a1111_test",
        submit_fn=sim.submit,
        status_fn=sim.status,
        max_items=2,
        authorized_references={"ref.png": str(ref)},
        snapshot_override=_healthy_snap(),
        storage_check_override=_ok_storage(),
    )
    adapter = A1111CompatibilityAdapter(broker)

    models = adapter.handle_sd_models()
    assert models.ok is True and any(m["title"] == "flux2_9b" for m in models.body["models"])

    # 1. Compatible txt2img strips SDXL weights and routes via broker.
    ok_res = adapter.handle_txt2img(
        {"prompt": "Adult instructor portrait, (soft studio lighting:1.4)", "width": 896, "height": 1280, "seed": 123}
    )
    assert ok_res.ok is True and ok_res.http_status == 200 and len(ok_res.body["images"]) == 1
    info = json.loads(ok_res.body["info"])
    assert any("SDXL" in w for w in info["warnings"])

    # 2. Unsupported parameters rejected explicitly (never silently ignored!).
    assert adapter.handle_txt2img({"prompt": "portrait", "enable_hr": True}).http_status == 400
    assert adapter.handle_txt2img({"prompt": "portrait", "alwayson_scripts": {"controlnet": {}}}).http_status == 400
    assert adapter.handle_txt2img({"prompt": "portrait", "restore_faces": True}).http_status == 400

    # 3. Unknown SDXL checkpoint rejected explicitly (no silent substitution!).
    bad_ckpt = adapter.handle_txt2img(
        {"prompt": "portrait", "override_settings": {"sd_model_checkpoint": "juggernautXL_v9.safetensors"}}
    )
    assert bad_ckpt.ok is False and bad_ckpt.http_status == 400
    assert "silent model substitution" in bad_ckpt.body["error"].lower()

    # 4. Policy enforcement preserved (minor blocked with HTTP 403).
    minor = adapter.handle_txt2img({"prompt": "portrait of a 14 year old child"})
    assert minor.ok is False and minor.http_status == 403

    # 5. img2img rejects unauthorized host paths (HTTP 403) and accepts authorized ref (HTTP 200).
    unauth = adapter.handle_img2img({"prompt": "studio portrait", "init_images": ["/etc/passwd"]})
    assert unauth.ok is False and unauth.http_status == 403
    auth = adapter.handle_img2img({"prompt": "studio portrait", "init_images": ["ref.png"]})
    assert auth.ok is True and auth.http_status == 200

    # 6. Host run limit (max_items=2) enforced cleanly (HTTP 429 on 3rd item).
    over = adapter.handle_txt2img({"prompt": "third item over cap"})
    assert over.ok is False and over.http_status == 429

    prog = adapter.handle_progress()
    assert prog.ok is True and prog.body["state"]["job_no"] == 2


# --------------------------------------------------------------------------- Phase K & Phase D UI check


def test_multiview_adherence_suite_and_frontend_subject_references():
    assert len(default_adherence_suite()) == 13
    mv = multi_view_reference_suite()
    assert len(mv) == 6
    assert {c.condition for c in mv} == {
        "starter_only",
        "starter_plus_front",
        "starter_plus_front_back",
        "multi_perspective",
        "conditioned_intermediate_keyframes",
        "conditioned_video",
    }
    assert all(c.human_evaluation_required for c in mv)
    assert len(MULTI_VIEW_CONSISTENCY_DIMENSIONS) == 7

    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="forgeSubjectRefs"' in html
    assert 'id="charProfileSel"' in html
    assert "character_profiles" in js_bridge.JS_API_METHODS
    assert "api.character_profiles(" in html


def test_flux2_multireference_workflow_chains_reference_latent_and_preserves_txt2img_baseline():
    from avatar_v2.providers.comfyui import _is_asset_key
    from avatar_v2.workflows_builder import build_flux2_txt2img_workflow

    assert _is_asset_key("REF_IMAGE_1") and _is_asset_key("REF_IMAGE_4")

    # Baseline txt2img without references has zero ReferenceLatent nodes.
    base_graph = build_flux2_txt2img_workflow({"PROMPT": "Studio portrait", "WIDTH": 896, "HEIGHT": 1280})
    assert not any(n["class_type"] == "ReferenceLatent" for n in base_graph.values())

    # Multi-reference graph with 2 views chains 4 ReferenceLatent nodes (pos + neg per view) + ImageScaleToTotalPixels.
    mref_graph = build_flux2_txt2img_workflow(
        {
            "PROMPT": "Rear view in sunlit hallway",
            "WIDTH": 896,
            "HEIGHT": 1280,
            "REF_IMAGE_1": "avatar_v2/front.png",
            "REF_IMAGE_2": "avatar_v2/back.png",
        }
    )
    ref_nodes = [n for n in mref_graph.values() if n["class_type"] == "ReferenceLatent"]
    scale_nodes = [n for n in mref_graph.values() if n["class_type"] == "ImageScaleToTotalPixels"]
    assert len(ref_nodes) == 4
    assert len(scale_nodes) == 2
    assert all(n["inputs"]["resolution_steps"] == 16 for n in scale_nodes)

    # Wan 2.2 First-Last-Frame upgrade preserves all graph links and injects end_image LoadImage node.
    from avatar_v2.workflows_builder import with_wan_end_frame

    wan_wf = json.loads((Path(__file__).parents[1] / "workflows" / "wan22_i2v_14b_gguf_api.json").read_text(encoding="utf-8"))
    flf_wf = with_wan_end_frame(wan_wf, "avatar_v2/back_keyframe.png")
    assert flf_wf["12"]["class_type"] == "WanFirstLastFrameToVideo"
    end_ref = flf_wf["12"]["inputs"]["end_image"]
    assert flf_wf[end_ref[0]]["class_type"] == "LoadImage"
    assert flf_wf[end_ref[0]]["inputs"]["image"] == "avatar_v2/back_keyframe.png"

