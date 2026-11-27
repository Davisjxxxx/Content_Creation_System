"""Adversarial regression suite for Astra RC1 Gate 3 (RC-10, RC-16, RC-18).

Covers:
- RC-10: Job lifecycle phases, real QA/correction cancellation preserving valid original outputs,
         durable local prompt_id before wait, and recovery_required (never blind resubmit) on restart.
- RC-16: ComfyUI error status with partial outputs NEVER marked done; output_dir traversal blocked.
- RC-18: Unique atomic write_durable, corrupt JSON quarantine across stores, worker survival under
         disk-full OSError, and single-writer directory flock.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from avatar_v2.library import LibraryStore
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec, SubjectSpec
from avatar_v2.ownership import read_json_safe
from avatar_v2.providers.comfyui import ComfyUICancelledError, ComfyUIExecutionError, ComfyUIProvider
from avatar_v2.render_queue import QueuedRender, RenderQueue
from avatar_v2.subject_references import CharacterProfile, CharacterProfileStore


def _make_job(job_id: str = "job-rc3") -> RenderJob:
    return RenderJob(
        job_id=job_id,
        avatar=AvatarManifest(
            avatar_id="av1",
            display_name="Test",
            subjects=[SubjectSpec(id="s1", kind="synthetic", age_verified_18_plus=True, consent_confirmed=True)],
        ),
        shot=ShotSpec(shot_id="s1", user_prompt="test prompt", seed=41001, width=512, height=512, frames=1),
        selected_engine="flux2",
        prompt="test prompt",
        negative_prompt="",
        asset_map={},
    )


def _make_item(tmp_path: Path, **kwargs: Any) -> QueuedRender:
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, Any] = {
        "job": _make_job(),
        "workflow": "builtin:flux2_txt2img",
        "provider_config": ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:18199", output_dir=str(out_dir)),
        "profile": "auto",
        "output_dir": str(out_dir),
    }
    defaults.update(kwargs)
    return QueuedRender(**defaults)


def _wait_for_status(queue: RenderQueue, item_id: str, targets: set[str], timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for entry in queue.list():
            if entry["id"] == item_id and entry["status"] in targets:
                return entry
        time.sleep(0.03)
    raise AssertionError(f"Item {item_id} did not reach {targets}; current={queue.list()}")


@pytest.mark.usefixtures("measured_local_gpu")
def test_rc16_error_with_partial_outputs_raises_and_never_marks_done(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ComfyUI history with BOTH status_str=error AND partial node outputs must fail, NOT succeed."""
    payload = {
        "pid-partial": {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_start", {"prompt_id": "pid-partial"}],
                    [
                        "execution_error",
                        {
                            "node_id": "9",
                            "node_type": "SaveVideo",
                            "exception_message": "ffmpeg failed mid-write",
                        },
                    ],
                ],
            },
            "outputs": {
                "4": {"images": [{"filename": "partial_preview.png", "subfolder": "tmp"}]},
            },
        }
    }

    class _FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return payload

    class _FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, url: str) -> _FakeResp:
            return _FakeResp()

    monkeypatch.setattr("avatar_v2.ollama_client.fetch_ollama_ps_state", lambda *a, **kw: ("online", [], ""))
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    provider = ComfyUIProvider(ProviderConfig(base_url="http://127.0.0.1:18199", timeout_s=30))
    with pytest.raises(ComfyUIExecutionError) as exc_info:
        provider.wait("pid-partial")
    assert exc_info.value.partial_outputs == ["tmp/partial_preview.png"]
    assert "SaveVideo" in str(exc_info.value)

    # Also verify RenderQueue records status=error, outputs=[], partial_outputs=[...]
    class _FailingPartialProvider:
        uses_local_gpu = True

        def render(self, workflow: str, job: RenderJob, on_submitted=None, cancel_event=None) -> dict[str, Any]:
            if on_submitted:
                on_submitted("pid-partial")
            raise ComfyUIExecutionError("node 9 failed", partial_outputs=["tmp/partial_preview.png"])

        def free_memory(self, **kw: Any) -> None:
            pass

        def queue_status(self) -> dict[str, Any]:
            return {"running": 0, "pending": 0, "busy": False}

    queue = RenderQueue(work_dir=tmp_path / "q", history_dir=tmp_path / "h")
    try:
        queue.storage_guard_enabled = False  # type: ignore[attr-defined]
        monkeypatch.setattr(queue, "_make_provider", lambda cfg: _FailingPartialProvider())
        item = _make_item(tmp_path)
        item.runtime["recover_attempts"] = 1
        qid = queue.add(item)
        done = _wait_for_status(queue, qid, {"error", "done"})
        assert done["status"] == "error"
        assert done["phase"] == "failed"
        assert done["outputs"] == []
        assert len(done["partial_outputs"]) == 1
        assert done["partial_outputs"][0].endswith("tmp/partial_preview.png")
    finally:
        queue.shutdown()


def test_rc16_output_path_traversal_blocked(tmp_path: Path) -> None:
    """Renderer metadata attempting parent traversal or absolute path escape is rejected."""
    with pytest.raises(ValueError, match="unsafe ComfyUI output path"):
        ComfyUIProvider.output_files(
            {"outputs": {"1": {"images": [{"filename": "desktop.json", "subfolder": "../../.avatar_v2"}]}}}
        )
    with pytest.raises(ValueError, match="unsafe ComfyUI output path"):
        ComfyUIProvider.output_files(
            {"outputs": {"1": {"images": [{"filename": "/etc/passwd", "subfolder": ""}]}}}
        )

    queue = RenderQueue(work_dir=tmp_path / "q", history_dir=tmp_path / "h")
    try:
        item = _make_item(tmp_path)
        with pytest.raises(ValueError, match="parent traversal"):
            queue._resolve_outputs(item, ["../../etc/passwd"])
    finally:
        queue.shutdown()


@pytest.mark.usefixtures("measured_local_gpu")
def test_rc10_job_stays_running_during_qa_and_cancel_preserves_original_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Job stays status=running during QA/correction; cancelling during QA stops it and preserves valid outputs."""
    out_file = tmp_path / "outputs" / "orig.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes((Path(__file__).parent / "fixtures/rc5/still.png").read_bytes())

    qa_entered = threading.Event()
    qa_saw_cancel = threading.Event()
    status_during_qa: dict[str, str] = {}

    class _OkProvider:
        uses_local_gpu = True

        def render(self, workflow: str, job: RenderJob, on_submitted=None, cancel_event=None) -> dict[str, Any]:
            if on_submitted:
                on_submitted("pid-qa-cancel")
            return {"prompt_id": "pid-qa-cancel", "outputs": ["orig.png"]}

        def free_memory(self, **kw: Any) -> None:
            pass

        def queue_status(self) -> dict[str, Any]:
            return {"running": 0, "pending": 0, "busy": False}

    def _qa_runner(item: QueuedRender) -> dict[str, Any]:
        status_during_qa["status"] = item.status
        status_during_qa["phase"] = item.phase
        qa_entered.set()
        ev = queue.get_cancel_event(item.id)
        if ev.wait(timeout=3.0):
            qa_saw_cancel.set()
        return {"ok": True, "decision": "needs_correction"}

    queue = RenderQueue(work_dir=tmp_path / "q", history_dir=tmp_path / "h", candidate_qa_runner=_qa_runner)
    try:
        queue.storage_guard_enabled = False  # type: ignore[attr-defined]
        monkeypatch.setattr(queue, "_make_provider", lambda cfg: _OkProvider())
        item = _make_item(tmp_path, runtime={"candidate_qa": True})
        qid = queue.add(item)
        assert qa_entered.wait(timeout=3.0)
        assert status_during_qa["status"] == "running"
        assert status_during_qa["phase"] == "inspecting"
        assert queue.cancel(qid) is True
        assert qa_saw_cancel.wait(timeout=2.0)
        final = _wait_for_status(queue, qid, {"cancelled", "done", "error"})
        assert final["status"] == "cancelled"
        assert final["phase"] == "cancelled"
        # Valid completed original output is preserved!
        assert final["outputs"] == [str(out_file.resolve())]
    finally:
        queue.shutdown()


def test_rc10_post_submit_restart_marks_recovery_required_instead_of_blind_resubmit(tmp_path: Path) -> None:
    """A local job that already persisted prompt_id (or post-submit phase) must NEVER blindly resubmit on restart."""
    q_dir = tmp_path / "q"
    h_dir = tmp_path / "h"
    submitted_item = _make_item(
        tmp_path,
        id="submitted_1",
        status="running",
        phase="rendering",
        prompt_id="comfy-pid-999",
        started_at=time.time() - 20.0,
    )
    pre_submit_item = _make_item(
        tmp_path,
        id="presubmit_1",
        status="running",
        phase="preparing",
        prompt_id=None,
        started_at=time.time() - 10.0,
    )
    q1 = RenderQueue(work_dir=q_dir, history_dir=h_dir)
    try:
        q1._persist_item_durable(submitted_item)
        q1._persist_item_durable(pre_submit_item)
    finally:
        q1.shutdown()

    q2 = RenderQueue(work_dir=q_dir, history_dir=h_dir)
    try:
        by_id = {it["id"]: it for it in q2.list()}
        assert by_id["submitted_1"]["status"] == "error"
        assert by_id["submitted_1"]["phase"] == "recovery_required"
        assert by_id["submitted_1"]["error_category"] == "recovery_required"
        assert (h_dir / "submitted_1.json").is_file()
        assert not (q_dir / "items" / "submitted_1.json").exists()

        # Pre-submit item safely requeues.
        assert by_id["presubmit_1"]["status"] == "queued"
        assert by_id["presubmit_1"]["phase"] == "queued"
    finally:
        q2.shutdown()


def test_rc18_corrupt_json_quarantined_across_stores(tmp_path: Path) -> None:
    """Corrupt JSON in profiles, library, settings, and queue items is quarantined rather than silently overwritten."""
    # 1. CharacterProfileStore
    prof_path = tmp_path / "profiles.json"
    prof_path.write_text("{broken json", encoding="utf-8")
    cstore = CharacterProfileStore(prof_path)
    assert cstore.list_profiles() == []
    quarantined_prof = list(tmp_path.glob("profiles.json.corrupt.*"))
    assert len(quarantined_prof) == 1
    assert quarantined_prof[0].read_text(encoding="utf-8") == "{broken json"
    # Saving a new profile leaves the quarantined corrupt file intact!
    cstore.save_profile(CharacterProfile(name="Recovered"))
    assert quarantined_prof[0].is_file()

    # 2. LibraryStore
    lib_dir = tmp_path / "library"
    lib_dir.mkdir()
    (lib_dir / "vault.json").write_text("NOT_JSON!!!", encoding="utf-8")
    lib = LibraryStore(lib_dir)
    assert lib.avatars() == []
    assert len(list(lib_dir.glob("vault.json.corrupt.*"))) == 1

    # 3. Generic read_json_safe (used by desktop.json)
    cfg = tmp_path / "desktop.json"
    cfg.write_text("{truncated", encoding="utf-8")
    assert read_json_safe(cfg, {"ok": True}, quarantine=True) == {"ok": True}
    assert len(list(tmp_path.glob("desktop.json.corrupt.*"))) == 1

    # 4. Queue items on restore
    q_dir = tmp_path / "q"
    (q_dir / "items").mkdir(parents=True)
    (q_dir / "items" / "bad_item.json").write_text("{bad", encoding="utf-8")
    q = RenderQueue(work_dir=q_dir, history_dir=tmp_path / "h")
    try:
        assert len(list((q_dir / "items").glob("bad_item.json.corrupt.*"))) == 1
    finally:
        q.shutdown()


@pytest.mark.usefixtures("measured_local_gpu")
def test_rc18_worker_survives_disk_full_and_single_writer_lock_enforced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Persistence OSError must not kill the worker thread; concurrent writer lock must fail closed."""
    out_file = tmp_path / "outputs" / "ok.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes((Path(__file__).parent / "fixtures/rc5/still.png").read_bytes())

    class _OkProvider:
        uses_local_gpu = True

        def render(self, workflow: str, job: RenderJob, on_submitted=None, cancel_event=None) -> dict[str, Any]:
            from uuid import uuid4
            prompt_id = uuid4().hex
            if on_submitted:
                on_submitted(prompt_id)
            return {"prompt_id": prompt_id, "outputs": ["ok.png"]}

        def free_memory(self, **kw: Any) -> None:
            pass

        def queue_status(self) -> dict[str, Any]:
            return {"running": 0, "pending": 0, "busy": False}

    q_dir = tmp_path / "q_lock"
    h_dir = tmp_path / "h_lock"
    q1 = RenderQueue(work_dir=q_dir, history_dir=h_dir)
    try:
        # Second concurrent instance on same work_dir must fail closed!
        with pytest.raises(RuntimeError, match="already holds the writer lock"):
            RenderQueue(work_dir=q_dir, history_dir=h_dir)

        q1.storage_guard_enabled = False  # type: ignore[attr-defined]
        monkeypatch.setattr(q1, "_make_provider", lambda cfg: _OkProvider())

        # Inject ENOSPC on first _persist_history call only
        orig_persist_history = q1._persist_history
        fail_once = {"flag": True}

        def _flaky_persist_history(item: QueuedRender) -> None:
            if fail_once["flag"]:
                fail_once["flag"] = False
                raise OSError(28, "No space left on device")
            orig_persist_history(item)

        monkeypatch.setattr(q1, "_persist_history", _flaky_persist_history)

        id1 = q1.add(_make_item(tmp_path, id="diskfull_1"))
        res1 = _wait_for_status(q1, id1, {"error", "done"})
        assert res1["status"] == "error"
        assert res1["error_category"] == "persistence_failed"
        assert q1._thread.is_alive()

        # Worker thread is still alive and completes the next job cleanly!
        id2 = q1.add(_make_item(tmp_path, id="after_enospc_2"))
        res2 = _wait_for_status(q1, id2, {"done", "error"})
        assert res2["status"] == "done"
        assert res2["phase"] == "complete"
    finally:
        q1.shutdown()

    # After q1.shutdown(), a new instance can cleanly acquire the lock!
    q2 = RenderQueue(work_dir=q_dir, history_dir=h_dir)
    q2.shutdown()
