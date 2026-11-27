"""Adversarial RC3 Gate 3 tests: durable job submission, fail-closed budgets,
strict terminal ComfyUI outputs, single-writer lock across slow shutdown, and JSON schema validation.

Directly reproduces Astra RC2 independent recheck findings:
- RC-10: Crash window after renderer acceptance before on_submitted(prompt_id)
- RC-11: Budget ledger write failure allowing detached double-charge past max_items
- RC-16: Non-terminal ComfyUI preview output promoted as completed render
- RC-18: Single-writer lock released while worker thread still alive; desktop.json `[]` TypeError
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import avatar_v2.desktop as desktop
import avatar_v2.providers.comfyui as comfyui_mod
import avatar_v2.render_queue as rq
import avatar_v2.script_broker as sb
from avatar_v2.composer import build_job
from avatar_v2.gpu_memory import GPUMemoryManager
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from avatar_v2.providers.comfyui import ComfyUIProvider


def _make_job(job_id: str = "rc3-job-1") -> RenderJob:
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
            "user_prompt": "A ceramic teacup on a wooden table",
        }
    )
    job = build_job(avatar, shot)
    job.job_id = job_id
    return job


# ---------------------------------------------------------------------------
# RC-10: Pre-acknowledgement crash window & durability failure before dispatch
# ---------------------------------------------------------------------------


def test_rc10_crash_between_provider_queue_and_on_submitted_never_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-10 counterexample:
    Renderer accepts the prompt, process crashes BEFORE on_submitted(prompt_id) executes.
    Restarting RenderQueue MUST mark the job recovery_required and NEVER replay it.
    """
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    snapshot_dir = tmp_path / "crash_snapshot"
    accepted_prompts: list[str] = []
    crash_captured = threading.Event()

    class CrashBeforeAckProvider:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            self.cfg = cfg

        def render(
            self,
            workflow: str,
            job: RenderJob,
            *,
            on_submitted=None,
            cancel_event=None,
        ) -> dict[str, Any]:
            # Renderer accepts the prompt...
            accepted_prompts.append("prompt-accepted-001")
            # ...and process crashes BEFORE on_submitted(prompt_id) can run!
            shutil.copytree(work_dir, snapshot_dir)
            crash_captured.set()
            raise RuntimeError("simulated hard crash before on_submitted")

    monkeypatch.setattr(rq, "ComfyUIProvider", CrashBeforeAckProvider)
    q1 = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        q1.add(
            rq.QueuedRender(
                id="job-rc10-crash",
                job=_make_job("job-rc10-crash"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        assert crash_captured.wait(timeout=5.0), "provider was not invoked"
    finally:
        q1.shutdown()

    assert accepted_prompts == ["prompt-accepted-001"]
    # Verify on-disk snapshot captured at crash instant ALREADY recorded phase="submitted" + submission_token
    snap_item = json.loads((snapshot_dir / "items" / "job-rc10-crash.json").read_text(encoding="utf-8"))
    assert snap_item["phase"] == "submitted"
    assert snap_item["runtime"].get("submission_token")

    # Remove copied lock file before restarting on snapshot_dir
    (snapshot_dir / ".writer.lock").unlink(missing_ok=True)

    # Restart a new RenderQueue from the exact crash snapshot!
    replayed: list[str] = []

    class ReplayDetectorProvider:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            replayed.append(job.job_id)
            return {"prompt_id": "replay", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", ReplayDetectorProvider)
    q2 = rq.RenderQueue(
        work_dir=snapshot_dir,
        history_dir=tmp_path / "history_after_crash",
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        time.sleep(0.3)
        items = q2.list()
        assert len(items) == 1
        assert items[0]["status"] == "error"
        assert items[0]["phase"] == "recovery_required"
        assert replayed == [], f"Crash recovery replayed an already-accepted render: {replayed}"
    finally:
        q2.shutdown()


def test_rc10_durability_write_failure_refuses_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If persisting the pre-submission marker fails (OSError), provider.render MUST NOT be called."""
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    dispatched: list[str] = []

    class SpyProvider:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            dispatched.append(job.job_id)
            return {"prompt_id": "should-not-happen", "outputs": []}

    monkeypatch.setattr(rq, "ComfyUIProvider", SpyProvider)
    q = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
    )
    try:
        def _fail_persist(item: rq.QueuedRender) -> None:
            raise OSError("disk full during durable pre-submit write")

        monkeypatch.setattr(q, "_persist_item_durable", _fail_persist)
        jid = q.add(
            rq.QueuedRender(
                id="job-rc10-diskfull",
                job=_make_job("job-rc10-diskfull"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == jid), None)
            if cur and cur["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.05)
        cur = next(x for x in q.list() if x["id"] == jid)
        assert cur["status"] == "error"
        assert dispatched == [], "Provider was dispatched despite pre-submit durability failure"
    finally:
        q.shutdown()


# ---------------------------------------------------------------------------
# RC-11: Fail-closed script budget persistence (no double-charge on write OSError)
# ---------------------------------------------------------------------------


def test_rc11_budget_write_failure_blocks_charge_and_prevents_double_charge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-11 counterexample:
    Budget limit=1. First charge fails write_json_atomic with OSError.
    The live charge MUST fail closed (False), leave disk at 0, and NEVER authorize a charge
    that later allows a second detached charge past max_items=1.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    budget_path = runs_dir / "run-rc11.budget.json"

    broker = sb.GenerationBroker(
        run_id="run-rc11",
        submit_fn=lambda ctx: "job-1",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        max_items=1,
        budget_path=budget_path,
    )
    assert json.loads(budget_path.read_text(encoding="utf-8"))["submitted"] == 0

    real_write = sb.write_json_atomic

    def _boom(path: Path, data: Any) -> None:
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr(sb, "write_json_atomic", _boom)

    ok1, msg1 = broker.charge_corrective_attempt("parent-1", note="attempt-1")
    assert ok1 is False, f"Expected fail-closed refusal on write OSError, got ok=True ({msg1})"
    assert "failed" in msg1.lower() or "refusing" in msg1.lower()
    assert broker.snapshot_status()["submitted"] == 0
    assert json.loads(budget_path.read_text(encoding="utf-8"))["submitted"] == 0

    # Restore writes: now exactly ONE charge succeeds (live OR detached), never two!
    monkeypatch.setattr(sb, "write_json_atomic", real_write)

    ok_live, _ = broker.charge_corrective_attempt("parent-1", note="attempt-live")
    assert ok_live is True
    assert json.loads(budget_path.read_text(encoding="utf-8"))["submitted"] == 1

    ok_detached, msg_detached = sb.charge_durable_script_budget(runs_dir, "run-rc11", "parent-1", note="detached")
    assert ok_detached is False, f"Detached caller double-charged past limit=1: {msg_detached}"
    assert "exhausted" in msg_detached.lower()

    # Also verify charge_durable_script_budget fails closed on write OSError
    budget_path2 = runs_dir / "run-rc11-detached.budget.json"
    real_write(budget_path2, {"run_id": "run-rc11-detached", "state": "running", "submitted": 0, "max_items": 1})
    monkeypatch.setattr(sb, "write_json_atomic", _boom)
    ok_d_fail, msg_d_fail = sb.charge_durable_script_budget(runs_dir, "run-rc11-detached", "p2")
    assert ok_d_fail is False
    assert "failed" in msg_d_fail.lower()
    assert json.loads(budget_path2.read_text(encoding="utf-8"))["submitted"] == 0


# ---------------------------------------------------------------------------
# RC-16: ComfyUIProvider.wait() MUST NOT promote non-terminal preview outputs
# ---------------------------------------------------------------------------


def test_rc16_wait_does_not_promote_non_terminal_preview_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces Astra's exact RC-16 counterexample:
    /history/<pid> initially returns outputs={'9': {'images': [{'filename': 'preview.png'}]}} with status={}.
    ComfyUIProvider.wait() MUST NOT return success until status reports terminal completion!
    """
    polls = {"count": 0}

    class FakeResp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str) -> FakeResp:
            polls["count"] += 1
            if polls["count"] < 3:
                return FakeResp({
                    "pid-preview": {
                        "status": {},
                        "outputs": {"9": {"images": [{"filename": "preview.png", "subfolder": ""}]}},
                    }
                })
            return FakeResp({
                "pid-preview": {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"10": {"images": [{"filename": "final.png", "subfolder": ""}]}},
                }
            })

    monkeypatch.setattr(comfyui_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(comfyui_mod.time, "sleep", lambda _: None)

    provider = ComfyUIProvider(ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188", timeout_s=30))
    history = provider.wait("pid-preview")
    assert polls["count"] == 3, f"wait() returned prematurely on poll {polls['count']} before terminal status!"
    assert provider.output_files(history) == ["final.png"]


def test_rc16_wait_times_out_if_only_non_terminal_preview_ever_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "pid-stuck": {
                    "status": {},
                    "outputs": {"9": {"images": [{"filename": "preview.png", "subfolder": ""}]}},
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args) -> None:
            pass

        def get(self, url: str) -> FakeResp:
            return FakeResp()

    clock = iter([0.0, 1.0, 2.0, 35.0])
    monkeypatch.setattr(comfyui_mod.httpx, "Client", FakeClient)
    monkeypatch.setattr(comfyui_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(comfyui_mod.time, "monotonic", lambda: next(clock))

    provider = ComfyUIProvider(ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188", timeout_s=30))
    with pytest.raises(TimeoutError):
        provider.wait("pid-stuck")


# ---------------------------------------------------------------------------
# RC-18: Single-writer lock across slow shutdown + desktop.json `[]` schema check
# ---------------------------------------------------------------------------


def test_rc18_shutdown_holds_dir_lock_while_worker_thread_is_still_alive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-18 counterexample:
    Worker is inside a slow QA hook that outlives shutdown()'s 2.0s join timeout.
    The first RenderQueue MUST retain its DirectoryWriterLock until _worker exits,
    preventing a second overlapping RenderQueue from taking over work_dir!
    """
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    qa_entered = threading.Event()
    qa_release = threading.Event()

    class FastProvider:
        uses_local_gpu = False

        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            return {"prompt_id": "p1", "outputs": [str(Path(__file__).parent / "fixtures/rc5/still.png")]}

    def slow_qa(item: rq.QueuedRender) -> dict[str, Any]:
        qa_entered.set()
        qa_release.wait(timeout=10.0)
        return {"ok": True}

    monkeypatch.setattr(rq, "RunwayProvider", FastProvider)
    q1 = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
        candidate_qa_runner=slow_qa,
    )
    q1.add(
        rq.QueuedRender(
            id="job-slow-qa",
            job=_make_job("job-slow-qa"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="runway"),
            profile="auto",
            runtime={"candidate_qa": True},
        )
    )
    assert qa_entered.wait(timeout=5.0), "slow QA did not start"

    # Call shutdown() while QA is blocked; join(2.0) will time out while worker is STILL alive.
    q1.shutdown()
    assert q1._thread.is_alive(), "Expected worker thread to still be alive in slow QA"

    # A second RenderQueue on the SAME work_dir MUST be blocked while first worker holds the lock!
    with pytest.raises((RuntimeError, OSError)):
        rq.RenderQueue(
            work_dir=work_dir,
            history_dir=history_dir,
            memory_manager=GPUMemoryManager(enabled=False),
        )

    # Now let QA finish; once worker exits, _dir_lock is released cleanly and a new RenderQueue succeeds!
    qa_release.set()
    q1._thread.join(timeout=5.0)
    assert not q1._thread.is_alive()

    q2 = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
    )
    q2.shutdown()


def test_rc18_desktop_json_wrong_root_type_is_quarantined_without_typeerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra's exact RC-18 counterexample:
    desktop.json containing valid JSON `[]` instead of `{}` MUST NOT raise TypeError.
    """
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    config_path = app_home / "desktop.json"
    config_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", config_path)
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")

    api = desktop.DesktopAPI()
    try:
        settings = api._settings()
        assert isinstance(settings, dict)
        quarantined = list(app_home.glob("desktop.json.corrupt.*"))
        assert len(quarantined) == 1, f"Expected desktop.json `[]` to be quarantined, found {quarantined}"
    finally:
        api.queue.shutdown()
