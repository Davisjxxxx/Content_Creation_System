"""Independent Adversarial Regression Tests — Astra RC1 Gate 2 (RC-04, RC-22, RC-11, RC-15, RC-17)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from avatar_v2 import desktop
from avatar_v2.inspection_contract import CorrectionBudget
from avatar_v2.resource_monitor import StorageCheck
from avatar_v2.script_broker import (
    EXECUTABLE_SCRIPT_MODELS,
    GenerationBroker,
    charge_durable_script_budget,
)
from avatar_v2.script_runner import (
    SandboxLimits,
    SandboxUnavailableError,
    ScriptRunManager,
    probe_sandbox_support,
)


HTML = (Path(__file__).resolve().parent.parent / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- RC-04 & RC-22: Provider Protection & Offline Zero-Egress Startup


def test_rc04_local_engine_loss_never_silently_switches_to_paid_runway() -> None:
    # Verify the offending auto-switch line is completely gone from UI engLoad()!
    assert "stChipSet('moveEngine','best')" not in HTML
    assert "Local GPU unavailable" in HTML


def test_rc22_startup_does_not_auto_query_runway_balance_in_offline_mode() -> None:
    # stInit must NOT automatically invoke stRefreshBalance() (external account lookup) on startup!
    init_section = HTML[HTML.index("function stInit()") : HTML.index("/* ---------- in-app self-test")]
    assert "stRefreshBalance()" not in init_section


def test_rc04_studio_go_local_move_refuses_when_engine_unavailable_never_dispatches_cloud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = tmp_path / "home"
    app_home.mkdir()
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")

    api = desktop.DesktopAPI()
    try:
        runway_calls: list[Any] = []
        monkeypatch.setattr(api, "queue_runway", lambda payload: (runway_calls.append(payload) or {"ok": True}))
        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "stopped", "message": "Your GPU engine is stopped."})
        monkeypatch.setattr(api, "engine_autostart", lambda: {"ok": False, "error": "offline"})

        res = api.studio_go({"action": "move", "engine": "local", "motion": "She smiles"})
        assert res["ok"] is False
        assert "stopped" in res["error"].lower() or "ready" in res["error"].lower()
        assert runway_calls == [], "Local move disappearance silently triggered Runway!"
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- RC-11: Correction & Durable Script Budgets


def test_rc11_correction_budget_enforces_zero_video_and_retries_used() -> None:
    # Zero global max_retries MUST yield 0 for BOTH image and video!
    zero_budget = CorrectionBudget(max_retries=0, max_image_retries=2, max_video_retries=1)
    assert zero_budget.limit_for_media("image") == 0
    assert zero_budget.limit_for_media("video") == 0

    # Retries already used MUST be subtracted!
    used_budget = CorrectionBudget(max_retries=2, max_image_retries=2, max_video_retries=1, retries_used=1)
    assert used_budget.limit_for_media("image") == 1
    assert used_budget.limit_for_media("video") == 0

    exhausted = CorrectionBudget(max_retries=2, max_image_retries=2, max_video_retries=1, retries_used=2)
    assert exhausted.limit_for_media("image") == 0
    assert exhausted.limit_for_media("video") == 0


def test_rc11_durable_script_budget_survives_detach_and_fails_closed(tmp_path: Path) -> None:
    runs_dir = tmp_path / "script_runs"
    runs_dir.mkdir()
    budget_file = runs_dir / "srun_detached.budget.json"

    # Broker submits 1 of 2 items -> persists submitted=1 durably
    broker = GenerationBroker(
        run_id="srun_detached",
        submit_fn=lambda ctx: f"j_{ctx.item_index}",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        max_items=2,
        budget_path=budget_file,
        storage_check_override=StorageCheck(True, "normal", 80.0, 10.0, 79.9, 15.0),
    )
    ok1, _ = broker.charge_corrective_attempt("j_1", note="attempt_1")
    assert ok1 is True

    # Simulate broker detached/restarted: durable charge_durable_script_budget still sees 1/2 used!
    ok2, msg2 = charge_durable_script_budget(runs_dir, "srun_detached", "j_1", note="detached_attempt_2")
    assert ok2 is True and "2/2" in msg2

    # 3rd charge MUST be refused even without active in-memory broker!
    ok3, msg3 = charge_durable_script_budget(runs_dir, "srun_detached", "j_1", note="detached_attempt_3")
    assert ok3 is False and "exhausted" in msg3.lower()

    # Unknown/missing run ledger MUST fail closed!
    ok_missing, msg_missing = charge_durable_script_budget(runs_dir, "srun_unknown", "j_x")
    assert ok_missing is False and "missing" in msg_missing.lower()


# --------------------------------------------------------------------------- RC-15: Script Model Capabilities Match Actual Builders


def test_rc15_script_broker_excludes_deferred_funcontrol_and_rejects_uninstalled(tmp_path: Path) -> None:
    # Create a fake ComfyUI dir with ONLY flux2_4b weights installed (no 9b, no wan, no funcontrol)
    comfy = tmp_path / "comfyui"
    for folder, fname in (
        ("diffusion_models", "flux-2-klein-4b-Q5_K_M.gguf"),
        ("text_encoders", "Qwen3-4B-Q5_K_M.gguf"),
        ("vae", "flux2-vae.safetensors"),
    ):
        p = comfy / "models" / folder / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")

    broker = GenerationBroker(
        run_id="srun_cap_test",
        submit_fn=lambda ctx: f"job_{ctx.item_index}",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        max_items=3,
        comfyui_dir=comfy,
        storage_check_override=StorageCheck(True, "normal", 80.0, 10.0, 79.9, 15.0),
    )
    caps = broker._op_capabilities()
    listed_ids = {m["model_id"] for m in caps["models"]}
    assert "wan22_funcontrol" not in listed_ids
    assert listed_ids == {"flux2_4b"}
    assert "wan22_funcontrol" not in EXECUTABLE_SCRIPT_MODELS

    # Attempting to submit deferred wan22_funcontrol is explicitly rejected BEFORE queueing!
    res = broker.handle_request(
        {
            "token": broker.token,
            "contract_version": "1.0",
            "op": "submit",
            "payload": {"model_id": "wan22_funcontrol", "spec": {"original_prompt": "A dancer turning smoothly"}},
        }
    )
    assert res["ok"] is False
    assert res["code"] == "unsupported_model"


# --------------------------------------------------------------------------- RC-17: Bounded Script Limiter, Streaming Output & Serialized Start


def test_rc17_missing_prlimit_fails_closed_and_bounded_output_prevents_unbounded_ram(tmp_path: Path) -> None:
    # 1. Missing prlimit MUST fail closed!
    probe_no_prlimit = probe_sandbox_support(prlimit_override="/nonexistent/prlimit_missing")
    assert probe_no_prlimit.available is False
    assert "prlimit" in probe_no_prlimit.error.lower()

    mgr_no_lim = ScriptRunManager(
        runs_dir=tmp_path / "runs_nolim",
        submit_fn=lambda ctx: "never",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        prlimit_override="/nonexistent/prlimit_missing",
    )
    script = tmp_path / "simple.py"
    script.write_text("import avatar_v2_sdk as av2\nav2.connect()\n", encoding="utf-8")
    with pytest.raises(SandboxUnavailableError):
        mgr_no_lim.start_sandboxed_run(script_path=script, max_items=1)

    # 2. Live Bubblewrap bounded-tail test (spamming 512 KiB stdout stays bounded to <= 4000 chars!)
    live_probe = probe_sandbox_support()
    if not live_probe.available:
        pytest.skip(f"Bubblewrap/prlimit unavailable: {live_probe.error}")

    noisy = tmp_path / "noisy_script.py"
    noisy.write_text(
        "import sys, avatar_v2_sdk as av2\n"
        "for i in range(256):\n"
        "    sys.stdout.write(('X' * 2048) + f'\\nLINE_{i}\\n')\n"
        "sys.stdout.write('TAIL_SENTINEL_END\\n')\n",
        encoding="utf-8",
    )
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs_noisy",
        submit_fn=lambda ctx: "j1",
        status_fn=lambda jid: {"status": "done", "outputs": []},
        storage_check_override=StorageCheck(True, "normal", 80.0, 10.0, 79.9, 15.0),
    )
    assert mgr.select_script(noisy)["ok"] is True
    started = mgr.start_sandboxed_run(max_items=1, limits=SandboxLimits(wall_timeout_s=10.0), allow_review_classification=True)
    assert started["ok"] is True, started

    deadline = time.monotonic() + 12.0
    rec: dict[str, Any] = {}
    while time.monotonic() < deadline:
        rec = mgr.status().get("record") or {}
        if rec.get("status") in {"completed", "error", "stopped"}:
            break
        time.sleep(0.05)

    assert rec["status"] == "completed", rec
    assert len(rec["stdout_tail"]) <= 4000
    assert "TAIL_SENTINEL_END" in rec["stdout_tail"]
