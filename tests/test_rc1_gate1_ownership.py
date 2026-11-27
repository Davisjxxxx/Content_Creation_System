"""Independent Adversarial Regression Tests — Astra RC1 Gate 1 (P0-A RC-01, P0-B RC-02, P1 RC-03).

Uses ONLY harmless disposable child processes and temporary scratch media; never targets
owner processes, live ~/.avatar_v2, or production bundles.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop, ownership
from avatar_v2.js_bridge import JsBridge
from avatar_v2.models import ProviderConfig
from avatar_v2.providers.comfyui import ComfyUIProvider


def _make_png(path: Path, color: tuple[int, int, int] = (40, 120, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color).save(path, format="PNG")
    return path


def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    app_home = tmp_path / "app_home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "COMFY_PID_PATH", app_home / "comfyui.pid")
    monkeypatch.setattr(desktop, "COMFY_OWNER_PATH", app_home / "comfyui_owner.json")
    monkeypatch.setattr(desktop, "COMFY_LOG_PATH", app_home / "logs" / "comfyui.log")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")
    monkeypatch.setattr(desktop, "RUNWAY_AVATARS_PATH", app_home / "runway_avatars.json")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    monkeypatch.setattr(desktop, "VOICEOVER_DIR", app_home / "voiceover")
    monkeypatch.setattr(desktop, "IMPORTS_DIR", app_home / "library" / "imports")
    monkeypatch.setattr(desktop, "LESSONS_PATH", app_home / "lessons.json")
    monkeypatch.setattr(desktop, "QA_DIR", app_home / "qa")
    monkeypatch.setattr(desktop, "PROVIDER_PROFILES_PATH", app_home / "provider_profiles.json")
    desktop._write_json(
        app_home / "desktop.json",
        {
            **desktop.DEFAULT_SETTINGS,
            "comfyui_url": "http://127.0.0.1:18991",
            "archive_dir": str(app_home / "archive"),
        },
    )
    return app_home


def _spawn_harmless_sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


# --------------------------------------------------------------------------- P0-A (RC-01): Process Ownership


def test_p0a_rc01_stale_legacy_pid_never_signals_unrelated_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = _isolate_home(monkeypatch, tmp_path)
    bystander = _spawn_harmless_sleeper()
    try:
        # Legacy bare integer PID file pointing at a live unrelated process!
        (app_home / "comfyui.pid").write_text(str(bystander.pid), encoding="utf-8")

        class DeadComfy:
            def __init__(self, cfg: ProviderConfig) -> None:
                pass

            def health(self) -> dict[str, Any]:
                raise ConnectionError("not running")

        monkeypatch.setattr(desktop, "ComfyUIProvider", DeadComfy)
        api = desktop.DesktopAPI()
        try:
            st = api.comfyui_status()
            assert st["ownership"] == "stale_or_reused"
            assert st["managed"] is False
            assert st["pid"] is None

            res = api.comfyui_stop(wait_s=1.0, force=True)
            assert res["signalled"] is False
            assert bystander.poll() is None, "Bystander process was signalled via bare legacy PID!"
            assert not (app_home / "comfyui.pid").exists()
        finally:
            api.queue.shutdown()
    finally:
        bystander.terminate()
        bystander.wait(timeout=5)


def test_p0a_rc01_reused_pid_and_mismatched_identity_never_signalled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = _isolate_home(monkeypatch, tmp_path)
    bystander = _spawn_harmless_sleeper()
    try:
        real_rec = ownership.ProcessOwnershipRecord.capture_for_pid(bystander.pid, port=18991)
        # 1. Recycled PID: different start_ticks!
        reused_rec = ownership.ProcessOwnershipRecord(
            pid=bystander.pid,
            start_ticks=(real_rec.start_ticks or 100) + 99999,
            boot_id=real_rec.boot_id,
            exe_realpath=real_rec.exe_realpath,
            cmdline=real_rec.cmdline,
            port=18991,
            owned_by_app=True,
        )
        ownership.save_ownership_record(app_home / "comfyui_owner.json", reused_rec)

        ok, reason = ownership.verify_process_ownership(reused_rec)
        assert ok is False and reason == "pid_reused"

        # 2. Unrelated matching executable (different port/cmdline)
        wrong_port_rec = ownership.ProcessOwnershipRecord(
            pid=bystander.pid,
            start_ticks=real_rec.start_ticks,
            boot_id=real_rec.boot_id,
            exe_realpath=real_rec.exe_realpath,
            cmdline=[sys.executable, "main.py", "--port", "8188"],
            port=8188,
            owned_by_app=True,
        )
        ok_port, reason_port = ownership.verify_process_ownership(wrong_port_rec)
        assert ok_port is False and reason_port in {"cmdline_mismatch", "port_mismatch"}

        class DeadComfy:
            def __init__(self, cfg: ProviderConfig) -> None:
                pass

            def health(self) -> dict[str, Any]:
                raise ConnectionError("not running")

        monkeypatch.setattr(desktop, "ComfyUIProvider", DeadComfy)
        api = desktop.DesktopAPI()
        try:
            res = api.comfyui_stop(wait_s=1.0, force=True)
            assert res["signalled"] is False
            assert bystander.poll() is None, "Bystander process was killed despite PID reuse!"
        finally:
            api.queue.shutdown()
    finally:
        bystander.terminate()
        bystander.wait(timeout=5)


def test_p0a_rc01_shared_external_renderer_never_freed_or_stopped_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_home(monkeypatch, tmp_path)
    free_calls: list[dict[str, Any]] = []

    class SharedExternalComfy:
        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
            free_calls.append({"unload": unload_models, "free": free_memory})

    monkeypatch.setattr(desktop, "ComfyUIProvider", SharedExternalComfy)
    api = desktop.DesktopAPI()
    st = api.comfyui_status()
    assert st["ready"] is True
    assert st["ownership"] == "shared_or_external"
    assert st["managed"] is False

    stop_res = api.comfyui_stop(wait_s=1.0, force=True)
    assert stop_res["ok"] is False
    assert stop_res["signalled"] is False

    # Shutdown MUST NOT call /free on a shared/external renderer!
    api.shutdown_engine()
    assert free_calls == [], f"shutdown_engine disrupted a shared external renderer: {free_calls}"


def _spawn_listening_child() -> tuple[subprocess.Popen[str], int]:
    """Spawn a real child process bound to an ephemeral loopback TCP port."""
    script = (
        "import socket, sys, time; "
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        "s.bind(('127.0.0.1', 0)); "
        "s.listen(5); "
        "print(s.getsockname()[1], flush=True); "
        "time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    port = int(proc.stdout.readline().strip())
    return proc, port


def test_p0a_rc01_owned_child_lifecycle_exit_race_repeated_shutdown_and_restart_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = _isolate_home(monkeypatch, tmp_path)
    free_calls: list[int] = []

    class OwnedComfyStub:
        def __init__(self, cfg: ProviderConfig) -> None:
            pass

        def health(self) -> dict[str, Any]:
            return {"system": {"comfyui_version": "0.18.1"}}

        def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
            free_calls.append(1)

    monkeypatch.setattr(desktop, "ComfyUIProvider", OwnedComfyStub)

    # 1. Owned child stopped cleanly via OwnedProcessHandle (pidfd) + repeated shutdown idempotent
    child1, port1 = _spawn_listening_child()
    try:
        (app_home / "desktop.json").write_text(
            json.dumps({"comfyui_url": f"http://127.0.0.1:{port1}"}),
            encoding="utf-8",
        )
        api = desktop.DesktopAPI()
        rec1 = ownership.ProcessOwnershipRecord.capture_for_pid(child1.pid, port=port1)
        api._owned_comfy_handle = ownership.OwnedProcessHandle(child1, rec1)
        ownership.save_ownership_record(app_home / "comfyui_owner.json", rec1)
        assert api.comfyui_status()["ownership"] == "owned_private"

        api.shutdown_engine()
        child1.wait(timeout=5)
        assert child1.poll() is not None
        assert len(free_calls) == 1
        assert not (app_home / "comfyui_owner.json").exists()

        # Repeated shutdown is safe and does not double-free or error
        api.shutdown_engine()
        assert len(free_calls) == 1
    finally:
        if child1.poll() is None:
            child1.kill()
            child1.wait(timeout=5)

    # 2. Crash recovery across restart: new DesktopAPI instance stops owned child via verified OS record
    child2, port2 = _spawn_listening_child()
    try:
        (app_home / "desktop.json").write_text(
            json.dumps({"comfyui_url": f"http://127.0.0.1:{port2}"}),
            encoding="utf-8",
        )
        rec2 = ownership.ProcessOwnershipRecord.capture_for_pid(child2.pid, port=port2)
        ownership.save_ownership_record(app_home / "comfyui_owner.json", rec2)
        api2 = desktop.DesktopAPI()
        try:
            assert api2.comfyui_status()["ownership"] == "owned_private"
            res = api2.comfyui_stop(wait_s=5.0, force=True)
            assert res["ok"] is True and res["signalled"] is True
            child2.wait(timeout=5)
            assert child2.poll() is not None
        finally:
            api2.queue.shutdown()
    finally:
        if child2.poll() is None:
            child2.kill()
            child2.wait(timeout=5)

    # 3. Process exits between ownership verification and cleanup
    child3 = _spawn_harmless_sleeper()
    rec3 = ownership.ProcessOwnershipRecord.capture_for_pid(child3.pid, port=18991)
    ownership.save_ownership_record(app_home / "comfyui_owner.json", rec3)
    child3.terminate()
    child3.wait(timeout=5)
    api3 = desktop.DesktopAPI()
    try:
        sent, reason = ownership.send_signal_if_owned(rec3, signal.SIGTERM)
        assert sent is False and reason == "process_exited"
    finally:
        api3.queue.shutdown()


def test_p0a_rc01_comfyui_interrupt_is_prompt_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[tuple[str, dict[str, Any]]] = []

    class FakeResp:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return self._data

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *a: Any) -> None:
            pass

        def get(self, url: str) -> FakeResp:
            return FakeResp({"queue_running": [[0, "other_user_job"]], "queue_pending": [[1, "my_queued_job"]]})

        def post(self, url: str, json: dict[str, Any]) -> FakeResp:
            posts.append((url, json))
            return FakeResp({})

    import avatar_v2.providers.comfyui as comfy_mod

    monkeypatch.setattr(comfy_mod.httpx, "Client", FakeClient)
    prov = ComfyUIProvider(ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"))

    # Unscoped global interrupt refused by default!
    assert prov.interrupt() is False
    assert posts == []

    # Requesting cancel of a job NOT running and NOT pending does NOT interrupt other_user_job!
    assert prov.interrupt("my_finished_job") is False
    assert posts == []

    # Requesting cancel of queued job deletes ONLY that prompt via /queue (no global interrupt!)
    assert prov.interrupt("my_queued_job") is True
    assert posts == [("http://127.0.0.1:8188/queue", {"delete": ["my_queued_job"]})]

    # Requesting cancel of the running prompt sends prompt-scoped /interrupt
    posts.clear()
    assert prov.interrupt("other_user_job") is True
    assert posts == [("http://127.0.0.1:8188/interrupt", {"prompt_id": "other_user_job"})]


# --------------------------------------------------------------------------- P0-B (RC-02): Media Ownership


def test_p0b_rc02_external_original_master_and_links_never_unlinked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = _isolate_home(monkeypatch, tmp_path)
    external_dir = tmp_path / "owner_photos"
    ext_master = _make_png(external_dir / "owner_portrait.png", (190, 80, 90))
    ext_bytes = ext_master.read_bytes()
    ext_sha = hashlib.sha256(ext_bytes).hexdigest()

    api = desktop.DesktopAPI()
    try:
        # Attach external original as presenter master (legacy/imported external path)
        api._studio_profile()
        pick = api.studio_pick({"path": str(ext_master), "slot_id": "identity_front", "keep": True})
        assert pick["ok"] is True

        # 1. Calling studio_delete on external master DETACHES without deleting external bytes!
        res = api.studio_delete({"path": str(ext_master)})
        assert res["ok"] is True
        assert res["detached"] is True
        assert res["deleted_file"] is False
        assert ext_master.is_file()
        assert hashlib.sha256(ext_master.read_bytes()).hexdigest() == ext_sha
        master_after, _ = api._studio_master()
        assert master_after is None

        # 2. Explicit detach_only also preserves external original
        api.studio_pick({"path": str(ext_master), "slot_id": "identity_front", "keep": True})
        res_det = api.studio_delete({"path": str(ext_master), "detach_only": True})
        assert res_det["ok"] is True and res_det["detached"] is True and res_det["deleted_file"] is False
        assert ext_master.is_file()

        # 3. Symlink inside archive pointing to external original -> REFUSED, external file untouched!
        archive_dir = app_home / "archive" / "job_sym"
        archive_dir.mkdir(parents=True, exist_ok=True)
        sym_path = archive_dir / "sneaky_link.png"
        os.symlink(ext_master, sym_path)
        (app_home / "history" / "job_sym.json").write_text(
            json.dumps({"id": "job_sym", "status": "done", "outputs": [str(sym_path)]}), encoding="utf-8"
        )
        res_sym = api.studio_delete({"path": str(sym_path)})
        assert res_sym["ok"] is False
        assert res_sym["deleted_file"] is False
        assert ext_master.is_file()
        assert hashlib.sha256(ext_master.read_bytes()).hexdigest() == ext_sha

        # 4. Hard-link inside archive pointing to external original (st_nlink > 1) -> REFUSED!
        hard_path = archive_dir / "hard_link.png"
        os.link(ext_master, hard_path)
        (app_home / "history" / "job_hard.json").write_text(
            json.dumps({"id": "job_hard", "status": "done", "outputs": [str(hard_path)]}), encoding="utf-8"
        )
        res_hard = api.studio_delete({"path": str(hard_path)})
        assert res_hard["ok"] is False
        assert res_hard["ownership"] == "hardlinked_shared"
        assert ext_master.is_file()
        assert hard_path.is_file()
        hard_path.unlink()  # clean up test hardlink

        # 5. Verified managed archive output: failed unlink leaves history intact (no orphaning!)
        from avatar_v2.ownership import register_managed_asset

        managed_img = _make_png(app_home / "runs" / "outputs" / "job_own" / "gen_001.png", (60, 160, 90))
        register_managed_asset(desktop.MANAGED_ASSETS_PATH, managed_img, origin="managed_archive_copy", job_id="job_own")
        hist_file = app_home / "history" / "job_own.json"
        hist_file.write_text(
            json.dumps({"id": "job_own", "status": "done", "outputs": [str(managed_img)]}), encoding="utf-8"
        )

        orig_os_unlink = os.unlink

        def _fail_os_unlink(path: Any, *a: Any, **kw: Any) -> None:
            if str(path) == managed_img.name or str(path) == str(managed_img):
                raise PermissionError("simulated read-only filesystem")
            return orig_os_unlink(path, *a, **kw)

        monkeypatch.setattr(os, "unlink", _fail_os_unlink)
        res_fail = api.studio_delete({"path": str(managed_img)})
        assert res_fail["ok"] is False
        assert hist_file.is_file(), "History was mutated despite failed unlink!"
        assert managed_img.is_file()

        # 6. Restore unlink: managed archive output is cleanly deleted & history removed!
        monkeypatch.setattr(os, "unlink", orig_os_unlink)
        res_ok = api.studio_delete({"path": str(managed_img)})
        assert res_ok["ok"] is True
        assert res_ok["deleted_file"] is True
        assert not managed_img.exists()
        assert not hist_file.exists()
    finally:
        api.queue.shutdown()


# --------------------------------------------------------------------------- P1 (RC-03): JsBridge File & Settings Authority


def test_rc03_js_bridge_enforces_settings_allowlist_and_file_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app_home = _isolate_home(monkeypatch, tmp_path)
    secret_img = _make_png(tmp_path / "private_unselected" / "secret.png", (220, 40, 40))
    secret_vid = tmp_path / "private_unselected" / "secret.mp4"
    secret_vid.write_bytes(b"not-authorized-mp4")
    unicode_img = _make_png(tmp_path / "picked café" / "photo 日本.png", (40, 200, 120))

    api = desktop.DesktopAPI()
    bridge = JsBridge(api)
    try:
        # 1. Privileged engine/path settings REFUSED via JsBridge!
        bad_set = bridge.save_settings({"comfyui_python": "/tmp/fake_py", "comfyui_dir": "/tmp/fake_comfy"})
        assert bad_set["ok"] is False
        assert "comfyui_python" in bad_set["error"]

        # Allowed UI settings SUCCEED via JsBridge!
        good_set = bridge.save_settings({"runway_job_limit": 42, "gpu_autoclean": True})
        assert good_set["ok"] is True
        assert good_set["settings"]["runway_job_limit"] == 42

        # 2. Unselected external video, thumbnail, image, open_path & staging REFUSED via JsBridge!
        assert bridge.studio_video(str(secret_vid))["ok"] is False
        assert bridge.studio_thumb(str(secret_vid))["ok"] is False
        assert bridge.studio_image(str(secret_img))["ok"] is False
        assert bridge.open_path(str(secret_img))["ok"] is False
        # APP_HOME/desktop.json is NOT inside trusted media subdirectories!
        assert bridge.open_path(str(app_home / "desktop.json"))["ok"] is False

        # Forge staging via JsBridge refuses unselected external image!
        monkeypatch.setattr(api, "engine_status", lambda: {"ok": True, "state": "ready", "message": "ready"})
        bad_forge = bridge.forge_go({"action": "photo", "prompt": "Portrait", "image": str(secret_img)})
        assert bad_forge["ok"] is False
        assert "not selected" in bad_forge["error"].lower() or "installed" in bad_forge["error"].lower()

        # 3. Picker-authorized Unicode image IS authorized through JsBridge!
        monkeypatch.setattr(api, "choose_files", lambda *a, **kw: [str(unicode_img)])
        picked = bridge.pick_local_images({"workflow": "single_start"})
        assert picked["ok"] is True
        img_res = bridge.studio_image(str(unicode_img))
        assert img_res["ok"] is True
        assert img_res["width"] == 256
    finally:
        api.queue.shutdown()
