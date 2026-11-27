"""Adversarial reproductions for Astra RC2 P0/P1 Gates 1 & 2 (RC-01, RC-02, RC-03).

Reproduces every exact counterexample reported in ASTRA_RC2_INDEPENDENT_RECHECK.md:
- RC-02 (P0): External original located INSIDE configured preferred_output_dir, selected as
              presenter master, then targeted by ordinary studio_delete() -> MUST NOT delete file!
              Also tests copied, moved, overwritten/replaced (inode/SHA mismatch), symlinked,
              hard-linked, and forged-history assets.
- RC-01 (P0): Valid fixture process record for port 18189 + healthy endpoint redirected to 28189 ->
              comfyui_status() MUST report shared_or_external, and shutdown_engine()/gpu_free()
              MUST send ZERO HTTP requests to 28189! Also tests atomic pidfd restart signalling.
- RC-03 (P1): Page-facing JsBridge studio_presenter_add() and studio_pick() MUST refuse arbitrary
              unselected host image pathnames.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop
from avatar_v2.js_bridge import JsBridge
from avatar_v2.ownership import (
    ProcessOwnershipRecord,
    register_managed_asset,
    save_ownership_record,
    send_signal_if_owned,
    verify_endpoint_ownership,
)


def _make_png(path: Path, color: tuple[int, int, int] = (120, 80, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color).save(path, format="PNG")
    return path


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, settings_extra: dict[str, Any] | None = None) -> Path:
    app_home = tmp_path / "avatar_home"
    app_home.mkdir(parents=True, exist_ok=True)
    for name, val in {
        "APP_HOME": app_home,
        "CONFIG_PATH": app_home / "desktop.json",
        "RUNS_DIR": app_home / "runs",
        "QUEUE_DIR": app_home / "queue",
        "HISTORY_DIR": app_home / "history",
        "LIBRARY_DIR": app_home / "library",
        "COMFY_PID_PATH": app_home / "comfyui.pid",
        "COMFY_OWNER_PATH": app_home / "comfyui_owner.json",
        "COMFY_LOG_PATH": app_home / "logs" / "comfyui.log",
        "CLOUD_LEDGER_PATH": app_home / "cloud_ledger.jsonl",
        "MANAGED_ASSETS_PATH": app_home / "managed_assets.json",
    }.items():
        monkeypatch.setattr(desktop, name, val)
    cfg = {**desktop.DEFAULT_SETTINGS, "comfyui_url": "http://127.0.0.1:28199", **(settings_extra or {})}
    desktop._write_json(app_home / "desktop.json", cfg)
    return app_home


def test_gate1_rc02_astra_preferred_output_dir_presenter_master_original_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exact Astra RC2 P0 counterexample: original photo inside preferred_output_dir set as presenter master."""
    owner_dir = tmp_path / "OwnerPhotos"
    original = _make_png(owner_dir / "synthetic_original.png", color=(210, 45, 70))
    orig_sha = hashlib.sha256(original.read_bytes()).hexdigest()

    app_home = _isolate(monkeypatch, tmp_path, settings_extra={"preferred_output_dir": str(owner_dir)})
    api = desktop.DesktopAPI()
    try:
        # Select the synthetic original as presenter master (zero history matches).
        pick = api.studio_pick({"path": str(original), "slot_id": "identity_front", "keep": True})
        assert pick["ok"] is True

        # Ordinary library deletion (NOT detach_only) MUST detach cleanly and NEVER delete the original!
        res = api.studio_delete({"path": str(original)})
        assert res["ok"] is True
        assert res["detached"] is True
        assert res["deleted_file"] is False
        assert original.is_file()
        assert hashlib.sha256(original.read_bytes()).hexdigest() == orig_sha

        # Even if a forged history JSON references the original inside preferred_output_dir,
        # without a verified ManagedAssetRegistry record it STILL cannot be deleted!
        (app_home / "history" / "forged.json").write_text(
            json.dumps({"id": "forged", "status": "done", "outputs": [str(original)]}),
            encoding="utf-8",
        )
        res_forged = api.studio_delete({"path": str(original)})
        assert res_forged["ok"] is False
        assert res_forged["deleted_file"] is False
        assert original.is_file()
        assert hashlib.sha256(original.read_bytes()).hexdigest() == orig_sha
    finally:
        api.queue.shutdown()


def test_gate1_rc02_copied_moved_replaced_symlink_and_hardlink_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test copied, moved, replaced-over-same-path, symlinked, and hardlinked assets against durable ownership."""
    app_home = _isolate(monkeypatch, tmp_path)
    out_dir = app_home / "runs" / "outputs" / "j1"
    api = desktop.DesktopAPI()
    try:
        # 1. Legitimate generated output registered by Avatar V2
        gen = _make_png(out_dir / "gen.png", color=(50, 150, 200))
        rec = register_managed_asset(desktop.MANAGED_ASSETS_PATH, gen, origin="generated_output", job_id="j1")
        assert rec is not None

        # 2. Copied asset at a new path has NO ownership record -> refused!
        copied = out_dir / "gen_copy.png"
        shutil.copy2(gen, copied)
        (app_home / "history" / "h_copy.json").write_text(
            json.dumps({"id": "h_copy", "status": "done", "outputs": [str(copied)]}), encoding="utf-8"
        )
        r_copy = api.studio_delete({"path": str(copied)})
        assert r_copy["ok"] is False and copied.is_file()

        # 3. Replaced-over-same-path (user overwrites gen.png with an external original!) -> inode/SHA mismatch -> refused!
        user_orig = _make_png(tmp_path / "external_replacement.png", color=(255, 220, 10))
        os.replace(user_orig, gen)
        (app_home / "history" / "h_gen.json").write_text(
            json.dumps({"id": "h_gen", "status": "done", "outputs": [str(gen)]}), encoding="utf-8"
        )
        r_repl = api.studio_delete({"path": str(gen)})
        assert r_repl["ok"] is False and gen.is_file()

        # 4. Moved registered asset -> refused!
        gen2 = _make_png(out_dir / "gen2.png", color=(10, 200, 10))
        register_managed_asset(desktop.MANAGED_ASSETS_PATH, gen2, origin="generated_output", job_id="j2")
        moved = out_dir / "gen2_moved.png"
        gen2.rename(moved)
        (app_home / "history" / "h_moved.json").write_text(
            json.dumps({"id": "h_moved", "status": "done", "outputs": [str(moved)]}), encoding="utf-8"
        )
        r_moved = api.studio_delete({"path": str(moved)})
        assert r_moved["ok"] is False and moved.is_file()

        # 5. Hard-linked registered asset (user creates hardlink after generation) -> st_nlink > 1 -> refused!
        gen3 = _make_png(out_dir / "gen3.png", color=(90, 90, 240))
        register_managed_asset(desktop.MANAGED_ASSETS_PATH, gen3, origin="generated_output", job_id="j3")
        hard_ext = tmp_path / "user_hardlink.png"
        os.link(gen3, hard_ext)
        (app_home / "history" / "h_hard.json").write_text(
            json.dumps({"id": "h_hard", "status": "done", "outputs": [str(gen3)]}), encoding="utf-8"
        )
        r_hard = api.studio_delete({"path": str(gen3)})
        assert r_hard["ok"] is False and gen3.is_file() and hard_ext.is_file()

        # 6. Symlink inside output dir -> refused!
        sym = out_dir / "link.png"
        sym.symlink_to(hard_ext)
        r_sym = api.studio_delete({"path": str(sym)})
        assert r_sym["ok"] is False and sym.is_symlink() and hard_ext.is_file()

        # 7. Genuine unmodified registered asset WITH history -> cleanly deleted!
        gen4 = _make_png(out_dir / "gen4.png", color=(33, 66, 99))
        register_managed_asset(desktop.MANAGED_ASSETS_PATH, gen4, origin="managed_archive_copy", job_id="j4")
        (app_home / "history" / "h_ok.json").write_text(
            json.dumps({"id": "h_ok", "status": "done", "outputs": [str(gen4)]}), encoding="utf-8"
        )
        r_ok = api.studio_delete({"path": str(gen4)})
        assert r_ok["ok"] is True and r_ok["deleted_file"] is True and not gen4.exists()
    finally:
        api.queue.shutdown()


def test_gate2_rc01_endpoint_redirection_18189_to_28189_and_atomic_restart_pidfd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exact Astra RC2 P0 counterexample: owned process on port A, healthy endpoint redirected to port B."""
    # Pick two free loopback ports dynamically (mirroring 18189 -> 28189)
    with socket.socket() as s1, socket.socket() as s2:
        s1.bind(("127.0.0.1", 0))
        s2.bind(("127.0.0.1", 0))
        owned_port = s1.getsockname()[1]
        external_port = s2.getsockname()[1]

    # Start a harmless disposable child listening on owned_port (simulates app-owned ComfyUI on 18189)
    child_code = (
        "import socket, time; "
        f"s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        f"s.bind(('127.0.0.1', {owned_port})); s.listen(5); time.sleep(60)"
    )
    owned_child = subprocess.Popen([sys.executable, "-c", child_code])

    # Start a healthy external HTTP spy server on external_port (simulates external ComfyUI on 28189)
    spy_requests: list[tuple[str, str]] = []

    class _SpyHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            spy_requests.append(("GET", self.path))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"system": {"comfyui_version": "0.18.1"}}')

        def do_POST(self) -> None:
            spy_requests.append(("POST", self.path))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    ext_server = ThreadingHTTPServer(("127.0.0.1", external_port), _SpyHandler)
    ext_thread = threading.Thread(target=ext_server.serve_forever, daemon=True)
    ext_thread.start()

    try:
        # Wait briefly for owned_child to bind owned_port
        time.sleep(0.25)
        app_home = _isolate(
            monkeypatch,
            tmp_path,
            settings_extra={"comfyui_url": f"http://127.0.0.1:{external_port}"},
        )
        owner_rec = ProcessOwnershipRecord.capture_for_pid(owned_child.pid, port=owned_port)
        save_ownership_record(desktop.COMFY_OWNER_PATH, owner_rec)

        api = desktop.DesktopAPI()
        try:
            # 1. Endpoint ownership verification MUST reject port mismatch!
            ep_ok, ep_reason = verify_endpoint_ownership(
                owner_rec, f"http://127.0.0.1:{external_port}", require_listening=True
            )
            assert ep_ok is False
            assert "endpoint_port_mismatch" in ep_reason

            # 2. comfyui_status() MUST report shared_or_external (NEVER owned_private!)
            st = api.comfyui_status()
            assert st["ready"] is True
            assert st["ownership"] == "shared_or_external"
            assert st["managed"] is False

            # 3. gpu_free() and comfyui_stop() and shutdown_engine() MUST send ZERO POSTs to external_port!
            spy_requests.clear()
            free_res = api.gpu_free()
            assert free_res["ok"] is False
            stop_res = api.comfyui_stop(wait_s=0.5, force=True)
            assert stop_res["ok"] is False
            assert stop_res["signalled"] is False
            api.shutdown_engine()

            post_paths = [path for method, path in spy_requests if method == "POST"]
            assert post_paths == [], f"Unauthorized POST sent to external endpoint {external_port}: {post_paths}"
            # Owned child was not spuriously killed when redirected endpoint was external
            assert owned_child.poll() is None

            # 4. Restart signalling (handle=None) uses atomic kernel pidfd_open -> verify -> pidfd_send_signal!
            ok_own, _ = verify_endpoint_ownership(owner_rec, f"http://127.0.0.1:{owned_port}", require_listening=True)
            assert ok_own is True
            sent, sig_reason = send_signal_if_owned(owner_rec, signal.SIGTERM, handle=None)
            assert sent is True
            assert sig_reason == "signalled_via_reopened_pidfd"
            owned_child.wait(timeout=5.0)
        finally:
            api.queue.shutdown()
    finally:
        ext_server.shutdown()
        ext_server.server_close()
        if owned_child.poll() is None:
            owned_child.kill()
            owned_child.wait(timeout=3.0)


def test_gate1_rc03_js_bridge_refuses_unselected_presenter_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Page-facing JsBridge studio_presenter_add and studio_pick refuse arbitrary host image pathnames."""
    _isolate(monkeypatch, tmp_path)
    secret_img = _make_png(tmp_path / "outside_host" / "private.png", color=(10, 20, 30))

    api = desktop.DesktopAPI()
    try:
        bridge = JsBridge(api)
        r_add = bridge.studio_presenter_add({"path": str(secret_img)})
        assert r_add["ok"] is False
        assert "not generated" in r_add["error"].lower() or "picker" in r_add["error"].lower()

        r_pick = bridge.studio_pick({"path": str(secret_img), "slot_id": "identity_front", "keep": True})
        assert r_pick["ok"] is False
        assert "not generated" in r_pick["error"].lower() or "picker" in r_pick["error"].lower()
        assert api._studio_master() == (None, None)
    finally:
        api.queue.shutdown()
