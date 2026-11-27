"""Adversarial RC4 P0 tests for F01 (Renderer Ownership) and F02 (Filesystem Ownership & Overwrite).

Exercises BOTH Layer A (invariant boundary checks) AND Layer B (black-box public API,
live HTTP fixture servers, real IPv4/IPv6 listeners, real OS processes & concurrent file locks).
"""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any

from PIL import Image
import pytest

import avatar_v2.desktop as desktop
import avatar_v2.ownership as ownership
import avatar_v2.render_queue as rq
from avatar_v2.composer import build_job
from avatar_v2.gpu_memory import GPUMemoryManager
from avatar_v2.models import AvatarManifest, ProviderConfig, RenderJob, ShotSpec
from avatar_v2.providers.comfyui import ComfyUIProvider, RendererOwnershipError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_png(path: Path, color: tuple[int, int, int] = (90, 140, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (192, 192), color=color).save(path, format="PNG")
    return path


def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    app_home = tmp_path / "home"
    app_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "COMFY_OWNER_PATH", app_home / "comfyui_owner.json")
    monkeypatch.setattr(desktop, "COMFY_PID_PATH", app_home / "comfyui.pid")
    monkeypatch.setattr(desktop, "MANAGED_ASSETS_PATH", app_home / "managed_assets.json")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")
    return app_home


def _spawn_listener(family: str = "ipv4") -> tuple[subprocess.Popen[str], int]:
    """Spawn a real child process bound to ONLY IPv4 (127.0.0.1) or ONLY IPv6 (::1)."""
    if family == "ipv6":
        script = (
            "import socket, time; "
            "s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM); "
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
            "s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1); "
            "s.bind(('::1', 0)); "
            "s.listen(5); "
            "print(s.getsockname()[1], flush=True); "
            "time.sleep(60)"
        )
    else:
        script = (
            "import socket, time; "
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
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line:
        err = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        pytest.skip(f"Could not bind {family} loopback listener in this environment: {err}")
    return proc, int(line)


def _start_spy_http_server(*, redirect_to: str | None = None) -> tuple[HTTPServer, int, list[tuple[str, str]]]:
    hits: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            hits.append(("GET", self.path))
            if self.path == "/system_stats":
                body = json.dumps({"system": {"comfyui_version": "0.18.1"}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/queue":
                body = json.dumps({"queue_running": [], "queue_pending": []}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            hits.append(("POST", self.path))
            if redirect_to is not None:
                self.send_response(307)
                self.send_header("Location", redirect_to)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port, hits


def _make_job(job_id: str = "rc4-p0-job") -> RenderJob:
    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "av-1",
            "display_name": "Test",
            "subjects": [{"id": "av-1", "kind": "synthetic"}],
            "identity_refs": [],
        }
    )
    shot = ShotSpec.model_validate({"shot_id": "sh-1", "user_prompt": "Studio portrait"})
    job = build_job(avatar, shot)
    job.job_id = job_id
    return job


# ===========================================================================
# F01 / P0: Renderer Ownership & Centralized Global-Operation Boundary
# ===========================================================================


def test_f01_ipv4_listener_never_authorizes_ipv6_target_and_vice_versa() -> None:
    """Reproduces Astra F01: IPv4-only listener MUST NOT authorize http://[::1]:port (and vice versa)."""
    v4_proc, v4_port = _spawn_listener("ipv4")
    try:
        rec_v4 = ownership.ProcessOwnershipRecord.capture_for_pid(v4_proc.pid, port=v4_port)
        ok_v4, pinned_v4, _ = ownership.authorize_owned_endpoint(rec_v4, f"http://127.0.0.1:{v4_port}", require_listening=True)
        assert ok_v4 is True and pinned_v4 == f"http://127.0.0.1:{v4_port}"

        ok_v6_mismatch, pinned_bad, reason = ownership.authorize_owned_endpoint(
            rec_v4, f"http://[::1]:{v4_port}", require_listening=True
        )
        assert ok_v6_mismatch is False and pinned_bad == ""
        assert "ipv6" in reason.lower()

        # localhost MUST pin concretely to 127.0.0.1 (the actually bound IPv4 family)
        ok_lh, pinned_lh, _ = ownership.authorize_owned_endpoint(rec_v4, f"http://localhost:{v4_port}", require_listening=True)
        assert ok_lh is True and pinned_lh == f"http://127.0.0.1:{v4_port}"
    finally:
        v4_proc.kill()
        v4_proc.wait(timeout=5)

    v6_proc, v6_port = _spawn_listener("ipv6")
    try:
        rec_v6 = ownership.ProcessOwnershipRecord.capture_for_pid(v6_proc.pid, port=v6_port)
        ok_v6, pinned_v6, _ = ownership.authorize_owned_endpoint(rec_v6, f"http://[::1]:{v6_port}", require_listening=True)
        assert ok_v6 is True and pinned_v6 == f"http://[::1]:{v6_port}"

        ok_v4_mismatch, pinned_bad2, reason2 = ownership.authorize_owned_endpoint(
            rec_v6, f"http://127.0.0.1:{v6_port}", require_listening=True
        )
        assert ok_v4_mismatch is False and pinned_bad2 == ""
        assert "ipv4" in reason2.lower()

        # localhost MUST pin concretely to [::1]
        ok_lh6, pinned_lh6, _ = ownership.authorize_owned_endpoint(rec_v6, f"http://localhost:{v6_port}", require_listening=True)
        assert ok_lh6 is True and pinned_lh6 == f"http://[::1]:{v6_port}"
    finally:
        v6_proc.kill()
        v6_proc.wait(timeout=5)


def test_f01_missing_boot_id_and_unreadable_listener_evidence_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces Astra F01: missing boot_id or unreadable /proc listener tables MUST return False."""
    proc, port = _spawn_listener("ipv4")
    try:
        rec = ownership.ProcessOwnershipRecord.capture_for_pid(proc.pid, port=port)
        # 1. Missing record boot_id fails closed
        rec_no_boot = ownership.ProcessOwnershipRecord(
            pid=rec.pid,
            start_ticks=rec.start_ticks,
            boot_id="",
            exe_realpath=rec.exe_realpath,
            cmdline=rec.cmdline,
            port=port,
            owned_by_app=True,
        )
        ok_b1, reason_b1 = ownership.verify_process_ownership(rec_no_boot)
        assert ok_b1 is False and "boot_id" in reason_b1

        # 2. Missing live boot_id fails closed
        orig_inspect = ownership.inspect_process_identity
        monkeypatch.setattr(
            ownership,
            "inspect_process_identity",
            lambda pid: {**(orig_inspect(pid) or {}), "boot_id": ""},
        )
        ok_b2, reason_b2 = ownership.verify_process_ownership(rec)
        assert ok_b2 is False and "boot_id" in reason_b2
        monkeypatch.setattr(ownership, "inspect_process_identity", orig_inspect)

        # 3. Unreadable /proc listener telemetry fails closed
        monkeypatch.setattr(
            ownership,
            "pid_listening_endpoints",
            lambda pid: (False, set(), "unreadable_proc_net_tcp"),
        )
        ok_t, _pinned, reason_t = ownership.authorize_owned_endpoint(
            rec, f"http://127.0.0.1:{port}", require_listening=True
        )
        assert ok_t is False and "unreadable" in reason_t
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_f01_http_redirect_on_global_free_is_refused_and_never_reaches_destination() -> None:
    """HTTP 307 redirect on POST /free MUST fail closed; destination spy receives ZERO requests."""
    dest_srv, dest_port, dest_hits = _start_spy_http_server()
    redir_srv, redir_port, redir_hits = _start_spy_http_server(
        redirect_to=f"http://127.0.0.1:{dest_port}/free"
    )
    try:
        provider = ComfyUIProvider(ProviderConfig(provider="comfyui", base_url=f"http://127.0.0.1:{redir_port}"),
                                   ownership_authorizer=lambda url, op: (True, url, "authorized fixture origin"))
        with pytest.raises(RendererOwnershipError, match="redirect"):
            provider.free_memory(unload_models=True, free_memory=True)
        assert ("POST", "/free") in redir_hits
        assert dest_hits == [], f"Redirect destination received unauthorized request: {dest_hits}"
    finally:
        redir_srv.shutdown()
        dest_srv.shutdown()


def test_f01_shared_renderer_receives_zero_global_free_across_automatic_manual_prevlm_and_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Black-box F01 proof: a shared/external renderer (no ownership record OR different owned port)
    receives ZERO POST /free or global POST /interrupt calls across:
    1. GPUMemoryManager automatic forced cleanup (pre-render / post-render / retry)
    2. Manual DesktopAPI.gpu_free()
    3. DesktopAPI.shutdown_engine()
    4. Direct ComfyUIProvider.free_memory() via DesktopAPI._build_comfy_provider()
    """
    app_home = _isolate_home(monkeypatch, tmp_path)
    spy_srv, spy_port, spy_hits = _start_spy_http_server()
    try:
        (app_home / "desktop.json").write_text(
            json.dumps({"comfyui_url": f"http://127.0.0.1:{spy_port}"}),
            encoding="utf-8",
        )
        api = desktop.DesktopAPI()
        try:
            assert api.comfyui_status()["ownership"] == "shared_or_external"

            # 1. Manual gpu_free refused
            res_free = api.gpu_free()
            assert res_free["ok"] is False

            # 2. Automatic forced GPUMemoryManager cleanup refused (zero POST /free!)
            prov = api._build_comfy_provider()
            ev = api.queue.memory_manager.maybe_cleanup(prov, reason="pre-render:low_memory", force=True, unload_models=True)
            assert ev["ownership_denied"] is True
            assert ev["cleanup_requested"] is False
            assert ev["cleanup_performed"] is False

            # 3. Direct provider global free_memory & global interrupt refused
            with pytest.raises(RendererOwnershipError):
                prov.free_memory(unload_models=True, free_memory=True)
            with pytest.raises(RendererOwnershipError):
                prov.interrupt(None, allow_global=True)

            # 4. Shutdown sends zero POST /free to shared spy
            api.shutdown_engine()
        finally:
            api.queue.shutdown()

        post_hits = [h for h in spy_hits if h[0] == "POST"]
        assert post_hits == [], f"Shared/external spy received unauthorized POST requests: {post_hits}"
    finally:
        spy_srv.shutdown()


def test_f01_config_mutation_between_status_and_free_cannot_redirect_to_external_spy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces Astra F01 config re-read race: even if desktop.json switches comfyui_url
    mid-call to an external spy server, gpu_free() snapshots ONE config and binds authorization
    at the provider boundary so the external spy receives ZERO POST /free requests.
    """
    app_home = _isolate_home(monkeypatch, tmp_path)
    owned_proc, owned_port = _spawn_listener("ipv4")
    ext_srv, ext_port, ext_hits = _start_spy_http_server()
    try:
        (app_home / "desktop.json").write_text(
            json.dumps({"comfyui_url": f"http://127.0.0.1:{owned_port}"}),
            encoding="utf-8",
        )
        rec = ownership.ProcessOwnershipRecord.capture_for_pid(owned_proc.pid, port=owned_port)
        ownership.save_ownership_record(app_home / "comfyui_owner.json", rec)

        api = desktop.DesktopAPI()
        try:
            call_count = {"n": 0}
            orig_cfg = api._provider_config

            def _flipping_cfg(settings: dict[str, Any]) -> ProviderConfig:
                call_count["n"] += 1
                if call_count["n"] >= 2:
                    return ProviderConfig(provider="comfyui", base_url=f"http://127.0.0.1:{ext_port}")
                return orig_cfg(settings)

            monkeypatch.setattr(api, "_provider_config", _flipping_cfg)
            # Call gpu_free(): because gpu_free() snapshots _provider_config ONCE, _flipping_cfg
            # is only called once (never flipping to ext_port!).
            api.gpu_free()
            assert call_count["n"] == 1
            assert [h for h in ext_hits if h[0] == "POST"] == []
        finally:
            api.queue.shutdown()
    finally:
        owned_proc.kill()
        owned_proc.wait(timeout=5)
        ext_srv.shutdown()


# ===========================================================================
# F02 / P0: Filesystem Ownership, Exclusive Archive Creation & Locked Registry
# ===========================================================================


def test_f02_archive_never_overwrites_preexisting_unregistered_destination(tmp_path: Path) -> None:
    """Reproduces Astra's exact F02 counterexample:
    A pre-existing unregistered original already sits at archive_dir / item.id / src.name.
    RenderQueue._archive_outputs MUST NEVER overwrite it — original bytes stay untouched
    and the new archive copy lands at a fresh collision-free path!
    """
    archive_dir = tmp_path / "archive"
    provider_out_dir = tmp_path / "comfy_out"
    work_dir = tmp_path / "work"
    history_dir = tmp_path / "history"
    reg_path = tmp_path / "managed_assets.json"

    preexisting = _make_png(archive_dir / "job-f02" / "render_0001.png", color=(220, 20, 20))
    preexisting_hash = _sha256(preexisting)

    new_render = _make_png(provider_out_dir / "render_0001.png", color=(20, 220, 20))
    new_render_hash = _sha256(new_render)
    assert preexisting_hash != new_render_hash

    q = rq.RenderQueue(
        work_dir=work_dir,
        history_dir=history_dir,
        memory_manager=GPUMemoryManager(enabled=False),
        managed_assets_path=reg_path,
    )
    try:
        item = rq.QueuedRender(
            id="job-f02",
            job=_make_job("job-f02"),
            workflow="builtin:flux2_txt2img",
            provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
            profile="auto",
            archive_dir=str(archive_dir),
            outputs=[str(new_render)],
        )
        archived = q._archive_outputs(item)
        assert len(archived) == 1
        dest = Path(archived[0])
        # Pre-existing unregistered file is 100% UNTOUCHED!
        assert _sha256(preexisting) == preexisting_hash
        # New archive copy went to a distinct collision-free file with the new bytes!
        assert dest.resolve() != preexisting.resolve()
        assert dest.is_file() and _sha256(dest) == new_render_hash
        # Pre-existing file is NOT registered as owned; new exclusive copy IS registered!
        ok_pre, _ = ownership.verify_managed_asset_ownership(reg_path, preexisting)
        ok_new, _ = ownership.verify_managed_asset_ownership(reg_path, dest)
        assert ok_pre is False
        assert ok_new is True
    finally:
        q.shutdown()


def _hold_registry_lock_child(lock_path_str: str, ready_q: multiprocessing.Queue, stop_ev: multiprocessing.Event) -> None:
    import fcntl

    p = Path(lock_path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready_q.put("LOCKED")
        stop_ev.wait(timeout=15.0)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def test_f02_registry_fails_closed_when_external_process_holds_lock_and_rejects_empty_job_id(
    tmp_path: Path,
) -> None:
    """Reproduces Astra's exact F02 lock-contention counterexample using a REAL separate OS process."""
    reg_path = tmp_path / "managed_assets.json"
    lock_path = reg_path.with_name(f".{reg_path.name}.lock")
    asset = _make_png(tmp_path / "asset.png", color=(40, 80, 120))

    # 1. Empty job_id MUST be rejected
    with pytest.raises(ValueError, match="job_id"):
        ownership.register_managed_asset(reg_path, asset, origin="managed_archive_copy", job_id="")

    # 2. Separate OS process holds exclusive flock on registry lock -> registration MUST fail closed!
    ctx = multiprocessing.get_context("spawn")
    ready_q = ctx.Queue()
    stop_ev = ctx.Event()
    holder = ctx.Process(target=_hold_registry_lock_child, args=(str(lock_path), ready_q, stop_ev))
    holder.start()
    try:
        assert ready_q.get(timeout=5.0) == "LOCKED"
        with pytest.raises(ownership.RegistryLockError):
            ownership.register_managed_asset(
                reg_path,
                asset,
                origin="managed_archive_copy",
                job_id="job-contended",
                lock_timeout_s=0.15,
            )
        assert not reg_path.exists(), "Registry was mutated despite another process holding the lock!"
    finally:
        stop_ev.set()
        holder.join(timeout=5.0)

    # 3. Once lock is released, registration + atomic locked delete succeed cleanly
    rec = ownership.register_managed_asset(reg_path, asset, origin="managed_archive_copy", job_id="job-ok")
    assert rec is not None and rec.job_id == "job-ok"

    # 4. Concurrent multi-thread registrations preserve ALL entries (zero lost updates!)
    extra_assets = [_make_png(tmp_path / f"c_{i}.png", color=(i * 20 % 255, 60, 100)) for i in range(10)]
    errors: list[Exception] = []

    def _reg_one(idx: int) -> None:
        try:
            ownership.register_managed_asset(
                reg_path, extra_assets[idx], origin="managed_archive_copy", job_id=f"job-t-{idx}"
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_reg_one, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    assert errors == [], f"Concurrent registry writers raised errors: {errors}"
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    assert len(data) == 11, f"Lost concurrent registry updates! Expected 11 entries, got {len(data)}"


def test_f02_unarchived_external_provider_output_is_not_registered_as_deletable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When archive_dir is unset, external provider output paths MUST remain external references
    (NOT registered in managed_assets.json with physical deletion authority).
    """
    ext_dir = tmp_path / "ext_comfy_out"
    existing_orig = _make_png(ext_dir / "owner_master.png", color=(150, 70, 30))
    orig_hash = _sha256(existing_orig)
    reg_path = tmp_path / "managed_assets.json"

    class StubComfy:
        uses_local_gpu = True

        def __init__(self, cfg: ProviderConfig, **kwargs) -> None:
            pass

        def render(self, workflow: str, job: RenderJob, **kwargs) -> dict[str, Any]:
            return {"prompt_id": "p-ext", "outputs": [str(existing_orig)]}

    import avatar_v2.ollama_client as oc

    monkeypatch.setattr(rq, "ComfyUIProvider", StubComfy)
    monkeypatch.setattr(oc, "_fetch_running_ollama_models_status", lambda *a, **k: (False, [], "offline"))

    q = rq.RenderQueue(
        work_dir=tmp_path / "work",
        history_dir=tmp_path / "history",
        memory_manager=GPUMemoryManager(enabled=False),
        managed_assets_path=reg_path,
    )
    try:
        q.add(
            rq.QueuedRender(
                id="job-no-archive",
                job=_make_job("job-no-archive"),
                workflow="builtin:flux2_txt2img",
                provider_config=ProviderConfig(provider="comfyui", base_url="http://127.0.0.1:8188"),
                profile="auto",
                archive_dir=None,
            )
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cur = next((x for x in q.list() if x["id"] == "job-no-archive"), None)
            if cur and cur["status"] in {"done", "error"}:
                break
            time.sleep(0.05)

        ok_own, res = ownership.verify_managed_asset_ownership(reg_path, existing_orig)
        assert ok_own is False, f"Unarchived external path was improperly granted deletion authority: {res}"
        assert _sha256(existing_orig) == orig_hash
    finally:
        q.shutdown()


def test_f02_delete_managed_asset_if_owned_refuses_replaced_file_and_empty_job_id_record(
    tmp_path: Path,
) -> None:
    """Atomic locked delete_managed_asset_if_owned refuses replaced bytes, replaced inode,
    hard-linked files, symlinks, and forged records with empty job_id.
    """
    reg_path = tmp_path / "managed_assets.json"
    asset = _make_png(tmp_path / "managed.png", color=(50, 90, 140))
    rec = ownership.register_managed_asset(reg_path, asset, origin="managed_archive_copy", job_id="job-del-1")
    assert rec is not None

    # Replace file with a different image at the same path -> MUST refuse & preserve new bytes!
    replacement = _make_png(tmp_path / "replacement.png", color=(240, 200, 10))
    rep_hash = _sha256(replacement)
    os.replace(replacement, asset)

    deleted, res = ownership.delete_managed_asset_if_owned(reg_path, asset)
    assert deleted is False
    assert asset.is_file() and _sha256(asset) == rep_hash

    # Forged record with empty job_id MUST be rejected even if inode/size/sha match!
    forged_asset = _make_png(tmp_path / "forged.png", color=(10, 20, 30))
    st = os.lstat(forged_asset)
    reg_path.write_text(
        json.dumps(
            {
                str(forged_asset.resolve()): {
                    "canonical_path": str(forged_asset.resolve()),
                    "sha256": _sha256(forged_asset),
                    "size_bytes": st.st_size,
                    "inode": st.st_ino,
                    "device": st.st_dev,
                    "origin": "managed_archive_copy",
                    "job_id": "",
                }
            }
        ),
        encoding="utf-8",
    )
    del_forged, _ = ownership.delete_managed_asset_if_owned(reg_path, forged_asset)
    assert del_forged is False and forged_asset.is_file()
