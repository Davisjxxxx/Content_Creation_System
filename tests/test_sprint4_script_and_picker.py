"""Sprint 4 comprehensive CPU-only and live Bubblewrap OS-isolation verification suite.

Covers Sections 2–12 of the Sprint 4 authorization:
- Shared local reference-image validation, access control, staging, and original preservation.
- Static AST inspection across all 5 classifications (with zero-execution proof).
- Safe Static Structured Prompt Extractor & deterministic spec materialization.
- Native Avatar V2 Python SDK + host GenerationBroker (schema, capability, policy, local-only,
  ResourcePlanner admission, generation limits, concurrency limits, pause/resume/stop).
- Real OS Bubblewrap isolation tests (network off, host files/env/GPU blocked, Unix socket IPC,
  fail-closed when bwrap unavailable, runaway subprocess cleanup).
- Sprint 3 carry-forward refinements (warm FLUX.2 tiled-VAE selection, Wan 2.2 CFG-aware compiler).
- Minimal Script mode & Browse Local Files frontend regression checks.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop, js_bridge
from avatar_v2.model_registry import DEFAULT_REGISTRY
from avatar_v2.prompt_compilers import compile_prompt_for_model
from avatar_v2.prompt_extractor import build_specs_from_extraction, extract_prompt_data
from avatar_v2.prompt_schema import GenerationIntent, PromptAttribute, StructuredPromptSpec
from avatar_v2.reference_picker import (
    LocalReferenceManager,
    default_browse_directory,
    validate_local_image,
    validate_reference_batch,
)
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceGuard, ResourceSnapshot, StorageCheck, StorageStatus
from avatar_v2.resource_planner import ResourcePlanner, WorkloadRequest
from avatar_v2.script_broker import BrokerSubmitContext, GenerationBroker
from avatar_v2.script_inspector import inspect_script, inspect_script_source
from avatar_v2.script_runner import (
    SandboxLimits,
    SandboxUnavailableError,
    ScriptRunManager,
    probe_sandbox_support,
)
from avatar_v2.sdk import AvatarV2Client, RunLimitReached, RunStopped, SDKError


def _make_img(path: Path, size: tuple[int, int] = (512, 512), fmt: str = "PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(90, 140, 210))
    img.save(path, format=fmt)
    return path


def _healthy_snapshot(gpu_free_mb: int = 1200) -> ResourceSnapshot:
    guard = ResourceGuard()
    host = HostMemoryStats(
        ram_total_mb=64000,
        ram_available_mb=40000,
        ram_available_pct=62.5,
        swap_total_mb=8192,
        swap_used_mb=100,
        swap_used_pct=1.2,
        pswpin=10,
        pswpout=20,
        psi_some_avg10=0.0,
        psi_full_avg10=0.0,
    )
    gpus = [GPUTelemetry(0, "RTX 4070", 8188 - gpu_free_mb, 8188, gpu_free_mb, temp_c=52, util_pct=5)]
    storage = StorageStatus("/sim", 915.0, 60.0, "normal")
    return guard.evaluate(host=host, gpus=gpus, storage=storage, now=100.0)


def _ok_storage() -> StorageCheck:
    return StorageCheck(True, "normal", 60.0, 0.05, 59.95, 15.0)


class _SimulatedQueue:
    """Thread-safe simulated RenderQueue for CPU-only broker & sandbox tests."""

    def __init__(self, tmp_path: Path, *, auto_complete: bool = True, delay_s: float = 0.0) -> None:
        self.tmp_path = tmp_path
        self.auto_complete = auto_complete
        self.delay_s = delay_s
        self._lock = threading.RLock()
        self._counter = 0
        self.jobs: dict[str, dict[str, Any]] = {}
        self.contexts: list[BrokerSubmitContext] = []

    def submit(self, ctx: BrokerSubmitContext) -> str:
        with self._lock:
            self._counter += 1
            jid = f"sim_{self._counter:04d}"
            self.contexts.append(ctx)
            out_file = self.tmp_path / f"{jid}.png"
            self.jobs[jid] = {"status": "queued", "outputs": [], "error": ""}

        if self.auto_complete:
            def _finish() -> None:
                if self.delay_s > 0:
                    time.sleep(self.delay_s)
                with self._lock:
                    if self.jobs[jid]["status"] == "cancelled":
                        return
                    self.jobs[jid]["status"] = "running"
                _make_img(out_file, (256, 256))
                with self._lock:
                    if self.jobs[jid]["status"] != "cancelled":
                        self.jobs[jid]["status"] = "done"
                        self.jobs[jid]["outputs"] = [str(out_file)]

            if self.delay_s > 0:
                threading.Thread(target=_finish, daemon=True).start()
            else:
                _finish()
        return jid

    def status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self.jobs.get(job_id) or {"status": "unknown", "outputs": []})

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job["status"] in {"queued", "running"}:
                job["status"] = "cancelled"
                return True
            return False


# --------------------------------------------------------------------------- Section 2: Shared local reference-image selection


def test_local_reference_validation_formats_missing_corrupted_and_multi(tmp_path: Path):
    png = _make_img(tmp_path / "portrait.png", (640, 896), "PNG")
    jpg = _make_img(tmp_path / "photo.jpg", (512, 768), "JPEG")
    webp = _make_img(tmp_path / "ref.webp", (512, 512), "WEBP")
    tiny = _make_img(tmp_path / "tiny.png", (32, 32), "PNG")
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not-a-valid-png-payload")
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")

    assert validate_local_image(png).ok is True
    assert validate_local_image(jpg).format == "JPEG"
    assert validate_local_image(webp).format == "WEBP"
    assert validate_local_image(tmp_path / "missing.png").ok is False
    assert "missing" in validate_local_image(tmp_path / "missing.png").error.lower()
    assert validate_local_image(corrupt).ok is False
    assert "corrupt" in validate_local_image(corrupt).error.lower()
    assert validate_local_image(txt).ok is False
    assert "unsupported" in validate_local_image(txt).error.lower()
    assert validate_local_image(tiny, workflow="image_to_video").ok is False

    # Multi-reference enforces workflow limit cleanly.
    batch_single = validate_reference_batch([png, jpg], workflow="single_start")
    assert len(batch_single["items"]) == 1 and batch_single["errors"]
    batch_multi = validate_reference_batch([png, jpg, webp], workflow="multi_reference")
    assert len(batch_multi["items"]) == 3 and not batch_multi["errors"]


def test_local_reference_manager_access_control_staging_and_original_preservation(tmp_path: Path):
    external_dir = tmp_path / "Pictures"
    trusted_dir = tmp_path / "app_home"
    staging_dir = trusted_dir / "staged_refs"
    ext_img = _make_img(external_dir / "vacation.png", (512, 768))
    unpicked_img = _make_img(external_dir / "private_unpicked.png", (512, 512))
    my_stuff_img = _make_img(trusted_dir / "archive" / "mine.png", (512, 512))

    orig_bytes = ext_img.read_bytes()
    orig_mtime = ext_img.stat().st_mtime_ns

    mgr = LocalReferenceManager(staging_dir=staging_dir, trusted_roots=[trusted_dir])
    # Frontend cannot access arbitrary host paths without native selection.
    assert mgr.is_path_authorized(unpicked_img) is False
    assert mgr.authorize_existing(unpicked_img).ok is False
    assert mgr.is_path_authorized(my_stuff_img) is True

    # Native selection authorizes ext_img.
    res = mgr.register_selected([ext_img], workflow="image_to_video")
    assert res["ok"] is True and len(res["items"]) == 1
    assert mgr.is_path_authorized(ext_img) is True

    # Staging creates one hash-deduplicated copy, preserves original untouched, and survives original removal.
    staged1 = mgr.stage_for_job(ext_img, workflow="image_to_video")
    staged2 = mgr.stage_for_job(ext_img, workflow="image_to_video")
    assert staged1["copied"] is True and staged2["copied"] is False
    assert staged1["staged_path"] == staged2["staged_path"]
    assert ext_img.read_bytes() == orig_bytes and ext_img.stat().st_mtime_ns == orig_mtime

    # Moving/deleting external original does NOT break already-staged job input.
    ext_img.unlink()
    assert Path(staged1["staged_path"]).is_file()
    assert mgr.authorize_existing(ext_img).ok is False  # Future selection fails gracefully!

    # Trusted My Stuff image is NOT duplicated when staged.
    staged_mine = mgr.stage_for_job(my_stuff_img, workflow="image_to_video")
    assert staged_mine["copied"] is False and staged_mine["staged_path"] == str(my_stuff_img.resolve())


def test_desktop_pick_local_images_cancellation_memory_and_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(desktop, "APP_HOME", tmp_path / "home")
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "home" / "desktop.json")
    monkeypatch.setattr(desktop, "HISTORY_DIR", tmp_path / "home" / "history")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)

    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)
    api._self_window = None

    pic_dir = tmp_path / "external_pics"
    img_a = _make_img(pic_dir / "a.png", (512, 512))
    img_b = _make_img(pic_dir / "b.jpg", (512, 512), "JPEG")
    unpicked = _make_img(pic_dir / "secret.png", (512, 512))

    # 1. Cancelled dialog returns cleanly.
    monkeypatch.setattr(api, "choose_files", lambda kind, multiple=True, directory="": [])
    cancelled = api.pick_local_images({"workflow": "image_to_video"})
    assert cancelled["ok"] is True and cancelled["cancelled"] is True and cancelled["items"] == []

    # 2. Selecting img_a remembers directory and authorizes studio_image preview.
    seen_dirs: list[str] = []

    def _choose_a(kind: str, multiple: bool = True, directory: str = "") -> list[str]:
        seen_dirs.append(directory)
        return [str(img_a)]

    monkeypatch.setattr(api, "choose_files", _choose_a)
    picked = api.pick_local_images({"workflow": "image_to_video"})
    assert picked["ok"] is True and picked["items"][0]["path"] == str(img_a.resolve())
    assert api.studio_image(str(img_a))["ok"] is True
    assert api.studio_image(str(unpicked))["ok"] is False  # Unauthorized host path blocked!

    # 3. Subsequent browse starts from remembered directory and replaces with img_b.
    def _choose_b(kind: str, multiple: bool = True, directory: str = "") -> list[str]:
        seen_dirs.append(directory)
        return [str(img_b)]

    monkeypatch.setattr(api, "choose_files", _choose_b)
    replaced = api.pick_local_images({"workflow": "image_to_video"})
    assert replaced["ok"] is True and replaced["items"][0]["path"] == str(img_b.resolve())
    assert seen_dirs[-1] == str(pic_dir.resolve())
    assert default_browse_directory(str(pic_dir)) == str(pic_dir)


# --------------------------------------------------------------------------- Section 4 & 5: Static AST inspection & Extractor


def test_ast_inspector_never_executes_and_classifies_all_five_categories(tmp_path: Path):
    marker = tmp_path / "pwned.txt"
    malicious = tmp_path / "side_effect.py"
    malicious.write_text(
        f"import os, subprocess, requests\n"
        f"open({str(marker)!r}, 'w').write('executed!')\n"
        f"eval('1+1')\n"
        f"k = os.environ['RUNWAYML_API_SECRET']\n",
        encoding="utf-8",
    )
    rep_review = inspect_script(malicious)
    assert not marker.exists(), "Static AST inspector MUST NEVER execute inspected code!"
    assert rep_review.ok is True and rep_review.syntax_valid is True
    assert rep_review.classification == "requires_review"
    assert rep_review.uses_subprocess and rep_review.uses_network and rep_review.uses_filesystem_writes
    assert rep_review.uses_dynamic_exec and rep_review.uses_sensitive_env
    assert "does NOT prove" in rep_review.safety_disclaimer

    # Compatible native SDK script
    native = inspect_script_source(
        "import avatar_v2_sdk\nclient = avatar_v2_sdk.connect()\nclient.generate_image('Studio portrait')\n",
        script_path="native.py",
    )
    assert native.classification == "compatible_native" and native.expected_backend == "avatar_v2_sdk"

    # Statically extractable pure prompt library
    extractable = inspect_script_source(
        "LIGHTING = ['soft window light', 'golden hour']\n"
        "ENVIRONMENT = ['modern studio', 'sunlit library']\n"
        "SCENE_RECIPES = {'lesson': {'lighting': 'soft studio', 'environment': 'minimalist office'}}\n",
        script_path="library.py",
    )
    assert extractable.classification == "statically_extractable"
    assert extractable.extraction_summary["has_extractable_data"] is True

    # Legacy A1111 script (continuous_generator_juggernaut.py style)
    legacy = inspect_script_source(
        "import requests\n"
        "URL = 'http://127.0.0.1:7860/sdapi/v1/txt2img'\n"
        "LIGHTING = ['(cinematic lighting:1.4)', 'soft window light']\n"
        "while True:\n"
        "    pass\n",
        script_path="legacy_a1111.py",
    )
    assert legacy.classification == "potentially_compatible_legacy"
    assert legacy.expected_backend == "a1111" and legacy.has_unbounded_loop is True

    # Unsupported syntax error & unsupported Runway cloud script
    bad_syntax = inspect_script_source("def broken(:\n    pass\n", script_path="bad.py")
    assert bad_syntax.ok is False and bad_syntax.syntax_valid is False and bad_syntax.classification == "unsupported"
    cloud = inspect_script_source("URL = 'https://api.dev.runwayml.com/v1/tasks'\n", script_path="cloud.py")
    assert cloud.classification == "unsupported" and cloud.expected_backend == "runway_cloud"


def test_static_prompt_extractor_sanitizes_filters_and_materializes_specs(tmp_path: Path):
    script = tmp_path / "prompts.py"
    script.write_text(
        "SUBJECTS = ['editorial portrait of an adult architect, (photorealistic:1.4), masterpiece, best quality', 'a 12 year old child']\n"
        "LIGHTING = ['warm golden hour light', 'soft north window light']\n"
        "NSFW_DEFAULTS = ['should never be imported']\n"
        "FORCE_AGE = True\n"
        "DYNAMIC_POOL = [x.upper() for x in LIGHTING]\n"
        "PROFILE = {'width': 896, 'height': 1280, 'steps': 20, 'cfg': 4.0}\n"
        "SCENE_RECIPES = {'desk': {'environment': 'sunlit desk', 'lighting': 'soft key light'}}\n"
        "PROMPT_TEMPLATE = '{SUBJECTS}, {LIGHTING}'\n",
        encoding="utf-8",
    )
    ext = extract_prompt_data(script)
    assert ext.ok is True and ext.has_extractable_data is True
    # SDXL weights stripped, quality spam stripped, minor filtered out, legacy sensitive symbols skipped.
    assert ext.categories["SUBJECTS"] == ["editorial portrait of an adult architect, photorealistic"]
    assert any(s["name"] == "NSFW_DEFAULTS" for s in ext.skipped_legacy)
    assert any(s["name"] == "FORCE_AGE" for s in ext.skipped_legacy)
    assert any(u["name"] == "DYNAMIC_POOL" for u in ext.unextractable)
    assert len(ext.profiles) == 1 and ext.profiles[0]["width"] == 896
    assert len(ext.recipes) == 1 and ext.recipes[0]["id"] == "desk"

    specs = build_specs_from_extraction(ext, count=3, seed=42, model_preference="flux2_9b")
    assert len(specs) == 3
    compiled = compile_prompt_for_model(specs[0], "flux2_9b")
    assert compiled.ok is True and "(photorealistic:1.4)" not in compiled.positive


# --------------------------------------------------------------------------- Section 6, 8, 9, 10: SDK + Broker limits/policy/pause/resume


def test_sdk_broker_enforces_validation_policy_cloud_lock_limits_concurrency_and_pause(tmp_path: Path):
    sim = _SimulatedQueue(tmp_path, auto_complete=True)
    ref_img = _make_img(tmp_path / "ref.png", (512, 768))
    broker = GenerationBroker(
        run_id="run_test_1",
        submit_fn=sim.submit,
        status_fn=sim.status,
        cancel_fn=sim.cancel,
        max_items=3,
        max_concurrent=1,
        allow_cloud=False,
        authorized_references={"/workspace/refs/ref.png": str(ref_img)},
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    client = AvatarV2Client(token=broker.token, transport=broker.handle_request)

    caps = client.capabilities()
    assert caps["local_only"] is True and caps["max_items"] == 3
    assert any(m["model_id"] == "flux2_9b" for m in caps["models"])

    # 1. Policy block (minor)
    with pytest.raises(SDKError) as exc_minor:
        client.generate_image("portrait of a 15 year old teenager in studio")
    assert exc_minor.value.code == "policy_blocked"

    # 2. Local-only cloud block
    with pytest.raises(SDKError) as exc_cloud:
        client.submit_prompt("portrait", model_id="runway_gen4_image")
    assert exc_cloud.value.code == "cloud_blocked"

    # 3. Model capability check (video task on still model)
    with pytest.raises(SDKError) as exc_cap:
        client.submit_prompt("she waves", task="video_generate", model_id="flux2_4b")
    assert exc_cap.value.code == "unsupported_capability"

    # 4. Unauthorized arbitrary host path blocked
    with pytest.raises(SDKError) as exc_path:
        client.generate_video("she nods", start_image="/etc/passwd", model_id="wan22_a14b")
    assert exc_path.value.code == "unauthorized_path"

    # 5. Authorized reference passes and queues video job cleanly
    vid = client.generate_video("She smiles and nods gently", start_image="/workspace/refs/ref.png", model_id="wan22_a14b")
    assert vid.status == "done" and len(vid.outputs) == 1

    # 6. Host-enforced run limit (max_items=3) cannot be exceeded
    client.generate_image("Studio portrait 2", model_id="flux2_4b")
    client.generate_image("Studio portrait 3", model_id="flux2_4b")
    with pytest.raises(RunLimitReached):
        client.generate_image("Studio portrait 4 should fail", model_id="flux2_4b")

    # 7. Default cap prevents requesting >10 without extended_authorized=True
    broker_default = GenerationBroker(
        run_id="run_cap",
        submit_fn=sim.submit,
        status_fn=sim.status,
        max_items=50,
        extended_authorized=False,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    assert broker_default.max_items == 10
    broker_ext = GenerationBroker(
        run_id="run_ext",
        submit_fn=sim.submit,
        status_fn=sim.status,
        max_items=50,
        extended_authorized=True,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    assert broker_ext.max_items == 50


def test_broker_concurrency_limit_and_pause_resume_stop_preserve_completed_outputs(tmp_path: Path):
    slow_sim = _SimulatedQueue(tmp_path, auto_complete=False)
    broker = GenerationBroker(
        run_id="run_conc",
        submit_fn=slow_sim.submit,
        status_fn=slow_sim.status,
        cancel_fn=slow_sim.cancel,
        max_items=5,
        max_concurrent=1,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    client = AvatarV2Client(token=broker.token, transport=broker.handle_request)
    first = client.generate_image("First item", wait=False)
    assert first.status == "queued"

    # Second concurrent submission is rejected while first is still active!
    with pytest.raises(SDKError) as exc_conc:
        client.generate_image("Concurrent bypass attempt", wait=False)
    assert exc_conc.value.code == "concurrency_limit"

    # Complete first job and test pause -> resume boundary.
    out1 = _make_img(tmp_path / "done1.png")
    slow_sim.jobs[first.job_id]["status"] = "done"
    slow_sim.jobs[first.job_id]["outputs"] = [str(out1)]

    broker.pause()
    submitted_after_resume: list[str] = []

    def _submit_while_paused() -> None:
        r = client.generate_image("Waits until resumed", wait=False)
        submitted_after_resume.append(r.job_id)

    t = threading.Thread(target=_submit_while_paused, daemon=True)
    t.start()
    time.sleep(0.15)
    assert submitted_after_resume == [], "Submission must block cleanly while paused!"
    broker.resume()
    t.join(timeout=3.0)
    assert len(submitted_after_resume) == 1

    # Stop cancels pending job 2, rejects further submissions, and preserves completed output 1!
    broker.stop(cancel_pending=True)
    snap = broker.snapshot_status()
    assert snap["state"] == "stopped"
    assert str(out1) in snap["outputs"]
    assert slow_sim.jobs[submitted_after_resume[0]]["status"] == "cancelled"
    with pytest.raises(RunStopped):
        client.generate_image("After stop", wait=False)


# --------------------------------------------------------------------------- Section 7: Live Bubblewrap OS isolation & fail-closed tests


def test_sandbox_fails_closed_when_bwrap_unavailable(tmp_path: Path):
    script = tmp_path / "ok_native.py"
    script.write_text("import avatar_v2_sdk\nprint('hi')\n", encoding="utf-8")
    sim = _SimulatedQueue(tmp_path)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        bwrap_override="/nonexistent/bwrap-binary",
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    sel = mgr.select_script(script)
    assert sel["ok"] is True and sel["executed"] is False and sel["sandbox"]["available"] is False
    with pytest.raises(SandboxUnavailableError):
        mgr.start_sandboxed_run(script_path=script)


def test_live_bubblewrap_isolation_blocks_network_secrets_gpu_and_allows_broker_ipc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    probe = probe_sandbox_support()
    if not probe.available:
        pytest.skip(f"Bubblewrap not available on this host: {probe.error}")

    # Inject a fake cloud credential into host env to verify --clearenv hides it completely.
    monkeypatch.setenv("RUNWAYML_API_SECRET", "rw_super_secret_should_never_leak")

    # Create a secret file on the host to verify filesystem isolation hides it.
    host_secret = tmp_path / "host_secret.txt"
    host_secret.write_text("TOP_SECRET", encoding="utf-8")

    # Open a loopback TCP listener on host to verify --unshare-net prevents reaching host loopback (e.g. ComfyUI).
    loop_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    loop_srv.bind(("127.0.0.1", 0))
    loop_srv.listen(1)
    loop_port = loop_srv.getsockname()[1]

    script = tmp_path / "sandbox_probe_native.py"
    script.write_text(
        f"import os, socket\n"
        f"import avatar_v2_sdk\n"
        f"assert 'RUNWAYML_API_SECRET' not in os.environ, 'host secret leaked!'\n"
        f"assert not os.path.exists({str(host_secret)!r}), 'host secret file visible!'\n"
        f"assert not os.path.exists('/dev/nvidia0'), 'GPU device exposed inside sandbox!'\n"
        f"try:\n"
        f"    s = socket.create_connection(('127.0.0.1', {loop_port}), timeout=0.5)\n"
        f"    s.close()\n"
        f"    raise AssertionError('Host loopback TCP reachable despite --unshare-net!')\n"
        f"except OSError:\n"
        f"    pass\n"
        f"client = avatar_v2_sdk.connect()\n"
        f"res = client.generate_image('Minimalist sunlit studio portrait of an adult designer', model_id='flux2_4b', wait=True)\n"
        f"assert res.status == 'done' and len(res.outputs) == 1\n"
        f"print('SANDBOX_VERIFIED_OK')\n",
        encoding="utf-8",
    )

    sim = _SimulatedQueue(tmp_path, auto_complete=True)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        cancel_fn=sim.cancel,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    try:
        # Without explicit review acknowledgement, scripts importing socket are held back.
        blocked = mgr.start_sandboxed_run(script_path=script, max_items=2, limits=SandboxLimits(wall_timeout_s=20))
        assert blocked["ok"] is False and "review" in blocked["error"].lower()

        started = mgr.start_sandboxed_run(
            script_path=script,
            max_items=2,
            limits=SandboxLimits(wall_timeout_s=20),
            allow_review_classification=True,
        )
        assert started["ok"] is True
        final = mgr.wait_for_run(timeout_s=20)
        assert final["record"]["status"] == "completed", final["record"].get("stderr_tail")
        assert "SANDBOX_VERIFIED_OK" in final["record"]["stdout_tail"]
        assert final["run"]["completed"] == 1
        assert (tmp_path / "runs" / f"{started['run_id']}.json").is_file()
    finally:
        loop_srv.close()


def test_live_bubblewrap_kills_runaway_script_cleanly(tmp_path: Path):
    probe = probe_sandbox_support()
    if not probe.available:
        pytest.skip(f"Bubblewrap not available on this host: {probe.error}")

    runaway = tmp_path / "runaway.py"
    runaway.write_text("import time, avatar_v2_sdk\nwhile True:\n    time.sleep(0.2)\n", encoding="utf-8")
    sim = _SimulatedQueue(tmp_path)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    started = mgr.start_sandboxed_run(script_path=runaway, limits=SandboxLimits(wall_timeout_s=1.2))
    assert started["ok"] is True
    final = mgr.wait_for_run(timeout_s=10)
    assert final["record"]["status"] == "error"
    assert "timeout" in final["record"]["error"].lower()


def test_static_extracted_run_executes_zero_script_code(tmp_path: Path):
    marker = tmp_path / "must_not_run.txt"
    legacy = tmp_path / "legacy_extract.py"
    legacy.write_text(
        f"open({str(marker)!r}, 'w').write('RAN')\n"
        "SUBJECTS = ['editorial portrait of an adult engineer']\n"
        "LIGHTING = ['soft diffused daylight', 'warm tungsten lamp light']\n",
        encoding="utf-8",
    )
    sim = _SimulatedQueue(tmp_path, auto_complete=True)
    mgr = ScriptRunManager(
        runs_dir=tmp_path / "runs",
        submit_fn=sim.submit,
        status_fn=sim.status,
        snapshot_override=_healthy_snapshot(),
        storage_check_override=_ok_storage(),
    )
    started = mgr.start_extracted_run(script_path=legacy, max_items=2, model_id="flux2_4b")
    assert started["ok"] is True
    final = mgr.wait_for_run(timeout_s=10)
    assert not marker.exists()
    assert final["record"]["status"] == "completed"
    assert final["record"]["execution_mode"] == "static_extracted"
    assert final["run"]["completed"] == 2


# --------------------------------------------------------------------------- Section 11: Carry-forward Sprint 3 refinements


def test_carry_forward_warm_flux2_avoids_unnecessary_tiled_vae_and_wan_compiler_respects_cfg():
    planner = ResourcePlanner()
    # Warm FLUX.2 9B at ~86% VRAM (1,150 MB free) at standard 896x1280 should keep standard VAE!
    warm_plan = planner.plan_execution(
        WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img", width=896, height=1280, frames=1),
        current_model_id="flux2_9b",
        snapshot=_healthy_snapshot(gpu_free_mb=1150),
        storage_check=_ok_storage(),
    )
    assert warm_plan.vae_strategy == "standard"
    assert warm_plan.respect_retention_above_threshold is True

    # High-res still (>1024x1280) still switches to tiled VAE automatically.
    hires_plan = planner.plan_execution(
        WorkloadRequest(model_id="flux2_9b", workflow="builtin:flux2_txt2img", width=1536, height=1536, frames=1),
        current_model_id="flux2_9b",
        snapshot=_healthy_snapshot(gpu_free_mb=1150),
        storage_check=_ok_storage(),
    )
    assert hires_plan.vae_strategy == "tiled"

    # Wan 2.2 prompt compiler adapts to 4-step CFG=1.0 vs strong-CFG Fun-Control.
    spec = StructuredPromptSpec(
        original_prompt="Instructor at desk",
        intent=GenerationIntent(task="video_generate"),
        motion=PromptAttribute(value="smiles and gestures calmly", state="specified", provenance="guided_ui"),
        excluded_attributes=["blurry", "watermark"],
    )
    cfg1 = compile_prompt_for_model(spec, "wan22_a14b", registry=DEFAULT_REGISTRY)
    strong = compile_prompt_for_model(spec, "wan22_funcontrol", registry=DEFAULT_REGISTRY)
    assert any("CFG 1.0" in w for w in cfg1.warnings)
    assert not any("CFG 1.0" in w for w in strong.warnings)
    assert "blurry" in strong.negative


# --------------------------------------------------------------------------- Section 2 & 3: Frontend regression


def test_frontend_has_three_input_modes_script_panel_and_browse_local_files():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'data-chips="forgeInputMode"' in html
    assert 'data-v="automatic"' in html and 'data-v="guided"' in html and 'data-v="script"' in html
    assert 'id="forgeScriptPanel"' in html
    assert "Select Python file" in html and "Browse local files" in html
    for method in ("pick_local_images", "script_select", "script_control"):
        assert method in js_bridge.JS_API_METHODS
        assert f"api.{method}(" in html
