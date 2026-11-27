"""Release gates using generated media, temporary state and an instrumented loopback renderer."""
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from unittest.mock import patch

import pytest
from PIL import Image

from avatar_v2 import gpu_memory, ownership
from avatar_v2.inspection_contract import present_job
from avatar_v2.media_validation import validate_media
from avatar_v2.models import ProviderConfig
from avatar_v2.ollama_client import _complete_residency
from avatar_v2.providers.comfyui import ComfyUIProvider, RendererOwnershipError
from avatar_v2.reference_picker import LocalReferenceManager
from avatar_v2.render_queue import AdmissionRejected, RenderQueue
from avatar_v2.safe_output import publish_output
from test_render_queue_lifecycle import _item


def test_publish_collision_after_producer_starts_and_partial_production(tmp_path):
    target = tmp_path / "scene.png"
    def produce(staged):
        assert staged.parent.stat().st_mode & 0o777 == 0o700
        target.write_bytes(b"existing media")
        staged.write_bytes(b"new media")
    sibling = publish_output(target, produce)
    assert sibling != target and sibling.read_bytes() == b"new media"
    def interrupted(staged):
        staged.write_bytes(b"partial")
        raise OSError("fixture write failure")
    with pytest.raises(OSError):
        publish_output(target, interrupted)
    assert target.read_bytes() == b"existing media"
    assert set(tmp_path.iterdir()) == {target, sibling}


def test_direct_queue_and_connected_renderer_authority(tmp_path):
    requests = []
    class Renderer(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Renderer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    queue = RenderQueue(tmp_path)
    try:
        cfg = ProviderConfig(base_url=f"http://127.0.0.1:{server.server_port}")
        for provider in (ComfyUIProvider(cfg), queue._make_provider(cfg)):
            with pytest.raises(RendererOwnershipError):
                provider.free_memory()
            with pytest.raises(RendererOwnershipError):
                provider.interrupt(allow_global=True)
        record = ownership.ProcessOwnershipRecord.capture_for_pid(os.getpid(), port=server.server_port)
        def authorize(url, operation, *, connection=None):
            if connection and not ownership.owns_connected_peer(record, *connection):
                return False, "", "connected peer unavailable"
            return ownership.authorize_owned_endpoint(record, url)
        provider = ComfyUIProvider(cfg, ownership_authorizer=authorize)
        with patch.object(ownership, "owns_connected_peer", return_value=False) as census:
            with pytest.raises(RendererOwnershipError):
                provider.free_memory()
            assert census.called
        assert requests == []
        provider.free_memory()
        assert requests == ["/free"]
    finally:
        queue.shutdown()
        server.shutdown()
        thread.join()
        server.server_close()


def test_uncommitted_batch_and_failed_rollback_never_restore(tmp_path):
    queue = RenderQueue(tmp_path)
    queue.shutdown()
    items = [_item(tmp_path), _item(tmp_path)]
    with patch.object(queue, "_commit_admission", side_effect=OSError("no journal commit")), \
         patch.object(queue, "_drop_item_file", side_effect=OSError("read only")):
        with pytest.raises(AdmissionRejected):
            queue.add_many(items)
    assert len(list((tmp_path / "items").glob("*.json"))) == 2
    with patch.object(RenderQueue, "_execute") as execute:
        restored = RenderQueue(tmp_path)
        try:
            assert restored.list() == []
            with pytest.raises(RuntimeError, match="writer lock"):
                queue.add(_item(tmp_path))
        finally:
            restored.shutdown()
        execute.assert_not_called()


@pytest.mark.parametrize("data", [b"x", (Path(__file__).parent / "fixtures/rc5/still.png").read_bytes()[:32]])
def test_decoder_rejects_incomplete_image(data, tmp_path):
    bad = tmp_path / "frame.png"
    bad.write_bytes(data)
    with pytest.raises((ValueError, OSError)):
        validate_media(bad)
    Image.new("RGB", (12, 16)).save(bad, format="JPEG")
    with pytest.raises(ValueError, match="format"):
        validate_media(bad)


def test_public_history_never_reuses_cached_verdict_or_conflicting_state(tmp_path):
    output = tmp_path / "output.png"
    Image.new("RGB", (16, 16)).save(output)
    fake = {"output_path": str(output), "ui_status": "requirements_verified", "ui_status_label": "Requirements verified"}
    saved = {"status": "done", "phase": "complete", "outputs": [str(output)],
             "candidate_qa": {"recovery_required": True, "inspection": fake, "output_inspections": {str(output): fake}},
             "verification_record": {"inspection_result": fake, "correction_session": dict(fake)}}
    view = present_job(saved)
    assert (view["status"], view["phase"]) == ("error", "recovery_required")
    assert view["candidate_qa"]["inspection"]["ui_status_label"] == "Not inspected"
    assert view["verification_record"]["inspection_result"]["ui_status_label"] == "Not inspected"
    assert view["verification_record"]["correction_session"]["ui_status_label"] == "Not inspected"
    assert saved["verification_record"]["inspection_result"]["ui_status"] == "requirements_verified"


def test_partial_residency_and_partial_device_inventory_cannot_verify():
    assert _complete_residency([]) and _complete_residency([{"name": "model", "size_vram": 0}])
    for entry in ({}, {"name": 1, "size_vram": 0}, {"name": "m", "size_vram": False},
                  {"name": "m", "size_vram": -1}, {"name": "m"}):
        assert not _complete_residency([{"name": "other", "size_vram": 0}, entry])
    assert gpu_memory.parse_nvidia_smi_memory("0, GPU, 100, 300\n1, GPU, N/A, 300") == []


def test_failed_unpin_is_observable_and_reconciled_without_losing_live_pin(tmp_path):
    cache = tmp_path / "cache"
    mgr = LocalReferenceManager(staging_dir=cache)
    image = tmp_path / "original.png"
    Image.new("RGB", (512, 512)).save(image)
    mgr.register_selected([image])
    staged = mgr.stage_for_job(image)["staged_path"]
    for job in ("finished", "live"):
        mgr.pin_staged(staged, job)
    with patch.object(mgr, "_save_pins_locked", side_effect=OSError("write failed")), pytest.raises(OSError):
        mgr.unpin_job("finished")
    reopened = LocalReferenceManager(staging_dir=cache)
    reopened.reconcile_job_pins({"live"})
    assert LocalReferenceManager(staging_dir=cache)._pinned == {staged: {"live"}}
    assert image.is_file()


def test_cli_uses_durable_admission_and_no_clobber_export(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from avatar_v2 import cli, desktop

    item = _item(tmp_path)
    job = tmp_path / "job.json"
    job.write_text(item.job.model_dump_json())
    config = tmp_path / "provider.yaml"
    config.write_text("provider: comfyui\n")
    graph = tmp_path / "workflow.json"
    graph.write_text("{}")
    export = tmp_path / "my-file.json"
    export.write_text("preserved")
    cli.render(job, config, graph, dry_run=True, resolved_out=export)
    assert export.read_text() == "preserved"
    assert len(list(tmp_path.glob("my-file_*.json"))) == 1
    queue = RenderQueue(tmp_path / "queue")
    monkeypatch.setattr(desktop, "DesktopAPI", lambda: SimpleNamespace(queue=queue))
    with patch.object(queue, "_commit_admission", side_effect=OSError("fixture disk error")), \
         patch.object(ComfyUIProvider, "render") as dispatch, pytest.raises(AdmissionRejected):
        cli.render(job, config, graph, dry_run=False, resolved_out=export)
    dispatch.assert_not_called()
    assert not list(queue.history_dir.glob("*.json"))


def test_correction_without_vlm_cannot_skip_residency_check(tmp_path, monkeypatch):
    from avatar_v2 import desktop, ollama_client
    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)
    queue = RenderQueue(tmp_path / "queue")
    api.queue = queue
    item = _item(tmp_path, runtime={"visual_inspection_vlm": False})
    monkeypatch.setattr(api, "_settings", lambda: {})
    monkeypatch.setattr(ollama_client, "fetch_ollama_ps_state", lambda *a, **k: ("unknown", [], "incomplete"))
    try:
        with patch.object(api, "_build_comfy_provider") as provider, pytest.raises(RuntimeError, match="unknown"):
            api._execute_local_correction_step(item, None)
        provider.assert_not_called()
        assert not item.runtime.get("corrective_submissions")
    finally:
        queue.shutdown()


def test_output_changed_during_inspection_cannot_acquire_a_bound_pass(tmp_path, monkeypatch):
    from avatar_v2 import visual_inspector
    from avatar_v2.workflow_router import plan_intelligent_generation
    path = tmp_path / "still.png"
    Image.new("RGB", (256, 256), "red").save(path)
    record = plan_intelligent_generation("Two cups", requested_action="photo", use_llm=False).verification_record
    original = visual_inspector.inspect_media_deterministic
    def replace_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        Image.new("RGB", (256, 256), "blue").save(path)
        return result
    monkeypatch.setattr(visual_inspector, "inspect_media_deterministic", replace_after_read)
    result = visual_inspector.VisualInspector().inspect_image(path, record, use_vlm=False)
    assert result.ui_status_label == "Not inspected"
