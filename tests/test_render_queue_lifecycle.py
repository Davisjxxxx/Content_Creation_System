import json
import threading
from pathlib import Path
from types import SimpleNamespace

from avatar_v2.composer import build_job
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2 import render_queue
from avatar_v2.render_queue import QueuedRender, RenderQueue


def _item(tmp_path: Path, **updates) -> QueuedRender:
    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "ava",
            "display_name": "Ava",
            "subjects": [{"id": "ava", "kind": "synthetic"}],
            "identity_refs": ["front.png"],
        }
    )
    shot = ShotSpec.model_validate(
        {
            "shot_id": "shot",
            "user_prompt": "walk",
            "engine_preference": "h3_fl2va",
            "references": {"init_image": "front.png"},
        }
    )
    data = {
        "job": build_job(avatar, shot),
        "workflow": "builtin:h3_fl2va",
        "provider_config": ProviderConfig(base_url="http://127.0.0.1:8188"),
        "profile": "low_memory_quality",
        "runtime": {"model": "h3", "steps": 4, "cfg": 1.0},
        "output_dir": str(tmp_path / "outputs"),
        "label": "test render",
    }
    data.update(updates)
    return QueuedRender(**data)


def _queue_shell(tmp_path: Path, items=None):
    queue = RenderQueue.__new__(RenderQueue)
    queue.work_dir = tmp_path / "queue"
    queue.work_dir.mkdir(parents=True, exist_ok=True)
    queue.history_dir = tmp_path / "history"
    queue.history_dir.mkdir(parents=True, exist_ok=True)
    queue._items = list(items or [])
    queue._lock = threading.RLock()
    queue._wake = threading.Event()
    queue._stop = threading.Event()
    queue._active_provider = None
    queue._active_id = None
    queue.candidate_qa_runner = None
    queue._engine_restart = None
    queue.memory_manager = SimpleNamespace(maybe_cleanup=lambda *a, **k: {"reason": "pre-retry"})
    return queue


def test_chained_scene_stops_at_the_first_broken_shot(tmp_path):
    first = _item(tmp_path, status="error", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    third = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": second.id})
    queue = _queue_shell(tmp_path, [first, second, third])
    assert queue._cancel_downstream(first.id) == 2
    assert second.status == "cancelled" and third.status == "cancelled"
    assert "previous shot failed" in second.error and "previous shot failed" in third.error


def test_engine_crash_recovers_once_then_stops_the_scene(tmp_path):
    import httpx

    first = _item(tmp_path, status="running", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    queue = _queue_shell(tmp_path, [first, second])
    queue.engine_ready_wait_s = 0.5  # the real app waits for ComfyUI to boot; the test doesn't
    restarts = []
    queue._engine_restart = lambda: restarts.append("restarted")
    queue._handle_failure(first, httpx.ConnectError("[Errno 111] Connection refused"))
    assert first.status == "queued" and first.runtime["recover_attempts"] == 1
    assert first.runtime["retry_after"] > __import__("time").time()
    assert restarts == ["restarted"]
    assert any(e.get("reason") == "engine-crash-recovery" for e in first.memory_events)
    assert any(e.get("reason") == "engine-ready-wait" for e in first.memory_events)
    assert any(e.get("reason") == "pre-retry" for e in first.memory_events)  # the CUDA clean runs before the retry
    assert queue._next() is None  # the failed shot waits for its restart, and its follower waits for it
    queue._handle_failure(first, httpx.ConnectError("[Errno 111] Connection refused"))
    assert first.status == "error" and second.status == "cancelled"  # one retry only; the scene stops cleanly


def test_cancel_downstream_is_durable_so_restarts_cannot_resurrect_shots(tmp_path: Path):
    import time as _time

    first = _item(tmp_path, status="error", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    third = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": second.id})
    queue = _queue_shell(tmp_path, [first, second, third])
    # The queued shots' durable item files exist before the cancellation, as the app wrote them.
    queue._persist_item_durable(second)
    queue._persist_item_durable(third)
    assert (queue.work_dir / "items" / f"{second.id}.json").is_file()
    assert queue._cancel_downstream(first.id) == 2
    # Durable: the stale "queued" item files are retired and history records hold the cancel,
    # so the next app start cannot bring the cancelled shots back to life.
    assert not (queue.work_dir / "items" / f"{second.id}.json").exists()
    assert not (queue.work_dir / "items" / f"{third.id}.json").exists()
    for item in (second, third):
        record = json.loads((queue.history_dir / f"{item.id}.json").read_text(encoding="utf-8"))
        assert record["status"] == "cancelled" and "previous shot failed" in record["error"]
        assert item.finished_at and item.finished_at > _time.time() - 60


def test_chaining_failure_is_clear_and_stops_the_scene(tmp_path: Path):
    from avatar_v2.errors import classify_exception

    first = _item(tmp_path, status="running", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    queue = _queue_shell(tmp_path, [first, second])
    broken = _item(tmp_path, status="running", chain_from_previous=True,
                   runtime={"studio_action": "scene", "chain_from_id": "gone"})
    with __import__("pytest").raises(RuntimeError) as raised:
        queue._execute(broken)  # no provider needed: the chain check fails before any render work
    classified = classify_exception(raised.value)
    assert classified.category == "scene_chain_broken"  # never mistaken for an engine crash
    assert "previous shot" in classified.message
    queue._handle_failure(second, raised.value)  # the follower chain stops visibly, not with a retry
    assert second.status == "error" and second.error_category == "scene_chain_broken"


def test_queue_add_many_list_clear_and_public_metadata(tmp_path: Path):
    first = _item(tmp_path)
    second = _item(tmp_path, status="done", outputs=["out.mp4"], started_at=10, finished_at=13)
    queue = _queue_shell(tmp_path)
    assert queue.add(first) == first.id
    assert queue.add_many([second]) == [second.id]
    public = queue.list()
    assert public[0]["avatar"] == "Ava"
    assert public[1]["elapsed_s"] == 3
    assert public[1]["resolution"] == [720, 1280]
    assert queue.clear_finished() == 1
    assert [entry["id"] for entry in queue.list()] == [first.id]


def test_queue_cancel_queued_running_and_finished(tmp_path: Path):
    queued = _item(tmp_path)
    running = _item(tmp_path, status="running")
    done = _item(tmp_path, status="done")
    queue = _queue_shell(tmp_path, [queued, running, done])
    assert queue.cancel(queued.id) is True
    assert queued.status == "cancelled"
    assert (queue.history_dir / f"{queued.id}.json").is_file()

    class Provider:
        calls = 0

        def interrupt(self, prompt_id=None, allow_global=False):
            assert not allow_global
            self.calls += 1

    provider = Provider()
    queue._active_provider = provider
    queue._active_id = running.id
    assert queue.cancel(running.id) is True
    assert provider.calls == 1
    assert queue.cancel(done.id) is False
    assert queue.cancel("missing") is False


def test_queue_item_round_trip_and_restore_marks_running_interrupted(tmp_path: Path):
    item = _item(tmp_path, status="running")
    queue = _queue_shell(tmp_path)
    data = queue._item_to_dict(item)
    restored = queue._item_from_dict(data)
    assert restored.status == "error"
    assert restored.error_category == "render_failed"
    assert "application restart" in restored.error

    items_dir = queue.work_dir / "items"
    items_dir.mkdir()
    (items_dir / "valid.json").write_text(json.dumps(queue._item_to_dict(_item(tmp_path))))
    (items_dir / "bad.json").write_text("not json")
    queue._items = []
    queue._restore()
    assert len(queue._items) == 1


def test_restore_requeues_interrupted_local_render_for_one_fresh_attempt(tmp_path: Path):
    # A scene shot that was rendering when the app quit must not strand its chain:
    # it goes back to "queued" with a short delay so the engine has time to start.
    import time

    item = _item(tmp_path, status="running", runtime={"studio_action": "scene", "chain_from_id": "other"})
    queue = _queue_shell(tmp_path)
    items_dir = queue.work_dir / "items"
    items_dir.mkdir()
    (items_dir / f"{item.id}.json").write_text(json.dumps(queue._item_to_dict(item)))
    queue._restore()
    assert len(queue._items) == 1
    restored = queue._items[0]
    assert restored.status == "queued" and restored.error == "" and restored.started_at is None
    assert restored.runtime["retry_after"] > time.time()
    assert any(e.get("reason") == "interrupted-by-restart-retry" for e in restored.memory_events)
    assert (items_dir / f"{item.id}.json").is_file()  # the durable record stays until it finishes


def test_restore_keeps_cloud_interruption_as_an_error(tmp_path: Path):
    from avatar_v2.models import ProviderConfig

    item = _item(tmp_path, status="running", provider_config=ProviderConfig(provider="runway"))
    queue = _queue_shell(tmp_path)
    items_dir = queue.work_dir / "items"
    items_dir.mkdir()
    (items_dir / f"{item.id}.json").write_text(json.dumps(queue._item_to_dict(item)))
    queue._restore()
    assert len(queue._items) == 1 and queue._items[0].status == "error"
    assert (items_dir / f"{item.id}.json").exists() is False  # retired once recorded


def test_next_stops_the_chain_when_the_previous_shot_failed(tmp_path: Path):
    first = _item(tmp_path, status="error", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    third = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": second.id})
    queue = _queue_shell(tmp_path, [first, second, third])
    assert queue._next() is None  # nothing can run; the chain is stopped, not stuck
    assert second.status == "cancelled" and "previous shot failed" in second.error
    assert third.status == "cancelled" and "previous shot failed" in third.error


def test_next_waits_for_a_running_predecessor(tmp_path: Path):
    first = _item(tmp_path, status="running", runtime={"studio_action": "scene"})
    second = _item(tmp_path, status="queued", runtime={"studio_action": "scene", "chain_from_id": first.id})
    queue = _queue_shell(tmp_path, [first, second])
    assert queue._next() is None
    assert second.status == "queued"  # still waiting, not cancelled


def test_previous_output_resolution_archive_and_candidate_failures(tmp_path: Path):
    video = tmp_path / "prior.mp4"
    video.write_bytes(b"video")
    prior = _item(tmp_path, status="done", outputs=[str(tmp_path / "image.png"), str(video)])
    current = _item(tmp_path)
    queue = _queue_shell(tmp_path, [prior, current])
    assert queue._previous_output(current) == str(video)
    assert queue._previous_output(_item(tmp_path)) is None
    assert queue._resolve_outputs(current, ["relative.mp4", str(video)]) == [
        str(tmp_path / "outputs" / "relative.mp4"),
        str(video),
    ]

    current.archive_dir = str(tmp_path / "archive")
    current.outputs = [str(video), str(tmp_path / "missing.mp4")]
    archived = queue._archive_outputs(current)
    assert Path(archived[0]).read_bytes() == b"video"
    assert archived[1].endswith("missing.mp4")

    current.outputs = []
    try:
        queue._extract_candidate_frame(current)
    except RuntimeError as exc:
        assert "without a readable video" in str(exc)
    else:
        raise AssertionError("missing video should fail")


def test_execute_forces_cleanup_renders_archives_and_extracts_candidate(tmp_path: Path, monkeypatch):
    output = tmp_path / "provider" / "clip.mp4"
    output.parent.mkdir()
    output.write_bytes((Path(__file__).parent / "fixtures/rc5/clip.mp4").read_bytes())
    item = _item(
        tmp_path,
        output_dir=str(output.parent),
        archive_dir=str(tmp_path / "archive"),
        runtime={"extract_candidate_frame": True},
    )
    queue = _queue_shell(tmp_path, [item])

    class Memory:
        def __init__(self):
            self.events = []

        def maybe_cleanup(self, provider, **kwargs):
            self.events.append(kwargs)
            return {"reason": kwargs["reason"], "forced": kwargs["force"]}

        def start_trace(self, _label):
            pass

        def stop_trace(self):
            return {}

        def start_trace(self, label):
            self.label = label

        def stop_trace(self):
            return {"peak": [{"used_mb": 7000}]}

    class Provider:
        def __init__(self, config):
            self.config = config

        def render(self, workflow, job):
            return {"prompt_id": "p1", "outputs": ["clip.mp4"]}

    queue.memory_manager = Memory()
    monkeypatch.setattr(render_queue, "ComfyUIProvider", Provider)
    (tmp_path / "candidate.png").write_bytes((Path(__file__).parent / "fixtures/rc5/still.png").read_bytes())
    queue._extract_candidate_frame = lambda rendered: rendered.outputs.append(str(tmp_path / "candidate.png"))
    queue._execute(item)
    assert item.prompt_id == "p1"
    assert Path(item.outputs[0]).read_bytes() == output.read_bytes()
    assert item.outputs[-1].endswith("candidate.png")
    assert item.gpu_trace["peak"][0]["used_mb"] == 7000
    assert queue.memory_manager.events[0]["force"] is True
    assert queue.memory_manager.events[0]["unload_models"] is True  # video jobs unload after render


def test_image_jobs_keep_models_loaded_for_fast_replay(tmp_path: Path, monkeypatch):
    """FLUX.2 photo jobs keep their models loaded between runs (pre and post cleanup never
    unload); video jobs keep the unload behavior so the big Wan weights release the GPU."""
    image_item = _item(tmp_path, workflow="builtin:flux2_txt2img",
                       runtime={"studio_action": "forge_photo"})
    video_item = _item(tmp_path)  # default builtin:h3_fl2va workflow

    class Memory:
        def __init__(self):
            self.events = []

        def maybe_cleanup(self, provider, **kwargs):
            self.events.append(kwargs)
            return {"reason": kwargs["reason"], "forced": kwargs["force"]}

        def start_trace(self, _label):
            pass

        def stop_trace(self):
            return {}

    stub_png = tmp_path / "stub_out.png"
    stub_png.write_bytes((Path(__file__).parent / "fixtures/rc5/still.png").read_bytes())
    stub_mp4 = tmp_path / "stub_clip.mp4"
    stub_mp4.write_bytes((Path(__file__).parent / "fixtures/rc5/clip.mp4").read_bytes())

    class Provider:
        def __init__(self, config):
            self.config = config

        def render(self, workflow, job):
            out = stub_png if workflow == "builtin:flux2_txt2img" else stub_mp4
            return {"prompt_id": "p1", "outputs": [str(out)]}

    queue = _queue_shell(tmp_path, [image_item])
    queue.memory_manager = Memory()
    monkeypatch.setattr(render_queue, "ComfyUIProvider", Provider)
    queue._execute(image_item)
    pre = queue.memory_manager.events[0]
    assert pre["reason"].startswith("pre-render:") and pre["unload_models"] is False
    assert pre["force"] is True  # the low-memory preflight still runs (cache freed, models kept)
    post = queue._post_cleanup(image_item, Provider(image_item.provider_config))
    assert post["reason"] == "post-render:queued · keep image models"
    queue.memory_manager.events = []

    queue2 = _queue_shell(tmp_path, [video_item])
    queue2.memory_manager = Memory()
    monkeypatch.setattr(render_queue, "ComfyUIProvider", Provider)
    queue2._execute(video_item)
    assert queue2.memory_manager.events[0]["unload_models"] is True
    post2 = queue2._post_cleanup(video_item, Provider(video_item.provider_config))
    assert "keep image models" not in post2["reason"]


def test_execute_chaining_rebinds_primary_h3_picture(tmp_path: Path, monkeypatch):
    prior_video = tmp_path / "prior.mp4"
    prior_video.write_bytes(b"video")
    prior = _item(tmp_path, status="done", outputs=[str(prior_video)])
    current = _item(tmp_path, chain_from_previous=True)
    current.job.selected_engine = "h3_ref2va"
    current.job.asset_map["H3_PICTURE_1"] = "old.png"
    queue = _queue_shell(tmp_path, [prior, current])
    frame = tmp_path / "chain.png"
    frame.write_bytes(b"frame")
    queue._extract_last_frame = lambda *_args: frame

    class Memory:
        def maybe_cleanup(self, *_args, **kwargs):
            return kwargs

        def start_trace(self, _label):
            pass

        def stop_trace(self):
            return {}

    chained_mp4 = tmp_path / "chained.mp4"
    chained_mp4.write_bytes((Path(__file__).parent / "fixtures/rc5/clip.mp4").read_bytes())

    class Provider:
        def __init__(self, _config):
            pass

        def render(self, _workflow, job):
            assert job.asset_map["INIT_IMAGE"] == str(frame)
            assert job.asset_map["H3_PICTURE_1"] == str(frame)
            return {"prompt_id": "p2", "outputs": [str(chained_mp4)]}

    queue.memory_manager = Memory()
    monkeypatch.setattr(render_queue, "ComfyUIProvider", Provider)
    queue._execute(current)
    assert current.prompt_id == "p2"


def test_shutdown_stops_trace_and_wakes_worker(tmp_path: Path):
    queue = _queue_shell(tmp_path)

    class Memory:
        called = False

        def stop_trace(self):
            self.called = True

    queue.memory_manager = Memory()
    queue.shutdown()
    assert queue._stop.is_set()
    assert queue._wake.is_set()
    assert queue.memory_manager.called is True


def test_candidate_qa_is_advisory_and_persisted_in_public_record(tmp_path: Path):
    item = _item(tmp_path, status="done", runtime={"candidate_qa": True})
    queue = _queue_shell(tmp_path, [item])
    queue.candidate_qa_runner = lambda queued: {
        "ok": True,
        "decision": "review",
        "overall_score": 0.68,
        "candidate_path": "/candidate.png",
    }
    queue._run_candidate_qa(item)
    assert item.public()["candidate_qa"]["decision"] == "review"
    restored = queue._item_from_dict(queue._item_to_dict(item))
    assert restored.candidate_qa["overall_score"] == 0.68

    queue.candidate_qa_runner = lambda _queued: (_ for _ in ()).throw(RuntimeError("vision down"))
    queue._run_candidate_qa(item)
    assert item.candidate_qa["decision"] == "unavailable"
    assert "vision down" in item.candidate_qa["error"]
