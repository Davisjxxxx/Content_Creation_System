import json
import threading
from pathlib import Path

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
    return queue


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

        def interrupt(self):
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
    output.write_bytes(b"video")
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
    queue._extract_candidate_frame = lambda rendered: rendered.outputs.append(str(tmp_path / "candidate.png"))
    queue._execute(item)
    assert item.prompt_id == "p1"
    assert Path(item.outputs[0]).read_bytes() == b"video"
    assert item.outputs[-1].endswith("candidate.png")
    assert item.gpu_trace["peak"][0]["used_mb"] == 7000
    assert queue.memory_manager.events[0]["force"] is True


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

    class Provider:
        def __init__(self, _config):
            pass

        def render(self, _workflow, job):
            assert job.asset_map["INIT_IMAGE"] == str(frame)
            assert job.asset_map["H3_PICTURE_1"] == str(frame)
            return {"prompt_id": "p2", "outputs": []}

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
