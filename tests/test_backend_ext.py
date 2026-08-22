import json
from pathlib import Path

import pytest

from avatar_v2.errors import classify_exception
from avatar_v2.library import AvatarProfile, LibraryStore, MotionAsset, SceneAsset
from avatar_v2.render_queue import QueuedRender
from avatar_v2 import render_queue


def test_classify_cuda_oom():
    classified = classify_exception(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert classified.category == "cuda_oom"
    assert "smaller canvas" in classified.message


def test_classify_timeout():
    classified = classify_exception(TimeoutError("too slow"))
    assert classified.category == "timeout"


def test_classify_missing_model():
    classified = classify_exception(RuntimeError("PytorchStreamReader failed reading zip archive"))
    assert classified.category == "missing_model"


def test_classify_fallback():
    classified = classify_exception(ValueError("something odd"))
    assert classified.category == "render_failed"


def test_library_avatar_crud(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.put_avatar({
        "avatar_id": "ava-01",
        "display_name": "Ava",
        "identity_refs": {"three_quarter_left": "/a.png", "front": "/b.png"},
        "persistent_features": ["freckles"],
    })
    assert len(store.avatars()) == 1
    profile = AvatarProfile.model_validate(store.get_avatar("ava-01"))
    assert profile.ordered_identity == ["/b.png", "/a.png"]
    assert store.delete_avatar("ava-01")
    assert store.avatars() == []


def test_library_rejects_bad_id(tmp_path: Path):
    store = LibraryStore(tmp_path)
    with pytest.raises(ValueError):
        store.put_avatar({"avatar_id": "Bad ID!", "display_name": "x", "identity_refs": {"front": "/a.png"}})


def test_library_motion_and_scene(tmp_path: Path):
    store = LibraryStore(tmp_path)
    motion = store.put_motion({
        "motion_id": "walk_confident_01",
        "label": "Confident walk",
        "category": "walk_confident",
        "path": "/videos/walk.mp4",
    })
    assert MotionAsset.model_validate(motion).category == "walk_confident"
    scene = store.put_scene({
        "scene_id": "waterfront",
        "label": "Waterfront",
        "path": "/scenes/waterfront.png",
        "location": "outdoor",
        "aspect": "vertical",
    })
    assert SceneAsset.model_validate(scene).aspect == "vertical"
    assert store.delete_motion("walk_confident_01")
    assert store.delete_scene("waterfront")


def test_profile_to_avatar_payload(tmp_path: Path):
    store = LibraryStore(tmp_path)
    store.put_avatar({
        "avatar_id": "ava-01",
        "display_name": "Ava",
        "identity_refs": {"front": "/b.png", "profile_left": "/c.png"},
        "body_refs": {"full_front": "/body.png"},
        "hair_refs": {"loose": "/hair.png"},
        "wardrobe_refs": {"jacket": "/jacket.png"},
    })
    payload = store.profile_to_avatar_payload("ava-01")
    assert payload["identity_refs"] == ["/b.png", "/c.png"]
    assert payload["body_refs"] == ["/body.png"]
    assert payload["hair_refs"] == ["/hair.png"]
    assert payload["wardrobe_refs"] == ["/jacket.png"]


def test_queue_history_manifest_written(tmp_path: Path, monkeypatch):
    from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
    from avatar_v2.composer import build_job

    avatar = AvatarManifest.model_validate({
        "avatar_id": "ava",
        "display_name": "Ava",
        "subjects": [{"id": "ava", "kind": "synthetic"}],
        "identity_refs": ["front.png"],
    })
    shot = ShotSpec.model_validate({"shot_id": "s1", "user_prompt": "walk"})
    job = build_job(avatar, shot)
    item = QueuedRender(
        job=job,
        workflow="wf.json",
        provider_config=ProviderConfig(base_url="http://127.0.0.1:8188"),
        profile="auto",
        runtime={"model": None},
        label="test",
    )

    history_dir = tmp_path / "history"
    queue = render_queue.RenderQueue.__new__(render_queue.RenderQueue)
    queue.work_dir = tmp_path / "queue"
    queue.work_dir.mkdir(parents=True)
    queue.history_dir = history_dir
    queue.history_dir.mkdir(parents=True)
    queue._persist_history(item)  # type: ignore[attr-defined]
    saved = json.loads((history_dir / f"{item.id}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "queued"
    assert saved["seed"] == shot.seed
    assert saved["resolution"] == [shot.width, shot.height]
