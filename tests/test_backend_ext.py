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
        "anatomy_refs": {"pelvis_front": "/pelvis.png"},
        "hair_refs": {"loose": "/hair.png"},
        "wardrobe_refs": {"jacket": "/jacket.png"},
    })
    payload = store.profile_to_avatar_payload("ava-01")
    assert payload["identity_refs"] == ["/b.png", "/c.png"]
    assert payload["body_refs"] == ["/body.png", "/pelvis.png"]
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


def test_wan_frame_count_snaps_to_4k_plus_1():
    from avatar_v2.workflows_builder import wan_frame_count
    for duration, expect_min in [(1, 16), (3, 49), (6, 97)]:
        frames = wan_frame_count(duration)
        assert (frames - 1) % 4 == 0
        assert frames >= expect_min
    assert wan_frame_count(3) == 49
    assert wan_frame_count(6) == 97


def test_wan_funcontrol_builder_wires_identity_and_motion():
    from avatar_v2.workflows_builder import build_wan_funcontrol_workflow
    graph = build_wan_funcontrol_workflow({
        "PROMPT": "walk", "NEGATIVE_PROMPT": "bad",
        "WIDTH": 832, "HEIGHT": 480, "FRAMES": 49, "FPS": 16,
        "SEED": 1, "STEPS": 20, "CFG": 1.0,
        "SAMPLER_NAME": "uni_pc", "SCHEDULER": "simple",
        "WAN_DIFFUSION_MODEL": "m.safetensors",
        "WAN_TEXT_ENCODER": "t.safetensors", "WAN_VAE": "v.safetensors",
        "INIT_IMAGE": "id.png", "MOTION_VIDEO": "motion.mp4",
        "OUTPUT_PREFIX": "p",
    })
    kinds = {nid: n["class_type"] for nid, n in graph.items()}
    assert "Wan22FunControlToVideo" in kinds.values()
    task_id = next(nid for nid, k in kinds.items() if k == "Wan22FunControlToVideo")
    inputs = graph[task_id]["inputs"]
    assert inputs["ref_image"] == [str(int(task_id) - 2), 0]
    assert inputs["control_video"][0] == str(int(task_id) - 1)
    assert inputs["length"] == 49


def test_wan_canvas_snaps_to_multiple_of_32():
    from avatar_v2.workflows_builder import snap_wan_canvas
    assert snap_wan_canvas(720, 720) == (704, 704)
    snapped = snap_wan_canvas(480, 272)
    assert snapped[0] % 32 == 0 and snapped[1] % 32 == 0
    assert snap_wan_canvas(832, 480) == (832, 480)


def test_classify_rope_canvas_error():
    classified = classify_exception(RuntimeError("apply_rope freqs shape is not broadcastable to input"))
    assert classified.category == "invalid_canvas"


def test_upscale_builder_wires_video_and_model():
    from avatar_v2.workflows_builder import build_upscale_workflow
    graph = build_upscale_workflow({
        "SOURCE_VIDEO": "src.mp4", "UPSCALE_MODEL": "4x-UltraSharp.pth",
        "FPS": 16, "OUTPUT_PREFIX": "up",
    })
    kinds = {nid: n["class_type"] for nid, n in graph.items()}
    assert set(kinds.values()) == {"VHS_LoadVideoPath", "UpscaleModelLoader", "ImageUpscaleWithModel", "VHS_VideoCombine"}
    loader = next(n for n in graph.values() if n["class_type"] == "VHS_LoadVideoPath")
    assert loader["inputs"]["video"] == "input/src.mp4"


def test_upscale_builder_requires_source():
    import pytest
    from avatar_v2.workflows_builder import build_upscale_workflow
    with pytest.raises(RuntimeError):
        build_upscale_workflow({"UPSCALE_MODEL": "m.pth"})


def test_model_names_are_not_treated_as_assets():
    from avatar_v2.providers.comfyui import _is_asset_key
    assert _is_asset_key("H3_PICTURE_1")
    assert _is_asset_key("H3_AUDIO_1")
    assert not _is_asset_key("H3_AUDIO_VAE")
    assert not _is_asset_key("H3_VIDEO_VAE")
    assert not _is_asset_key("WAN_DIFFUSION_MODEL")
    assert not _is_asset_key("H3_DIFFUSION_MODEL")
