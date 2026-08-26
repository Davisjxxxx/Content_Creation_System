from pathlib import Path

import pytest

from avatar_v2.desktop import DesktopAPI
from avatar_v2.reference_cache import (
    _visual_facts_to_role,
    infer_reference_role,
    ollama_vision_available,
    scan_reference_cache,
)


def test_filename_angle_inference_is_conservative():
    assert infer_reference_role(Path("ava_face_front.jpg")) == ("identity_refs", "front")
    assert infer_reference_role(Path("ava_face_34_left.png")) == ("identity_refs", "three_quarter_left")
    assert infer_reference_role(Path("ava_full_body_rear.webp")) == ("body_refs", "rear")
    assert infer_reference_role(Path("ava_body_top_front.png")) == ("body_refs", "top_front")
    assert infer_reference_role(Path("ava_pelvis_rear.png")) == ("anatomy_refs", "pelvis_rear")
    assert infer_reference_role(Path("ava_hair_tied.png")) == ("hair_refs", "tied")
    assert infer_reference_role(Path("random_001.png")) is None


def test_cache_scan_maps_images_and_separates_motion_media(tmp_path: Path):
    names = [
        "ava_face_front.jpg",
        "ava_face_profile_left.png",
        "ava_full_body_front.webp",
        "ava_full_body_rear.jpg",
        "ava_hands.png",
        "mystery.png",
        "turntable.gif",
        "walk.mp4",
        "voice.wav",
        "notes.txt",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"data")
    result = scan_reference_cache(tmp_path)
    assert result["assignments"]["identity_refs"]["front"].endswith("ava_face_front.jpg")
    assert result["assignments"]["body_refs"]["rear"].endswith("ava_full_body_rear.jpg")
    assert result["assignments"]["anatomy_refs"]["hands"].endswith("ava_hands.png")
    assert result["primary_source"].endswith("ava_face_front.jpg")
    assert len(result["unclassified_images"]) == 1
    assert len(result["media"]["gifs"]) == 1
    assert len(result["media"]["videos"]) == 1
    assert "motion examples" in " ".join(result["warnings"])


def test_desktop_cache_scan_reports_invalid_folder():
    api = DesktopAPI.__new__(DesktopAPI)
    result = api.avatar_designer_scan_cache({"cache_dir": "/definitely/not/a/cache"})
    assert result["ok"] is False
    assert "folder does not exist" in result["error"]


def test_visual_scan_classifies_opaque_filenames_and_ranks_duplicate_roles(tmp_path: Path):
    paths = [tmp_path / "Screenshot_001.jpg", tmp_path / "Screenshot_002.jpg", tmp_path / "Screenshot_003.jpg"]
    for path in paths:
        path.write_bytes(b"image pixels are supplied to the injected classifier")

    results = {
        "Screenshot_001.jpg": {"role": "identity_front", "confidence": 0.72, "visible_evidence": "face toward camera"},
        "Screenshot_002.jpg": {"role": "body_rear", "confidence": 0.94, "visible_evidence": "body viewed from rear"},
        "Screenshot_003.jpg": {"role": "identity_front", "confidence": 0.96, "visible_evidence": "stronger frontal face"},
    }
    result = scan_reference_cache(
        tmp_path,
        use_visual=True,
        visual_classifier=lambda path: results[path.name],
    )

    assert result["angle_method"] == "local_ollama_vision"
    assert result["visual_images_analyzed"] == 3
    assert result["assignments"]["identity_refs"]["front"].endswith("Screenshot_003.jpg")
    assert result["assignments"]["body_refs"]["rear"].endswith("Screenshot_002.jpg")
    assert result["primary_source"].endswith("Screenshot_003.jpg")
    assert result["duplicate_role_candidates"][0].endswith("Screenshot_001.jpg")
    assert result["unclassified_images"] == []


def test_visual_service_is_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        ollama_vision_available("https://vision.example.com", "qwen3-vl:2b-instruct")


def test_visual_facts_map_to_profile_roles_deterministically():
    assert _visual_facts_to_role(
        {"framing": "face_closeup", "facing": "three_quarter_left", "camera_pitch": "overhead", "region": "face"}
    ) == "identity_three_quarter_left"
    assert _visual_facts_to_role(
        {"framing": "full_body", "facing": "rear", "camera_pitch": "overhead", "region": "full_body"}
    ) == "body_top_rear"
    assert _visual_facts_to_role(
        {"framing": "body_detail", "facing": "rear", "camera_pitch": "eye_level", "region": "buttocks"}
    ) == "anatomy_pelvis_rear"
    assert _visual_facts_to_role(
        {"framing": "full_body", "facing": "front", "camera_pitch": "eye_level", "region": "full_body", "pose": "walking"}
    ) == "body_walking"


def test_quick_builder_exposes_ai_analysis_and_saved_anatomy_selection():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "Local AI understands the views" in html
    assert "AI analyze cache" in html
    assert "Use saved anatomy references" in html
    assert "Advanced · Add a new anatomy reference" in html
    assert "Add to this avatar" in html
    assert "Queue ${missing.length} missing H3 profile views now?" in html
