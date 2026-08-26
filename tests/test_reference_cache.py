from pathlib import Path

from avatar_v2.desktop import DesktopAPI
from avatar_v2.reference_cache import infer_reference_role, scan_reference_cache


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
