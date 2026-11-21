from pathlib import Path
from types import SimpleNamespace

import pytest

from avatar_v2.desktop import DesktopAPI
from avatar_v2.reference_cache import (
    DEFAULT_VISION_MODEL,
    IDENTITY_PROMPT,
    IDENTITY_SCHEMA,
    VISION_PROMPT,
    VISION_SCHEMA,
    _quality_result,
    _visual_facts_to_role,
    _visual_facts_to_roles,
    infer_reference_role,
    ollama_vision_available,
    scan_reference_cache,
)


def test_default_vision_model_is_the_measured_8gb_recommendation():
    assert DEFAULT_VISION_MODEL == "qwen3-vl:2b-instruct"


def test_filename_angle_inference_is_conservative():
    assert infer_reference_role(Path("ava_face_front.jpg")) == ("identity_refs", "front")
    assert infer_reference_role(Path("ava_face_34_left.png")) == ("identity_refs", "three_quarter_left")
    assert infer_reference_role(Path("ava_full_body_rear.webp")) == ("body_refs", "rear")
    assert infer_reference_role(Path("ava_body_top_front.png")) == ("body_refs", "top_front")
    assert infer_reference_role(Path("ava_pelvis_rear.png")) == ("anatomy_refs", "pelvis_rear")
    assert infer_reference_role(Path("ava_hair_tied.png")) == ("hair_refs", "tied")
    assert infer_reference_role(Path("random_001.png")) is None
    assert infer_reference_role(Path("Screenshot_001.png")) is None


def test_visual_classifier_schema_is_geometry_and_quality_only():
    assert set(VISION_SCHEMA["properties"]) == {
        "framing",
        "facing",
        "camera_pitch",
        "region",
        "usable_regions",
        "pose",
        "face_visibility",
        "phone_ui",
        "watermark",
        "heavy_occlusion",
        "source_quality",
        "confidence",
    }
    prompt = VISION_PROMPT.lower()
    assert "do not identify the person" in prompt
    assert "do not infer or report sex" in prompt
    assert "classify only view geometry and source suitability" in prompt
    assert "source quality" in prompt
    assert "do not lower source quality solely for phone chrome or a watermark" in prompt
    identity_prompt = IDENTITY_PROMPT.lower()
    assert "visible_evidence" not in VISION_SCHEMA["properties"]
    assert set(IDENTITY_SCHEMA["properties"]) == {"same_subject", "confidence"}
    assert "private consistency check, not identification" in identity_prompt
    assert "never name, recognize, search for" in identity_prompt


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


def test_cache_scan_can_analyze_only_explicitly_selected_images(tmp_path: Path):
    selected = tmp_path / "selected_face_front.JPG"
    sibling = tmp_path / "unselected_full_body_rear.jpg"
    selected.write_bytes(b"selected")
    sibling.write_bytes(b"not selected")

    result = scan_reference_cache("", selected_files=[selected])

    assert result["selection_mode"] == "files"
    assert result["selected_file_count"] == 1
    assert result["media"]["images"] == [str(selected.resolve())]
    assert str(sibling.resolve()) not in result["media"]["images"]
    assert result["assignments"]["identity_refs"]["front"] == str(selected.resolve())


def test_image_picker_accepts_backend_formats_uppercase_and_has_fallback(monkeypatch):
    calls = []

    class Window:
        def create_file_dialog(self, *args, **kwargs):
            calls.append((args, kwargs))
            return []

    monkeypatch.setitem(__import__("sys").modules, "webview", SimpleNamespace(OPEN_DIALOG="open"))
    api = DesktopAPI.__new__(DesktopAPI)
    api._self_window = Window()

    assert api.choose_files("image", True) == []
    file_types = calls[0][1]["file_types"]
    assert "*.JPG" in file_types[0]
    assert "*.tif" in file_types[0]
    assert "*.TIFF" in file_types[0]
    assert file_types[-1] == "All files (*.*)"


def test_desktop_cache_scan_reports_invalid_folder():
    api = DesktopAPI.__new__(DesktopAPI)
    result = api.avatar_designer_scan_cache({"cache_dir": "/definitely/not/a/cache"})
    assert result["ok"] is False
    assert "folder does not exist" in result["error"]


def test_desktop_cache_scan_forwards_exact_image_selection(monkeypatch, tmp_path: Path):
    selected = tmp_path / "selected.JPG"
    selected.write_bytes(b"image")
    captured = {}

    def fake_scan(root, **kwargs):
        captured["root"] = root
        captured.update(kwargs)
        return {"selection_mode": "files", "media": {"images": [str(selected)]}}

    monkeypatch.setattr("avatar_v2.desktop.scan_reference_cache", fake_scan)
    api = DesktopAPI.__new__(DesktopAPI)

    result = api.avatar_designer_scan_cache(
        {
            "image_paths": [str(selected)],
            "vision_model": "qwen2.5vl:7b",
            "single_subject_confirmed": True,
        }
    )

    assert result["ok"] is True
    assert captured["root"] == ""
    assert captured["selected_files"] == [str(selected)]
    assert captured["use_visual"] is True
    assert captured["vision_model"] == "qwen2.5vl:7b"
    assert captured["single_subject_confirmed"] is True


def test_cache_scan_reports_byte_identical_selected_images(tmp_path: Path):
    first = tmp_path / "face_front.jpg"
    duplicate = tmp_path / "copy" / "face_front_copy.jpg"
    distinct = tmp_path / "body_rear.jpg"
    duplicate.parent.mkdir()
    first.write_bytes(b"same pixels")
    duplicate.write_bytes(b"same pixels")
    distinct.write_bytes(b"different pixels")

    result = scan_reference_cache("", selected_files=[first, duplicate, distinct])

    assert result["selected_image_count"] == 3
    assert result["distinct_image_count"] == 2
    assert len(result["duplicate_content_images"]) == 1
    assert Path(result["duplicate_content_images"][0]).read_bytes() == b"same pixels"


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


def test_visual_scan_holds_low_quality_phone_screenshot(tmp_path: Path):
    path = tmp_path / "Screenshot_001.jpg"
    path.write_bytes(b"injected")
    result = scan_reference_cache(
        tmp_path,
        use_visual=True,
        visual_classifier=lambda _: {
            "role": "identity_front",
            "confidence": 0.98,
            "source_quality": 0.78,
            "face_visibility": "partial",
            "phone_ui": True,
            "watermark": True,
            "heavy_occlusion": False,
        },
    )

    item = result["classification_results"][0]
    assert item["quality_pass"] is False
    assert "identity views require a clearly visible complete face" in item["quality_reasons"]
    assert result["assignments"]["identity_refs"] == {}
    assert result["quality_rejected_images"] == [str(path)]


def test_phone_ui_and_watermark_warn_without_double_penalizing_clear_geometry():
    quality, reasons, passed = _quality_result(
        {
            "source_quality": 0.72,
            "face_visibility": "clear",
            "phone_ui": True,
            "watermark": True,
            "heavy_occlusion": False,
        },
        "identity_front",
    )

    assert quality == 0.72
    assert reasons == []
    assert passed is True


def test_identity_gate_keeps_same_subject_and_quarantines_mismatch(tmp_path: Path):
    anchor = tmp_path / "anchor.jpg"
    same = tmp_path / "same.jpg"
    different = tmp_path / "different.jpg"
    for path in (anchor, same, different):
        path.write_bytes(b"injected")
    classifications = {
        "anchor.jpg": {"role": "identity_front", "confidence": 0.99, "source_quality": 0.95, "face_visibility": "clear"},
        "same.jpg": {"role": "identity_three_quarter_left", "confidence": 0.91, "source_quality": 0.90, "face_visibility": "clear"},
        "different.jpg": {"role": "body_full_front", "confidence": 0.93, "source_quality": 0.92, "face_visibility": "partial"},
    }

    def compare(_anchor: Path, candidate: Path):
        if candidate.name == "same.jpg":
            return {"same_subject": "same", "confidence": 0.88, "visible_evidence": "stable face geometry"}
        return {"same_subject": "different", "confidence": 0.94, "visible_evidence": "face geometry conflicts"}

    result = scan_reference_cache(
        tmp_path,
        use_visual=True,
        visual_classifier=lambda path: classifications[path.name],
        identity_comparator=compare,
    )

    assert result["identity_gate_active"] is True
    assert result["identity_anchor"] == str(anchor)
    assert result["assignments"]["identity_refs"]["front"] == str(anchor)
    assert result["assignments"]["identity_refs"]["three_quarter_left"] == str(same)
    assert result["assignments"]["body_refs"] == {}
    assert result["identity_quarantined_images"] == [str(different)]
    assert set(result["generation_eligible_images"]) == {str(anchor), str(same)}


def test_confirmed_single_avatar_cache_admits_no_face_but_not_visible_mismatch(tmp_path: Path):
    anchor = tmp_path / "anchor.jpg"
    body_only = tmp_path / "body_only.jpg"
    different = tmp_path / "different.jpg"
    for path in (anchor, body_only, different):
        path.write_bytes(path.name.encode())
    classifications = {
        "anchor.jpg": {
            "role": "identity_front",
            "confidence": 0.99,
            "source_quality": 0.95,
            "face_visibility": "clear",
        },
        "body_only.jpg": {
            "role": "body_rear",
            "confidence": 0.92,
            "source_quality": 0.91,
            "face_visibility": "absent",
        },
        "different.jpg": {
            "role": "body_full_front",
            "confidence": 0.93,
            "source_quality": 0.92,
            "face_visibility": "clear",
        },
    }

    result = scan_reference_cache(
        tmp_path,
        use_visual=True,
        visual_classifier=lambda path: classifications[path.name],
        identity_comparator=lambda _anchor, _candidate: {
            "same_subject": "different",
            "confidence": 0.94,
        },
        single_subject_confirmed=True,
    )

    items = {Path(item["path"]).name: item for item in result["classification_results"]}
    assert items["body_only.jpg"]["identity_status"] == "confirmed_no_face"
    assert items["body_only.jpg"]["generation_eligible"] is True
    assert items["different.jpg"]["identity_status"] == "different"
    assert items["different.jpg"]["generation_eligible"] is False
    assert result["single_subject_confirmed"] is True
    assert set(result["generation_eligible_images"]) == {str(anchor), str(body_only)}


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
        {
            "framing": "body_detail",
            "facing": "front",
            "camera_pitch": "low_angle",
            "region": "pelvis",
            "pose": "seated",
        }
    ) == "unclassified"
    assert _visual_facts_to_role(
        {"framing": "full_body", "facing": "front", "camera_pitch": "eye_level", "region": "full_body", "pose": "walking"}
    ) == "body_walking"


def test_visual_facts_map_one_image_to_every_usable_visible_region():
    roles = _visual_facts_to_roles(
        {
            "framing": "upper_body",
            "facing": "front",
            "camera_pitch": "eye_level",
            "region": "torso",
            "pose": "seated",
            "face_visibility": "clear",
            "usable_regions": ["face", "torso", "pelvis", "hands"],
        }
    )

    assert roles == [
        "anatomy_torso_front",
        "identity_front",
        "anatomy_hands",
    ]


def test_visual_scan_assigns_one_image_to_multiple_supported_slots(tmp_path: Path):
    path = tmp_path / "opaque.jpg"
    path.write_bytes(b"injected")

    result = scan_reference_cache(
        tmp_path,
        use_visual=True,
        visual_classifier=lambda _: {
            "role": "identity_front",
            "roles": ["identity_front", "anatomy_torso_front", "anatomy_pelvis_front"],
            "confidence": 0.96,
            "source_quality": 0.94,
            "face_visibility": "clear",
        },
    )

    assert result["assignments"]["identity_refs"]["front"] == str(path)
    assert result["assignments"]["anatomy_refs"]["torso_front"] == str(path)
    assert result["assignments"]["anatomy_refs"]["pelvis_front"] == str(path)
    assert result["classification_results"][0]["roles"] == [
        "identity_front",
        "anatomy_torso_front",
        "anatomy_pelvis_front",
    ]


def test_quick_builder_exposes_ai_analysis_and_saved_anatomy_selection():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    desktop = (Path(__file__).parents[1] / "avatar_v2" / "desktop.py").read_text(encoding="utf-8")
    assert "Internal evidence gates" in html
    assert "same-subject consistency" in html
    assert "AI analyze references" in html
    assert 'id="designerVisionModel"' in html
    assert 'value="qwen3-vl:2b-instruct-q8_0"' in html
    assert 'value="qwen3-vl:8b-instruct"' in html
    assert 'id="designerSingleSubject"' in html
    assert "generation-eligible image${eligible.length===1?'':'s'} analyzed" in html
    assert "Qwen3-VL 8B Q4 · recommended, strict identity and view check" in html
    assert '<option value="qwen3-vl:8b-instruct" selected>' in html
    assert '"vision_classifier_model": "qwen3-vl:8b-instruct"' in desktop
    assert "Use saved anatomy references" in html
    assert "Advanced · Add a new anatomy reference" in html
    assert "Add to this avatar" in html
    assert "Queue ${missing.length} missing H3 profile views now?" in html
