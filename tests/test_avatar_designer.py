from pathlib import Path

import pytest

from avatar_v2.avatar_designer import (
    build_designer_prompt,
    coverage_report,
    designer_spec,
    slot_by_id,
)
from avatar_v2.desktop import DesktopAPI
from avatar_v2.library import AnatomyGuide, AvatarProfile, LibraryStore
from avatar_v2.models import ProviderConfig
from avatar_v2.render_queue import RenderQueue


def _adult_profile(**updates):
    data = {
        "avatar_id": "ava-01",
        "display_name": "Ava",
        "apparent_age_years": 29,
        "subject_kind": "synthetic",
        "age_verified_18_plus": True,
        "consent_confirmed": True,
    }
    data.update(updates)
    return data


def test_designer_spec_covers_requested_angles_details_and_anatomy():
    spec = designer_spec()
    ids = {slot["id"] for slot in spec["slots"]}
    assert {
        "identity_front",
        "body_full_rear",
        "body_full_left",
        "body_full_right",
        "body_top_front",
        "body_low_rear",
        "anatomy_torso_front",
        "anatomy_pelvis_front",
        "anatomy_pelvis_rear",
        "anatomy_hands",
        "anatomy_feet",
    } <= ids
    assert {"freckles_moles", "scars", "tattoos", "body_type", "anatomy_details"} <= set(spec["appearance_fields"])


def test_coverage_report_reads_all_persistent_reference_groups():
    profile = AvatarProfile.model_validate(
        _adult_profile(
            identity_refs={"front": "/front.png"},
            body_refs={"full_front": "/body.png", "top_front": "/top.png"},
            anatomy_refs={"pelvis_front": "/pelvis.png"},
        )
    )
    report = coverage_report(profile)
    assert report["filled"] == 4
    assert report["required"] == len(designer_spec()["slots"])
    assert "identity_front" not in report["missing"]
    assert "anatomy_pelvis_front" not in report["missing"]


def test_designer_prompt_is_identity_and_landmark_specific():
    prompt = build_designer_prompt(
        "anatomy_pelvis_front",
        {"body_type": "athletic pear shape", "freckles_moles": "mole on left hip"},
        29,
        sex="female",
        coverage_mode="nude",
    )
    assert "H3 photorealistic avatar reference turn" in prompt
    assert "neutral clinical adult anatomy reference" in prompt
    assert "mole on left hip" in prompt
    assert "visibly a 29-year-old adult" in prompt
    assert "do not make the subject look younger or older" in prompt
    assert "Hold that target view nearly motionless for the final 30%" in prompt
    assert "Do not beautify, reshape, add, remove, mirror, duplicate, or relocate" in prompt
    assert "selected sex anatomy is female" in prompt
    assert "anatomy-first nude validation pass" in prompt


def test_designer_prompt_supports_male_clothed_continuity():
    prompt = build_designer_prompt(
        "body_full_front",
        apparent_age_years=104,
        sex="male",
        coverage_mode="clothed",
    )
    assert "selected sex anatomy is male" in prompt
    assert "104-year-old adult" in prompt
    assert "plain, opaque, close-fitting neutral clothing" in prompt


def test_library_round_trips_designer_profile_and_applies_anatomy(tmp_path: Path):
    store = LibraryStore(tmp_path)
    saved = store.put_avatar(
        _adult_profile(
            identity_refs={"front": "/front.png"},
            body_refs={"full_front": "/body.png"},
            anatomy_refs={"pelvis_front": "/pelvis.png"},
            source_refs=["/source.png"],
            appearance_manifest={"tattoos": "small star on right shoulder"},
            sex="female",
            nude_mode="both",
            designer_iteration=3,
        )
    )
    assert saved["designer_iteration"] == 3
    assert saved["apparent_age_years"] == 29
    assert saved["sex"] == "female"
    assert saved["nude_mode"] == "both"
    payload = store.profile_to_avatar_payload("ava-01")
    assert payload["apparent_age_years"] == 29
    assert payload["body_refs"] == ["/body.png", "/pelvis.png"]
    assert "tattoos: small star on right shoulder" in payload["persistent_features"]


def test_appearance_component_requires_enough_authorized_examples_before_selection(tmp_path: Path):
    store = LibraryStore(tmp_path / "library")
    paths = []
    for index in range(3):
        path = tmp_path / f"torso-{index}.png"
        path.write_bytes(b"image")
        paths.append(str(path))
    saved = store.put_appearance_component(
        {
            "option_id": "athletic_torso",
            "label": "Athletic torso",
            "category": "torso",
            "paths": paths,
            "description": "defined athletic torso proportions",
            "provenance": "locally created synthetic examples",
            "source_kind": "synthetic",
            "adult_subject_confirmed": True,
            "rights_confirmed": True,
            "minimum_examples": 3,
        }
    )
    assert saved["ready"] is True
    store.put_avatar(_adult_profile(component_option_ids=["athletic_torso"]))
    payload = store.profile_to_avatar_payload("ava-01")
    assert payload["body_refs"] == paths
    assert "selected torso: Athletic torso" in " ".join(payload["persistent_features"])


def test_appearance_component_below_threshold_is_not_applied(tmp_path: Path):
    image = tmp_path / "shape.png"
    image.write_bytes(b"image")
    store = LibraryStore(tmp_path / "library")
    saved = store.put_appearance_component(
        {
            "option_id": "shape_draft",
            "label": "Shape draft",
            "category": "body_type",
            "paths": [str(image)],
            "description": "draft body shape",
            "provenance": "synthetic source",
            "source_kind": "synthetic",
            "adult_subject_confirmed": True,
            "rights_confirmed": True,
            "minimum_examples": 3,
        }
    )
    assert saved["ready"] is False
    assert store.selected_appearance_components(["shape_draft"]) == []


def test_ready_appearance_component_drives_body_reference_prompt(tmp_path: Path):
    store = LibraryStore(tmp_path / "library")
    paths = []
    for index in range(3):
        path = tmp_path / f"shape-{index}.png"
        path.write_bytes(b"image")
        paths.append(str(path))
    store.put_appearance_component(
        {
            "option_id": "pear_shape",
            "label": "Pear shape",
            "category": "proportions",
            "paths": paths,
            "description": "balanced pear-shaped proportions",
            "provenance": "authorized synthetic set",
            "source_kind": "synthetic",
            "adult_subject_confirmed": True,
            "rights_confirmed": True,
        }
    )
    api = DesktopAPI.__new__(DesktopAPI)
    api.library = store
    result = api.avatar_designer_prompt(
        {
            "avatar": _adult_profile(component_option_ids=["pear_shape"]),
            "slot_id": "body_full_front",
            "render_mode": "body_ref2va",
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["component_count"] == 1
    assert "Pear shape (proportions)" in result["prompt"]


def test_face_shape_component_routes_identity_slot_through_ref2va(tmp_path: Path):
    store = LibraryStore(tmp_path / "library")
    paths = []
    for index in range(3):
        path = tmp_path / f"face-{index}.png"
        path.write_bytes(b"image")
        paths.append(str(path))
    store.put_appearance_component(
        {
            "option_id": "oval_face",
            "label": "Oval face",
            "category": "face_shape",
            "paths": paths,
            "description": "soft oval facial structure",
            "provenance": "authorized synthetic set",
            "source_kind": "synthetic",
            "adult_subject_confirmed": True,
            "rights_confirmed": True,
        }
    )
    api = DesktopAPI.__new__(DesktopAPI)
    api.library = store
    result = api.avatar_designer_prompt(
        {
            "avatar": _adult_profile(component_option_ids=["oval_face"]),
            "slot_id": "identity_front",
            "render_mode": "body_ref2va",
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["body_ref_count"] == 3
    assert "Oval face (face shape)" in result["prompt"]


def test_desktop_designer_prompt_gates_unverified_adult_anatomy():
    api = DesktopAPI.__new__(DesktopAPI)
    denied = api.avatar_designer_prompt(
        {
            "avatar": _adult_profile(subject_kind="unknown", age_verified_18_plus=False, consent_confirmed=False),
            "slot_id": "anatomy_pelvis_front",
        }
    )
    assert denied["ok"] is False
    allowed = api.avatar_designer_prompt(
        {"avatar": _adult_profile(sex="female", nude_mode="both"), "slot_id": "anatomy_pelvis_front"}
    )
    assert allowed["ok"] is True
    assert allowed["engine"] == "h3_fl2va"
    assert "29-year-old adult" in allowed["prompt"]
    assert "selected sex anatomy is female" in allowed["prompt"]
    assert "anatomy-first nude validation pass" in allowed["prompt"]


def test_designer_rejects_apparent_age_below_18():
    with pytest.raises(ValueError, match="greater than or equal to 18"):
        AvatarProfile.model_validate(_adult_profile(apparent_age_years=17))


def test_designer_accepts_apparent_age_above_100():
    profile = AvatarProfile.model_validate(_adult_profile(apparent_age_years=104))
    assert profile.apparent_age_years == 104


def test_designer_rejects_unknown_subject_kind_values():
    with pytest.raises(ValueError, match="synthetic"):
        AvatarProfile.model_validate(_adult_profile(subject_kind="minor"))


def _guide_record(path: Path, **updates):
    data = {
        "guide_id": "pelvis-guide-01",
        "label": "Clinical pelvis front",
        "path": str(path),
        "region": "pelvis_front",
        "source_kind": "synthetic",
        "provenance": "Locally authored synthetic reference",
        "adult_subject_confirmed": True,
        "rights_confirmed": True,
        "consent_or_release_confirmed": False,
    }
    data.update(updates)
    return data


def test_anatomy_guide_library_validates_provenance_and_resolves_avatar_selection(tmp_path: Path):
    image = tmp_path / "guide.png"
    image.write_bytes(b"png")
    store = LibraryStore(tmp_path / "library")
    saved = store.put_anatomy_guide(_guide_record(image))
    assert AnatomyGuide.model_validate(saved).region == "pelvis_front"

    with pytest.raises(ValueError, match="consent or model-release"):
        store.put_anatomy_guide(
            _guide_record(
                image,
                guide_id="photo-guide-01",
                source_kind="licensed_adult_photo",
                consent_or_release_confirmed=False,
            )
        )
    omitted_release = _guide_record(
        image,
        guide_id="photo-guide-02",
        source_kind="consenting_adult_photo",
    )
    omitted_release.pop("consent_or_release_confirmed")
    with pytest.raises(ValueError, match="consent or model-release"):
        store.put_anatomy_guide(omitted_release)

    store.put_avatar(_adult_profile(anatomy_guide_ids=[saved["guide_id"]]))
    payload = store.profile_to_avatar_payload("ava-01")
    assert payload["anatomy_guides"] == [
        {
            "guide_id": "pelvis-guide-01",
            "label": "Clinical pelvis front",
            "path": str(image),
            "region": "pelvis_front",
        }
    ]
    assert store.delete_anatomy_guide("pelvis-guide-01")
    assert store.get_avatar("ava-01")["anatomy_guide_ids"] == []


def test_desktop_guided_prompt_uses_validated_guides_only(tmp_path: Path):
    image = tmp_path / "guide.png"
    image.write_bytes(b"png")
    store = LibraryStore(tmp_path / "library")
    store.put_anatomy_guide(_guide_record(image))
    api = DesktopAPI.__new__(DesktopAPI)
    api.library = store

    result = api.avatar_designer_prompt(
        {
            "avatar": _adult_profile(anatomy_guide_ids=["pelvis-guide-01"]),
            "slot_id": "anatomy_pelvis_front",
            "render_mode": "guided_ref2va",
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["guide_count"] == 1
    assert "guide-only clinical topology references" in result["prompt"]
    assert "never copy their face, identity, age, ethnicity" in result["prompt"]


def test_guided_designer_queue_forces_low_memory_ref2va(tmp_path: Path):
    source = tmp_path / "source.png"
    guide = tmp_path / "guide.png"
    source.write_bytes(b"source")
    guide.write_bytes(b"guide")
    store = LibraryStore(tmp_path / "library")
    store.put_anatomy_guide(_guide_record(guide))
    captured = {}

    class FakeQueue:
        def add(self, item):
            captured["item"] = item
            return "queue-guided-01"

    api = DesktopAPI.__new__(DesktopAPI)
    api.library = store
    api.queue = FakeQueue()
    api._settings = lambda: {"default_h3_preset": "vertical_4070"}
    api._archive_dir = lambda: None
    api._provider_config = lambda payload: ProviderConfig(base_url="http://127.0.0.1:8188")
    api.avatar_designer_renderer_status = lambda payload=None: {"ready": True, "guided_ready": True}

    def resolve(render_payload, avatar, shot):
        from avatar_v2.composer import build_job

        assert render_payload["runtime_profile"] == "low_memory"
        assert render_payload["h3_preset"] == "vertical_fast"
        assert render_payload["advanced"]["H3_REF_IMAGE_SIZE"] == "match"
        return build_job(avatar, shot), {
            "profile": "low_memory",
            "local_h3_allowed": True,
        }, "builtin:h3_ref2va"

    api._resolve_runtime = resolve
    result = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(
                anatomy_guide_ids=["pelvis-guide-01"],
                source_refs=[str(source)],
            ),
            "slot_id": "anatomy_pelvis_front",
            "render_mode": "guided_ref2va",
            "source_image": str(source),
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["guide_count"] == 1
    assert result["reference_bindings"]["h3_pictures"] == 2
    assert captured["item"].runtime["designer_render_mode"] == "guided_ref2va"


def test_body_reference_designer_queue_binds_identity_and_body_refs(tmp_path: Path):
    source = tmp_path / "source.png"
    body_front = tmp_path / "body-front.png"
    body_rear = tmp_path / "body-rear.png"
    source.write_bytes(b"source")
    body_front.write_bytes(b"front")
    body_rear.write_bytes(b"rear")
    captured = {}

    class FakeQueue:
        def add(self, item):
            captured["item"] = item
            return "queue-body-ref-01"

    api = DesktopAPI.__new__(DesktopAPI)
    api.queue = FakeQueue()
    api._settings = lambda: {"default_h3_preset": "vertical_4070"}
    api._archive_dir = lambda: None
    api._provider_config = lambda payload: ProviderConfig(base_url="http://127.0.0.1:8188")
    api.avatar_designer_renderer_status = lambda payload=None: {"ready": True, "guided_ready": True}

    def resolve(render_payload, avatar, shot):
        from avatar_v2.composer import build_job

        assert render_payload["runtime_profile"] == "low_memory"
        assert render_payload["h3_preset"] == "vertical_fast"
        assert render_payload["reference_mode"] == "multi_match"
        assert render_payload["advanced"]["H3_REF_IMAGE_SIZE"] == "match"
        assert avatar.identity_refs == [str(source)]
        assert avatar.body_refs == [str(body_front), str(body_rear)]
        assert avatar.sex == "female"
        return build_job(avatar, shot), {
            "profile": "low_memory",
            "local_h3_allowed": True,
        }, "builtin:h3_ref2va"

    api._resolve_runtime = resolve
    result = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(
                sex="female",
                source_refs=[str(source)],
                body_refs={"full_front": str(body_front), "rear": str(body_rear)},
            ),
            "slot_id": "body_full_left",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["body_ref_count"] == 2
    assert result["reference_bindings"]["source_counts"]["identity"] == 1
    assert result["reference_bindings"]["source_counts"]["body"] == 2
    assert result["reference_bindings"]["h3_pictures"] == 3
    assert captured["item"].runtime["designer_render_mode"] == "body_ref2va"
    assert captured["item"].runtime["profile_coverage"] == "nude"
    assert "same-avatar" in result["prompt"]


def test_body_reference_mode_keeps_identity_slots_on_fl2va(tmp_path: Path):
    source = tmp_path / "source.png"
    body = tmp_path / "body.png"
    source.write_bytes(b"source")
    body.write_bytes(b"body")
    captured = {}

    class FakeQueue:
        def add(self, item):
            captured["item"] = item
            return "queue-identity-01"

    api = DesktopAPI.__new__(DesktopAPI)
    api.queue = FakeQueue()
    api._settings = lambda: {"default_h3_preset": "vertical_4070"}
    api._archive_dir = lambda: None
    api._provider_config = lambda payload: ProviderConfig(base_url="http://127.0.0.1:8188")
    api.avatar_designer_renderer_status = lambda payload=None: {"ready": True, "guided_ready": True}

    def resolve(render_payload, avatar, shot):
        from avatar_v2.composer import build_job

        assert render_payload["reference_mode"] == "exact_primary"
        assert avatar.body_refs == []
        return build_job(avatar, shot), {
            "profile": "low_memory_quality",
            "local_h3_allowed": True,
        }, "builtin:h3_fl2va"

    api._resolve_runtime = resolve
    result = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(source_refs=[str(source)], body_refs={"full_front": str(body)}),
            "slot_id": "identity_front",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert result["ok"] is True
    assert result["engine"] == "h3_fl2va"
    assert result["body_ref_count"] == 0
    assert captured["item"].runtime["designer_render_mode"] == "fl2va"


def test_body_reference_mode_rejects_missing_or_unapproved_references(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    api = DesktopAPI.__new__(DesktopAPI)

    no_refs = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(source_refs=[str(source)]),
            "slot_id": "body_full_front",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert no_refs["ok"] is False
    assert "body / proportions reference" in no_refs["error"]

    missing = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(
                source_refs=[str(source)],
                body_refs={"full_front": str(tmp_path / "missing.png")},
            ),
            "slot_id": "body_full_front",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert missing["ok"] is False
    assert "body reference files are missing" in missing["error"]

    body = tmp_path / "body.png"
    body.write_bytes(b"body")
    denied = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(
                subject_kind="unknown",
                age_verified_18_plus=False,
                consent_confirmed=False,
                source_refs=[str(source)],
                body_refs={"full_front": str(body)},
            ),
            "slot_id": "body_full_front",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert denied["ok"] is False
    assert "verified 18+ age and confirmed consent" in denied["error"]


def test_body_reference_mode_caps_h3_pack_and_prioritizes_requested_role(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    refs = {}
    for index in range(10):
        path = tmp_path / f"body-{index}.png"
        path.write_bytes(str(index).encode())
        refs[f"role_{index}"] = str(path)
    requested = tmp_path / "requested.png"
    requested.write_bytes(b"requested")
    refs["left"] = str(requested)
    captured = {}

    class FakeQueue:
        def add(self, item):
            captured["item"] = item
            return "queue-capped-01"

    api = DesktopAPI.__new__(DesktopAPI)
    api.queue = FakeQueue()
    api._settings = lambda: {"default_h3_preset": "vertical_4070"}
    api._archive_dir = lambda: None
    api._provider_config = lambda payload: ProviderConfig(base_url="http://127.0.0.1:8188")
    api.avatar_designer_renderer_status = lambda payload=None: {"ready": True, "guided_ready": True}

    def resolve(render_payload, avatar, shot):
        from avatar_v2.composer import build_job

        assert len(avatar.body_refs) == 8
        assert avatar.body_refs[0] == str(requested)
        return build_job(avatar, shot), {"profile": "low_memory", "local_h3_allowed": True}, "builtin:h3_ref2va"

    api._resolve_runtime = resolve
    result = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(source_refs=[str(source)], body_refs=refs),
            "slot_id": "body_full_left",
            "render_mode": "body_ref2va",
            "source_image": str(source),
        }
    )
    assert result["ok"] is True
    assert result["body_ref_count"] == 8
    assert result["saved_body_ref_count"] == 11
    assert result["reference_bindings"]["h3_pictures"] == 9


def test_complete_validation_rotates_full_pool_and_prioritizes_accepted_target():
    primary = "/refs/source-0.png"
    sources = [f"/refs/source-{index}.png" for index in range(12)]
    target = "/accepted/body-left.png"
    profile = AvatarProfile.model_validate(
        _adult_profile(
            source_refs=sources,
            identity_refs={"front": "/accepted/face-front.png"},
            body_refs={"left": target},
            anatomy_refs={"pelvis_front": "/accepted/pelvis-front.png"},
        )
    )
    slot = slot_by_id("body_full_left")

    batches = [
        DesktopAPI._designer_body_refs(
            profile,
            slot,
            full_reference_pool=True,
            primary_source=primary,
            cycle_index=index,
            cycle_total=3,
        )
        for index in range(3)
    ]

    assert all(len(batch) == 8 for batch in batches)
    assert all(batch[0] == target for batch in batches)
    assert all(primary not in batch for batch in batches)
    assert set().union(*map(set, batches)) == {
        *sources[1:],
        "/accepted/face-front.png",
        target,
        "/accepted/pelvis-front.png",
    }


def test_complete_validation_queue_uses_ref2va_and_marks_candidates(tmp_path: Path):
    paths = []
    for index in range(10):
        path = tmp_path / f"reference-{index}.png"
        path.write_bytes(str(index).encode())
        paths.append(str(path))
    captured = {}

    class FakeQueue:
        def add(self, item):
            captured["item"] = item
            return "queue-validation-01"

    api = DesktopAPI.__new__(DesktopAPI)
    api.queue = FakeQueue()
    api._settings = lambda: {"default_h3_preset": "vertical_4070"}
    api._archive_dir = lambda: None
    api._provider_config = lambda payload: ProviderConfig(base_url="http://127.0.0.1:8188")
    api.avatar_designer_renderer_status = lambda payload=None: {"ready": True, "guided_ready": True}

    def resolve(render_payload, avatar, shot):
        from avatar_v2.composer import build_job

        assert render_payload["reference_mode"] == "multi_match"
        assert avatar.identity_refs == [paths[0]]
        assert paths[0] not in avatar.body_refs
        assert len(avatar.body_refs) == 8
        return build_job(avatar, shot), {"profile": "low_memory", "local_h3_allowed": True}, "builtin:h3_ref2va"

    api._resolve_runtime = resolve
    result = api.queue_avatar_designer(
        {
            "avatar": _adult_profile(
                source_refs=paths,
                identity_refs={"front": paths[1]},
                body_refs={"left": paths[2]},
                anatomy_refs={"pelvis_front": paths[3]},
            ),
            "slot_id": "body_full_left",
            "render_mode": "fl2va",
            "source_image": paths[0],
            "validation_run": True,
            "full_reference_pool": True,
            "reference_cycle_index": 1,
            "reference_cycle_total": 3,
        }
    )

    assert result["ok"] is True
    assert result["engine"] == "h3_ref2va"
    assert result["validation_run"] is True
    assert result["full_reference_pool"] is True
    assert captured["item"].runtime["validation_run"] is True
    assert captured["item"].runtime["full_reference_pool"] is True
    assert captured["item"].label.startswith("Avatar Validation H3")
    assert captured["item"].job.asset_map["OUTPUT_PREFIX"].startswith("avatar_validation_h3_")


def test_h3_candidate_frame_is_appended_to_designer_outputs(tmp_path: Path, monkeypatch):
    video = tmp_path / "designer.mp4"
    video.write_bytes(b"video")
    item = type("Item", (), {"outputs": [str(video)]})()
    queue = RenderQueue.__new__(RenderQueue)

    def fake_extract(source, target):
        assert source == str(video)
        target.write_bytes(b"png")
        return target

    monkeypatch.setattr(queue, "_extract_last_frame", fake_extract)
    queue._extract_candidate_frame(item)

    assert item.outputs[-1].endswith("designer_candidate.png")
    assert Path(item.outputs[-1]).read_bytes() == b"png"


def test_desktop_exposes_profile_build_controls():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="designerRunNext"' in html
    assert 'id="designerRunProfile"' in html
    assert "function designerRunNext()" in html
    assert "function designerRunProfile()" in html
    assert 'id="designerRunFullProfile"' in html
    assert "function designerRunFullProfile()" in html
    assert "full_reference_pool:validationRun" in html
    assert "referencePasses=Math.max" in html
    assert "full rotating image pool" in html
    assert "existing accepted images stay preserved" in html
    assert "Existing accepted views will be skipped" not in html
    assert "complete ${slots.length}-view" not in html
    assert 'id="designerRunProfile" onclick="designerRunProfile()">' in html
    assert "function designerEnsureIdentityDefaults()" in html
    assert "function designerShowPrerequisiteErrors(errors)" in html
    assert 'id="vApparentAge"' in html
    assert 'id="vApparentAge" type="number" min="18" value="25"' in html
    assert 'id="vApparentAge" type="number" min="18" max=' not in html
    assert "apparent_age_years:Number($('vApparentAge').value)||null" in html
    assert "errors.push('adult apparent age of 18 or older')" in html
    assert 'id="vSex"' in html
    assert '<option value="female">female</option>' in html
    assert '<option value="male">male</option>' in html
    assert 'id="vAdultAuth"' in html
    assert 'id="vNude"' in html
    assert '<option value="both">Nude, then clothed</option>' in html
    assert "coverage_variant:coverage" in html
    assert "coverageVariants=$('vNude').value==='both'?['nude','clothed']" in html
    assert "Do not identify the person and do not infer or report sex" not in html
    assert "It does not identify the person or infer sex" in html
    assert "prompt_has_age" in html
    assert 'id="designerRenderMode"' in html
    assert '<option value="body_ref2va">Identity + body refs · Ref2VA · 8 GB experimental</option>' in html
    assert "function designerBodyRefs()" in html
    assert "BODY / COMPONENT'} REF ×${referenceCountLabel}" in html
    assert "function designerReferenceCountForSlot(slot)" in html
    assert "bodyReferenced=mode==='body_ref2va'&&(slot.group!=='identity'||identityComponent)" in html
    assert 'id="designerCacheDir"' in html
    assert "function designerAnalyzeCache()" in html
    assert "function designerResetCacheMappings()" in html
    assert "previous working mappings replaced" in html
    assert 'id="componentSelectors"' in html
    assert "function componentSave()" in html
    assert 'id="guideList"' in html
    assert "function guideFormEntry()" in html
    assert 'id="gStatus"' in html
    assert 'onclick="guidePickImage()"' in html
    assert "function guideValidationErrors(entry)" in html
    assert "function guideShowErrors(errors)" in html
    assert "function guideSuggestedIdentity(path)" in html
    assert "Save guide + attach to avatar" in html
    assert "await window.pywebview.api.vault_save(avatarEntry)" in html
    assert "Guide save failed:" in html
    assert "anatomy_guide_ids:[...state.designer.anatomy_guide_ids]" in html
    assert "Confirm Profile Build" in html
    assert "profileConfirmUntil=Date.now()+15000" in html
    assert "const r=await vaultSave();if(r.ok)" in html
    assert "else $('designerIteration').value=previous" in html
    assert "A1111" not in html
    assert "Juggernaut" not in html
