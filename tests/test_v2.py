from avatar_v2.composer import build_job
from avatar_v2.models import AvatarManifest, ShotSpec
from avatar_v2.policy import evaluate_policy
from avatar_v2.providers.comfyui import replace_placeholders
from avatar_v2.router import choose_engine


def synthetic_avatar():
    return AvatarManifest.model_validate({
        "avatar_id": "ava",
        "display_name": "Ava",
        "subjects": [{
            "id": "ava",
            "kind": "synthetic",
            "age_verified_18_plus": True,
            "consent_confirmed": True,
        }],
        "identity_refs": ["front.png"],
        "persistent_features": ["freckles"],
    })


def test_general_job_allowed():
    shot = ShotSpec.model_validate({"shot_id": "s1", "user_prompt": "walk toward camera"})
    decision = evaluate_policy(synthetic_avatar(), shot)
    assert decision.allowed


def test_adult_job_requires_verified_consent():
    avatar = synthetic_avatar().model_copy(deep=True)
    avatar.subjects[0].consent_confirmed = False
    shot = ShotSpec.model_validate({
        "shot_id": "s1", "content_class": "adult_sexual", "user_prompt": "adult scene"
    })
    assert not evaluate_policy(avatar, shot).allowed


def test_verified_synthetic_adult_job_allowed():
    shot = ShotSpec.model_validate({
        "shot_id": "s1", "content_class": "adult_nudity", "user_prompt": "adult scene"
    })
    assert evaluate_policy(synthetic_avatar(), shot).allowed


def test_motion_reference_routes_to_wan():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "walk",
        "references": {"motion_video": "walk.mp4"},
    })
    assert choose_engine(shot) == "wan22"


def test_last_frame_routes_to_ltx():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "turn",
        "references": {"last_frame": "end.png"},
    })
    assert choose_engine(shot) == "ltx25"


def test_composer_builds_temporal_prompt():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "walk",
        "duration_s": 4,
        "action_beats": [
            {"start_s": 0, "end_s": 2, "description": "walk"},
            {"start_s": 2, "end_s": 4, "description": "smile"},
        ],
    })
    job = build_job(synthetic_avatar(), shot)
    assert "[0.000-2.000] walk" in job.prompt
    assert job.asset_map["FRAMES"] == 96


def test_placeholder_replacement_preserves_numeric_types():
    workflow = {"a": "${WIDTH}", "b": "x-${FPS}", "c": ["${PROMPT}"]}
    result = replace_placeholders(workflow, {"WIDTH": 720, "FPS": 24, "PROMPT": "hello"})
    assert result["a"] == 720
    assert result["b"] == "x-24"
    assert result["c"] == ["hello"]
