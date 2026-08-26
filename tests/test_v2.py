from avatar_v2.composer import build_job
from avatar_v2.h3_access import recommend_local_h3
from avatar_v2.h3_runtime import actual_duration_s, preset_by_id, valid_frame_count
from avatar_v2.models import AvatarManifest, ShotSpec
from avatar_v2.policy import evaluate_policy
from avatar_v2.providers.comfyui import replace_placeholders
from avatar_v2.router import choose_engine
from avatar_v2.runtime_profiles import resolve_profile
from avatar_v2.workflows_builder import build_h3_ref2va_workflow


def synthetic_avatar():
    return AvatarManifest.model_validate({
        "avatar_id": "ava",
        "display_name": "Ava",
        "apparent_age_years": 29,
        "subjects": [{
            "id": "ava",
            "kind": "synthetic",
            "age_verified_18_plus": True,
            "consent_confirmed": True,
        }],
        "identity_refs": ["front.png", "three-quarter.png"],
        "body_refs": ["body.png"],
        "persistent_features": ["freckles"],
    })


def test_general_job_allowed():
    shot = ShotSpec.model_validate({"shot_id": "s1", "user_prompt": "walk toward camera"})
    assert evaluate_policy(synthetic_avatar(), shot).allowed


def test_prompt_includes_stable_apparent_age_anchor():
    shot = ShotSpec.model_validate({
        "shot_id": "age-anchor",
        "user_prompt": "turn toward camera",
        "engine_preference": "h3_fl2va",
        "references": {"init_image": "front.png"},
    })
    prompt = build_job(synthetic_avatar(), shot).prompt
    assert "subject remains visibly 29 years old in every frame" in prompt
    assert "do not make the subject look younger or older" in prompt


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


def test_motion_reference_routes_to_h3_ref2va():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "walk",
        "references": {"motion_video": "walk.mp4"},
    })
    assert choose_engine(shot) == "h3_ref2va"


def test_last_frame_routes_to_h3_fl2va():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "turn",
        "references": {"last_frame": "end.png"},
    })
    assert choose_engine(shot) == "h3_fl2va"


def test_h3_prompt_uses_official_ref2va_section_order_and_roles():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "walk naturally",
        "engine_preference": "h3_ref2va",
        "references": {"motion_video": "walk.mp4", "scene_image": "street.png"},
    })
    job = build_job(synthetic_avatar(), shot)
    sections = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    positions = [job.prompt.index(section) for section in sections]
    assert positions == sorted(positions)
    assert "<Subject 1> is the target synthetic adult person" in job.prompt
    assert "<Video 1>: attribute_transfer" in job.prompt
    assert "without transferring source identity" in job.prompt
    assert job.asset_map["H3_PICTURE_1"] == "front.png"
    assert job.asset_map["H3_VIDEO_1"] == "walk.mp4"


def test_h3_ref2va_keeps_anatomy_guides_separate_from_identity():
    avatar_data = synthetic_avatar().model_dump()
    avatar_data["anatomy_guides"] = [
        {
            "guide_id": "pelvis-guide-01",
            "label": "Clinical pelvis front",
            "path": "guide.png",
            "region": "pelvis_front",
        }
    ]
    avatar = AvatarManifest.model_validate(avatar_data)
    shot = ShotSpec.model_validate({
        "shot_id": "guided",
        "content_class": "adult_nudity",
        "user_prompt": "neutral clinical profile view",
        "engine_preference": "h3_ref2va",
    })
    job = build_job(avatar, shot)
    assert job.asset_map["H3_PICTURE_1"] == "front.png"
    assert job.asset_map["H3_PICTURE_3"] == "guide.png"
    assert "Preserve appearance and identity from <Picture 1>, <Picture 2>;" in job.prompt
    assert "Preserve appearance and identity from <Picture 1>, <Picture 2>, <Picture 3>" not in job.prompt
    assert "only as clinical topology and landmark guides" in job.prompt
    assert "Never copy their face, identity, age, ethnicity" in job.prompt
    assert "<Picture 3>: weak_reference" in job.prompt


def test_h3_primary_reference_is_bound_at_best_fidelity():
    shot = ShotSpec.model_validate({
        "shot_id": "primary-ref",
        "user_prompt": "hold identity",
        "engine_preference": "h3_ref2va",
        "references": {"init_image": "front.png"},
    })
    job = build_job(synthetic_avatar(), shot)
    graph = build_h3_ref2va_workflow(job.asset_map)
    image_nodes = {node_id: node for node_id, node in graph.items() if node["class_type"] == "LoadImage"}
    task = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo")
    assert len(image_nodes) == 3
    assert task["inputs"]["ref_images"][0][0] in image_nodes
    assert image_nodes[task["inputs"]["ref_images"][0][0]]["inputs"]["image"] == "front.png"
    assert task["inputs"]["ref_image_size"] == "max"


def test_fl2va_prompt_uses_three_core_fields():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "turn smoothly",
        "engine_preference": "h3_fl2va",
        "references": {"init_image": "start.png", "last_frame": "end.png"},
    })
    job = build_job(synthetic_avatar(), shot)
    assert "integrated_multimodal_description:" in job.prompt
    assert "overall_soundscape:" in job.prompt
    assert "non_diegetic_music:" in job.prompt
    assert "<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark" in job.prompt


def test_adult_sexual_prompt_preprograms_anatomy_and_motion_fidelity():
    shot = ShotSpec.model_validate({
        "shot_id": "adult-motion",
        "content_class": "adult_sexual",
        "user_prompt": "perform a moving adult scene",
        "duration_s": 6,
        "engine_preference": "h3_fl2va",
        "references": {"init_image": "start.png"},
    })
    prompt = build_job(synthetic_avatar(), shot).prompt
    assert "external genital structures across every frame" in prompt
    assert "Contact mechanics must remain physically coherent" in prompt
    assert "stable pelvic orientation" in prompt
    assert "Prefer slower speed, smaller range" in prompt
    assert "Reveal anatomy outside the opening crop gradually" in prompt


def test_general_prompt_does_not_add_adult_anatomy_language():
    shot = ShotSpec.model_validate({
        "shot_id": "general-motion",
        "content_class": "general",
        "user_prompt": "walk through the room",
        "engine_preference": "h3_fl2va",
        "references": {"init_image": "start.png"},
    })
    prompt = build_job(synthetic_avatar(), shot).prompt
    assert "external genital structures" not in prompt
    assert "Contact mechanics must remain physically coherent" not in prompt


def test_composer_builds_temporal_prompt():
    shot = ShotSpec.model_validate({
        "shot_id": "s1",
        "user_prompt": "walk",
        "duration_s": 4,
        "engine_preference": "h3_fl2va",
        "action_beats": [
            {"start_s": 0, "end_s": 2, "description": "walk"},
            {"start_s": 2, "end_s": 4, "description": "smile"},
        ],
    })
    job = build_job(synthetic_avatar(), shot)
    assert "[0.000-2.000] walk" in job.prompt
    assert job.asset_map["FRAMES"] == 96


def test_h3_native_frame_grid_and_preset():
    frames = valid_frame_count(5)
    assert frames >= 120
    assert (frames - 5) % 17 == 0
    assert actual_duration_s(5) == frames / 24
    preset = preset_by_id("vertical_4070")
    assert (preset.width, preset.height) == (480, 864)


def test_12gb_auto_prefers_installed_int8():
    stats = {"devices": [{"vram_total": 12 * 1024**3}]}
    models = [
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_bf16.safetensors",
    ]
    rec = recommend_local_h3(
        mode="ref2va",
        system_stats=stats,
        diffusion_models=models,
        local_h3_authorized=True,
    )
    assert rec.model == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def test_local_h3_disabled_without_authorization():
    stats = {"devices": [{"vram_total": 12 * 1024**3}]}
    rec = recommend_local_h3(
        mode="ref2va",
        system_stats=stats,
        diffusion_models=["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        local_h3_authorized=False,
    )
    assert rec.route == "wan22"
    assert rec.local_allowed is False


def test_low_memory_profile_forces_fast_vertical_canvas():
    stats = {"devices": [{"vram_total": 12 * 1024**3}]}
    runtime = resolve_profile(
        profile_id="low_memory",
        engine="h3_ref2va",
        system_stats=stats,
        diffusion_models=["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        local_h3_authorized=True,
        requested_preset="vertical_max",
        duration_s=5,
    )
    assert runtime["model"].endswith("pruned_int8_convrot.safetensors")
    assert (runtime["width"], runtime["height"]) == (352, 608)
    assert runtime["steps"] == 12


def test_forced_int8_profile_falls_back_if_int8_missing():
    stats = {"devices": [{"vram_total": 24 * 1024**3}]}
    runtime = resolve_profile(
        profile_id="int8_12gb",
        engine="h3_fl2va",
        system_stats=stats,
        diffusion_models=["minimax_h3_fl2va_bf16.safetensors"],
        local_h3_authorized=True,
        requested_preset="vertical_4070",
        duration_s=5,
    )
    assert runtime["engine"] == "wan22"
    assert runtime["model"] is None


def test_placeholder_replacement_preserves_numeric_types():
    workflow = {
        "a": "${WIDTH}",
        "b": "x-${FPS}",
        "c": ["${PROMPT}"],
        "d": "${H3_DIFFUSION_MODEL}",
    }
    result = replace_placeholders(
        workflow,
        {"WIDTH": 480, "FPS": 24, "PROMPT": "hello", "H3_DIFFUSION_MODEL": "model.safetensors"},
    )
    assert result["a"] == 480
    assert result["b"] == "x-24"
    assert result["c"] == ["hello"]
    assert result["d"] == "model.safetensors"
