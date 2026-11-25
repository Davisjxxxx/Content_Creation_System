"""CPU-only unit tests for the Structured Prompt Schema and Prompt Adherence Benchmark Foundation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from avatar_v2.adherence_benchmark import (
    ALL_ADHERENCE_CATEGORIES,
    AdherenceCaseResult,
    RequirementVerdict,
    default_adherence_suite,
    summarize_adherence_results,
)
from avatar_v2.prompt_schema import (
    EnhancementItem,
    GenerationConstraints,
    PromptAttribute,
    ReferenceAssetItem,
    RequirementItem,
    StructuredPromptSpec,
    SubjectAttributeSpec,
)


def test_automatic_mode_preserves_original_prompt_and_roundtrips_json():
    spec = StructuredPromptSpec.from_natural_language(
        "  Candid studio portrait of an adult woman instructor  ",
        task="image_generate",
        excluded=["watermark", "cartoon"],
    )
    assert spec.original_prompt == "Candid studio portrait of an adult woman instructor"
    assert spec.intent.input_mode == "automatic"
    assert spec.constraints.local_only is True
    assert spec.required_attributes[0].provenance == "user_explicit"

    restored = StructuredPromptSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec
    positive, negative = restored.to_baseline_prompt()
    assert "Candid studio portrait" in positive
    assert negative == "watermark, cartoon"


def test_guided_locking_and_deterministic_controlled_randomization():
    spec = StructuredPromptSpec.from_guided(
        original_prompt="Editorial portrait of an adult presenter",
        attributes={"lighting": "soft window light", "visual_style": "35mm film"},
        locked_fields={"lighting"},
        randomized_fields={
            "environment": ["minimalist loft", "sunlit cafe", "modern studio"],
            "atmosphere": ["warm and candid", "calm and focused"],
        },
        excluded=["cafe"],  # randomizer must filter out excluded candidates!
        seed=12345,
    )
    assert spec.locked_attributes() == {"lighting": "soft window light"}
    assert spec.lighting.state == "locked"
    assert spec.visual_style.state == "specified"
    assert spec.environment.state == "randomized"

    run1 = spec.apply_randomization(seed=12345)
    run2 = spec.apply_randomization(seed=12345)
    assert run1.environment.value == run2.environment.value
    assert run1.atmosphere.value == run2.atmosphere.value
    assert "cafe" not in run1.environment.value.lower()
    assert run1.environment.provenance == "randomizer"
    # Locked and specified fields must never be mutated by randomization.
    assert run1.lighting.value == "soft window light" and run1.lighting.state == "locked"
    assert run1.visual_style.value == "35mm film" and run1.visual_style.state == "specified"


def test_explicit_user_requirements_override_automatic_enhancements():
    spec = StructuredPromptSpec.from_guided(
        original_prompt="Studio headshot",
        attributes={"lighting": "dramatic rim light"},
        locked_fields={"lighting"},
        excluded=["neon glow"],
    )
    # Cannot overwrite locked/specified lighting.
    unchanged, applied = spec.apply_enhancement("lighting", "flat overcast light", provenance="local_llm")
    assert applied is False and unchanged.lighting.value == "dramatic rim light"

    # Cannot apply enhancement matching excluded_attributes.
    unchanged2, applied2 = spec.apply_enhancement("atmosphere", "cyberpunk neon glow", provenance="local_llm")
    assert applied2 is False and unchanged2.atmosphere.value == ""

    # Can populate an automatic/empty attribute cleanly with provenance recorded.
    enhanced, applied3 = spec.apply_enhancement("composition", "medium close-up at eye level", provenance="local_llm")
    assert applied3 is True
    assert enhanced.composition.value == "medium close-up at eye level"
    assert enhanced.composition.provenance == "local_llm"
    assert enhanced.composition.state == "automatic"


def test_conflict_detection_catches_contradictions_missing_refs_and_policy_blocks():
    # 1. Required vs excluded + semantic antonym + blocked optional enhancement + missing edit_source + cloud in local_only.
    spec = StructuredPromptSpec(
        original_prompt="Adult woman smiling indoors in daylight wearing a red dress",
        intent={"task": "image_edit", "input_mode": "guided"},
        environment=PromptAttribute(value="outdoors at night", state="locked", provenance="guided_ui"),
        excluded_attributes=["red dress"],
        optional_enhancements=[EnhancementItem(field="environment", value="quiet studio")],
        constraints=GenerationConstraints(local_only=True, model_preference="runway:gen4.5"),
    )
    codes = {c.code for c in spec.detect_conflicts()}
    assert "required_excluded_contradiction" in codes
    assert "semantic_contradiction" in codes
    assert "enhancement_blocked_by_explicit_attribute" in codes
    assert "missing_edit_source" in codes
    assert "cloud_model_disallowed_in_local_only_mode" in codes

    # 2. Adult-only safety gate catches minor phrasing.
    minor_spec = StructuredPromptSpec.from_natural_language("Portrait of a 14-year-old schoolgirl")
    minor_codes = {c.code for c in minor_spec.detect_conflicts()}
    assert "policy_minor_depiction_blocked" in minor_codes

    # 3. Clean video spec has zero conflicts and auto-syncs frames from duration_s * fps.
    clean_video = StructuredPromptSpec(
        original_prompt="Adult presenter smiling and nodding gently at a studio desk",
        intent={"task": "video_generate", "input_mode": "automatic"},
        subjects=[SubjectAttributeSpec(id="presenter", apparent_age_years=32)],
        reference_assets=[ReferenceAssetItem(asset_id="start", role="start_frame", path="/tmp/start.png")],
        constraints=GenerationConstraints(width=480, height=832, fps=16, duration_s=5.0, frames=1),
    )
    assert clean_video.constraints.frames == 80
    assert clean_video.detect_conflicts() == []


def test_adherence_benchmark_suite_covers_all_13_categories_and_separates_scores():
    suite = default_adherence_suite()
    categories = [c.category for c in suite]
    assert len(suite) == 13
    assert set(categories) == set(ALL_ADHERENCE_CATEGORIES)
    assert sum(1 for c in suite if c.target_model == "flux2_9b") == 10
    assert sum(1 for c in suite if c.target_model == "wan22_14b") == 3

    for case in suite:
        # Every benchmark case must have verifiable mandatory requirements and zero schema conflicts.
        assert len(case.spec.required_attributes) >= 2
        assert case.spec.detect_conflicts() == []
        pub = case.public()
        assert pub["baseline_positive"] and pub["mandatory_requirements"]

    # Score aggregation keeps objective mandatory pass rate independent from subjective quality.
    r1 = AdherenceCaseResult(
        case_id="adh_03_quantities",
        category="object_quantities",
        target_model="flux2_9b",
        requirement_verdicts=[RequirementVerdict("three_mugs", True), RequirementVerdict("two_notebooks", False)],
        subjective_quality_1_to_5=4.5,  # High visual quality, but missed a mandatory count!
    )
    r2 = AdherenceCaseResult(
        case_id="adh_07_typography",
        category="typography",
        target_model="flux2_9b",
        requirement_verdicts=[RequirementVerdict("exact_text", True), RequirementVerdict("legible_contrast", True)],
        subjective_quality_1_to_5=3.5,
    )
    summary = summarize_adherence_results([r1, r2])
    assert r1.all_mandatory_passed is False and r1.mandatory_pass_rate == 0.5
    assert r2.all_mandatory_passed is True and r2.mandatory_pass_rate == 1.0
    assert summary["strict_all_mandatory_cases"] == 1
    assert summary["overall_mandatory_pass_rate"] == 0.75
    assert summary["mean_subjective_quality_1_to_5"] == 4.0


def test_benchmark_stage_timing_parser_and_list_adherence_cli():
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import gpu_memory_benchmark as bench

    sample_log = (
        "[INFO] loaded partially; 212.82 MB usable, 0.00 MB loaded, 8475.46 MB offloaded\n"
        "50%|█████ | 1/2 [01:03<01:03, 63.78s/it]\n"
        "100%|██████████| 2/2 [02:06<00:00, 63.27s/it]\n"
        "[INFO] Prompt executed in 311.49 seconds\n"
    )
    parsed = bench.parse_comfyui_stage_timings(sample_log)
    assert parsed["comfy_total_s"] == 311.49
    assert parsed["mean_s_per_it"] == 63.53
    assert parsed["loaded_mb"] == [0.0] and parsed["offloaded_mb"] == [8475.46]
