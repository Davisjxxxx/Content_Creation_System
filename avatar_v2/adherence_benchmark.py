"""Prompt Adherence & Generation Quality Benchmark Foundation (Sprint 2, Section 8).

Defines controlled, repeatable benchmark cases for FLUX.2 9B and Wan 2.2 14B covering all
13 required categories:
  1. photorealism          8. character_consistency
  2. composition           9. reference_adherence
  3. object_quantities    10. complex_natural_language
  4. spatial_relationships 11. video_motion
  5. colors               12. camera_movement
  6. clothing_appearance  13. temporal_consistency
  7. typography

Mandatory requirements are scored independently (pass/fail per requirement) from
subjective visual quality (1-5 scale) so future model compilers, local prompt intelligence,
and automated corrective generation can be evaluated rigorously against this baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .prompt_schema import (
    GenerationConstraints,
    GenerationIntent,
    PromptAttribute,
    ReferenceAssetItem,
    RequirementItem,
    StructuredPromptSpec,
)

AdherenceCategory = Literal[
    "photorealism",
    "composition",
    "object_quantities",
    "spatial_relationships",
    "colors",
    "clothing_appearance",
    "typography",
    "character_consistency",
    "reference_adherence",
    "complex_natural_language",
    "video_motion",
    "camera_movement",
    "temporal_consistency",
]

ALL_ADHERENCE_CATEGORIES: tuple[AdherenceCategory, ...] = (
    "photorealism",
    "composition",
    "object_quantities",
    "spatial_relationships",
    "colors",
    "clothing_appearance",
    "typography",
    "character_consistency",
    "reference_adherence",
    "complex_natural_language",
    "video_motion",
    "camera_movement",
    "temporal_consistency",
)


@dataclass(frozen=True)
class AdherenceBenchmarkCase:
    case_id: str
    category: AdherenceCategory
    target_model: Literal["flux2_9b", "wan22_14b"]
    title: str
    spec: StructuredPromptSpec
    requires_reference_image: bool = False

    def public(self) -> dict[str, Any]:
        positive, negative = self.spec.to_baseline_prompt()
        return {
            "case_id": self.case_id,
            "category": self.category,
            "target_model": self.target_model,
            "title": self.title,
            "requires_reference_image": self.requires_reference_image,
            "mandatory_requirements": [r.model_dump() for r in self.spec.required_attributes],
            "baseline_positive": positive,
            "baseline_negative": negative,
            "constraints": self.spec.constraints.model_dump(),
        }


@dataclass(frozen=True)
class RequirementVerdict:
    requirement_id: str
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class AdherenceCaseResult:
    """Separates objective mandatory-requirement adherence from subjective visual quality."""

    case_id: str
    category: AdherenceCategory
    target_model: str
    requirement_verdicts: list[RequirementVerdict]
    subjective_quality_1_to_5: float | None = None
    output_path: str = ""
    elapsed_s: float | None = None
    peak_vram_mb: int | None = None
    notes: str = ""

    @property
    def mandatory_pass_count(self) -> int:
        return sum(1 for v in self.requirement_verdicts if v.passed)

    @property
    def mandatory_total_count(self) -> int:
        return len(self.requirement_verdicts)

    @property
    def mandatory_pass_rate(self) -> float:
        return round(self.mandatory_pass_count / self.mandatory_total_count, 3) if self.mandatory_total_count else 0.0

    @property
    def all_mandatory_passed(self) -> bool:
        return self.mandatory_total_count > 0 and self.mandatory_pass_count == self.mandatory_total_count

    def public(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "target_model": self.target_model,
            "mandatory_passed": self.mandatory_pass_count,
            "mandatory_total": self.mandatory_total_count,
            "mandatory_pass_rate": self.mandatory_pass_rate,
            "all_mandatory_passed": self.all_mandatory_passed,
            "subjective_quality_1_to_5": self.subjective_quality_1_to_5,
            "requirement_verdicts": [asdict(v) for v in self.requirement_verdicts],
            "output_path": self.output_path,
            "elapsed_s": self.elapsed_s,
            "peak_vram_mb": self.peak_vram_mb,
            "notes": self.notes,
        }


def summarize_adherence_results(results: list[AdherenceCaseResult]) -> dict[str, Any]:
    """Aggregate objective requirement adherence independently from subjective quality ratings."""
    total_reqs = sum(r.mandatory_total_count for r in results)
    passed_reqs = sum(r.mandatory_pass_count for r in results)
    strict_cases = sum(1 for r in results if r.all_mandatory_passed)
    rated = [r.subjective_quality_1_to_5 for r in results if r.subjective_quality_1_to_5 is not None]
    by_category: dict[str, dict[str, Any]] = {}
    for r in results:
        by_category[r.category] = {
            "case_id": r.case_id,
            "target_model": r.target_model,
            "mandatory_pass_rate": r.mandatory_pass_rate,
            "all_mandatory_passed": r.all_mandatory_passed,
            "subjective_quality_1_to_5": r.subjective_quality_1_to_5,
        }
    return {
        "cases_evaluated": len(results),
        "strict_all_mandatory_cases": strict_cases,
        "mandatory_requirements_passed": passed_reqs,
        "mandatory_requirements_total": total_reqs,
        "overall_mandatory_pass_rate": round(passed_reqs / total_reqs, 3) if total_reqs else 0.0,
        "mean_subjective_quality_1_to_5": round(sum(rated) / len(rated), 2) if rated else None,
        "by_category": by_category,
    }


def _img_case(
    case_id: str,
    category: AdherenceCategory,
    title: str,
    prompt: str,
    reqs: list[tuple[str, str]],
    *,
    excluded: list[str] | None = None,
    requires_ref: bool = False,
    seed: int = 51001,
) -> AdherenceBenchmarkCase:
    refs = (
        [ReferenceAssetItem(asset_id="ref_primary", role="identity", path="REF_IMAGE_PLACEHOLDER", locked=True)]
        if requires_ref
        else []
    )
    spec = StructuredPromptSpec(
        original_prompt=prompt,
        intent=GenerationIntent(
            task="image_generate",
            input_mode="automatic",
            goal=title,
            preserve_identity=requires_ref,
        ),
        reference_assets=refs,
        required_attributes=[
            RequirementItem(id=rid, category=category, description=desc) for rid, desc in reqs
        ],
        excluded_attributes=list(excluded or ["watermark", "extra fingers", "distorted face"]),
        constraints=GenerationConstraints(width=896, height=1280, frames=1, seed=seed, model_preference="flux2_9b"),
    )
    return AdherenceBenchmarkCase(
        case_id=case_id,
        category=category,
        target_model="flux2_9b",
        title=title,
        spec=spec,
        requires_reference_image=requires_ref,
    )


def _vid_case(
    case_id: str,
    category: AdherenceCategory,
    title: str,
    prompt: str,
    motion: str,
    camera_move: str,
    reqs: list[tuple[str, str]],
    *,
    seed: int = 61001,
) -> AdherenceBenchmarkCase:
    spec = StructuredPromptSpec(
        original_prompt=prompt,
        intent=GenerationIntent(
            task="video_generate",
            input_mode="automatic",
            goal=title,
            preserve_identity=True,
            preserve_scene=True,
        ),
        motion=PromptAttribute(value=motion, state="specified", provenance="user_explicit"),
        camera_movement=PromptAttribute(value=camera_move, state="specified", provenance="user_explicit"),
        reference_assets=[
            ReferenceAssetItem(asset_id="start_frame", role="start_frame", path="REF_IMAGE_PLACEHOLDER", locked=True)
        ],
        required_attributes=[
            RequirementItem(id=rid, category=category, description=desc) for rid, desc in reqs
        ],
        excluded_attributes=["blurry", "morphing", "flicker", "extra fingers", "frozen still image"],
        constraints=GenerationConstraints(
            width=480, height=832, fps=16, duration_s=5.0, frames=81, seed=seed, model_preference="wan22_14b"
        ),
    )
    return AdherenceBenchmarkCase(
        case_id=case_id,
        category=category,
        target_model="wan22_14b",
        title=title,
        spec=spec,
        requires_reference_image=True,
    )


def default_adherence_suite() -> list[AdherenceBenchmarkCase]:
    """Return the complete 13-category controlled benchmark suite (10 FLUX.2 9B + 3 Wan 2.2 14B)."""
    return [
        _img_case(
            "adh_01_photorealism",
            "photorealism",
            "Natural-light studio portrait skin & lens realism",
            "Candid natural-light portrait of a 34-year-old woman instructor near a north-facing window, visible natural skin pores, fine strands of brown hair, shot on 85mm lens, shallow depth of field.",
            [
                ("natural_skin", "Natural skin texture without plastic/airbrushed smoothing"),
                ("adult_subject", "Clearly adult woman (~30s)"),
                ("window_light", "Directional natural window light with soft shadows"),
            ],
            seed=51001,
        ),
        _img_case(
            "adh_02_composition",
            "composition",
            "Rule-of-thirds off-center full-body framing",
            "Full-body photograph of an adult man standing on the right third of a quiet brick courtyard, head-to-toe fully inside the frame with open negative space on the left.",
            [
                ("full_body", "Entire body from head to shoes visible without cropping"),
                ("right_third", "Subject placed on the right third with open space on the left"),
            ],
            seed=51002,
        ),
        _img_case(
            "adh_03_quantities",
            "object_quantities",
            "Exact countable tabletop objects",
            "Overhead natural-light photograph of a clean oak desk holding exactly three ceramic coffee mugs and exactly two closed notebooks.",
            [
                ("three_mugs", "Exactly 3 ceramic coffee mugs visible"),
                ("two_notebooks", "Exactly 2 closed notebooks visible"),
            ],
            seed=51003,
        ),
        _img_case(
            "adh_04_spatial",
            "spatial_relationships",
            "Explicit left/right and foreground/background placement",
            "Interior studio shot: a green glass vase sits to the left of a brass lamp on a wooden table, while a tall potted fern stands behind the table near a white curtain.",
            [
                ("vase_left_of_lamp", "Green glass vase is positioned to the left of the brass lamp"),
                ("fern_behind_table", "Potted fern stands behind the table near a white curtain"),
            ],
            seed=51004,
        ),
        _img_case(
            "adh_05_colors",
            "colors",
            "Distinct color binding without color bleed",
            "Editorial portrait of an adult woman wearing a cobalt blue tailored blazer over a crisp white shirt, holding a bright yellow leather notebook against a warm charcoal backdrop.",
            [
                ("cobalt_blazer", "Blazer is cobalt blue"),
                ("white_shirt", "Shirt underneath is white without blue bleed"),
                ("yellow_notebook", "Notebook is bright yellow"),
            ],
            seed=51005,
        ),
        _img_case(
            "adh_06_clothing",
            "clothing_appearance",
            "Specific wardrobe, hairstyle, and accessory fidelity",
            "Medium portrait of an adult woman with shoulder-length auburn hair in a neat half-up style, wearing tortoiseshell glasses, a dark emerald knit turtleneck, and small gold hoop earrings.",
            [
                ("auburn_half_up", "Shoulder-length auburn hair in half-up style"),
                ("tortoiseshell_glasses", "Wearing tortoiseshell glasses"),
                ("emerald_turtleneck", "Dark emerald knit turtleneck"),
                ("gold_hoops", "Small gold hoop earrings"),
            ],
            seed=51006,
        ),
        _img_case(
            "adh_07_typography",
            "typography",
            "Legible exact sign typography",
            "Clean daytime photograph of a modern studio door with a matte black plaque reading 'STUDIO 04' in crisp white uppercase sans-serif letters.",
            [
                ("exact_text", "Sign reads exactly 'STUDIO 04' with zero misspellings or garbled glyphs"),
                ("legible_contrast", "White uppercase text clearly readable on matte black plaque"),
            ],
            excluded=["gibberish text", "misspelled letters", "watermark"],
            seed=51007,
        ),
        _img_case(
            "adh_08_character_consistency",
            "character_consistency",
            "Consistent persona across new angle and lighting",
            "Three-quarter candid portrait of the same adult woman presenter from the reference photo, smiling warmly in a sunlit cafe wearing a tan trench coat.",
            [
                ("face_match", "Preserves facial geometry, hairline, and eye shape of reference persona"),
                ("new_context", "Placed in a sunlit cafe wearing a tan trench coat"),
            ],
            requires_ref=True,
            seed=51008,
        ),
        _img_case(
            "adh_09_reference_adherence",
            "reference_adherence",
            "Reference outfit and identity preservation",
            "Medium shot of the exact adult presenter from the reference image standing beside a whiteboard in a bright office, retaining her exact facial features and hairstyle.",
            [
                ("identity_preserved", "Matches reference identity accurately"),
                ("whiteboard_office", "Bright office background with a whiteboard beside her"),
            ],
            requires_ref=True,
            seed=51009,
        ),
        _img_case(
            "adh_10_complex_nl",
            "complex_natural_language",
            "Multi-clause natural-language scene with exclusions",
            "An adult architect in her late 30s leans over a drafting table covered with rolled blueprints in a brick loft at dusk; warm tungsten desk lamp illuminates her hands holding a silver mechanical pencil while cool blue twilight enters through tall arched windows behind her; no laptop or phone on the desk.",
            [
                ("architect_pose", "Adult woman leaning over drafting table with rolled blueprints"),
                ("silver_pencil", "Holding a silver mechanical pencil"),
                ("dual_lighting", "Warm desk lamp foreground + cool blue twilight through arched windows"),
                ("no_electronics", "No laptop or phone on the desk"),
            ],
            excluded=["laptop", "smartphone", "phone", "computer monitor"],
            seed=51010,
        ),
        _vid_case(
            "adh_11_video_motion",
            "video_motion",
            "Controlled subject gesture and expression transition",
            "The presenter looks into the camera, smiles warmly, raises her right hand to wave hello, then lowers her hand smoothly.",
            motion="Smooth right-hand wave followed by natural settle; stable facial identity",
            camera_move="Locked tripod medium shot, zero unintended camera drift",
            reqs=[
                ("right_hand_wave", "Subject clearly raises right hand to wave and lowers it smoothly"),
                ("five_fingers", "Hand maintains natural anatomy without extra or melted fingers"),
            ],
            seed=61001,
        ),
        _vid_case(
            "adh_12_camera_movement",
            "camera_movement",
            "Slow deliberate camera push-in",
            "The presenter sits calmly at a desk facing the lens, maintaining steady eye contact and subtle breathing as the camera executes a smooth slow push-in toward a closer portrait.",
            motion="Minimal natural breathing and slight warm nod",
            camera_move="Smooth slow forward push-in over 5 seconds",
            reqs=[
                ("forward_push_in", "Camera visibly and smoothly pushes closer to the subject"),
                ("stable_geometry", "Background lines and desk geometry remain coherent without warping"),
            ],
            seed=61002,
        ),
        _vid_case(
            "adh_13_temporal_consistency",
            "temporal_consistency",
            "Identity, wardrobe, and background stability over 81 frames",
            "The presenter turns her head slightly left toward a notepad, glances down briefly, then looks back into the camera with a gentle smile while keeping the same clothing, hairstyle, and studio lighting throughout.",
            motion="Head turn left, brief glance down, return to camera eye contact",
            camera_move="Static locked-off studio camera",
            reqs=[
                ("identity_stable", "Face and hair remain consistent across all 81 frames"),
                ("wardrobe_stable", "Clothing color, collar, and texture do not morph or flicker"),
                ("lighting_stable", "Background and lighting stay flicker-free"),
            ],
            seed=61003,
        ),
    ]
