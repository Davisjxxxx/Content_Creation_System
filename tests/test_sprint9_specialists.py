"""Sprint 9 Regression Tests — Specialist Registry, Acquisition Gate, Blind-Grid Localization, Masked Editing & Wan Base Comparison.

Clearly distinguishes deterministic/simulated tests (this file) from live GPU ComfyUI + Qwen3-VL
benchmarks (executed via ``scripts/eval_sprint9_specialists.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from avatar_v2.correction_engine import SUPPORTED_CORRECTION_STRATEGIES, validate_correction_strategy
from avatar_v2.inspection_contract import AcceptanceCriterion
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceSnapshot, StorageStatus
from avatar_v2.resource_planner import ResourcePlanner
from avatar_v2.specialist_registry import DEFAULT_SPECIALIST_REGISTRY
from avatar_v2.specialist_vision import (
    generate_region_mask,
    measure_outside_mask_preservation,
    reconcile_criterion_with_detections,
)
from avatar_v2.workflows_builder import build_flux2_masked_edit_workflow, without_wan_lightning


def _snap(*, gpu_free_mb: int = 7200, ram_sev: str = "normal") -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=1.0,
        host=HostMemoryStats(
            ram_total_mb=64000,
            ram_available_mb=32000 if ram_sev == "normal" else 1800,
            ram_available_pct=50.0 if ram_sev == "normal" else 2.8,
        ),
        gpus=[GPUTelemetry(index=0, name="RTX 4070", used_mb=8188 - gpu_free_mb, total_mb=8188, free_mb=gpu_free_mb, temp_c=52, util_pct=5)],
        storage=StorageStatus(path="/tmp", total_gb=1000.0, free_gb=43.0, tier="normal"),
        ram_severity=ram_sev,  # type: ignore[arg-type]
        swap_severity="normal",
        temp_severity="normal",
        storage_severity="normal",
        overall_severity=ram_sev,  # type: ignore[arg-type]
    )


def test_specialist_registry_compatibility_and_acquisition_gate() -> None:
    """Sections 4, 8, 11: Blocks incompatible SDXL ControlNets & unapproved downloads; allows zero-download built-ins."""
    reg = DEFAULT_SPECIALIST_REGISTRY

    # SDXL ControlNet MUST NEVER be allowed on FLUX.2 or Wan 2.2!
    ok_flux, reason_flux = reg.check_compatibility("sdxl_controlnet_pack", "flux2")
    assert ok_flux is False
    assert "incompatible" in reason_flux.lower()
    ok_wan, _ = reg.check_compatibility("sdxl_controlnet_pack", "wan22")
    assert ok_wan is False

    # Already-available built-in masked edit passes with 0 MB download!
    acq_local = reg.evaluate_acquisition("flux2_masked_latent_composite")
    assert acq_local["allowed"] is True
    assert acq_local["required_mb"] == 0

    # Large model (FLUX.2 Dev 32B) classified operationally_impractical & blocked even with approval!
    acq_impractical = reg.evaluate_acquisition("flux2_dev_32b_gguf_q4", explicit_approval=True)
    assert acq_impractical["allowed"] is False
    assert acq_impractical["classification"] == "operationally_impractical"

    # Lightweight specialist (Grounding DINO Tiny) requires explicit authorization!
    acq_unapproved = reg.evaluate_acquisition("grounding_dino_tiny", explicit_approval=False)
    assert acq_unapproved["allowed"] is False
    assert "authorization" in acq_unapproved["reason"].lower()


def test_blind_localization_reconciliation_count_state_and_2d_spatial() -> None:
    """Sections 5 & 13: Reconciles handled mugs vs bowls, open vs closed notebooks, 2D left/right, and occlusion."""
    # Case 1: adh_03_quantities (2 handled mugs + 2 handleless bowls; 1 open + 1 closed notebook)
    det_adh03 = [
        {"category": "coffee mug", "subtype_or_details": "white ceramic mug", "has_handle": True, "open_or_closed": "not_applicable", "x_center_pct": 49.0, "y_center_pct": 22.0, "bbox_norm": [40, 12, 60, 32], "occluded": False},
        {"category": "coffee mug", "subtype_or_details": "grey ceramic mug", "has_handle": True, "open_or_closed": "not_applicable", "x_center_pct": 84.0, "y_center_pct": 35.0, "bbox_norm": [72, 25, 96, 45], "occluded": False},
        {"category": "bowl", "subtype_or_details": "white ceramic bowl", "has_handle": False, "open_or_closed": "not_applicable", "x_center_pct": 20.0, "y_center_pct": 32.0, "bbox_norm": [8, 22, 33, 42], "occluded": False},
        {"category": "bowl", "subtype_or_details": "white ceramic bowl", "has_handle": False, "open_or_closed": "not_applicable", "x_center_pct": 63.0, "y_center_pct": 46.0, "bbox_norm": [50, 36, 77, 57], "occluded": False},
        {"category": "notebook", "subtype_or_details": "lined pages exposed", "has_handle": None, "open_or_closed": "open", "x_center_pct": 34.0, "y_center_pct": 55.0, "bbox_norm": [14, 38, 54, 72], "occluded": False},
        {"category": "notebook", "subtype_or_details": "brown hardcover", "has_handle": None, "open_or_closed": "closed", "x_center_pct": 57.0, "y_center_pct": 52.0, "bbox_norm": [38, 38, 77, 67], "occluded": False},
    ]
    c_mugs = AcceptanceCriterion(criterion_id="mugs", category="object_quantities", description="Exactly 3 ceramic coffee mugs with handles are visible")
    c_books = AcceptanceCriterion(criterion_id="books", category="object_quantities", description="Exactly 2 closed notebooks are visible (none open)")
    r_mugs = reconcile_criterion_with_detections(c_mugs, det_adh03)
    r_books = reconcile_criterion_with_detections(c_books, det_adh03)
    assert r_mugs["status"] == "failed" and r_mugs["matched_count"] == 2
    assert r_books["status"] == "failed" and "open" in r_books["evidence"].lower()

    # Case 2: adh_04_spatial (brass lamp at x=20% LEFT of green vase at x=35% -> contradicts vase_left_of_lamp!)
    det_adh04 = [
        {"category": "brass lamp", "subtype_or_details": "desk lamp", "x_center_pct": 20.0, "y_center_pct": 65.0, "bbox_norm": [2, 46, 43, 88], "occluded": False},
        {"category": "green glass vase", "subtype_or_details": "glass vase", "x_center_pct": 35.0, "y_center_pct": 75.0, "bbox_norm": [24, 67, 46, 85], "occluded": False},
    ]
    c_sp = AcceptanceCriterion(criterion_id="sp", category="spatial_relationships", description="Green glass vase is positioned to the viewer's left of the brass lamp")
    r_sp = reconcile_criterion_with_detections(c_sp, det_adh04)
    assert r_sp["status"] == "failed"
    assert r_sp["coordinate_type"] == "2d_image_coordinates"

    # Case 3: Occluded items stay uncertain rather than falsely failing or passing!
    det_occ = [{"category": "coffee mug", "has_handle": True, "open_or_closed": "not_applicable", "x_center_pct": 50.0, "y_center_pct": 50.0, "bbox_norm": [40, 40, 60, 60], "occluded": True}]
    r_occ = reconcile_criterion_with_detections(c_mugs, det_occ)
    assert r_occ["status"] == "uncertain"


def test_pillow_mask_generation_and_outside_mask_preservation_metric(tmp_path: Path) -> None:
    """Sections 6 & 7: Quantitatively detects 0.0% outside-mask drift vs unmasked background drift."""
    orig = tmp_path / "orig.png"
    masked_edit = tmp_path / "masked_edit.png"
    unmasked_drift = tmp_path / "unmasked_drift.png"
    mask_path = tmp_path / "mask.png"

    im = Image.new("RGB", (512, 512), (210, 200, 190))
    ImageDraw.Draw(im).rectangle((40, 40, 200, 200), fill=(60, 120, 70))
    im.save(orig)

    generate_region_mask(512, 512, boxes_norm=[(55, 55, 90, 90)], pad_pct=0.0, feather_px=0, out_path=mask_path)

    # Masked edit changes ONLY inside (55..90%, 55..90%) -> 0.0% outside drift!
    im_m = im.copy()
    ImageDraw.Draw(im_m).rectangle((290, 290, 450, 450), fill=(220, 40, 40))
    im_m.save(masked_edit)
    stats_m = measure_outside_mask_preservation(orig, masked_edit, mask_path)
    assert stats_m["strictly_preserved_outside_mask"] is True
    assert stats_m["outside_mask_pct_changed"] == 0.0
    assert stats_m["inside_mask_pct_changed"] >= 75.0

    # Unmasked edit alters background globally -> flagged as NOT strictly preserved!
    im_u = Image.new("RGB", (512, 512), (140, 150, 160))
    im_u.save(unmasked_drift)
    stats_u = measure_outside_mask_preservation(orig, unmasked_drift, mask_path)
    assert stats_u["strictly_preserved_outside_mask"] is False
    assert stats_u["outside_mask_pct_changed"] > 90.0


def test_masked_edit_workflow_and_wan_standard_workflow_builders() -> None:
    """Sections 7, 9, 12: ComfyUI masked-edit and Wan non-Lightning graphs wire verified core nodes."""
    g = build_flux2_masked_edit_workflow({
        "SOURCE_IMAGE": "src.png",
        "MASK_IMAGE": "mask.png",
        "PROMPT": "Replace bowl with handled mug",
        "WIDTH": 896,
        "HEIGHT": 1280,
        "STEPS": 28,
        "CFG": 3.5,
        "SEED": 41001,
    })
    classes = {node["class_type"] for node in g.values()}
    assert {"UnetLoaderGGUF", "DifferentialDiffusion", "ImageToMask", "ReferenceLatent", "SetLatentNoiseMask", "ImageCompositeMasked", "SaveImage"} <= classes
    assert validate_correction_strategy("masked_localized_edit", media_kind="image")[0] is True
    assert SUPPORTED_CORRECTION_STRATEGIES["masked_localized_edit"]["workflow"] == "builtin:flux2_masked_edit"

    # Wan 2.2 A14B non-Lightning conversion strips LoraLoaderModelOnly and sets CFG 3.5 + even step split.
    wan_wf = json.loads((Path(__file__).resolve().parents[1] / "workflows" / "wan22_i2v_14b_gguf_api.json").read_text(encoding="utf-8"))
    std_wf = without_wan_lightning(wan_wf, steps=12, cfg=3.5)
    std_classes = {n["class_type"] for n in std_wf.values()}
    assert "LoraLoaderModelOnly" not in std_classes
    samplers = [n["inputs"] for n in std_wf.values() if n["class_type"] == "KSamplerAdvanced"]
    assert len(samplers) == 2
    assert samplers[0]["cfg"] == 3.5 and samplers[0]["end_at_step"] == 6
    assert samplers[1]["start_at_step"] == 6 and samplers[1]["end_at_step"] == 12


def test_resource_planner_specialist_scheduling_and_frontend_preserved() -> None:
    """Sections 15 & 16: Warm ComfyUI defers GPU specialist; CPU mask runs immediately; frontend unchanged."""
    planner = ResourcePlanner()
    p_cpu = planner.plan_specialist_execution("pillow_bbox_mask_generator", snapshot=_snap())
    assert p_cpu["allowed"] is True and p_cpu["execution_mode"] == "cpu_only"

    p_warm = planner.plan_specialist_execution("qwen3_vl_8b_blind_grid", comfy_models_warm=True, snapshot=_snap())
    assert p_warm["allowed"] is True and p_warm["execution_mode"] == "deferred_batch_sequential_gpu"

    p_incompat = planner.plan_specialist_execution("sdxl_controlnet_pack", target_family="flux2", snapshot=_snap())
    assert p_incompat["allowed"] is False

    html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "specialist-config-panel" not in html
    assert "view-qa-dashboard" not in html
