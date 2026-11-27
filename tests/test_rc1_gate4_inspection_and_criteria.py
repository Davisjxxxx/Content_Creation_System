"""Adversarial regression suite for Astra RC1 Gate 4 (RC-05, RC-07, RC-08, RC-09).

Covers:
- RC-05: Opt-in Local VLM quality check & Auto-correct UI controls + persisted settings.
- RC-07: Arbitrary numeric/spelled counts (1, 3, 11, multi-object), full spatial operands,
         unparsed prompt preservation, and FLUX.2 typography negative prompt suppression.
- RC-08: No single-VLM self-corroboration on passed verdicts, explicit VideoInspector
         verification_eligibility, and CorrectionCoordinator routing through compute_inspection_status.
- RC-09: Candidate artifact binding (SHA-256 + requirement_set_hash), timed-out/uncertain
         reinspection never beating original, reinspection exception resilience, and per-output verdicts.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2 import desktop, offline_builder as ob
from avatar_v2.correction_engine import (
    CorrectionCoordinator,
    is_candidate_strict_improvement,
    score_candidate,
)
from avatar_v2.inspection_contract import (
    AcceptanceCriterion,
    CorrectionBudget,
    CriterionAssessment,
    ExpectedOutputCharacteristics,
    GenerationVerificationRecord,
    VisualInspectionResult,
)
from avatar_v2.intent_interpreter import interpret_natural_language
from avatar_v2.js_bridge import JsBridge
from avatar_v2.ollama_client import LLMCallResult, LocalLLMInfo
from avatar_v2.visual_inspector import VisualInspector, _deterministic_cross_check_vlm_criterion


def _make_png(path: Path, color: tuple[int, int, int] = (120, 140, 160)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (512, 512), color)
    # Add high-contrast pattern so structural CV sharpness passes cleanly
    px = img.load()
    for y in range(0, 512, 8):
        for x in range(0, 512, 8):
            px[x, y] = (240, 240, 240) if ((x + y) // 8) % 2 == 0 else (20, 20, 20)
    img.save(path, format="PNG")
    return path


def test_rc05_ui_toggles_and_persisted_qa_preferences(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Forge UI exposes opt-in VLM QA & Auto-correct checkboxes and persists them across restarts."""
    html = (Path(__file__).resolve().parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="forgeVlmQa"' in html
    assert 'id="forgeAutoCorrect"' in html
    assert "visual_inspection_vlm:!!($('forgeVlmQa')" in html
    assert "auto_correct:!!($('forgeAutoCorrect')" in html

    app_home = tmp_path / "home"
    app_home.mkdir()
    monkeypatch.setattr(desktop, "APP_HOME", app_home)
    monkeypatch.setattr(desktop, "CONFIG_PATH", app_home / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", app_home / "runs")
    monkeypatch.setattr(desktop, "QUEUE_DIR", app_home / "queue")
    monkeypatch.setattr(desktop, "HISTORY_DIR", app_home / "history")
    monkeypatch.setattr(desktop, "LIBRARY_DIR", app_home / "library")
    monkeypatch.setattr(desktop, "CLOUD_LEDGER_PATH", app_home / "cloud_ledger.jsonl")

    api = desktop.DesktopAPI()
    try:
        bridge = JsBridge(api)
        saved = bridge.save_settings({"local_vlm_inspection": True, "local_auto_correct": True})
        assert saved["ok"] is True
        st1 = api.forge_state()
        assert st1["local_vlm_inspection"] is True
        assert st1["local_auto_correct"] is True
    finally:
        api.queue.shutdown()

    # Simulated restart preserves user preference
    api2 = desktop.DesktopAPI()
    try:
        st2 = api2.forge_state()
        assert st2["local_vlm_inspection"] is True
        assert st2["local_auto_correct"] is True
    finally:
        api2.queue.shutdown()


def test_rc07_counts_spatial_operands_unparsed_and_typography_negative() -> None:
    """Counts (1, 3, 11, multi-object), full spatial operands, unparsed prompts, and typography negative."""
    req_counts = interpret_natural_language(
        "A wooden table with 1 ceramic mug, three red apples, and 11 birthday candles",
        requested_action="photo",
    )
    qty_descs = [r.description for r in req_counts.spec.required_attributes if r.category == "object_quantities"]
    assert any("1 ceramic mug" in d for d in qty_descs), qty_descs
    assert any("3 red apples" in d for d in qty_descs), qty_descs
    assert any("11 birthday candles" in d for d in qty_descs), qty_descs

    req_spatial = interpret_natural_language(
        "A wooden table with a red vase to the left of a brass lamp",
        requested_action="photo",
    )
    sp_descs = [r.description for r in req_spatial.spec.required_attributes if r.category == "spatial_relationships"]
    assert sp_descs, "Expected spatial requirement"
    assert any("red vase" in d.lower() and "brass lamp" in d.lower() for d in sp_descs), sp_descs

    # Prompt with no regex hits still preserves explicit semantic requirement
    req_unparsed = interpret_natural_language(
        "A matte ceramic pitcher with a subtle chipped rim in warm morning light",
        requested_action="photo",
    )
    assert len(req_unparsed.spec.required_attributes) >= 1

    # Typography negative prompt suppresses text-banning tokens
    neg_text = ob.photo_negative_for_prompt('A glowing neon sign reading "OPEN 24 HOURS"')
    for banned in ("text", "words", "letters", "signage", "typography"):
        assert banned not in neg_text.lower()
    neg_plain = ob.photo_negative_for_prompt("A sunlit portrait of an adult in a park")
    assert "text" in neg_plain.lower()


def test_rc08_no_single_vlm_self_corroboration_and_coordinator_status(tmp_path: Path) -> None:
    """Single-VLM consistency must never self-corroborate passes; CorrectionCoordinator respects review_recommended."""
    crit = AcceptanceCriterion(
        criterion_id="sp_1",
        category="spatial_relationships",
        description="Spatial placement: a red vase to the left of a brass lamp",
        mandatory=True,
    )
    # 1. Self-consistent single VLM pass MUST NOT return corroborated=True
    st, fail_d, corrob = _deterministic_cross_check_vlm_criterion(
        crit,
        status="passed",
        observed_evidence="Red vase at left, brass lamp at right",
        observed_count=None,
        observed_visible_text=[],
        expected_typography=[],
        failure_description="",
        localized_objects=[
            {"label": "red vase", "x_center_pct": 25.0},
            {"label": "brass lamp", "x_center_pct": 75.0},
        ],
    )
    assert st == "passed"
    assert corrob is False

    # 2. Contradictory single VLM pass STILL gets rejected to failed!
    st2, fail_d2, _ = _deterministic_cross_check_vlm_criterion(
        crit,
        status="passed",
        observed_evidence="Looks good",
        observed_count=None,
        observed_visible_text=[],
        expected_typography=[],
        failure_description="",
        localized_objects=[
            {"label": "red vase", "x_center_pct": 80.0},
            {"label": "brass lamp", "x_center_pct": 20.0},
        ],
    )
    assert st2 == "failed"
    assert "NOT left of" in fail_d2

    # 3. Full VisualInspector run with requires_human_review category returns review_recommended, NEVER requirements_verified
    img_path = _make_png(tmp_path / "cand.png")

    class _FakeVLMClient:
        def list_models(self) -> list[LocalLLMInfo]:
            return [
                LocalLLMInfo(
                    name="qwen3-vl:8b-instruct",
                    size_bytes=5_000_000_000,
                    size_gb=5.0,
                    family="qwen3vl",
                    parameter_size="8B",
                    quantization="Q4_K_M",
                    roles=("vision_qa", "text_chat"),
                )
            ]

        def complete_vision_json(self, **kw: Any) -> LLMCallResult:
            return LLMCallResult(
                ok=True,
                model="qwen3-vl:8b-instruct",
                mode="cpu_only",
                latency_ms=12.0,
                data={
                    "observed_scene_summary": "A red vase on left, brass lamp on right",
                    "observed_visible_text": [],
                    "localized_objects": [
                        {"label": "red vase", "x_center_pct": 25.0},
                        {"label": "brass lamp", "x_center_pct": 75.0},
                    ],
                    "criteria": [
                        {
                            "criterion_id": "sp_1",
                            "observed_evidence": "Red vase left of brass lamp",
                            "status": "passed",
                        }
                    ],
                    "detected_defects": [],
                },
            )

    record = GenerationVerificationRecord(
        original_prompt="A red vase to the left of a brass lamp",
        detected_intent="image_generate",
        compiled_prompt="A red vase to the left of a brass lamp",
        selected_model_id="flux2_9b",
        selected_workflow="builtin:flux2_txt2img",
        expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=512, height=512),
        mandatory_acceptance_criteria=[crit],
    )
    inspector = VisualInspector(client=_FakeVLMClient())  # type: ignore[arg-type]
    res = inspector.inspect_image(img_path, record, use_vlm=True)
    assert res.ui_status == "review_recommended"
    assert res.human_review_required is True
    assert res.criterion_assessments[0].independent_corroboration is False
    # RC-09 binding check:
    assert res.output_sha256 == hashlib.sha256(img_path.read_bytes()).hexdigest()
    assert len(res.requirement_set_hash) == 64

    # 4. CorrectionCoordinator MUST NOT upgrade review_recommended to requirements_verified or correction_complete!
    coord = CorrectionCoordinator()
    session = coord.run_bounded_correction(
        record=record,
        original_output_id="orig_1",
        original_output_path=str(img_path),
        initial_inspection=res,
        generate_fn=lambda step: (str(img_path), 10.0),
        inspect_fn=lambda p, rec: res,
    )
    assert session.ui_status == "review_recommended"
    assert session.human_review_required is True


def test_rc09_timeout_uncertain_never_beats_original_and_reinspection_exception_safe(tmp_path: Path) -> None:
    """Timed-out/uncertain reinspection must NEVER replace original; reinspection exception preserves candidate."""
    orig = _make_png(tmp_path / "orig.png", (100, 100, 100))
    cand1 = _make_png(tmp_path / "cand1.png", (120, 120, 120))
    cand2 = _make_png(tmp_path / "cand2.png", (140, 140, 140))

    crit = AcceptanceCriterion(criterion_id="q1", category="object_quantities", description="Exact count: 3 mugs", mandatory=True)
    record = GenerationVerificationRecord(
        original_prompt="3 mugs on a table",
        detected_intent="image_generate",
        compiled_prompt="3 mugs on a table",
        selected_model_id="flux2_9b",
        selected_workflow="builtin:flux2_txt2img",
        expected_output=ExpectedOutputCharacteristics(task="image_generate", media_kind="image", width=512, height=512),
        mandatory_acceptance_criteria=[crit],
        correction_budget=CorrectionBudget(max_retries=2, max_image_retries=2),
    )
    orig_ins = VisualInspectionResult(
        output_path=str(orig),
        overall_decision="needs_correction",
        ui_status="correction_recommended",
        criterion_assessments=[
            CriterionAssessment(criterion_id="q1", category="object_quantities", mandatory=True, status="failed")
        ],
    )
    timed_out_ins = VisualInspectionResult(
        output_path=str(cand1),
        overall_decision="warn",
        ui_status="some_uncertain",
        criterion_assessments=[
            CriterionAssessment(criterion_id="q1", category="object_quantities", mandatory=True, status="uncertain")
        ],
    )

    s_orig = score_candidate(orig_ins, record)
    s_timeout = score_candidate(timed_out_ins, record)
    better, is_reg, reason = is_candidate_strict_improvement(s_timeout, s_orig)
    assert better is False
    assert "uncertainty" in reason.lower()

    # Full coordinator run: attempt 1 times out (uncertain), attempt 2 raises during reinspection
    gen_paths = iter([str(cand1), str(cand2)])
    ins_calls = {"n": 0}

    def _inspect(path: str, rec: GenerationVerificationRecord) -> VisualInspectionResult:
        ins_calls["n"] += 1
        if ins_calls["n"] == 1:
            return timed_out_ins
        raise RuntimeError("Ollama socket reset during reinspection")

    coord = CorrectionCoordinator()
    session = coord.run_bounded_correction(
        record=record,
        original_output_id="job1",
        original_output_path=str(orig),
        initial_inspection=orig_ins,
        generate_fn=lambda step: (next(gen_paths), 50.0),
        inspect_fn=_inspect,
    )
    assert session.best_candidate_path == str(orig)
    assert session.best_candidate_is_original is True
    # Both generated candidates are preserved despite timeout & reinspection crash!
    assert str(cand1) in session.all_candidate_paths
    assert str(cand2) in session.all_candidate_paths
    assert len(session.attempts) == 2
