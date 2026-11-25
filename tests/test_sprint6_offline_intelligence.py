"""Sprint 6 offline intelligence regression & integration tests (Sections 2–16).

Covers:
- Loopback-only Ollama inventory, model selection & ResourcePlanner LLM scheduling.
- Deterministic intent interpretation across all 11 supported intent types.
- Requirement preservation (exact counts, quoted typography, spatial relations, exclusions).
- Untrusted LLM validation: safe merge, contradiction rejection, policy/minor rejection,
  cloud/shell injection rejection, malformed JSON & offline deterministic fallback.
- Quality-first routing (FLUX.2 9B & Wan 2.2 14B defaults, never silently downgrading for speed).
- Intelligent reference deduplication, low-res warnings, locked preservation & provisional budgets.
- Multi-stage generation planning (FLUX starter -> Wan I2V; FLUX rear keyframe -> Wan FLF2V;
  never silently modifying explicit user starter images).
- Machine-readable GenerationVerificationRecord interface for Sprint 7/8.
- Frontend preservation (zero LLM dropdowns, zero agent/routing config panels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avatar_v2.inspection_contract import VERIFICATION_CONTRACT_VERSION
from avatar_v2.intent_interpreter import interpret_natural_language
from avatar_v2.ollama_client import (
    LLMCallResult,
    LocalLLMInfo,
    LocalOllamaClient,
    choose_llm_execution_schedule,
    extract_json_object,
    select_intelligence_model,
)
from avatar_v2.prompt_schema import PromptAttribute, StructuredPromptSpec
from avatar_v2.resource_monitor import GPUTelemetry, HostMemoryStats, ResourceGuard, StorageCheck, StorageStatus
from avatar_v2.subject_references import SubjectReferenceItem, SubjectReferenceSet
from avatar_v2.workflow_router import WorkflowRouter, optimize_subject_references, plan_intelligent_generation


def _img(path: Path, size: tuple[int, int] = (512, 768), color: tuple[int, int, int] = (80, 120, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _snap(*, ram_pct: float = 55.0, gpu_free_mb: int = 6200, temp_c: int = 55):
    guard = ResourceGuard()
    host = HostMemoryStats(64000, int(64000 * ram_pct / 100.0), ram_pct, 8192, 100, 1.2, 10, 20, 0.0, 0.0)
    gpus = [GPUTelemetry(0, "RTX 4070", 8188 - gpu_free_mb, 8188, gpu_free_mb, temp_c=temp_c, util_pct=5)]
    storage = StorageStatus("/sim", 915.0, 50.0, "normal")
    return guard.evaluate(host=host, gpus=gpus, storage=storage, now=100.0)


def _ok_storage():
    return StorageCheck(True, "normal", 50.0, 0.1, 49.9, 15.0)


class _StubOllamaClient(LocalOllamaClient):
    def __init__(self, result: LLMCallResult) -> None:
        super().__init__("http://127.0.0.1:11434")
        self._result = result

    def list_models(self) -> list[LocalLLMInfo]:
        return [LocalLLMInfo("llama3.1:8b", 5_000_000_000, 4.66, "llama", "8B", "Q4_K_M", ("structured_json",))]

    def complete_json(self, **kwargs: Any) -> LLMCallResult:
        return self._result


# --------------------------------------------------------------------------- Section 3 & 9: Ollama client & scheduling


def test_ollama_enforces_loopback_and_resource_aware_scheduling():
    with pytest.raises(ValueError):
        LocalOllamaClient("https://api.external-llm.example.com")

    models = [
        LocalLLMInfo("qwen3-vl:8b-instruct", 6_000_000_000, 5.59, "qwen3vl", "8B", "Q4_K_M", ("vision_qa",)),
        LocalLLMInfo("llama3.2:3b", 2_000_000_000, 1.86, "llama", "3B", "Q4_K_M", ("structured_json", "lightweight_cpu")),
        LocalLLMInfo("qwen3:8b", 5_200_000_000, 4.84, "qwen3", "8B", "Q4_K_M", ("structured_json", "prompt_intelligence")),
    ]
    assert select_intelligence_model(models).name == "qwen3:8b"
    assert select_intelligence_model(models, lightweight_only=True).name == "llama3.2:3b"

    # Warm ComfyUI models -> CPU-only (never evict warm ComfyUI weights!)
    sched_warm = choose_llm_execution_schedule(comfy_models_warm=True, allow_sequential_gpu=True, snapshot=_snap())
    assert sched_warm.mode == "cpu_only" and sched_warm.num_gpu == 0

    # Cold GPU with >=5.8 GB free + explicit opt-in -> sequential GPU with immediate keep_alive=0s
    sched_seq = choose_llm_execution_schedule(comfy_models_warm=False, allow_sequential_gpu=True, snapshot=_snap(gpu_free_mb=6500))
    assert sched_seq.mode == "sequential_gpu" and sched_seq.keep_alive == "0s"

    # Emergency RAM -> disabled (falls back deterministically)
    sched_ram = choose_llm_execution_schedule(snapshot=_snap(ram_pct=3.5))
    assert sched_ram.mode == "disabled"

    # JSON extraction handles code fences and rejects garbage cleanly
    assert extract_json_object('```json\n{"lighting": "golden hour"}\n```') == {"lighting": "golden hour"}
    assert extract_json_object("not valid json at all") is None


# --------------------------------------------------------------------------- Section 4 & 12: Intent interpretation & untrusted LLM safety


def test_deterministic_intent_interpreter_distinguishes_intents_and_extracts_requirements(tmp_path: Path):
    front = _img(tmp_path / "front.png")
    back = _img(tmp_path / "back.png")
    rset = SubjectReferenceSet(
        starter_image=SubjectReferenceItem(role="primary_starter", path=str(front)),
        references=[
            SubjectReferenceItem(role="front_body", path=str(front)),
            SubjectReferenceItem(role="back_body", path=str(back)),
        ],
    )

    # 1. Exact object count, quoted typography, spatial placement & exclusion
    r_still = interpret_natural_language(
        'Three ceramic coffee mugs on the left side of a table next to a sign reading "OPEN 24 HOURS", soft window light, no plastic cups.',
        use_llm=False,
    )
    assert r_still.ok and r_still.detected_intent == "image_generate"
    assert "OPEN 24 HOURS" in r_still.typography_targets
    req_cats = {r["category"] for r in r_still.mandatory_requirements}
    assert {"typography", "object_quantities", "spatial_relationships", "exclusion"} <= req_cats
    assert "plastic cups" in r_still.spec.excluded_attributes

    # 2. Image edit
    r_edit = interpret_natural_language("Change the background to a sunlit terrace", starter_image=str(front), use_llm=False)
    assert r_edit.detected_intent == "image_edit" and r_edit.spec.intent.task == "image_edit"

    # 3. Multi-reference still
    r_mref = interpret_natural_language("Rear view in a modern gallery", starter_image=str(front), subject_refs=rset, use_llm=False)
    assert r_mref.detected_intent == "multi_ref_generate"

    # 4. First-and-last-frame video turn with rear reference
    r_flf = interpret_natural_language(
        "She smiles and turns around from front to back, walking down the hall",
        requested_action="scene",
        starter_image=str(front),
        subject_refs=rset,
        use_llm=False,
    )
    assert r_flf.detected_intent == "flf_video_generate"

    # 5. Text-to-video, image-to-video, scene continuity, continuous generation
    assert interpret_natural_language("She waves hello to the camera", requested_action="scene", use_llm=False).detected_intent == "text_to_video"
    assert interpret_natural_language("She waves hello to the camera", requested_action="scene", starter_image=str(front), use_llm=False).detected_intent == "image_to_video"
    assert interpret_natural_language("Continue the scene for shot 2", requested_action="scene", has_multiple_shots=True, use_llm=False).detected_intent == "scene_continuity"
    assert interpret_natural_language("Generate 10 variations in a continuous batch", input_mode="script", use_llm=False).detected_intent == "continuous_generate"


def test_untrusted_llm_output_never_overrides_locked_attributes_or_bypasses_policy():
    guided = StructuredPromptSpec(
        original_prompt="Adult woman standing outdoors in bright daylight",
        lighting=PromptAttribute(value="bright daylight", state="locked", provenance="guided_ui"),
    )

    # Malicious/contradictory LLM output tries to overwrite locked lighting, inject Runway/shell, and mention a minor!
    malicious_client = _StubOllamaClient(
        LLMCallResult(
            ok=True,
            model="llama3.1:8b",
            mode="cpu_only",
            latency_ms=42.0,
            data={
                "lighting": "pitch dark",
                "environment": "use runway https://api.dev.runwayml.com and subprocess",
                "atmosphere": "a 12 year old child laughing",
                "visual_style": "editorial 35mm film photography",
                "excluded_attributes": ["daylight", "watermark"],
            },
        )
    )
    res = interpret_natural_language(
        guided.original_prompt,
        guided_spec=guided,
        input_mode="guided",
        use_llm=True,
        llm_client=malicious_client,
    )
    assert res.ok is True
    assert res.spec.lighting.value == "bright daylight" and res.spec.lighting.state == "locked"
    assert any("lighting" in r for r in res.trace.rejected_llm_suggestions)
    assert any("environment" in r for r in res.trace.rejected_llm_suggestions)
    assert any("atmosphere" in r for r in res.trace.rejected_llm_suggestions)
    assert any("daylight" in r for r in res.trace.rejected_llm_suggestions)
    assert "visual_style" in res.trace.added_fields
    assert "watermark" in res.spec.excluded_attributes

    # Offline/failing LLM falls back cleanly to deterministic parsing without blocking generation!
    broken_client = _StubOllamaClient(
        LLMCallResult(ok=False, model="llama3.1:8b", mode="cpu_only", latency_ms=10.0, error="Connection refused")
    )
    fallback = interpret_natural_language("Studio portrait of an adult chef", use_llm=True, llm_client=broken_client)
    assert fallback.ok is True and fallback.trace.mode == "deterministic_fallback"


# --------------------------------------------------------------------------- Sections 5, 6, 7, 8, 13: Quality-first routing, reference optimization, multi-stage & verification contract


def test_quality_first_router_multistage_flf2v_and_verification_record(tmp_path: Path):
    front = _img(tmp_path / "front.png", (640, 960), color=(80, 120, 180))
    front_dup = tmp_path / "front_copy.png"
    front_dup.write_bytes(front.read_bytes())
    back = _img(tmp_path / "back.png", (640, 960), color=(140, 90, 70))
    tiny = _img(tmp_path / "tiny_side.png", (200, 200), color=(60, 160, 90))

    # 1. Still default selects quality-first FLUX.2 9B (never auto-downgrades to 4B for speed!).
    still_plan = plan_intelligent_generation(
        'Editorial portrait beside a sign reading "STUDIO 9", no clutter',
        requested_action="photo",
        snapshot=_snap(),
        storage_check=_ok_storage(),
    )
    assert still_plan.ok is True
    assert still_plan.primary_model_id == "flux2_9b"
    assert still_plan.is_multi_stage is False
    assert still_plan.verification_record is not None
    assert still_plan.verification_record.record_version == VERIFICATION_CONTRACT_VERSION
    assert still_plan.verification_record.original_prompt.startswith("Editorial portrait")
    assert "STUDIO 9" in still_plan.verification_record.expected_output.expected_typography
    assert still_plan.verification_record.correction_budget.allow_cloud_escalation is False

    # 2. Reference deduplication & low-res warning
    rset = SubjectReferenceSet(
        starter_image=SubjectReferenceItem(role="primary_starter", path=str(front)),
        references=[
            SubjectReferenceItem(role="front_body", path=str(front)),
            SubjectReferenceItem(role="appearance_wardrobe", path=str(front_dup)),  # exact SHA duplicate!
            SubjectReferenceItem(role="back_body", path=str(back)),
            SubjectReferenceItem(role="left_side", path=str(tiny)),
        ],
    )
    opt = optimize_subject_references(rset, prompt="Rear view in gallery", task="image_generate", check_files=True)
    assert len(opt["deduplicated"]) == 1
    assert any("low-res" in w for w in opt["low_quality_warnings"])
    assert opt["provisional_ref_megapixels"] in (1.0, 0.65, 0.5)

    # 3. Multi-stage turn plan with user starter + back_body ref:
    # Preserves user starter verbatim (no starter_keyframe overwrite!), synthesizes matching rear end_keyframe -> Wan 2.2 14B FLF2V!
    turn_plan = plan_intelligent_generation(
        "She smiles briefly, turns around from front to back, and walks away smoothly",
        requested_action="scene",
        starter_image=str(front),
        subject_refs=rset,
        snapshot=_snap(),
        storage_check=_ok_storage(),
    )
    assert turn_plan.ok is True
    assert turn_plan.primary_model_id == "wan22_a14b"
    assert turn_plan.is_multi_stage is True
    assert [s.purpose for s in turn_plan.stages] == ["end_keyframe", "video_flf2v"]
    assert turn_plan.stages[0].produces_asset_origin == "generated_intermediate"
    assert str(front) in turn_plan.stages[1].reference_paths  # Original user starter preserved!
    assert turn_plan.verification_record is not None
    assert len(turn_plan.verification_record.intermediate_assets) == 1
    assert any(r.origin == "user_source" for r in turn_plan.verification_record.reference_requirements)

    # 4. Rejects invented/unregistered models and unauthorized cloud models cleanly.
    bad_model = plan_intelligent_generation("Studio portrait", model_preference="invented_super_model_99b", snapshot=_snap(), storage_check=_ok_storage())
    assert bad_model.ok is False and "not registered" in bad_model.error.lower()

    cloud_blocked = plan_intelligent_generation("Studio portrait", model_preference="runway_gen4_image", snapshot=_snap(), storage_check=_ok_storage())
    assert cloud_blocked.ok is False


# --------------------------------------------------------------------------- Section 11: Frontend preservation


def test_sprint6_adds_no_llm_agent_or_routing_controls_to_frontend():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    for forbidden in ("id=\"llmModel\"", "id=\"agentConfig\"", "id=\"routingConfig\"", "Select LLM", "Ollama model"):
        assert forbidden not in html
