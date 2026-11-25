"""Deterministic Model-Specific Prompt Compilers for Avatar V2 (Sprint 3, Section 5).

Translates canonical ``StructuredPromptSpec`` v1.0 objects into model-appropriate prompts
without requiring a local LLM and without applying legacy SDXL ``(tag:1.4)`` weighting or
quality-tag spam to FLUX.2 or Wan 2.2.

Guarantees:
- Original user prompt and explicit requirements are always preserved first.
- Unsupported mandatory capabilities are NEVER silently dropped; they are reported in
  ``CompiledPrompt.unsupported_capabilities`` and ``warnings``.
- Attribute provenance is preserved for auditing and future Sprint 6-8 refinement.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .model_registry import DEFAULT_REGISTRY, ModelRegistry, ModelSpec
from .prompt_schema import PromptConflict, StructuredPromptSpec

# Regex to detect legacy A1111/SDXL weight wrappers like "(photorealistic:1.4)".
_SDXL_WEIGHT_RE = re.compile(r"\(([^():]+):\s*\d+(?:\.\d+)?\)")


def strip_sdxl_weights(text: str) -> tuple[str, bool]:
    """Strip SDXL/A1111 emphasis syntax ``(phrase:1.4)`` into clean natural prose."""
    cleaned, count = _SDXL_WEIGHT_RE.subn(r"\1", text or "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, count > 0


@dataclass(frozen=True)
class CompiledPrompt:
    model_id: str
    model_family: str
    positive: str
    negative: str
    preserved_requirements: list[str]
    dropped_enhancements: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_capabilities: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(c.get("severity") == "error" for c in self.conflicts) and not self.unsupported_capabilities

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _dedupe_clauses(clauses: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in clauses:
        clean = " ".join(str(raw or "").split()).strip(" ,.")
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _subject_clauses(spec: StructuredPromptSpec) -> list[str]:
    clauses: list[str] = []
    for subj in spec.subjects:
        bits = [
            getattr(subj, name).value.strip()
            for name in ("description", "appearance", "clothing", "expression", "pose")
            if getattr(subj, name).value.strip()
        ]
        if subj.count > 1 and bits:
            bits[0] = f"{subj.count} {bits[0]}"
        if subj.persistent_features:
            bits.append(", ".join(subj.persistent_features))
        if bits:
            clauses.append(", ".join(bits))
    return clauses


class Flux2PromptCompiler:
    """Deterministic natural-language compiler for FLUX.2 klein (9B Base & 4B, Qwen3 encoder).

    FLUX.2 uses a Qwen3 LLM text encoder and rectified-flow transformer:
    - Responds best to clear, natural declarative sentences rather than SDXL tag soup.
    - Strips legacy SDXL ``(token:weight)`` syntax while warning the caller.
    - Reports explicit unsupported capability errors if mandatory video motion/camera-movement
      requirements are targeted at a still-image model.
    """

    def compile(self, spec: StructuredPromptSpec, model: ModelSpec) -> CompiledPrompt:
        conflicts = [c.model_dump() for c in spec.detect_conflicts()]
        warnings: list[str] = []
        unsupported: list[str] = []
        dropped_enhancements: list[str] = []

        # Explicitly flag unsupported mandatory capabilities instead of silently dropping them.
        if spec.intent.task in {"video_generate", "video_scene"}:
            unsupported.append(f"Model '{model.model_id}' is a still-image model and cannot execute task '{spec.intent.task}'.")
        if spec.motion.is_protected:
            unsupported.append(f"Mandatory motion attribute ('{spec.motion.value}') is not supported by still-image model '{model.model_id}'.")
        if spec.camera_movement.is_protected:
            unsupported.append(
                f"Mandatory camera_movement attribute ('{spec.camera_movement.value}') is not supported by still-image model '{model.model_id}'."
            )
        if spec.intent.task == "image_edit" and not model.supports_edit:
            unsupported.append(f"Task 'image_edit' is not enabled on workflow '{model.workflow}' for '{model.model_id}'.")

        clauses: list[str] = []
        had_sdxl_weights = False

        def _add(text: str) -> None:
            nonlocal had_sdxl_weights
            if not text.strip():
                return
            clean, stripped = strip_sdxl_weights(text)
            had_sdxl_weights = had_sdxl_weights or stripped
            clauses.append(clean)

        # 1. Explicit original user prompt always comes first.
        _add(spec.original_prompt)

        # 2. Subject & appearance details.
        for sc in _subject_clauses(spec):
            _add(sc)

        # 3. Declarative photography order for FLUX.2: composition & camera -> environment -> lighting & atmosphere -> materials & style.
        for field_name in (
            "composition",
            "camera_perspective",
            "environment",
            "lighting",
            "atmosphere",
            "colors_and_materials",
            "visual_style",
        ):
            attr = getattr(spec, field_name)
            if attr.value.strip():
                _add(attr.value)

        # 4. Optional enhancements only fill empty, non-protected, non-excluded fields.
        for enh in sorted(spec.optional_enhancements, key=lambda e: e.priority):
            if enh.field in {"motion", "camera_movement"}:
                dropped_enhancements.append(f"{enh.field}: skipped on still model")
                continue
            attr = getattr(spec, enh.field, None)
            if attr is not None and (attr.is_protected or attr.value.strip()):
                dropped_enhancements.append(f"{enh.field}: preserved explicit/populated value")
                continue
            _add(enh.value)

        if had_sdxl_weights:
            warnings.append("Stripped legacy SDXL (token:weight) syntax into natural prose for FLUX.2 Qwen3 encoder.")

        negative_terms = _dedupe_clauses(spec.excluded_attributes)
        if negative_terms and model.negative_prompt_strength == "light":
            warnings.append(
                f"{model.label} runs distilled low-CFG sampling (CFG ~1.2); negative exclusions have light influence compared to FLUX.2 9B Base."
            )

        positive = ". ".join(_dedupe_clauses(clauses))
        negative = ", ".join(negative_terms)
        preserved = [r.id for r in spec.required_attributes]
        return CompiledPrompt(
            model_id=model.model_id,
            model_family=model.family,
            positive=positive,
            negative=negative,
            preserved_requirements=preserved,
            dropped_enhancements=dropped_enhancements,
            warnings=warnings,
            unsupported_capabilities=unsupported,
            conflicts=conflicts,
            provenance=spec.attribute_provenance_map(),
        )


class Wan22PromptCompiler:
    """Deterministic motion-first compiler for Wan 2.2 (A14B Lightning, 5B Turbo, Fun-Control).

    Wan 2.2 uses UMT5-XXL and image-to-video temporal diffusion:
    - Leads with explicit subject action/motion and camera movement so the 4-step Lightning
      experts prioritize temporal dynamics rather than freezing on the initial frame.
    - Preserves identity/scene continuity cues cleanly.
    - Notes explicitly when Lightning CFG=1.0 ignores negative prompt tokens and reinforces
      core quality stability cleanly without SDXL tag spam.
    """

    def compile(self, spec: StructuredPromptSpec, model: ModelSpec) -> CompiledPrompt:
        conflicts = [c.model_dump() for c in spec.detect_conflicts()]
        warnings: list[str] = []
        unsupported: list[str] = []
        dropped_enhancements: list[str] = []

        if spec.intent.task in {"image_generate", "image_edit"}:
            unsupported.append(f"Model '{model.model_id}' is a video model and cannot execute still task '{spec.intent.task}'.")
        has_start_ref = any(r.role in {"start_frame", "identity"} for r in spec.reference_assets)
        if model.requires_start_image and not has_start_ref and spec.intent.task == "video_generate":
            warnings.append(
                f"{model.label} requires a start reference image; caller must supply or pre-generate a start frame before queueing."
            )

        clauses: list[str] = []
        had_sdxl_weights = False

        def _add(text: str) -> None:
            nonlocal had_sdxl_weights
            if not text.strip():
                return
            clean, stripped = strip_sdxl_weights(text)
            had_sdxl_weights = had_sdxl_weights or stripped
            clauses.append(clean)

        # 1. Explicit original user prompt first (preserves user wording verbatim).
        _add(spec.original_prompt)

        # 2. Motion & camera dynamics next (highest leverage for Wan 2.2 temporal adherence).
        if spec.motion.value.strip():
            _add(spec.motion.value)
        if spec.camera_movement.value.strip():
            _add(spec.camera_movement.value)

        # 3. Subject continuity, framing, setting, lighting, atmosphere, and visual style.
        for sc in _subject_clauses(spec):
            _add(sc)
        for field_name in (
            "composition",
            "camera_perspective",
            "environment",
            "lighting",
            "atmosphere",
            "colors_and_materials",
            "visual_style",
        ):
            attr = getattr(spec, field_name)
            if attr.value.strip():
                _add(attr.value)

        for enh in sorted(spec.optional_enhancements, key=lambda e: e.priority):
            attr = getattr(spec, enh.field, None)
            if attr is not None and (attr.is_protected or attr.value.strip()):
                dropped_enhancements.append(f"{enh.field}: preserved explicit/populated value")
                continue
            _add(enh.value)

        if had_sdxl_weights:
            warnings.append("Stripped legacy SDXL (token:weight) syntax into natural motion prose for Wan 2.2 UMT5 encoder.")

        negative_terms = _dedupe_clauses(spec.excluded_attributes)
        if negative_terms and model.negative_prompt_strength == "ignored_cfg1":
            warnings.append(
                f"{model.label} runs at CFG 1.0 (4-step Lightning/Turbo), where negative prompts have no effect; express critical constraints positively in the prompt."
            )

        positive = ". ".join(_dedupe_clauses(clauses))
        negative = ", ".join(negative_terms)
        preserved = [r.id for r in spec.required_attributes]
        return CompiledPrompt(
            model_id=model.model_id,
            model_family=model.family,
            positive=positive,
            negative=negative,
            preserved_requirements=preserved,
            dropped_enhancements=dropped_enhancements,
            warnings=warnings,
            unsupported_capabilities=unsupported,
            conflicts=conflicts,
            provenance=spec.attribute_provenance_map(),
        )


_COMPILERS_BY_FAMILY = {
    "flux2": Flux2PromptCompiler(),
    "wan22": Wan22PromptCompiler(),
}


def compile_prompt_for_model(
    spec: StructuredPromptSpec,
    model_id: str,
    *,
    registry: ModelRegistry = DEFAULT_REGISTRY,
) -> CompiledPrompt:
    """Compile a canonical StructuredPromptSpec for any registered local model."""
    model = registry.get(model_id)
    if model is None:
        raise ValueError(f"Unknown model_id '{model_id}' in ModelRegistry")
    compiler = _COMPILERS_BY_FAMILY.get(model.family)
    if compiler is None:
        positive, negative = spec.to_baseline_prompt()
        return CompiledPrompt(
            model_id=model.model_id,
            model_family=model.family,
            positive=positive,
            negative=negative,
            preserved_requirements=[r.id for r in spec.required_attributes],
            warnings=[f"No specialized compiler for family '{model.family}'; used baseline prompt assembly."],
            conflicts=[c.model_dump() for c in spec.detect_conflicts()],
            provenance=spec.attribute_provenance_map(),
        )
    return compiler.compile(spec, model)
