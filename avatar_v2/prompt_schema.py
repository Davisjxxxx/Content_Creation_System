"""Model-independent Structured Prompt Schema for Avatar V2.

Serves as the canonical prompt contract across:
- Automatic mode (natural-language prompt entry -> structured spec)
- Guided mode (optional structured creative controls, locking, controlled randomization)
- Script mode (Native SDK & Static Structured Prompt Extractor)
- Image generation, image editing, video generation, multi-reference, and continuity
- Future Sprint 3 model-specific compilers & Sprint 6 local prompt intelligence

Design guarantees:
1. Explicit user requirements ("specified" / "locked") always take precedence over
   automatic or optional enhancements.
2. Every creative attribute tracks its state (automatic / specified / randomized / locked)
   and provenance (user_explicit / guided_ui / script / recipe / randomizer / local_llm / system_default).
3. Explicit conflict detection surfaces contradictions deterministically without silently
   dropping user intent.
4. Zero external frameworks beyond Pydantic (already a core dependency) and stdlib.
"""

from __future__ import annotations

import random
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .policy import depicts_minor

AttributeState = Literal["automatic", "specified", "randomized", "locked"]
AttributeProvenance = Literal[
    "user_explicit",
    "guided_ui",
    "script",
    "recipe",
    "randomizer",
    "local_llm",
    "system_default",
]
GenerationTask = Literal[
    "image_generate",
    "image_edit",
    "video_generate",
    "video_scene",
    "multi_ref_generate",
]
InputMode = Literal["automatic", "guided", "script"]
ReferenceRole = Literal[
    "primary_starter",
    "face_identity",
    "front_body",
    "back_body",
    "left_side",
    "right_side",
    "front_three_quarter",
    "rear_three_quarter",
    "appearance_wardrobe",
    "identity",
    "style",
    "scene",
    "edit_source",
    "start_frame",
    "end_frame",
    "motion_video",
    "audio",
    "anatomy_guide",
]
ConflictSeverity = Literal["error", "warning"]

SCHEMA_VERSION = "1.0"

# Pairs of mutually exclusive semantic descriptors checked during conflict detection.
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("indoors", "outdoors"),
    ("indoor", "outdoor"),
    ("daylight", "night"),
    ("daytime", "nighttime"),
    ("bright sunlight", "pitch dark"),
    ("smiling", "frowning"),
    ("standing", "sitting"),
    ("monochrome", "vibrant color"),
    ("black and white", "full color"),
    ("static camera", "tracking shot"),
    ("frozen still", "running"),
)

_EXPLICIT_PROVENANCES = frozenset({"user_explicit", "guided_ui", "script"})
_PROTECTED_STATES = frozenset({"specified", "locked"})


class PromptAttribute(BaseModel):
    """One creative attribute with explicit state, provenance, and randomization pool."""

    value: str = ""
    state: AttributeState = "automatic"
    provenance: AttributeProvenance = "system_default"
    candidates: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def is_protected(self) -> bool:
        """True when explicit user/script input must not be overwritten automatically."""
        return self.state in _PROTECTED_STATES and bool(self.value.strip())

    def with_auto_value(self, value: str, *, provenance: AttributeProvenance = "local_llm") -> tuple["PromptAttribute", bool]:
        """Set value only if this attribute is not already locked or user-specified."""
        if self.is_protected or self.state == "locked":
            return self, False
        return (
            self.model_copy(
                update={
                    "value": value.strip(),
                    "state": "automatic",
                    "provenance": provenance,
                }
            ),
            True,
        )


class SubjectAttributeSpec(BaseModel):
    """Structured description of one subject in the scene."""

    id: str = "primary"
    role: str = "main subject"
    description: PromptAttribute = Field(default_factory=PromptAttribute)
    count: int = Field(default=1, ge=1, le=20)
    appearance: PromptAttribute = Field(default_factory=PromptAttribute)
    clothing: PromptAttribute = Field(default_factory=PromptAttribute)
    expression: PromptAttribute = Field(default_factory=PromptAttribute)
    pose: PromptAttribute = Field(default_factory=PromptAttribute)
    persistent_features: list[str] = Field(default_factory=list)
    apparent_age_years: int | None = Field(default=None, ge=18)


class ReferenceAssetItem(BaseModel):
    """One reference image, video, or audio asset bound to a semantic role and optional subject."""

    asset_id: str
    role: ReferenceRole
    path: str
    subject_id: str = "primary"
    view_orientation: str = ""
    source_provenance: str = ""
    asset_fingerprint: str = ""
    preserve_identity: bool = True
    preserve_appearance: bool = True
    preserve_wardrobe: bool = True
    locked_attributes: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    locked: bool = False
    description: str = ""


class RequirementItem(BaseModel):
    """A mandatory, verifiable requirement preserved independently of subjective quality."""

    id: str
    category: str
    description: str
    provenance: AttributeProvenance = "user_explicit"
    locked: bool = True


class EnhancementItem(BaseModel):
    """An optional non-mandatory enhancement that yields to explicit user requirements."""

    field: str
    value: str
    provenance: AttributeProvenance = "local_llm"
    priority: int = Field(default=1, ge=1, le=10)


class GenerationIntent(BaseModel):
    """High-level task and interaction mode."""

    task: GenerationTask = "image_generate"
    input_mode: InputMode = "automatic"
    goal: str = ""
    preserve_identity: bool = True
    preserve_scene: bool = False


class RandomizationConfig(BaseModel):
    """Deterministic controlled randomization configuration."""

    enabled: bool = False
    seed: int = Field(default=41001, ge=0)
    pools: dict[str, list[str]] = Field(default_factory=dict)


class GenerationConstraints(BaseModel):
    """Execution bounds and resource-neutral target constraints."""

    width: int = Field(default=896, ge=256, le=4096)
    height: int = Field(default=1280, ge=256, le=4096)
    fps: int = Field(default=16, ge=1, le=60)
    duration_s: float | None = Field(default=None, gt=0, le=120)
    frames: int = Field(default=1, ge=1, le=1921)
    shots: int = Field(default=1, ge=1, le=12)
    seed: int | None = Field(default=None, ge=0)
    model_preference: str = "auto"
    local_only: bool = True
    max_retries: int = Field(default=2, ge=0, le=3)
    quality_priority: Literal["balanced", "quality", "speed"] = "quality"


class ContinuitySpec(BaseModel):
    """Character and multi-shot scene continuity metadata."""

    avatar_id: str | None = None
    scene_id: str | None = None
    shot_index: int = Field(default=0, ge=0)
    previous_shot_id: str | None = None
    carry_over_attributes: list[str] = Field(default_factory=list)


class PromptConflict(BaseModel):
    """An explicitly detected contradiction or constraint violation."""

    code: str
    severity: ConflictSeverity
    fields: list[str]
    message: str


CREATIVE_FIELD_NAMES: tuple[str, ...] = (
    "environment",
    "composition",
    "camera_perspective",
    "lighting",
    "atmosphere",
    "visual_style",
    "colors_and_materials",
    "motion",
    "camera_movement",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


_NEGATION_PREFIX = re.compile(r"\b(?:no|without|avoid|never|zero|not|free of|neither|nor)\b[^.,;]{0,32}$")
_NEGATION_SUFFIX = re.compile(r"^\s*-\s*free\b")


def _has_positive_mention(text: str, term: str) -> bool:
    """Return True when ``term`` appears as a positive request rather than inside a negation clause."""
    norm_text = _norm(text)
    norm_term = _norm(term)
    if not norm_term:
        return False
    for match in re.finditer(rf"\b{re.escape(norm_term)}\b", norm_text):
        prefix = norm_text[max(0, match.start() - 40):match.start()]
        suffix = norm_text[match.end():match.end() + 16]
        if not _NEGATION_PREFIX.search(prefix) and not _NEGATION_SUFFIX.search(suffix):
            return True
    return False


class StructuredPromptSpec(BaseModel):
    """Canonical model-independent prompt specification shared by all Avatar V2 modes."""

    schema_version: str = SCHEMA_VERSION
    original_prompt: str = ""
    intent: GenerationIntent = Field(default_factory=GenerationIntent)
    subjects: list[SubjectAttributeSpec] = Field(default_factory=list)

    environment: PromptAttribute = Field(default_factory=PromptAttribute)
    composition: PromptAttribute = Field(default_factory=PromptAttribute)
    camera_perspective: PromptAttribute = Field(default_factory=PromptAttribute)
    lighting: PromptAttribute = Field(default_factory=PromptAttribute)
    atmosphere: PromptAttribute = Field(default_factory=PromptAttribute)
    visual_style: PromptAttribute = Field(default_factory=PromptAttribute)
    colors_and_materials: PromptAttribute = Field(default_factory=PromptAttribute)
    motion: PromptAttribute = Field(default_factory=PromptAttribute)
    camera_movement: PromptAttribute = Field(default_factory=PromptAttribute)

    reference_assets: list[ReferenceAssetItem] = Field(default_factory=list)
    required_attributes: list[RequirementItem] = Field(default_factory=list)
    excluded_attributes: list[str] = Field(default_factory=list)
    optional_enhancements: list[EnhancementItem] = Field(default_factory=list)
    randomization: RandomizationConfig = Field(default_factory=RandomizationConfig)
    constraints: GenerationConstraints = Field(default_factory=GenerationConstraints)
    continuity: ContinuitySpec = Field(default_factory=ContinuitySpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_video_frames(self) -> "StructuredPromptSpec":
        if self.constraints.duration_s is not None and self.intent.task in {"video_generate", "video_scene"}:
            if self.constraints.frames <= 1:
                self.constraints.frames = max(1, round(self.constraints.duration_s * self.constraints.fps))
        return self

    @classmethod
    def from_natural_language(
        cls,
        prompt: str,
        *,
        task: GenerationTask = "image_generate",
        references: list[ReferenceAssetItem] | None = None,
        constraints: GenerationConstraints | None = None,
        excluded: list[str] | None = None,
    ) -> "StructuredPromptSpec":
        """Create an Automatic-mode spec directly from a user's natural-language prompt."""
        clean = " ".join(str(prompt or "").split())
        req = (
            [RequirementItem(id="user_prompt", category="natural_language", description=clean, provenance="user_explicit")]
            if clean
            else []
        )
        return cls(
            original_prompt=clean,
            intent=GenerationIntent(task=task, input_mode="automatic", goal=clean[:240]),
            reference_assets=list(references or []),
            required_attributes=req,
            excluded_attributes=list(excluded or []),
            constraints=constraints or GenerationConstraints(),
        )

    @classmethod
    def from_guided(
        cls,
        *,
        original_prompt: str = "",
        task: GenerationTask = "image_generate",
        attributes: dict[str, str | PromptAttribute] | None = None,
        locked_fields: set[str] | list[str] | None = None,
        randomized_fields: dict[str, list[str]] | None = None,
        excluded: list[str] | None = None,
        references: list[ReferenceAssetItem] | None = None,
        constraints: GenerationConstraints | None = None,
        seed: int = 41001,
    ) -> "StructuredPromptSpec":
        """Create a Guided-mode spec with explicit locking and controlled randomization."""
        locked = set(locked_fields or ())
        rand_pools = dict(randomized_fields or {})
        kwargs: dict[str, Any] = {}
        for name in CREATIVE_FIELD_NAMES:
            raw = (attributes or {}).get(name)
            if isinstance(raw, PromptAttribute):
                attr = raw
                if name in locked:
                    attr = attr.model_copy(update={"state": "locked"})
                kwargs[name] = attr
            elif isinstance(raw, str) and raw.strip():
                kwargs[name] = PromptAttribute(
                    value=raw.strip(),
                    state="locked" if name in locked else "specified",
                    provenance="guided_ui",
                    candidates=list(rand_pools.get(name, [])),
                )
            elif name in rand_pools and rand_pools[name]:
                kwargs[name] = PromptAttribute(
                    value="",
                    state="randomized",
                    provenance="guided_ui",
                    candidates=list(rand_pools[name]),
                )
        return cls(
            original_prompt=original_prompt.strip(),
            intent=GenerationIntent(task=task, input_mode="guided", goal=original_prompt.strip()[:240]),
            reference_assets=list(references or []),
            excluded_attributes=list(excluded or []),
            randomization=RandomizationConfig(enabled=bool(rand_pools), seed=seed, pools=rand_pools),
            constraints=constraints or GenerationConstraints(),
            **kwargs,
        )

    def locked_attributes(self) -> dict[str, str]:
        """Return all locked top-level and subject attributes."""
        out: dict[str, str] = {}
        for name in CREATIVE_FIELD_NAMES:
            attr: PromptAttribute = getattr(self, name)
            if attr.state == "locked":
                out[name] = attr.value
        for subj in self.subjects:
            for sub_name in ("description", "appearance", "clothing", "expression", "pose"):
                attr = getattr(subj, sub_name)
                if attr.state == "locked":
                    out[f"subject.{subj.id}.{sub_name}"] = attr.value
        return out

    def attribute_provenance_map(self) -> dict[str, dict[str, str]]:
        """Inspect state and provenance of every populated or locked creative attribute."""
        out: dict[str, dict[str, str]] = {}
        for name in CREATIVE_FIELD_NAMES:
            attr: PromptAttribute = getattr(self, name)
            if attr.value or attr.state != "automatic":
                out[name] = {"state": attr.state, "provenance": attr.provenance, "value": attr.value}
        return out

    def apply_enhancement(
        self,
        field_name: str,
        value: str,
        *,
        provenance: AttributeProvenance = "local_llm",
    ) -> tuple["StructuredPromptSpec", bool]:
        """Apply an automatic enhancement ONLY if it does not overwrite explicit/locked intent
        or contradict excluded attributes."""
        if field_name not in CREATIVE_FIELD_NAMES:
            return self, False
        clean_val = value.strip()
        if not clean_val:
            return self, False
        lower_val = _norm(clean_val)
        for excl in self.excluded_attributes:
            ex = _norm(excl)
            if ex and (ex in lower_val or lower_val in ex):
                return self, False
        current: PromptAttribute = getattr(self, field_name)
        updated_attr, changed = current.with_auto_value(clean_val, provenance=provenance)
        if not changed:
            return self, False
        return self.model_copy(update={field_name: updated_attr}), True

    def apply_randomization(self, seed: int | None = None) -> "StructuredPromptSpec":
        """Deterministically resolve all 'randomized' fields without touching locked/specified fields."""
        effective_seed = self.randomization.seed if seed is None else int(seed)
        rng = random.Random(effective_seed)
        updates: dict[str, Any] = {}
        for name in CREATIVE_FIELD_NAMES:
            attr: PromptAttribute = getattr(self, name)
            if attr.state != "randomized":
                continue
            pool = attr.candidates or self.randomization.pools.get(name) or []
            # Filter out any candidate that violates excluded_attributes.
            safe_pool = [
                cand
                for cand in pool
                if cand.strip()
                and not any(_norm(ex) and _norm(ex) in _norm(cand) for ex in self.excluded_attributes)
            ]
            if not safe_pool:
                continue
            chosen = rng.choice(safe_pool).strip()
            updates[name] = attr.model_copy(
                update={"value": chosen, "provenance": "randomizer", "candidates": list(pool)}
            )
        new_subjects: list[SubjectAttributeSpec] = []
        subjects_changed = False
        for subj in self.subjects:
            s_updates: dict[str, Any] = {}
            for sub_name in ("description", "appearance", "clothing", "expression", "pose"):
                attr = getattr(subj, sub_name)
                if attr.state == "randomized" and attr.candidates:
                    safe_pool = [
                        c
                        for c in attr.candidates
                        if c.strip()
                        and not any(_norm(ex) and _norm(ex) in _norm(c) for ex in self.excluded_attributes)
                    ]
                    if safe_pool:
                        s_updates[sub_name] = attr.model_copy(
                            update={"value": rng.choice(safe_pool).strip(), "provenance": "randomizer"}
                        )
            if s_updates:
                subjects_changed = True
                new_subjects.append(subj.model_copy(update=s_updates))
            else:
                new_subjects.append(subj)
        if subjects_changed:
            updates["subjects"] = new_subjects
        return self.model_copy(update=updates) if updates else self

    def detect_conflicts(self) -> list[PromptConflict]:
        """Deterministically detect contradictions, missing required assets, and policy violations."""
        conflicts: list[PromptConflict] = []

        # 1. Gather all active textual assertions with their field paths.
        active_texts: list[tuple[str, str]] = []
        if self.original_prompt.strip():
            active_texts.append(("original_prompt", self.original_prompt))
        for req in self.required_attributes:
            if req.description.strip():
                active_texts.append((f"required_attributes.{req.id}", req.description))
        for name in CREATIVE_FIELD_NAMES:
            attr: PromptAttribute = getattr(self, name)
            if attr.value.strip():
                active_texts.append((name, attr.value))
        for subj in self.subjects:
            for sub_name in ("description", "appearance", "clothing", "expression", "pose"):
                attr = getattr(subj, sub_name)
                if attr.value.strip():
                    active_texts.append((f"subjects.{subj.id}.{sub_name}", attr.value))

        # 2. Required / active attribute vs excluded_attributes contradiction (ignoring negated mentions like "no laptop").
        for excl in self.excluded_attributes:
            ex_norm = _norm(excl)
            if not ex_norm:
                continue
            for field_path, text in active_texts:
                if _has_positive_mention(text, ex_norm):
                    conflicts.append(
                        PromptConflict(
                            code="required_excluded_contradiction",
                            severity="error",
                            fields=[field_path, "excluded_attributes"],
                            message=f"'{excl}' is excluded, but also requested in {field_path}.",
                        )
                    )

        # 3. Cross-field semantic antonym contradictions.
        for left, right in _ANTONYM_PAIRS:
            left_hits = [fp for fp, txt in active_texts if re.search(rf"\b{re.escape(left)}\b", _norm(txt))]
            right_hits = [fp for fp, txt in active_texts if re.search(rf"\b{re.escape(right)}\b", _norm(txt))]
            if left_hits and right_hits:
                conflicts.append(
                    PromptConflict(
                        code="semantic_contradiction",
                        severity="warning",
                        fields=sorted(set(left_hits + right_hits)),
                        message=f"Contradictory descriptors '{left}' and '{right}' appear across {', '.join(sorted(set(left_hits + right_hits)))}.",
                    )
                )

        # 4. Optional enhancements targeting protected/locked fields.
        for enh in self.optional_enhancements:
            if enh.field in CREATIVE_FIELD_NAMES:
                attr = getattr(self, enh.field)
                if attr.is_protected and _norm(attr.value) != _norm(enh.value):
                    conflicts.append(
                        PromptConflict(
                            code="enhancement_blocked_by_explicit_attribute",
                            severity="warning",
                            fields=[enh.field, "optional_enhancements"],
                            message=f"Optional enhancement for '{enh.field}' conflicts with {attr.state} value '{attr.value}' and will be skipped.",
                        )
                    )

        # 5. Task & reference constraints.
        roles = {ref.role for ref in self.reference_assets}
        if self.intent.task == "image_edit" and "edit_source" not in roles:
            conflicts.append(
                PromptConflict(
                    code="missing_edit_source",
                    severity="error",
                    fields=["intent.task", "reference_assets"],
                    message="Image editing requires at least one reference asset with role='edit_source'.",
                )
            )
        if self.intent.task == "multi_ref_generate" and len(self.reference_assets) < 2:
            conflicts.append(
                PromptConflict(
                    code="insufficient_multi_references",
                    severity="warning",
                    fields=["intent.task", "reference_assets"],
                    message="Multi-reference generation expects at least 2 reference assets.",
                )
            )
        if self.intent.task in {"video_generate", "video_scene"} and self.constraints.frames <= 1 and self.constraints.duration_s is None:
            conflicts.append(
                PromptConflict(
                    code="video_missing_duration",
                    severity="error",
                    fields=["intent.task", "constraints.frames"],
                    message="Video generation requires frames > 1 or duration_s > 0.",
                )
            )
        if self.constraints.local_only and self.constraints.model_preference.lower().startswith("runway"):
            conflicts.append(
                PromptConflict(
                    code="cloud_model_disallowed_in_local_only_mode",
                    severity="error",
                    fields=["constraints.local_only", "constraints.model_preference"],
                    message="Cloud model preference conflicts with local_only=True.",
                )
            )

        # 5b. Multi-view subject reference consistency checks (Sprint 5, Phase E).
        starters = [ref for ref in self.reference_assets if ref.role in {"primary_starter", "start_frame"}]
        if len(starters) > 1 and len({r.path for r in starters}) > 1:
            conflicts.append(
                PromptConflict(
                    code="competing_starter_frames",
                    severity="error",
                    fields=["reference_assets"],
                    message="Multiple distinct primary_starter/start_frame assets conflict; designate one starter image and assign view roles to supplemental references.",
                )
            )
        singular_views = {"front_body", "back_body", "left_side", "right_side", "front_three_quarter", "rear_three_quarter"}
        by_subj_role: dict[tuple[str, str], set[str]] = {}
        by_subj_path: dict[tuple[str, str], set[str]] = {}
        for ref in self.reference_assets:
            if ref.role in singular_views:
                by_subj_role.setdefault((ref.subject_id, ref.role), set()).add(ref.path)
            by_subj_path.setdefault((ref.subject_id, ref.path), set()).add(ref.role)
        for (subj_id, role), paths in by_subj_role.items():
            if len(paths) > 1:
                conflicts.append(
                    PromptConflict(
                        code="duplicate_singular_subject_view",
                        severity="error",
                        fields=["reference_assets"],
                        message=f"Subject '{subj_id}' has multiple conflicting images assigned to singular role '{role}'.",
                    )
                )
        for (subj_id, path), assigned in by_subj_path.items():
            if {"front_body", "back_body"} <= assigned or {"left_side", "right_side"} <= assigned:
                conflicts.append(
                    PromptConflict(
                        code="contradictory_subject_view_orientation",
                        severity="error",
                        fields=["reference_assets"],
                        message=f"Same reference '{path}' is assigned to opposite views ({', '.join(sorted(assigned))}) for subject '{subj_id}'.",
                    )
                )

        # 6. Mandatory adult-safety policy check (avatar_v2.policy.depicts_minor).
        combined_text = " ".join(txt for _fp, txt in active_texts)
        minor_phrase = depicts_minor(combined_text)
        if minor_phrase:
            conflicts.append(
                PromptConflict(
                    code="policy_minor_depiction_blocked",
                    severity="error",
                    fields=["original_prompt"],
                    message=f"Blocked by adult-only policy: prompt contains disallowed minor phrasing ('{minor_phrase}').",
                )
            )

        return conflicts

    def to_baseline_prompt(self) -> tuple[str, str]:
        """Deterministic model-neutral baseline prompt assembly (pre-Sprint 3 compilers).

        Returns ``(positive_prompt, negative_prompt)``. Explicit user wording and locked
        attributes are always included first.
        """
        parts: list[str] = []
        if self.original_prompt.strip():
            parts.append(self.original_prompt.strip())
        for subj in self.subjects:
            subj_bits = [
                getattr(subj, f).value.strip()
                for f in ("description", "appearance", "clothing", "expression", "pose")
                if getattr(subj, f).value.strip()
            ]
            if subj.count > 1 and subj_bits:
                subj_bits[0] = f"{subj.count} {subj_bits[0]}"
            if subj_bits:
                parts.append(", ".join(subj_bits))
        for name in CREATIVE_FIELD_NAMES:
            attr: PromptAttribute = getattr(self, name)
            if attr.value.strip():
                parts.append(attr.value.strip())
        # Append optional enhancements only when their target field is still empty.
        for enh in sorted(self.optional_enhancements, key=lambda e: e.priority):
            if enh.field in CREATIVE_FIELD_NAMES:
                attr = getattr(self, enh.field)
                if not attr.value.strip() and not attr.is_protected and enh.value.strip():
                    parts.append(enh.value.strip())

        # Deduplicate while preserving exact order.
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            key = _norm(part)
            if key and key not in seen:
                seen.add(key)
                ordered.append(part)

        positive = ". ".join(ordered)
        negative = ", ".join(dict.fromkeys(ex.strip() for ex in self.excluded_attributes if ex.strip()))
        return positive, negative
