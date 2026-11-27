"""Local Intent Interpreter converting natural language into validated StructuredPromptSpec (Sprint 6).

Architecture:
- Deterministic Python parser runs first and handles straightforward/explicit prompts with
  zero LLM latency and 100% reproducibility.
- Optional local Ollama LLM semantic assist is invoked only when enabled AND useful beyond
  deterministic rules, scheduled via ResourcePlanner (CPU-only when ComfyUI is warm).
- Treats ALL LLM output as untrusted structured data: never mutates ``original_prompt``,
  never overrides explicit/locked user requirements, never introduces policy violations,
  and falls back cleanly to deterministic parsing if the LLM is offline or malformed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .ollama_client import (
    DEFAULT_OLLAMA_URL,
    LLMExecutionSchedule,
    LocalOllamaClient,
    choose_llm_execution_schedule,
    select_intelligence_model,
)
from .policy import depicts_minor
from .prompt_schema import (
    CREATIVE_FIELD_NAMES,
    EnhancementItem,
    GenerationConstraints,
    GenerationIntent,
    PromptAttribute,
    ReferenceAssetItem,
    RequirementItem,
    StructuredPromptSpec,
)
from .resource_monitor import ResourceSnapshot
from .subject_references import SubjectReferenceSet, infer_reference_scenario

InterpretedIntentKind = Literal[
    "image_generate",
    "image_edit",
    "image_to_video",
    "text_to_video",
    "multi_ref_generate",
    "character_continuity",
    "scene_continuity",
    "keyframe_generate",
    "flf_video_generate",
    "continuous_generate",
    "multi_stage_generate",
]

# Deterministic semantic extraction patterns.
_EDIT_RE = re.compile(
    r"\b(edit this|change the|replace the|replace her|replace his|remove the|swap the|recolor|inpaint|retouch|turn the background into|put her in|place him in)\b",
    re.IGNORECASE,
)
_VIDEO_RE = re.compile(
    r"\b(video|clip|animate|animation|movie|footage|seconds? long|5s|10s|shot by shot|walks? |turns? |waves? |nods? |speaks? |explains? |camera push|camera pan|camera orbit|dolly|tracking shot)\b",
    re.IGNORECASE,
)
_CONTINUOUS_RE = re.compile(r"\b(continuous|batch of \d+|generate \d+ variations?|keep generating|series of \d+)\b", re.IGNORECASE)
_SCENE_CONT_RE = re.compile(r"\b(next shot|continue the scene|continuing from|shot \d+|multi[- ]shot|same scene)\b", re.IGNORECASE)
_CHAR_CONT_RE = re.compile(r"\b(same person|same woman|same man|same character|same subject|consistent identity|keep her face|keep his face)\b", re.IGNORECASE)
_KEYFRAME_RE = re.compile(r"\b(keyframe|end frame|start and end frame|intermediate view|turnaround sheet)\b", re.IGNORECASE)
_TURN_RE = re.compile(r"\b(turns? around|turning from front to back|turns? to her left|turns? away|walks? away|360|orbit)\b", re.IGNORECASE)

_SPELLED_COUNTS: dict[str, int] = {
    "one": 1,
    "single": 1,
    "two": 2,
    "pair": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "dozen": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_QUOTED_TEXT_RE = re.compile(r'(?:reading|says?|text|sign|label|title|printed)\s+["\']([^"\']{1,60})["\']|["\']([A-Z0-9][A-Za-z0-9 &!?:.-]{1,40})["\']')
_COUNT_RE = re.compile(
    r"\b(?:exactly\s+)?(one|single|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|dozen|pair\s+of|[1-9]\d{0,2})\s+([a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2})\b",
    re.IGNORECASE,
)
_SPATIAL_REL_RE = re.compile(
    r"\b((?:an?|the|\d+|one|two|three|four|five)\s+[a-z0-9][a-z0-9\s-]{1,32}?)\s+"
    r"(to the left of|to the right of|in front of|behind|above|below|underneath|beside|next to|between)\s+"
    r"((?:an?|the|\d+|one|two|three|four|five)\s+[a-z0-9][a-z0-9\s-]{1,32}?)(?=[,.;]|$|\band\b|\bwith\b|\bwhile\b)",
    re.IGNORECASE,
)
_SPATIAL_SIDE_RE = re.compile(
    r"\b((?:an?|the|\d+|one|two|three|four|five)\s+[a-z0-9][a-z0-9\s-]{1,32}?)\s+"
    r"(on the left|on the right|in the foreground|in the background|in the center|centered|on the right third|on the left third)\b",
    re.IGNORECASE,
)
_SPATIAL_RE = re.compile(
    r"\b(on the left|on the right|to the left of|to the right of|in front of|behind|above|below|underneath|beside|next to|between|foreground|background|right third|left third|centered)\b",
    re.IGNORECASE,
)
_EXCLUSION_RE = re.compile(
    r"\b(?:no|without|exclude|avoid)\s+([a-z0-9][a-z0-9\s-]{1,35}?)(?=[,.;]|$|\band\b|\bwhile\b|\bwith\b)",
    re.IGNORECASE,
)
_CAMERA_MOVE_RE = re.compile(
    r"\b(slow push[- ]?in|dolly in|dolly out|pull back|pan left|pan right|tilt up|tilt down|camera orbit|tracking shot|handheld|locked tripod|static camera)\b",
    re.IGNORECASE,
)
_LIGHTING_RE = re.compile(
    r"\b(golden hour|soft window light|natural window light|studio lighting|rim light|backlit|warm tungsten|neon lighting|overcast daylight|chiaroscuro|dramatic side lighting)\b",
    re.IGNORECASE,
)
_COMPOSITION_RE = re.compile(
    r"\b(close[- ]?up|extreme close[- ]?up|medium shot|medium close[- ]?up|full[- ]?body|full[- ]?length|wide shot|headshot|three[- ]?quarter portrait|rule of thirds)\b",
    re.IGNORECASE,
)

# Words after a number that are ages/durations/lens sizes, NOT object counts.
_NON_OBJECT_UNITS = frozenset(
    {"year", "years", "yr", "yrs", "second", "seconds", "sec", "minute", "minutes", "min", "mm", "px", "mp", "gb", "mb", "step", "steps", "fps", "k", "am", "pm", "s"}
)

_LLM_SYSTEM_PROMPT = (
    "You are a strict structured-intent parser for an offline local image/video generator. "
    "Return ONLY valid JSON with optional keys: "
    "'intent_kind' (one of image_generate, image_edit, image_to_video, text_to_video, multi_ref_generate, "
    "character_continuity, scene_continuity, keyframe_generate, flf_video_generate, continuous_generate, multi_stage_generate), "
    "'environment', 'composition', 'camera_perspective', 'lighting', 'atmosphere', 'visual_style', "
    "'colors_and_materials', 'motion', 'camera_movement' (short concise strings, omit if not implied), "
    "'mandatory_requirements' (list of short strings explicitly stated by user), "
    "'excluded_attributes' (list of short strings the user asked to avoid). "
    "Rules: Never invent unrequested objects or clothing, never mention minors, never output shell commands, "
    "never select cloud services, and never contradict the user prompt."
)


@dataclass
class InterpretationTrace:
    mode: Literal["deterministic", "llm_assisted", "deterministic_fallback"]
    detected_intent: InterpretedIntentKind
    scenario: str
    llm_invoked: bool = False
    llm_model: str = ""
    llm_schedule_mode: str = ""
    llm_latency_ms: float = 0.0
    added_fields: list[str] = field(default_factory=list)
    rejected_llm_suggestions: list[str] = field(default_factory=list)
    fallback_reason: str = ""

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterpretedRequest:
    ok: bool
    detected_intent: InterpretedIntentKind
    spec: StructuredPromptSpec
    expected_orientation: str
    typography_targets: list[str]
    mandatory_requirements: list[dict[str, Any]]
    optional_preferences: list[str]
    inferred_defaults: list[str]
    trace: InterpretationTrace
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detected_intent": self.detected_intent,
            "spec": self.spec.model_dump(),
            "expected_orientation": self.expected_orientation,
            "typography_targets": list(self.typography_targets),
            "mandatory_requirements": list(self.mandatory_requirements),
            "optional_preferences": list(self.optional_preferences),
            "inferred_defaults": list(self.inferred_defaults),
            "trace": self.trace.public(),
            "warnings": list(self.warnings),
            "error": self.error,
        }


def _detect_intent_deterministic(
    prompt: str,
    *,
    requested_action: str | None = None,
    has_starter_image: bool = False,
    subject_refs: SubjectReferenceSet | None = None,
    input_mode: str = "automatic",
    has_multiple_shots: bool = False,
) -> InterpretedIntentKind:
    text = str(prompt or "").strip()
    supp_count = len(subject_refs.references) if subject_refs else 0
    has_rear_ref = bool(
        subject_refs and any(r.role in {"back_body", "rear_three_quarter"} for r in subject_refs.references)
    )

    if input_mode == "script" and _CONTINUOUS_RE.search(text):
        return "continuous_generate"
    if _CONTINUOUS_RE.search(text) and requested_action != "photo":
        return "continuous_generate"

    wants_video = (
        requested_action in {"scene", "video", "video_generate", "video_scene", "move"}
        or has_multiple_shots
        or (requested_action is None and bool(_VIDEO_RE.search(text)))
    )
    if wants_video:
        if has_multiple_shots or _SCENE_CONT_RE.search(text):
            return "scene_continuity"
        if _TURN_RE.search(text) and (has_rear_ref or supp_count >= 2):
            return "flf_video_generate"
        if has_starter_image:
            return "image_to_video"
        if supp_count > 0:
            return "multi_stage_generate"
        return "text_to_video"

    if _KEYFRAME_RE.search(text):
        return "keyframe_generate"
    if (has_starter_image or supp_count > 0) and _EDIT_RE.search(text):
        return "image_edit"
    if supp_count >= 2 or (has_starter_image and supp_count >= 1):
        return "multi_ref_generate"
    if (has_starter_image or supp_count == 1) and _CHAR_CONT_RE.search(text):
        return "character_continuity"
    return "image_generate"


def _map_intent_to_schema_task(intent_kind: InterpretedIntentKind) -> str:
    if intent_kind in {"image_to_video", "text_to_video", "flf_video_generate", "multi_stage_generate"}:
        return "video_generate"
    if intent_kind == "scene_continuity":
        return "video_scene"
    if intent_kind == "image_edit":
        return "image_edit"
    if intent_kind == "multi_ref_generate":
        return "multi_ref_generate"
    return "image_generate"


def _should_invoke_llm(prompt: str, *, use_llm: bool | Literal["auto"]) -> bool:
    if use_llm is False:
        return False
    if use_llm is True:
        return True
    # "auto": invoke local LLM only for complex multi-clause prompts (>140 chars or >=3 clauses).
    text = prompt.strip()
    clauses = [c for c in re.split(r"[,;.]", text) if c.strip()]
    return len(text) >= 140 or len(clauses) >= 3


def interpret_natural_language(
    prompt: str,
    *,
    requested_action: str | None = None,
    starter_image: str | None = None,
    subject_refs: SubjectReferenceSet | None = None,
    guided_spec: StructuredPromptSpec | None = None,
    input_mode: Literal["automatic", "guided", "script"] = "automatic",
    has_multiple_shots: bool = False,
    model_preference: str = "auto",
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    use_llm: bool | Literal["auto"] = False,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str | None = None,
    llm_client: LocalOllamaClient | None = None,
    llm_schedule: LLMExecutionSchedule | None = None,
    snapshot: ResourceSnapshot | None = None,
    comfy_models_warm: bool = False,
    active_generation_running: bool = False,
) -> InterpretedRequest:
    """Convert natural language + optional references/guided controls into a validated StructuredPromptSpec."""
    clean_prompt = " ".join(str(prompt or "").split())
    has_starter = bool(starter_image and str(starter_image).strip())
    detected = _detect_intent_deterministic(
        clean_prompt,
        requested_action=requested_action,
        has_starter_image=has_starter,
        subject_refs=subject_refs,
        input_mode=input_mode,
        has_multiple_shots=has_multiple_shots,
    )
    scenario = infer_reference_scenario(clean_prompt)
    schema_task = _map_intent_to_schema_task(detected)
    is_video = schema_task in {"video_generate", "video_scene"}

    # Start from guided_spec if supplied (preserving all locked & specified guided attributes!).
    base_spec = guided_spec.model_copy(deep=True) if guided_spec is not None else StructuredPromptSpec(
        original_prompt=clean_prompt,
        intent=GenerationIntent(task=schema_task, input_mode=input_mode),  # type: ignore[arg-type]
    )
    if clean_prompt and not base_spec.original_prompt:
        base_spec.original_prompt = clean_prompt
    base_spec.intent.task = schema_task  # type: ignore[assignment]
    base_spec.intent.input_mode = input_mode

    # Build reference assets list cleanly (never silently dropping user references).
    ref_assets: list[ReferenceAssetItem] = list(base_spec.reference_assets)
    existing_paths = {(r.role, r.path) for r in ref_assets}
    if has_starter:
        starter_role = "edit_source" if schema_task == "image_edit" else ("start_frame" if is_video else "primary_starter")
        if (starter_role, str(starter_image)) not in existing_paths:
            ref_assets.insert(
                0,
                ReferenceAssetItem(
                    asset_id="starter_0",
                    role=starter_role,  # type: ignore[arg-type]
                    path=str(starter_image),
                    subject_id="primary",
                    view_orientation="composition",
                    source_provenance="local",
                    locked=True,
                ),
            )
            existing_paths.add((starter_role, str(starter_image)))
    if subject_refs is not None:
        for idx, sref in enumerate(subject_refs.references):
            if (sref.role, sref.path) in existing_paths:
                continue
            ref_assets.append(
                ReferenceAssetItem(
                    asset_id=sref.ref_id or f"sref_{idx}",
                    role=sref.role,
                    path=sref.path,
                    subject_id=sref.subject_id,
                    view_orientation=sref.resolved_orientation(),
                    source_provenance=sref.source_provenance,
                    asset_fingerprint=sref.asset_fingerprint,
                    preserve_identity=sref.preserve_identity,
                    preserve_appearance=sref.preserve_appearance,
                    preserve_wardrobe=sref.preserve_wardrobe,
                    locked_attributes=list(sref.locked_attributes),
                    locked=sref.locked,
                )
            )
            existing_paths.add((sref.role, sref.path))
    base_spec.reference_assets = ref_assets

    # Extract explicit deterministic requirements (always marked user_explicit & locked).
    reqs: list[RequirementItem] = list(base_spec.required_attributes)
    seen_req_desc = {r.description.lower() for r in reqs}

    def _add_req(cat: str, desc: str) -> None:
        key = desc.strip().lower()
        if key and key not in seen_req_desc:
            seen_req_desc.add(key)
            reqs.append(
                RequirementItem(
                    id=f"req_{len(reqs) + 1}_{cat}",
                    category=cat,
                    description=desc.strip(),
                    provenance="user_explicit",
                    locked=True,
                )
            )

    typography_targets: list[str] = []
    for m in _QUOTED_TEXT_RE.finditer(clean_prompt):
        quoted = (m.group(1) or m.group(2) or "").strip()
        if quoted and quoted not in typography_targets:
            typography_targets.append(quoted)
            _add_req("typography", f'Exact visible text: "{quoted}"')

    for m in _COUNT_RE.finditer(clean_prompt):
        raw_qty, obj = m.group(1).strip().lower(), m.group(2).strip()
        if raw_qty.startswith("pair"):
            qty_num = 2
        elif raw_qty.isdigit():
            qty_num = int(raw_qty)
        else:
            qty_num = _SPELLED_COUNTS.get(raw_qty, 0)
        # Trim trailing stop-words from greedy 3-word object span (e.g. "mugs on the" -> "mugs")
        obj_tokens = obj.split()
        while obj_tokens and obj_tokens[-1].lower() in {"on", "in", "at", "of", "to", "with", "and", "or", "the", "a", "an", "by", "from", "for"}:
            obj_tokens.pop()
        obj_clean = " ".join(obj_tokens)
        first_word = obj_tokens[0].lower() if obj_tokens else ""
        if qty_num > 0 and first_word and first_word not in _NON_OBJECT_UNITS and first_word not in {"of", "more", "less", "other", "another"}:
            _add_req("object_quantities", f"Exact count: {qty_num} {obj_clean}")

    spatial_added = False
    for m in _SPATIAL_REL_RE.finditer(clean_prompt):
        subj_a = " ".join(m.group(1).split())
        rel = " ".join(m.group(2).split())
        subj_b = " ".join(m.group(3).split())
        _add_req("spatial_relationships", f"Spatial placement: {subj_a} {rel} {subj_b}")
        spatial_added = True
    for m in _SPATIAL_SIDE_RE.finditer(clean_prompt):
        subj_a = " ".join(m.group(1).split())
        side = " ".join(m.group(2).split())
        _add_req("spatial_relationships", f"Spatial placement: {subj_a} {side}")
        spatial_added = True
    if not spatial_added:
        for m in _SPATIAL_RE.finditer(clean_prompt):
            # Capture local context around bare spatial preposition so operands are never lost!
            start = max(0, m.start() - 32)
            end = min(len(clean_prompt), m.end() + 32)
            snippet = clean_prompt[start:end].strip(" ,.;")
            _add_req("spatial_relationships", f"Spatial placement: {snippet}")

    exclusions = list(base_spec.excluded_attributes)
    for m in _EXCLUSION_RE.finditer(clean_prompt):
        term = m.group(1).strip(" .,:;")
        if term and len(term) >= 2 and term.lower() not in {e.lower() for e in exclusions}:
            exclusions.append(term)
            _add_req("exclusion", f"No {term}")
    base_spec.excluded_attributes = exclusions

    # RC-07: Preserve explicit user prompt requirements even when no narrow regex matched (never leave 0 semantic requirements!).
    if clean_prompt and not reqs:
        _add_req("complex_nl", f"Explicit prompt fidelity: {clean_prompt[:220]}")

    # Populate empty non-protected creative fields from explicit prompt phrases.
    optional_prefs: list[str] = []
    inferred_defaults: list[str] = []
    for regex, field_name in (
        (_LIGHTING_RE, "lighting"),
        (_COMPOSITION_RE, "composition"),
        (_CAMERA_MOVE_RE, "camera_movement"),
    ):
        match = regex.search(clean_prompt)
        if match:
            if field_name == "camera_movement" and not is_video:
                continue
            attr: PromptAttribute = getattr(base_spec, field_name)
            if not attr.is_protected and not attr.value.strip():
                setattr(
                    base_spec,
                    field_name,
                    PromptAttribute(value=match.group(1), state="specified", provenance="user_explicit"),
                )

    if scenario in {"rear_facing", "left_profile", "right_profile", "side_profile", "turn_or_orbit"}:
        _add_req("orientation", f"Requested subject orientation/motion: {scenario.replace('_', ' ')}")

    base_spec.required_attributes = reqs

    # Apply constraints cleanly without silently changing explicit settings.
    c_updates: dict[str, Any] = {"local_only": True}
    if model_preference and model_preference != "auto":
        c_updates["model_preference"] = model_preference
    if width is not None:
        c_updates["width"] = int(width)
    elif is_video and base_spec.constraints.width == 896:
        c_updates["width"] = 480
        inferred_defaults.append("Default vertical Wan video width=480")
    if height is not None:
        c_updates["height"] = int(height)
    elif is_video and base_spec.constraints.height == 1280:
        c_updates["height"] = 832
        inferred_defaults.append("Default vertical Wan video height=832")
    if frames is not None:
        c_updates["frames"] = int(frames)
    elif is_video and base_spec.constraints.frames <= 1:
        c_updates["frames"] = 81
        inferred_defaults.append("Default Wan 5-second frame count=81 (16 fps)")
    if seed is not None:
        c_updates["seed"] = int(seed)
    base_spec.constraints = base_spec.constraints.model_copy(update=c_updates)

    # Optional local Ollama LLM semantic assist (untrusted output, strict merge, deterministic fallback).
    trace = InterpretationTrace(mode="deterministic", detected_intent=detected, scenario=scenario)
    warnings: list[str] = []

    if _should_invoke_llm(clean_prompt, use_llm=use_llm):
        sched = llm_schedule or choose_llm_execution_schedule(
            comfy_models_warm=comfy_models_warm,
            active_generation_running=active_generation_running,
            snapshot=snapshot,
        )
        trace.llm_schedule_mode = sched.mode
        client = llm_client or LocalOllamaClient(base_url=ollama_url)
        installed = client.list_models() if ollama_model is None else []
        chosen_name = ollama_model or (
            m.name if (m := select_intelligence_model(installed)) is not None else ""
        )
        if not chosen_name or sched.mode == "disabled":
            trace.mode = "deterministic_fallback"
            trace.fallback_reason = sched.reason if sched.mode == "disabled" else "No local Ollama text model available."
            warnings.append(f"Used deterministic interpretation ({trace.fallback_reason}).")
        else:
            trace.llm_invoked = True
            trace.llm_model = chosen_name
            llm_res = client.complete_json(
                model=chosen_name,
                system_prompt=_LLM_SYSTEM_PROMPT,
                user_prompt=(
                    f"User prompt: {clean_prompt}\n"
                    f"Has starter image: {has_starter}\n"
                    f"Supplemental reference roles: {[r.role for r in (subject_refs.references if subject_refs else [])]}\n"
                    "Return JSON only."
                ),
                schedule=sched,
            )
            trace.llm_latency_ms = llm_res.latency_ms
            if not llm_res.ok or not isinstance(llm_res.data, dict):
                trace.mode = "deterministic_fallback"
                trace.fallback_reason = llm_res.error or "Invalid local LLM response."
                warnings.append(f"Local LLM unavailable/invalid ({trace.fallback_reason}); used deterministic fallback.")
            else:
                trace.mode = "llm_assisted"
                _merge_untrusted_llm_output(base_spec, llm_res.data, is_video=is_video, trace=trace)

    # Final policy & conflict validation.
    conflicts = base_spec.detect_conflicts()
    blocking = [c.message for c in conflicts if c.severity == "error"]
    for c in conflicts:
        if c.severity == "warning":
            warnings.append(c.message)

    return InterpretedRequest(
        ok=not blocking,
        detected_intent=detected,
        spec=base_spec,
        expected_orientation=scenario,
        typography_targets=typography_targets,
        mandatory_requirements=[r.model_dump() for r in base_spec.required_attributes],
        optional_preferences=optional_prefs,
        inferred_defaults=inferred_defaults,
        trace=trace,
        warnings=warnings,
        error=blocking[0] if blocking else "",
    )


def _merge_untrusted_llm_output(
    spec: StructuredPromptSpec,
    data: dict[str, Any],
    *,
    is_video: bool,
    trace: InterpretationTrace,
) -> None:
    """Safely merge untrusted LLM JSON fields into spec without ever overriding explicit user intent."""
    # Reject any attempt to smuggle cloud models, shell ops, or minor phrasing.
    for field_name in CREATIVE_FIELD_NAMES:
        raw_val = data.get(field_name)
        if not isinstance(raw_val, str) or not raw_val.strip():
            continue
        clean = " ".join(raw_val.split())[:180]
        if not is_video and field_name in {"motion", "camera_movement"}:
            trace.rejected_llm_suggestions.append(f"{field_name}: skipped on still-image intent")
            continue
        if depicts_minor(clean):
            trace.rejected_llm_suggestions.append(f"{field_name}: blocked by adult-only policy")
            continue
        if any(tok in clean.lower() for tok in ("runway", "http://", "https://", "rm -rf", "subprocess", "eval(")):
            trace.rejected_llm_suggestions.append(f"{field_name}: rejected unsafe/external directive")
            continue

        existing: PromptAttribute = getattr(spec, field_name)
        if existing.is_protected or existing.value.strip():
            trace.rejected_llm_suggestions.append(f"{field_name}: preserved explicit/populated value")
            continue

        # Test tentatively for semantic contradictions before accepting.
        trial = spec.model_copy(deep=True)
        setattr(trial, field_name, PromptAttribute(value=clean, state="automatic", provenance="local_llm"))
        if len(trial.detect_conflicts()) > len(spec.detect_conflicts()):
            trace.rejected_llm_suggestions.append(f"{field_name}: skipped contradictory suggestion '{clean}'")
            continue

        # Store as non-mandatory EnhancementItem so explicit user wording always stays authoritative.
        spec.optional_enhancements.append(
            EnhancementItem(field=field_name, value=clean, provenance="local_llm", priority=2)
        )
        trace.added_fields.append(field_name)

    for excl in data.get("excluded_attributes") or []:
        if not isinstance(excl, str) or not excl.strip():
            continue
        term = " ".join(excl.split())[:60]
        if term.lower() in spec.original_prompt.lower():
            trace.rejected_llm_suggestions.append(f"excluded '{term}': contradicts original prompt")
            continue
        if term.lower() not in {e.lower() for e in spec.excluded_attributes}:
            spec.excluded_attributes.append(term)
            trace.added_fields.append(f"exclude:{term}")
