"""Safe Static Structured Prompt Extractor for Avatar V2 (Sprint 4, Section 5).

Extracts compatible declarative prompt libraries, attribute categories, scene recipes,
generation profiles, prompt templates, and candidate pools from Python scripts (or companion
JSON/YAML configs) strictly via Python AST literal analysis — NEVER importing or executing
untrusted Python expressions.

Design guarantees:
- Zero eval(), exec(), or importlib execution of user scripts.
- Strips legacy SDXL ``(token:1.4)`` weight syntax cleanly with explicit warnings.
- Filters out policy-prohibited minor references via ``avatar_v2.policy.depicts_minor``.
- Does NOT blindly import legacy adult defaults or unsupported hardcoded parameters.
- Explicitly lists every dynamic / non-literal construct that could not be extracted statically.
- Outputs validated structures compatible with ``StructuredPromptSpec`` and ``prompt_library``.
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .policy import depicts_minor
from .prompt_compilers import strip_sdxl_weights
from .prompt_schema import (
    CREATIVE_FIELD_NAMES,
    GenerationConstraints,
    GenerationIntent,
    PromptAttribute,
    StructuredPromptSpec,
    SubjectAttributeSpec,
)

# Map common script variable/dict-key patterns onto canonical StructuredPromptSpec fields.
_FIELD_ALIASES: dict[str, str] = {
    "composition": "composition",
    "framing": "composition",
    "shot": "composition",
    "shot_type": "composition",
    "camera_angle": "camera_perspective",
    "camera_perspective": "camera_perspective",
    "camera": "camera_perspective",
    "perspective": "camera_perspective",
    "angle": "camera_perspective",
    "environment": "environment",
    "environments": "environment",
    "setting": "environment",
    "settings": "environment",
    "location": "environment",
    "locations": "environment",
    "background": "environment",
    "backgrounds": "environment",
    "lighting": "lighting",
    "lightings": "lighting",
    "light": "lighting",
    "atmosphere": "atmosphere",
    "mood": "atmosphere",
    "moods": "atmosphere",
    "weather": "atmosphere",
    "visual_style": "visual_style",
    "style": "visual_style",
    "styles": "visual_style",
    "aesthetic": "visual_style",
    "colors_and_materials": "colors_and_materials",
    "color": "colors_and_materials",
    "colors": "colors_and_materials",
    "palette": "colors_and_materials",
    "materials": "colors_and_materials",
    "textures": "colors_and_materials",
    "motion": "motion",
    "motions": "motion",
    "action": "motion",
    "actions": "motion",
    "camera_movement": "camera_movement",
    "camera_motion": "camera_movement",
    "subject": "subject_description",
    "subjects": "subject_description",
    "character": "subject_description",
    "characters": "subject_description",
    "appearance": "subject_appearance",
    "hair": "subject_appearance",
    "clothing": "subject_clothing",
    "wardrobe": "subject_clothing",
    "outfit": "subject_clothing",
    "outfits": "subject_clothing",
    "expression": "subject_expression",
    "expressions": "subject_expression",
    "pose": "subject_pose",
    "poses": "subject_pose",
}

# Variable names commonly containing legacy adult/NSFW defaults that should NOT be silently imported.
_SENSITIVE_DEFAULT_TOKENS = (
    "nsfw", "nude", "nudity", "explicit", "fetish", "erotic", "lingerie_explicit",
    "force_age", "age_list", "ages", "underage", "teen",
)

# Common legacy A1111/SDXL quality-tag spam tokens to strip cleanly when migrating.
_LEGACY_TAG_SPAM_RE = re.compile(
    r"\b(masterpiece|best quality|ultra detailed|8k uhd|absurdres|highres|score_9|score_8_up)\b",
    re.IGNORECASE,
)

_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class ExtractedProfile:
    name: str
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None
    model_hint: str = ""
    raw_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRecipe:
    id: str
    label: str
    attributes: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""
    excluded: list[str] = field(default_factory=list)
    source_symbol: str = ""


@dataclass
class ExtractionResult:
    ok: bool
    source_path: str
    categories: dict[str, list[str]] = field(default_factory=dict)
    mapped_fields: dict[str, str] = field(default_factory=dict)
    recipes: list[dict[str, Any]] = field(default_factory=list)
    profiles: list[dict[str, Any]] = field(default_factory=list)
    templates: list[dict[str, Any]] = field(default_factory=list)
    candidate_pools: dict[str, list[str]] = field(default_factory=dict)
    unextractable: list[dict[str, Any]] = field(default_factory=list)
    skipped_legacy: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def has_extractable_data(self) -> bool:
        return bool(self.categories or self.recipes or self.templates or self.candidate_pools)

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_extractable_data"] = self.has_extractable_data
        return data


def _sanitize_phrase(raw: str, warnings: list[str]) -> str | None:
    """Clean a candidate prompt phrase; return None if empty or policy-blocked."""
    text = " ".join(str(raw or "").split()).strip(" ,;")
    if not text:
        return None
    cleaned, had_weights = strip_sdxl_weights(text)
    if had_weights:
        msg = "Stripped legacy SDXL (token:weight) syntax from extracted prompt values."
        if msg not in warnings:
            warnings.append(msg)
    cleaned_no_spam = _LEGACY_TAG_SPAM_RE.sub("", cleaned)
    if cleaned_no_spam != cleaned:
        msg = "Removed legacy SDXL quality-tag spam tokens during extraction."
        if msg not in warnings:
            warnings.append(msg)
        cleaned = re.sub(r"[\s,]{2,}", ", ", cleaned_no_spam).strip(" ,;")
    if not cleaned:
        return None
    if depicts_minor(cleaned):
        msg = "Filtered out extracted prompt value(s) referencing minors (adult-only policy)."
        if msg not in warnings:
            warnings.append(msg)
        return None
    return cleaned


def _safe_eval_node(node: ast.AST, env: dict[str, Any]) -> tuple[bool, Any, str]:
    """Evaluate strictly literal Python AST nodes (and previously resolved literal names).

    NEVER invokes eval/exec or calls arbitrary Python functions.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return True, node.value, ""
        return False, None, f"unsupported constant type {type(node.value).__name__}"

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        ok, val, reason = _safe_eval_node(node.operand, env)
        if ok and isinstance(val, (int, float)) and not isinstance(val, bool):
            return True, (-val if isinstance(node.op, ast.USub) else +val), ""
        return False, None, reason or "non-numeric unary op"

    if isinstance(node, ast.Name):
        if node.id in env:
            return True, env[node.id], ""
        return False, None, f"unresolved symbol '{node.id}'"

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out = []
        for elt in node.elts:
            ok, val, reason = _safe_eval_node(elt, env)
            if not ok:
                return False, None, reason
            out.append(val)
        return True, out, ""

    if isinstance(node, ast.Dict):
        out_dict: dict[Any, Any] = {}
        for k_node, v_node in zip(node.keys, node.values):
            if k_node is None:
                return False, None, "dict unpacking (**expr) cannot be evaluated statically"
            ok_k, k_val, reason_k = _safe_eval_node(k_node, env)
            if not ok_k or not isinstance(k_val, (str, int, float)):
                return False, None, reason_k or "non-literal dict key"
            ok_v, v_val, reason_v = _safe_eval_node(v_node, env)
            if not ok_v:
                return False, None, reason_v
            out_dict[str(k_val)] = v_val
        return True, out_dict, ""

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        ok_l, left, reason_l = _safe_eval_node(node.left, env)
        ok_r, right, reason_r = _safe_eval_node(node.right, env)
        if ok_l and ok_r:
            if isinstance(left, str) and isinstance(right, str):
                return True, left + right, ""
            if isinstance(left, list) and isinstance(right, list):
                return True, left + right, ""
        return False, None, reason_l or reason_r or "unsupported '+' operands"

    if isinstance(node, ast.JoinedStr):
        # Convert simple f-strings like f"{subject} in {environment}" into declarative "{subject} in {environment}" templates.
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue) and isinstance(piece.value, ast.Name):
                parts.append("{" + piece.value.id + "}")
            else:
                return False, None, "dynamic expression inside f-string"
        return True, "".join(parts), ""

    return False, None, f"dynamic {type(node).__name__} expression"


def _infer_canonical_field(symbol: str) -> str | None:
    norm = re.sub(r"[^a-z0-9]+", "_", symbol.lower()).strip("_")
    if norm in _FIELD_ALIASES:
        return _FIELD_ALIASES[norm]
    for token, target in _FIELD_ALIASES.items():
        if norm.endswith("_" + token) or norm.startswith(token + "_"):
            return target
    return None


def _is_sensitive_symbol(symbol: str) -> bool:
    low = symbol.lower()
    return any(tok in low for tok in _SENSITIVE_DEFAULT_TOKENS)


def _looks_like_profile_dict(val: dict[str, Any]) -> bool:
    profile_keys = {"width", "height", "steps", "cfg", "cfg_scale", "seed", "sampler", "model", "frames"}
    lower_keys = {str(k).lower() for k in val}
    return len(lower_keys & profile_keys) >= 2


def extract_prompt_data_from_source(source: str, *, source_path: str = "<memory>") -> ExtractionResult:
    """Statically extract compatible prompt libraries, recipes, profiles, and templates from Python source."""
    result = ExtractionResult(ok=True, source_path=source_path)
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        result.ok = False
        result.error = f"SyntaxError at line {exc.lineno}: {exc.msg}"
        return result

    env: dict[str, Any] = {}
    for node in tree.body:
        targets: list[str] = []
        value_node: ast.AST | None = None
        lineno = getattr(node, "lineno", 0)
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets = [node.target.id]
            value_node = node.value
        else:
            continue

        if not targets or value_node is None:
            continue

        for name in targets:
            if name.startswith("_"):
                continue
            if _is_sensitive_symbol(name):
                result.skipped_legacy.append(
                    {"name": name, "line": lineno, "reason": "Skipped legacy/sensitive default symbol."}
                )
                continue

            ok, val, reason = _safe_eval_node(value_node, env)
            if not ok:
                result.unextractable.append({"name": name, "line": lineno, "reason": reason})
                continue

            env[name] = val
            _classify_extracted_symbol(name, val, result)

    return result


def _classify_extracted_symbol(name: str, val: Any, result: ExtractionResult) -> None:
    low = name.lower()

    # 1. String list -> candidate pool / creative attribute category.
    if isinstance(val, list) and val and all(isinstance(item, str) for item in val):
        cleaned = []
        for item in val:
            phrase = _sanitize_phrase(item, result.warnings)
            if phrase and phrase not in cleaned:
                cleaned.append(phrase)
        if not cleaned:
            return
        canonical = _infer_canonical_field(name)
        if canonical:
            result.categories[name] = cleaned
            result.mapped_fields[name] = canonical
        else:
            result.candidate_pools[name] = cleaned
        return

    # 2. Dict of lists or recipes or generation profiles.
    if isinstance(val, dict) and val:
        if _looks_like_profile_dict(val):
            result.profiles.append(_extract_profile(name, val))
            return

        # Dict mapping names -> profile dicts (e.g. PROFILES = {"portrait": {"width": 896, "height": 1280}})
        if all(isinstance(v, dict) and _looks_like_profile_dict(v) for v in val.values()):
            for prof_name, prof_val in val.items():
                result.profiles.append(_extract_profile(f"{name}.{prof_name}", prof_val))
            return

        # Dict mapping category -> list[str] (e.g. PROMPT_LIBRARY = {"lighting": [...], "environment": [...]})
        if all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in val.values()):
            for sub_key, items in val.items():
                if _is_sensitive_symbol(sub_key):
                    result.skipped_legacy.append(
                        {"name": f"{name}.{sub_key}", "line": 0, "reason": "Skipped legacy/sensitive category."}
                    )
                    continue
                cleaned = [p for x in items if (p := _sanitize_phrase(x, result.warnings))]
                if not cleaned:
                    continue
                canonical = _infer_canonical_field(sub_key)
                if canonical:
                    result.categories[sub_key] = cleaned
                    result.mapped_fields[sub_key] = canonical
                else:
                    result.candidate_pools[f"{name}.{sub_key}"] = cleaned
            return

        # Dict mapping recipe_name -> {attr: value} (scene recipes)
        if any(tok in low for tok in ("recipe", "scene", "preset", "template")) or all(
            isinstance(v, dict) for v in val.values()
        ):
            for rec_id, rec_body in val.items():
                if not isinstance(rec_body, dict) or _is_sensitive_symbol(rec_id):
                    continue
                attrs: dict[str, str] = {}
                template_str = ""
                excluded: list[str] = []
                for k, v in rec_body.items():
                    if isinstance(v, str):
                        if k.lower() in {"prompt", "template", "prompt_template"}:
                            phrase = _sanitize_phrase(v, result.warnings)
                            if phrase:
                                template_str = phrase
                        elif k.lower() in {"negative", "negative_prompt", "exclude"}:
                            for part in v.split(","):
                                p = _sanitize_phrase(part, result.warnings)
                                if p:
                                    excluded.append(p)
                        else:
                            canonical = _infer_canonical_field(k)
                            phrase = _sanitize_phrase(v, result.warnings)
                            if canonical and phrase:
                                attrs[canonical] = phrase
                if attrs or template_str:
                    result.recipes.append(
                        asdict(
                            ExtractedRecipe(
                                id=re.sub(r"[^a-z0-9_]+", "_", rec_id.lower()).strip("_") or "recipe",
                                label=rec_id.replace("_", " ").strip().title(),
                                attributes=attrs,
                                prompt_template=template_str,
                                excluded=excluded,
                                source_symbol=name,
                            )
                        )
                    )
            return

    # 3. Declarative prompt template string with placeholders (e.g. PROMPT_TEMPLATE = "{subject}, {lighting}").
    if isinstance(val, str) and ("template" in low or "prompt" in low):
        fields = _TEMPLATE_FIELD_RE.findall(val)
        cleaned = _sanitize_phrase(val, result.warnings)
        if cleaned and (fields or "template" in low or low in {"prompt", "base_prompt", "default_prompt"}):
            result.templates.append({"name": name, "template": cleaned, "fields": fields})


def _extract_profile(name: str, data: dict[str, Any]) -> dict[str, Any]:
    def _opt_int(k: str) -> int | None:
        v = data.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def _opt_float(k: str) -> float | None:
        v = data.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return asdict(
        ExtractedProfile(
            name=name,
            width=_opt_int("width"),
            height=_opt_int("height"),
            steps=_opt_int("steps"),
            cfg=_opt_float("cfg") or _opt_float("cfg_scale"),
            seed=_opt_int("seed"),
            model_hint=str(data.get("model") or data.get("checkpoint") or ""),
            raw_fields={k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))},
        )
    )


def extract_prompt_data(path: str | Path) -> ExtractionResult:
    """Safely inspect and extract declarative prompt structures from a .py or .json file."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return ExtractionResult(ok=False, source_path=str(file_path), error=f"File not found: {file_path}")
    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ExtractionResult(ok=False, source_path=str(file_path), error=f"Could not read file: {exc}")

    if file_path.suffix.lower() == ".json":
        result = ExtractionResult(ok=True, source_path=str(file_path))
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            result.ok = False
            result.error = f"Invalid JSON: {exc}"
            return result
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if not _is_sensitive_symbol(str(k)):
                    _classify_extracted_symbol(str(k), v, result)
        return result

    return extract_prompt_data_from_source(raw_text, source_path=str(file_path))


def build_specs_from_extraction(
    extraction: ExtractionResult | dict[str, Any],
    *,
    count: int = 1,
    seed: int = 41001,
    task: str = "image_generate",
    model_preference: str | None = None,
    base_prompt: str = "",
) -> list[StructuredPromptSpec]:
    """Deterministically materialize StructuredPromptSpec objects from statically extracted data.

    Allows compatible extracted libraries to drive Avatar V2's native prompt compilers and
    RenderQueue without executing arbitrary Python code.
    """
    data = extraction.public() if isinstance(extraction, ExtractionResult) else dict(extraction)
    categories: dict[str, list[str]] = dict(data.get("categories") or {})
    mapped: dict[str, str] = dict(data.get("mapped_fields") or {})
    recipes: list[dict[str, Any]] = list(data.get("recipes") or [])
    templates: list[dict[str, Any]] = list(data.get("templates") or [])
    pools: dict[str, list[str]] = dict(data.get("candidate_pools") or {})

    n = max(1, min(int(count), 100))
    rng = random.Random(int(seed))
    specs: list[StructuredPromptSpec] = []

    for idx in range(n):
        field_values: dict[str, str] = {}
        subj_fields: dict[str, str] = {}
        excluded: list[str] = []
        prompt_text = base_prompt.strip()

        if recipes:
            recipe = recipes[idx % len(recipes)]
            for k, v in (recipe.get("attributes") or {}).items():
                if k in CREATIVE_FIELD_NAMES:
                    field_values[k] = str(v)
                elif k.startswith("subject_"):
                    subj_fields[k[len("subject_"):]] = str(v)
            if not prompt_text and recipe.get("prompt_template"):
                prompt_text = str(recipe["prompt_template"])
            excluded.extend(str(e) for e in (recipe.get("excluded") or []))

        for cat_name, options in categories.items():
            if not options:
                continue
            chosen = rng.choice(options)
            target = mapped.get(cat_name, "")
            if target in CREATIVE_FIELD_NAMES and target not in field_values:
                field_values[target] = chosen
            elif target.startswith("subject_"):
                sub_key = target[len("subject_"):]
                subj_fields.setdefault(sub_key, chosen)

        if not prompt_text and templates:
            tpl = str(templates[idx % len(templates)].get("template") or "")
            # Fill any {placeholder} from extracted categories/pools without eval().
            def _sub(match: re.Match[str]) -> str:
                key = match.group(1)
                if key in categories and categories[key]:
                    return rng.choice(categories[key])
                if key in pools and pools[key]:
                    return rng.choice(pools[key])
                mapped_key = _infer_canonical_field(key)
                if mapped_key and mapped_key in field_values:
                    return field_values[mapped_key]
                return ""

            prompt_text = re.sub(r"\s{2,}", " ", _TEMPLATE_FIELD_RE.sub(_sub, tpl)).strip(" ,.")

        if not prompt_text and not field_values and not subj_fields and pools:
            sampled = [rng.choice(vals) for vals in pools.values() if vals]
            prompt_text = ", ".join(sampled)

        spec_kwargs: dict[str, Any] = {
            "original_prompt": prompt_text,
            "intent": GenerationIntent(task=task, input_mode="script"),  # type: ignore[arg-type]
            "excluded_attributes": excluded,
            "constraints": GenerationConstraints(
                seed=int(seed) + idx,
                local_only=True,
                model_preference=model_preference or "auto",
            ),
        }
        if subj_fields:
            spec_kwargs["subjects"] = [
                SubjectAttributeSpec(
                    **{
                        k: PromptAttribute(value=v, state="randomized", provenance="script")
                        for k, v in subj_fields.items()
                        if k in {"description", "appearance", "clothing", "expression", "pose"}
                    }
                )
            ]
        for fname, fval in field_values.items():
            spec_kwargs[fname] = PromptAttribute(value=fval, state="randomized", provenance="script")

        specs.append(StructuredPromptSpec(**spec_kwargs))

    return specs
