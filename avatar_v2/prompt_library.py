"""Data-driven reusable structured prompt libraries and scene recipes (Sprint 3).

Loads validated creative attribute categories and scene recipes from
``avatar_v2/data/prompt_libraries.json`` (kept strictly separate from execution logic).
Zero local LLM dependency; zero legacy adult/hardcoded content defaults.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .prompt_schema import CREATIVE_FIELD_NAMES, PromptAttribute, StructuredPromptSpec

DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "prompt_libraries.json"

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "composition",
    "subject_attributes",
    "environment",
    "lighting",
    "camera_perspective",
    "focal_length",
    "depth_of_field",
    "visual_style",
    "atmosphere",
    "color_grading",
    "materials_and_textures",
    "expressions",
    "wardrobe",
    "motion",
    "camera_movement",
    "scene_continuity",
)

# Guided UI subset surfaced progressively per task (keeps the frontend minimal and uncluttered).
GUIDED_PHOTO_CATEGORIES: tuple[str, ...] = (
    "composition",
    "environment",
    "lighting",
    "camera_perspective",
    "visual_style",
    "atmosphere",
    "expressions",
    "wardrobe",
)
GUIDED_VIDEO_CATEGORIES: tuple[str, ...] = (
    "motion",
    "camera_movement",
    "lighting",
    "atmosphere",
    "scene_continuity",
)


@lru_cache(maxsize=4)
def load_prompt_library(path: str | Path | None = None) -> dict[str, Any]:
    """Load the structured prompt library JSON from disk."""
    target = Path(path) if path is not None else DEFAULT_LIBRARY_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "categories" not in data or "recipes" not in data:
        raise ValueError(f"Invalid prompt library format in {target}")
    return data


def validate_prompt_library(data: dict[str, Any] | None = None) -> list[str]:
    """Validate library completeness, unique IDs, and conflict-free scene recipes."""
    lib = data if data is not None else load_prompt_library()
    errors: list[str] = []
    categories = lib.get("categories") or {}
    for cat in REQUIRED_CATEGORIES:
        entry = categories.get(cat)
        if not isinstance(entry, dict) or not entry.get("options"):
            errors.append(f"Missing or empty required category: {cat}")
            continue
        seen_ids: set[str] = set()
        for opt in entry["options"]:
            oid = str(opt.get("id") or "")
            if not oid or oid in seen_ids:
                errors.append(f"Duplicate or empty option id '{oid}' in category '{cat}'")
            seen_ids.add(oid)
            if not str(opt.get("value") or "").strip():
                errors.append(f"Empty option value for '{oid}' in category '{cat}'")

    seen_recipes: set[str] = set()
    for recipe in lib.get("recipes") or []:
        rid = str(recipe.get("id") or "")
        if not rid or rid in seen_recipes:
            errors.append(f"Duplicate or empty recipe id '{rid}'")
            continue
        seen_recipes.add(rid)
        attrs = {
            k: v
            for k, v in (recipe.get("attributes") or {}).items()
            if k in CREATIVE_FIELD_NAMES and str(v or "").strip()
        }
        spec = StructuredPromptSpec.from_guided(
            original_prompt=f"Adult presenter portrait ({recipe.get('label', rid)})",
            attributes=attrs,
        )
        conflicts = [c for c in spec.detect_conflicts() if c.severity == "error"]
        if conflicts:
            errors.append(f"Recipe '{rid}' contains conflicting attributes: {[c.message for c in conflicts]}")
    return errors


def guided_ui_payload(task: str = "all") -> dict[str, Any]:
    """Return the curated progressive-disclosure Guided library payload for the desktop UI."""
    lib = load_prompt_library()
    categories = lib.get("categories") or {}
    wanted = (
        GUIDED_PHOTO_CATEGORIES
        if task == "photo"
        else GUIDED_VIDEO_CATEGORIES
        if task == "scene"
        else tuple(dict.fromkeys([*GUIDED_PHOTO_CATEGORIES, *GUIDED_VIDEO_CATEGORIES]))
    )
    out_cats = []
    for cat_id in wanted:
        entry = categories.get(cat_id)
        if not entry:
            continue
        out_cats.append(
            {
                "id": cat_id,
                "label": entry["label"],
                "applies_to": list(entry.get("applies_to") or ["image", "video"]),
                "options": list(entry.get("options") or []),
            }
        )
    return {
        "version": lib.get("version", "1.0"),
        "categories": out_cats,
        "recipes": list(lib.get("recipes") or []),
    }


def build_spec_from_guided_payload(
    *,
    prompt: str,
    task: str,
    guided: dict[str, Any] | None = None,
    model_preference: str = "auto",
    width: int = 896,
    height: int = 1280,
    frames: int = 1,
    references: list[dict[str, Any]] | None = None,
) -> StructuredPromptSpec:
    """Build and deterministically resolve a StructuredPromptSpec from optional Guided inputs.

    Maps extra guided categories (e.g. ``expressions``, ``wardrobe``, ``focal_length``,
    ``depth_of_field``, ``color_grading``, ``materials_and_textures``, ``scene_continuity``)
    into the canonical StructuredPromptSpec fields without losing provenance or lock states.
    """
    lib = load_prompt_library()
    categories = lib.get("categories") or {}
    guided = dict(guided or {})
    selected = dict(guided.get("attributes") or {})
    locked = set(guided.get("locked") or ())
    randomized = set(guided.get("randomized") or ())
    seed = int(guided.get("seed") or 41001)

    # Resolve option IDs to their natural-language values when IDs are supplied.
    resolved_values: dict[str, str] = {}
    rand_pools: dict[str, list[str]] = {}
    for cat_id, cat_data in categories.items():
        opts = cat_data.get("options") or []
        by_id = {str(o["id"]): str(o["value"]) for o in opts}
        if cat_id in randomized and cat_id not in locked:
            rand_pools[cat_id] = [str(o["value"]) for o in opts if str(o.get("value") or "").strip()]
        raw_val = str(selected.get(cat_id) or "").strip()
        if raw_val:
            resolved_values[cat_id] = by_id.get(raw_val, raw_val)

    # Map library category keys onto StructuredPromptSpec's canonical creative fields.
    direct_attrs: dict[str, str] = {}
    direct_pools: dict[str, list[str]] = {}
    direct_locked: set[str] = set()

    for key in CREATIVE_FIELD_NAMES:
        if key in resolved_values:
            direct_attrs[key] = resolved_values[key]
        if key in rand_pools:
            direct_pools[key] = rand_pools[key]
        if key in locked:
            direct_locked.add(key)

    # Merge auxiliary camera/color/material/subject attributes cleanly with provenance.
    def _merge(target: str, aux_keys: tuple[str, ...]) -> None:
        extras = [resolved_values[k] for k in aux_keys if resolved_values.get(k)]
        if extras:
            base = direct_attrs.get(target, "")
            direct_attrs[target] = ", ".join([p for p in [base, *extras] if p])
        if any(k in locked for k in aux_keys):
            direct_locked.add(target)

    _merge("camera_perspective", ("focal_length", "depth_of_field"))
    _merge("colors_and_materials", ("color_grading", "materials_and_textures"))
    _merge("visual_style", ("scene_continuity",))

    from .prompt_schema import GenerationConstraints, ReferenceAssetItem, SubjectAttributeSpec

    subj_kwargs: dict[str, Any] = {}
    for lib_key, subj_field in (("subject_attributes", "description"), ("expressions", "expression"), ("wardrobe", "clothing")):
        if lib_key in resolved_values:
            subj_kwargs[subj_field] = PromptAttribute(
                value=resolved_values[lib_key],
                state="locked" if lib_key in locked else "specified",
                provenance="guided_ui",
            )
        elif lib_key in rand_pools:
            subj_kwargs[subj_field] = PromptAttribute(
                value="",
                state="randomized",
                provenance="guided_ui",
                candidates=rand_pools[lib_key],
            )

    subjects = [SubjectAttributeSpec(id="primary", **subj_kwargs)] if subj_kwargs else []
    ref_items = [ReferenceAssetItem.model_validate(r) for r in (references or [])]

    gen_task = "video_generate" if task in {"scene", "video", "video_generate", "video_scene"} else "image_generate"
    spec = StructuredPromptSpec.from_guided(
        original_prompt=prompt,
        task=gen_task,  # type: ignore[arg-type]
        attributes=direct_attrs,
        locked_fields=direct_locked,
        randomized_fields=direct_pools,
        references=ref_items,
        constraints=GenerationConstraints(
            width=width,
            height=height,
            frames=frames,
            duration_s=(frames / 16.0) if gen_task == "video_generate" and frames > 1 else None,
            model_preference=model_preference,
            local_only=True,
        ),
        seed=seed,
    )
    if subjects:
        spec = spec.model_copy(update={"subjects": subjects})
    if guided.get("randomized"):
        spec = spec.apply_randomization(seed=seed)
    return spec
