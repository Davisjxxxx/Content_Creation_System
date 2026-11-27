"""Role-aware Multi-View Subject Reference Architecture & Reusable Character Profiles (Sprint 5).

Supports optional multi-view subject reference collections alongside a primary starter image:
- Distinguishes the primary starter image (initial composition / video start frame) from
  supplemental subject references (facial identity, front/back/side/3-quarter body views,
  appearance/wardrobe details).
- Supports multiple distinct subjects without accidentally merging reference sets.
- Detects conflicting or incompatible reference assignments explicitly.
- Provides deterministic orientation-aware reference selection (preparing for Sprint 6
  local intelligence without requiring an LLM for straightforward cases).
- Provides lightweight local CharacterProfileStore reusing original paths without
  modifying or duplicating original user assets.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .reference_picker import LocalReferenceManager, validate_local_image

SubjectReferenceRole = Literal[
    "primary_starter",
    "face_identity",
    "front_body",
    "back_body",
    "left_side",
    "right_side",
    "front_three_quarter",
    "rear_three_quarter",
    "appearance_wardrobe",
]

ViewOrientation = Literal[
    "composition",
    "front",
    "back",
    "left",
    "right",
    "front_three_quarter",
    "rear_three_quarter",
    "detail",
]

NON_3D_RECONSTRUCTION_NOTE = (
    "Multiple reference views provide stronger identity, wardrobe, and angle guidance to reduce "
    "invented subject traits, but do not produce an exact 3D geometric reconstruction."
)

ROLE_CATALOG: tuple[dict[str, Any], ...] = (
    {"id": "primary_starter", "label": "Primary starter image", "orientation": "composition", "singular": True, "kind": "starter"},
    {"id": "face_identity", "label": "Facial identity", "orientation": "front", "singular": False, "kind": "subject"},
    {"id": "front_body", "label": "Front body", "orientation": "front", "singular": True, "kind": "subject"},
    {"id": "back_body", "label": "Back body", "orientation": "back", "singular": True, "kind": "subject"},
    {"id": "left_side", "label": "Left side", "orientation": "left", "singular": True, "kind": "subject"},
    {"id": "right_side", "label": "Right side", "orientation": "right", "singular": True, "kind": "subject"},
    {"id": "front_three_quarter", "label": "Front three-quarter", "orientation": "front_three_quarter", "singular": True, "kind": "subject"},
    {"id": "rear_three_quarter", "label": "Rear three-quarter", "orientation": "rear_three_quarter", "singular": True, "kind": "subject"},
    {"id": "appearance_wardrobe", "label": "Appearance / wardrobe", "orientation": "detail", "singular": False, "kind": "subject"},
)

ROLE_BY_ID: dict[str, dict[str, Any]] = {entry["id"]: entry for entry in ROLE_CATALOG}
SUPPLEMENTAL_ROLE_IDS: tuple[str, ...] = tuple(entry["id"] for entry in ROLE_CATALOG if entry["id"] != "primary_starter")


class SubjectReferenceItem(BaseModel):
    """One role-aware reference image attached to a specific subject."""

    ref_id: str = Field(default_factory=lambda: f"sref_{secrets.token_hex(4)}")
    role: SubjectReferenceRole
    subject_id: str = "primary"
    view_orientation: ViewOrientation | None = None
    path: str
    source_provenance: Literal["local", "my_stuff", "character_profile", "script"] = "local"
    asset_fingerprint: str = ""
    width: int = 0
    height: int = 0
    preserve_identity: bool = True
    preserve_appearance: bool = True
    preserve_wardrobe: bool = True
    locked_attributes: list[str] = Field(default_factory=list)
    locked: bool = False
    note: str = ""

    def resolved_orientation(self) -> ViewOrientation:
        if self.view_orientation is not None:
            return self.view_orientation
        return ROLE_BY_ID[self.role]["orientation"]  # type: ignore[return-value]


class ReferenceConflict(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    subject_id: str = ""
    roles: list[str] = Field(default_factory=list)
    message: str


class SubjectReferenceSet(BaseModel):
    """Reusable role-aware reference collection distinguishing starter frame from subject views."""

    starter_image: SubjectReferenceItem | None = None
    references: list[SubjectReferenceItem] = Field(default_factory=list)
    preserve_identity: bool = True
    preserve_appearance: bool = True
    preserve_wardrobe: bool = True
    character_profile_id: str | None = None
    non_3d_disclaimer: str = NON_3D_RECONSTRUCTION_NOTE

    def all_items(self) -> list[SubjectReferenceItem]:
        out: list[SubjectReferenceItem] = []
        if self.starter_image is not None:
            out.append(self.starter_image)
        out.extend(self.references)
        return out

    def by_subject(self) -> dict[str, list[SubjectReferenceItem]]:
        grouped: dict[str, list[SubjectReferenceItem]] = {}
        for item in self.references:
            grouped.setdefault(item.subject_id, []).append(item)
        return grouped

    def subject_ids(self) -> list[str]:
        return sorted(self.by_subject().keys())

    def detect_conflicts(self, *, check_disk: bool = False) -> list[ReferenceConflict]:
        """Detect competing starter frames, duplicate singular views, missing files, and subject mixing."""
        conflicts: list[ReferenceConflict] = []

        # Supplemental references must never be treated as competing primary starter frames.
        extra_starters = [r for r in self.references if r.role == "primary_starter"]
        if extra_starters:
            conflicts.append(
                ReferenceConflict(
                    code="competing_starter_frame",
                    severity="error",
                    roles=["primary_starter"],
                    message=(
                        "Supplemental references cannot use role 'primary_starter'; assign only one "
                        "starter_image and use view roles (front_body, back_body, etc.) for extra views."
                    ),
                )
            )

        # Singular body-view roles should not have conflicting distinct images for the same subject.
        for subj_id, items in self.by_subject().items():
            seen_role_paths: dict[str, set[str]] = {}
            for item in items:
                meta = ROLE_BY_ID.get(item.role) or {}
                if meta.get("singular"):
                    seen_role_paths.setdefault(item.role, set()).add(item.path)
            for role, paths in seen_role_paths.items():
                if len(paths) > 1:
                    conflicts.append(
                        ReferenceConflict(
                            code="duplicate_singular_view",
                            severity="error",
                            subject_id=subj_id,
                            roles=[role],
                            message=(
                                f"Subject '{subj_id}' has {len(paths)} conflicting images assigned to "
                                f"singular role '{role}'."
                            ),
                        )
                    )

            # Same image path assigned to opposite orientations (e.g. front_body AND back_body).
            by_path: dict[str, set[str]] = {}
            for item in items:
                by_path.setdefault(item.path, set()).add(item.role)
            for path, roles in by_path.items():
                if {"front_body", "back_body"} <= roles or {"left_side", "right_side"} <= roles:
                    conflicts.append(
                        ReferenceConflict(
                            code="contradictory_view_assignment",
                            severity="error",
                            subject_id=subj_id,
                            roles=sorted(roles),
                            message=(
                                f"The same image '{Path(path).name}' is assigned to opposite views "
                                f"({', '.join(sorted(roles))}) for subject '{subj_id}'."
                            ),
                        )
                    )

        if check_disk:
            for item in self.all_items():
                checked = validate_local_image(item.path, workflow="multi_reference")
                if not checked.ok:
                    conflicts.append(
                        ReferenceConflict(
                            code="invalid_reference_file",
                            severity="error",
                            subject_id=item.subject_id,
                            roles=[item.role],
                            message=checked.error,
                        )
                    )

        return conflicts

    def stage_all(self, manager: LocalReferenceManager, *, require_authorization: bool = False) -> "SubjectReferenceSet":
        """Validate and stage any external references into controlled storage without mutating originals."""
        staged_starter = None
        if self.starter_image is not None:
            info = manager.stage_for_job(
                self.starter_image.path,
                workflow="multi_reference",
                require_authorization=require_authorization,
            )
            staged_starter = self.starter_image.model_copy(
                update={
                    "path": info["staged_path"],
                    "asset_fingerprint": info["sha256"],
                    "width": info["width"],
                    "height": info["height"],
                }
            )
        staged_refs: list[SubjectReferenceItem] = []
        for item in self.references:
            info = manager.stage_for_job(
                item.path,
                workflow="multi_reference",
                require_authorization=require_authorization,
            )
            staged_refs.append(
                item.model_copy(
                    update={
                        "path": info["staged_path"],
                        "asset_fingerprint": info["sha256"],
                        "width": info["width"],
                        "height": info["height"],
                    }
                )
            )
        return self.model_copy(update={"starter_image": staged_starter, "references": staged_refs})


# --------------------------------------------------------------------------- Phase H: Deterministic Reference Selection

_REAR_PATTERNS = re.compile(
    r"\b(from behind|back view|rear view|seen from the back|facing away|walking away|turned away|her back|his back|their back)\b",
    re.IGNORECASE,
)
_TURN_PATTERNS = re.compile(
    r"\b(turns? around|turning around|spin|spins|rotate|rotates|360|orbit|orbits|walks? past|turning from front to back|changes? angle)\b",
    re.IGNORECASE,
)
_LEFT_PATTERNS = re.compile(r"\b(left profile|left side|from the left|facing left)\b", re.IGNORECASE)
_RIGHT_PATTERNS = re.compile(r"\b(right profile|right side|from the right|facing right)\b", re.IGNORECASE)
_SIDE_PATTERNS = re.compile(r"\b(side profile|profile view|side view|sideways|in profile)\b", re.IGNORECASE)
_PORTRAIT_PATTERNS = re.compile(r"\b(portrait|headshot|close[- ]?up|facing camera|front view|looking at the camera)\b", re.IGNORECASE)
_NEW_ENV_PATTERNS = re.compile(r"\b(new environment|different background|placed in|standing in|outdoors|in a cafe|on a beach|in a studio)\b", re.IGNORECASE)


def infer_reference_scenario(prompt: str, *, explicit_scenario: str | None = None) -> str:
    """Deterministically infer the visual perspective/motion scenario from prompt text (zero LLM)."""
    if explicit_scenario:
        return explicit_scenario
    text = str(prompt or "")
    if _TURN_PATTERNS.search(text):
        return "turn_or_orbit"
    if _REAR_PATTERNS.search(text):
        return "rear_facing"
    if _LEFT_PATTERNS.search(text):
        return "left_profile"
    if _RIGHT_PATTERNS.search(text):
        return "right_profile"
    if _SIDE_PATTERNS.search(text):
        return "side_profile"
    if _NEW_ENV_PATTERNS.search(text):
        return "new_environment"
    if _PORTRAIT_PATTERNS.search(text):
        return "frontal_portrait"
    return "general"


_ROLE_PRIORITY_BY_SCENARIO: dict[str, tuple[str, ...]] = {
    "frontal_portrait": ("face_identity", "front_body", "front_three_quarter", "appearance_wardrobe"),
    "left_profile": ("left_side", "face_identity", "front_three_quarter", "rear_three_quarter", "appearance_wardrobe"),
    "right_profile": ("right_side", "face_identity", "front_three_quarter", "rear_three_quarter", "appearance_wardrobe"),
    "side_profile": ("left_side", "right_side", "face_identity", "front_three_quarter", "appearance_wardrobe"),
    "rear_facing": ("back_body", "rear_three_quarter", "appearance_wardrobe", "left_side", "right_side"),
    "turn_or_orbit": (
        "front_body",
        "back_body",
        "left_side",
        "right_side",
        "front_three_quarter",
        "rear_three_quarter",
        "face_identity",
        "appearance_wardrobe",
    ),
    "new_environment": ("face_identity", "front_body", "appearance_wardrobe", "front_three_quarter", "back_body"),
    "general": (
        "face_identity",
        "front_body",
        "front_three_quarter",
        "appearance_wardrobe",
        "left_side",
        "right_side",
        "back_body",
        "rear_three_quarter",
    ),
}


def select_relevant_references(
    ref_set: SubjectReferenceSet,
    *,
    prompt: str,
    task: str = "image_generate",
    subject_id: str | None = None,
    max_supplemental: int = 4,
    explicit_scenario: str | None = None,
    workflow_supports_multiref: bool = True,
) -> dict[str, Any]:
    """Deterministically select the highest-value references for a generation request.

    - Always keeps locked user references.
    - Never silently drops rear/side references when a workflow cannot directly consume them:
      instead reports explicit ``workflow_limitations`` and ``recommended_strategy``.
    - Scopes strictly to ``subject_id`` when provided so multiple subjects are never blended.
    """
    scenario = infer_reference_scenario(prompt, explicit_scenario=explicit_scenario)
    priority = _ROLE_PRIORITY_BY_SCENARIO.get(scenario, _ROLE_PRIORITY_BY_SCENARIO["general"])

    candidates = [
        r for r in ref_set.references
        if r.role != "primary_starter" and (subject_id is None or r.subject_id == subject_id)
    ]

    # Sort: locked first, then scenario priority order, then original order.
    def _rank(item: SubjectReferenceItem) -> tuple[int, int]:
        lock_rank = 0 if item.locked else 1
        role_rank = priority.index(item.role) if item.role in priority else len(priority)
        return (lock_rank, role_rank)

    ranked = sorted(candidates, key=_rank)
    selected: list[SubjectReferenceItem] = []
    omitted: list[SubjectReferenceItem] = []

    limit = max(0, int(max_supplemental))
    for item in ranked:
        if len(selected) < limit or item.locked:
            selected.append(item)
        else:
            omitted.append(item)

    limitations: list[str] = []
    recommended_strategy = "direct_multiref" if workflow_supports_multiref else "starter_only"

    if not workflow_supports_multiref and selected:
        if task in {"video_generate", "video_scene"}:
            has_rear = any(r.role in {"back_body", "rear_three_quarter"} for r in selected)
            if scenario in {"turn_or_orbit", "rear_facing"} and has_rear:
                recommended_strategy = "flux_keyframe_plus_wan_flf2v"
                limitations.append(
                    "Wan 2.2 I2V does not accept arbitrary multi-view subject refs directly; "
                    "use FLUX.2 multi-reference keyframe synthesis + Wan First-Last-Frame / segmented clips "
                    "so rear/side references guide the turn instead of being discarded."
                )
            else:
                recommended_strategy = "flux_conditioned_starter_then_wan"
                limitations.append(
                    "Wan 2.2 I2V conditions directly on a single start frame; synthesize a FLUX.2 "
                    "reference-conditioned starter image first to carry supplemental views into video."
                )
        else:
            limitations.append(
                f"Active workflow does not consume {len(selected)} supplemental reference(s) directly."
            )

    if omitted:
        limitations.append(
            f"Model reference budget ({limit}) prioritized {', '.join(r.role for r in selected)} "
            f"for scenario '{scenario}'; omitted {', '.join(r.role for r in omitted)}."
        )

    return {
        "scenario": scenario,
        "starter_image": ref_set.starter_image.model_dump() if ref_set.starter_image else None,
        "selected": [r.model_dump() for r in selected],
        "omitted": [r.model_dump() for r in omitted],
        "recommended_strategy": recommended_strategy,
        "workflow_limitations": limitations,
        "non_3d_disclaimer": NON_3D_RECONSTRUCTION_NOTE,
    }


# --------------------------------------------------------------------------- Phase I: Reusable Character Profiles


class CharacterProfile(BaseModel):
    """Lightweight reusable local Character Profile (references originals without duplicating them)."""

    profile_id: str = Field(default_factory=lambda: f"char_{secrets.token_hex(5)}")
    name: str
    reference_set: SubjectReferenceSet = Field(default_factory=SubjectReferenceSet)
    appearance_attributes: dict[str, str] = Field(default_factory=dict)
    locked_attributes: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CharacterProfileStore:
    """Local JSON store for Character Profiles; never mutates or deletes original source assets."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = Path(store_path)

    def _load_raw(self) -> dict[str, dict[str, Any]]:
        from .ownership import read_json_safe

        data = read_json_safe(self.store_path, {}, quarantine=True, expected_type=dict)
        return data if isinstance(data, dict) else {}

    def _save_raw(self, data: dict[str, dict[str, Any]]) -> None:
        from .ownership import write_json_atomic

        write_json_atomic(self.store_path, data)

    def list_profiles(self) -> list[dict[str, Any]]:
        raw = self._load_raw()
        out = list(raw.values())
        out.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return out

    def get_profile(self, profile_id: str) -> CharacterProfile | None:
        entry = self._load_raw().get(str(profile_id))
        return CharacterProfile.model_validate(entry) if entry else None

    def save_profile(self, profile: CharacterProfile | dict[str, Any]) -> CharacterProfile:
        model = profile if isinstance(profile, CharacterProfile) else CharacterProfile.model_validate(profile)
        conflicts = [c for c in model.reference_set.detect_conflicts(check_disk=False) if c.severity == "error"]
        if conflicts:
            raise ValueError(conflicts[0].message)
        raw = self._load_raw()
        existing = raw.get(model.profile_id)
        updated = model.model_copy(
            update={
                "created_at": str(existing.get("created_at")) if isinstance(existing, dict) and existing.get("created_at") else model.created_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        raw[updated.profile_id] = updated.model_dump()
        self._save_raw(raw)
        return updated

    def delete_profile(self, profile_id: str) -> bool:
        """Remove a saved profile record without touching or deleting any original reference files."""
        raw = self._load_raw()
        if str(profile_id) not in raw:
            return False
        raw.pop(str(profile_id), None)
        self._save_raw(raw)
        return True
