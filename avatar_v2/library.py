from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import SubjectKind

LocationKind = Literal["indoor", "outdoor", "studio"]
AspectKind = Literal["vertical", "landscape", "square"]
AnatomyRegion = Literal[
    "full_body",
    "torso_front",
    "torso_rear",
    "pelvis_front",
    "pelvis_three_quarter",
    "pelvis_rear",
    "hands",
    "feet",
]
GuideSourceKind = Literal[
    "synthetic",
    "medical_illustration",
    "licensed_adult_photo",
    "consenting_adult_photo",
]
AppearanceComponentCategory = Literal[
    "face_shape",
    "body_type",
    "height",
    "proportions",
    "torso",
    "buttocks",
    "pelvis",
    "hands",
    "feet",
    "skin",
]

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


def _validate_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError("id must be 2-64 chars of lowercase letters, digits, '-' or '_'")
    return value


class AvatarProfile(BaseModel):
    avatar_id: str
    display_name: str
    subject_kind: SubjectKind = "unknown"
    apparent_age_years: int | None = Field(default=None, ge=18, le=100)
    age_verified_18_plus: bool = False
    consent_confirmed: bool = False
    identity_refs: dict[str, str] = Field(default_factory=dict)
    body_refs: dict[str, str] = Field(default_factory=dict)
    hair_refs: dict[str, str] = Field(default_factory=dict)
    wardrobe_refs: dict[str, str] = Field(default_factory=dict)
    anatomy_refs: dict[str, str] = Field(default_factory=dict)
    anatomy_guide_ids: list[str] = Field(default_factory=list, max_length=8)
    component_option_ids: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(default_factory=list)
    appearance_manifest: dict[str, str] = Field(default_factory=dict)
    designer_iteration: int = Field(default=0, ge=0)
    persistent_features: list[str] = Field(default_factory=list)
    notes: str = ""

    _check_id = field_validator("avatar_id")(_validate_id)

    @field_validator("anatomy_guide_ids")
    @classmethod
    def _validate_guide_ids(cls, values: list[str]) -> list[str]:
        output: list[str] = []
        for value in values:
            clean = _validate_id(str(value))
            if clean not in output:
                output.append(clean)
        return output

    @field_validator("component_option_ids")
    @classmethod
    def _validate_component_ids(cls, values: list[str]) -> list[str]:
        return cls._validate_guide_ids(values)

    @property
    def ordered_identity(self) -> list[str]:
        order = ("front", "three_quarter_left", "three_quarter_right", "profile_left", "profile_right")
        ordered = [self.identity_refs[key] for key in order if self.identity_refs.get(key)]
        ordered += [value for key, value in self.identity_refs.items() if key not in order and value]
        return ordered


class AnatomyGuide(BaseModel):
    guide_id: str
    label: str
    path: str
    region: AnatomyRegion
    source_kind: GuideSourceKind
    provenance: str = Field(min_length=3, max_length=1000)
    adult_subject_confirmed: bool
    rights_confirmed: bool
    consent_or_release_confirmed: bool = False
    notes: str = Field(default="", max_length=1000)

    _check_id = field_validator("guide_id")(_validate_id)

    @field_validator("label", "path", "provenance")
    @classmethod
    def _required_text(cls, value: str) -> str:
        clean = str(value).strip()
        if not clean:
            raise ValueError("field must not be empty")
        return clean

    @field_validator("adult_subject_confirmed")
    @classmethod
    def _require_adult(cls, value: bool) -> bool:
        if not value:
            raise ValueError("anatomy guides require confirmed adult subject matter")
        return value

    @field_validator("rights_confirmed")
    @classmethod
    def _require_rights(cls, value: bool) -> bool:
        if not value:
            raise ValueError("anatomy guides require confirmed usage rights")
        return value

    @model_validator(mode="after")
    def _require_photo_consent(self) -> "AnatomyGuide":
        if self.source_kind in {"licensed_adult_photo", "consenting_adult_photo"} and not self.consent_or_release_confirmed:
            raise ValueError("real-person anatomy guides require consent or model-release confirmation")
        return self


class AppearanceComponentOption(BaseModel):
    option_id: str
    label: str
    category: AppearanceComponentCategory
    paths: list[str] = Field(min_length=1, max_length=8)
    description: str = Field(min_length=3, max_length=1000)
    provenance: str = Field(min_length=3, max_length=1000)
    source_kind: GuideSourceKind
    adult_subject_confirmed: bool
    rights_confirmed: bool
    consent_or_release_confirmed: bool = False
    minimum_examples: int = Field(default=3, ge=2, le=8)

    _check_id = field_validator("option_id")(_validate_id)

    @model_validator(mode="after")
    def _validate_authority(self) -> "AppearanceComponentOption":
        if not self.adult_subject_confirmed or not self.rights_confirmed:
            raise ValueError("appearance components require confirmed adult subject matter and usage rights")
        if self.source_kind in {"licensed_adult_photo", "consenting_adult_photo"} and not self.consent_or_release_confirmed:
            raise ValueError("real-person appearance components require consent or model-release confirmation")
        return self


class MotionAsset(BaseModel):
    motion_id: str
    label: str
    category: str = "custom"
    path: str
    description: str = ""
    role: str = "motion and camera choreography"

    _check_id = field_validator("motion_id")(_validate_id)


class SceneAsset(BaseModel):
    scene_id: str
    label: str
    path: str
    location: LocationKind = "outdoor"
    environment: str = ""
    lighting: str = ""
    time_of_day: str = ""
    camera_style: str = ""
    aspect: AspectKind = "vertical"

    _check_id = field_validator("scene_id")(_validate_id)


class LibraryStore:
    """Local-first JSON stores for avatars, motion and scene references.

    Everything lives under ~/.avatar_v2/library and never leaves the machine.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / name

    def _read(self, name: str) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._path(name).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, name: str, data: dict[str, dict[str, Any]]) -> None:
        self._path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _crud(self, name: str, model: type[BaseModel], entry_id: str) -> None:
        pass  # placeholder to satisfy coverage of future generic CRUD

    def avatars(self) -> list[dict[str, Any]]:
        return [entry for entry in self._read("vault.json").values()]

    def get_avatar(self, avatar_id: str) -> dict[str, Any] | None:
        return self._read("vault.json").get(avatar_id)

    def put_avatar(self, entry: dict[str, Any]) -> dict[str, Any]:
        profile = AvatarProfile.model_validate(entry)
        data = self._read("vault.json")
        data[profile.avatar_id] = profile.model_dump()
        self._write("vault.json", data)
        return data[profile.avatar_id]

    def delete_avatar(self, avatar_id: str) -> bool:
        data = self._read("vault.json")
        if avatar_id not in data:
            return False
        del data[avatar_id]
        self._write("vault.json", data)
        return True

    def anatomy_guides(self) -> list[dict[str, Any]]:
        return [entry for entry in self._read("anatomy_guides.json").values()]

    def get_anatomy_guide(self, guide_id: str) -> dict[str, Any] | None:
        return self._read("anatomy_guides.json").get(guide_id)

    def put_anatomy_guide(self, entry: dict[str, Any]) -> dict[str, Any]:
        guide = AnatomyGuide.model_validate(entry)
        path = Path(guide.path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("anatomy guide image does not exist")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            raise ValueError("anatomy guide must be a supported image file")
        data = self._read("anatomy_guides.json")
        data[guide.guide_id] = guide.model_copy(update={"path": str(path)}).model_dump()
        self._write("anatomy_guides.json", data)
        return data[guide.guide_id]

    def delete_anatomy_guide(self, guide_id: str) -> bool:
        data = self._read("anatomy_guides.json")
        if guide_id not in data:
            return False
        del data[guide_id]
        self._write("anatomy_guides.json", data)
        avatars = self._read("vault.json")
        changed = False
        for avatar in avatars.values():
            selected = list(avatar.get("anatomy_guide_ids") or [])
            if guide_id in selected:
                avatar["anatomy_guide_ids"] = [item for item in selected if item != guide_id]
                changed = True
        if changed:
            self._write("vault.json", avatars)
        return True

    def selected_anatomy_guides(self, guide_ids: list[str]) -> list[dict[str, Any]]:
        data = self._read("anatomy_guides.json")
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for guide_id in guide_ids:
            if guide_id in seen or guide_id not in data:
                continue
            seen.add(guide_id)
            selected.append(AnatomyGuide.model_validate(data[guide_id]).model_dump())
        return selected

    def appearance_components(self) -> list[dict[str, Any]]:
        return [entry for entry in self._read("appearance_components.json").values()]

    def put_appearance_component(self, entry: dict[str, Any]) -> dict[str, Any]:
        option = AppearanceComponentOption.model_validate(entry)
        resolved: list[str] = []
        for raw in option.paths:
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"appearance component image does not exist: {path.name}")
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                raise ValueError("appearance component examples must be still images")
            value = str(path)
            if value not in resolved:
                resolved.append(value)
        data = self._read("appearance_components.json")
        saved = option.model_copy(update={"paths": resolved}).model_dump()
        saved["ready"] = len(resolved) >= option.minimum_examples
        data[option.option_id] = saved
        self._write("appearance_components.json", data)
        return saved

    def delete_appearance_component(self, option_id: str) -> bool:
        data = self._read("appearance_components.json")
        if option_id not in data:
            return False
        del data[option_id]
        self._write("appearance_components.json", data)
        avatars = self._read("vault.json")
        changed = False
        for avatar in avatars.values():
            selected = list(avatar.get("component_option_ids") or [])
            if option_id in selected:
                avatar["component_option_ids"] = [item for item in selected if item != option_id]
                changed = True
        if changed:
            self._write("vault.json", avatars)
        return True

    def selected_appearance_components(self, option_ids: list[str]) -> list[dict[str, Any]]:
        data = self._read("appearance_components.json")
        return [data[option_id] for option_id in option_ids if option_id in data and data[option_id].get("ready")]

    def motions(self) -> list[dict[str, Any]]:
        return [entry for entry in self._read("motions.json").values()]

    def put_motion(self, entry: dict[str, Any]) -> dict[str, Any]:
        asset = MotionAsset.model_validate(entry)
        data = self._read("motions.json")
        data[asset.motion_id] = asset.model_dump()
        self._write("motions.json", data)
        return data[asset.motion_id]

    def delete_motion(self, motion_id: str) -> bool:
        data = self._read("motions.json")
        if motion_id not in data:
            return False
        del data[motion_id]
        self._write("motions.json", data)
        return True

    def scenes(self) -> list[dict[str, Any]]:
        return [entry for entry in self._read("scenes.json").values()]

    def put_scene(self, entry: dict[str, Any]) -> dict[str, Any]:
        asset = SceneAsset.model_validate(entry)
        data = self._read("scenes.json")
        data[asset.scene_id] = asset.model_dump()
        self._write("scenes.json", data)
        return data[asset.scene_id]

    def delete_scene(self, scene_id: str) -> bool:
        data = self._read("scenes.json")
        if scene_id not in data:
            return False
        del data[scene_id]
        self._write("scenes.json", data)
        return True

    def profile_to_avatar_payload(self, avatar_id: str) -> dict[str, Any] | None:
        entry = self.get_avatar(avatar_id)
        if not entry:
            return None
        profile = AvatarProfile.model_validate(entry)
        guides = self.selected_anatomy_guides(profile.anatomy_guide_ids)
        components = self.selected_appearance_components(profile.component_option_ids)
        return {
            "avatar_id": profile.avatar_id,
            "display_name": profile.display_name,
            "subject_kind": profile.subject_kind,
            "apparent_age_years": profile.apparent_age_years,
            "age_verified_18_plus": profile.age_verified_18_plus,
            "consent_confirmed": profile.consent_confirmed,
            "identity_refs": profile.ordered_identity,
            "body_refs": [
                value
                for value in (
                    *profile.body_refs.values(),
                    *profile.anatomy_refs.values(),
                    *(path for component in components for path in component.get("paths") or []),
                )
                if value
            ][:8],
            "hair_refs": [value for value in profile.hair_refs.values() if value],
            "wardrobe_refs": [value for value in profile.wardrobe_refs.values() if value],
            "anatomy_guides": [
                {
                    "guide_id": guide["guide_id"],
                    "label": guide["label"],
                    "path": guide["path"],
                    "region": guide["region"],
                }
                for guide in guides
            ],
            "persistent_features": [
                *profile.persistent_features,
                *[
                    f"selected {component['category'].replace('_', ' ')}: {component['label']} - {component['description']}"
                    for component in components
                ],
                *[
                    f"{key.replace('_', ' ')}: {value}"
                    for key, value in profile.appearance_manifest.items()
                    if str(value).strip()
                ],
            ],
        }


def history_manifest(item: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Build a durable run manifest for one finished queue item."""
    now = datetime.now(timezone.utc).isoformat()
    manifest = dict(item)
    manifest["run_id"] = f"run-{item.get('id', 'unknown')}-{now[:19].replace(':', '')}"
    manifest["saved_at"] = now
    manifest["runs_dir"] = str(runs_dir)
    return manifest
