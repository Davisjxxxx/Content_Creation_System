from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

LocationKind = Literal["indoor", "outdoor", "studio"]
AspectKind = Literal["vertical", "landscape", "square"]

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


def _validate_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError("id must be 2-64 chars of lowercase letters, digits, '-' or '_'")
    return value


class AvatarProfile(BaseModel):
    avatar_id: str
    display_name: str
    subject_kind: str = "synthetic"
    age_verified_18_plus: bool = True
    consent_confirmed: bool = True
    identity_refs: dict[str, str] = Field(default_factory=dict)
    body_refs: dict[str, str] = Field(default_factory=dict)
    hair_refs: dict[str, str] = Field(default_factory=dict)
    wardrobe_refs: dict[str, str] = Field(default_factory=dict)
    persistent_features: list[str] = Field(default_factory=list)
    notes: str = ""

    _check_id = field_validator("avatar_id")(_validate_id)

    @property
    def ordered_identity(self) -> list[str]:
        order = ("front", "three_quarter_left", "three_quarter_right", "profile_left", "profile_right")
        ordered = [self.identity_refs[key] for key in order if self.identity_refs.get(key)]
        ordered += [value for key, value in self.identity_refs.items() if key not in order and value]
        return ordered


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
        return {
            "avatar_id": profile.avatar_id,
            "display_name": profile.display_name,
            "subject_kind": profile.subject_kind,
            "age_verified_18_plus": profile.age_verified_18_plus,
            "consent_confirmed": profile.consent_confirmed,
            "identity_refs": profile.ordered_identity,
            "body_refs": [value for value in profile.body_refs.values() if value],
            "hair_refs": [value for value in profile.hair_refs.values() if value],
            "wardrobe_refs": [value for value in profile.wardrobe_refs.values() if value],
            "persistent_features": profile.persistent_features,
        }


def history_manifest(item: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Build a durable run manifest for one finished queue item."""
    now = datetime.now(timezone.utc).isoformat()
    manifest = dict(item)
    manifest["run_id"] = f"run-{item.get('id', 'unknown')}-{now[:19].replace(':', '')}"
    manifest["saved_at"] = now
    manifest["runs_dir"] = str(runs_dir)
    return manifest
