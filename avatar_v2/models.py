from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator

ContentClass = Literal["general", "adult_nudity", "adult_sexual"]
SubjectKind = Literal["synthetic", "consenting_adult", "unknown"]
EngineName = Literal["auto", "wan22", "ltx25", "seedance", "custom"]


class SubjectSpec(BaseModel):
    id: str
    kind: SubjectKind = "synthetic"
    age_verified_18_plus: bool = False
    consent_confirmed: bool = False


class AvatarManifest(BaseModel):
    avatar_id: str
    display_name: str
    subjects: list[SubjectSpec] = Field(default_factory=list)
    identity_refs: list[str] = Field(default_factory=list)
    body_refs: list[str] = Field(default_factory=list)
    hair_refs: list[str] = Field(default_factory=list)
    wardrobe_refs: list[str] = Field(default_factory=list)
    persistent_features: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_subject_and_identity(self) -> "AvatarManifest":
        if not self.subjects:
            raise ValueError("avatar manifest requires at least one subject")
        if not self.identity_refs:
            raise ValueError("avatar manifest requires at least one identity reference")
        return self


class ActionBeat(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    description: str

    @model_validator(mode="after")
    def validate_range(self) -> "ActionBeat":
        if self.end_s <= self.start_s:
            raise ValueError("action beat end_s must be greater than start_s")
        return self


class CameraSpec(BaseModel):
    framing: str = "medium portrait"
    movement: str = "subtle handheld"
    lens_equivalent_mm: int | None = None
    height: str | None = None


class WindSpec(BaseModel):
    direction: str = "none"
    speed_mps: float = Field(default=0.0, ge=0)
    gust_variation_pct: int = Field(default=0, ge=0, le=100)


class ReferenceSpec(BaseModel):
    init_image: str | None = None
    last_frame: str | None = None
    motion_video: str | None = None
    scene_image: str | None = None
    audio: str | None = None


class ShotSpec(BaseModel):
    shot_id: str
    content_class: ContentClass = "general"
    user_prompt: str
    duration_s: float = Field(default=6.0, gt=0, le=60)
    fps: int = Field(default=24, ge=1, le=60)
    width: int = Field(default=720, ge=256)
    height: int = Field(default=1280, ge=256)
    seed: int = Field(default=1, ge=0)
    engine_preference: EngineName = "auto"
    local_only: bool = False
    action_beats: list[ActionBeat] = Field(default_factory=list)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    wind: WindSpec = Field(default_factory=WindSpec)
    wardrobe: str | None = None
    environment: str | None = None
    negative_prompt: str | None = None
    references: ReferenceSpec = Field(default_factory=ReferenceSpec)

    @model_validator(mode="after")
    def beats_fit_duration(self) -> "ShotSpec":
        for beat in self.action_beats:
            if beat.end_s > self.duration_s:
                raise ValueError(
                    f"action beat ending at {beat.end_s}s exceeds duration {self.duration_s}s"
                )
        return self

    @property
    def frames(self) -> int:
        return max(1, round(self.duration_s * self.fps))


class ProviderConfig(BaseModel):
    provider: Literal["comfyui"] = "comfyui"
    base_url: str = "http://127.0.0.1:8188"
    input_dir: str | None = None
    output_dir: str | None = None
    timeout_s: int = Field(default=900, ge=30)


class RenderJob(BaseModel):
    job_id: str
    avatar: AvatarManifest
    shot: ShotSpec
    selected_engine: str
    prompt: str
    negative_prompt: str
    asset_map: dict[str, str | int | float | None]
