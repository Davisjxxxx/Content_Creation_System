"""Studio: the app's simple start screen for making the BRAND presenter.

Each Studio action turns a few plain choices into a normal Runway job and queues it through the
same pipeline as the advanced screens (``_runway_build`` → ``queue_runway``), so pricing, uploads,
cost records and outputs behave identically. Jobs start on one click; the price is shown on the
button before you click. Like the presenter mixin, nothing here is stored as a public attribute,
because pywebview would expose it to page JavaScript.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import secrets
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from . import brand_instructor as bi
from . import cloud_ledger
from .integrity import sha256_file
from .library import AvatarProfile
from .providers import runway as runway_api
from .secret_store import secret_status

STUDIO_ACTIONS = ("presenter", "angles", "voiceover", "sample", "talk", "act", "move", "sharpen")
PHOTO_RATIO = "1080:1440"  # 3:4 portrait, 1080p tier (8 credits per Gen-4 image)
VIDEO_SHAPES = {"tall": "720:1280", "portrait": "832:1104"}  # phone 9:16, or the photo's own 3:4
MOVE_MODELS = {"best": "gen4.5", "fast": "gen4_turbo"}
SAMPLE_LINE = "Hi! I'm your instructor. Let's learn something new together today."
DEFAULT_MOTION = "She smiles warmly, nods, and starts explaining something with a small, natural hand gesture."
DEFAULT_TTS_VOICE = "Maya"
# Runway's avatar voices that suit the presenter; any of runway_api.VOICE_PRESETS is accepted.
TALKER_VOICES = (
    "clara", "emma", "victoria", "maya", "nina", "georgia", "ruby", "summer", "skye", "mia", "aurora",
    "petra", "violet", "luna",
)
TALKER_NAME = "BRAND Instructor"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
RECENT_FAILURE_WINDOW = timedelta(hours=6)


def _desktop():
    from . import desktop  # late import: desktop imports this module

    return desktop


def _kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "file"


def _angle_slots() -> list[dict[str, Any]]:
    return [slot for slot in bi.REFERENCE_SLOTS if not slot.get("master")]


def _slot_label(slot_id: str) -> str:
    try:
        return str(bi.slot_by_id(slot_id)["label"])
    except ValueError:
        return slot_id


def _choice(value: Any, allowed: dict[str, Any] | tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


class StudioMixin:
    # ------------------------------------------------------------------ helpers

    def _studio_profile(self) -> AvatarProfile | None:
        """The BRAND presenter's vault profile, created on first use (local only)."""
        library = self._library_or_none()
        if library is None:
            return None
        entry = library.get_avatar(bi.PROFILE_ID)
        if not entry:
            self.brand_instructor_install()
            entry = library.get_avatar(bi.PROFILE_ID)
        return AvatarProfile.model_validate(entry) if entry else None

    def _studio_master(self, profile: AvatarProfile | None = None) -> tuple[str | None, str | None]:
        profile = profile or self._studio_profile()
        if profile is None:
            return None, None
        path = profile.identity_refs.get("front")
        if not path or not Path(path).is_file():
            return None, None
        return str(path), profile.reference_hashes.get("front") or sha256_file(path)

    def _studio_talker(self, master_sha: str | None) -> dict[str, Any] | None:
        """The Runway avatar made from the current master photo, if one exists."""
        if not master_sha:
            return None
        matches = [
            {"id": avatar_id, **entry}
            for avatar_id, entry in self._runway_avatar_index().items()
            if isinstance(entry, dict)
            and entry.get("vault_avatar_id") == bi.PROFILE_ID
            and entry.get("source_sha256") == master_sha
        ]
        matches.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return matches[0] if matches else None

    def _studio_voice(self, requested: Any = None) -> str:
        voice = str(requested or "").strip()
        if voice and runway_api._TTS_PRESET_RE.match(voice):
            return voice
        profile = self._voiceovers().get_profile(bi.VOICE_PROFILE["id"]) or {}
        saved = str((profile.get("voice") or {}).get("presetId") or "")
        return saved if runway_api._TTS_PRESET_RE.match(saved) else DEFAULT_TTS_VOICE

    def _studio_remember_voice(self, voice: str) -> None:
        store = self._voiceovers()
        profile = store.get_profile(bi.VOICE_PROFILE["id"]) or dict(bi.VOICE_PROFILE)
        if (profile.get("voice") or {}) != {"type": "runway-preset", "presetId": voice}:
            profile["voice"] = {"type": "runway-preset", "presetId": voice}
            store.save_profile(profile)

    def _studio_payload(self, options: dict[str, Any]) -> dict[str, Any]:
        """Map a Studio action and its simple choices onto a regular Runway job payload."""
        action = str(options.get("action") or "")
        if action not in STUDIO_ACTIONS:
            raise ValueError(f"Unknown Studio action '{action}'.")
        settings = self._settings()
        preset = self.brand_instructor_preset()
        avatar = dict(preset["avatar"])
        shot: dict[str, Any] = {
            **preset["shot"],
            "shot_id": f"studio-{action}",
            "duration_s": 6,
            "engine_preference": "runway",
            "wind": {},
            "references": {},
        }
        look = _choice(options.get("look") or settings.get("studio_look"), bi.LOOKS, bi.DEFAULT_LOOK)
        extra = options.get("extra") if options.get("extra") is not None else settings.get("studio_extra")
        shot["user_prompt"] = bi.look_prompt(look, str(extra or ""))
        master, master_sha = self._studio_master()
        narration = str(options.get("narration_id") or "").strip()
        shape = VIDEO_SHAPES[_choice(options.get("shape"), VIDEO_SHAPES, "tall")]
        runway: dict[str, Any]

        def need_master() -> str:
            if not master:
                raise ValueError("Pick your presenter photo first (step 1).")
            return master

        if action == "presenter":
            avatar["identity_refs"] = []  # design a new face from the description alone
            count = max(1, min(int(options.get("count") or 4), 8))
            shot["seed"] = secrets.randbelow(2_000_000_000) + 1  # fresh options every time
            runway = {"task": "still", "model": "gen4_image", "ratio": PHOTO_RATIO, "slot": "identity_front"}
            if count > 1:
                runway["candidates"] = count
            label = "New presenter option"
        elif action == "angles":
            need_master()
            allowed = {slot["id"] for slot in _angle_slots()}
            slots = [str(item) for item in options.get("slots") or [] if str(item) in allowed]
            if not slots:
                raise ValueError("Tick at least one kind of photo.")
            # Her face comes from the master photo; extra design notes would crowd out the angle direction.
            shot["user_prompt"] = bi.look_prompt(look)
            runway = {"task": "variations", "model": "gen4_image", "ratio": PHOTO_RATIO, "variations": slots}
            label = "Photo of her"
        elif action in ("voiceover", "sample"):
            voice = self._studio_voice(options.get("voice"))
            script = SAMPLE_LINE if action == "sample" else str(options.get("script") or "").strip()
            if not script:
                raise ValueError("Type what she should say.")
            runway = {
                "task": "narration",
                "model": "eleven_v4",
                "script": script,
                "voice_profile_id": bi.VOICE_PROFILE["id"],
                "tts_voice": voice,
            }
            label = f"{'Voice sample' if action == 'sample' else 'Voiceover'} · {voice}"
        elif action == "talk":
            need_master()
            script = str(options.get("script") or "").strip()
            if not narration and not script:
                raise ValueError("Type what she should say, or pick one of your voiceovers.")
            runway = {
                "task": "avatar_video",
                "speech": "audio" if narration else "text",
                "script": script,
                "voiceover_id": narration,
            }
            talker = self._studio_talker(master_sha)
            if talker:
                runway["avatar_id"] = talker["id"]
            else:
                # Price only: the talking avatar is set up first (studio_go does that), then this job runs.
                runway["avatar_preset"] = runway_api.AVATAR_PRESETS[0]
            label = "Talking video"
        elif action == "act":
            character = need_master()
            video = str(options.get("video") or "").strip()
            if not video:
                raise ValueError("Choose the video of you acting it out.")
            shot["references"] = {"init_image": character, "motion_video": video}
            runway = {
                "task": "performance",
                "model": "act_two",
                "ratio": shape,
                "body_control": options.get("body", True) is not False,
                "expression_intensity": 3,
                "output_audio": "voiceover" if narration else "mute",
                "voiceover_id": narration,
            }
            label = "Act-it-out video"
        elif action == "move":
            image = str(options.get("image") or "").strip() or need_master()
            seconds = 10 if int(options.get("seconds") or 5) >= 10 else 5
            shot["user_prompt"] = " ".join(str(options.get("motion") or "").split())[:600] or DEFAULT_MOTION
            shot["references"] = {"init_image": image}
            runway = {
                "task": "image_to_video",
                "model": MOVE_MODELS[_choice(options.get("quality"), MOVE_MODELS, "best")],
                "ratio": shape,
                "duration": seconds,
            }
            if narration:
                runway.update({"output_audio": "voiceover", "voiceover_id": narration})
            label = f"Motion clip · {seconds} s"
        else:  # sharpen
            image = str(options.get("image") or "").strip() or need_master()
            runway = {"task": "upscale", "image": image, "scale_factor": 2, "flavor": "photo"}
            label = "Sharper photo"
        return {
            "settings": {},
            "avatar": avatar,
            "shot": shot,
            "runway": runway,
            "label": label,
            "studio_action": action,
        }

    # ------------------------------------------------------------------ actions

    def studio_price(self, options: dict[str, Any]) -> dict[str, Any]:
        """What a Studio action would cost (local estimate, no network, nothing queued)."""
        try:
            payload = self._studio_payload(dict(options or {}))
        except (ValueError, runway_api.RunwayError) as exc:
            return {"ok": False, "error": str(exc)}
        estimate = self.runway_estimate(payload)
        if not estimate.get("ok"):
            return {"ok": False, "error": estimate.get("error")}
        return {
            "ok": True,
            "action": payload["studio_action"],
            "credits": estimate["estimate"]["credits"],
            "usd": estimate["estimate"]["usd"],
            "upper_bound": bool(estimate["estimate"].get("upper_bound")),
            "basis": estimate["estimate"].get("basis"),
            "warnings": estimate.get("warnings") or [],
            "limit_error": estimate.get("limit_error"),
            "needs_talker": bool(payload["runway"].get("avatar_preset")),
        }

    def studio_go(self, options: dict[str, Any]) -> dict[str, Any]:
        """Start a Studio action right away (no confirmation step)."""
        options = dict(options or {})
        action = str(options.get("action") or "")
        if action in ("voiceover", "sample"):
            self._studio_remember_voice(self._studio_voice(options.get("voice")))
        try:
            payload = self._studio_payload(options)
        except (ValueError, runway_api.RunwayError) as exc:
            return {"ok": False, "error": str(exc)}
        if payload["runway"].get("avatar_preset"):
            setup = self.studio_setup_talker({"voice": options.get("talker_voice")})
            if not setup.get("ok"):
                return setup
            return {
                "ok": False,
                "setup_started": True,
                "talker": setup.get("talker"),
                "message": "Setting up her talking avatar on Runway (usually about a minute). "
                "The video starts automatically when it's ready.",
            }
        if action == "presenter":
            self.save_settings(
                {
                    "studio_look": _choice(options.get("look"), bi.LOOKS, bi.DEFAULT_LOOK),
                    "studio_extra": " ".join(str(options.get("extra") or "").split())[:300],
                }
            )
        result = self.queue_runway(payload)
        return {**result, "action": action}

    def studio_pick(self, options: dict[str, Any]) -> dict[str, Any]:
        """Keep (or toss) a generated photo. Keeping a presenter option makes it her master photo."""
        keep = (options or {}).get("keep", True) is not False
        self._studio_profile()  # make sure the profile exists
        return self.avatar_candidate_decide(
            {
                "avatar_id": bi.PROFILE_ID,
                "slot_id": str((options or {}).get("slot_id") or "identity_front"),
                "decision": "accepted" if keep else "rejected",
                "path": str((options or {}).get("path") or ""),
            }
        )

    def studio_setup_talker(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create her Runway talking avatar from the master photo (once per master photo)."""
        master, master_sha = self._studio_master()
        if not master:
            return {"ok": False, "error": "Pick your presenter photo first (step 1)."}
        existing = self._studio_talker(master_sha)
        if existing:
            return {"ok": True, "existing": True, "talker": {"id": existing["id"], "voice": existing.get("voice")}}
        voice = str((options or {}).get("voice") or "").strip() or runway_api.DEFAULT_AVATAR_VOICE
        avatar = {**self.brand_instructor_preset()["avatar"], "identity_refs": [master]}
        created = self.runway_avatar_create(
            {"avatar": avatar, "reference_image": master, "name": TALKER_NAME, "voice": voice}
        )
        if not created.get("ok"):
            return created
        return {"ok": True, "existing": False, "talker": {**created["avatar"], "voice": voice}}

    def studio_talker_status(self) -> dict[str, Any]:
        """Whether her talking avatar exists and is ready (one free Runway status check)."""
        _master, master_sha = self._studio_master()
        talker = self._studio_talker(master_sha)
        if not talker:
            return {"ok": True, "talker": None}
        status = self.runway_avatar_status(talker["id"])
        if not status.get("ok"):
            return {"ok": False, "error": status.get("error"), "talker": {"id": talker["id"]}}
        return {"ok": True, "talker": {**status["avatar"], "voice": talker.get("voice")}}

    def studio_thumb(self, path: str) -> dict[str, Any]:
        """Small preview for an image or (first frame of) a video in the Studio results."""
        source = Path(str(path or "")).expanduser()
        kind = _kind(str(source))
        if kind == "image":
            return self.reference_thumbnail(str(source))
        if kind != "video" or not source.is_file():
            return {"ok": False, "error": "No preview for this file."}
        try:
            stat = source.stat()
            key = hashlib.sha1(f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:20]
            cached = _desktop().APP_HOME / "thumbs" / f"{key}.jpg"
            if not cached.is_file():
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    return {"ok": False, "error": "ffmpeg is not installed."}
                cached.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [ffmpeg, "-v", "error", "-y", "-ss", "0.5", "-i", str(source), "-frames:v", "1",
                     "-vf", "scale=320:-2", str(cached)],
                    capture_output=True, timeout=20, check=False,
                )
                if not cached.is_file():  # very short clips: take the first frame
                    subprocess.run(
                        [ffmpeg, "-v", "error", "-y", "-i", str(source), "-frames:v", "1", "-vf", "scale=320:-2",
                         str(cached)],
                        capture_output=True, timeout=20, check=False,
                    )
            with Image.open(cached) as opened:
                preview = opened.convert("RGB")
                buffer = io.BytesIO()
                preview.save(buffer, format="JPEG", quality=78)
            return {"ok": True, "data_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def studio_image(self, path: str) -> dict[str, Any]:
        """Full-size copy of a still for the big viewer, so you can judge it before keeping it."""
        source = Path(str(path or "")).expanduser()
        if _kind(str(source)) != "image" or not source.is_file():
            return {"ok": False, "error": "That photo is no longer on this computer."}
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            if max(width, height) > 2400:  # Runway stills are 1080x1440, so they stay at full size
                image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=93)
            return {
                "ok": True,
                "data_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "width": width,
                "height": height,
                "name": source.name,
            }
        except OSError as exc:
            return {"ok": False, "error": f"Could not open that photo ({type(exc).__name__})."}

    # ------------------------------------------------------------------ state for the Studio screen

    def _studio_results(self, limit: int = 24) -> list[dict[str, Any]]:
        history_dir = _desktop().HISTORY_DIR
        try:
            files = sorted(history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        except OSError:
            return []
        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        for path in files[: limit * 4]:
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict) or item.get("provider") != "runway":
                continue
            outputs = [
                {"path": str(output), "kind": _kind(str(output)), "name": Path(str(output)).name}
                for output in item.get("outputs") or []
                if isinstance(output, str) and Path(output).is_file()
            ]
            finished = item.get("finished_at")
            finished_dt = datetime.fromtimestamp(float(finished), timezone.utc) if finished else None
            failed = item.get("status") in {"error", "cancelled"}
            if failed and (finished_dt is None or now - finished_dt > RECENT_FAILURE_WINDOW):
                continue  # keep the screen about recent work; old failures stay in Pro tools → Results
            if not outputs and not failed:
                continue
            receipt = item.get("cloud_receipt") or {}
            results.append(
                {
                    "id": item.get("id"),
                    "label": item.get("label") or "Runway job",
                    "action": item.get("studio_action"),
                    "status": item.get("status"),
                    "finished_at": finished,
                    "outputs": outputs,
                    "error": item.get("error_message") or (item.get("error") if failed else None),
                    "credits": receipt.get("credits_charged"),
                    "charge_status": receipt.get("charge_status"),
                }
            )
            if len(results) >= limit:
                break
        return results

    def studio_state(self) -> dict[str, Any]:
        """Everything the Studio screen shows, in one local call (no network)."""
        profile = self._studio_profile()
        master, master_sha = self._studio_master(profile)
        settings = self._settings()
        angle_ids = {slot["id"] for slot in _angle_slots()}
        pending_master: list[dict[str, Any]] = []
        pending_photos: list[dict[str, Any]] = []
        photos: list[dict[str, Any]] = []
        if profile is not None:
            for candidate in sorted(profile.build_manifest.candidates, key=lambda c: c.created_at, reverse=True):
                if candidate.status != "pending" or not Path(candidate.path).is_file():
                    continue
                entry = {"path": candidate.path, "slot_id": candidate.slot_id, "label": _slot_label(candidate.slot_id)}
                if candidate.slot_id == "identity_front":
                    pending_master.append(entry)
                elif candidate.slot_id in angle_ids:
                    pending_photos.append(entry)
            for slot in _angle_slots():
                path = profile.identity_refs.get(slot["role"])
                if path and Path(path).is_file():
                    photos.append({"slot_id": slot["id"], "label": slot["label"], "path": path})
        narrations = [
            {
                "id": record["id"],
                "label": record.get("label") or record["id"],
                "duration_s": record.get("duration_s"),
                "path": record.get("path"),
                "script": str(record.get("script") or "")[:160],
                "created_at": record.get("created_at"),
            }
            for record in self._voiceovers().list()
            if record.get("status") != "rejected" and record.get("exists")
        ][:30]
        talker = self._studio_talker(master_sha)
        jobs = [
            {
                "id": item["id"],
                "label": item.get("label"),
                "status": item.get("status"),
                "action": item.get("studio_action"),
                "progress": (item.get("cloud_receipt") or {}).get("progress"),
                "credits": item.get("cloud_estimate"),
            }
            for item in self.queue.list()
            if item.get("status") in {"queued", "running"}
        ]
        spend = cloud_ledger.spend_summary(_desktop().CLOUD_LEDGER_PATH)
        return {
            "ok": True,
            "key": secret_status("runway"),
            "master": master,
            "pending_master": pending_master,
            "photos": photos,
            "pending_photos": pending_photos,
            "angles": [{"id": slot["id"], "label": slot["label"]} for slot in _angle_slots()],
            "looks": [{"id": key, "label": look["label"], "hint": look["hint"]} for key, look in bi.LOOKS.items()],
            "look": _choice(settings.get("studio_look"), bi.LOOKS, bi.DEFAULT_LOOK),
            "extra": str(settings.get("studio_extra") or ""),
            "voice": self._studio_voice(),
            "voices": list(runway_api.TTS_PRESET_SUGGESTIONS),
            "talker": {"id": talker["id"], "voice": talker.get("voice")} if talker else None,
            "talker_voices": list(TALKER_VOICES),
            "default_talker_voice": runway_api.DEFAULT_AVATAR_VOICE,
            "narrations": narrations,
            "jobs": jobs,
            "results": self._studio_results(),
            "spend": {
                "today": round(spend["today_total"], 2),
                "month": round(spend["month_total"], 2),
                "unresolved": spend["unresolved"],
            },
            "credit_usd": runway_api.CREDIT_USD,
        }
