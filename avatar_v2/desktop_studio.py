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
from . import cloud_ledger, local_engine
from .composer import build_job
from .integrity import sha256_file
from .library import AvatarProfile
from .policy import depicts_minor
from .presenter import compile_presenter_prompt
from .providers import runway as runway_api
from .render_queue import QueuedRender
from .secret_store import secret_status
from .voiceover import segment_script

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
TALK_PART_CHARS = 4500  # Runway avatar videos take up to 5,000 characters; longer scripts become parts
MOVE_ENGINES = ("local", "best", "fast")  # your GPU, Runway Gen-4.5, Runway Gen-4 Turbo


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
            quality = options.get("engine") if options.get("engine") in MOVE_MODELS else options.get("quality")
            runway = {
                "task": "image_to_video",
                "model": MOVE_MODELS[_choice(quality, MOVE_MODELS, "best")],
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

    def _studio_local_move(self, options: dict[str, Any]) -> QueuedRender:
        """A "Moving clip" rendered on this computer's GPU (Wan 2.2 14B through ComfyUI; free)."""
        info = local_engine.inspect(self._settings())
        if not info["installed"] or info["missing_models"]:
            raise ValueError(
                "Your GPU engine isn't installed yet. Run scripts/setup_local_engine.sh, or pick a Runway option."
            )
        master, _sha = self._studio_master()
        image = str(options.get("image") or "").strip() or master
        if not image:
            raise ValueError("Pick your presenter photo first (step 1).")
        motion = " ".join(str(options.get("motion") or "").split())[:600] or DEFAULT_MOTION
        phrase = depicts_minor(motion)
        if phrase:
            raise ValueError(f'The presenter has to be an adult (18+). Remove "{phrase}" from the description.')
        preset = self.brand_instructor_preset()
        avatar_payload = {**preset["avatar"], "identity_refs": [], "hair_refs": [], "wardrobe_refs": []}
        shot_payload = {
            **preset["shot"],
            "shot_id": "studio-move-local",
            "engine_preference": "wan22",
            "duration_s": local_engine.SECONDS,
            "fps": local_engine.FPS,
            "user_prompt": motion,
            "wind": {},
            "references": {"init_image": image},
        }
        avatar, shot = self._models_from_payload({"avatar": avatar_payload, "shot": shot_payload})
        job = build_job(avatar, shot)
        prompt, _dropped = compile_presenter_prompt(avatar, shot, budget=900, motion=True)
        width, height = local_engine.SHAPES[_choice(options.get("shape"), local_engine.SHAPES, "tall")]
        job.prompt = prompt
        job.asset_map.update(
            {
                "PROMPT": prompt,
                "INIT_IMAGE": image,
                "WAN_NEGATIVE": local_engine.NEGATIVE,
                "WAN_WIDTH": width,
                "WAN_HEIGHT": height,
                "WAN_LENGTH": local_engine.FRAMES,
                "WAN_SEED": secrets.randbelow(2_000_000_000) + 1,
                "WAN_PREFIX": f"avatar_v2/studio_move_{job.job_id[:12]}",
                **local_engine.model_placeholders(),
            }
        )
        voiceover = ""
        if str(options.get("narration_id") or "").strip():
            voiceover = self._narration_file({"voiceover_id": str(options["narration_id"]).strip()}, job)
        runtime = {
            "provider": "comfyui",
            "profile": local_engine.PROFILE,
            "engine": "wan22_i2v_14b_gguf",
            "model": local_engine.ENGINE_LABEL,
            "steps": 4,
            "cfg": 1.0,
            "width": width,
            "height": height,
            "frames": local_engine.FRAMES,
            "studio_action": "move",
            "estimate_credits": 0,
            "reason": "Moving clip on your GPU (free)",
        }
        if voiceover:
            runtime.update({"studio_voiceover": voiceover, "candidate_qa": True})
        return QueuedRender(
            job=job,
            workflow=str(local_engine.WORKFLOW),
            provider_config=self._provider_config({}),
            profile=local_engine.PROFILE,
            runtime=runtime,
            archive_dir=str(self._archive_dir()),
            label=f"Motion clip · {local_engine.SECONDS} s · your GPU",
        )

    def _studio_payloads(self, options: dict[str, Any]) -> list[dict[str, Any]]:
        """Usually one job; a typed talking-video script longer than Runway's limit becomes parts."""
        script = str(options.get("script") or "").strip()
        if options.get("action") == "talk" and not options.get("narration_id") and len(script) > TALK_PART_CHARS:
            parts: list[str] = []
            for piece in segment_script(script, TALK_PART_CHARS):  # pack paragraphs into as few videos as fit
                if parts and len(parts[-1]) + 2 + len(piece) <= TALK_PART_CHARS:
                    parts[-1] = f"{parts[-1]}\n\n{piece}"
                else:
                    parts.append(piece)
            payloads = []
            for index, part in enumerate(parts, start=1):
                payload = self._studio_payload({**options, "script": part})
                payload["label"] = f"Talking video · part {index} of {len(parts)}"
                payloads.append(payload)
            return payloads
        return [self._studio_payload(options)]

    # ------------------------------------------------------------------ actions

    def studio_price(self, options: dict[str, Any]) -> dict[str, Any]:
        """What a Studio action would cost (local estimate, no network, nothing queued)."""
        options = dict(options or {})
        if options.get("action") == "move" and options.get("engine") == "local":
            try:
                self._studio_local_move(options)
            except (ValueError, OSError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "action": "move", "credits": 0, "usd": 0.0, "upper_bound": False, "local": True,
                    "minutes": local_engine.ESTIMATED_MINUTES, "warnings": [], "limit_error": None, "needs_talker": False}
        try:
            payloads = self._studio_payloads(options)
        except (ValueError, runway_api.RunwayError) as exc:
            return {"ok": False, "error": str(exc)}
        total, warnings, limit_error, upper = 0.0, [], None, False
        for payload in payloads:
            estimate = self.runway_estimate(payload)
            if not estimate.get("ok"):
                return {"ok": False, "error": estimate.get("error")}
            total += estimate["estimate"]["credits"]
            upper = upper or bool(estimate["estimate"].get("upper_bound"))
            warnings.extend(w for w in estimate.get("warnings") or [] if w not in warnings)
            limit_error = limit_error or estimate.get("limit_error")
        return {
            "ok": True,
            "action": payloads[0]["studio_action"],
            "credits": total,
            "usd": round(total * runway_api.CREDIT_USD, 2),
            "upper_bound": upper,
            "parts": len(payloads),
            "warnings": warnings,
            "limit_error": limit_error,
            "needs_talker": bool(payloads[0]["runway"].get("avatar_preset")),
        }

    def studio_go(self, options: dict[str, Any]) -> dict[str, Any]:
        """Start a Studio action right away (no confirmation step)."""
        options = dict(options or {})
        action = str(options.get("action") or "")
        if action == "move" and options.get("engine") == "local":
            engine = self.engine_status()
            if engine["state"] != "ready":
                if engine["state"] in ("stopped", "error"):
                    self.engine_autostart()
                return {"ok": False, "error": engine["message"] + " Try again when it says ready."}
            try:
                item = self._studio_local_move(options)
            except (ValueError, OSError) as exc:
                return {"ok": False, "error": str(exc)}
            queue_id = self.queue.add(item)
            return {"ok": True, "action": action, "queue_id": queue_id, "queued": [{"queue_id": queue_id}],
                    "total_credits": 0, "local": True, "minutes": local_engine.ESTIMATED_MINUTES}
        if action in ("voiceover", "sample"):
            self._studio_remember_voice(self._studio_voice(options.get("voice")))
        try:
            payloads = self._studio_payloads(options)
        except (ValueError, runway_api.RunwayError) as exc:
            return {"ok": False, "error": str(exc)}
        payload = payloads[0]
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
        if len(payloads) == 1:
            return {**self.queue_runway(payload), "action": action}
        queued, total = [], 0.0
        for part in payloads:  # long talking-video scripts: one job per part, in order
            result = self.queue_runway(part)
            if not result.get("ok"):
                return {**result, "action": action, "queued": queued, "total_credits": total,
                        "error": f"{result.get('error')} ({len(queued)} of {len(payloads)} parts were started.)"}
            queued.extend(result.get("queued") or [])
            total += float(result.get("total_credits") or 0)
        return {"ok": True, "action": action, "queue_id": queued[0]["queue_id"], "queued": queued,
                "total_credits": total, "parts": len(payloads)}

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

    def _studio_mux_voiceover(self, item: Any) -> dict[str, Any]:
        """After a local clip renders, add her voiceover as its sound track (the silent original is kept)."""
        voiceover = Path(str(item.runtime.get("studio_voiceover") or ""))
        video = next((Path(o) for o in item.outputs if Path(o).suffix.lower() in VIDEO_SUFFIXES and Path(o).is_file()), None)
        if video is None or not voiceover.is_file():
            return {"ok": False, "voiceover": "skipped: video or voiceover file missing"}
        try:
            voiced = runway_api.replace_audio_track(video, "voiceover", voiceover)
        except runway_api.RunwayError as exc:
            return {"ok": False, "voiceover": f"not added: {exc}"}
        item.outputs.insert(0, str(voiced))
        item.output_hashes[str(voiced)] = sha256_file(voiced)
        return {"ok": True, "voiceover": "added", "file": voiced.name}

    def engine_status(self) -> dict[str, Any]:
        """The local GPU engine (ComfyUI + Wan 2.2 14B): installed, starting, ready or stopped."""
        info = local_engine.inspect(self._settings())
        gpu: dict[str, Any] = {}
        status_fn = getattr(getattr(self, "queue", None), "memory_status", None)
        if status_fn is not None:
            try:
                gpus = (status_fn() or {}).get("gpus") or []
                if gpus:
                    gpu = {"name": gpus[0].get("name"), "vram_gb": round(float(gpus[0].get("total_mb") or 0) / 1024, 1),
                           "used_gb": round(float(gpus[0].get("used_mb") or 0) / 1024, 1)}
            except Exception:  # nvidia-smi missing or failing: report no GPU details
                gpu = {}
        if not info["installed"]:
            state, message = "not_installed", "Your GPU engine isn't installed (run scripts/setup_local_engine.sh)."
        elif info["missing_models"]:
            state, message = "missing_models", f"Your GPU engine is missing {len(info['missing_models'])} model file(s); run scripts/setup_local_engine.sh again."
        else:
            comfy = self.comfyui_status()
            state = str(comfy.get("state") or "STOPPED").lower()
            message = {
                "ready": "Your GPU engine is ready.",
                "starting": "Your GPU engine is starting (about 30 seconds).",
                "stopped": "Your GPU engine is stopped.",
                "error": "Your GPU engine stopped unexpectedly; it restarts when you try again.",
            }.get(state, "Your GPU engine is " + state + ".")
        return {"ok": True, "state": state, "message": message, "installed": info["installed"],
                "missing_models": info["missing_models"], "gpu": gpu, "engine": local_engine.ENGINE_LABEL,
                "minutes": local_engine.ESTIMATED_MINUTES, "seconds": local_engine.SECONDS}

    def engine_autostart(self) -> dict[str, Any]:
        """Start ComfyUI when the engine is installed and not already running (called when the app opens)."""
        status = self.engine_status()
        if status["state"] in ("stopped", "error"):
            return self.comfyui_start()
        return {"ok": True, "state": status["state"]}

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

    def studio_library(self, kind: str = "all") -> dict[str, Any]:
        """Everything made with Studio and Runway, newest first, one tile per file."""
        kind = kind if kind in ("all", "image", "video", "audio") else "all"
        tiles: list[dict[str, Any]] = []
        for result in self._studio_results(limit=400, include_failures=False):
            for output in result["outputs"]:
                if kind != "all" and output["kind"] != kind:
                    continue
                tiles.append({**output, "id": result["id"], "label": result["label"], "action": result["action"],
                              "finished_at": result["finished_at"], "credits": result["credits"],
                              "local": result["local"]})
        return {"ok": True, "kind": kind, "items": tiles[:400]}

    def _studio_results(self, limit: int = 24, include_failures: bool = True) -> list[dict[str, Any]]:
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
            if not isinstance(item, dict):
                continue
            local = item.get("provider") == "comfyui" and bool(item.get("studio_action"))
            if item.get("provider") != "runway" and not local:
                continue
            outputs = [
                {"path": str(output), "kind": _kind(str(output)), "name": Path(str(output)).name}
                for output in item.get("outputs") or []
                if isinstance(output, str) and Path(output).is_file()
            ]
            finished = item.get("finished_at")
            finished_dt = datetime.fromtimestamp(float(finished), timezone.utc) if finished else None
            failed = item.get("status") in {"error", "cancelled"}
            if failed and not include_failures:
                continue
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
                    "credits": 0 if local else receipt.get("credits_charged"),
                    "charge_status": "none" if local else receipt.get("charge_status"),
                    "local": local,
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
