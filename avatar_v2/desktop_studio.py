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
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, ImageOps

from . import brand_instructor as bi
from . import cloud_ledger, local_engine
from . import offline_builder as ob
from .composer import build_job
from .integrity import sha256_file
from .js_bridge import in_js_bridge
from .library import AvatarProfile
from .ownership import classify_media_ownership, write_json_atomic
from .policy import depicts_minor
from .presenter import MOTION_REALISM, NO_TEXT, compile_presenter_prompt
from .prompt_compilers import compile_prompt_for_model
from .prompt_library import build_spec_from_guided_payload, guided_ui_payload
from .providers import runway as runway_api
from .reference_picker import (
    LocalReferenceManager,
    WORKFLOW_REFERENCE_LIMITS,
    default_browse_directory,
    validate_local_image,
)
from .render_queue import QueuedRender
from .script_broker import BrokerSubmitContext
from .script_runner import SandboxUnavailableError, ScriptRunManager
from .secret_store import secret_status
from .subject_references import (
    ROLE_CATALOG,
    CharacterProfile,
    CharacterProfileStore,
    SubjectReferenceItem,
    SubjectReferenceSet,
    select_relevant_references,
)
from .workflow_router import plan_intelligent_generation
from .voiceover import segment_script

STUDIO_ACTIONS = ("presenter", "angles", "voiceover", "sample", "talk", "act", "move", "sharpen", "scene")
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
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}

_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class _LocalMediaServer:
    """Loopback-only HTTP file server with byte-range support for in-app video playback."""

    def __init__(self, max_entries: int = 512) -> None:
        self._lock = threading.Lock()
        self._by_token: dict[str, Path] = {}
        self._by_path: dict[Path, str] = {}
        self._max_entries = max_entries
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_HEAD(self) -> None:
                self._serve(send_body=False)

            def do_GET(self) -> None:
                self._serve(send_body=True)

            def _serve(self, *, send_body: bool) -> None:
                raw_path = urlsplit(self.path).path
                if not raw_path.startswith("/media/"):
                    self._send_empty(404)
                    return
                stem = raw_path[len("/media/"):].split("/", 1)[0].split(".", 1)[0]
                target = server_ref._lookup(stem)
                if target is None or not target.is_file():
                    self._send_empty(404)
                    return
                try:
                    size = target.stat().st_size
                except OSError:
                    self._send_empty(404)
                    return
                mime = _VIDEO_MIME.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                start, end = 0, max(size - 1, 0)
                status = 200
                range_header = (self.headers.get("Range") or "").strip()
                if range_header and size > 0:
                    match = _RANGE_RE.match(range_header)
                    if not match:
                        self._send_empty(416, extra={"Content-Range": f"bytes */{size}"})
                        return
                    raw_start, raw_end = match.groups()
                    if raw_start == "" and raw_end == "":
                        self._send_empty(416, extra={"Content-Range": f"bytes */{size}"})
                        return
                    if raw_start == "":
                        suffix_len = int(raw_end)
                        if suffix_len <= 0:
                            self._send_empty(416, extra={"Content-Range": f"bytes */{size}"})
                            return
                        start = max(size - suffix_len, 0)
                        end = size - 1
                    else:
                        start = int(raw_start)
                        end = int(raw_end) if raw_end != "" else size - 1
                        if start >= size or end < start:
                            self._send_empty(416, extra={"Content-Range": f"bytes */{size}"})
                            return
                        end = min(end, size - 1)
                    status = 206
                length = (end - start + 1) if size > 0 else 0
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", mime)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    if status == 206:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    if not send_body or length == 0:
                        return
                    with target.open("rb") as handle:
                        handle.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = handle.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _send_empty(self, code: int, extra: dict[str, str] | None = None) -> None:
                try:
                    self.send_response(code)
                    self.send_header("Content-Length", "0")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    for key, value in (extra or {}).items():
                        self.send_header(key, value)
                    self.end_headers()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="avatar-v2-media", daemon=True)
        self._thread.start()

    def _lookup(self, token: str) -> Path | None:
        with self._lock:
            return self._by_token.get(token)

    def register(self, path: Path) -> str:
        resolved = path.resolve()
        with self._lock:
            token = self._by_path.get(resolved)
            if token is None:
                token = secrets.token_urlsafe(16)
                self._by_token[token] = resolved
                self._by_path[resolved] = token
                while len(self._by_token) > self._max_entries:
                    oldest_token = next(iter(self._by_token))
                    oldest_path = self._by_token.pop(oldest_token)
                    self._by_path.pop(oldest_path, None)
        return f"http://127.0.0.1:{self.port}/media/{token}{resolved.suffix.lower()}"

    def close(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass


def _probe_video_meta(source: Path) -> dict[str, Any]:
    """Read width, height, duration, and fps with ffprobe when available (best-effort)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
                "-of", "json", str(source),
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        out: dict[str, Any] = {}
        if stream.get("width") and stream.get("height"):
            out["width"] = int(stream["width"])
            out["height"] = int(stream["height"])
        raw_dur = stream.get("duration") or fmt.get("duration")
        if raw_dur is not None:
            try:
                dur = float(raw_dur)
                if dur > 0:
                    out["duration_s"] = round(dur, 3)
            except (TypeError, ValueError):
                pass
        for key in ("avg_frame_rate", "r_frame_rate"):
            rate = str(stream.get(key) or "")
            if "/" in rate:
                num, den = rate.split("/", 1)
                try:
                    n_val, d_val = float(num), float(den)
                    if n_val > 0 and d_val > 0:
                        out["fps"] = round(n_val / d_val, 3)
                        break
                except ValueError:
                    pass
        return out
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return {}
RECENT_FAILURE_WINDOW = timedelta(hours=6)
TALK_PART_CHARS = 4500  # Runway avatar videos take up to 5,000 characters; longer scripts become parts
MOVE_ENGINES = ("local", "best", "fast")  # your GPU, Runway Gen-4.5, Runway Gen-4 Turbo
SCENE_MAX_SHOTS = 12  # each shot is about 5 s; shots continue from the previous shot's last frame (12 ≈ 1 minute)


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

    def _reference_manager(self) -> LocalReferenceManager:
        """Shared LocalReferenceManager guarding external file access and controlled job staging.

        Trusts ONLY app-owned media subdirectories (never raw APP_HOME config/logs/ledger!).
        """
        mgr: LocalReferenceManager | None = getattr(self, "_local_ref_manager", None)
        d = _desktop()
        app_home = d.APP_HOME
        if mgr is None:
            mgr = LocalReferenceManager(staging_dir=app_home / "staged_refs")
            self._local_ref_manager = mgr
        for media_root in (app_home, d.RUNS_DIR, d.IMPORTS_DIR, d.VOICEOVER_DIR, app_home / "thumbs"):
            mgr.add_trusted_root(media_root)
        try:
            mgr.add_trusted_root(self._archive_dir())
        except Exception:
            pass
        try:
            out_dir = str(self._settings().get("comfyui_output") or "").strip()
            if out_dir:
                mgr.add_trusted_root(Path(out_dir))
        except Exception:
            pass
        return mgr

    def _stage_reference_if_external(self, path: str, *, workflow: str = "single_start") -> str:
        """Validate and stage an external reference image for a queued job without mutating the original."""
        cleaned = str(path or "").strip()
        if not cleaned:
            return ""
        mgr = self._reference_manager()
        # Enforce strict native-picker / trusted-root authorization for page-facing JsBridge calls (RC-03).
        staged = mgr.stage_for_job(cleaned, workflow=workflow, require_authorization=in_js_bridge())
        return str(staged["staged_path"])

    def _remember_directory(self, setting_key: str, path: str | Path) -> None:
        try:
            parent = Path(path).expanduser().resolve().parent
            if not parent.is_dir():
                return
            d = _desktop()
            settings = self._settings()
            if settings.get(setting_key) != str(parent):
                settings[setting_key] = str(parent)
                d._write_json(d.CONFIG_PATH, settings)
        except Exception:
            pass

    def pick_local_images(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Shared native local reference-image selector (My Stuff alternative, no import required).

        Prevents unrestricted filesystem access: only paths returned by the native OS file
        dialog (or already inside trusted Avatar V2 roots) are authorized for preview and queueing.
        """
        opts = dict(options or {})
        workflow = str(opts.get("workflow") or "single_start")
        if workflow not in WORKFLOW_REFERENCE_LIMITS:
            workflow = "single_start"
        rule = WORKFLOW_REFERENCE_LIMITS[workflow]
        action = str(opts.get("action") or "browse")
        mgr = self._reference_manager()

        if action == "validate":
            raw_path = str(opts.get("path") or "").strip()
            checked = mgr.authorize_existing(raw_path, workflow=workflow)
            return {
                "ok": checked.ok,
                "item": checked.public() if checked.ok else None,
                "error": checked.error,
            }

        multiple = bool(opts.get("multiple")) and int(rule["max_images"]) > 1
        settings = self._settings()
        start_dir = default_browse_directory(str(settings.get("last_image_dir") or ""))
        selected = self.choose_files("image", multiple=multiple, directory=start_dir)
        if not selected:
            return {"ok": True, "cancelled": True, "items": [], "errors": [], "workflow": workflow}

        self._remember_directory("last_image_dir", selected[0])
        batch = mgr.register_selected(selected, workflow=workflow)
        return {
            "ok": bool(batch["items"]),
            "cancelled": False,
            "workflow": workflow,
            "max_images": batch["max_images"],
            "items": batch["items"],
            "errors": batch["errors"],
            "error": batch["errors"][0] if (not batch["items"] and batch["errors"]) else "",
        }

    # ------------------------------------------------------------------ Character Profiles & Multi-View Refs (Sprint 5)

    def _character_profile_store(self) -> CharacterProfileStore:
        return CharacterProfileStore(_desktop().APP_HOME / "character_profiles.json")

    def character_profiles(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """CRUD for lightweight local Character Profiles (never alters or deletes original assets)."""
        opts = dict(options or {})
        action = str(opts.get("action") or "list").strip().lower()
        store = self._character_profile_store()
        mgr = self._reference_manager()

        if action == "list":
            profiles = store.list_profiles()
            # Re-authorize saved profile reference paths for thumbnail preview on reload.
            for prof in profiles:
                rset = prof.get("reference_set") or {}
                paths = [r.get("path") for r in (rset.get("references") or []) if isinstance(r, dict) and r.get("path")]
                if isinstance(rset.get("starter_image"), dict) and rset["starter_image"].get("path"):
                    paths.append(rset["starter_image"]["path"])
                existing = [p for p in paths if Path(str(p)).is_file()]
                if existing:
                    mgr.register_selected(existing, workflow="multi_reference")
            return {"ok": True, "profiles": profiles}

        if action == "delete":
            pid = str(opts.get("profile_id") or "").strip()
            if not pid:
                return {"ok": False, "error": "Select a profile to remove."}
            removed = store.delete_profile(pid)
            return {"ok": removed, "profiles": store.list_profiles(), "error": "" if removed else "Profile not found."}

        if action == "save":
            name = " ".join(str(opts.get("name") or "").split())[:80]
            if not name:
                return {"ok": False, "error": "Enter a name for this Character Profile."}
            raw_refs = opts.get("references") or []
            starter_path = str(opts.get("starter_image") or "").strip()
            starter_item = None
            if starter_path:
                checked = mgr.authorize_existing(starter_path, workflow="multi_reference")
                if not checked.ok:
                    return {"ok": False, "error": checked.error}
                starter_item = SubjectReferenceItem(
                    role="primary_starter",
                    path=checked.path,
                    source_provenance="my_stuff" if checked.source == "my_stuff" else "local",
                    asset_fingerprint=checked.sha256,
                    width=checked.width,
                    height=checked.height,
                )
            items: list[SubjectReferenceItem] = []
            for entry in raw_refs:
                if not isinstance(entry, dict) or not str(entry.get("path") or "").strip():
                    continue
                checked = mgr.authorize_existing(str(entry["path"]), workflow="multi_reference")
                if not checked.ok:
                    return {"ok": False, "error": checked.error}
                items.append(
                    SubjectReferenceItem(
                        role=str(entry.get("role") or "front_body"),  # type: ignore[arg-type]
                        subject_id=str(entry.get("subject_id") or "primary"),
                        path=checked.path,
                        source_provenance="my_stuff" if checked.source == "my_stuff" else "local",
                        asset_fingerprint=checked.sha256,
                        width=checked.width,
                        height=checked.height,
                        preserve_identity=bool(entry.get("preserve_identity", True)),
                        preserve_appearance=bool(entry.get("preserve_appearance", True)),
                        preserve_wardrobe=bool(entry.get("preserve_wardrobe", True)),
                        locked=bool(entry.get("locked", False)),
                    )
                )
            if not items and starter_item is None:
                return {"ok": False, "error": "Add at least one reference image before saving a Character Profile."}
            ref_set = SubjectReferenceSet(
                starter_image=starter_item,
                references=items,
                preserve_identity=bool(opts.get("preserve_identity", True)),
                preserve_appearance=bool(opts.get("preserve_appearance", True)),
                preserve_wardrobe=bool(opts.get("preserve_wardrobe", True)),
            )
            profile_kwargs: dict[str, Any] = {
                "name": name,
                "reference_set": ref_set,
                "appearance_attributes": dict(opts.get("appearance_attributes") or {}),
                "locked_attributes": list(opts.get("locked_attributes") or []),
            }
            if opts.get("profile_id"):
                profile_kwargs["profile_id"] = str(opts["profile_id"])
            try:
                saved = store.save_profile(CharacterProfile(**profile_kwargs))
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "profile": saved.model_dump(), "profiles": store.list_profiles()}

        return {"ok": False, "error": f"Unknown character_profiles action '{action}'."}

    def _resolve_subject_references(
        self,
        *,
        starter_image: str,
        raw_subject_refs: dict[str, Any] | None,
        prompt: str,
        task: str,
        workflow_supports_multiref: bool,
        stage: bool = False,
    ) -> tuple[list[str], list[str], dict[str, Any] | None]:
        """Validate, detect conflicts, select relevant multi-view references, and optionally stage them."""
        if not isinstance(raw_subject_refs, dict):
            return [], [], None
        entries = [e for e in (raw_subject_refs.get("references") or []) if isinstance(e, dict) and str(e.get("path") or "").strip()]
        if not entries:
            return [], [], None
        mgr = self._reference_manager()
        starter_item = (
            SubjectReferenceItem(role="primary_starter", path=starter_image)
            if starter_image and Path(starter_image).is_file()
            else None
        )
        items: list[SubjectReferenceItem] = []
        for entry in entries:
            items.append(
                SubjectReferenceItem(
                    role=str(entry.get("role") or "front_body"),  # type: ignore[arg-type]
                    subject_id=str(entry.get("subject_id") or "primary"),
                    path=str(entry["path"]),
                    preserve_identity=bool(raw_subject_refs.get("preserve_identity", True)),
                    preserve_appearance=bool(raw_subject_refs.get("preserve_appearance", True)),
                    preserve_wardrobe=bool(raw_subject_refs.get("preserve_wardrobe", True)),
                    locked=bool(entry.get("locked", False)),
                )
            )
        ref_set = SubjectReferenceSet(
            starter_image=starter_item,
            references=items,
            preserve_identity=bool(raw_subject_refs.get("preserve_identity", True)),
            preserve_appearance=bool(raw_subject_refs.get("preserve_appearance", True)),
            preserve_wardrobe=bool(raw_subject_refs.get("preserve_wardrobe", True)),
        )
        conflicts = [c for c in ref_set.detect_conflicts(check_disk=True) if c.severity == "error"]
        if conflicts:
            raise ValueError(conflicts[0].message)
        if stage:
            ref_set = ref_set.stage_all(mgr, require_authorization=in_js_bridge())
        selection = select_relevant_references(
            ref_set,
            prompt=prompt,
            task=task,
            max_supplemental=4,
            workflow_supports_multiref=workflow_supports_multiref,
        )
        selected_paths = [str(r["path"]) for r in selection["selected"]]
        return selected_paths, list(selection["workflow_limitations"]), selection

    # ------------------------------------------------------------------ Script mode (Sprint 4)

    def _script_manager(self) -> ScriptRunManager:
        mgr: ScriptRunManager | None = getattr(self, "_script_run_manager", None)
        comfy_raw = str(self._settings().get("comfyui_dir") or "").strip()
        comfy_path = Path(comfy_raw).expanduser() if comfy_raw else None
        if mgr is None:
            mgr = ScriptRunManager(
                runs_dir=_desktop().APP_HOME / "script_runs",
                submit_fn=self._script_submit_job,
                status_fn=self._script_job_status,
                cancel_fn=self._script_cancel_job,
                reference_manager=self._reference_manager(),
                planner=getattr(self.queue, "resource_planner", None),
                comfyui_dir=comfy_path if (comfy_path and comfy_path.is_dir()) else None,
            )
            self._script_run_manager = mgr
        return mgr

    def _script_submit_job(self, ctx: BrokerSubmitContext) -> str:
        """Route a validated SDK BrokerSubmitContext through existing RenderQueue builders (RC-15)."""
        engine = self.engine_status()
        if engine["state"] != "ready":
            raise RuntimeError(engine["message"] + " Start the GPU engine before running scripts.")
        ver_record = None
        try:
            routed = plan_intelligent_generation(
                ctx.spec.original_prompt or ctx.compiled_positive,
                requested_action="scene" if ctx.model_id.startswith("wan22_") else "photo",
                starter_image=ctx.staged_references[0] if ctx.staged_references else None,
                guided_spec=ctx.spec,
                input_mode="script",
                model_preference=ctx.model_id,
                use_llm=False,
            )
            if routed.ok and routed.verification_record is not None:
                ver_record = routed.verification_record.public()
                ver_record["compiled_prompt"] = ctx.compiled_positive
        except Exception:
            ver_record = None

        if ctx.model_id in {"flux2_4b", "flux2_9b"}:
            photo_model = "9b" if ctx.model_id == "flux2_9b" else "4b"
            item = self._forge_photo_item(
                prompt=ctx.compiled_positive,
                shape="portrait",
                photo_model=photo_model,
                action="script_photo",
                label=ctx.label,
                extra_runtime={
                    "script_run_id": ctx.run_id,
                    "script_item": ctx.item_index,
                    "verification_record": ver_record,
                },
                reference_images=ctx.staged_references,
            )
            if ctx.compiled_negative:
                item.job.asset_map["NEGATIVE_PROMPT"] = ctx.compiled_negative
            if ctx.spec.constraints.width and ctx.spec.constraints.height:
                item.job.asset_map["WIDTH"] = int(ctx.spec.constraints.width)
                item.job.asset_map["HEIGHT"] = int(ctx.spec.constraints.height)
                item.runtime["width"] = int(ctx.spec.constraints.width)
                item.runtime["height"] = int(ctx.spec.constraints.height)
            if ctx.spec.constraints.seed is not None:
                item.job.asset_map["SEED"] = int(ctx.spec.constraints.seed)
                item.runtime["seed"] = int(ctx.spec.constraints.seed)
        elif ctx.model_id in {"wan22_a14b", "wan22_5b"}:
            video_model = "5b" if ctx.model_id == "wan22_5b" else "a14b"
            start_img = ctx.staged_references[0] if ctx.staged_references else ""
            item = self._local_wan_item(
                image=start_img,
                text=ctx.compiled_positive,
                shape="tall",
                label=ctx.label,
                action="script_video",
                video_model=video_model,
                vram_mode="cpu_offload" if ctx.execution_plan.wan_clip_device == "cpu" else "balanced",
                verbatim=True,
            )
            item.runtime.update(
                {
                    "script_run_id": ctx.run_id,
                    "script_item": ctx.item_index,
                    "verification_record": ver_record,
                }
            )
            if ctx.spec.constraints.width and ctx.spec.constraints.height:
                item.job.asset_map["WAN_WIDTH"] = int(ctx.spec.constraints.width)
                item.job.asset_map["WAN_HEIGHT"] = int(ctx.spec.constraints.height)
                item.runtime["width"] = int(ctx.spec.constraints.width)
                item.runtime["height"] = int(ctx.spec.constraints.height)
            if ctx.spec.constraints.frames:
                item.job.asset_map["WAN_LENGTH"] = int(ctx.spec.constraints.frames)
                item.runtime["frames"] = int(ctx.spec.constraints.frames)
            if ctx.spec.constraints.seed is not None:
                item.job.asset_map["WAN_SEED"] = int(ctx.spec.constraints.seed)
                item.runtime["seed"] = int(ctx.spec.constraints.seed)
        else:
            raise ValueError(f"Unsupported or deferred Script model '{ctx.model_id}'; refusing silent model substitution.")
        return self.queue.add(item)

    def _script_job_status(self, job_id: str) -> dict[str, Any]:
        for entry in self.queue.list():
            if entry.get("id") == job_id:
                return {
                    "status": entry.get("status") or "queued",
                    "outputs": list(entry.get("outputs") or []),
                    "error": entry.get("error_message") or entry.get("error") or "",
                }
        hist = _desktop().HISTORY_DIR / f"{job_id}.json"
        if hist.is_file():
            try:
                data = json.loads(hist.read_text(encoding="utf-8"))
                return {
                    "status": data.get("status") or "done",
                    "outputs": list(data.get("outputs") or []),
                    "error": data.get("error_message") or data.get("error") or "",
                }
            except (OSError, json.JSONDecodeError):
                pass
        return {"status": "unknown", "outputs": [], "error": ""}

    def _script_cancel_job(self, job_id: str) -> bool:
        try:
            return bool(self.queue.cancel(job_id))
        except Exception:
            return False

    def script_select(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Select a local Python script via native dialog and statically inspect it (NEVER executes)."""
        opts = dict(options or {})
        settings = self._settings()
        start_dir = default_browse_directory(str(settings.get("last_script_dir") or ""))
        selected = self.choose_files("python", multiple=False, directory=start_dir)
        if not selected:
            return {"ok": True, "cancelled": True, "executed": False}
        chosen = selected[0]
        self._remember_directory("last_script_dir", chosen)
        result = self._script_manager().select_script(chosen)
        result["cancelled"] = False
        return result

    def script_control(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Host-enforced lifecycle controller for Script mode (status/start/run_extracted/pause/resume/stop/add_config)."""
        opts = dict(options or {})
        action = str(opts.get("action") or "status").strip().lower()
        mgr = self._script_manager()

        if action == "status":
            return mgr.status()
        if action == "pause":
            return mgr.pause()
        if action == "resume":
            return mgr.resume()
        if action == "stop":
            return mgr.stop()
        if action == "add_config":
            settings = self._settings()
            start_dir = default_browse_directory(str(settings.get("last_script_dir") or ""))
            files = self.choose_files("config", multiple=True, directory=start_dir)
            if not files:
                return {"ok": True, "cancelled": True, "configs": mgr.status().get("configs") or []}
            self._remember_directory("last_script_dir", files[0])
            res = mgr.add_config_files(files)
            res["cancelled"] = False
            return res

        raw_refs = [str(r) for r in (opts.get("references") or []) if str(r or "").strip()]
        ref_mgr = self._reference_manager()
        verified_refs: list[str] = []
        for ref in raw_refs:
            checked = ref_mgr.authorize_existing(ref, workflow="script")
            if not checked.ok:
                return {"ok": False, "error": checked.error}
            verified_refs.append(checked.path)

        max_items = int(opts.get("max_items") or 10)
        extended = bool(opts.get("extended_authorized"))

        if action == "run_extracted":
            photo_model = ob.photo_model_choice(opts.get("photo_model"))
            return mgr.start_extracted_run(
                max_items=max_items,
                extended_authorized=extended,
                model_id=f"flux2_{photo_model}",
                references=verified_refs,
            )

        if action == "start":
            try:
                return mgr.start_sandboxed_run(
                    max_items=max_items,
                    extended_authorized=extended,
                    references=verified_refs,
                    allow_review_classification=bool(opts.get("acknowledge_review")),
                )
            except SandboxUnavailableError as exc:
                return {"ok": False, "error": str(exc), "code": "sandbox_unavailable"}

        return {"ok": False, "error": f"Unknown script action '{action}'."}

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
            allowed = {slot["id"] for slot in _angle_slots()}
            slots = [str(item) for item in options.get("slots") or [] if str(item) in allowed]
            if not slots:
                raise ValueError("Tick at least one kind of photo.")
            anchor = str(options.get("image") or "").strip() or need_master()
            if not Path(anchor).is_file():
                raise ValueError("Pick a start photo first (step 1).")
            anchor = self._stage_reference_if_external(anchor, workflow="avatar_generation")
            # The same face as the chosen picture, from the ticked angles. Her master photo is the
            # default; any picture in My stuff can be the anchor instead.
            avatar["identity_refs"] = [anchor]
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
            character = str(options.get("image") or "").strip() or need_master()
            if not Path(character).is_file():
                raise ValueError("Pick a start photo first (step 1).")
            character = self._stage_reference_if_external(character, workflow="character_reference")
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
            if Path(image).is_file():
                image = self._stage_reference_if_external(image, workflow="image_to_video")
            seconds = 10 if int(options.get("seconds") or 5) >= 10 else 5
            shot["user_prompt"] = " ".join(str(options.get("motion") or "").split())[:1000] or DEFAULT_MOTION  # Runway video models allow up to 1000
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
            if Path(image).is_file():
                image = self._stage_reference_if_external(image, workflow="image_edit")
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

    def _is_presenter_image(self, path: str) -> bool:
        profile = self._studio_profile()
        if profile is None or not path:
            return False
        target = str(Path(path).expanduser().resolve())
        return any(ref and str(Path(ref).expanduser().resolve()) == target for ref in profile.identity_refs.values())

    def _local_wan_item(
        self,
        *,
        image: str,
        text: str,
        shape: str,
        label: str,
        action: str,
        video_model: str = "a14b",
        vram_mode: str = "balanced",
        verbatim: bool = False,
    ) -> QueuedRender:
        """One free 5-second clip on this computer's GPU (Wan 2.2 through ComfyUI)."""
        model = local_engine.VIDEO_MODELS.get(video_model) or local_engine.VIDEO_MODELS["a14b"]
        info = local_engine.inspect(self._settings())
        model_info = (info.get("video_models") or {}).get(video_model) or {}
        if not info["installed"] or model_info.get("missing_models"):
            raise ValueError(
                "Your GPU engine isn't installed yet. Run scripts/setup_local_engine.sh, or pick a Runway option."
            )
        if not image or not Path(image).is_file():
            raise ValueError("Pick a start picture first.")
        image = self._stage_reference_if_external(image, workflow="image_to_video")
        text = " ".join(str(text or "").split())[:ob.PROMPT_MAX_CHARS] or DEFAULT_MOTION
        phrase = depicts_minor(text)
        if phrase:
            raise ValueError(f'People shown have to be adults (18+). Remove "{phrase}" from the description.')
        preset = self.brand_instructor_preset()
        avatar_payload = {**preset["avatar"], "identity_refs": [], "hair_refs": [], "wardrobe_refs": []}
        shot_payload = {
            **preset["shot"],
            "shot_id": f"studio-{action}-local",
            "engine_preference": "wan22",  # the ShotSpec literal; the detailed model lives in runtime
            "duration_s": local_engine.SECONDS,
            "fps": local_engine.FPS,
            "user_prompt": text,
            "wind": {},
            "references": {"init_image": image},
        }
        avatar, shot = self._models_from_payload({"avatar": avatar_payload, "shot": shot_payload})
        job = build_job(avatar, shot)
        if self._is_presenter_image(image):
            # Budget covers identity, realism and camera blocks plus the whole description (up to the 4000-char sanity cap).
            prompt, _dropped = compile_presenter_prompt(avatar, shot, budget=5000, motion=True)
        elif verbatim:
            # The Free builder sends the user's words exactly as written; no baked-in wording.
            prompt = text
        else:  # some other picture: describe the motion only, never the presenter's identity
            prompt = f"{text if text.endswith(('.', '!', '?')) else text + '.'} {MOTION_REALISM} {NO_TEXT}"
        width, height = local_engine.SHAPES[_choice(shape, local_engine.SHAPES, "tall")]
        job.prompt = prompt
        placeholders = (
            local_engine.model_placeholders()
            if video_model == "a14b"
            else {
                **local_engine.model_placeholders_5b(),
                "WAN_TEXT_ENCODER": local_engine.MODELS["WAN_TEXT_ENCODER"][1],
                "WAN_VAE": local_engine.MODELS_5B["WAN5B_VAE"][1],
            }
        )
        job.asset_map.update(
            {
                "PROMPT": prompt,
                "INIT_IMAGE": image,
                "WAN_NEGATIVE": local_engine.NEGATIVE,
                "WAN_CLIP_DEVICE": ob.vram_mode_choice(vram_mode) == "cpu_offload" and "cpu" or "default",
                "WAN_WIDTH": width,
                "WAN_HEIGHT": height,
                "WAN_LENGTH": local_engine.FRAMES,
                "WAN_SEED": secrets.randbelow(2_000_000_000) + 1,
                "WAN_PREFIX": f"avatar_v2/studio_{action}_{job.job_id[:12]}",
                **placeholders,
            }
        )
        runtime = {
            "provider": "comfyui",
            "profile": local_engine.PROFILE,
            "engine": model["engine"],
            "model": model["label"],
            "steps": 4,
            "cfg": 1.0,
            "width": width,
            "height": height,
            "frames": local_engine.FRAMES,
            "studio_action": action,
            "estimate_credits": 0,
            "reason": "Rendered on your GPU (free)",
        }
        return QueuedRender(
            job=job,
            workflow=str(model["workflow"]),
            provider_config=self._provider_config({}),
            profile=local_engine.PROFILE,
            runtime=runtime,
            archive_dir=str(self._archive_dir()),
            label=label,
        )

    def _studio_local_move(self, options: dict[str, Any]) -> QueuedRender:
        """A "Moving clip" rendered on this computer's GPU (free)."""
        master, _sha = self._studio_master()
        image = str(options.get("image") or "").strip() or master
        if not image:
            raise ValueError("Pick your presenter photo first (step 1).")
        item = self._local_wan_item(image=image, text=str(options.get("motion") or ""), shape=str(options.get("shape") or ""),
                                    label=f"Motion clip · {local_engine.SECONDS} s · your GPU", action="move")
        if str(options.get("narration_id") or "").strip():
            voiceover = self._narration_file({"voiceover_id": str(options["narration_id"]).strip()}, item.job)
            item.runtime.update({"studio_voiceover": voiceover, "candidate_qa": True})
        return item

    def _studio_scene_items(
        self,
        options: dict[str, Any],
        *,
        video_model: str = "a14b",
        vram_mode: str = "balanced",
        action: str = "scene",
        verbatim: bool = False,
    ) -> list[QueuedRender]:
        """A video scene made offline: 1-12 shots (about a minute), each continuing from the previous shot's last frame."""
        image = str(options.get("image") or "").strip()  # a blank slate: never default to the master photo
        if not image or not Path(image).is_file():
            raise ValueError("Pick a start picture first (step 1).")
        image = self._stage_reference_if_external(image, workflow="image_to_video")
        shots = [" ".join(str(text or "").split()) for text in options.get("shots") or []]
        shots = [text for text in shots if text][:SCENE_MAX_SHOTS]
        if not shots:
            raise ValueError("Describe at least one shot (step 2).")
        shape = str(options.get("shape") or "")
        items: list[QueuedRender] = []
        for index, text in enumerate(shots, start=1):
            item = self._local_wan_item(
                image=image,
                text=text,
                shape=shape,
                action=action,
                video_model=video_model,
                vram_mode=vram_mode,
                verbatim=verbatim,
                label=f"Scene · shot {index} of {len(shots)} · your GPU",
            )
            if items:
                item.chain_from_previous = True
                item.runtime["chain_from_id"] = items[-1].id
            items.append(item)
        last = items[-1]
        last.runtime.update({"scene_shots": [item.id for item in items], "candidate_qa": True})
        if str(options.get("narration_id") or "").strip():
            last.runtime["studio_voiceover"] = self._narration_file(
                {"voiceover_id": str(options["narration_id"]).strip()}, last.job
            )
        return items

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
        if options.get("action") == "scene" or (options.get("action") == "move" and options.get("engine") == "local"):
            try:
                if options.get("action") == "scene":
                    shots = len(self._studio_scene_items(options))
                else:
                    self._studio_local_move(options)
                    shots = 1
            except (ValueError, OSError) as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "action": options["action"], "credits": 0, "usd": 0.0, "upper_bound": False, "local": True,
                    "shots": shots, "minutes": local_engine.ESTIMATED_MINUTES * shots, "warnings": [], "limit_error": None,
                    "needs_talker": False}
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
        if action == "scene" or (action == "move" and options.get("engine") == "local"):
            engine = self.engine_status()
            if engine["state"] != "ready":
                if engine["state"] in ("stopped", "error"):
                    self.engine_autostart()
                return {"ok": False, "error": engine["message"] + " Try again when it says ready."}
            try:
                items = self._studio_scene_items(options) if action == "scene" else [self._studio_local_move(options)]
            except (ValueError, OSError) as exc:
                return {"ok": False, "error": str(exc)}
            queue_ids = self.queue.add_many(items)
            return {"ok": True, "action": action, "queue_id": queue_ids[0],
                    "queued": [{"queue_id": queue_id} for queue_id in queue_ids], "total_credits": 0, "local": True,
                    "shots": len(items), "minutes": local_engine.ESTIMATED_MINUTES * len(items)}
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

    def _shot_video(self, queue_id: str, current: Any) -> Path | None:
        if current is not None and current.id == queue_id:
            outputs = current.outputs
        else:
            outputs = next((i.get("outputs") or [] for i in self.queue.list() if i.get("id") == queue_id), None)
            if outputs is None:
                try:
                    outputs = json.loads((_desktop().HISTORY_DIR / f"{queue_id}.json").read_text(encoding="utf-8")).get("outputs") or []
                except (OSError, json.JSONDecodeError):
                    outputs = []
        return next((Path(o) for o in outputs if Path(o).suffix.lower() in VIDEO_SUFFIXES and Path(o).is_file()
                     and not Path(o).name.startswith("scene_")), None)

    def _studio_finish_scene(self, item: Any) -> dict[str, Any]:
        """After a scene's last shot renders, join every shot into one video (and add her voiceover)."""
        videos = [self._shot_video(str(queue_id), item) for queue_id in item.runtime.get("scene_shots") or []]
        if not videos or any(video is None for video in videos):
            return {"ok": False, "scene": "not joined: a shot has no video"}
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {"ok": False, "scene": "not joined: ffmpeg is not installed"}
        target = videos[-1].with_name(f"scene_{item.id}.mp4")
        command = [ffmpeg, "-y", "-v", "error"]
        for video in videos:
            command += ["-i", str(video)]
        # Every shot after the first starts on the previous shot's last frame, so drop that repeated frame.
        parts = [f"[{i}:v]{'trim=start_frame=1,' if i else ''}setpts=PTS-STARTPTS[v{i}]" for i in range(len(videos))]
        joined = "".join(f"[v{i}]" for i in range(len(videos)))
        command += ["-filter_complex", ";".join(parts) + f";{joined}concat=n={len(videos)}:v=1:a=0[out]",
                    "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(target)]
        try:
            subprocess.run(command, capture_output=True, timeout=600, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            return {"ok": False, "scene": f"not joined: {type(exc).__name__}"}
        final = target
        voiceover = Path(str(item.runtime.get("studio_voiceover") or ""))
        if item.runtime.get("studio_voiceover") and voiceover.is_file():
            try:
                final = runway_api.replace_audio_track(target, "voiceover", voiceover)
            except runway_api.RunwayError:
                final = target
        item.outputs.insert(0, str(final))
        if final != target:
            item.outputs.insert(1, str(target))  # the silent version is kept too
        for output in {final, target}:
            item.output_hashes[str(output)] = sha256_file(output)
        return {"ok": True, "scene": "joined", "shots": len(videos), "file": final.name}

    def studio_avatars(self) -> dict[str, Any]:
        """Saved avatar profiles from the vault, with the pictures each one keeps (local only)."""
        library = self._library_or_none()
        avatars = []
        for entry in (library.avatars() if library is not None else []):
            profile = AvatarProfile.model_validate(entry)
            pictures = []
            for group in (profile.ordered_identity, *(list(getattr(profile, key).values()) for key in ("body_refs", "hair_refs", "wardrobe_refs"))):
                for path in group:
                    if path and Path(str(path)).is_file() and str(path) not in pictures:
                        pictures.append(str(path))
            if pictures:
                avatars.append({"id": profile.avatar_id, "name": profile.display_name, "pictures": pictures})
        avatars.sort(key=lambda a: (a["id"] != bi.PROFILE_ID, a["name"].lower()))
        return {"ok": True, "items": avatars}

    def studio_avatar_delete(self, options: dict[str, Any]) -> dict[str, Any]:
        """Remove a saved avatar so it stops being an option; its picture files stay in My stuff."""
        avatar_id = str((options or {}).get("avatar_id") or "").strip()
        if not avatar_id:
            return {"ok": False, "error": "No avatar given."}
        if avatar_id == bi.PROFILE_ID:
            return {"ok": False, "error": "That's your current presenter. Use Remove on her photo in the look recipe instead."}
        library = self._library_or_none()
        if library is None or not library.remove_avatar(avatar_id):
            return {"ok": False, "error": "That avatar isn't saved."}
        return {"ok": True, "deleted": avatar_id}

    def _managed_media_roots(self) -> tuple[list[Path], list[Path], list[Path]]:
        """Explicit app-owned media directories eligible for managed deletion (Astra RC-02)."""
        d = _desktop()
        output_roots: list[Path] = [d.APP_HOME, d.RUNS_DIR, d.VOICEOVER_DIR]
        try:
            output_roots.append(self._archive_dir())
        except Exception:
            pass
        import_roots: list[Path] = [d.IMPORTS_DIR]
        staging_roots: list[Path] = [d.APP_HOME / "staged_refs"]
        return output_roots, import_roots, staging_roots

    def studio_delete(self, options: dict[str, Any]) -> dict[str, Any]:
        """Detach a reference or delete ONE app-owned managed output from My stuff (Astra RC-02).

        Strict media-ownership invariants:
        - External original files, symlinks, and hard-linked files are NEVER unlinked on disk.
        - Removing an externally backed presenter master detaches the reference cleanly while
          preserving the original bytes untouched.
        - For verified app-owned managed files, physical unlink runs BEFORE history mutation so a
          filesystem error (permission denied / busy / read-only) never orphans the file from history.
        """
        opts = dict(options or {})
        path = str(opts.get("path") or "").strip()
        detach_only = bool(opts.get("detach_only"))
        if not path:
            return {"ok": False, "error": "No file given."}
        target = Path(path).expanduser()
        profile = self._studio_profile()
        front = str(profile.identity_refs.get("front") or "") if profile is not None else ""
        was_master = bool(front) and (
            Path(front).expanduser() == target
            or (target.exists() and Path(front).expanduser().exists() and Path(front).expanduser().resolve() == target.resolve())
        )

        # Explicit detach (e.g., presenter master Remove control) NEVER deletes files from disk.
        if detach_only:
            if not was_master:
                return {"ok": False, "error": "That photo is not currently set as the presenter master."}
            self._clear_master_refs()
            return {
                "ok": True,
                "detached": True,
                "deleted_file": False,
                "deleted": target.name,
                "records_updated": 0,
                "ownership": "detached_reference",
            }

        # Refuse deleting non-media files (configs, logs, JSON records, ledgers) under all circumstances!
        if _kind(str(target)) == "file":
            return {"ok": False, "deleted_file": False, "error": "Only media files can be managed through My stuff."}

        out_roots, imp_roots, stg_roots = self._managed_media_roots()
        ownership = classify_media_ownership(
            target,
            managed_output_roots=out_roots,
            managed_import_roots=imp_roots,
            staging_roots=stg_roots,
        )

        # External original / symlink / hardlink: NEVER unlink! If it was the presenter master, detach safely.
        if not ownership.deletable and ownership.classification != "missing":
            if was_master:
                self._clear_master_refs()
                return {
                    "ok": True,
                    "detached": True,
                    "deleted_file": False,
                    "deleted": target.name,
                    "records_updated": 0,
                    "ownership": ownership.classification,
                    "message": "Detached presenter master reference; external original file was preserved untouched.",
                }
            return {
                "ok": False,
                "deleted_file": False,
                "ownership": ownership.classification,
                "error": ownership.reason,
            }

        # Scan history records (read-only first!) to verify this managed file belongs to My stuff.
        history_dir = _desktop().HISTORY_DIR
        matching_records: list[tuple[Path, dict[str, Any], list[str]]] = []
        canon_target = Path(ownership.canonical_path) if ownership.canonical_path else target
        for record_file in history_dir.glob("*.json"):
            try:
                record = json.loads(record_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            outputs = record.get("outputs") or []
            kept: list[str] = []
            matched_here = False
            for o in outputs:
                op = Path(str(o)).expanduser()
                if op == target or (ownership.canonical_path and op.exists() and Path(os.path.realpath(op)) == canon_target):
                    matched_here = True
                else:
                    kept.append(str(o))
            if matched_here:
                matching_records.append((record_file, record, kept))

        if not matching_records and not was_master:
            return {"ok": False, "deleted_file": False, "error": "That file isn't in My stuff."}

        # Perform physical unlink BEFORE mutating history so a failed unlink never orphans the file!
        deleted_file = False
        if ownership.deletable and target.exists():
            try:
                target.unlink()
                deleted_file = True
            except OSError as exc:
                return {
                    "ok": False,
                    "deleted_file": False,
                    "error": f"Could not delete the file (history left unchanged): {exc}",
                }

        updated = 0
        for record_file, record, kept in matching_records:
            try:
                if kept:
                    record["outputs"] = kept
                    write_json_atomic(record_file, record)
                else:
                    record_file.unlink(missing_ok=True)
                updated += 1
            except OSError:
                continue

        if was_master:
            self._clear_master_refs()
        return {
            "ok": True,
            "detached": was_master and not deleted_file,
            "deleted_file": deleted_file,
            "deleted": target.name,
            "records_updated": updated,
            "ownership": ownership.classification,
        }

    def _clear_master_refs(self) -> None:
        """After her master photo is deleted, drop it from the vault so the app starts from a blank slate."""
        library = self._library_or_none()
        if library is None:
            return
        entry = library.get_avatar(bi.PROFILE_ID)
        if not entry:
            return
        profile = AvatarProfile.model_validate(entry)
        identity = dict(profile.identity_refs or {})
        hashes = dict(profile.reference_hashes or {})
        identity.pop("front", None)
        hashes.pop("front", None)
        library.put_avatar(profile.model_copy(update={"identity_refs": identity, "reference_hashes": hashes}).model_dump())

    def studio_presenter_add(self, options: dict[str, Any]) -> dict[str, Any]:
        """Put any picture from My stuff into the 'Your presenter' options (pending, never auto-accepted).

        Only the user's manual pick (studio_pick) ever makes one the master photo.
        """
        path = str((options or {}).get("path") or "").strip()
        if not path:
            return {"ok": False, "error": "No picture given."}
        source = Path(path).expanduser()
        if _kind(str(source)) != "image" or not source.is_file():
            return {"ok": False, "error": "That isn't a picture."}
        library = self._library_or_none()
        if library is None or library.get_avatar(bi.PROFILE_ID) is None:
            self.brand_instructor_install()
            library = self._library_or_none()
        if library is None or library.get_avatar(bi.PROFILE_ID) is None:
            return {"ok": False, "error": "The presenter profile is missing."}
        profile = AvatarProfile.model_validate(library.get_avatar(bi.PROFILE_ID))
        if any(c.path == str(source) and c.slot_id == "identity_front" for c in profile.build_manifest.candidates):
            return {"ok": True, "path": str(source), "already": True}
        library.record_candidate(
            bi.PROFILE_ID,
            {"path": str(source), "source_path": str(source), "slot_id": "identity_front",
             "group": "identity", "status": "pending"},
        )
        return {"ok": True, "path": str(source), "already": False}

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

    # ------------------------------------------------------------------ the Offline Builder (Free builder)

    def _forge_shots(self, options: dict[str, Any]) -> list[str]:
        """Shot prompts from a manual list when given, otherwise split from the one long prompt."""
        manual = [" ".join(str(text or "").split()) for text in options.get("shots") or []]
        manual = [text for text in manual if text]
        if manual:
            return manual[:ob.SCENE_MAX_SHOTS]
        prompt = " ".join(str(options.get("prompt") or "").split())[:ob.PROMPT_MAX_CHARS]
        return ob.split_scene_prompt(prompt)

    def _forge_photo_item(
        self,
        *,
        prompt: str,
        shape: str,
        photo_model: str,
        label: str,
        action: str = "forge_photo",
        extra_runtime: dict[str, Any] | None = None,
        reference_images: list[str] | None = None,
    ) -> QueuedRender:
        """One avatar photo made offline: FLUX.2 klein through ComfyUI (free)."""
        info = ob.inspect(self._settings())
        model_info = (info.get("models") or {}).get(photo_model)
        if model_info is None or model_info.get("missing_models"):
            raise ValueError(
                "That image model isn't installed yet. Run scripts/setup_offline_builder.sh (about 6 GB)."
            )
        prompt = " ".join(str(prompt or "").split())[:ob.PROMPT_MAX_CHARS]
        if not prompt:
            raise ValueError("Describe what you want first.")
        phrase = depicts_minor(prompt)
        if phrase:
            raise ValueError(f'People shown have to be adults (18+). Remove "{phrase}" from the description.')
        model = ob.PHOTO_MODELS[photo_model]
        preset = self.brand_instructor_preset()
        avatar_payload = {**preset["avatar"], "identity_refs": [], "hair_refs": [], "wardrobe_refs": []}
        shot_payload = {
            **preset["shot"],
            "shot_id": f"studio-{action}-forge",
            "engine_preference": "custom",  # the ShotSpec literal; the detailed model lives in runtime
            "user_prompt": prompt,
            "wind": {},
            "references": {},
        }
        avatar, shot = self._models_from_payload({"avatar": avatar_payload, "shot": shot_payload})
        job = build_job(avatar, shot)
        width, height = ob.PHOTO_SHAPES[_choice(shape, ob.PHOTO_SHAPES, "portrait")]
        job.prompt = prompt
        job.asset_map.update(
            {
                "PROMPT": prompt,
                "NEGATIVE_PROMPT": ob.PHOTO_NEGATIVE,
                "WIDTH": width,
                "HEIGHT": height,
                "SEED": secrets.randbelow(2_000_000_000) + 1,
                "STEPS": model["steps"],
                "CFG": model["cfg"],
                "SAMPLER_NAME": ob.PHOTO_SAMPLER,
                "OUTPUT_PREFIX": f"avatar_v2/forge_{job.job_id[:12]}",
                **ob.image_placeholders(photo_model),
            }
        )
        for r_idx, r_path in enumerate((reference_images or [])[:4], start=1):
            if r_path and Path(r_path).is_file():
                job.asset_map[f"REF_IMAGE_{r_idx}"] = str(r_path)
        runtime: dict[str, Any] = {
            "provider": "comfyui",
            "profile": local_engine.PROFILE,
            "engine": "flux2_txt2img",
            "model": model["label"],
            "steps": model["steps"],
            "cfg": model["cfg"],
            "width": width,
            "height": height,
            "studio_action": action,
            "estimate_credits": 0,
            "reason": "Rendered on your GPU (free)",
        }
        if extra_runtime:
            runtime.update(extra_runtime)
        return QueuedRender(
            job=job,
            workflow="builtin:flux2_txt2img",
            provider_config=self._provider_config({}),
            profile=local_engine.PROFILE,
            runtime=runtime,
            archive_dir=str(self._archive_dir()),
            label=label,
        )

    def _forge_continue_scene(self, item: QueuedRender) -> dict[str, Any]:
        """After the start photo renders, queue the scene shots that use it (prompt-only scenes)."""
        spec = dict(item.runtime.get("forge_scene") or {})
        photo = next(
            (Path(output) for output in item.outputs if Path(output).suffix.lower() in IMAGE_SUFFIXES and Path(output).is_file()),
            None,
        )
        if photo is None:
            return {"ok": False, "scene": "not started: the start photo has no usable output"}
        try:
            items = self._studio_scene_items(
                {
                    "image": str(photo),
                    "shots": spec.get("shots") or [],
                    "shape": str(spec.get("shape") or ""),
                },
                video_model=str(spec.get("video_model") or "a14b"),
                vram_mode=str(spec.get("vram_mode") or "balanced"),
                action="forge_scene",
                verbatim=True,
            )
        except ValueError as exc:
            return {"ok": False, "scene": f"not started: {exc}"}
        if not items:
            return {"ok": False, "scene": "not started: no shots"}
        queue_ids = self.queue.add_many(items)
        return {"ok": True, "scene": "queued", "shots": len(items), "queue_ids": queue_ids}

    def _compile_guided_prompt(
        self,
        *,
        prompt: str,
        action: str,
        guided: dict[str, Any] | None,
        model_id: str,
        width: int = 896,
        height: int = 1280,
        frames: int = 1,
    ) -> tuple[str, list[str]]:
        """Compile optional Guided attributes deterministically via StructuredPromptSpec.

        When no guided attributes or randomized fields are active, returns the raw prompt
        untouched so Automatic mode behaves identically.
        """
        g = dict(guided or {})
        has_attrs = any(str(v or "").strip() for v in (g.get("attributes") or {}).values())
        has_rand = bool(g.get("randomized"))
        if not has_attrs and not has_rand:
            return prompt, []
        spec = build_spec_from_guided_payload(
            prompt=prompt,
            task=action,
            guided=g,
            model_preference=model_id,
            width=width,
            height=height,
            frames=frames,
        )
        compiled = compile_prompt_for_model(spec, model_id)
        errors = [c["message"] for c in compiled.conflicts if c.get("severity") == "error"]
        if errors:
            raise ValueError(errors[0])
        if compiled.unsupported_capabilities:
            raise ValueError(compiled.unsupported_capabilities[0])
        return compiled.positive or prompt, list(compiled.warnings)

    def forge_state(self) -> dict[str, Any]:
        """Everything the Free builder screen shows, in one local call (no network)."""
        settings = self._settings()
        photos = ob.inspect(settings)
        videos = local_engine.video_model_info(settings)
        engine = self.engine_status()
        return {
            "ok": True,
            "installed": photos["installed"],
            "photo_models": [photos["models"][model_id] for model_id in ob.PHOTO_MODEL_IDS],
            "video_models": [videos[model_id] for model_id in local_engine.video_model_ids()],
            "vram_modes": [{"id": key, **value} for key, value in ob.VRAM_MODES.items()],
            "shapes": list(ob.PHOTO_SHAPES),
            "scene_shapes": list(local_engine.SHAPES),
            "max_shots": ob.SCENE_MAX_SHOTS,
            "shot_seconds": local_engine.SECONDS,
            "max_prompt_chars": ob.PROMPT_MAX_CHARS,
            "engine_state": engine["state"],
            "engine_message": engine["message"],
            "guided": guided_ui_payload("all"),
            "reference_roles": [r for r in ROLE_CATALOG if r["id"] != "primary_starter"],
            "character_profiles": self._character_profile_store().list_profiles(),
        }

    def forge_price(self, options: dict[str, Any]) -> dict[str, Any]:
        """What a Free builder action would take (local estimate, nothing queued)."""
        options = dict(options or {})
        action = "scene" if str(options.get("action") or "") == "scene" else "photo"
        prompt = " ".join(str(options.get("prompt") or "").split())[:ob.PROMPT_MAX_CHARS]
        photo_model = ob.photo_model_choice(options.get("photo_model"))
        video_model = str(options.get("video_model") or "a14b")
        if video_model not in local_engine.VIDEO_MODELS:
            video_model = "a14b"
        vram_mode = ob.vram_mode_choice(options.get("vram_mode"))
        image = str(options.get("image") or "").strip()
        guided = options.get("guided") if isinstance(options.get("guided"), dict) else None
        subject_refs = options.get("subject_references") if isinstance(options.get("subject_references"), dict) else None
        guided_warnings: list[str] = []
        ref_warnings: list[str] = []
        try:
            reg_model = f"flux2_{photo_model}" if action == "photo" else ("wan22_a14b" if video_model == "a14b" else "wan22_5b")
            if guided:
                prompt, guided_warnings = self._compile_guided_prompt(
                    prompt=prompt,
                    action=action,
                    guided=guided,
                    model_id=reg_model,
                    frames=1 if action == "photo" else local_engine.FRAMES,
                )
            if subject_refs:
                _sel, ref_warnings, _meta = self._resolve_subject_references(
                    starter_image=image,
                    raw_subject_refs=subject_refs,
                    prompt=prompt,
                    task="image_generate" if action == "photo" else "video_scene",
                    workflow_supports_multiref=(action == "photo" or not image),
                    stage=False,
                )
            if not prompt and action == "photo":
                raise ValueError("Describe what you want first.")
            phrase = depicts_minor(prompt)
            if phrase:
                raise ValueError(f'People shown have to be adults (18+). Remove "{phrase}" from the description.')
            if image and not Path(image).expanduser().is_file():
                raise ValueError("That start picture isn't on this computer anymore.")
            photos = ob.inspect(self._settings())
            model_info = (photos.get("models") or {}).get(photo_model)
            if model_info is None or model_info.get("missing_models"):
                raise ValueError(
                    f"The {model_info['label'] if model_info else photo_model} image model isn't installed yet. "
                    "Run scripts/setup_offline_builder.sh."
                )
            videos = local_engine.video_model_info(self._settings())
            if action == "scene" and videos[video_model].get("missing_models"):
                raise ValueError(
                    f"The {videos[video_model]['label']} video model isn't installed yet. "
                    "Run scripts/setup_offline_builder.sh for the 5B, or setup_local_engine.sh for the 14B."
                )
            if action == "photo":
                count = max(1, min(int(options.get("count") or 1), 4))
                shots = count
                minutes = ob.PHOTO_MODELS[photo_model]["estimated_minutes"] * count
            else:
                shots = len(self._forge_shots(options))
                if not shots:
                    raise ValueError("Describe at least one shot (it becomes the video scene).")
                photo_minutes = 0 if image else ob.PHOTO_MODELS[photo_model]["estimated_minutes"]
                minutes = photo_minutes + shots * local_engine.VIDEO_MODELS[video_model]["estimated_minutes"]
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "action": action,
            "credits": 0,
            "usd": 0.0,
            "upper_bound": False,
            "local": True,
            "shots": shots,
            "minutes": minutes,
            "photo_first": action == "scene" and not image,
            "warnings": [
                *(
                    ["The 9B model is non-commercial: only for testing and research."]
                    if photo_model == "9b"
                    else []
                ),
                *guided_warnings,
                *ref_warnings,
            ],
            "limit_error": None,
            "needs_talker": False,
        }

    def forge_go(self, options: dict[str, Any]) -> dict[str, Any]:
        """Start a Free builder action right away (no confirmation step)."""
        options = dict(options or {})
        action = "scene" if str(options.get("action") or "") == "scene" else "photo"
        engine = self.engine_status()
        if engine["state"] != "ready":
            if engine["state"] in ("stopped", "error"):
                self.engine_autostart()
            return {"ok": False, "error": engine["message"] + " Try again when it says ready."}
        prompt = " ".join(str(options.get("prompt") or "").split())[:ob.PROMPT_MAX_CHARS]
        photo_model = ob.photo_model_choice(options.get("photo_model"))
        video_model = str(options.get("video_model") or "a14b")
        if video_model not in local_engine.VIDEO_MODELS:
            video_model = "a14b"
        vram_mode = ob.vram_mode_choice(options.get("vram_mode"))
        shape = str(options.get("shape") or "")
        image = str(options.get("image") or "").strip()
        guided = options.get("guided") if isinstance(options.get("guided"), dict) else None
        subject_refs = options.get("subject_references") if isinstance(options.get("subject_references"), dict) else None
        try:
            staged_subject_refs: list[str] = []
            ref_selection_meta: dict[str, Any] | None = None
            if subject_refs:
                staged_subject_refs, _ref_warns, ref_selection_meta = self._resolve_subject_references(
                    starter_image=image,
                    raw_subject_refs=subject_refs,
                    prompt=prompt,
                    task="image_generate" if action == "photo" else "video_scene",
                    workflow_supports_multiref=(action == "photo" or not image),
                    stage=True,
                )
            if action == "photo":
                raw_user_prompt = prompt
                guided_spec = None
                if guided:
                    g_dict = dict(guided)
                    if any(str(v or "").strip() for v in (g_dict.get("attributes") or {}).values()) or g_dict.get("randomized"):
                        guided_spec = build_spec_from_guided_payload(
                            prompt=prompt,
                            task="photo",
                            guided=g_dict,
                            model_preference=f"flux2_{photo_model}",
                            frames=1,
                        )
                    prompt, _ = self._compile_guided_prompt(
                        prompt=prompt,
                        action="photo",
                        guided=guided,
                        model_id=f"flux2_{photo_model}",
                        frames=1,
                    )
                count = max(1, min(int(options.get("count") or 1), 4))
                short_label = ob.PHOTO_MODELS[photo_model]["label"].split(" (")[0]
                photo_refs = list(staged_subject_refs)
                if image and Path(image).expanduser().is_file():
                    staged_starter = self._stage_reference_if_external(image, workflow="multi_reference")
                    if staged_starter not in photo_refs:
                        photo_refs.insert(0, staged_starter)
                ver_record = None
                try:
                    routed = plan_intelligent_generation(
                        raw_user_prompt or prompt,
                        requested_action="photo",
                        starter_image=image or None,
                        guided_spec=guided_spec,
                        input_mode="guided" if guided else "automatic",
                        model_preference=f"flux2_{photo_model}",
                        use_llm=False,
                    )
                    if routed.ok and routed.verification_record is not None:
                        ver_record = routed.verification_record.public()
                        ver_record["compiled_prompt"] = prompt
                except Exception:
                    ver_record = None
                opt_extra: dict[str, Any] = {
                    "guided_mode": bool(guided),
                    "reference_selection": ref_selection_meta,
                    "verification_record": ver_record,
                }
                if options.get("auto_correct") is not None:
                    opt_extra["auto_correct"] = bool(options.get("auto_correct"))
                if options.get("visual_inspection_vlm") is not None:
                    opt_extra["visual_inspection_vlm"] = bool(options.get("visual_inspection_vlm"))
                if options.get("visual_inspection_model"):
                    opt_extra["visual_inspection_model"] = str(options.get("visual_inspection_model"))
                items = [
                    self._forge_photo_item(
                        prompt=prompt,
                        shape=shape,
                        photo_model=photo_model,
                        label=f"Avatar photo {index} of {count} · {short_label} · your GPU",
                        extra_runtime=dict(opt_extra),
                        reference_images=photo_refs,
                    )
                    for index in range(1, count + 1)
                ]
                total_minutes = ob.PHOTO_MODELS[photo_model]["estimated_minutes"] * count
                two_stage = False
            else:
                shots = self._forge_shots(options)
                if not shots:
                    raise ValueError("Describe at least one shot (it becomes the video scene).")
                if guided:
                    reg_vid = "wan22_a14b" if video_model == "a14b" else "wan22_5b"
                    shots = [
                        self._compile_guided_prompt(
                            prompt=shot_text,
                            action="scene",
                            guided=guided,
                            model_id=reg_vid,
                            frames=local_engine.FRAMES,
                        )[0]
                        for shot_text in shots
                    ]
                    if not image:
                        prompt, _ = self._compile_guided_prompt(
                            prompt=prompt,
                            action="photo",
                            guided=guided,
                            model_id=f"flux2_{photo_model}",
                            frames=1,
                        )
                if image and not Path(image).expanduser().is_file():
                    raise ValueError("That start picture isn't on this computer anymore.")
                photo_minutes = ob.PHOTO_MODELS[photo_model]["estimated_minutes"]
                if image:
                    end_image = str(options.get("end_image") or "").strip()
                    if end_image and not Path(end_image).expanduser().is_file():
                        raise ValueError("That end picture isn't on this computer anymore.")
                    items = self._studio_scene_items(
                        {"image": image, "shots": shots, "shape": shape},
                        video_model=video_model,
                        vram_mode=vram_mode,
                        action="forge_scene",
                        verbatim=True,
                    )
                    if end_image:
                        staged_end = self._stage_reference_if_external(end_image, workflow="image_to_video")
                        items[-1].job.asset_map["LAST_FRAME"] = staged_end
                    vid_ver = None
                    try:
                        routed_vid = plan_intelligent_generation(
                            prompt or shots[0],
                            requested_action="scene",
                            starter_image=image,
                            input_mode="guided" if guided else "automatic",
                            model_preference="wan22_a14b" if video_model == "a14b" else "wan22_5b",
                            use_llm=False,
                        )
                        if routed_vid.ok and routed_vid.verification_record is not None:
                            vid_ver = routed_vid.verification_record.public()
                    except Exception:
                        vid_ver = None
                    for it in items:
                        it.runtime["reference_selection"] = ref_selection_meta
                        if vid_ver is not None:
                            it.runtime["verification_record"] = vid_ver
                        if options.get("visual_inspection_vlm") is not None:
                            it.runtime["visual_inspection_vlm"] = bool(options.get("visual_inspection_vlm"))
                        if options.get("visual_inspection_model"):
                            it.runtime["visual_inspection_model"] = str(options.get("visual_inspection_model"))
                        if options.get("auto_correct") is not None:
                            it.runtime["auto_correct"] = bool(options.get("auto_correct"))
                    total_minutes = len(shots) * local_engine.VIDEO_MODELS[video_model]["estimated_minutes"]
                    two_stage = False
                else:
                    items = [
                        self._forge_photo_item(
                            prompt=prompt,
                            shape=shape,
                            photo_model=photo_model,
                            action="forge_scene",
                            label=f"Scene start photo · {ob.PHOTO_MODELS[photo_model]['label'].split(' (')[0]} · your GPU",
                            extra_runtime={
                                "candidate_qa": True,
                                "reference_selection": ref_selection_meta,
                                "forge_scene": {
                                    "shots": shots,
                                    "shape": shape,
                                    "video_model": video_model,
                                    "vram_mode": vram_mode,
                                },
                            },
                            reference_images=staged_subject_refs,
                        )
                    ]
                    total_minutes = photo_minutes + len(shots) * local_engine.VIDEO_MODELS[video_model]["estimated_minutes"]
                    two_stage = True
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        queue_ids = self.queue.add_many(items)
        return {
            "ok": True,
            "action": action,
            "queue_id": queue_ids[0],
            "queued": [{"queue_id": queue_id} for queue_id in queue_ids],
            "total_credits": 0,
            "local": True,
            "shots": len(items),
            "minutes": total_minutes,
            "photo_first": two_stage,
        }

    def studio_thumb(self, path: str) -> dict[str, Any]:
        """Small preview for an image or (first frame of) a video in the Studio results."""
        source = Path(str(path or "")).expanduser()
        kind = _kind(str(source))
        if kind == "image":
            if not self._reference_manager().is_path_authorized(source):
                return {"ok": False, "error": "That image was not selected through Avatar V2."}
            return self.reference_thumbnail(str(source))
        if kind != "video" or not source.is_file():
            return {"ok": False, "error": "No preview for this file."}
        if not self._reference_manager().is_path_authorized(source):
            return {"ok": False, "error": "That video was not selected through Avatar V2."}
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
        if not self._reference_manager().is_path_authorized(source):
            return {"ok": False, "error": "That photo was not selected through Avatar V2."}
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

    def _ensure_media_server(self) -> _LocalMediaServer:
        server = getattr(self, "_media_server", None)
        if server is None:
            server = _LocalMediaServer()
            self._media_server = server
        return server

    def _close_media_server(self) -> None:
        server = getattr(self, "_media_server", None)
        if server is not None:
            server.close()
            self._media_server = None

    def studio_video(self, path: str) -> dict[str, Any]:
        """Serve a local video over loopback HTTP (with Range seeking) for the in-app review viewer."""
        source = Path(str(path or "")).expanduser()
        if _kind(str(source)) != "video" or not source.is_file():
            return {"ok": False, "error": "That video is no longer on this computer."}
        if not self._reference_manager().is_path_authorized(source):
            return {"ok": False, "error": "That video was not selected through Avatar V2."}
        try:
            url = self._ensure_media_server().register(source)
            meta = _probe_video_meta(source)
            return {
                "ok": True,
                "url": url,
                "name": source.name,
                "width": meta.get("width"),
                "height": meta.get("height"),
                "duration_s": meta.get("duration_s"),
                "fps": meta.get("fps"),
            }
        except OSError as exc:
            return {"ok": False, "error": f"Could not open that video ({type(exc).__name__})."}

    # ------------------------------------------------------------------ state for the Studio screen

    def studio_library(self, kind: str = "all") -> dict[str, Any]:
        """Everything made with Studio and Runway, newest first, one tile per file."""
        kind = kind if kind in ("all", "image", "video", "audio") else "all"
        tiles: list[dict[str, Any]] = []
        for result in self._studio_results(limit=5000, include_failures=False):
            for output in result["outputs"]:
                if kind != "all" and output["kind"] != kind:
                    continue
                tiles.append({**output, "id": result["id"], "label": result["label"], "action": result["action"],
                              "finished_at": result["finished_at"], "credits": result["credits"],
                              "local": result["local"], "source": result["source"]})
        return {"ok": True, "kind": kind, "items": tiles}

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
            # Everything you've made counts: Runway jobs, recipes on your GPU, and the old app's local renders.
            source = "runway" if item.get("provider") == "runway" else ("gpu" if item.get("studio_action") else "old")
            local = source != "runway"
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
            qa_data = item.get("candidate_qa") if isinstance(item.get("candidate_qa"), dict) else {}
            inspection = qa_data.get("inspection") if isinstance(qa_data.get("inspection"), dict) else None
            correction = qa_data.get("correction") if isinstance(qa_data.get("correction"), dict) else None
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
                    "source": source,
                    "inspection": inspection,
                    "correction": correction,
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
