"""Presenter-production APIs mixed into the desktop bridge (DesktopAPI).

Kept in its own module so desktop.py stays reviewable. Methods without a leading underscore are
exposed to the desktop UI by pywebview; stores are built on demand (never stored as public
attributes) so pywebview does not expose their internals to page JavaScript.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from . import brand_instructor as bi
from . import cloud_ledger, realism_qa
from .avatar_designer import slot_by_id
from .integrity import sha256_file, verify_hashes
from .lessons import LessonStore
from .library import AvatarProfile
from .providers import runway as runway_api
from .reference_import import SOURCES as IMPORT_SOURCES
from .reference_import import import_images, list_imports
from .secret_store import secret_status
from .voiceover import VoiceoverStore, apply_pronunciations, script_stats, segment_script

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _desktop():
    from . import desktop  # late import: desktop imports this module

    return desktop


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PresenterMixin:
    # ------------------------------------------------------------------ shared helpers

    def _voiceovers(self) -> VoiceoverStore:
        return VoiceoverStore(_desktop().VOICEOVER_DIR)

    def _lessons(self) -> LessonStore:
        return LessonStore(_desktop().LESSONS_PATH)

    def _library_or_none(self):
        return getattr(self, "library", None)

    def _identity_lock(self, avatar: Any) -> dict[str, Any]:
        """Approved reference hashes for the avatar's vault profile, and the hash of its primary reference."""
        primary = avatar.identity_refs[0] if getattr(avatar, "identity_refs", None) else None
        lock: dict[str, Any] = {"profile_id": None, "approved": {}, "master_sha256": None,
                                "primary_sha256": sha256_file(primary) if primary else None}
        library = self._library_or_none()
        entry = library.get_avatar(avatar.avatar_id) if library is not None and getattr(avatar, "avatar_id", "") else None
        if entry:
            profile = AvatarProfile.model_validate(entry)
            approved = {role: digest for role, digest in profile.reference_hashes.items() if digest}
            lock.update({"profile_id": profile.avatar_id, "approved": approved, "master_sha256": approved.get("front")})
        return lock

    @staticmethod
    def _enforce_identity_lock(lock: dict[str, Any], path: Any, *, role: str, any_approved: bool = False) -> None:
        """Refuse to switch the character silently: identity uploads must match approved references."""
        approved = lock.get("approved") or {}
        if not approved or not path:
            return
        digest = sha256_file(path)
        allowed = set(approved.values()) if any_approved else {approved.get("front")}
        if lock.get("master_sha256") is None and not any_approved:
            return
        if digest not in allowed:
            raise runway_api.RunwayValidationError(
                f"The {role} is not the approved {'reference set' if any_approved else 'master identity'} for this "
                "profile (its content hash differs). Apply the profile from Build Avatar or approve the new image "
                "first, so the character never changes silently."
            )

    def _approved_narration(self, options: dict[str, Any], job: Any) -> str:
        store = self._voiceovers()
        if options.get("voiceover_id"):
            return store.approved_path(str(options["voiceover_id"]))
        audio = job.asset_map.get("AUDIO")
        if audio and store.approved_by_hash(audio):
            return str(audio)
        raise runway_api.RunwayValidationError(
            "Talking-head audio must be an approved narration. Import or generate it under Voice & narration and "
            "approve it first."
        )

    def _attach_voiceover(self, options: dict[str, Any], job: Any, local_media: dict[str, Any]) -> None:
        job.asset_map["RUNWAY_VOICEOVER"] = self._approved_narration(options, job)
        local_media["voiceover"] = {"asset": "RUNWAY_VOICEOVER"}

    # ------------------------------------------------------------------ budgets and cost accounting

    def _pending_cloud_estimates(self) -> float:
        lister = getattr(getattr(self, "queue", None), "list", None)
        if lister is None:
            return 0.0
        total = 0.0
        for item in lister():
            receipt = item.get("cloud_receipt") or {}
            if item.get("provider") == "runway" and item.get("status") in {"queued", "running"} and not receipt.get("task_id"):
                total += float(item.get("cloud_estimate") or 0)
        return total

    def _budget_status(self, settings: dict[str, Any], credits: float) -> dict[str, Any]:
        summary = cloud_ledger.spend_summary(_desktop().CLOUD_LEDGER_PATH)
        pending = self._pending_cloud_estimates()
        daily_cap = int(settings.get("runway_daily_credit_cap") or 300)
        monthly_cap = int(settings.get("runway_monthly_credit_cap") or 3000)
        today = summary["today_total"] + pending + float(credits)
        month = summary["month_total"] + pending + float(credits)
        error = None
        if today > daily_cap:
            error = f"This would bring today's Runway spend to {today:g} credits, above the daily cap of {daily_cap}."
        elif month > monthly_cap:
            error = f"This would bring this month's Runway spend to {month:g} credits, above the monthly cap of {monthly_cap}."
        return {
            "ok": error is None,
            "error": error,
            "daily_cap": daily_cap,
            "monthly_cap": monthly_cap,
            "today_projected": round(today, 2),
            "month_projected": round(month, 2),
            "pending_queued": round(pending, 2),
            "today_charged": round(summary["today_charged"], 2),
            "today_unconfirmed": round(summary["today_unconfirmed"], 2),
            "today_reserved": round(summary["today_reserved"], 2),
            "month_charged": round(summary["month_charged"], 2),
            "month_unconfirmed": round(summary["month_unconfirmed"], 2),
            "month_reserved": round(summary["month_reserved"], 2),
            "unresolved": summary["unresolved"],
        }

    def _unresolved_summary(self) -> dict[str, Any]:
        items = cloud_ledger.unresolved(_desktop().CLOUD_LEDGER_PATH)
        return {"count": len(items), "credits": round(sum(i["reserved_credits"] for i in items), 2)}

    def runway_unresolved(self) -> dict[str, Any]:
        """Cloud jobs whose charge is unknown; each counts at its estimate until resolved."""
        items = [
            {
                "key": item["key"],
                "task_id": item.get("task_id"),
                "queue_id": item.get("queue_id"),
                "label": item.get("label"),
                "model": item.get("model"),
                "status": item.get("status"),
                "reserved_credits": item["reserved_credits"],
                "created_ts": item.get("created_ts"),
            }
            for item in cloud_ledger.unresolved(_desktop().CLOUD_LEDGER_PATH)
        ]
        return {"ok": True, "items": items}

    def runway_resolve_charge(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Settle an unknown charge after checking Runway usage. Never resubmits anything."""
        key = str(payload.get("key") or "").strip()
        states = cloud_ledger.task_states(_desktop().CLOUD_LEDGER_PATH)
        state = states.get(key)
        if not state or cloud_ledger.classify(state)[1] != "unconfirmed":
            return {"ok": False, "error": "That entry is not an unresolved cloud charge."}
        resolution = str(payload.get("resolution") or "")
        if resolution == "not_billed":
            charge, credits = "none", 0
        elif resolution == "billed":
            try:
                credits = float(payload.get("credits"))
            except (TypeError, ValueError):
                return {"ok": False, "error": "Enter the credits Runway charged (from your Runway usage page)."}
            if not 0 <= credits <= 100000:
                return {"ok": False, "error": "Credits must be between 0 and 100000."}
            charge = "actual"
        else:
            return {"ok": False, "error": "Resolution must be billed or not_billed."}
        entry = {"status": "RESOLVED", "charge_status": charge, "credits_charged": credits,
                 "resolution_note": str(payload.get("note") or "")[:300]}
        entry["task_id" if state.get("task_id") else "request_key"] = key
        cloud_ledger.record(_desktop().CLOUD_LEDGER_PATH, entry)
        return {"ok": True, "key": key, "charge_status": charge, "credits_charged": credits}

    def runway_reconcile_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Check a task id you found in the Runway dashboard; download its outputs if it succeeded."""
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        key = str(payload.get("key") or "").strip()
        task_id = str(payload.get("task_id") or "").strip()
        if not runway_api._UUID_RE.match(task_id):
            return {"ok": False, "error": "Paste the task id (a UUID) from your Runway dashboard."}
        from .models import ProviderConfig

        config = ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE,
                                output_dir=str(_desktop().RUNS_DIR / "runway"))
        try:
            result = runway_api.RunwayProvider(config).reconcile_task(task_id)
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        if not result["terminal"]:
            return {"ok": True, "terminal": False, "status": result["status"],
                    "message": "The task is still running on Runway; check again later. Nothing was resubmitted."}
        receipt = result["cloud_receipt"]
        ledger = _desktop().CLOUD_LEDGER_PATH
        states = cloud_ledger.task_states(ledger)
        base = {"label": (states.get(key) or {}).get("label"), "queue_id": (states.get(key) or {}).get("queue_id"),
                "model": (states.get(key) or {}).get("model"),
                "estimate_credits": (states.get(key) or {}).get("estimate_credits")}
        charge = receipt.get("charge_status") or "unknown"
        cloud_ledger.record(ledger, {**base, "task_id": task_id, "status": result["status"], "charge_status": charge,
                                     "credits_charged": receipt.get("credits_charged")})
        if key and key != task_id and key in states:
            # The uncertain request turned out to be this task: settle the request entry against it.
            cloud_ledger.record(ledger, {"request_key": key, "status": "RESOLVED", "charge_status": "none",
                                         "credits_charged": 0, "resolution_note": f"reconciled as task {task_id}"})
        outputs = result["outputs"]
        return {"ok": True, "terminal": True, "status": result["status"], "outputs": outputs,
                "output_hashes": {path: sha256_file(path) for path in outputs}, "charge_status": charge,
                "credits_charged": receipt.get("credits_charged")}

    def _unresolved_guard(self, payload: dict[str, Any]) -> str | None:
        """Unknown earlier charges require an explicit acknowledgement before any new paid job."""
        unresolved = self._unresolved_summary()
        if unresolved["count"] and payload.get("acknowledge_unresolved") is not True:
            return (
                f"{unresolved['count']} earlier Runway submission(s) have unknown charges ({unresolved['credits']:g} "
                "credits reserved). Resolve them under Presenter → Providers & budget, or confirm that you are "
                "deliberately submitting new work anyway."
            )
        return None

    def runway_spend(self) -> dict[str, Any]:
        settings = self._settings()
        summary = cloud_ledger.spend_summary(_desktop().CLOUD_LEDGER_PATH)
        recent = sorted(
            cloud_ledger.task_states(_desktop().CLOUD_LEDGER_PATH).values(),
            key=lambda item: item.get("created_ts", ""),
            reverse=True,
        )[:50]
        return {"ok": True, "summary": summary, "budget": self._budget_status(settings, 0), "recent": recent,
                "credit_usd": runway_api.CREDIT_USD}

    # ------------------------------------------------------------------ queue post-processing

    def _runway_postprocess(self, item: Any) -> dict[str, Any]:
        """Register finished Runway outputs: slot candidates (pending), narration files, lesson renders."""
        runtime = item.runtime
        result: dict[str, Any] = {"ok": True, "decision": "review", "advisory": True, "recorded": []}
        library = self._library_or_none()
        slot_id, avatar_id = runtime.get("candidate_slot"), runtime.get("candidate_avatar_id")
        if slot_id and avatar_id and library is not None and library.get_avatar(avatar_id):
            master = str(runtime.get("candidate_reference") or "")
            settings = self._settings()
            images = [o for o in item.outputs if Path(o).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and Path(o).is_file()]
            for image in images:
                qa: dict[str, Any] = {"decision": "unavailable", "scores": {}, "reasons": ["Review manually."]}
                if master and Path(master).is_file():
                    try:
                        from .candidate_qa import compare_candidate_to_reference

                        qa = compare_candidate_to_reference(
                            Path(master), Path(image), slot_id=str(slot_id), slot_group="identity",
                            base_url=str(settings.get("vision_classifier_url") or "http://127.0.0.1:11434"),
                            model=str(settings.get("vision_classifier_model") or "qwen3-vl:8b-instruct"),
                        )
                    except Exception as exc:  # advisory only; never blocks registration
                        qa = {"decision": "unavailable", "scores": {}, "reasons": [f"Local QA unavailable: {type(exc).__name__}"]}
                library.record_candidate(
                    avatar_id,
                    {
                        "path": str(Path(image).resolve()),
                        "source_path": master or "text-to-image (no source reference)",
                        "slot_id": str(slot_id),
                        "group": "identity",
                        "status": "pending",
                        "qa_decision": qa.get("decision", "unavailable"),
                        "scores": qa.get("scores") or {},
                        "reasons": (qa.get("reasons") or [])[:6],
                    },
                )
                result["recorded"].append(Path(image).name)
                result["decision"] = qa.get("decision", "unavailable")
            result.update({"avatar_id": avatar_id, "slot_id": slot_id, "manifest_recorded": bool(images)})
        if runtime.get("register_voiceover"):
            store = self._voiceovers()
            records = []
            for output in item.outputs:
                if Path(output).suffix.lower() in runway_api.AUDIO_TYPES and Path(output).is_file():
                    records.append(
                        store.add_file(
                            output, label=f"Narration · {item.label}"[:120], origin="runway_tts",
                            script=str(runtime.get("script") or ""), voice_profile_id=str(runtime.get("voice_profile_id") or ""),
                            authorization="Runway preset or designed voice (no cloning)",
                        )["id"]
                    )
            result["voiceovers"] = records
        if runtime.get("lesson_id") and runtime.get("segment_id"):
            video = next((o for o in item.outputs if Path(o).suffix.lower() in runway_api.VIDEO_TYPES), "")
            lesson = self._lessons().get(str(runtime["lesson_id"])) or {}
            latest = ((lesson.get("renders") or {}).get(str(runtime["segment_id"])) or {}).get("queue_id")
            if latest not in (None, item.id):
                # A newer render was queued for this segment; never let a stale result overwrite it.
                result["lesson_render"] = "stale result ignored"
                return result
            self._lessons().record_render(
                runtime["lesson_id"], runtime["segment_id"],
                {"status": "done", "queue_id": item.id, "input_hash": runtime.get("lesson_input_hash"),
                 "output": video, "output_sha256": item.output_hashes.get(video) or (sha256_file(video) if video else None)},
            )
        return result

    def review_queue(self) -> dict[str, Any]:
        """Pending candidates across profiles; accept or reject them with avatar_candidate_decide."""
        library = self._library_or_none()
        items = []
        for entry in (library.avatars() if library is not None else []):
            profile = AvatarProfile.model_validate(entry)
            for candidate in profile.build_manifest.candidates:
                if candidate.status != "pending":
                    continue
                try:
                    label = slot_by_id(candidate.slot_id).label
                except ValueError:
                    label = candidate.slot_id
                items.append(
                    {
                        "avatar_id": profile.avatar_id,
                        "display_name": profile.display_name,
                        "slot_id": candidate.slot_id,
                        "slot_label": label,
                        "path": candidate.path,
                        "exists": Path(candidate.path).is_file(),
                        "source_path": candidate.source_path,
                        "qa_decision": candidate.qa_decision,
                        "scores": candidate.scores,
                        "reasons": candidate.reasons,
                        "created_at": candidate.created_at,
                    }
                )
        items.sort(key=lambda item: item["created_at"], reverse=True)
        return {"ok": True, "items": items}

    # ------------------------------------------------------------------ BRAND instructor profile

    def _presenter_state(self) -> dict[str, Any]:
        return {
            "brand": bi.spec_payload(),
            "tts_voice_suggestions": list(runway_api.TTS_PRESET_SUGGESTIONS),
            "voice_design_models": list(runway_api.VOICE_DESIGN_MODELS),
            "import_sources": dict(IMPORT_SOURCES),
            "qa_criteria": [{"id": key, "label": label} for key, label in realism_qa.HUMAN_CRITERIA],
        }

    def brand_instructor_spec(self) -> dict[str, Any]:
        library = self._library_or_none()
        installed = bool(library is not None and library.get_avatar(bi.PROFILE_ID))
        return {"ok": True, "installed": installed, **bi.spec_payload()}

    def brand_instructor_install(self) -> dict[str, Any]:
        """Create the vault profile and default voice profile if missing. Never overwrites approvals."""
        library = self._library_or_none()
        if library is None:
            return {"ok": False, "error": "The avatar library is not available."}
        store = self._voiceovers()
        if not store.get_profile(bi.VOICE_PROFILE["id"]):
            store.save_profile(dict(bi.VOICE_PROFILE))
        existing = library.get_avatar(bi.PROFILE_ID)
        if existing:
            return {"ok": True, "installed": False, "avatar": existing,
                    "note": "Already installed; approved references and hashes were left untouched."}
        return {"ok": True, "installed": True, "avatar": library.put_avatar(bi.profile_entry())}

    def brand_instructor_preset(self) -> dict[str, Any]:
        """Everything the Create screen needs to load the BRAND Instructor in one click."""
        library = self._library_or_none()
        avatar = library.profile_to_avatar_payload(bi.PROFILE_ID) if library is not None else None
        installed = avatar is not None
        if avatar is None:
            entry = bi.profile_entry()
            avatar = {**entry, "identity_refs": [], "body_refs": [], "hair_refs": [], "wardrobe_refs": [], "anatomy_guides": []}
        avatar["body_refs"] = []  # presenter jobs never use body references
        return {
            "ok": True,
            "installed": installed,
            "avatar": avatar,
            "shot": {
                "user_prompt": bi.IDENTITY_PROMPT_COMPACT,
                "content_class": "general",
                "seed": 41001,
                "camera": {"framing": "chest-up portrait", "movement": "locked-off tripod", "lens_equivalent_mm": 50},
                "environment": "neutral studio background",
                "wardrobe": "professional smart-casual clothing",
                "negative_prompt": bi.NEGATIVE_PROMPT,
            },
            "recommended": bi.RECOMMENDED,
            "slots": [dict(slot) for slot in bi.REFERENCE_SLOTS],
            "voice_profile": self._voiceovers().get_profile(bi.VOICE_PROFILE["id"]) or dict(bi.VOICE_PROFILE),
        }

    # ------------------------------------------------------------------ voice and narration

    def voice_profiles_list(self) -> dict[str, Any]:
        return {"ok": True, "items": self._voiceovers().profiles()}

    def voice_profile_save(self, profile: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "profile": self._voiceovers().save_profile(dict(profile or {}))}
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}

    def voiceover_import(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Copy prerecorded narration into the library (pending approval). Authorization is required."""
        authorization = str(payload.get("authorization") or "").strip()
        if len(authorization) < 10:
            return {
                "ok": False,
                "error": "Describe your authorization for this narration (for example: recorded by the hired voice "
                "actor under contract, or my own voice). Real people's voices need their permission.",
            }
        imported, errors = [], []
        for path in payload.get("paths") or []:
            try:
                imported.append(
                    self._voiceovers().add_file(
                        path, label=str(payload.get("label") or ""), origin="imported",
                        script=str(payload.get("script") or ""), voice_profile_id=str(payload.get("voice_profile_id") or ""),
                        authorization=authorization,
                    )
                )
            except ValueError as exc:
                errors.append({"file": Path(str(path)).name, "error": str(exc)})
        return {"ok": bool(imported), "imported": imported, "errors": errors}

    def voiceover_list(self) -> dict[str, Any]:
        return {"ok": True, "items": self._voiceovers().list()}

    def voiceover_decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            record = self._voiceovers().decide(str(payload.get("id") or ""), str(payload.get("decision") or ""),
                                               str(payload.get("note") or ""))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "item": record}

    def script_analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = self._voiceovers().get_profile(str(payload.get("voice_profile_id") or "")) or {}
        model = runway_api.model_spec(str(payload.get("model") or profile.get("model") or "eleven_v4"))
        spoken = apply_pronunciations(str(payload.get("script") or ""), profile.get("pronunciations"))
        wpm = tuple(profile.get("target_wpm") or (140, 160))
        try:
            segments = segment_script(spoken, model.prompt_max or 2500) if spoken.strip() else []
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "model": model.id,
            "limit": model.prompt_max,
            "stats": script_stats(spoken, target_wpm=(int(wpm[0]), int(wpm[1]))),
            "spoken_text": spoken,
            "segments": [{"index": i + 1, "characters": len(s), "text": s} for i, s in enumerate(segments)],
            "estimate": runway_api.estimate_credits(model.id, {"prompt": spoken}) if spoken.strip() else None,
        }

    def runway_voice_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Audition a voice designed from a text description (no real person's voice involved)."""
        blocked = self._runway_gate(self._settings(), uploads=True)
        if blocked:
            return {"ok": False, "error": blocked}
        if payload.get("confirm_cloud_upload") is not True:
            return {"ok": False, "error": "Confirm that the voice description will be sent to Runway."}
        prompt, model = str(payload.get("prompt") or "").strip(), str(payload.get("model") or "eleven_ttv_v3")
        try:
            with runway_api.client_from_store(timeout_s=120) as client:
                preview = client.preview_voice(prompt, model)
                target_dir = _desktop().VOICEOVER_DIR / "previews"
                stem = f"voice_preview_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                local = client.download(str(preview.get("url") or ""), target_dir, stem)
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": str(local), "duration_s": preview.get("durationSecs")}

    def runway_voice_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=True)
        if blocked:
            return {"ok": False, "error": blocked}
        if payload.get("confirm_cloud_upload") is not True:
            return {"ok": False, "error": "Confirm that the voice description will be sent to Runway."}
        try:
            with runway_api.client_from_store(timeout_s=60) as client:
                created = client.create_designed_voice(
                    str(payload.get("name") or "").strip(), str(payload.get("prompt") or "").strip(),
                    description=str(payload.get("description") or "") or None,
                    model=str(payload.get("model") or "eleven_ttv_v3"),
                )
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        profile_id = str(payload.get("voice_profile_id") or "").strip()
        if profile_id and created.get("id"):
            store = self._voiceovers()
            profile = store.get_profile(profile_id) or {"id": profile_id, "label": profile_id}
            profile["voice"] = {"type": "custom", "id": created["id"]}
            profile["design_prompt"] = str(payload.get("prompt") or "")
            store.save_profile(profile)
        return {"ok": True, "voice": created}

    def runway_voices_list(self) -> dict[str, Any]:
        blocked = self._runway_gate(self._settings(), uploads=False)
        if blocked:
            return {"ok": False, "error": blocked}
        try:
            with runway_api.client_from_store(timeout_s=30) as client:
                data = client.list_voices()
        except runway_api.RunwayError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "items": [
            {"id": v.get("id"), "name": v.get("name"), "status": v.get("status"), "created_at": v.get("createdAt")}
            for v in data.get("data") or []
        ]}

    # ------------------------------------------------------------------ manual reference import (Midjourney)

    def reference_import(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = import_images(
                [str(path) for path in payload.get("paths") or []],
                store_dir=_desktop().IMPORTS_DIR,
                source=str(payload.get("source") or "midjourney"),
                attested=payload.get("attested") is True,
                prompt=str(payload.get("prompt") or ""),
                job_reference=str(payload.get("job_reference") or ""),
                notes=str(payload.get("notes") or ""),
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        registered: list[str] = []
        avatar_id, slot_id = str(payload.get("avatar_id") or ""), str(payload.get("slot_id") or "")
        if avatar_id and slot_id and result["imported"]:
            library = self._library_or_none()
            try:
                slot = slot_by_id(slot_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc), **result}
            if slot.group != "identity":
                return {"ok": False, "error": "Imported images can only become identity or presenter candidates.", **result}
            if library is None or not library.get_avatar(avatar_id):
                return {"ok": False, "error": f"Avatar profile '{avatar_id}' not found.", **result}
            for record in result["imported"]:
                library.record_candidate(
                    avatar_id,
                    {"path": record["path"], "source_path": f"{record['source_label']}: {record['original_name']}",
                     "slot_id": slot.id, "group": "identity", "status": "pending", "qa_decision": "unavailable",
                     "reasons": ["Imported manually; check identity before accepting."]},
                )
                registered.append(record["path"])
        return {"ok": True, **result, "registered_as_pending": registered}

    def reference_imports_list(self) -> dict[str, Any]:
        return {"ok": True, "items": list_imports(_desktop().IMPORTS_DIR)}

    # ------------------------------------------------------------------ realism QA

    def _qa_report_path(self, raw: str) -> Path:
        root = _desktop().QA_DIR.resolve()
        path = Path(str(raw or "")).expanduser().resolve()
        if root not in path.parents or path.suffix != ".json":
            raise ValueError("QA reports must come from the app's QA folder.")
        return path

    def qa_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the automated realism checks locally (nothing leaves this computer)."""
        video = str(payload.get("path") or "")
        master = str(payload.get("master") or "")
        avatar_id = str(payload.get("avatar_id") or "")
        library = self._library_or_none()
        if not master and avatar_id and library is not None and library.get_avatar(avatar_id):
            master = str(AvatarProfile.model_validate(library.get_avatar(avatar_id)).identity_refs.get("front") or "")
        try:
            report = realism_qa.analyze(video, master_image=master or None, sample_fps=float(payload.get("sample_fps") or 10))
        except (FileNotFoundError, RuntimeError, subprocess.SubprocessError, OSError, ValueError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        folder = _desktop().QA_DIR
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = folder / f"{(report.get('sha256') or 'unknown')[:16]}_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {"ok": True, "report": report, "report_path": str(path)}

    def qa_review_save(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._qa_report_path(str(payload.get("report_path") or ""))
            report = json.loads(path.read_text(encoding="utf-8"))
            updated = realism_qa.apply_human_review(
                report, reviewer=str(payload.get("reviewer") or ""),
                ratings={k: int(v) for k, v in dict(payload.get("ratings") or {}).items() if str(v).strip().isdigit()},
                notes=dict(payload.get("notes") or {}),
            )
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}
        path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return {"ok": True, "report": updated, "report_path": str(path)}

    def qa_reports_list(self) -> dict[str, Any]:
        items = []
        for path in sorted(_desktop().QA_DIR.glob("*.json"), reverse=True)[:100]:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            gate = report.get("production_gate") or {}
            items.append({"report_path": str(path), "file": Path(str(report.get("file") or "")).name,
                          "measured_at": report.get("measured_at"), "ready": gate.get("ready"), "reasons": gate.get("reasons"),
                          "reviewed": (report.get("human_review") or {}).get("complete")})
        return {"ok": True, "items": items}

    # ------------------------------------------------------------------ lessons (changed-segment re-rendering)

    def _narration_hashes(self) -> dict[str, str | None]:
        return {record["id"]: record.get("sha256") for record in self._voiceovers().list() if record.get("status") == "approved"}

    def lesson_save(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": True, "lesson": self._lessons().save(dict(payload.get("lesson") or payload))}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def lesson_list(self) -> dict[str, Any]:
        return {"ok": True, "items": self._lessons().list()}

    def lesson_plan(self, lesson_id: str) -> dict[str, Any]:
        store = self._lessons()
        try:
            rows = store.plan(str(lesson_id), self._narration_hashes())
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        lesson = store.get(str(lesson_id)) or {}
        history_dir = _desktop().HISTORY_DIR
        for row in rows:  # reconcile queued renders with their finished queue records
            render = (lesson.get("renders") or {}).get(row["segment_id"]) or {}
            if row["state"] in {"pending", "changed"} and _SAFE_ID.match(str(render.get("queue_id") or "")):
                history = history_dir / f"{render['queue_id']}.json"
                if history.is_file():
                    record = json.loads(history.read_text(encoding="utf-8"))
                    if record.get("error_category") in {"cloud_submission_uncertain", "cloud_persistence"}:
                        # The earlier submission may have been billed: never re-queue it automatically.
                        row["state"] = "unresolved"
                    elif row["state"] == "pending" and record.get("status") in {"error", "cancelled"}:
                        row["state"] = "failed"
        blocked = {"unchanged", "pending", "unresolved"}
        return {"ok": True, "rows": rows, "to_render": [r["segment_id"] for r in rows if r["state"] not in blocked]}

    def lesson_queue_changed(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Queue only segments whose inputs changed (or failed / lost their output). Use dry_run to estimate."""
        lesson_id = str(payload.get("lesson_id") or "")
        lesson = self._lessons().get(lesson_id)
        if not lesson:
            return {"ok": False, "error": f"Lesson '{lesson_id}' not found."}
        if not lesson.get("runway_avatar_id"):
            return {"ok": False, "error": "Choose the lesson's Runway presenter avatar first."}
        plan = self.lesson_plan(lesson_id)
        if not plan.get("ok"):
            return plan
        targets = set(plan["to_render"])
        builds = []
        base = copy.deepcopy(payload.get("base") or {})
        for row in plan["rows"]:
            if row["segment_id"] not in targets:
                continue
            segment = next(s for s in lesson["segments"] if s["id"] == row["segment_id"])
            candidate = copy.deepcopy(base)
            candidate.setdefault("shot", {})["engine_preference"] = "runway"
            candidate["runway"] = {
                "task": "avatar_video", "avatar_id": lesson["runway_avatar_id"],
                "speech": "audio" if segment["mode"] == "narration" else "text",
                "script": segment["script"], "voiceover_id": segment.get("voiceover_id") or "",
                "voice": segment.get("voice") or lesson.get("voice") or "",
            }
            try:
                build = self._runway_build(candidate)
            except (runway_api.RunwayError, ValueError) as exc:
                return {"ok": False, "error": f"Segment '{segment['id']}': {exc}"}
            build["runtime"].update({"candidate_qa": True, "lesson_id": lesson_id, "segment_id": segment["id"],
                                     "lesson_input_hash": row["input_hash"]})
            builds.append((segment, build))
        total = sum(build["estimate"]["credits"] for _segment, build in builds)
        summary = {"to_render": [segment["id"] for segment, _ in builds],
                   "unchanged": [r["segment_id"] for r in plan["rows"] if r["state"] == "unchanged"],
                   "total_credits": total}
        if payload.get("dry_run") or not builds:
            return {"ok": True, "dry_run": True, **summary}
        settings = self._settings()
        blocked = self._runway_gate(settings, uploads=True)
        if blocked:
            return {"ok": False, "error": blocked}
        confirmation = self._runway_confirmation_error(payload, total)
        if confirmation:
            return {"ok": False, "error": confirmation}
        batch_cap = int(settings.get("runway_max_credits_per_batch") or 400)
        if any(build["estimate"]["credits"] > build["cap"] for _s, build in builds) or total > batch_cap:
            return {"ok": False, "error": f"The changed segments need {total} credits, above a per-job or batch cap."}
        budget = self._budget_status(settings, total)
        if not budget["ok"]:
            return {"ok": False, "error": budget["error"]}
        guard = self._unresolved_guard(payload)
        if guard:
            return {"ok": False, "error": guard}
        items = [self._runway_queue_item(build, payload, f"{lesson['title']} · {segment['title']}") for segment, build in builds]
        queue_ids = self.queue.add_many(items)
        store = self._lessons()
        for (segment, build), queue_id in zip(builds, queue_ids):
            store.record_render(lesson_id, segment["id"], {"status": "queued", "queue_id": queue_id,
                                                           "input_hash": build["runtime"]["lesson_input_hash"]})
        return {"ok": True, "queued": list(zip(summary["to_render"], queue_ids)), **summary}

    # ------------------------------------------------------------------ availability, integrity, saved profiles

    def provider_status(self) -> dict[str, Any]:
        settings = self._settings()
        comfy = {"available": False}
        try:
            with httpx.Client(timeout=3) as client:
                client.get(f"{str(settings.get('comfyui_url') or 'http://127.0.0.1:8188').rstrip('/')}/system_stats").raise_for_status()
            comfy = {"available": True}
        except httpx.HTTPError as exc:
            comfy["detail"] = f"{type(exc).__name__}"
        vision = {"available": False}
        try:
            with httpx.Client(timeout=3) as client:
                tags = client.get(f"{str(settings.get('vision_classifier_url') or 'http://127.0.0.1:11434').rstrip('/')}/api/tags")
                tags.raise_for_status()
            names = [m.get("name") for m in tags.json().get("models", [])]
            vision = {"available": True, "model_installed": settings.get("vision_classifier_model") in names}
        except (httpx.HTTPError, ValueError) as exc:
            vision["detail"] = f"{type(exc).__name__}"
        gate = self._runway_gate(settings, uploads=True)
        return {
            "ok": True,
            "providers": {
                "comfyui": {**comfy, "label": "Local ComfyUI (H3 / Wan / AnimateDiff)"},
                "local_vision": {**vision, "label": "Local vision QA (Ollama)"},
                "runway": {"available": gate is None, "enabled": bool(settings.get("runway_enabled")),
                           "upload_consent": bool(settings.get("runway_upload_consent")),
                           "key_configured": secret_status("runway")["configured"], "detail": gate,
                           "label": "Runway (cloud, opt-in)"},
                "midjourney": {"available": True, "mode": "manual import only",
                               "detail": "No official public API; automated access is not permitted, so images are imported manually.",
                               "label": "Midjourney (manual import)"},
                "ffmpeg": {"available": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "label": "ffmpeg / ffprobe"},
            },
        }

    def verify_outputs(self, item_id: str) -> dict[str, Any]:
        if not _SAFE_ID.match(str(item_id or "")):
            return {"ok": False, "error": "Invalid queue item id."}
        history = _desktop().HISTORY_DIR / f"{item_id}.json"
        if not history.is_file():
            return {"ok": False, "error": "No finished job with that id."}
        recorded = json.loads(history.read_text(encoding="utf-8")).get("output_hashes") or {}
        if not recorded:
            return {"ok": False, "error": "This job finished before output hashes were recorded."}
        return verify_hashes(recorded)

    def _profiles_path(self) -> Path:
        return _desktop().PROVIDER_PROFILES_PATH

    def provider_profiles_list(self) -> dict[str, Any]:
        try:
            data = json.loads(self._profiles_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        return {"ok": True, "items": sorted(data.values(), key=lambda item: item.get("name", "")) if isinstance(data, dict) else []}

    def provider_profile_save(self, profile: dict[str, Any]) -> dict[str, Any]:
        name = str((profile or {}).get("name") or "").strip()
        if not 1 <= len(name) <= 60:
            return {"ok": False, "error": "Give the provider profile a name (1-60 characters)."}
        task = str(profile.get("task") or "still")
        try:
            model = runway_api.model_spec(str(profile.get("model") or ""))
        except runway_api.RunwayValidationError as exc:
            return {"ok": False, "error": str(exc)}
        if runway_api.TASK_ENDPOINT_TASK.get(task) != model.task:
            return {"ok": False, "error": f"{model.label} cannot run the '{task}' task."}
        clean = {"name": name, "provider": "runway", "task": task, "model": model.id, "saved_at": _now()}
        for key in ("ratio", "duration", "resolution", "output_count", "prompt_expansion", "candidates"):
            if profile.get(key) not in (None, ""):
                clean[key] = profile[key]
        if clean.get("ratio") and clean["ratio"] not in model.ratios:
            return {"ok": False, "error": f"Ratio {clean['ratio']} is not supported by {model.label}."}
        current = {item["name"]: item for item in self.provider_profiles_list()["items"]}
        current[name] = clean
        path = self._profiles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return {"ok": True, "profile": clean}

    def provider_profile_delete(self, name: str) -> dict[str, Any]:
        current = {item["name"]: item for item in self.provider_profiles_list()["items"]}
        removed = current.pop(str(name), None) is not None
        path = self._profiles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return {"ok": True, "removed": removed}
