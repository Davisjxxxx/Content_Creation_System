"""Lessons made of talking-presenter segments, re-rendered only where inputs changed.

Each segment's render inputs (script or approved narration hash, Runway avatar, model and
voice settings) are hashed. A segment is re-queued only when that hash differs from its last
successful render or its recorded output file no longer verifies.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import sha256_file

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")
SEGMENT_MODES = ("script", "narration")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def segment_input_hash(lesson: dict[str, Any], segment: dict[str, Any], narration_sha256: str | None = None) -> str:
    material = {
        "mode": segment.get("mode"),
        "script": " ".join(str(segment.get("script") or "").split()),
        "narration_sha256": narration_sha256,
        "runway_avatar_id": lesson.get("runway_avatar_id"),
        "voice": segment.get("voice") or lesson.get("voice") or "",
        "model": "gwm1_avatars",
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


class LessonStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def list(self) -> list[dict[str, Any]]:
        return sorted(self._read().values(), key=lambda item: item.get("updated_at", ""), reverse=True)

    def get(self, lesson_id: str) -> dict[str, Any] | None:
        return self._read().get(str(lesson_id))

    def save(self, lesson: dict[str, Any]) -> dict[str, Any]:
        lesson_id = str(lesson.get("id") or "").strip()
        if not _ID_RE.match(lesson_id):
            raise ValueError("Lesson ids use lowercase letters, digits, - and _ (2-64 characters).")
        segments = []
        seen: set[str] = set()
        for raw in lesson.get("segments") or []:
            segment_id = str(raw.get("id") or "").strip()
            if not _ID_RE.match(segment_id) or segment_id in seen:
                raise ValueError(f"Segment id '{segment_id}' is invalid or repeated.")
            seen.add(segment_id)
            mode = str(raw.get("mode") or "script")
            if mode not in SEGMENT_MODES:
                raise ValueError("Segment mode must be script or narration.")
            script = str(raw.get("script") or "").strip()
            if mode == "script" and not script:
                raise ValueError(f"Segment '{segment_id}' needs a script.")
            if len(script) > 5000:
                raise ValueError(f"Segment '{segment_id}' is over 5000 characters; split it.")
            if mode == "narration" and not str(raw.get("voiceover_id") or "").strip():
                raise ValueError(f"Segment '{segment_id}' needs an approved narration.")
            segments.append(
                {
                    "id": segment_id,
                    "title": str(raw.get("title") or segment_id)[:120],
                    "mode": mode,
                    "script": script,
                    "voiceover_id": str(raw.get("voiceover_id") or ""),
                    "voice": str(raw.get("voice") or ""),
                }
            )
        data = self._read()
        previous = data.get(lesson_id) or {}
        record = {
            "id": lesson_id,
            "title": str(lesson.get("title") or lesson_id)[:160],
            "profile_id": str(lesson.get("profile_id") or ""),
            "runway_avatar_id": str(lesson.get("runway_avatar_id") or ""),
            "voice": str(lesson.get("voice") or ""),
            "segments": segments,
            "renders": previous.get("renders") or {},
            "updated_at": _now(),
        }
        data[lesson_id] = record
        self._write(data)
        return record

    def record_render(self, lesson_id: str, segment_id: str, render: dict[str, Any]) -> None:
        data = self._read()
        lesson = data.get(str(lesson_id))
        if not lesson:
            return
        lesson.setdefault("renders", {})[str(segment_id)] = {**render, "recorded_at": _now()}
        data[str(lesson_id)] = lesson
        self._write(data)

    def plan(self, lesson_id: str, narration_hashes: dict[str, str | None]) -> list[dict[str, Any]]:
        """Which segments are unchanged, new, changed, failed, or missing their output."""
        lesson = self.get(lesson_id)
        if not lesson:
            raise ValueError(f"Lesson '{lesson_id}' not found.")
        rows = []
        for segment in lesson["segments"]:
            current = segment_input_hash(lesson, segment, narration_hashes.get(segment.get("voiceover_id") or ""))
            render = (lesson.get("renders") or {}).get(segment["id"]) or {}
            if not render:
                state = "new"
            elif render.get("status") in {"queued", "running"}:
                # A render in flight for older inputs is stale: the segment still needs a new render.
                state = "pending" if render.get("input_hash") == current else "changed"
            elif render.get("status") != "done":
                state = "failed"
            elif render.get("input_hash") != current:
                state = "changed"
            elif not render.get("output") or sha256_file(render["output"]) != render.get("output_sha256"):
                state = "output_missing_or_changed"
            else:
                state = "unchanged"
            rows.append({"segment_id": segment["id"], "title": segment["title"], "state": state, "input_hash": current})
        return rows
