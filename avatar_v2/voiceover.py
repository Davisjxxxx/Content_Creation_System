"""Narration library and voice profiles, kept independent of visual identity.

* Voice profiles describe how narration should sound (provider, voice, pacing, pronunciation)
  and are never stored on an avatar's visual profile.
* Narration files (imported recordings or generated speech) are copied into a managed store,
  hashed, measured, and must be explicitly approved before talking-head or voiceover use.
* No voice cloning is offered here. Only prerecorded narration you are authorized to use,
  Runway preset voices, or voices designed from a text description.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import sha256_file
from .providers.runway import AUDIO_TYPES, media_duration_s

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")
_WORD_RE = re.compile(r"\S+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STATUSES = ("pending", "approved", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------- script tools


def apply_pronunciations(text: str, pronunciations: dict[str, str] | None) -> str:
    """Whole-word, case-insensitive respellings, for example {"SQL": "sequel"}."""
    output = str(text or "")
    for word, spoken in sorted((pronunciations or {}).items(), key=lambda item: -len(item[0])):
        word, spoken = str(word).strip(), str(spoken).strip()
        if word and spoken:
            output = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", spoken, output, flags=re.IGNORECASE)
    return output


def segment_script(text: str, max_chars: int) -> list[str]:
    """Split a script at paragraph, then sentence boundaries so each piece fits a model limit."""
    if max_chars < 50:
        raise ValueError("Segment limit must be at least 50 characters.")
    pieces: list[str] = []
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]:
        current = ""
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            while len(sentence) > max_chars:  # a single overlong sentence: break at a word boundary
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            candidate = f"{current} {sentence}".strip()
            if len(candidate) > max_chars and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)
    return pieces


def script_stats(text: str, *, target_wpm: tuple[int, int] = (140, 160)) -> dict[str, Any]:
    """Timing estimate and pronunciation hints for a narration script."""
    words = _WORD_RE.findall(str(text or ""))
    sentences = [s for s in _SENTENCE_RE.split(str(text or "").strip()) if s.strip()]
    longest = max((len(_WORD_RE.findall(s)) for s in sentences), default=0)
    low, high = target_wpm
    flags: list[str] = []
    if any(re.search(r"\d", word) for word in words):
        flags.append("Contains numerals: check how numbers, dates and units should be read aloud.")
    acronyms = sorted({w.strip(".,;:!?()\"'") for w in words if re.fullmatch(r"[A-Z]{2,}[0-9]*[.,;:!?]?", w)})
    if acronyms:
        flags.append("Acronyms to confirm (spell out or add a pronunciation): " + ", ".join(acronyms[:12]))
    if longest > 35:
        flags.append(f"Longest sentence has {longest} words; split it so breathing and pauses sound natural.")
    return {
        "characters": len(str(text or "")),
        "words": len(words),
        "sentences": len(sentences),
        "longest_sentence_words": longest,
        "estimated_seconds": [round(len(words) / high * 60, 1), round(len(words) / low * 60, 1)] if words else [0, 0],
        "pause_tags": str(text or "").count("[pause]"),
        "flags": flags,
    }


# ---------------------------------------------------------------------------- stores


class VoiceoverStore:
    """Managed narration files plus voice profiles, under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.index_path = self.root / "voiceovers.json"
        self.profiles_path = self.root / "voice_profiles.json"

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # -- narration files

    def add_file(
        self,
        source: str | Path,
        *,
        label: str = "",
        origin: str = "imported",
        script: str = "",
        voice_profile_id: str = "",
        authorization: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(str(source)).expanduser()
        if not path.is_file():
            raise ValueError(f"Narration file not found: {path.name}")
        suffix = path.suffix.lower()
        if suffix not in AUDIO_TYPES:
            raise ValueError(f"{path.name}: use MP3, WAV, FLAC, M4A or AAC narration.")
        digest = sha256_file(path)
        if not digest:
            raise ValueError(f"{path.name} could not be read.")
        index = self._read(self.index_path)
        for record in index.values():
            if record.get("sha256") == digest:
                return {**record, "duplicate": True}
        self.files_dir.mkdir(parents=True, exist_ok=True)
        stored = self.files_dir / f"{digest[:16]}{suffix}"
        if not stored.exists():
            from .ownership import copy_file_exclusive

            stored = copy_file_exclusive(path, self.files_dir, stored.name)
        record_id = f"vo_{uuid.uuid4().hex[:10]}"
        record = {
            "id": record_id,
            "label": (label or path.stem)[:120],
            "origin": origin,
            "status": "pending",
            "path": str(stored.resolve()),
            "original_name": path.name,
            "sha256": digest,
            "duration_s": media_duration_s(stored),
            "script": script[:20000],
            "voice_profile_id": voice_profile_id,
            "authorization": authorization[:500],
            "created_at": _now(),
            **(extra or {}),
        }
        index[record_id] = record
        self._write(self.index_path, index)
        return record

    def list(self) -> list[dict[str, Any]]:
        records = list(self._read(self.index_path).values())
        for record in records:
            record["exists"] = Path(record.get("path", "")).is_file()
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return self._read(self.index_path).get(str(record_id))

    def decide(self, record_id: str, decision: str, note: str = "") -> dict[str, Any]:
        if decision not in ("approved", "rejected"):
            raise ValueError("Decision must be approved or rejected.")
        index = self._read(self.index_path)
        record = index.get(str(record_id))
        if not record:
            raise ValueError(f"Narration '{record_id}' not found.")
        if decision == "approved" and sha256_file(record["path"]) != record.get("sha256"):
            raise ValueError("The stored narration file changed or is missing; re-import it before approving.")
        record.update({"status": decision, "decided_at": _now(), "decision_note": note[:500]})
        self._write(self.index_path, index)
        return record

    def approved_path(self, record_id: str) -> str:
        record = self.get(record_id)
        if not record or record.get("status") != "approved":
            raise ValueError("Use an approved narration from the Voice & narration library.")
        if sha256_file(record["path"]) != record.get("sha256"):
            raise ValueError("The approved narration file changed or is missing; re-import and re-approve it.")
        return str(record["path"])

    def approved_by_hash(self, path: str | Path) -> dict[str, Any] | None:
        digest = sha256_file(path)
        if not digest:
            return None
        return next(
            (r for r in self._read(self.index_path).values() if r.get("sha256") == digest and r.get("status") == "approved"),
            None,
        )

    # -- voice profiles

    def profiles(self) -> list[dict[str, Any]]:
        return sorted(self._read(self.profiles_path).values(), key=lambda item: item.get("label", ""))

    def save_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(profile.get("id") or "").strip()
        if not _ID_RE.match(profile_id):
            raise ValueError("Voice profile ids use lowercase letters, digits, - and _ (2-64 characters).")
        voice = dict(profile.get("voice") or {})
        if voice.get("type") not in (None, "runway-preset", "custom"):
            raise ValueError("Voice type must be runway-preset or custom (a designed voice).")
        speed = float(profile.get("speed", 1.0))
        if not 0.7 <= speed <= 1.2:
            raise ValueError("Narration speed must be between 0.7 and 1.2.")
        clean = {
            "id": profile_id,
            "label": str(profile.get("label") or profile_id)[:120],
            "independent_of_visual_identity": True,
            "source": str(profile.get("source") or "runway_tts"),
            "model": str(profile.get("model") or "eleven_v4"),
            "voice": voice,
            "design_prompt": str(profile.get("design_prompt") or "")[:1000],
            "speed": speed,
            "stability": float(profile.get("stability", 0.6)),
            "similarity_boost": float(profile.get("similarity_boost", 0.75)),
            "style": float(profile.get("style", 0.15)),
            "language_code": str(profile.get("language_code") or "en")[:16],
            "target_wpm": list(profile.get("target_wpm") or [140, 160])[:2],
            "pronunciations": {str(k)[:60]: str(v)[:120] for k, v in dict(profile.get("pronunciations") or {}).items()},
            "updated_at": _now(),
        }
        data = self._read(self.profiles_path)
        data[profile_id] = clean
        self._write(self.profiles_path, data)
        return clean

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        return self._read(self.profiles_path).get(str(profile_id))
