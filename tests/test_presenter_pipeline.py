"""Phases 3-10: reconciliation, cost ledger, integrity, audio separation, voice, BRAND profile,
identity locking, manual imports, realism QA and changed-segment lessons.

No test contacts Runway, Ollama or ComfyUI: HTTP goes through httpx.MockTransport or is monkeypatched.
Media fixtures are generated locally with ffmpeg and Pillow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

from avatar_v2 import brand_instructor as bi
from avatar_v2 import cloud_ledger, desktop, realism_qa, render_queue, secret_store
from avatar_v2.avatar_designer import COVERAGE_SLOTS, slot_by_id
from avatar_v2.composer import build_job
from avatar_v2.desktop import DesktopAPI
from avatar_v2.integrity import sha256_file, verify_hashes
from avatar_v2.lessons import LessonStore
from avatar_v2.library import AvatarProfile, LibraryStore
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.policy import evaluate_policy
from avatar_v2.presenter import compile_presenter_prompt
from avatar_v2.providers import runway as runway_api
from avatar_v2.providers.runway import RunwayClient, RunwayProvider, RunwayValidationError, utf16_len
from avatar_v2.reference_import import import_images, list_imports
from avatar_v2.render_queue import QueuedRender, RenderQueue
from avatar_v2.voiceover import VoiceoverStore, apply_pronunciations, script_stats, segment_script

KEY = "rw_test_key_0123456789abcdef"
TASK_ID = "11111111-2222-3333-4444-555555555555"
AVATAR_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
VOICE_UUID = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")


def _png(path: Path, size=(720, 1280), color=(200, 160, 140)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _video(path: Path, seconds: float = 3.0, *, audio: bool = True, size: str = "320x568") -> Path:
    command = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={seconds}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    command += ["-c:v", "mpeg4", "-q:v", "5"]
    command += ["-c:a", "aac"] if audio else []
    command += ["-shortest", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


def _audio(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}", str(path)],
        check=True, capture_output=True,
    )
    return path


def _streams(path: Path) -> set[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], check=True, capture_output=True)
    return {s["codec_type"] for s in json.loads(out.stdout)["streams"]}


class FakeMemory:
    def __init__(self):
        self.calls = []

    def update_settings(self, **_kwargs):
        pass

    def maybe_cleanup(self, provider, **kwargs):
        self.calls.append(kwargs)
        return {"reason": kwargs["reason"]}

    def start_trace(self, _label):
        pass

    def stop_trace(self):
        return {}


class FakeQueue:
    def __init__(self):
        self.items: list[QueuedRender] = []
        self.memory_manager = FakeMemory()

    def add(self, item):
        self.items.append(item)
        return item.id

    def add_many(self, items):
        self.items.extend(items)
        return [item.id for item in items]

    def list(self):
        return [item.public() for item in self.items]


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name, value in {
        "CONFIG_PATH": tmp_path / "desktop.json",
        "RUNS_DIR": tmp_path / "runs",
        "HISTORY_DIR": tmp_path / "history",
        "RUNWAY_AVATARS_PATH": tmp_path / "runway_avatars.json",
        "CLOUD_LEDGER_PATH": tmp_path / "cloud_ledger.jsonl",
        "VOICEOVER_DIR": tmp_path / "voiceover",
        "IMPORTS_DIR": tmp_path / "imports",
        "LESSONS_PATH": tmp_path / "lessons.json",
        "QA_DIR": tmp_path / "qa",
        "PROVIDER_PROFILES_PATH": tmp_path / "provider_profiles.json",
    }.items():
        monkeypatch.setattr(desktop, name, value)
    (tmp_path / "history").mkdir()
    monkeypatch.setattr(secret_store, "SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)
    return tmp_path


@pytest.fixture
def api(home):
    instance = DesktopAPI.__new__(DesktopAPI)
    instance.queue = FakeQueue()
    instance.library = LibraryStore(home / "library")
    return instance


def _enable(api, **extra):
    """Runway needs only a saved key; ``extra`` sets optional owner limits."""
    if extra:
        api.save_settings(extra)
    assert api.runway_save_key(KEY)["ok"] is True


def _brand_payload(api, home, *, task="still", refs=None, **options):
    api.brand_instructor_install()
    preset = api.brand_instructor_preset()
    avatar = {**preset["avatar"], "subject_kind": "synthetic"}
    return {
        "settings": {},
        "avatar": avatar,
        "shot": {**preset["shot"], "shot_id": "brand", "duration_s": 6, "engine_preference": "runway",
                 "wind": {}, "references": refs or {}},
        "runway": {"task": task, **options},
    }


def _confirm(payload, credits):
    """Legacy confirmation fields: still accepted, no longer required."""
    return {**payload, "confirm_cloud_upload": True, "confirmed_estimate_credits": credits}


def _accept_master(api, home, name="master.png", color=(190, 150, 130)):
    master = _png(home / name, color=color)
    result = api.avatar_candidate_decide(
        {"avatar_id": bi.PROFILE_ID, "slot_id": "identity_front", "decision": "accepted", "path": str(master)}
    )
    assert result["ok"] is True, result
    return master


# --------------------------------------------------------------------------- reconciliation and ledger


def _runway_item(tmp_path, **updates):
    avatar = AvatarManifest.model_validate(
        {"avatar_id": "p", "display_name": "P", "subjects": [{"id": "p", "kind": "synthetic"}],
         "identity_refs": [str(_png(tmp_path / "id.png"))]}
    )
    shot = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "smile", "engine_preference": "runway"})
    data = {
        "job": build_job(avatar, shot),
        "workflow": "runway:still:gen4_image",
        "provider_config": ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE),
        "profile": "cloud_runway",
        "runtime": {"estimate_credits": 8},
        "label": "cloud",
    }
    data.update(updates)
    return QueuedRender(**data)


def _queue(tmp_path, items=()):
    queue = RenderQueue.__new__(RenderQueue)
    queue.work_dir = tmp_path / "queue"
    queue.work_dir.mkdir(parents=True, exist_ok=True)
    queue.history_dir = tmp_path / "qhistory"
    queue.history_dir.mkdir(parents=True, exist_ok=True)
    queue._ledger_path = tmp_path / "ledger.jsonl"
    queue._items = list(items)
    queue._lock = threading.RLock()
    queue._wake = threading.Event()
    queue._stop = threading.Event()
    queue._active_provider = None
    queue._active_id = None
    queue.candidate_qa_runner = None
    queue.memory_manager = FakeMemory()
    return queue


def test_restart_resumes_created_cloud_tasks_instead_of_failing_or_rebilling(tmp_path):
    queue = _queue(tmp_path)
    created = _runway_item(tmp_path, status="running", prompt_id=TASK_ID, cloud_receipt={"task_id": TASK_ID, "uploads": [{"file": "id.png"}]})
    never_sent = _runway_item(tmp_path, status="running", cloud_receipt={"submission": {"state": "preparing"}})
    unknown = _runway_item(tmp_path, status="running")  # no durable marker: the request may have been sent
    restored = queue._item_from_dict(queue._item_to_dict(created))
    assert restored.status == "queued" and restored.prompt_id == TASK_ID
    assert restored.cloud_receipt["reconcile_after_restart"] is True
    failed = queue._item_from_dict(queue._item_to_dict(never_sent))
    assert failed.status == "error" and "no generation was billed" in failed.error
    uncertain = queue._item_from_dict(queue._item_to_dict(unknown))
    assert uncertain.status == "error" and uncertain.error_category == "cloud_submission_uncertain"
    assert uncertain.cloud_receipt["charge_status"] == "unknown" and "billed" not in uncertain.error


def test_resumed_job_polls_the_existing_task_and_keeps_the_original_receipt(tmp_path, monkeypatch):
    seen = {}

    class ResumingRunway:
        uses_local_gpu = False

        def __init__(self, config):
            self.on_task_created = None

        def render(self, workflow, job):
            seen["resume"] = job.asset_map.get("RUNWAY_RESUME_TASK_ID")
            return {"prompt_id": TASK_ID, "outputs": [str(Path(__file__).parent / "fixtures/rc5/still.png")], "cloud_receipt": {"status": "SUCCEEDED", "credits_charged": 8, "resumed": True}}

    monkeypatch.setattr(render_queue, "RunwayProvider", ResumingRunway)
    item = _runway_item(tmp_path, prompt_id=TASK_ID, cloud_receipt={"task_id": TASK_ID, "uploads": [{"file": "id.png"}]})
    queue = _queue(tmp_path, [item])
    queue._execute(item)
    assert seen["resume"] == TASK_ID
    assert item.cloud_receipt["uploads"] == [{"file": "id.png"}] and item.cloud_receipt["resumed"] is True


def test_provider_resume_path_never_creates_a_new_task(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", KEY)
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path == f"/v1/tasks/{TASK_ID}":
            return httpx.Response(200, json={"id": TASK_ID, "status": "SUCCEEDED", "cost": {"credits": 8}, "output": ["https://cdn.example/o.png"]})
        if request.url.host == "cdn.example":
            return httpx.Response(200, content=(Path(__file__).parent / "fixtures/rc5/still.png").read_bytes(), headers={"content-type": "image/png"})
        return httpx.Response(500)

    item = _runway_item(tmp_path)
    spec = {"version": 1, "model": "gen4_image", "params": {"prompt": "@presenter smiles", "ratio": "1080:1440"},
            "media": {"references": [{"asset": "IDENTITY_REF_1", "tag": "presenter"}]}}
    item.job.asset_map["RUNWAY_SPEC"] = json.dumps(spec)
    item.job.asset_map["RUNWAY_RESUME_TASK_ID"] = TASK_ID
    config = ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE, output_dir=str(tmp_path / "out"))
    result = RunwayProvider(config, transport=httpx.MockTransport(handler), sleep=lambda _s: None).render(item.workflow, item.job)
    assert ("POST", "/v1/text_to_image") not in calls
    assert result["cloud_receipt"]["resumed"] is True and result["cloud_receipt"]["credits_charged"] == 8
    assert list(result["cloud_receipt"]["downloads_sha256"].values()) == [sha256_file(result["outputs"][0])]


def test_task_creation_is_persisted_and_logged_before_polling(tmp_path):
    item = _runway_item(tmp_path)
    queue = _queue(tmp_path, [item])
    queue._record_cloud_task(item, TASK_ID, {"model": "gen4_image", "estimate": {"credits": 8}, "api_estimated_credits": 8})
    saved = json.loads((queue.work_dir / "items" / f"{item.id}.json").read_text())
    assert saved["prompt_id"] == TASK_ID
    item.status = "done"
    item.cloud_receipt.update({"status": "SUCCEEDED", "credits_charged": 7, "charge_status": "actual"})
    queue._record_cloud_final(item)
    states = cloud_ledger.task_states(queue._ledger_path)
    assert states[TASK_ID]["status"] == "SUCCEEDED" and states[TASK_ID]["credits_charged"] == 7


def test_spend_summary_counts_charged_and_reserved_credits(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    cloud_ledger.record(ledger, {"task_id": "a", "status": "CREATED", "api_estimated_credits": 60, "model": "gen4.5"})
    cloud_ledger.record(ledger, {"task_id": "b", "status": "CREATED", "api_estimated_credits": 8, "model": "gen4_image"})
    cloud_ledger.record(ledger, {"task_id": "b", "status": "SUCCEEDED", "credits_charged": 8, "model": "gen4_image"})
    summary = cloud_ledger.spend_summary(ledger)
    assert summary["today_charged"] == 8 and summary["today_reserved"] == 60
    assert summary["today_total"] == 68 and summary["by_model"] == {"gen4_image": 8.0}


def test_no_daily_limit_by_default_and_spend_counts_unlogged_queued_jobs(api, home):
    _enable(api)
    cloud_ledger.record(desktop.CLOUD_LEDGER_PATH, {"task_id": "x", "status": "SUCCEEDED", "credits_charged": 990})
    payload = _brand_payload(api, home, ratio="1080:1440")
    first = api.queue_runway(payload)
    assert first["ok"] is True and first["budget"]["today_projected"] == 998 and first["budget"]["daily_cap"] == 0
    assert api.queue_runway(payload)["ok"] is True  # no built-in cap
    spend = api.runway_spend()
    assert spend["summary"]["today_charged"] == 990 and spend["budget"]["pending_queued"] == 16


def test_a_daily_limit_the_owner_sets_stops_new_jobs(api, home):
    _enable(api, runway_daily_limit=20)
    cloud_ledger.record(desktop.CLOUD_LEDGER_PATH, {"task_id": "x", "status": "SUCCEEDED", "credits_charged": 10})
    payload = _brand_payload(api, home, ratio="1080:1440")
    first = api.queue_runway(payload)
    assert first["ok"] is True and first["budget"]["today_projected"] == 18
    second = api.queue_runway(payload)
    assert second["ok"] is False and "daily limit you set (20)" in second["error"]


# --------------------------------------------------------------------------- integrity


@pytest.mark.usefixtures("measured_local_gpu")
def test_queue_hashes_outputs_and_verify_detects_changes(tmp_path, monkeypatch):
    output = tmp_path / "out.png"
    output.write_bytes((Path(__file__).parent / "fixtures/rc5/still.png").read_bytes())

    class Local:
        def __init__(self, config):
            pass

        def render(self, workflow, job):
            return {"prompt_id": "p", "outputs": [str(output)]}

    monkeypatch.setattr(render_queue, "ComfyUIProvider", Local)
    item = _runway_item(tmp_path, provider_config=ProviderConfig())
    queue = _queue(tmp_path, [item])
    queue._execute(item)
    assert item.output_hashes == {str(output): sha256_file(output)}
    assert verify_hashes(item.output_hashes)["ok"] is True
    output.write_bytes(b"tampered")
    report = verify_hashes(item.output_hashes)
    assert report["changed"] == [str(output)] and report["ok"] is False


def test_verify_outputs_reads_the_history_record(api, home):
    good = home / "good.png"
    good.write_bytes(b"x")
    (home / "history" / "abc123.json").write_text(json.dumps({"output_hashes": {str(good): sha256_file(good)}}))
    assert api.verify_outputs("abc123")["ok"] is True
    good.unlink()
    assert api.verify_outputs("abc123")["missing"] == [str(good)]
    assert api.verify_outputs("../etc")["ok"] is False


# --------------------------------------------------------------------------- performance transfer audio


@needs_ffmpeg
def test_act_two_strips_performer_audio_before_upload_and_mutes_or_replaces_output(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", KEY)
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    performance = _video(tmp_path / "me_teaching.mp4", 3.0, audio=True)
    narration = _audio(tmp_path / "narration.wav", 2.0)
    uploaded = {}

    def handler(request):
        if request.url.path == "/v1/character_performance":
            body = json.loads(request.content)
            uri = body["reference"]["uri"]
            data = __import__("base64").b64decode(uri.split(",", 1)[1])
            (tmp_path / "uploaded.mp4").write_bytes(data)
            uploaded["streams"] = _streams(tmp_path / "uploaded.mp4")
            return httpx.Response(200, json={"id": TASK_ID})
        if request.url.path == f"/v1/tasks/{TASK_ID}":
            return httpx.Response(200, json={"id": TASK_ID, "status": "SUCCEEDED", "output": ["https://cdn.example/out.mp4"]})
        if request.url.host == "cdn.example":
            return httpx.Response(200, content=(tmp_path / "me_teaching.mp4").read_bytes(), headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    item = _runway_item(tmp_path)
    item.job.asset_map.update({"MOTION_VIDEO": str(performance), "RUNWAY_VOICEOVER": str(narration)})
    spec = {"version": 1, "model": "act_two", "params": {"ratio": "720:1280", "expression_intensity": 3, "output_audio": "voiceover"},
            "media": {"character": {"asset": "INIT_IMAGE", "type": "image"}, "reference_video": {"asset": "MOTION_VIDEO"}},
            "local": {"voiceover": {"asset": "RUNWAY_VOICEOVER"}}}
    item.job.asset_map["RUNWAY_SPEC"] = json.dumps(spec)
    config = ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE, output_dir=str(tmp_path / "out"))
    result = RunwayProvider(config, transport=httpx.MockTransport(handler), sleep=lambda _s: None).render(item.workflow, item.job)

    assert uploaded["streams"] == {"video"}  # the performer's voice never left the machine
    assert result["cloud_receipt"]["source_audio"] == "removed before upload"
    voiced = Path(result["outputs"][0])
    assert voiced.name.endswith("_voiceover.mp4") and _streams(voiced) == {"video", "audio"}
    assert not (tmp_path / "out" / "_staging").exists()  # stripped copy cleaned up

    muted = runway_api.replace_audio_track(Path(result["outputs"][1]), "mute")
    assert _streams(muted) == {"video"}


def test_performance_defaults_to_mute_and_uses_any_library_voiceover(api, home):
    _enable(api)
    video = home / "drive.mp4"
    video.write_bytes(b"0" * 4096)
    payload = _brand_payload(api, home, task="performance", refs={"motion_video": str(video), "init_image": str(_png(home / "char.png"))})
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True
    assert any("audio track is removed" in w for w in estimate["warnings"])
    api.queue_runway(_confirm(payload, estimate["estimate"]["credits"]))
    spec = json.loads(api.queue.items[-1].job.asset_map["RUNWAY_SPEC"])
    assert spec["params"]["output_audio"] == "mute" and "local" not in spec
    pending = api.voiceover_import({"paths": [str(_audio(home / "vo.wav"))]})  # no note, no approval step
    assert pending["imported"][0]["status"] == "pending"
    voiced = {**payload, "runway": {**payload["runway"], "voiceover_id": pending["imported"][0]["id"]}}
    usable = api.runway_estimate(voiced)
    assert usable["ok"] is True and all(u["file"] != "vo.wav" for u in usable["uploads"])  # voiceover stays local
    missing = {**payload, "runway": {**payload["runway"], "voiceover_id": "vo_nothere"}}
    assert "no longer in the library" in api.runway_estimate(missing)["error"]


# --------------------------------------------------------------------------- narration and voices


def test_tts_models_validate_build_and_price():
    v4 = runway_api.model_spec("eleven_v4")
    params = {"prompt": "Welcome back. [pause] Today: fractions.", "voice": {"type": "runway-preset", "presetId": "Maya"},
              "speed": 0.95, "previous_text": "Last time we covered decimals.", "seed": 7}
    spec = {"version": 1, "model": "eleven_v4", "params": params, "media": {}}
    runway_api.validate_spec(spec)
    endpoint, body = runway_api.build_request(v4, params, {})
    assert endpoint == "/v1/text_to_speech"
    assert body == {"model": "eleven_v4", "promptText": params["prompt"], "voice": {"type": "runway-preset", "presetId": "Maya"},
                    "speed": 0.95, "previousText": "Last time we covered decimals.", "seed": 7}
    assert runway_api.estimate_credits("eleven_v4", {"prompt": "a" * 2000})["credits"] == 10
    assert runway_api.estimate_credits("eleven_v3", {"prompt": "a" * 120})["credits"] == 3
    with pytest.raises(RunwayValidationError, match="neighbouring"):
        runway_api.validate_spec({**spec, "model": "eleven_v3"})
    with pytest.raises(RunwayValidationError, match="0.7 and 1.2"):
        runway_api.validate_spec({**spec, "params": {**params, "speed": 1.5}})
    with pytest.raises(RunwayValidationError, match="preset voice"):
        runway_api.validate_spec({**spec, "params": {**params, "voice": {"type": "runway-preset", "presetId": "not valid!"}}})
    custom = {**spec, "params": {"prompt": "Hi there", "voice": {"type": "custom", "id": VOICE_UUID}}}
    runway_api.validate_spec(custom)
    with pytest.raises(RunwayValidationError, match="does not support"):
        runway_api.validate_spec({"version": 1, "model": "eleven_multilingual_v2", "params": {"prompt": "x", "voice": {"type": "custom", "id": VOICE_UUID}, "speed": 1.0}})


def test_voice_design_uses_text_descriptions_only():
    requests = []

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/voices/preview":
            return httpx.Response(200, json={"url": "https://cdn.example/p.mp3", "durationSecs": 6.2})
        return httpx.Response(200, json={"id": VOICE_UUID})

    client = RunwayClient(KEY, transport=httpx.MockTransport(handler), sleep=lambda _s: None)
    with client:
        assert client.preview_voice("Warm, clear, confident female presenter voice, calm pace.")["durationSecs"] == 6.2
        assert client.create_designed_voice("BRAND voice", "Warm, clear, confident female presenter voice, calm pace.")["id"] == VOICE_UUID
    assert requests[1][1]["from"] == {"type": "text", "prompt": "Warm, clear, confident female presenter voice, calm pace.", "model": "eleven_ttv_v3"}
    assert not any(name for name in dir(RunwayClient) if "clone" in name.lower())
    with pytest.raises(RunwayValidationError, match="20-1000"):
        runway_api.validate_voice_design("v", "too short", "eleven_ttv_v3")


def test_voiceover_store_imports_without_a_note_and_still_tracks_file_integrity(api, home):
    clip = _audio(home / "lesson1.wav")
    plain = api.voiceover_import({"paths": [str(clip)], "authorization": ""})
    assert plain["ok"] is True and plain["imported"][0]["authorization"] == "Imported by the owner (no note)"
    imported = api.voiceover_import({"paths": [str(_audio(home / "lesson1b.wav", seconds=1.5))], "authorization": "My own recording for BRAND", "label": "Lesson 1"})["imported"][0]
    assert imported["status"] == "pending" and imported["duration_s"] and imported["authorization"] == "My own recording for BRAND"
    store = VoiceoverStore(desktop.VOICEOVER_DIR)
    with pytest.raises(ValueError, match="approved narration"):
        store.approved_path(imported["id"])
    Path(imported["path"]).write_bytes(b"corrupted")
    assert "changed or is missing" in api.voiceover_decide({"id": imported["id"], "decision": "approved"})["error"]


def test_narration_task_registers_generated_audio_as_pending(api, home):
    _enable(api)
    payload = _brand_payload(api, home, task="narration", script="Welcome to BRAND. SQL basics today.", voice_profile_id=bi.VOICE_PROFILE["id"], tts_voice="Leslie")
    api.voice_profile_save({**bi.VOICE_PROFILE, "pronunciations": {"SQL": "sequel"}})
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and estimate["uploads"] == [] and estimate["prompt"] == "Welcome to BRAND. sequel basics today."
    api.queue_runway(_confirm(payload, estimate["estimate"]["credits"]))
    item = api.queue.items[-1]
    spec = json.loads(item.job.asset_map["RUNWAY_SPEC"])
    assert spec["params"]["voice"] == {"type": "runway-preset", "presetId": "Leslie"} and spec["params"]["speed"] == 0.95
    audio = _audio(home / "generated.mp3") if FFMPEG else None
    if audio:
        item.outputs = [str(audio)]
        result = api._candidate_qa(item)
        assert len(result["voiceovers"]) == 1
        assert api.voiceover_list()["items"][0]["status"] == "pending"


def test_script_tools_segment_respell_and_flag():
    assert apply_pronunciations("Use SQL and sql daily", {"SQL": "sequel"}) == "Use sequel and sequel daily"
    text = ("One two three. " * 40).strip()
    pieces = segment_script(text, 100)
    assert all(len(piece) <= 100 for piece in pieces) and " ".join(pieces).split() == text.split()
    stats = script_stats("In 2026 the API returned 42 results. " + " ".join(["word"] * 40) + ".")
    assert any("numerals" in flag for flag in stats["flags"]) and any("API" in flag for flag in stats["flags"])
    assert any("Longest sentence" in flag for flag in stats["flags"])


# --------------------------------------------------------------------------- BRAND profile and identity lock


def test_brand_profile_is_valid_brand_safe_and_installs_idempotently(api, home):
    profile = AvatarProfile.model_validate(bi.profile_entry())
    assert profile.display_name == "BRAND_INSTRUCTOR_PHOTOREAL_V1" and profile.apparent_age_years == 26
    assert profile.nude_mode == "clothed" and profile.subject_kind == "synthetic"
    manifest = AvatarManifest.model_validate(
        {"avatar_id": profile.avatar_id, "display_name": profile.display_name,
         "subjects": [{"id": profile.avatar_id, "kind": "synthetic", "age_verified_18_plus": True, "consent_confirmed": False}]}
    )
    adult = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "x", "content_class": "adult_nudity"})
    assert evaluate_policy(manifest, adult).allowed is False  # adult classes stay locked for the presenter
    assert api.brand_instructor_install()["installed"] is True
    _accept_master(api, home)
    again = api.brand_instructor_install()
    assert again["installed"] is False and again["avatar"]["reference_hashes"]["front"]
    assert api.voice_profiles_list()["items"][0]["id"] == bi.VOICE_PROFILE["id"]


def test_brand_slots_cover_the_brief_and_resolve_for_explicit_acceptance():
    ids = [slot["id"] for slot in bi.REFERENCE_SLOTS]
    assert len(ids) == 9 and len(set(ids)) == 9
    for slot_id in ids:
        assert slot_by_id(slot_id).group == "identity"
    assert "presenter_upper_body" not in {slot.id for slot in COVERAGE_SLOTS}  # H3 designer coverage unchanged


def test_brand_prompt_fits_gen4_with_tag_slot_and_realism_guards(api, home):
    payload = _brand_payload(api, home, task="variations", variations=["presenter_expression_speaking"], ratio="1080:1440")
    _accept_master(api, home)
    payload["avatar"]["identity_refs"] = [api.library.get_avatar(bi.PROFILE_ID)["identity_refs"]["front"]]
    estimate = api.runway_estimate(payload)
    prompt = estimate["prompt"]
    assert estimate["ok"] is True and utf16_len(prompt) <= 1000
    assert prompt.startswith("@presenter, a woman in her mid-20s") and "mouth naturally open" in prompt
    assert "natural matte sheen" in prompt  # BRAND realism guards replace the generic block


def test_identity_check_warns_about_a_different_primary_and_records_hashes(api, home):
    _enable(api)
    payload = _brand_payload(api, home, ratio="1080:1440")
    master = _accept_master(api, home)
    impostor = _png(home / "someone_else.png", color=(10, 200, 30))
    payload["avatar"]["identity_refs"] = [str(impostor)]
    different = api.runway_estimate(payload)
    assert different["ok"] is True  # a warning, not a block
    assert any("not the chosen master photo" in w for w in different["warnings"])
    assert api.queue_runway(payload)["ok"] is True
    payload["avatar"]["identity_refs"] = [str(master)]
    allowed = api.runway_estimate(payload)
    assert allowed["ok"] is True and not any("master photo" in w for w in allowed["warnings"])
    assert allowed["identity"] == {"profile_id": bi.PROFILE_ID, "master_sha256": sha256_file(master), "primary_sha256": sha256_file(master)}


def test_slot_outputs_become_pending_candidates_never_accepted(api, home, monkeypatch):
    _enable(api)
    api.brand_instructor_install()
    master = _accept_master(api, home)
    payload = _brand_payload(api, home, task="variations", variations=["identity_three_quarter_left"], ratio="1080:1440")
    payload["avatar"]["identity_refs"] = [str(master)]
    api.queue_runway(_confirm(payload, 8))
    item = api.queue.items[-1]
    assert item.runtime["candidate_slot"] == "identity_three_quarter_left" and item.runtime["candidate_qa"] is True
    output = _png(home / "candidate.png", color=(180, 140, 120))
    item.outputs = [str(output)]
    import avatar_v2.candidate_qa as cqa

    monkeypatch.setattr(cqa, "compare_candidate_to_reference", lambda *a, **k: {"decision": "review", "scores": {"slot_match": 0.8}, "reasons": ["ok"]})
    result = api._candidate_qa(item)
    assert result["recorded"] == ["candidate.png"]
    profile = AvatarProfile.model_validate(api.library.get_avatar(bi.PROFILE_ID))
    candidate = next(c for c in profile.build_manifest.candidates if c.path == str(output.resolve()))
    assert candidate.status == "pending" and "three_quarter_left" not in profile.identity_refs
    queue_items = api.review_queue()["items"]
    assert any(entry["slot_id"] == "identity_three_quarter_left" for entry in queue_items)


def test_seeded_master_candidates_batch(api, home):
    _enable(api)
    payload = _brand_payload(api, home, ratio="1080:1440", candidates=4, slot="identity_front")
    payload["avatar"]["identity_refs"] = []
    estimate = api.runway_estimate(payload)
    assert estimate["estimate"]["credits"] == 32 and len(estimate["breakdown"]) == 4
    queued = api.queue_runway(_confirm(payload, 32))
    seeds = [json.loads(i.job.asset_map["RUNWAY_SPEC"])["params"]["seed"] for i in api.queue.items]
    assert queued["ok"] is True and seeds == [41001, 41002, 41003, 41004]
    assert all(i.runtime["candidate_slot"] == "identity_front" for i in api.queue.items)


def test_talking_head_uses_library_narration_and_flags_unlocked_avatars(api, home):
    _enable(api)
    api.brand_instructor_install()
    _accept_master(api, home)
    payload = _brand_payload(api, home, task="avatar_video", avatar_id=AVATAR_UUID, speech="audio")
    assert "Pick a voiceover" in api.runway_estimate(payload)["error"]  # it needs some audio to lip-sync to
    record = api.voiceover_import({"paths": [str(_audio(home / "n.wav"))], "authorization": "Hired voice actor, contract #12"})["imported"][0]
    payload["runway"]["voiceover_id"] = record["id"]  # usable without an approval step
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and estimate["uploads"][0]["role"] == "narration audio"
    assert any("not created from the approved master" in w for w in estimate["warnings"])


# --------------------------------------------------------------------------- manual imports (Midjourney)


def test_manual_import_validates_hashes_and_never_auto_accepts(api, home):
    good = _png(home / "mj_upscale.png", size=(1024, 1536))
    small = _png(home / "thumb.png", size=(300, 300))
    gif = home / "anim.gif"
    Image.new("RGB", (800, 800)).save(gif, save_all=True, append_images=[Image.new("RGB", (800, 800), (9, 9, 9))])
    api.brand_instructor_install()
    unattested = api.reference_import({"paths": [str(_png(home / "plain.png", size=(1024, 1536), color=(90, 80, 70)))], "source": "midjourney"})
    assert unattested["ok"] is True and unattested["imported"][0]["attested"] is False  # recorded, not required
    result = api.reference_import(
        {"paths": [str(good), str(good), str(small), str(gif)], "source": "midjourney", "attested": True,
         "prompt": "brunette presenter --ar 2:3", "avatar_id": bi.PROFILE_ID, "slot_id": "identity_front"}
    )
    assert result["ok"] is True and len(result["imported"]) == 1
    reasons = {r["file"]: r["reason"] for r in result["rejected"]}
    assert "duplicate" in reasons["mj_upscale.png"] and "too small" in reasons["thumb.png"] and "GIF" in reasons["anim.gif"]
    stored = Path(result["imported"][0]["path"])
    assert sha256_file(stored) == sha256_file(good)  # bytes unchanged
    profile = AvatarProfile.model_validate(api.library.get_avatar(bi.PROFILE_ID))
    assert profile.identity_refs == {} and profile.build_manifest.candidates[0].status == "pending"
    assert list_imports(desktop.IMPORTS_DIR)[0]["verified"] is True
    body = api.reference_import({"paths": [str(_png(home / "b.png", (900, 900)))], "source": "midjourney", "attested": True,
                                 "avatar_id": bi.PROFILE_ID, "slot_id": "body_full_front"})
    assert body["ok"] is False and "identity or presenter" in body["error"]


# --------------------------------------------------------------------------- realism QA


def test_lag_correlation_and_blink_counting_heuristics():
    reference = [0, 0, 1, 5, 9, 5, 1, 0, 0, 0, 2, 8, 2, 0, 0, 0] * 3
    delayed = [0, 0] + reference[:-2]
    corr, lag = realism_qa.best_lag_correlation(delayed, reference, max_lag=5)
    assert lag == 2 and corr > 0.95
    eyes = [10.0] * 20 + [3.0, 2.5] + [10.0] * 20 + [3.0] + [10.0] * 10
    assert realism_qa.count_blinks(eyes, fps=10) == 2


@needs_ffmpeg
def test_realism_qa_measures_a_clip_and_gates_on_duration_and_review(api, home):
    clip = _video(home / "segment.mp4", 3.0, audio=True, size="360x640")
    master = _png(home / "master_qa.png", size=(360, 640))
    run = api.qa_run({"path": str(clip), "master": str(master), "sample_fps": 8})
    assert run["ok"] is True, run
    report = run["report"]
    for name in ("skin_detail_vs_master", "lighting_continuity", "temporal_flicker", "identity_drift",
                 "facial_geometry_stability", "blink_rate", "lip_sync", "av_duration_alignment", "frozen_or_black_frames"):
        assert name in report["metrics"], name
    assert report["production_gate"]["ready"] is False
    assert any("20-30 s" in reason for reason in report["production_gate"]["reasons"])
    bad = api.qa_review_save({"report_path": run["report_path"], "reviewer": "QA lead", "ratings": {"skin_detail": 5}})
    assert bad["ok"] is False
    ratings = {key: 5 for key, _label in realism_qa.HUMAN_CRITERIA}
    saved = api.qa_review_save({"report_path": run["report_path"], "reviewer": "QA lead", "ratings": ratings})
    assert saved["report"]["production_gate"]["human_ok"] is True and saved["report"]["production_gate"]["ready"] is False
    assert api.qa_review_save({"report_path": str(home / "elsewhere.json"), "reviewer": "x", "ratings": ratings})["ok"] is False
    assert api.qa_reports_list()["items"][0]["reviewed"] is True


def test_production_gate_requires_every_condition():
    report = {"probe": {"duration_s": 25.0}, "metrics": {"lip_sync": {"status": "pass"}},
              "human_review": {"complete": True, "criteria": [{"rating": 5}] * 11}}
    assert realism_qa.production_gate(report)["ready"] is True
    report["metrics"]["lip_sync"]["status"] = "fail"
    assert realism_qa.production_gate(report)["ready"] is False


# --------------------------------------------------------------------------- lessons and provider profiles


def test_lessons_re_render_only_changed_segments(api, home):
    _enable(api)
    api.brand_instructor_install()
    base = _brand_payload(api, home)
    lesson = {"id": "intro-course", "title": "Intro", "profile_id": bi.PROFILE_ID, "runway_avatar_id": AVATAR_UUID,
              "segments": [{"id": "s1", "mode": "script", "script": "Welcome to the course."},
                           {"id": "s2", "mode": "script", "script": "Here is what you will learn."}]}
    assert api.lesson_save({"lesson": lesson})["ok"] is True
    dry = api.lesson_queue_changed({"lesson_id": "intro-course", "base": base, "dry_run": True})
    assert dry["to_render"] == ["s1", "s2"]
    queued = api.lesson_queue_changed(_confirm({"lesson_id": "intro-course", "base": base}, dry["total_credits"]))
    assert queued["ok"] is True and len(api.queue.items) == 2
    store = LessonStore(desktop.LESSONS_PATH)
    video = home / "s1.mp4"
    video.write_bytes(b"v1")
    first = api.queue.items[0]
    first.outputs, first.output_hashes = [str(video)], {str(video): sha256_file(video)}
    api._candidate_qa(first)
    assert {r["segment_id"]: r["state"] for r in store.plan("intro-course", {})}["s1"] == "unchanged"
    lesson["segments"][1]["script"] = "Here is what you will learn today."
    api.lesson_save({"lesson": lesson})
    plan = api.lesson_plan("intro-course")
    assert plan["to_render"] == ["s2"]  # s2's queued render is for the old script, so it is stale
    stale = api.queue.items[1]
    stale.outputs = [str(video)]
    store.record_render("intro-course", "s2", {"status": "queued", "queue_id": "newer-job", "input_hash": "x"})
    assert api._candidate_qa(stale)["lesson_render"] == "stale result ignored"
    video.write_bytes(b"edited")
    assert "s1" in api.lesson_plan("intro-course")["to_render"]


def test_provider_profiles_validate_against_the_registry(api):
    assert api.provider_profile_save({"name": "Master stills", "task": "still", "model": "gen4_image", "ratio": "1080:1440"})["ok"]
    assert "cannot run" in api.provider_profile_save({"name": "bad", "task": "still", "model": "gen4.5"})["error"]
    assert "not supported" in api.provider_profile_save({"name": "bad2", "task": "still", "model": "gen4_image", "ratio": "1:1"})["error"]
    assert [p["name"] for p in api.provider_profiles_list()["items"]] == ["Master stills"]
    assert api.provider_profile_delete("Master stills")["removed"] is True


def test_provider_status_reports_each_lane(api):
    api.save_settings({"comfyui_url": "http://127.0.0.1:9", "vision_classifier_url": "http://127.0.0.1:9"})
    status = api.provider_status()["providers"]
    assert status["comfyui"]["available"] is False and status["local_vision"]["available"] is False
    assert status["midjourney"]["mode"] == "manual import only"
    assert status["runway"]["available"] is False and "API key" in status["runway"]["detail"]
