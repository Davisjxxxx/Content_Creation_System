"""Desktop, policy, prompt-compiler and queue integration for the opt-in Runway provider."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from avatar_v2 import desktop, render_queue, secret_store
from avatar_v2.integrity import sha256_file
from avatar_v2.composer import build_job
from avatar_v2.desktop import DesktopAPI
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.policy import evaluate_cloud_policy
from avatar_v2.presenter import PRESENTER_VARIATIONS, compile_presenter_prompt, subject_phrase
from avatar_v2.providers import runway as runway_api
from avatar_v2.providers.runway import RunwayTaskFailed, utf16_len
from avatar_v2.render_queue import QueuedRender, RenderQueue

KEY = "rw_test_key_0123456789abcdef"
AVATAR_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _png(path: Path, size=(720, 1280)) -> Path:
    Image.new("RGB", size, (200, 160, 140)).save(path)
    return path


class FakeMemory:
    def __init__(self):
        self.calls = []

    def update_settings(self, **_kwargs):
        pass

    def maybe_cleanup(self, provider, **kwargs):
        self.calls.append(kwargs)
        return {"reason": kwargs["reason"], "forced": kwargs["force"]}

    def start_trace(self, _label):
        pass

    def stop_trace(self):
        return {"peak": []}


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


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "desktop.json")
    monkeypatch.setattr(desktop, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(desktop, "RUNWAY_AVATARS_PATH", tmp_path / "runway_avatars.json")
    monkeypatch.setattr(secret_store, "SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)
    instance = DesktopAPI.__new__(DesktopAPI)
    instance.queue = FakeQueue()
    return instance


def _enable(api, *, key=True, **extra):
    """Runway needs only a saved key; ``extra`` sets optional owner limits."""
    if extra:
        api.save_settings(extra)
    if key:
        assert api.runway_save_key(KEY)["ok"] is True


def _payload(tmp_path, *, task="still", kind="synthetic", content="general", refs=None, **options):
    front = _png(tmp_path / "front.png")
    return {
        "runtime_profile": "auto",
        "settings": {},
        "avatar": {
            "avatar_id": "brand-presenter",
            "display_name": "Course Presenter",
            "sex": "female",
            "apparent_age_years": 25,
            "subject_kind": kind,
            "age_verified_18_plus": True,
            "consent_confirmed": True,
            "identity_refs": [str(front)],
            "body_refs": [str(_png(tmp_path / "body.png"))],
            "hair_refs": [],
            "wardrobe_refs": [],
            "persistent_features": ["small beauty mark above the left lip"],
        },
        "shot": {
            "shot_id": "desktop-shot",
            "content_class": content,
            "user_prompt": "Warm, confident smile while introducing today's lesson.",
            "duration_s": 6,
            "seed": 41001,
            "engine_preference": "runway",
            "camera": {"framing": "medium portrait", "movement": "locked-off tripod", "lens_equivalent_mm": 50},
            "wind": {},
            "references": refs or {},
        },
        "runway": {"task": task, **options},
    }


def _confirm(payload, credits):
    """Legacy confirmation fields: still accepted, no longer required."""
    return {**payload, "confirm_cloud_upload": True, "confirmed_estimate_credits": credits}


# --------------------------------------------------------------------------- settings and secrets


def test_settings_never_persist_or_expose_credentials(api, tmp_path):
    saved = api.save_settings(
        {
            "runway_api_key": "x" * 32,
            "RUNWAY_SECRET": "y" * 32,
            "runway_job_limit": 999999,
            "runway_daily_limit": -5,
            "runway_default_video_model": "gen4_image",
        }
    )["settings"]
    stored = json.loads((tmp_path / "desktop.json").read_text())
    for blob in (saved, stored):
        assert "runway_api_key" not in blob and "RUNWAY_SECRET" not in blob
    assert stored["runway_job_limit"] == 100000  # clamped to the documented range
    assert stored["runway_daily_limit"] == 0  # 0 means no limit
    assert stored["runway_monthly_limit"] == 0 and stored["runway_batch_limit"] == 0  # off by default
    assert stored["runway_default_video_model"] == "gen4.5"

    (tmp_path / "desktop.json").write_text(json.dumps({"legacy_api_key": "z" * 32}))
    state = api.get_state()
    assert "legacy_api_key" not in state["settings"]
    assert "z" * 32 not in json.dumps(state)
    assert {engine["id"] for engine in state["engines"]} >= {"h3_ref2va", "runway"}
    assert state["runway"]["ready"] is False and state["runway"]["key"]["configured"] is False
    assert state["runway"]["limits"] == {"job": 0, "batch": 0, "daily": 0, "monthly": 0}


def test_key_management_never_echoes_the_key(api):
    assert api.runway_save_key("short")["ok"] is False
    saved = api.runway_save_key(KEY)
    assert saved["ok"] is True and saved["stored_in"] == "file"
    assert KEY not in json.dumps(saved)
    assert api.get_state()["runway"]["key"] == {
        "configured": True,
        "source": "file",
        "env_var": "RUNWAYML_API_SECRET",
        "keyring_available": False,
    }
    assert KEY not in json.dumps(api.get_state())
    cleared = api.runway_clear_key()
    assert cleared["removed"] is True and cleared["key"]["configured"] is False


def test_local_queue_render_refuses_cloud_engine(api, tmp_path):
    result = api.queue_render(_payload(tmp_path))
    assert result["ok"] is False and "Queue on Runway" in result["error"]
    assert api.queue.items == []


# --------------------------------------------------------------------------- gates


def test_queue_runway_needs_only_a_key_and_no_confirmation(api, tmp_path):
    payload = _payload(tmp_path, ratio="1080:1440")
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and "API key" in estimate["gate"]
    credits = estimate["estimate"]["credits"]
    assert credits == 8 and estimate["within_cap"] is True and estimate["cap"] == 0

    missing = api.queue_runway(payload)
    assert missing["ok"] is False and "API key" in missing["error"]
    assert api.queue.items == []
    api.runway_save_key(KEY)
    assert api.runway_estimate(payload)["gate"] is None

    result = api.queue_runway(payload)  # no enable switch, consent box, confirmation or estimate echo
    assert result["ok"] is True and result["total_credits"] == 8
    item = api.queue.items[0]
    assert item.provider_config.provider == "runway"
    assert item.provider_config.base_url == runway_api.RUNWAY_API_BASE
    assert item.profile == "cloud_runway" and item.workflow == "runway:still:gen4_image"
    assert item.job.shot.local_only is False and item.job.selected_engine == "runway"
    spec = json.loads(item.job.asset_map["RUNWAY_SPEC"])
    assert spec["max_credits"] is None and spec["estimate"]["credits"] == 8  # no per-job limit by default
    assert spec["media"]["references"] == [{"asset": "IDENTITY_REF_1", "tag": "presenter", "role": "primary identity"}]
    assert "@presenter" in spec["params"]["prompt"]


def test_payload_settings_cannot_change_the_owners_limits(api, tmp_path):
    _enable(api, runway_job_limit=5)
    payload = _payload(tmp_path)
    payload["settings"] = {"runway_job_limit": 0, "runway_daily_limit": 0}
    blocked = api.queue_runway(payload)
    assert blocked["ok"] is False and "limit you set" in blocked["error"]
    assert api.queue.items == []


def test_cloud_policy_blocks_adult_classes_and_unknown_provenance(api, tmp_path):
    adult = api.runway_estimate(_payload(tmp_path, content="adult_nudity", kind="synthetic"))
    assert adult["ok"] is False and "general content only" in adult["error"]
    unknown = api.runway_estimate(_payload(tmp_path, kind="unknown"))
    assert unknown["ok"] is False and "declared provenance" in unknown["error"]

    avatar = AvatarManifest.model_validate(
        {
            "avatar_id": "real-person",
            "display_name": "Real",
            "subjects": [{"id": "real-person", "kind": "consenting_adult", "age_verified_18_plus": True, "consent_confirmed": False}],
        }
    )
    shot = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "hi", "engine_preference": "runway"})
    assert evaluate_cloud_policy(avatar, shot).allowed is False


def test_per_job_limit_applies_only_when_the_owner_sets_one(api, tmp_path):
    payload = _payload(tmp_path, task="image_to_video", model="gen4.5", ratio="720:1280", duration=5)
    _enable(api)
    unlimited = api.runway_estimate(payload)
    assert unlimited["estimate"]["credits"] == 60 and unlimited["within_cap"] is True
    assert api.queue_runway(payload)["ok"] is True

    api.queue.items.clear()
    api.save_settings({"runway_job_limit": 10})
    estimate = api.runway_estimate(payload)
    assert estimate["within_cap"] is False and "per-job limit you set (10)" in estimate["limit_error"]
    result = api.queue_runway(payload)
    assert result["ok"] is False and "per-job limit you set" in result["error"]
    assert api.queue.items == []
    spec_limit = api._runway_build(payload)["spec"]["max_credits"]
    assert spec_limit == 10  # the provider cancels a task Runway estimates above it


# --------------------------------------------------------------------------- task builds


def test_still_uploads_only_identity_never_body_or_anatomy_refs(api, tmp_path):
    estimate = api.runway_estimate(_payload(tmp_path))
    assert [item["file"] for item in estimate["uploads"]] == ["front.png"]
    assert estimate["uploads"][0]["transfer"] == "inline data URI"
    assert estimate["prompt_receipt"]["chars"] <= 1000
    assert "a woman in her mid-20s" in estimate["prompt"]
    assert "small beauty mark above the left lip" in estimate["prompt"]


def test_text_only_still_designs_a_new_presenter(api, tmp_path):
    payload = _payload(tmp_path)
    payload["avatar"]["identity_refs"] = []
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and estimate["uploads"] == []
    assert "@presenter" not in estimate["prompt"]
    assert any("pure text-to-image" in warning for warning in estimate["warnings"])
    turbo = api.runway_estimate({**payload, "runway": {"task": "still", "model": "gen4_image_turbo"}})
    assert turbo["ok"] is False and "at least 1 reference" in turbo["error"]


def test_gemini_still_marks_identity_angles_as_human_subjects(api, tmp_path):
    payload = _payload(tmp_path, model="gemini_image3.1_flash", ratio="832:1248", output_count=4)
    extra = [str(_png(tmp_path / f"angle{i}.png")) for i in range(2, 8)]
    payload["avatar"]["identity_refs"] += extra
    payload["avatar"]["wardrobe_refs"] = [str(_png(tmp_path / "outfit.png"))]
    payload["shot"]["references"] = {"scene_image": str(_png(tmp_path / "classroom.png"))}
    _enable(api)
    estimate = api.runway_estimate(payload)
    assert estimate["estimate"]["credits"] == 28
    queued = api.queue_runway(_confirm(payload, 28))
    spec = json.loads(api.queue.items[0].job.asset_map["RUNWAY_SPEC"])
    references = spec["media"]["references"]
    assert sum(1 for ref in references if ref.get("subject") == "human") == 5
    assert [ref.get("tag") for ref in references if ref.get("tag")] == ["presenter", "outfit", "location"]
    assert "@outfit" in spec["params"]["prompt"] and "@location" in spec["params"]["prompt"]
    assert "seed" not in spec["params"]
    assert queued["ok"] is True


def test_image_to_video_builds_with_h3_max_keyframes_and_motion_prompt(api, tmp_path):
    last = str(_png(tmp_path / "end.png"))
    payload = _payload(tmp_path, task="image_to_video", model="h3_max", duration=8, resolution="768p", refs={"last_frame": last})
    _enable(api)
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and estimate["estimate"]["credits"] == 64
    api.queue_runway(_confirm(payload, 64))
    spec = json.loads(api.queue.items[0].job.asset_map["RUNWAY_SPEC"])
    assert spec["params"]["prompt_expansion"] == "disabled" and spec["params"]["resolution"] == "768p"
    assert spec["media"]["last_frame"]["optional"] is True
    assert "Natural blinking" in spec["params"]["prompt"] and "@presenter" not in spec["params"]["prompt"]
    assert api.queue.items[0].job.shot.duration_s == 8


def test_gen45_warns_about_center_crop_and_ignored_last_frame(api, tmp_path):
    payload = _payload(
        tmp_path, task="image_to_video", model="gen4.5", ratio="1280:720", duration=5, refs={"last_frame": str(_png(tmp_path / "end.png"))}
    )
    warnings = api.runway_estimate(payload)["warnings"]
    assert any("center-crop" in warning for warning in warnings)
    assert any("ignores the last frame" in warning for warning in warnings)


def test_talking_presenter_video_uses_script_and_custom_avatar(api, tmp_path):
    script = "Welcome back. Today we will learn how compound interest grows over time."
    missing = api.runway_estimate(_payload(tmp_path, task="avatar_video", script=script))
    assert missing["ok"] is False and "presenter avatar" in missing["error"]
    payload = _payload(tmp_path, task="avatar_video", avatar_id=AVATAR_UUID, script=script, voice="clara")
    _enable(api)
    estimate = api.runway_estimate(payload)
    assert estimate["ok"] is True and estimate["uploads"] == []
    credits = estimate["estimate"]["credits"]
    assert api.queue_runway(_confirm(payload, credits))["ok"] is True
    item = api.queue.items[0]
    spec = json.loads(item.job.asset_map["RUNWAY_SPEC"])
    assert spec["params"] == {
        "avatar": {"type": "custom", "avatarId": AVATAR_UUID},
        "speech": {"type": "text", "text": script, "voice": {"type": "preset", "presetId": "clara"}},
    }
    assert item.job.prompt == script


def test_performance_transfer_needs_a_motion_video(api, tmp_path):
    assert "performance video" in api.runway_estimate(_payload(tmp_path, task="performance"))["error"]
    video = tmp_path / "me_teaching.mp4"
    video.write_bytes(b"0" * 2048)
    estimate = api.runway_estimate(_payload(tmp_path, task="performance", refs={"motion_video": str(video)}))
    assert estimate["ok"] is True
    assert {item["file"] for item in estimate["uploads"]} == {"front.png", "me_teaching.mp4"}
    assert estimate["estimate"]["upper_bound"] is True  # unreadable test video: priced at the 30 s maximum


def test_upscale_prices_from_the_real_image_size(api, tmp_path):
    # The primary still is 720x1280: 2x gives a 2560 px long edge (25 credits); 4x gives 5120 px (150).
    double = api.runway_estimate(_payload(tmp_path, task="upscale", scale_factor=2))
    assert double["ok"] is True and double["estimate"]["credits"] == 25 and double["estimate"]["upper_bound"] is False
    quadruple = api.runway_estimate(_payload(tmp_path, task="upscale", scale_factor=4))
    assert quadruple["estimate"]["credits"] == 150
    explicit = api.runway_estimate(_payload(tmp_path, task="upscale", scale_factor=2, image=str(_png(tmp_path / "s.png", (640, 960)))))
    assert [item["file"] for item in explicit["uploads"]] == ["s.png"]


def test_variation_batch_reuses_identity_and_seed_and_respects_a_set_batch_limit(api, tmp_path):
    ids = [item["id"] for item in PRESENTER_VARIATIONS[:3]]
    payload = _payload(tmp_path, task="variations", ratio="1080:1440", variations=ids)
    _enable(api)
    estimate = api.runway_estimate(payload)
    assert estimate["estimate"]["credits"] == 24 and len(estimate["breakdown"]) == 3
    result = api.queue_runway(payload)
    assert result["ok"] is True and [entry["variation"] for entry in result["queued"]] == ids
    specs = [json.loads(item.job.asset_map["RUNWAY_SPEC"]) for item in api.queue.items]
    assert {spec["params"]["seed"] for spec in specs} == {41001}
    for spec, variation in zip(specs, PRESENTER_VARIATIONS[:3]):
        assert variation["prompt"].split(";")[0][:40] in spec["params"]["prompt"]
        assert spec["media"]["references"][0]["asset"] == "IDENTITY_REF_1"
    assert "presenter variation" in api.queue.items[0].label

    api.queue.items.clear()
    api.save_settings({"runway_batch_limit": 10})
    blocked = api.queue_runway(payload)
    assert blocked["ok"] is False and "batch limit you set" in blocked["error"]
    assert "Choose at least one" in api.runway_estimate(_payload(tmp_path, task="variations", variations=[]))["error"]


# --------------------------------------------------------------------------- account and avatars (network mocked)


class FakeClient:
    def __init__(self, **responses):
        self.responses = responses
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def organization(self):
        return {"creditBalance": 1234, "tier": {"maxMonthlyCreditSpend": 50000, "models": {"gen4.5": {}}}}

    def upload(self, filename, content_type, data):
        self.calls.append(("upload", filename))
        return "runway://upload-1"

    def create_avatar(self, body):
        self.calls.append(("create_avatar", body))
        return {"id": AVATAR_UUID, "name": body["name"], "status": "PROCESSING"}

    def list_avatars(self, limit=100):
        return {"data": [{"id": AVATAR_UUID, "name": "Course Presenter", "status": "READY", "voice": {"presetId": "clara"}}], "hasMore": False}

    def delete_avatar(self, avatar_id):
        self.calls.append(("delete_avatar", avatar_id))


def test_account_check_needs_only_a_key_and_reports_balance(api, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(runway_api, "client_from_store", lambda **_kw: fake)
    assert "API key" in api.runway_check()["error"]
    _enable(api)
    result = api.runway_check()
    assert result == {"ok": True, "credit_balance": 1234, "max_monthly_credit_spend": 50000, "models_with_limits": ["gen4.5"]}


def test_avatar_creation_uploads_primary_identity_and_links_the_vault_profile(api, tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(runway_api, "client_from_store", lambda **_kw: fake)
    payload = _payload(tmp_path)
    request = {"avatar": payload["avatar"], "name": "Course Presenter", "voice": "clara"}
    assert "API key" in api.runway_avatar_create(request)["error"]
    _enable(api)
    assert "avatar voices" in api.runway_avatar_create({**request, "voice": "robot"})["error"]
    unknown = {**request, "avatar": {**payload["avatar"], "subject_kind": "unknown"}}
    assert "declared provenance" in api.runway_avatar_create(unknown)["error"]  # third-party likeness rule stays

    created = api.runway_avatar_create(request)  # no confirmation step
    assert created["ok"] is True and created["avatar"]["id"] == AVATAR_UUID
    body = next(call[1] for call in fake.calls if call[0] == "create_avatar")
    assert body["voice"] == {"type": "runway-live-preset", "presetId": "clara"}
    assert body["referenceImage"].startswith("data:image/png;base64,")
    assert body["imageProcessing"] == "optimize" and "Course Presenter" in body["personality"]
    index = json.loads((tmp_path / "runway_avatars.json").read_text())
    assert index[AVATAR_UUID]["vault_avatar_id"] == "brand-presenter"
    assert index[AVATAR_UUID]["reference_file"] == "front.png"
    assert index[AVATAR_UUID]["source_sha256"] == sha256_file(tmp_path / "front.png")

    listed = api.runway_avatars_list()
    assert listed["items"][0]["vault_avatar_id"] == "brand-presenter"
    assert api.runway_avatar_delete("not-a-uuid")["ok"] is False
    assert api.runway_avatar_delete(AVATAR_UUID)["ok"] is True
    assert AVATAR_UUID not in json.loads((tmp_path / "runway_avatars.json").read_text())


# --------------------------------------------------------------------------- prompt compiler


def _avatar(**updates) -> AvatarManifest:
    data = {
        "avatar_id": "p",
        "display_name": "P",
        "sex": "female",
        "apparent_age_years": 25,
        "subjects": [{"id": "p", "kind": "synthetic"}],
        "persistent_features": ["light freckles across the nose"],
    }
    data.update(updates)
    return AvatarManifest.model_validate(data)


def test_subject_phrase_describes_adult_age_without_identity_claims():
    assert subject_phrase(_avatar()) == "a woman in her mid-20s"
    assert subject_phrase(_avatar(apparent_age_years=31)) == "a woman in her early 30s"
    assert subject_phrase(_avatar(sex="", apparent_age_years=None)) == "an adult person"
    assert subject_phrase(_avatar(sex="male", apparent_age_years=19)) == "an adult man aged 19"


def test_compiler_keeps_tags_and_drops_low_priority_sections_to_fit():
    shot = ShotSpec.model_validate(
        {"shot_id": "s", "user_prompt": "explains fractions with a calm smile " * 20, "engine_preference": "runway", "wardrobe": "navy blazer"}
    )
    prompt, dropped = compile_presenter_prompt(
        _avatar(), shot, budget=1000, identity_tag="presenter", location_tag="location", variation=PRESENTER_VARIATIONS[2]
    )
    assert utf16_len(prompt) <= 1000
    assert "@presenter" in prompt and "@location" in prompt
    assert dropped[0] == "no_text" and "realism" in dropped

    tight, dropped_tight = compile_presenter_prompt(_avatar(), shot, budget=300, identity_tag="presenter")
    assert utf16_len(tight) <= 300 and "@presenter" in tight
    assert "direction (shortened)" in dropped_tight


def test_compiler_includes_motion_cues_and_timed_beats_for_video():
    shot = ShotSpec.model_validate(
        {
            "shot_id": "s",
            "user_prompt": "turns from the whiteboard to the camera",
            "engine_preference": "runway",
            "duration_s": 6,
            "action_beats": [{"start_s": 0, "end_s": 2, "description": "looks at board"}, {"start_s": 2, "end_s": 6, "description": "turns and smiles"}],
        }
    )
    prompt, _dropped = compile_presenter_prompt(_avatar(), shot, budget=1000, motion=True)
    assert "[0-2s] looks at board" in prompt and "Natural blinking" in prompt
    assert "Photorealistic photograph" not in prompt


# --------------------------------------------------------------------------- queue routing


def _queue(tmp_path, items):
    queue = RenderQueue.__new__(RenderQueue)
    queue.work_dir = tmp_path / "queue"
    queue.work_dir.mkdir(parents=True, exist_ok=True)
    queue.history_dir = tmp_path / "history"
    queue.history_dir.mkdir(parents=True, exist_ok=True)
    queue._items = list(items)
    queue._lock = threading.RLock()
    queue._wake = threading.Event()
    queue._stop = threading.Event()
    queue._active_provider = None
    queue._active_id = None
    queue.candidate_qa_runner = None
    queue.memory_manager = FakeMemory()
    return queue


def _cloud_item(tmp_path, profile="cloud_runway"):
    avatar = _avatar(identity_refs=[str(_png(tmp_path / "q.png"))])
    shot = ShotSpec.model_validate({"shot_id": "cloud", "user_prompt": "smile", "engine_preference": "runway"})
    return QueuedRender(
        job=build_job(avatar, shot),
        workflow="runway:still:gen4_image",
        provider_config=ProviderConfig(provider="runway", base_url=runway_api.RUNWAY_API_BASE),
        profile=profile,
        label="cloud",
    )


def test_cloud_jobs_route_to_runway_and_skip_local_gpu_cleanup(tmp_path, monkeypatch):
    output = tmp_path / "out.png"
    output.write_bytes(b"png")

    class FakeRunway:
        uses_local_gpu = False

        def __init__(self, config):
            assert config.provider == "runway"

        def render(self, workflow, job):
            return {"prompt_id": "task-1", "outputs": [str(output)], "cloud_receipt": {"credits_charged": 8}}

    monkeypatch.setattr(render_queue, "RunwayProvider", FakeRunway)
    item = _cloud_item(tmp_path, profile="low_memory")  # even a memory-sensitive profile name must not trigger /free
    queue = _queue(tmp_path, [item])
    queue._execute(item)
    assert queue.memory_manager.calls == []
    assert item.memory_events[0]["skipped"] is True
    assert item.cloud_receipt["credits_charged"] == 8
    assert item.cloud_receipt["submission"] == {"state": "preparing"}  # durable pre-send marker
    public = item.public()
    assert public["provider"] == "runway" and public["cloud_receipt"]["credits_charged"] == 8
    restored = queue._item_from_dict(queue._item_to_dict(item))
    assert restored.cloud_receipt["credits_charged"] == 8 and restored.provider_config.provider == "runway"


def test_local_comfyui_jobs_keep_forced_preclean(tmp_path, monkeypatch):
    class FakeComfy:
        def __init__(self, config):
            pass

        def render(self, workflow, job):
            return {"prompt_id": "p", "outputs": []}

    monkeypatch.setattr(render_queue, "ComfyUIProvider", FakeComfy)
    item = _cloud_item(tmp_path, profile="int8_12gb")
    item.provider_config = ProviderConfig()
    queue = _queue(tmp_path, [item])
    queue._execute(item)
    assert queue.memory_manager.calls[0]["force"] is True
    assert queue.memory_manager.calls[0]["reason"] == "pre-render:int8_12gb"


def test_worker_keeps_the_cloud_receipt_when_a_runway_job_fails(tmp_path, monkeypatch):
    class FailingRunway:
        uses_local_gpu = False

        def __init__(self, config):
            pass

        def render(self, workflow, job):
            raise RunwayTaskFailed("task-9", "moderation", "SAFETY.INPUT.TEXT", receipt={"task_id": "task-9", "credits_charged": 0})

    monkeypatch.setattr(render_queue, "RunwayProvider", FailingRunway)
    item = _cloud_item(tmp_path)
    queue = _queue(tmp_path, [item])
    worker = threading.Thread(target=queue._worker, daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while item.finished_at is None and time.monotonic() < deadline:
        time.sleep(0.05)
    queue._stop.set()
    queue._wake.set()
    worker.join(timeout=5)
    assert item.status == "error"
    assert item.error_category == "cloud_moderation"
    assert item.cloud_receipt["task_id"] == "task-9"
    assert [(event["reason"], event["skipped"]) for event in item.memory_events] == [
        ("pre-render:cloud_runway", True),
        ("post-render:error", True),
    ]
    assert queue.memory_manager.calls == []
    history = json.loads((tmp_path / "history" / f"{item.id}.json").read_text())
    assert history["cloud_receipt"]["task_id"] == "task-9"


def test_cloud_and_adult_policies_refuse_prompts_describing_minors(api, tmp_path):
    from avatar_v2.policy import evaluate_policy

    minor = _payload(tmp_path)
    minor["shot"]["user_prompt"] = "Caucasion 15-year-old girl: attractive, intelligent, approachable."
    refused = api.runway_estimate(minor)
    assert refused["ok"] is False and "has to be an adult" in refused["error"]
    adult = _payload(tmp_path)
    adult["shot"]["user_prompt"] = "Fictional 18-year-old woman with minor asymmetry and baby hairs at the hairline."
    assert api.runway_estimate(adult)["ok"] is True

    avatar = AvatarManifest.model_validate(
        {"avatar_id": "s", "display_name": "S", "subjects": [{"id": "s", "kind": "synthetic", "age_verified_18_plus": True, "consent_confirmed": True}]}
    )
    teen = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "a teen on a bed", "content_class": "adult_nudity"})
    assert evaluate_policy(avatar, teen).allowed is False
    grown = ShotSpec.model_validate({"shot_id": "s", "user_prompt": "a 30-year-old woman", "content_class": "adult_nudity"})
    assert evaluate_policy(avatar, grown).allowed is True
    with pytest.raises(ValueError):
        AvatarManifest.model_validate({"avatar_id": "y", "display_name": "Y", "apparent_age_years": 15, "subjects": []})
