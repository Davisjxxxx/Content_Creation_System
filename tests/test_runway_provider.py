"""Runway cloud adapter: request shapes, limits, costs, media handling, polling and safety rails.

All HTTP goes through httpx.MockTransport; nothing here contacts Runway or spends credits.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
from pathlib import Path

import httpx
import pytest
from PIL import Image

from avatar_v2 import secret_store
from avatar_v2.composer import build_job
from avatar_v2.errors import classify_exception
from avatar_v2.models import AvatarManifest, ProviderConfig, ShotSpec
from avatar_v2.providers import runway
from avatar_v2.providers.runway import (
    RunwayAPIError,
    RunwayBudgetError,
    RunwayClient,
    RunwayConfigError,
    RunwayProvider,
    RunwayTaskCancelled,
    RunwayTaskFailed,
    RunwayTimeout,
    RunwayValidationError,
)

API_KEY = "rw_test_key_0123456789abcdef"
TASK_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def isolated_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(secret_store, "SECRETS_DIR", tmp_path / "secrets")
    monkeypatch.setattr(secret_store, "_keyring", lambda: None)
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)


def _png(path: Path, size=(96, 128), color=(180, 120, 90)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


class Recorder:
    """MockTransport handler that records requests and serves scripted responses."""

    def __init__(self, routes):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.host, request.url.path)
        handler = self.routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": f"unrouted {key}"})
        if isinstance(handler, list):
            handler = handler.pop(0) if len(handler) > 1 else handler[0]
        return handler(request) if callable(handler) else handler

    def find(self, method, path):
        return [r for r in self.requests if r.method == method and r.url.path == path]


def _client(recorder, **kwargs) -> RunwayClient:
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("jitter", lambda: 0.0)
    return RunwayClient(API_KEY, transport=httpx.MockTransport(recorder), **kwargs)


# --------------------------------------------------------------------------- registry and costs


def test_every_registered_model_has_a_cost_model_and_public_payload():
    for model in runway.MODELS.values():
        params = {
            "ratio": model.ratios[0] if model.ratios else None,
            "duration": model.default_duration,
            "resolution": model.resolutions[0] if model.resolutions else None,
            "speech": {"type": "text", "text": "Hello"},
        }
        estimate = runway.estimate_credits(model.id, params)
        assert estimate["credits"] >= 1
        assert estimate["usd"] == round(estimate["credits"] * 0.01, 2)
        assert model.public()["endpoint"].startswith("/v1/")


@pytest.mark.parametrize(
    "model_id,params,facts,credits",
    [
        ("gen4_image", {"ratio": "1280:720"}, None, 5),
        ("gen4_image", {"ratio": "1920:1080"}, None, 8),
        ("gen4_image_turbo", {"ratio": "720:1280"}, None, 2),
        ("gemini_image3.1_flash", {"ratio": "1024:1024", "output_count": 4}, None, 28),
        ("gemini_image3.1_flash", {"ratio": "2048:2048", "output_count": 1}, None, 11),
        ("gemini_image3.1_flash", {"ratio": "512:512", "output_count": 1}, None, 5),
        ("gen4.5", {"duration": 10}, None, 120),
        ("gen4_turbo", {"duration": 4}, None, 20),
        ("h3_max", {"duration": 15, "resolution": "480p"}, None, 75),
        ("h3_max", {"duration": 15, "resolution": "768p"}, None, 120),
        ("act_two", {}, {"reference_video_s": 12.2}, 65),
        ("magnific_precision_upscaler_v2", {"scale_factor": 2}, {"image_size": (1024, 1536)}, 25),
        ("magnific_precision_upscaler_v2", {"scale_factor": 4}, {"image_size": (1024, 1536)}, 150),
    ],
)
def test_estimates_follow_documented_pricing(model_id, params, facts, credits):
    assert runway.estimate_credits(model_id, params, facts)["credits"] == credits


def test_avatar_video_estimate_counts_video_seconds_and_script_characters():
    text = " ".join(["word"] * 150)  # ~60 s at 150 wpm, 749 characters
    estimate = runway.estimate_credits("gwm1_avatars", {"speech": {"type": "text", "text": text}})
    assert estimate["credits"] == 20 + 15
    unknown_audio = runway.estimate_credits("gwm1_avatars", {"speech": {"type": "audio"}})
    assert unknown_audio["upper_bound"] is True


def test_unknown_model_is_rejected_not_passed_through():
    with pytest.raises(RunwayValidationError, match="not registered"):
        runway.model_spec("gen9_mystery")


def test_utf16_length_counts_surrogate_pairs():
    assert runway.utf16_len("abc") == 3
    assert runway.utf16_len("🙂") == 2


# --------------------------------------------------------------------------- validation and request shapes


def _spec(model, params, media=None):
    return {"version": 1, "model": model, "params": params, "media": media or {}}


@pytest.mark.parametrize(
    "spec,message",
    [
        (_spec("gen4_image", {"prompt": "x", "ratio": "999:999"}), "not supported"),
        (_spec("gen4_image", {"prompt": "", "ratio": "1280:720"}), "needs a prompt"),
        (_spec("gen4_image", {"prompt": "🙂" * 501, "ratio": "1280:720"}), "UTF-16"),
        (_spec("gen4.5", {"prompt": "walk", "ratio": "1280:720", "duration": 11}, {"first_frame": {"path": "a.png"}}), "duration"),
        (_spec("gen4_image_turbo", {"prompt": "x", "ratio": "1280:720"}), "1-3 reference"),
        (
            _spec(
                "gen4_image",
                {"prompt": "@presenter waves", "ratio": "1280:720"},
                {"references": [{"path": "a.png", "tag": "1bad"}]},
            ),
            "must be 3-16",
        ),
        (
            _spec(
                "gen4_image",
                {"prompt": "@face and @face", "ratio": "1280:720"},
                {"references": [{"path": "a.png", "tag": "face"}, {"path": "b.png", "tag": "face"}]},
            ),
            "unique",
        ),
        (
            _spec("gen4_image", {"prompt": "no mention", "ratio": "1280:720"}, {"references": [{"path": "a.png", "tag": "presenter"}]}),
            "must mention @presenter",
        ),
        (
            _spec("gen4_image", {"prompt": "x", "ratio": "1280:720"}, {"references": [{"path": "a.png", "subject": "human"}]}),
            "human-subject",
        ),
        (
            _spec(
                "gemini_image3.1_flash",
                {"prompt": "x", "ratio": "1024:1024"},
                {"references": [{"path": f"{i}.png", "subject": "human"} for i in range(6)]},
            ),
            "at most 5",
        ),
        (_spec("gen4_image", {"prompt": "x", "ratio": "1280:720", "seed": -1}), "Seed"),
        (_spec("gemini_image3.1_flash", {"prompt": "x", "ratio": "1024:1024", "seed": 5}), "does not accept a seed"),
        (_spec("gemini_image3.1_flash", {"prompt": "x", "ratio": "1024:1024", "output_count": 2}), "output count"),
        (
            _spec("gen4.5", {"prompt": "walk", "ratio": "1280:720", "duration": 5}, {"first_frame": {"path": "a.png"}, "last_frame": {"path": "b.png"}}),
            "does not accept a last frame",
        ),
        (_spec("gen4.5", {"prompt": "walk", "ratio": "1280:720", "duration": 5}), "first-frame"),
        (
            _spec("h3_max", {"prompt": "walk", "resolution": "1080p", "duration": 6, "prompt_expansion": "disabled"}, {"first_frame": {"path": "a.png"}}),
            "supports 768p",
        ),
        (_spec("act_two", {"expression_intensity": 3}, {"character": {"path": "a.png", "type": "image"}}), "performance reference"),
        (_spec("act_two", {"prompt": "dance"}, {}), "does not take a text prompt"),
        (_spec("gwm1_avatars", {"avatar": {"type": "custom", "avatarId": "nope"}, "speech": {"type": "text", "text": "hi"}}), "UUID"),
        (
            _spec("gwm1_avatars", {"avatar": {"type": "custom", "avatarId": TASK_ID}, "speech": {"type": "text", "text": "a" * 5001}}),
            "5000",
        ),
        (
            _spec(
                "gwm1_avatars",
                {"avatar": {"type": "custom", "avatarId": TASK_ID}, "speech": {"type": "text", "text": "hi", "voice": {"type": "preset", "presetId": "hal9000"}}},
            ),
            "voice preset",
        ),
        (_spec("gwm1_avatars", {"avatar": {"type": "custom", "avatarId": TASK_ID}, "speech": {"type": "audio"}}), "narration audio"),
        (_spec("magnific_precision_upscaler_v2", {"scale_factor": 3}, {"image": {"path": "a.png"}}), "2, 4, 8 or 16"),
    ],
)
def test_validate_spec_enforces_documented_limits(spec, message):
    with pytest.raises(RunwayValidationError, match=message):
        runway.validate_spec(spec)


def test_validate_spec_checks_assets_exist_in_job_map():
    spec = _spec(
        "gen4_image",
        {"prompt": "@presenter smiles", "ratio": "1280:720"},
        {"references": [{"asset": "IDENTITY_REF_1", "tag": "presenter", "role": "primary identity"}]},
    )
    with pytest.raises(RunwayValidationError, match="has no file"):
        runway.validate_spec(spec, {"IDENTITY_REF_1": None})
    runway.validate_spec(spec, {"IDENTITY_REF_1": "/tmp/front.png"})


def test_build_request_shapes_match_api_reference():
    image = runway.model_spec("gemini_image3.1_flash")
    endpoint, body = runway.build_request(
        image,
        {"prompt": "@presenter at a desk", "ratio": "832:1248", "output_count": 4, "seed": 9},
        {"references": [{"uri": "data:a", "tag": "presenter", "subject": "human"}, {"uri": "data:b", "subject": "human"}]},
    )
    assert endpoint == "/v1/text_to_image"
    assert body == {
        "model": "gemini_image3.1_flash",
        "promptText": "@presenter at a desk",
        "ratio": "832:1248",
        "referenceImages": [{"uri": "data:a", "tag": "presenter", "subject": "human"}, {"uri": "data:b", "subject": "human"}],
        "outputCount": 4,
    }

    gen4 = runway.model_spec("gen4_image")
    _endpoint, body = runway.build_request(
        gen4, {"prompt": "@presenter", "ratio": "1280:720", "seed": 7}, {"references": [{"uri": "u", "tag": "presenter", "subject": "human"}]}
    )
    assert body["referenceImages"] == [{"uri": "u", "tag": "presenter"}]  # gen4 has no subject field
    assert body["seed"] == 7 and "outputCount" not in body

    endpoint, body = runway.build_request(
        runway.model_spec("h3_max"),
        {"prompt": "turns to camera", "resolution": "768p", "duration": 8, "seed": 3, "prompt_expansion": "disabled"},
        {"first_frame": "f", "last_frame": "l"},
    )
    assert endpoint == "/v1/image_to_video"
    assert body["promptImage"] == [{"uri": "f", "position": "first"}, {"uri": "l", "position": "last"}]
    assert body["promptExpansionMode"] == "disabled" and body["resolution"] == "768p" and "ratio" not in body

    _endpoint, body = runway.build_request(
        runway.model_spec("gen4.5"), {"prompt": "nods", "ratio": "720:1280", "duration": 5}, {"first_frame": "f", "last_frame": "l"}
    )
    assert body["promptImage"] == [{"uri": "f", "position": "first"}]

    endpoint, body = runway.build_request(
        runway.model_spec("act_two"),
        {"ratio": "720:1280", "body_control": False, "expression_intensity": 4},
        {"character": "c", "character_type": "image", "reference_video": "v"},
    )
    assert endpoint == "/v1/character_performance"
    assert body["character"] == {"type": "image", "uri": "c"}
    assert body["reference"] == {"type": "video", "uri": "v"}
    assert body["bodyControl"] is False and body["expressionIntensity"] == 4

    endpoint, body = runway.build_request(
        runway.model_spec("gwm1_avatars"),
        {"avatar": {"type": "custom", "avatarId": TASK_ID}, "speech": {"type": "text", "text": "Hi", "voice": {"type": "preset", "presetId": "clara"}}},
        {},
    )
    assert endpoint == "/v1/avatar_videos"
    assert body == {
        "model": "gwm1_avatars",
        "avatar": {"type": "custom", "avatarId": TASK_ID},
        "speech": {"type": "text", "text": "Hi", "voice": {"type": "preset", "presetId": "clara"}},
    }
    _endpoint, body = runway.build_request(
        runway.model_spec("gwm1_avatars"),
        {"avatar": {"type": "runway-preset", "presetId": "cooking-teacher"}, "speech": {"type": "audio"}},
        {"speech_audio": "a"},
    )
    assert body["speech"] == {"type": "audio", "audio": "a"}

    endpoint, body = runway.build_request(
        runway.model_spec("magnific_precision_upscaler_v2"), {"scale_factor": 4, "flavor": "photo", "sharpen": 140}, {"image": "i"}
    )
    assert endpoint == "/v1/image_upscale"
    assert body == {"model": "magnific_precision_upscaler_v2", "imageUri": "i", "scaleFactor": 4, "flavor": "photo", "sharpen": 100}


def test_redact_request_removes_inline_media():
    body = {"promptImage": [{"uri": "data:image/png;base64," + "A" * 400}], "promptText": "keep me"}
    redacted = runway.redact_request(body)
    assert "base64" not in json.dumps(redacted)
    assert redacted["promptText"] == "keep me"


def test_failure_codes_map_to_documented_retry_guidance():
    assert runway.classify_failure("SAFETY.INPUT.IMAGE")["retryable"] is False
    assert runway.classify_failure("INPUT_PREPROCESSING.SAFETY.TEXT")["category"] == "moderation"
    assert runway.classify_failure("INPUT_PREPROCESSING.INTERNAL")["retryable"] is True
    assert runway.classify_failure("INTERNAL.BAD_OUTPUT.CODE01")["category"] == "bad_output"
    assert runway.classify_failure("ASSET.INVALID")["retryable"] is False
    assert runway.classify_failure("THIRD_PARTY.UNAVAILABLE")["retryable"] is True
    assert runway.classify_failure(None)["category"] == "internal"
    failed = RunwayTaskFailed(TASK_ID, "blocked", "SAFETY.OUTPUT.VIDEO")
    classified = classify_exception(failed)
    assert classified.category == "cloud_moderation"
    assert "moderation" in classified.message


def test_network_errors_are_not_misreported_as_comfyui():
    classified = classify_exception(RunwayAPIError("Task lookup: network error (ConnectError)"))
    assert classified.category == "cloud_unreachable"


# --------------------------------------------------------------------------- media preparation


def test_small_png_travels_inline_and_large_files_use_ephemeral_upload(tmp_path, monkeypatch):
    small = _png(tmp_path / "front.png")
    prepared = runway.prepare_media(str(small), kind="image", role="identity", uploader=None)
    assert prepared.method == "inline_data_uri"
    assert prepared.uri.startswith("data:image/png;base64,")
    assert base64.b64decode(prepared.uri.split(",", 1)[1]) == small.read_bytes()

    calls = []
    monkeypatch.setattr(runway, "DATA_URI_LIMIT", 64)
    prepared = runway.prepare_media(
        str(small), kind="image", role="identity", uploader=lambda name, ctype, data: calls.append((name, ctype, len(data))) or "runway://abc"
    )
    assert prepared.method == "ephemeral_upload" and prepared.uri == "runway://abc"
    assert calls == [("front.png", "image/png", small.stat().st_size)]


def test_exif_is_stripped_and_unsupported_formats_are_converted(tmp_path):
    jpeg = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # Make
    Image.new("RGB", (64, 64), (10, 20, 30)).save(jpeg, exif=exif)
    prepared = runway.prepare_media(str(jpeg), kind="image", role="identity", uploader=None)
    assert prepared.metadata_stripped is True
    data = base64.b64decode(prepared.uri.split(",", 1)[1])
    reread = Image.open(__import__("io").BytesIO(data))
    assert len(reread.getexif()) == 0

    bmp = tmp_path / "legacy.bmp"
    Image.new("RGB", (32, 32)).save(bmp)
    converted = runway.prepare_media(str(bmp), kind="image", role="identity", uploader=None)
    assert converted.content_type == "image/png" and converted.filename == "legacy.png"


@pytest.mark.parametrize(
    "value,kind,message",
    [
        ("http://insecure.example/a.png", "image", "HTTPS"),
        ("data:image/png;base64,AAAA", "image", "HTTPS"),
        ("/nope/missing.png", "image", "does not exist"),
    ],
)
def test_prepare_media_rejects_unsafe_or_missing_inputs(value, kind, message):
    with pytest.raises(RunwayValidationError, match=message):
        runway.prepare_media(value, kind=kind, role="identity", uploader=None)


def test_prepare_media_passthrough_and_type_checks(tmp_path):
    assert runway.prepare_media("https://cdn.example/a.png", kind="image", role="r", uploader=None).method == "https_url"
    assert runway.prepare_media("runway://upload-1", kind="image", role="r", uploader=None).method == "runway_uri"
    video = tmp_path / "clip.avi"
    video.write_bytes(b"0" * 600)
    with pytest.raises(RunwayValidationError, match="MP4"):
        runway.prepare_media(str(video), kind="video", role="performance", uploader=None)
    small = _png(tmp_path / "tiny.png", size=(200, 200))
    with pytest.raises(RunwayValidationError, match="at least 256"):
        runway.prepare_media(str(small), kind="image", role="first frame", uploader=None, min_side=256)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image at all")
    with pytest.raises(RunwayValidationError, match="could not be read"):
        runway.prepare_media(str(broken), kind="image", role="identity", uploader=None)


# --------------------------------------------------------------------------- HTTP client


def test_client_refuses_non_official_hosts_and_missing_keys():
    with pytest.raises(RunwayConfigError, match="official API"):
        RunwayClient(API_KEY, base_url="https://evil.example")
    with pytest.raises(RunwayConfigError):
        RunwayClient("")


def test_api_calls_carry_auth_and_version_but_upload_and_download_hosts_never_see_the_key(tmp_path):
    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/uploads"): httpx.Response(
                200, json={"uploadUrl": "https://storage.example/bucket", "fields": {"policy": "p1"}, "runwayUri": "runway://up-1"}
            ),
            ("POST", "storage.example", "/bucket"): httpx.Response(204),
            ("GET", "cdn.example", "/out/clip"): httpx.Response(200, content=b"mp4", headers={"content-type": "video/mp4"}),
        }
    )
    with _client(recorder) as client:
        uri = client.upload("front.png", "image/png", b"x" * 600)
        target = client.download("https://cdn.example/out/clip", tmp_path / "dl", "runway_out")
    assert uri == "runway://up-1"
    start, storage, download = recorder.requests
    assert start.headers["Authorization"] == f"Bearer {API_KEY}"
    assert start.headers["X-Runway-Version"] == "2024-11-06"
    assert json.loads(start.content) == {"filename": "front.png", "type": "ephemeral"}
    assert "authorization" not in {key.lower() for key in storage.headers}
    assert b'name="policy"' in storage.content and b'name="file"; filename="front.png"' in storage.content
    assert "authorization" not in {key.lower() for key in download.headers}
    assert target.name == "runway_out.mp4" and target.read_bytes() == b"mp4"


def test_create_task_retries_rate_limits_but_surfaces_validation_detail():
    sleeps = []
    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): [
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json={"id": TASK_ID, "estimatedCost": {"credits": 8}}),
            ]
        }
    )
    with _client(recorder, sleep=sleeps.append) as client:
        assert client.create_task("/v1/text_to_image", {"model": "gen4_image"})["id"] == TASK_ID
    assert sleeps == [2.0]

    bad = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): httpx.Response(
                400, json={"error": "Validation failed", "issues": [{"path": ["ratio"], "message": "Invalid enum value"}]}
            )
        }
    )
    with _client(bad) as client, pytest.raises(RunwayAPIError, match="ratio: Invalid enum value") as caught:
        client.create_task("/v1/text_to_image", {"model": "gen4_image"})
    assert caught.value.status_code == 400
    assert API_KEY not in str(caught.value)


def test_create_task_does_not_retry_server_errors_to_avoid_duplicate_billing():
    recorder = Recorder({("POST", "api.dev.runwayml.com", "/v1/image_to_video"): httpx.Response(502, text="bad gateway")})
    with _client(recorder) as client, pytest.raises(RunwayAPIError):
        client.create_task("/v1/image_to_video", {"model": "gen4.5"})
    assert len(recorder.requests) == 1


def test_wait_for_task_polls_at_least_five_seconds_and_backs_off_on_errors():
    sleeps = []
    recorder = Recorder(
        {
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): [
                httpx.Response(200, json={"id": TASK_ID, "status": "PENDING"}),
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={"id": TASK_ID, "status": "THROTTLED"}),
                httpx.Response(200, json={"id": TASK_ID, "status": "RUNNING", "progress": 0.5}),
                httpx.Response(200, json={"id": TASK_ID, "status": "SUCCEEDED", "output": ["https://cdn.example/o.png"]}),
            ]
        }
    )
    updates = []
    with _client(recorder, sleep=sleeps.append) as client:
        task = client.wait_for_task(TASK_ID, timeout_s=600, poll_interval_s=1, on_update=updates.append)
    assert task["status"] == "SUCCEEDED"
    assert all(delay >= 5.0 for delay in sleeps)
    assert sleeps[1] == 10.0  # exponential backoff after the 503
    assert [u["status"] for u in updates] == ["PENDING", "THROTTLED", "RUNNING", "SUCCEEDED"]


def test_wait_for_task_times_out_and_cancels_the_remote_task():
    clock = [0.0]
    recorder = Recorder(
        {
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(200, json={"id": TASK_ID, "status": "RUNNING"}),
            ("DELETE", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(204),
        }
    )
    with _client(recorder, sleep=lambda s: clock.__setitem__(0, clock[0] + s), monotonic=lambda: clock[0]) as client:
        with pytest.raises(RunwayTimeout):
            client.wait_for_task(TASK_ID, timeout_s=30, poll_interval_s=6)
    assert recorder.find("DELETE", f"/v1/tasks/{TASK_ID}")


# --------------------------------------------------------------------------- provider end-to-end


def _avatar(tmp_path, kind="synthetic") -> AvatarManifest:
    return AvatarManifest.model_validate(
        {
            "avatar_id": "brand-presenter",
            "display_name": "Course Presenter",
            "sex": "female",
            "apparent_age_years": 25,
            "subjects": [{"id": "brand-presenter", "kind": kind, "age_verified_18_plus": True, "consent_confirmed": True}],
            "identity_refs": [str(_png(tmp_path / "front.png"))],
        }
    )


def _still_job(tmp_path, *, max_credits=150):
    avatar = _avatar(tmp_path)
    shot = ShotSpec.model_validate({"shot_id": "still", "user_prompt": "smiles at the camera", "engine_preference": "runway"})
    job = build_job(avatar, shot)
    spec = {
        "version": 1,
        "model": "gen4_image",
        "task": "still",
        "params": {"prompt": "@presenter smiles at the camera", "ratio": "1080:1440", "seed": 41001},
        "media": {"references": [{"asset": "IDENTITY_REF_1", "tag": "presenter", "role": "primary identity"}]},
        "estimate": {"credits": 8},
        "max_credits": max_credits,
    }
    job.asset_map["RUNWAY_SPEC"] = json.dumps(spec)
    return job


def _provider(tmp_path, recorder, **kwargs):
    config = ProviderConfig(provider="runway", base_url=runway.RUNWAY_API_BASE, output_dir=str(tmp_path / "runway"), timeout_s=600)
    kwargs.setdefault("sleep", lambda _s: None)
    return RunwayProvider(config, transport=httpx.MockTransport(recorder), **kwargs)


def test_provider_requires_a_configured_key(tmp_path):
    with pytest.raises(RunwayConfigError, match="No Runway API key"):
        _provider(tmp_path, Recorder({})).render("runway:still:gen4_image", _still_job(tmp_path))


def test_provider_uploads_creates_polls_downloads_and_keeps_a_redacted_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", API_KEY)
    created_bodies = []

    def create(request):
        created_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": TASK_ID, "estimatedCost": {"credits": 8}})

    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): create,
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): [
                httpx.Response(200, json={"id": TASK_ID, "status": "RUNNING"}),
                httpx.Response(
                    200,
                    json={"id": TASK_ID, "status": "SUCCEEDED", "cost": {"credits": 8}, "output": ["https://cdn.example/a/out.png"]},
                ),
            ],
            ("GET", "cdn.example", "/a/out.png"): httpx.Response(200, content=b"png-bytes", headers={"content-type": "image/png"}),
        }
    )
    result = _provider(tmp_path, recorder).render("runway:still:gen4_image", _still_job(tmp_path))

    body = created_bodies[0]
    assert body["model"] == "gen4_image" and body["ratio"] == "1080:1440" and body["seed"] == 41001
    assert body["referenceImages"][0]["tag"] == "presenter"
    assert body["referenceImages"][0]["uri"].startswith("data:image/png;base64,")
    assert result["prompt_id"] == TASK_ID
    output = Path(result["outputs"][0])
    assert output.read_bytes() == b"png-bytes" and output.parent == (tmp_path / "runway" / TASK_ID).resolve()
    receipt = result["cloud_receipt"]
    assert receipt["credits_charged"] == 8 and receipt["status"] == "SUCCEEDED"
    assert receipt["uploads"][0]["file"] == "front.png" and receipt["uploads"][0]["transfer"] == "inline_data_uri"
    assert "base64" not in json.dumps(receipt)
    assert "output" not in result["history"]


def test_provider_cancels_tasks_whose_server_estimate_exceeds_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", API_KEY)
    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): httpx.Response(
                200, json={"id": TASK_ID, "estimatedCost": {"credits": 500}}
            ),
            ("DELETE", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(204),
        }
    )
    with pytest.raises(RunwayBudgetError) as caught:
        _provider(tmp_path, recorder).render("runway:still:gen4_image", _still_job(tmp_path, max_credits=100))
    assert recorder.find("DELETE", f"/v1/tasks/{TASK_ID}")
    assert caught.value.receipt["task_id"] == TASK_ID
    assert caught.value.receipt["api_estimated_credits"] == 500


def test_provider_reports_task_failure_with_receipt_and_advice(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", API_KEY)
    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): httpx.Response(200, json={"id": TASK_ID}),
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(
                200, json={"id": TASK_ID, "status": "FAILED", "failure": "Bad output", "failureCode": "INTERNAL.BAD_OUTPUT.CODE01"}
            ),
        }
    )
    with pytest.raises(RunwayTaskFailed) as caught:
        _provider(tmp_path, recorder).render("runway:still:gen4_image", _still_job(tmp_path))
    assert caught.value.receipt["failure_code"] == "INTERNAL.BAD_OUTPUT.CODE01"
    assert classify_exception(caught.value).category == "cloud_bad_output"


def test_user_cancel_during_polling_cancels_the_remote_task(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", API_KEY)
    recorder = Recorder(
        {
            ("POST", "api.dev.runwayml.com", "/v1/text_to_image"): httpx.Response(200, json={"id": TASK_ID}),
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(200, json={"id": TASK_ID, "status": "PENDING"}),
            ("DELETE", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(204),
        }
    )
    holder = {}
    provider = _provider(tmp_path, recorder, sleep=lambda _s: holder["provider"]._cancel.set())
    holder["provider"] = provider
    with pytest.raises(RunwayTaskCancelled):
        provider.render("runway:still:gen4_image", _still_job(tmp_path))
    assert recorder.find("DELETE", f"/v1/tasks/{TASK_ID}")


def test_interrupt_cancels_the_active_remote_task(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNWAYML_API_SECRET", API_KEY)
    recorder = Recorder({("DELETE", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(204)})
    provider = _provider(tmp_path, recorder)
    provider._task_id = TASK_ID
    provider.interrupt()
    assert provider._cancel.is_set()
    assert recorder.find("DELETE", f"/v1/tasks/{TASK_ID}")
    assert RunwayProvider.uses_local_gpu is False


# --------------------------------------------------------------------------- credential store


def test_secret_store_prefers_environment_then_user_only_file(tmp_path, monkeypatch):
    assert secret_store.secret_status("runway")["configured"] is False
    assert secret_store.save_secret("runway", API_KEY) == "file"
    path = secret_store.SECRETS_DIR / "runway_api_key"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(secret_store.SECRETS_DIR.stat().st_mode) == 0o700
    assert secret_store.get_secret("runway") == (API_KEY, "file")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "env_key_abcdefghijklmnop")
    assert secret_store.get_secret("runway") == ("env_key_abcdefghijklmnop", "environment")
    status = secret_store.secret_status("runway")
    assert status == {"configured": True, "source": "environment", "env_var": "RUNWAYML_API_SECRET", "keyring_available": False}
    assert API_KEY not in json.dumps(status)
    monkeypatch.delenv("RUNWAYML_API_SECRET")
    assert secret_store.clear_secret("runway") is True
    assert secret_store.get_secret("runway") == (None, None)


@pytest.mark.parametrize("bad", ["", "short", "has space in the middle of it", "x" * 600])
def test_secret_store_rejects_implausible_keys_without_echoing_them(bad):
    with pytest.raises(secret_store.SecretStoreError) as caught:
        secret_store.save_secret("runway", bad)
    if bad:
        assert bad not in str(caught.value)


def test_secret_store_uses_keyring_when_available(monkeypatch):
    class FakeKeyring:
        store: dict = {}

        def get_password(self, service, name):
            return self.store.get((service, name))

        def set_password(self, service, name, value):
            self.store[(service, name)] = value

        def delete_password(self, service, name):
            self.store.pop((service, name))

    fake = FakeKeyring()
    monkeypatch.setattr(secret_store, "_keyring", lambda: fake)
    assert secret_store.save_secret("runway", API_KEY) == "keyring"
    assert not (secret_store.SECRETS_DIR / "runway_api_key").exists()
    assert secret_store.get_secret("runway") == (API_KEY, "keyring")
    assert secret_store.clear_secret("runway") is True
    assert fake.store == {}


def test_client_from_store_names_the_setup_step_when_no_key_exists():
    with pytest.raises(RunwayConfigError, match="RUNWAYML_API_SECRET"):
        runway.client_from_store()


def test_threading_event_cancel_path_without_injected_sleep():
    client_event = threading.Event()
    client_event.set()
    recorder = Recorder(
        {
            ("GET", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(200, json={"id": TASK_ID, "status": "PENDING"}),
            ("DELETE", "api.dev.runwayml.com", f"/v1/tasks/{TASK_ID}"): httpx.Response(204),
        }
    )
    client = RunwayClient(API_KEY, transport=httpx.MockTransport(recorder), jitter=lambda: 0.0)
    with client, pytest.raises(RunwayTaskCancelled):
        client.wait_for_task(TASK_ID, timeout_s=60, cancel_event=client_event)
