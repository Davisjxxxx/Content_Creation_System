"""Studio: the simple start screen maps plain choices onto normal Runway jobs (no network here)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from avatar_v2 import brand_instructor as bi
from avatar_v2 import desktop, desktop_studio, secret_store
from avatar_v2.desktop import DesktopAPI
from avatar_v2.integrity import sha256_file
from avatar_v2.js_bridge import JS_API_METHODS
from avatar_v2.library import AvatarProfile, LibraryStore
from avatar_v2.providers import runway as runway_api

KEY = "key_" + "s" * 40
TALKER_UUID = "11111111-2222-3333-4444-555555555555"


class _Memory:
    def update_settings(self, **_kwargs):
        pass


class FakeQueue:
    def __init__(self):
        self.items = []
        self.memory_manager = _Memory()

    def add_many(self, items):
        self.items.extend(items)
        return [item.id for item in items]

    def list(self):
        return [item.public() for item in self.items]


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name, value in {
        "APP_HOME": tmp_path,
        "CONFIG_PATH": tmp_path / "desktop.json",
        "RUNS_DIR": tmp_path / "runs",
        "HISTORY_DIR": tmp_path / "history",
        "RUNWAY_AVATARS_PATH": tmp_path / "runway_avatars.json",
        "CLOUD_LEDGER_PATH": tmp_path / "cloud_ledger.jsonl",
        "VOICEOVER_DIR": tmp_path / "voiceover",
        "LESSONS_PATH": tmp_path / "lessons.json",
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


def _png(path: Path, size=(1080, 1440), color=(150, 120, 110)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _video(path: Path, seconds: float = 6.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=480x854:rate=24", "-f", "lavfi",
         "-i", "sine=f=330", "-t", str(seconds), "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _pick_master(api, home) -> Path:
    master = _png(home / "master.png")
    assert api.studio_pick({"path": str(master), "slot_id": "identity_front"})["ok"] is True
    return master


def _spec(item) -> dict:
    return json.loads(item.job.asset_map["RUNWAY_SPEC"])


def test_studio_methods_are_bridged_and_the_internal_setup_is_not():
    for name in ("studio_state", "studio_price", "studio_go", "studio_pick", "studio_talker_status", "studio_thumb", "studio_image"):
        assert name in JS_API_METHODS
    assert "studio_setup_talker" not in JS_API_METHODS  # only reachable through studio_go


def test_state_installs_the_profile_and_offers_looks_angles_and_voices(api):
    state = api.studio_state()
    assert state["ok"] is True and state["master"] is None and state["key"]["configured"] is False
    assert [look["id"] for look in state["looks"]] == ["natural", "polished", "original"] and state["look"] == "natural"
    assert len(state["angles"]) == 8 and "identity_front" not in {a["id"] for a in state["angles"]}
    assert state["voice"] == "Maya" and "Leslie" in state["voices"]
    assert api.library.get_avatar(bi.PROFILE_ID) is not None  # created on first use, locally


def test_prices_match_runway_pricing_and_steps_explain_what_they_need(api, home):
    assert api.studio_price({"action": "presenter", "count": 4})["credits"] == 32
    assert api.studio_price({"action": "presenter", "count": 1})["credits"] == 8
    assert api.studio_price({"action": "sample"})["credits"] == 1
    for action in ("angles", "talk", "act", "move", "sharpen"):
        result = api.studio_price({"action": action, "slots": ["identity_three_quarter_left"], "script": "Hi"})
        assert result["ok"] is False and "Pick your presenter photo first" in result["error"]
    _pick_master(api, home)
    assert api.studio_price({"action": "angles", "slots": ["identity_three_quarter_left", "presenter_expression_smile"]})["credits"] == 16
    assert api.studio_price({"action": "move", "seconds": 10, "quality": "best"})["credits"] == 120
    assert api.studio_price({"action": "move", "seconds": 5, "quality": "fast"})["credits"] == 25
    assert api.studio_price({"action": "sharpen"})["credits"] == 25
    talk = api.studio_price({"action": "talk", "script": "Welcome back! Today we look at three quick ideas."})
    assert talk["ok"] is True and talk["needs_talker"] is True
    act = api.studio_price({"action": "act", "video": str(_video(home / "me.mp4", 6))})
    assert act["ok"] is True and act["credits"] == 30  # 5 credits per second of performance
    assert "Tick at least one" in api.studio_price({"action": "angles", "slots": []})["error"]
    assert "Type what she should say" in api.studio_price({"action": "voiceover", "script": "  "})["error"]


def test_go_without_a_key_explains_how_to_connect(api):
    result = api.studio_go({"action": "presenter", "count": 4})
    assert result["ok"] is False and "API key" in result["error"] and api.queue.items == []


def test_presenter_options_queue_in_one_click_with_fresh_seeds_and_the_chosen_look(api):
    api.runway_save_key(KEY)
    first = api.studio_go({"action": "presenter", "count": 4, "look": "natural", "extra": "short dark bob"})
    assert first["ok"] is True and first["total_credits"] == 32 and len(api.queue.items) == 4
    specs = [_spec(item) for item in api.queue.items]
    seeds = [spec["params"]["seed"] for spec in specs]
    assert seeds == list(range(seeds[0], seeds[0] + 4))
    prompt = specs[0]["params"]["prompt"]
    assert "no eyeliner" in prompt and "short dark bob." in prompt and len(prompt) <= 1000
    assert specs[0]["media"]["references"] == []  # a brand-new face from the description
    assert all(item.runtime["candidate_slot"] == "identity_front" for item in api.queue.items)
    assert all(item.runtime["studio_action"] == "presenter" for item in api.queue.items)
    assert api.queue.items[0].public()["studio_action"] == "presenter"
    second = api.studio_go({"action": "presenter", "count": 1, "look": "polished"})
    assert second["ok"] is True and _spec(api.queue.items[-1])["params"]["seed"] not in seeds
    saved = api.studio_state()
    assert saved["look"] == "polished" and saved["extra"] == ""


def test_picking_an_option_makes_it_the_master_and_angles_use_it(api, home):
    api.runway_save_key(KEY)
    master = _pick_master(api, home)
    profile = AvatarProfile.model_validate(api.library.get_avatar(bi.PROFILE_ID))
    assert profile.identity_refs["front"] == str(master.resolve())
    assert profile.reference_hashes["front"] == sha256_file(master)
    assert Path(api.studio_state()["master"]).name == "master.png"
    result = api.studio_go({"action": "angles", "slots": ["identity_three_quarter_left", "presenter_upper_body"], "extra": "x" * 160})
    assert result["ok"] is True and len(api.queue.items) == 2
    for item, slot in zip(api.queue.items, ("identity_three_quarter_left", "presenter_upper_body")):
        spec = _spec(item)
        assert spec["media"]["references"][0]["asset"] == "IDENTITY_REF_1"
        assert bi.slot_by_id(slot)["prompt"][:40] in spec["params"]["prompt"]  # the angle direction survives
        assert "x" * 40 not in spec["params"]["prompt"]


def test_tossing_an_option_rejects_it_without_touching_the_master(api, home):
    master = _pick_master(api, home)
    other = _png(home / "other.png", color=(20, 30, 40))
    api.library.record_candidate(bi.PROFILE_ID, {"path": str(other.resolve()), "source_path": "t2i", "slot_id": "identity_front",
                                                 "group": "identity", "status": "pending", "qa_decision": "review"})
    assert [c["path"] for c in api.studio_state()["pending_master"]] == [str(other.resolve())]
    assert api.studio_pick({"path": str(other), "slot_id": "identity_front", "keep": False})["ok"] is True
    state = api.studio_state()
    assert state["pending_master"] == [] and Path(state["master"]) == master.resolve()


def test_voiceover_remembers_her_voice_and_registers_the_audio(api):
    api.runway_save_key(KEY)
    result = api.studio_go({"action": "voiceover", "voice": "Leslie", "script": "Welcome to the course. Today: fractions."})
    assert result["ok"] is True
    spec = _spec(api.queue.items[-1])
    assert spec["params"]["voice"] == {"type": "runway-preset", "presetId": "Leslie"}
    assert api.queue.items[-1].runtime["register_voiceover"] is True
    assert api.studio_state()["voice"] == "Leslie"
    sample = api.studio_go({"action": "sample", "voice": "Maya"})
    assert sample["ok"] is True and sample["total_credits"] == 1
    assert _spec(api.queue.items[-1])["params"]["prompt"] == desktop_studio.SAMPLE_LINE


def test_talk_sets_up_the_talking_avatar_first_then_queues_the_video(api, home, monkeypatch):
    api.runway_save_key(KEY)
    master = _pick_master(api, home)
    created = []

    def fake_create(payload):
        created.append(payload)
        index = api._runway_avatar_index()
        index[TALKER_UUID] = {"vault_avatar_id": bi.PROFILE_ID, "name": payload["name"], "voice": payload["voice"],
                              "source_sha256": sha256_file(payload["reference_image"]), "created_at": "2026-11-22T00:00:00+00:00"}
        desktop._write_json(desktop.RUNWAY_AVATARS_PATH, index)
        return {"ok": True, "avatar": {"id": TALKER_UUID, "name": payload["name"], "status": "PROCESSING"}}

    monkeypatch.setattr(api, "runway_avatar_create", fake_create)
    first = api.studio_go({"action": "talk", "script": "Hi! Today I'll show you three quick ideas.", "talker_voice": "emma"})
    assert first["ok"] is False and first["setup_started"] is True and api.queue.items == []
    assert created[0]["reference_image"] == str(master.resolve()) and created[0]["voice"] == "emma"
    assert api.studio_state()["talker"] == {"id": TALKER_UUID, "voice": "emma"}
    second = api.studio_go({"action": "talk", "script": "Hi! Today I'll show you three quick ideas."})
    assert second["ok"] is True and len(api.queue.items) == 1 and len(created) == 1  # no second avatar
    spec = _spec(api.queue.items[0])
    assert spec["params"]["avatar"] == {"type": "custom", "avatarId": TALKER_UUID}
    assert spec["params"]["speech"]["type"] == "text"


def test_act_and_move_use_the_master_and_any_voiceover(api, home):
    api.runway_save_key(KEY)
    master = _pick_master(api, home)
    clip = home / "narration.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", str(clip)], check=True)
    narration = api.voiceover_import({"paths": [str(clip)]})["imported"][0]["id"]  # usable straight away
    acted = api.studio_go({"action": "act", "video": str(_video(home / "me.mp4", 5)), "body": False, "narration_id": narration})
    assert acted["ok"] is True
    act_item = api.queue.items[-1]
    spec = _spec(act_item)
    assert spec["params"]["output_audio"] == "voiceover" and spec["params"]["body_control"] is False
    assert spec["params"]["ratio"] == "720:1280" and act_item.job.asset_map["INIT_IMAGE"] == str(master.resolve())
    moved = api.studio_go({"action": "move", "motion": "She waves hello.", "seconds": 5, "quality": "fast", "shape": "portrait"})
    assert moved["ok"] is True
    move_spec = _spec(api.queue.items[-1])
    assert move_spec["model"] == "gen4_turbo" and move_spec["params"]["ratio"] == "832:1104"
    assert "She waves hello." in move_spec["params"]["prompt"]


def test_owner_limits_still_apply_through_studio(api):
    api.runway_save_key(KEY)
    api.save_settings({"runway_job_limit": 5})
    assert "limit you set" in api.studio_price({"action": "presenter", "count": 1})["limit_error"]
    refused = api.studio_go({"action": "presenter", "count": 1})
    assert refused["ok"] is False and "limit you set" in refused["error"] and api.queue.items == []


def test_full_size_viewer_image_and_video_thumbnails(api, home):
    still = _png(home / "big.png", size=(1080, 1440))
    image = api.studio_image(str(still))
    assert image["ok"] is True and (image["width"], image["height"]) == (1080, 1440)
    assert image["data_url"].startswith("data:image/jpeg;base64,")
    assert api.studio_image(str(home / "missing.png"))["ok"] is False
    assert api.studio_image(str(_video(home / "clip.mp4", 1)))["ok"] is False  # stills only
    thumb = api.studio_thumb(str(home / "clip.mp4"))
    assert thumb["ok"] is True and thumb["data_url"].startswith("data:image/jpeg;base64,")


def test_results_list_recent_runway_outputs_and_recent_failures_only(api, home):
    out = _png(home / "result.png", size=(400, 500))
    now = __import__("time").time()
    records = {
        "done": {"id": "done", "provider": "runway", "status": "done", "label": "Photo of her · Natural smile", "outputs": [str(out)],
                 "finished_at": now, "studio_action": "angles", "cloud_receipt": {"credits_charged": 8, "charge_status": "actual"}},
        "fresh_fail": {"id": "fresh_fail", "provider": "runway", "status": "error", "label": "Talking video", "outputs": [],
                       "finished_at": now, "error_message": "Runway said no.", "cloud_receipt": {"charge_status": "none"}},
        "old_fail": {"id": "old_fail", "provider": "runway", "status": "error", "label": "Old", "outputs": [], "finished_at": now - 86400},
        "local": {"id": "local", "provider": "comfyui", "status": "done", "label": "Local", "outputs": [str(out)], "finished_at": now},
    }
    for key, record in records.items():
        (home / "history" / f"{key}.json").write_text(json.dumps(record))
    results = {r["id"]: r for r in api.studio_state()["results"]}
    assert set(results) == {"done", "fresh_fail"}
    assert results["done"]["outputs"][0]["kind"] == "image" and results["done"]["credits"] == 8
    assert results["fresh_fail"]["error"] == "Runway said no." and results["fresh_fail"]["charge_status"] == "none"
