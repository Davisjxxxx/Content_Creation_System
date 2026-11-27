"""Studio: the simple start screen maps plain choices onto normal Runway jobs (no network here)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
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
        "MANAGED_ASSETS_PATH": tmp_path / "managed_assets.json",
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
    for name in ("studio_state", "studio_price", "studio_go", "studio_pick", "studio_talker_status", "studio_thumb",
                 "studio_image", "studio_video", "studio_library", "studio_avatars", "studio_delete", "engine_status",
                 "comfyui_start", "comfyui_stop"):
        assert name in JS_API_METHODS
    for name in ("studio_setup_talker", "engine_autostart", "queue_runway", "queue_render", "vault_save"):
        assert name not in JS_API_METHODS  # internal, or part of the removed advanced screens


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
    clip = _video(home / "clip.mp4", 1)
    assert api.studio_image(str(clip))["ok"] is False  # stills only
    thumb = api.studio_thumb(str(clip))
    assert thumb["ok"] is True and thumb["data_url"].startswith("data:image/jpeg;base64,")


def test_in_app_video_viewer_serves_local_video_with_range_seeking(api, home):
    import httpx

    clip = _video(home / "review.mp4", 1)
    still = _png(home / "still.png")
    try:
        assert api.studio_video(str(home / "missing.mp4"))["ok"] is False
        assert api.studio_video(str(still))["ok"] is False  # videos only

        info = api.studio_video(str(clip))
        assert info["ok"] is True
        assert info["url"].startswith("http://127.0.0.1:") and info["url"].endswith(".mp4")
        assert info["name"] == "review.mp4"
        assert info["width"] and info["height"] and info["duration_s"]

        raw = clip.read_bytes()
        full = httpx.get(info["url"], timeout=5.0)
        assert full.status_code == 200
        assert full.headers.get("accept-ranges") == "bytes"
        assert full.headers.get("content-type") == "video/mp4"
        assert full.content == raw

        partial = httpx.get(info["url"], headers={"Range": "bytes=0-127"}, timeout=5.0)
        assert partial.status_code == 206
        assert partial.headers.get("content-range") == f"bytes 0-127/{len(raw)}"
        assert partial.content == raw[:128]

        bad_range = httpx.get(info["url"], headers={"Range": f"bytes={len(raw) + 10}-"}, timeout=5.0)
        assert bad_range.status_code == 416

        base_url = info["url"].rsplit("/", 1)[0]
        assert httpx.get(f"{base_url}/not_registered.mp4", timeout=5.0).status_code == 404
    finally:
        api._close_media_server()


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
    assert set(results) == {"done", "fresh_fail", "local"}  # old local renders are shown again
    assert results["local"]["source"] == "old" and results["done"]["source"] == "runway"
    assert results["done"]["outputs"][0]["kind"] == "image" and results["done"]["credits"] == 8
    assert results["fresh_fail"]["error"] == "Runway said no." and results["fresh_fail"]["charge_status"] == "none"


def test_presenter_stays_an_adult_and_nothing_is_sent_for_minor_prompts(api, home):
    api.runway_save_key(KEY)
    refused = api.studio_price({"action": "presenter", "count": 4, "extra": "make her a 15-year-old girl"})
    assert refused["ok"] is False and "has to be an adult" in refused["error"] and "15-year-old" in refused["error"]
    go = api.studio_go({"action": "presenter", "count": 4, "extra": "Caucasion 15-year-old girl"})
    assert go["ok"] is False and api.queue.items == []
    assert api.studio_go({"action": "presenter", "count": 1, "extra": "looks about 18-year-old, freckles"})["ok"] is True
    _pick_master(api, home)
    assert "has to be an adult" in api.studio_price({"action": "move", "motion": "she turns into a schoolgirl"})["error"]


# --------------------------------------------------------------------------- local GPU engine (ComfyUI + Wan 2.2 14B)


def _fake_engine(tmp_path, monkeypatch, *, models=True):
    from avatar_v2 import local_engine

    comfy = tmp_path / "engine" / "comfyui"
    (comfy / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True)
    (comfy / "main.py").write_text("# fake")
    python = tmp_path / "engine" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    if models:
        for subfolder, name in local_engine.MODELS.values():
            (comfy / "models" / subfolder).mkdir(parents=True, exist_ok=True)
            (comfy / "models" / subfolder / name).write_bytes(b"x")
    monkeypatch.setattr(desktop, "ENGINE_HOME", tmp_path / "engine")
    return comfy


def test_engine_status_reports_not_installed_missing_and_ready(api, home, monkeypatch):
    monkeypatch.setattr(desktop, "ENGINE_HOME", home / "nothing-here")
    assert api.engine_status()["state"] == "not_installed"
    comfy = _fake_engine(home, monkeypatch, models=False)
    status = api.engine_status()
    assert status["state"] == "missing_models" and len(status["missing_models"]) == 6
    assert api._settings()["comfyui_dir"] == str(comfy)  # found without touching settings
    _fake_engine(home / "full", monkeypatch)
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "READY", "ready": True, "pid": 123})
    ready = api.engine_status()
    assert ready["state"] == "ready" and ready["installed"] is True and ready["seconds"] == 5
    started = []
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "STOPPED", "ready": False, "pid": None})
    monkeypatch.setattr(api, "comfyui_start", lambda: started.append(True) or {"ok": True, "state": "STARTING"})
    assert api.engine_autostart()["state"] == "STARTING" and started == [True]


def test_proc_alive_never_counts_zombies_or_missing_pids():
    import os

    from avatar_v2.desktop import _proc_alive

    assert _proc_alive(os.getpid()) is True
    assert _proc_alive(None) is False
    assert _proc_alive(99999999) is False


def test_local_moving_clip_is_free_and_fills_every_workflow_slot(api, home, monkeypatch):
    from avatar_v2 import local_engine
    from avatar_v2.providers.comfyui import replace_placeholders

    _fake_engine(home, monkeypatch)
    master = _pick_master(api, home)
    price = api.studio_price({"action": "move", "engine": "local", "motion": "She waves hello."})
    assert price["ok"] is True and price["credits"] == 0 and price["local"] is True
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "STARTING", "ready": False, "pid": 1})
    waiting = api.studio_go({"action": "move", "engine": "local", "motion": "She waves hello."})
    assert waiting["ok"] is False and "starting" in waiting["error"] and api.queue.items == []
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "READY", "ready": True, "pid": 1})
    go = api.studio_go({"action": "move", "engine": "local", "motion": "She waves hello.", "shape": "portrait"})
    assert go["ok"] is True and go["total_credits"] == 0 and go["local"] is True
    item = api.queue.items[0]
    assert item.provider_config.provider == "comfyui" and item.profile == "low_memory"  # forced GPU cleanup first
    assert item.workflow == str(local_engine.WORKFLOW) and item.runtime["studio_action"] == "move"
    workflow = json.loads(local_engine.WORKFLOW.read_text())
    resolved = replace_placeholders(workflow, item.job.asset_map)
    assert "${" not in json.dumps(resolved)  # every placeholder is filled
    assert resolved["11"]["inputs"]["image"] == str(master.resolve())
    assert (resolved["12"]["inputs"]["width"], resolved["12"]["inputs"]["height"]) == (480, 640)
    assert resolved["12"]["inputs"]["length"] == 81 and "She waves hello." in resolved["8"]["inputs"]["text"]
    high, low = resolved["13"]["inputs"], resolved["14"]["inputs"]
    assert (high["steps"], high["end_at_step"], low["start_at_step"], high["cfg"]) == (4, 2, 2, 1.0)  # Lightning settings
    assert resolved["5"]["inputs"]["shift"] == 5.0 and resolved["1"]["class_type"] == "UnetLoaderGGUF"
    refused = api.studio_price({"action": "move", "engine": "local", "motion": "she becomes a 15-year-old girl"})
    assert refused["ok"] is False and "adult" in refused["error"]


def test_local_clip_gets_her_voiceover_as_sound_and_keeps_the_silent_original(api, home, monkeypatch):
    _fake_engine(home, monkeypatch)
    _pick_master(api, home)
    clip = home / "vo.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=2", str(clip)], check=True)
    narration = api.voiceover_import({"paths": [str(clip)]})["imported"][0]["id"]
    item = api._studio_local_move({"engine": "local", "narration_id": narration})
    assert item.runtime["candidate_qa"] is True and Path(item.runtime["studio_voiceover"]).is_file()
    video = _video(home / "render.mp4", 2)
    item.outputs = [str(video)]
    item.output_hashes = {str(video): "x"}
    result = api._candidate_qa(item)
    assert result["ok"] is True and Path(item.outputs[0]).name == "render_voiceover.mp4"
    assert Path(item.outputs[1]) == video and item.output_hashes[item.outputs[0]]


def test_long_talking_scripts_become_parts(api, home, monkeypatch):
    api.runway_save_key(KEY)
    _pick_master(api, home)
    index = {TALKER_UUID: {"vault_avatar_id": bi.PROFILE_ID, "voice": "clara", "source_sha256": sha256_file(home / "master.png"),
                           "created_at": "2026-11-22T00:00:00+00:00"}}
    desktop._write_json(desktop.RUNWAY_AVATARS_PATH, index)
    paragraph = ("This is one paragraph of a long lesson that keeps going for quite a while. " * 12).strip()
    script = "\n\n".join([paragraph] * 8)  # about 7,000 characters
    price = api.studio_price({"action": "talk", "script": script})
    assert price["ok"] is True and price["parts"] == 2
    go = api.studio_go({"action": "talk", "script": script})
    assert go["ok"] is True and go["parts"] == 2 and len(api.queue.items) == 2
    assert [item.label for item in api.queue.items] == ["Talking video · part 1 of 2", "Talking video · part 2 of 2"]
    assert all(len(_spec(item)["params"]["speech"]["text"]) <= 4500 for item in api.queue.items)


def test_delete_removes_a_file_from_the_library_and_its_records(api, home):
    from avatar_v2.ownership import register_managed_asset

    still = _png(home / "toss.png", size=(300, 400))
    register_managed_asset(desktop.MANAGED_ASSETS_PATH, still, origin="generated_output", job_id="x")
    now = __import__("time").time()
    records = {
        "x": {"id": "x", "status": "done", "label": "Old render", "outputs": [str(still)], "finished_at": now},
        "y": {"id": "y", "status": "done", "label": "Another view", "outputs": [str(still)], "finished_at": now},
    }
    for key, record in records.items():
        (home / "history" / f"{key}.json").write_text(json.dumps(record))
    assert still.is_file()
    deleted = api.studio_delete({"path": str(still)})
    assert deleted["ok"] is True and deleted["records_updated"] == 2
    assert not still.is_file()
    assert not (home / "history" / "x.json").exists() and not (home / "history" / "y.json").exists()
    assert [item["path"] for item in api.studio_library("all")["items"]] == []
    # files the app never recorded cannot be removed through the bridge
    outside = _png(home / "untracked.png", size=(100, 100))
    refused = api.studio_delete({"path": str(outside)})
    assert refused["ok"] is False and outside.is_file()


def test_delete_of_the_master_photo_goes_back_to_a_blank_slate(api, home):
    # Unregistered master photo is DETACHED safely without deleting the original bytes (RC-02 Gate 1)!
    master = _pick_master(api, home)
    res = api.studio_delete({"path": str(master)})
    assert res["ok"] is True
    assert res["detached"] is True and res["deleted_file"] is False
    assert master.is_file()
    assert api._studio_master() == (None, None)  # the app asks which photo to start from now
    profile = api._library_or_none().get_avatar(bi.PROFILE_ID)
    assert "front" not in profile["identity_refs"] and "front" not in profile["reference_hashes"]


def test_angles_and_act_can_start_from_any_my_stuff_picture(api, home):
    other = _png(home / "someone_else.png", size=(300, 400))
    angle = api._studio_payload({"action": "angles", "slots": ["identity_three_quarter_left"], "image": str(other)})
    assert angle["avatar"]["identity_refs"] == [str(other)]  # the variations anchor on the chosen picture
    clip = _video(home / "acting.mp4", 4)
    act = api._studio_payload({"action": "act", "video": str(clip), "image": str(other)})
    assert act["shot"]["references"]["init_image"] == str(other)


def test_any_picture_can_be_added_to_presenter_options(api, home):
    master = _pick_master(api, home)
    old = _png(home / "old_render.png", size=(300, 400))
    (home / "history" / "oldjob.json").write_text(json.dumps(
        {"id": "oldjob", "status": "done", "label": "Old render", "outputs": [str(old)], "finished_at": 0}))
    added = api.studio_presenter_add({"path": str(old)})
    assert added["ok"] is True and added["already"] is False
    assert any(c["path"] == str(old) for c in api.studio_state()["pending_master"])  # now an option for her
    again = api.studio_presenter_add({"path": str(old)})
    assert again["ok"] is True and again["already"] is True  # never duplicated
    assert api.studio_presenter_add({"path": str(home / "missing.png")})["ok"] is False
    clip = _video(home / "clip.mp4", 1)
    assert api.studio_presenter_add({"path": str(clip)})["ok"] is False  # videos aren't face options
    picked = api.studio_pick({"path": str(old), "slot_id": "identity_front"})
    assert picked["ok"] is True and api._studio_master()[0] == str(old.resolve())


def test_delete_keeps_a_record_when_other_outputs_remain(api, home):
    from avatar_v2.ownership import register_managed_asset

    first = _png(home / "keep.png", size=(300, 400))
    second = _png(home / "toss.png", size=(300, 400))
    register_managed_asset(desktop.MANAGED_ASSETS_PATH, second, origin="generated_output", job_id="z")
    now = __import__("time").time()
    (home / "history" / "z.json").write_text(json.dumps(
        {"id": "z", "status": "done", "label": "Pair", "outputs": [str(first), str(second)], "finished_at": now}))
    assert api.studio_delete({"path": str(second)})["ok"] is True
    record = json.loads((home / "history" / "z.json").read_text())
    assert record["outputs"] == [str(first)]
    assert [item["path"] for item in api.studio_library("all")["items"]] == [str(first)]


def test_library_lists_every_output_including_free_gpu_clips(api, home):
    still = _png(home / "photo.png", size=(300, 400))
    clip = _video(home / "clip.mp4", 1)
    now = __import__("time").time()
    records = {
        "a": {"id": "a", "provider": "runway", "status": "done", "label": "Photo", "outputs": [str(still)], "finished_at": now,
              "cloud_receipt": {"credits_charged": 8}},
        "b": {"id": "b", "provider": "comfyui", "status": "done", "label": "Motion clip · 5 s · your GPU", "outputs": [str(clip)],
              "finished_at": now, "studio_action": "move"},
        "c": {"id": "c", "provider": "comfyui", "status": "done", "label": "Old local render", "outputs": [str(clip)], "finished_at": now},
    }
    for key, record in records.items():
        (home / "history" / f"{key}.json").write_text(json.dumps(record))
    everything = api.studio_library("all")["items"]
    assert {item["id"] for item in everything} == {"a", "b", "c"}  # including the old app's local render
    assert {item["id"]: item["source"] for item in everything} == {"a": "runway", "b": "gpu", "c": "old"}
    local = next(item for item in everything if item["id"] == "b")
    assert local["local"] is True and local["credits"] == 0 and local["kind"] == "video"
    assert [item["id"] for item in api.studio_library("image")["items"]] == ["a"]


# --------------------------------------------------------------------------- older work, saved avatars, offline scenes


def test_library_brings_back_the_old_apps_renders(api, home):
    still = _png(home / "old_candidate.png", size=(300, 400))
    clip = _video(home / "old_render.mp4", 1)
    now = __import__("time").time()
    old = {"id": "old1", "status": "done", "label": "avatar_validation_h3 · reddit_full_front", "outputs": [str(clip), str(still)],
           "finished_at": now - 90 * 86400}  # an old-app record: no provider field at all
    (home / "history" / "old1.json").write_text(json.dumps(old))
    items = api.studio_library("all")["items"]
    assert {item["path"] for item in items} == {str(clip), str(still)}
    assert all(item["source"] == "old" and item["local"] is True and item["credits"] == 0 for item in items)
    assert [item["path"] for item in api.studio_library("video")["items"]] == [str(clip)]


def test_saved_avatars_list_each_profiles_pictures(api, home):
    master = _pick_master(api, home)
    listed = api.studio_avatars()
    assert listed["ok"] is True
    brand = next(a for a in listed["items"] if a["id"] == bi.PROFILE_ID)
    assert brand["pictures"] == [str(master.resolve())] and listed["items"][0]["id"] == bi.PROFILE_ID


def test_bad_avatars_can_be_removed_but_not_the_current_presenter(api, home):
    _pick_master(api, home)
    other = _png(home / "old_face.png", size=(300, 400))
    entry = dict(bi.profile_entry())
    entry.update({"avatar_id": "old_avatar", "display_name": "Old Avatar", "identity_refs": {"front": str(other)},
                  "reference_hashes": {"front": "sha"}, "build_manifest": {}})
    api._library_or_none().put_avatar(entry)
    assert any(a["id"] == "old_avatar" for a in api.studio_avatars()["items"])
    removed = api.studio_avatar_delete({"avatar_id": "old_avatar"})
    assert removed["ok"] is True and other.is_file()  # the picture itself stays in My stuff
    assert all(a["id"] != "old_avatar" for a in api.studio_avatars()["items"])
    assert api.studio_avatar_delete({"avatar_id": "old_avatar"})["ok"] is False
    refused = api.studio_avatar_delete({"avatar_id": bi.PROFILE_ID})
    assert refused["ok"] is False and "presenter" in refused["error"]


def test_offline_scene_chains_shots_and_waits_for_the_engine(api, home, monkeypatch):
    from avatar_v2 import local_engine

    _fake_engine(home, monkeypatch)
    master = _pick_master(api, home)
    shots = ["She walks in and smiles.", "She sits down at the desk.", "She explains a point with her hands."]
    # a blank slate: even with a master set, a scene never defaults to it
    assert "start picture" in api.studio_price({"action": "scene", "shots": shots})["error"]
    price = api.studio_price({"action": "scene", "shots": shots, "shape": "wide", "image": str(master)})
    assert price["ok"] is True and price["credits"] == 0 and price["shots"] == 3
    assert price["minutes"] == 3 * local_engine.ESTIMATED_MINUTES
    assert "at least one shot" in api.studio_price({"action": "scene", "shots": ["  "], "image": str(master)})["error"]
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "STARTING", "ready": False, "pid": 1})
    assert api.studio_go({"action": "scene", "shots": shots, "image": str(master)})["ok"] is False and api.queue.items == []
    monkeypatch.setattr(api, "comfyui_status", lambda: {"state": "READY", "ready": True, "pid": 1})
    go = api.studio_go({"action": "scene", "shots": shots, "shape": "wide", "image": str(master)})
    assert go["ok"] is True and go["shots"] == 3 and len(api.queue.items) == 3
    first, second, third = api.queue.items
    assert [i.label for i in api.queue.items] == [f"Scene · shot {n} of 3 · your GPU" for n in (1, 2, 3)]
    assert first.chain_from_previous is False and first.job.asset_map["INIT_IMAGE"] == str(master.resolve())
    assert second.chain_from_previous and second.runtime["chain_from_id"] == first.id
    assert third.runtime["chain_from_id"] == second.id and third.runtime["scene_shots"] == [first.id, second.id, third.id]
    assert (first.job.asset_map["WAN_WIDTH"], first.job.asset_map["WAN_HEIGHT"]) == (832, 480)
    assert all(item.runtime["studio_action"] == "scene" and item.profile == "low_memory" for item in api.queue.items)
    refused = api.studio_price({"action": "scene", "shots": ["a 15-year-old girl walks in"], "image": str(master)})
    assert refused["ok"] is False and "adult" in refused["error"]


def test_scene_from_another_picture_describes_motion_only(api, home, monkeypatch):
    _fake_engine(home, monkeypatch)
    other = _png(home / "landscape.png", size=(832, 480), color=(40, 90, 140))
    item = api._studio_scene_items({"image": str(other), "shots": ["The camera pans across the room."]})[0]
    assert item.job.asset_map["INIT_IMAGE"] == str(other)
    assert "woman in her mid-20s" not in item.job.prompt and item.job.prompt.startswith("The camera pans across the room.")


def test_long_scene_descriptions_are_kept_whole(api, home, monkeypatch):
    _fake_engine(home, monkeypatch)
    long_text = " ".join(
        ["She walks to the whiteboard, uncaps a marker, and draws three boxes while explaining each one in turn."]
        * 18
    )  # about 1,900 characters: well past the old 600-character cut
    other = _png(home / "wide.png", size=(832, 480), color=(40, 90, 140))
    item = api._studio_scene_items({"image": str(other), "shots": [long_text]})[0]
    assert long_text in item.job.prompt  # any other picture: the description is used whole
    master = _pick_master(api, home)
    item = api._studio_scene_items({"image": str(master), "shots": [long_text]})[0]
    assert long_text in item.job.prompt  # the presenter's own picture: identity kept, description not cut


def test_scene_caps_at_twelve_shots(api, home, monkeypatch):
    _fake_engine(home, monkeypatch)
    master = _pick_master(api, home)
    shots = [f"Shot {n}: she continues from the exact pose the last shot ended on." for n in range(1, 14)]
    price = api.studio_price({"action": "scene", "shots": shots, "image": str(master)})
    assert price["ok"] is True and price["shots"] == 12  # the 13th is dropped at the cap
    items = api._studio_scene_items({"shots": shots, "image": str(master)})
    assert len(items) == 12 and items[-1].runtime["scene_shots"] == [item.id for item in items]


def test_finished_scene_joins_its_shots_into_one_video(api, home):
    def frames(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return int(out.stdout.strip())

    first = _video(home / "shot1.mp4", 1)
    second = _video(home / "shot2.mp4", 1)
    (home / "history" / "shot1id.json").write_text(json.dumps({"id": "shot1id", "status": "done", "outputs": [str(first)]}))
    last = SimpleNamespace(id="shot2id", outputs=[str(second)], output_hashes={}, runtime={"scene_shots": ["shot1id", "shot2id"]},
                           provider_config=SimpleNamespace(provider="comfyui"))
    result = api._candidate_qa(last)
    assert result["ok"] is True and result["shots"] == 2 and result["file"] == "scene_shot2id.mp4"
    joined = Path(last.outputs[0])
    assert joined.name == "scene_shot2id.mp4" and last.output_hashes[str(joined)]
    assert frames(joined) == frames(first) + frames(second) - 1  # the repeated boundary frame is dropped
    assert Path(last.outputs[1]) == second  # the shot itself is kept


def test_chained_shot_uses_exactly_its_previous_shot(tmp_path):
    import threading

    from avatar_v2.render_queue import RenderQueue

    queue = RenderQueue.__new__(RenderQueue)
    queue._lock, queue.history_dir = threading.RLock(), tmp_path
    clip = _video(tmp_path / "a.mp4", 1)
    older = SimpleNamespace(id="older", status="done", outputs=[str(clip)], runtime={})
    failed = SimpleNamespace(id="shot1", status="error", outputs=[], runtime={})
    current = SimpleNamespace(id="shot2", status="queued", outputs=[], runtime={"chain_from_id": "shot1"})
    queue._items = [older, failed, current]
    assert queue._previous_output(current) is None  # never falls back to an unrelated older video
    queue._items = [older, current]  # shot 1 was cleared from the list: its saved record is used
    (tmp_path / "shot1.json").write_text(json.dumps({"status": "done", "outputs": [str(clip)]}))
    assert queue._previous_output(current) == str(clip)
