"""The Free builder: prompt-first offline image + video scene toolset (no network here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avatar_v2 import desktop, local_engine, offline_builder as ob, secret_store
from avatar_v2.desktop import DesktopAPI
from avatar_v2.js_bridge import JS_API_METHODS
from avatar_v2.library import LibraryStore


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
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


def _fake_builder_engine(tmp_path: Path, monkeypatch, *, include_9b: bool = True, include_5b: bool = True) -> Path:
    comfy = tmp_path / "engine" / "comfyui"
    (comfy / "custom_nodes" / "ComfyUI-GGUF").mkdir(parents=True, exist_ok=True)
    (comfy / "main.py").write_text("# fake")
    python = tmp_path / "engine" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    files = [
        (subfolder, name)
        for subfolder, name in ob.SHARED_MODELS.values()
    ]
    files.append(("diffusion_models", ob.PHOTO_MODELS["4b"]["unet"]))
    files.append(("text_encoders", ob.PHOTO_MODELS["4b"]["text_encoder"]))
    if include_9b:
        files.append(("diffusion_models", ob.PHOTO_MODELS["9b"]["unet"]))
        files.append(("text_encoders", ob.PHOTO_MODELS["9b"]["text_encoder"]))
    else:
        (comfy / "models" / "diffusion_models" / ob.PHOTO_MODELS["9b"]["unet"]).unlink(missing_ok=True)
    files.extend(local_engine.MODELS.values())
    if include_5b:
        files.append(("diffusion_models", local_engine.MODELS_5B["WAN5B_MODEL"][1]))
        files.append(("vae", local_engine.MODELS_5B["WAN5B_VAE"][1]))
    for subfolder, name in files:
        target = comfy / "models" / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(b"x")
    monkeypatch.setattr(desktop, "ENGINE_HOME", tmp_path / "engine")
    return comfy


def _ready(api):
    monkeypatch_state = {"started": False}

    def status():
        return {"state": "READY", "ready": True, "pid": 1234}

    return status


# ------------------------------------------------------------------ prompt splitting


def test_split_scene_prompt_handles_sizes_and_caps():
    assert ob.split_scene_prompt("") == []
    assert ob.split_scene_prompt("   ") == []
    assert ob.split_scene_prompt("She waves.") == ["She waves."]
    short = "She smiles. She nods. She waves."
    assert ob.split_scene_prompt(short) == ["She smiles. She nods. She waves."]
    long_sentence = "x" * 300
    shots = ob.split_scene_prompt(f"{long_sentence}. Another {long_sentence}.")
    assert len(shots) == 2 and all(len(s) <= 320 for s in shots)
    many = ". ".join([f"shot number {i} " + "y" * 260 for i in range(30)]) + "."
    shots = ob.split_scene_prompt(many)
    assert len(shots) == ob.SCENE_MAX_SHOTS == 12
    assert ob.split_scene_prompt("no periods here just words" * 5) == ["no periods here just words" * 5]


# ------------------------------------------------------------------ workflow graphs


def test_flux2_txt2img_workflow_graph():
    from avatar_v2.workflows_builder import build_flux2_txt2img_workflow
    from avatar_v2.providers.comfyui import replace_placeholders

    mapping = {
        "PROMPT": "a woman",
        "NEGATIVE_PROMPT": "text",
        "WIDTH": 896,
        "HEIGHT": 1280,
        "SEED": 5,
        "STEPS": 8,
        "CFG": 1.2,
        "SAMPLER_NAME": "euler",
        "OUTPUT_PREFIX": "avatar_v2/test",
        **ob.image_placeholders("4b"),
    }
    graph = build_flux2_txt2img_workflow(mapping)
    classes = {node["class_type"] for node in graph.values()}
    assert {
        "UnetLoaderGGUF",
        "CLIPLoaderGGUF",
        "VAELoader",
        "CLIPTextEncode",
        "EmptySD3LatentImage",
        "Flux2Scheduler",
        "RandomNoise",
        "KSamplerSelect",
        "CFGGuider",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "SaveImage",
    } <= classes
    clip = next(node for node in graph.values() if node["class_type"] == "CLIPLoaderGGUF")
    assert clip["inputs"]["type"] == "flux2"  # Qwen3 GGUF encoder; CPU by default in ComfyUI 0.46
    assert clip["inputs"]["clip_name"] == ob.PHOTO_MODELS["4b"]["text_encoder"]
    latent = next(node for node in graph.values() if node["class_type"] == "EmptySD3LatentImage")
    assert (latent["inputs"]["width"], latent["inputs"]["height"]) == (896, 1280)
    scheduler = next(node for node in graph.values() if node["class_type"] == "Flux2Scheduler")
    assert scheduler["inputs"]["steps"] == 8
    resolved = replace_placeholders(graph, mapping)
    assert "${" not in json.dumps(resolved)


def test_wan5b_workflow_resolves_every_slot(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()  # installs the presenter profile the local jobs build from
    image = _png(home / "start.png")
    item = api._local_wan_item(image=str(image), text="She waves.", shape="tall", label="clip",
                               action="scene", video_model="5b")
    from avatar_v2.providers.comfyui import replace_placeholders

    workflow = json.loads(local_engine.WORKFLOW_5B.read_text())
    resolved = replace_placeholders(workflow, item.job.asset_map)
    assert "${" not in json.dumps(resolved)
    assert resolved["1"]["inputs"]["unet_name"] == local_engine.MODELS_5B["WAN5B_MODEL"][1]
    assert resolved["6"]["inputs"]["vae_name"] == local_engine.MODELS_5B["WAN5B_VAE"][1]  # 48-ch Wan 2.2 VAE
    assert resolved["3"]["inputs"]["device"] == "default"  # balanced mode keeps the encoder on the GPU
    assert resolved["9"]["inputs"]["steps"] == 4 and resolved["9"]["inputs"]["cfg"] == 1.0


def test_cpu_offload_moves_the_wan_encoder_to_the_cpu(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    image = _png(home / "start.png")
    item = api._local_wan_item(image=str(image), text="She waves.", shape="tall", label="clip",
                               action="scene", video_model="a14b", vram_mode="cpu_offload")
    assert item.job.asset_map["WAN_CLIP_DEVICE"] == "cpu"


# ------------------------------------------------------------------ forge actions


def test_forge_photo_is_free_and_queues_one_flux_item(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    price = api.forge_price({"action": "photo", "prompt": "A confident instructor in her 30s."})
    assert price["ok"] is True and price["credits"] == 0 and price["local"] is True and price["shots"] == 1
    go = api.forge_go({"action": "photo", "prompt": "A confident instructor in her 30s.", "photo_model": "4b"})
    assert go["ok"] is True and go["local"] is True and go["minutes"] == ob.PHOTO_MODELS["4b"]["estimated_minutes"]
    item = api.queue.items[0]
    assert item.workflow == "builtin:flux2_txt2img" and item.profile == "low_memory"  # forced GPU cleanup first
    assert item.runtime["studio_action"] == "forge_photo" and item.runtime["engine"] == "flux2_txt2img"
    assert item.job.asset_map["FORGE_TEXT_ENCODER"] == "Qwen3-4B-Q5_K_M.gguf"  # runs on the CPU by default
    assert item.job.asset_map["PROMPT"] == "A confident instructor in her 30s."
    assert (item.job.asset_map["WIDTH"], item.job.asset_map["HEIGHT"]) == ob.PHOTO_SHAPES["portrait"]


def test_forge_photo_9b_license_is_flagged_and_missing_models_refused(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch, include_9b=True)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    price = api.forge_price({"action": "photo", "prompt": "A woman.", "photo_model": "9b"})
    assert price["ok"] is True
    assert any("non-commercial" in warning for warning in price["warnings"])
    _fake_builder_engine(home, monkeypatch, include_9b=False)
    refused = api.forge_price({"action": "photo", "prompt": "A woman.", "photo_model": "9b"})
    assert refused["ok"] is False and "setup_offline_builder.sh" in refused["error"]


def test_forge_scene_splits_prompt_and_uses_the_chosen_video_model(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    image = _png(home / "start.png")
    prompt = ("She sits at the desk. " + "a" * 300 + ". Then she stands up. " + "b" * 300 + ".")
    price = api.forge_price({"action": "scene", "prompt": prompt, "image": str(image), "video_model": "5b"})
    assert price["ok"] is True and price["shots"] >= 2 and price["photo_first"] is False
    go = api.forge_go({"action": "scene", "prompt": prompt, "image": str(image), "video_model": "5b", "shape": "wide"})
    assert go["ok"] is True and go["shots"] == price["shots"]
    items = api.queue.items
    assert all(item.workflow == str(local_engine.WORKFLOW_5B) for item in items)
    assert all(item.runtime["studio_action"] == "forge_scene" for item in items)  # shown in the Free builder, not Make
    assert items[0].job.asset_map["INIT_IMAGE"] == str(image)
    assert items[1].chain_from_previous and items[1].runtime["chain_from_id"] == items[0].id
    assert items[-1].runtime.get("scene_shots") == [item.id for item in items]  # stitched on the last shot
    assert (items[0].job.asset_map["WAN_WIDTH"], items[0].job.asset_map["WAN_HEIGHT"]) == local_engine.SHAPES["wide"]


def test_forge_scene_without_image_is_two_stage_and_continues_itself(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    prompt = "She walks in. " + "c" * 300 + ". She sits down. " + "d" * 300 + "."
    go = api.forge_go({"action": "scene", "prompt": prompt, "video_model": "a14b"})
    assert go["ok"] is True and go["photo_first"] is True
    assert len(api.queue.items) == 1
    photo = api.queue.items[0]
    assert photo.runtime["studio_action"] == "forge_scene" and photo.runtime.get("candidate_qa") is True
    spec = photo.runtime["forge_scene"]
    assert len(spec["shots"]) >= 2 and spec["video_model"] == "a14b"
    # The start photo renders; the queue's done-hook turns it into the scene shots.
    photo.outputs = [str(_png(home / "made.png"))]
    result = api._forge_continue_scene(photo)
    assert result["ok"] is True and result["scene"] == "queued" and result["shots"] == len(spec["shots"])
    scene_items = api.queue.items[1:]
    assert len(scene_items) == len(spec["shots"])
    assert scene_items[0].job.asset_map["INIT_IMAGE"] == str(home / "made.png")
    assert scene_items[0].runtime["studio_action"] == "forge_scene"
    assert scene_items[1].chain_from_previous
    # No usable photo output: the hook reports it instead of queueing junk.
    broken = api.queue.items[0]
    broken.outputs = []
    assert api._forge_continue_scene(broken)["ok"] is False


def test_forge_scene_sends_the_users_words_verbatim(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    image = _png(home / "start.png")
    prompt = "She sits at the desk. " + "a" * 300 + ". Then she stands up. " + "b" * 300 + "."
    go = api.forge_go({"action": "scene", "prompt": prompt, "image": str(image), "video_model": "5b"})
    assert go["ok"] is True
    items = api.queue.items
    split = ob.split_scene_prompt(prompt)
    assert items[0].job.prompt == split[0]  # exactly the user's words
    assert "photorealistic motion" not in items[0].job.prompt.lower()  # no baked-in wording
    assert "no text" not in items[0].job.prompt.lower()
    # The Make recipes keep their prompt wrapping; only the Free builder is verbatim.
    make_item = api._local_wan_item(image=str(image), text="She waves.", shape="tall", label="clip", action="scene")
    assert "micro-expressions" in make_item.job.prompt.lower()  # the Make recipes' realism wrap


def test_forge_photo_makes_options_with_fresh_seeds(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    price = api.forge_price({"action": "photo", "prompt": "A woman.", "count": 4})
    assert price["ok"] is True and price["shots"] == 4
    assert price["minutes"] == ob.PHOTO_MODELS["4b"]["estimated_minutes"] * 4
    go = api.forge_go({"action": "photo", "prompt": "A woman.", "photo_model": "4b", "count": 4})
    assert go["ok"] is True and go["shots"] == 4 and len(api.queue.items) == 4
    seeds = {item.job.asset_map["SEED"] for item in api.queue.items}
    assert len(seeds) == 4  # every option gets its own seed, so the images differ
    assert all("Avatar photo" in item.label for item in api.queue.items)


def test_forge_accepts_the_full_prompt_length(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    monkeypatch.setattr(api, "comfyui_status", _ready(api))
    long_prompt = "A detailed description of the scene. " * 300  # ~12,300 chars, capped at 8,000
    go = api.forge_go({"action": "photo", "prompt": long_prompt, "count": 1})
    assert go["ok"] is True
    assert len(api.queue.items[0].job.asset_map["PROMPT"]) == ob.PROMPT_MAX_CHARS


def test_forge_refuses_minors_and_needs_a_prompt(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    api.studio_state()
    refused = api.forge_price({"action": "photo", "prompt": "a 15-year-old girl"})
    assert refused["ok"] is False and "adult" in refused["error"]
    assert api.forge_price({"action": "photo", "prompt": ""})["ok"] is False
    assert api.forge_price({"action": "scene", "prompt": ""})["ok"] is False


def test_forge_methods_are_bridged_and_state_reports_models(api, home, monkeypatch):
    _fake_builder_engine(home, monkeypatch)
    for name in ("forge_state", "forge_price", "forge_go"):
        assert name in JS_API_METHODS
    state = api.forge_state()
    assert state["ok"] is True and state["installed"] is True
    assert [m["id"] for m in state["photo_models"]] == ["4b", "9b"]
    assert state["photo_models"][0]["installed"] is True and state["photo_models"][0]["license"] == "apache-2.0"
    assert state["photo_models"][1]["license"] == "flux-non-commercial"
    assert [m["id"] for m in state["video_models"]] == ["a14b", "5b"]
    assert state["video_models"][0]["installed"] is True and state["video_models"][1]["installed"] is True
    assert state["max_shots"] == 12 and state["max_prompt_chars"] == 8000
