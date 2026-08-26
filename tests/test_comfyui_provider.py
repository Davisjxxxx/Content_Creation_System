import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from avatar_v2.models import ProviderConfig
from avatar_v2.providers import comfyui
from avatar_v2.providers.comfyui import ComfyUIExecutionError, ComfyUIProvider


class FakeResponse:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("failed", request=None, response=None)

    def json(self):
        return self.data


class FakeClient:
    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def _response(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeResponse(self.responses.pop(0))

    def get(self, url, **kwargs):
        return self._response("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._response("POST", url, **kwargs)


def _provider(**updates):
    return ComfyUIProvider(ProviderConfig(base_url="http://127.0.0.1:8188/", **updates))


def test_health_object_info_queue_interrupt_status_and_free(monkeypatch):
    responses = [
        {"system": "ok"},
        {"Node": {}},
        {"prompt_id": "prompt-1"},
        {},
        {"queue_running": [[1]], "queue_pending": [[2], [3]]},
        {},
    ]
    calls = []
    monkeypatch.setattr(comfyui.httpx, "Client", lambda **_kwargs: FakeClient(responses, calls))
    provider = _provider()

    assert provider.health() == {"system": "ok"}
    assert provider.object_info("Node") == {"Node": {}}
    assert provider.queue({"1": {}}) == "prompt-1"
    provider.interrupt()
    assert provider.queue_status() == {"running": 1, "pending": 2, "busy": True}
    provider.free_memory(unload_models=False)
    assert calls[1][1].endswith("/object_info/Node")
    assert calls[2][2]["json"] == {"prompt": {"1": {}}}
    assert calls[-1][2]["json"] == {"unload_models": False, "free_memory": True}


def test_queue_requires_prompt_id(monkeypatch):
    monkeypatch.setattr(comfyui.httpx, "Client", lambda **_kwargs: FakeClient([{}], []))
    with pytest.raises(RuntimeError, match="did not return prompt_id"):
        _provider().queue({})


def test_capabilities_filters_h3_nodes_and_models(monkeypatch):
    provider = _provider()
    provider.object_info = lambda node_name=None: {
        "MiniMaxH3ReferenceToVideo": {},
        "Other": {},
    }
    provider.list_models = lambda: {
        "diffusion_models": ["minimax_h3_ref2va.safetensors", "other.safetensors"],
        "text_encoders": ["qwen3vl.safetensors", "clip.safetensors"],
        "vae": ["minimax_h3_vae.safetensors", "vae.safetensors"],
    }
    result = provider.capabilities()
    assert result["h3_available"] is True
    assert result["h3_nodes"] == ["MiniMaxH3ReferenceToVideo"]
    assert result["h3_models"]["diffusion_models"] == ["minimax_h3_ref2va.safetensors"]
    assert result["h3_models"]["text_encoders"] == ["qwen3vl.safetensors"]


def test_list_models_accepts_both_combo_schema_shapes_and_skips_errors():
    provider = _provider()

    def object_info(node_name=None):
        if node_name == "UNETLoader":
            return {node_name: {"input": {"required": {"unet_name": [["a", "b"]]}}}}
        if node_name == "CLIPLoader":
            return {node_name: {"input": {"required": {"clip_name": ["STRING", {"options": ["c"]}]}}}}
        raise KeyError(node_name)

    provider.object_info = object_info
    assert provider.list_models() == {
        "diffusion_models": ["a", "b"],
        "text_encoders": ["c"],
        "vae": [],
    }


def test_stage_assets_separates_same_basename_and_deduplicates_same_path(tmp_path: Path):
    first = tmp_path / "one" / "body.png"
    second = tmp_path / "two" / "body.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    provider = _provider()
    uploads = []

    def upload(path, *, subfolder):
        uploads.append((Path(path), subfolder))
        return f"{subfolder}/{Path(path).name}"

    provider.upload_file = upload
    job = SimpleNamespace(
        job_id="job 1",
        asset_map={
            "H3_PICTURE_1": str(first),
            "H3_PICTURE_2": str(second),
            "H3_PICTURE_3": str(first),
            "H3_DIFFUSION_MODEL": "not-an-asset.safetensors",
        },
    )
    staged = provider._stage_assets(job)
    assert len(uploads) == 2
    assert uploads[0][1] == "avatar_v2/job_1/h3_picture_1"
    assert uploads[1][1] == "avatar_v2/job_1/h3_picture_2"
    assert staged["H3_PICTURE_1"] != staged["H3_PICTURE_2"]
    assert staged["H3_PICTURE_3"] == staged["H3_PICTURE_1"]


def test_stage_assets_copies_to_configured_input_and_rejects_missing(tmp_path: Path):
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    input_dir = tmp_path / "input"
    provider = _provider(input_dir=str(input_dir))
    job = SimpleNamespace(job_id="job", asset_map={"INIT_IMAGE": str(source)})
    staged = provider._stage_assets(job)
    assert (input_dir / staged["INIT_IMAGE"]).read_bytes() == b"image"

    job.asset_map["INIT_IMAGE"] = str(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError, match="INIT_IMAGE asset does not exist"):
        provider._stage_assets(job)


def test_resolve_custom_workflow_and_unknown_builtin(tmp_path: Path):
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"1": {"inputs": {"image": "${INIT_IMAGE}", "seed": "${SEED}"}}}))
    job = SimpleNamespace(job_id="job", asset_map={"INIT_IMAGE": "image.png", "SEED": 7})
    provider = _provider()
    resolved = provider.resolve_workflow(workflow, job, stage_assets=False)
    assert resolved["1"]["inputs"] == {"image": "image.png", "seed": 7}

    with pytest.raises(RuntimeError, match="No builtin workflow builder"):
        provider.resolve_workflow("builtin:not-real", job, stage_assets=False)


def test_wait_handles_completion_node_error_and_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        comfyui.httpx,
        "Client",
        lambda **_kwargs: FakeClient([{"p": {"outputs": {"1": {"images": []}}}}], calls),
    )
    assert _provider(timeout_s=30).wait("p")["outputs"]

    error = {
        "p": {
            "status": {
                "status_str": "error",
                "messages": [["execution_error", {"node_id": "4", "node_type": "Sampler", "exception_message": "CUDA OOM"}]],
            }
        }
    }
    monkeypatch.setattr(comfyui.httpx, "Client", lambda **_kwargs: FakeClient([error], []))
    with pytest.raises(ComfyUIExecutionError, match="node 4 .*CUDA OOM"):
        _provider(timeout_s=30).wait("p")

    clock = iter([0.0, 31.0])
    monkeypatch.setattr(comfyui.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(comfyui.httpx, "Client", lambda **_kwargs: FakeClient([], []))
    with pytest.raises(TimeoutError, match="exceeded"):
        _provider(timeout_s=30).wait("p")


def test_output_files_and_render_pipeline():
    provider = _provider()
    history = {
        "outputs": {
            "1": {
                "images": [{"filename": "a.png", "subfolder": "images"}],
                "videos": [{"filename": "b.mp4"}],
                "audio": ["invalid"],
            }
        }
    }
    assert provider.output_files(history) == ["images/a.png", "b.mp4"]
    provider.resolve_workflow = lambda *_args, **_kwargs: {"1": {}}
    provider.queue = lambda workflow: "prompt-2"
    provider.wait = lambda prompt_id: history
    result = provider.render("builtin:h3_ref2va", SimpleNamespace())
    assert result == {"prompt_id": "prompt-2", "outputs": ["images/a.png", "b.mp4"], "history": history}
