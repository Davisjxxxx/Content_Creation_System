import json
from pathlib import Path

import pytest
from PIL import Image

from avatar_v2.candidate_qa import compare_candidate_to_reference


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response({"message": {"content": json.dumps(self.content)}})


def _image(path: Path, color: str) -> Path:
    Image.new("RGB", (64, 96), color).save(path)
    return path


def test_candidate_qa_is_loopback_only_and_requires_local_files(tmp_path: Path):
    source = _image(tmp_path / "source.png", "red")
    candidate = _image(tmp_path / "candidate.png", "blue")
    with pytest.raises(ValueError, match="loopback"):
        compare_candidate_to_reference(
            source,
            candidate,
            slot_id="body_full_front",
            slot_group="body",
            base_url="https://example.com",
            model="vision",
        )


def test_candidate_qa_normalizes_scores_and_returns_advisory_pass(tmp_path: Path):
    source = _image(tmp_path / "source.png", "red")
    candidate = _image(tmp_path / "candidate.png", "blue")
    client = _Client(
        {
            "same_subject": "same",
            "identity_consistency": 0.9,
            "body_proportions": 0.84,
            "slot_match": 0.88,
            "persistent_features": 0.76,
            "artifact_free": 0.92,
            "reasons": ["Visible geometry is consistent."],
        }
    )
    result = compare_candidate_to_reference(
        source,
        candidate,
        slot_id="body_full_front",
        slot_group="body",
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b-instruct",
        client=client,
    )
    assert result["decision"] == "pass"
    assert result["advisory"] is True
    assert result["overall_score"] == 0.85
    assert client.calls[0][0] == "http://127.0.0.1:11434/api/chat"
    payload = client.calls[0][1]["json"]
    assert len(payload["messages"][0]["images"]) == 2
    assert "never identify" in payload["messages"][0]["content"]


def test_candidate_qa_rejects_visible_slot_or_artifact_failure(tmp_path: Path):
    source = _image(tmp_path / "source.png", "red")
    candidate = _image(tmp_path / "candidate.png", "blue")
    client = _Client(
        {
            "same_subject": "uncertain",
            "identity_consistency": 0.5,
            "body_proportions": 0.5,
            "slot_match": 0.8,
            "persistent_features": 0.5,
            "artifact_free": 0.2,
            "reasons": ["Visible deformation."],
        }
    )
    result = compare_candidate_to_reference(
        source,
        candidate,
        slot_id="body_full_rear",
        slot_group="body",
        base_url="http://localhost:11434",
        model="vision",
        client=client,
    )
    assert result["decision"] == "reject"
