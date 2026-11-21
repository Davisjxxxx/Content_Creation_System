from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from .reference_cache import _encoded_reference_image, _local_vision_url


CANDIDATE_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "same_subject": {"type": "string", "enum": ["same", "different", "uncertain"]},
        "identity_consistency": {"type": "number", "minimum": 0, "maximum": 1},
        "body_proportions": {"type": "number", "minimum": 0, "maximum": 1},
        "slot_match": {"type": "number", "minimum": 0, "maximum": 1},
        "persistent_features": {"type": "number", "minimum": 0, "maximum": 1},
        "artifact_free": {"type": "number", "minimum": 0, "maximum": 1},
        "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": [
        "same_subject",
        "identity_consistency",
        "body_proportions",
        "slot_match",
        "persistent_features",
        "artifact_free",
        "reasons",
    ],
}

CANDIDATE_QA_PROMPT = """Picture 1 is an authorized avatar source reference. Picture 2 is a locally generated candidate. Compare only visible consistency; never identify, name, recognize, search for, or report demographic traits about the person. Evaluate stable face geometry when visible, visible body proportions, whether Picture 2 matches the requested profile slot, visible persistent marks, and obvious generation artifacts. Do not invent details hidden or absent in Picture 1. A score means visual support, not identity proof. Return only the requested schema-valid JSON fields."""


def _score(value: Any) -> float:
    try:
        return round(max(0.0, min(float(value), 1.0)), 3)
    except (TypeError, ValueError):
        return 0.0


def compare_candidate_to_reference(
    reference: Path,
    candidate: Path,
    *,
    slot_id: str,
    slot_group: str,
    base_url: str,
    model: str,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run advisory, local-only visual QA after a designer render completes."""

    reference = Path(reference).expanduser().resolve()
    candidate = Path(candidate).expanduser().resolve()
    if not reference.is_file() or not candidate.is_file():
        raise ValueError("candidate QA requires readable local source and candidate images")
    url = _local_vision_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{CANDIDATE_QA_PROMPT}\nRequested slot: {slot_id}. Slot group: {slot_group}.",
                "images": [_encoded_reference_image(reference), _encoded_reference_image(candidate)],
            }
        ],
        "format": CANDIDATE_QA_SCHEMA,
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 360},
    }

    def run(active_client: httpx.Client) -> dict[str, Any]:
        response = active_client.post(f"{url}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        content = json.loads(response.json()["message"]["content"])
        same_subject = str(content.get("same_subject") or "uncertain")
        if same_subject not in {"same", "different", "uncertain"}:
            same_subject = "uncertain"
        scores = {
            key: _score(content.get(key))
            for key in (
                "identity_consistency",
                "body_proportions",
                "slot_match",
                "persistent_features",
                "artifact_free",
            )
        }
        visible_scores = [scores["slot_match"], scores["artifact_free"]]
        if slot_group == "identity":
            visible_scores.extend([scores["identity_consistency"], scores["persistent_features"]])
        else:
            visible_scores.extend([scores["body_proportions"], scores["persistent_features"]])
        overall = round(sum(visible_scores) / len(visible_scores), 3)
        if same_subject == "different" or min(scores["slot_match"], scores["artifact_free"]) < 0.45:
            decision = "reject"
        elif same_subject == "same" and overall >= 0.72 and min(visible_scores) >= 0.55:
            decision = "pass"
        else:
            decision = "review"
        reasons = [str(item).strip()[:240] for item in (content.get("reasons") or []) if str(item).strip()][:6]
        return {
            "ok": True,
            "decision": decision,
            "same_subject": same_subject,
            "overall_score": overall,
            "scores": scores,
            "reasons": reasons,
            "advisory": True,
        }

    if client is not None:
        return run(client)
    with httpx.Client(timeout=150) as active_client:
        return run(active_client)
