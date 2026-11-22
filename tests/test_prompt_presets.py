from pathlib import Path

from avatar_v2.prompt_presets import (
    ADULT_PROMPT_DEFAULTS,
    ADULT_PROMPT_PRESETS,
    adult_prompt_presets_payload,
)


def test_adult_prompt_presets_cover_all_requested_sections():
    expected = {"roles", "environments", "actions", "toys", "moves", "positions"}
    assert set(ADULT_PROMPT_PRESETS) == expected
    assert set(ADULT_PROMPT_DEFAULTS) == expected

    for category, options in ADULT_PROMPT_PRESETS.items():
        assert len(options) >= 6
        ids = {option["id"] for option in options}
        assert len(ids) == len(options)
        assert ADULT_PROMPT_DEFAULTS[category] in ids
        assert all(option["label"] and option["prompt"] for option in options)


def test_adult_prompt_payload_is_a_mutable_copy():
    first = adult_prompt_presets_payload()
    second = adult_prompt_presets_payload()
    first["categories"]["roles"][0]["label"] = "changed"
    assert second["categories"]["roles"][0]["label"] != "changed"


def test_adult_prompt_payload_merges_only_valid_custom_options():
    payload = adult_prompt_presets_payload({
        "roles": [
            {"id": "custom_director", "label": "Director", "prompt": "an adult scene director"},
            {"id": "broken", "label": "", "prompt": "missing label"},
        ],
        "not_a_category": [{"id": "x", "label": "X", "prompt": "ignored"}],
    })
    custom = payload["categories"]["roles"][-1]
    assert custom == {
        "id": "custom_director",
        "label": "Director",
        "prompt": "an adult scene director",
        "custom": True,
    }
    assert "not_a_category" not in payload["categories"]
