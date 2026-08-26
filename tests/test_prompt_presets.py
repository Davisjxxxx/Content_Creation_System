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


def test_desktop_ui_keeps_vault_provenance_and_black_select_text():
    html = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "-webkit-appearance:none;appearance:none" in html
    assert "-webkit-text-fill-color:#111!important" in html
    assert "background-color:#fff!important" in html
    assert "avatar:{...state.avatar" in html
    assert "state.avatar={avatar_id:a.avatar_id" in html
    assert "state.avatar.age_verified_18_plus===true" in html
    assert "state.avatar.consent_confirmed===true" in html
    for control in ("adultRole", "adultEnvironment", "adultAction", "adultToy", "adultMove", "adultPosition"):
        assert f'id="{control}"' in html
    assert "Other… · add and save" in html
    assert 'id="adultCustomEditor"' in html
    assert "save_adult_prompt_option" in html
    assert 'id="h3ReferenceMode"' in html
    assert "exact_primary" in html
    assert "setPrimaryIdentity()" in html
