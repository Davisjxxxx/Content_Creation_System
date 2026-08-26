import json

import avatar_v2.desktop as desktop


def test_save_adult_prompt_option_persists_and_reloads(monkeypatch, tmp_path):
    config = tmp_path / "desktop.json"
    monkeypatch.setattr(desktop, "CONFIG_PATH", config)
    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)

    result = api.save_adult_prompt_option("moves", "Shoulder roll", "perform a slow shoulder roll")

    assert result["ok"] is True
    assert result["item"]["id"] == "custom_shoulder_roll"
    stored = json.loads(config.read_text(encoding="utf-8"))
    assert stored["adult_prompt_custom"]["moves"][0]["label"] == "Shoulder roll"
    payload = api.get_state()["adult_prompt_presets"]
    assert payload["categories"]["moves"][-1]["prompt"] == "perform a slow shoulder roll"


def test_save_adult_prompt_option_rejects_invalid_input(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "desktop.json")
    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)
    assert not api.save_adult_prompt_option("unknown", "X", "Y")["ok"]
    assert not api.save_adult_prompt_option("roles", "", "Y")["ok"]


def test_exact_primary_reference_routes_to_first_frame_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(desktop, "CONFIG_PATH", tmp_path / "desktop.json")
    api = desktop.DesktopAPI.__new__(desktop.DesktopAPI)
    avatar, shot = api._models_from_payload({
        "reference_mode": "exact_primary",
        "avatar": {"identity_refs": ["primary.png"]},
        "shot": {
            "engine_preference": "h3_ref2va",
            "user_prompt": "preserve the subject",
            "references": {"init_image": "primary.png"},
        },
    })
    assert avatar.identity_refs == ["primary.png"]
    assert shot.engine_preference == "h3_fl2va"
