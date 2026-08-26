from pathlib import Path


def test_desktop_package_includes_ui_and_bundled_workflows():
    spec = (Path(__file__).parents[1] / "scripts" / "avatar_v2.spec").read_text(encoding="utf-8")
    assert '"avatar_v2", "ui", "index.html"' in spec
    assert '"animatediff_sd15_api.json"' in spec
    assert '"animatediff_sdxl_api.json"' in spec
    assert spec.count('"workflows"') >= 2
