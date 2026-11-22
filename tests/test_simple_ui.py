"""The desktop page is the simple Studio app: Make (recipes), My stuff, Settings."""

from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")


def test_three_views_and_six_step_by_step_recipes():
    for view in ("make", "library", "settings"):
        assert f'id="view-{view}"' in HTML
    recipes = re.findall(r'<section class="recipe" id="recipe-([a-z]+)"', HTML)
    assert recipes == ["look", "photos", "voice", "talk", "act", "move"]
    for recipe in recipes:
        section = HTML.split(f'id="recipe-{recipe}"', 1)[1].split("</section>", 1)[0]
        assert '<ol class="steps">' in section and 'class="btn primary go"' in section  # numbered steps + one button


def test_removed_screens_and_conflicting_controls_are_gone():
    for gone in ('id="adultRole"', 'id="designerRunNext"', 'id="rwTask"', 'id="profiles"', 'data-engine="h3_ref2va"',
                 'id="mCategory"', 'id="windStrength"', "Pro tools", 'id="gpuFreeButton"', 'id="termModal"'):
        assert gone not in HTML


def test_every_result_has_a_visible_thumbnail_and_a_full_size_viewer():
    assert "function makeTile(" in HTML and "stThumb(img,path)" in HTML
    assert 'id="viewer"' in HTML and "Open in its own window" in HTML and "Actual size" in HTML
    assert "slice(0,4)" not in HTML  # no options hidden behind a "see all" button


def test_dropdowns_stay_readable_and_the_self_test_hook_exists():
    assert "-webkit-appearance:none;appearance:none" in HTML
    assert "-webkit-text-fill-color:#111!important" in HTML and "background-color:#fff!important" in HTML
    assert "location.search.indexOf('uitest')" in HTML and "ui_test_report" in HTML
