"""The desktop page is the simple Studio app: Make (recipes), My stuff, Settings."""

from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).parents[1] / "avatar_v2" / "ui" / "index.html").read_text(encoding="utf-8")


def test_three_views_and_seven_step_by_step_recipes():
    for view in ("make", "library", "settings"):
        assert f'id="view-{view}"' in HTML
    recipes = re.findall(r'<section class="recipe" id="recipe-([a-z]+)"', HTML)
    assert recipes == ["look", "photos", "voice", "talk", "act", "move", "scene"]
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
    assert 'id="viewerVideo"' in HTML and "studio_video" in HTML
    assert "viewerReplay()" in HTML and "viewerToggleLoop()" in HTML and "viewerStepFrame(" in HTML
    assert "slice(0,4)" not in HTML  # no options hidden behind a "see all" button


def test_dropdowns_stay_readable_and_the_self_test_hook_exists():
    assert "-webkit-appearance:none;appearance:none" in HTML
    assert "-webkit-text-fill-color:#111!important" in HTML and "background-color:#fff!important" in HTML
    assert "location.search.indexOf('uitest')" in HTML and "ui_test_report" in HTML


def test_my_stuff_pages_large_libraries_and_lists_saved_avatars():
    assert 'data-k="avatars"' in HTML and "studio_avatars" in HTML
    assert "Show more (" in HTML and "IntersectionObserver" in HTML  # thumbnails load as they scroll into view
    assert "Make a scene from this" in HTML and "made with the old app" in HTML
    assert "studio_delete" in HTML and "Remove this from My stuff for good" in HTML  # every tile can be deleted


def test_every_image_recipe_picks_any_picture_from_my_stuff():
    for picker in ('id="moveImages"', 'id="sceneImages"', 'id="anglesImages"', 'id="actImages"'):
        assert picker in HTML  # video, scene, photos and act-it-out all have a start-photo picker
    assert "Choose from My stuff" in HTML  # the full library is one tap away, never dumped into the step
    assert "No presenter is set — pick the photo to start from" in HTML  # blank slate when no master exists
    assert "{requirePick:true}" in HTML and "Pick the photo to start from:" in HTML  # the scene never auto-loads a picture
    assert "Change…" in HTML and "hide every other picture" in HTML  # after picking, the other pictures collapse away
    assert "Detach her master photo reference" in HTML  # RC-02: presenter master Remove detaches without deleting external originals
    assert "studio_avatar_delete" in HTML and "Remove this avatar so it is no longer an option" in HTML  # bad avatars can go
    assert "studio_presenter_add" in HTML and "Add from My stuff" in HTML and "Presenter option" in HTML  # older pictures can join her face options


def test_scene_and_motion_descriptions_have_no_short_character_cap():
    assert 'maxlength="600"' not in HTML  # "What happens" / shot descriptions are never cut at a short limit


def test_scene_goes_to_twelve_shots_with_a_full_video_template():
    assert "st.sceneShots.length>=12" in HTML and "Fill in a full-video template" in HTML
    assert "SCENE_TEMPLATE" in HTML and "starts from the previous shot's last frame" in HTML
