"""The only object handed to pywebview as ``js_api``.

pywebview exposes every public method of the ``js_api`` object to page JavaScript, and it also
recurses into public non-callable attributes. Passing ``DesktopAPI`` directly therefore exposed
its internals (the render queue, the avatar library and their ``Path`` objects, for example
``queue.work_dir.unlink``). This facade exposes exactly the operations the desktop UI calls,
listed explicitly below, and holds the real API in a private slot that pywebview skips.

When the UI needs a new operation, add it to ``JS_API_METHODS``; the test suite checks that the
UI's calls and this list stay in sync.
"""

from __future__ import annotations

from typing import Any, Callable

JS_API_METHODS: tuple[str, ...] = (
    "anatomy_guides_delete", "anatomy_guides_list", "anatomy_guides_save",
    "appearance_components_delete", "appearance_components_list", "appearance_components_save",
    "avatar_candidate_decide", "avatar_designer_coverage", "avatar_designer_prompt",
    "avatar_designer_renderer_status", "avatar_designer_scan_cache", "avatar_manifest_open",
    "brand_instructor_install", "brand_instructor_preset", "brand_instructor_spec",
    "cancel_queue_item", "choose_files", "choose_folder", "clear_finished",
    "comfyui_start", "comfyui_status", "comfyui_stop", "compose", "doctor", "dry_run",
    "get_state", "gpu_details", "gpu_free", "gpu_status", "gpu_terminate_pids", "history",
    "lesson_list", "lesson_plan", "lesson_queue_changed", "lesson_save",
    "motions_delete", "motions_list", "motions_save", "open_output_dir", "open_path",
    "provider_profile_save", "provider_profiles_list", "provider_status",
    "qa_reports_list", "qa_review_save", "qa_run",
    "queue_avatar_designer", "queue_best_of", "queue_profile_sweep", "queue_render", "queue_runway",
    "queue_state", "queue_upscale",
    "reference_import", "reference_imports_list", "reference_thumbnail", "review_queue",
    "runway_avatar_create", "runway_avatars_list", "runway_check", "runway_clear_key", "runway_estimate",
    "runway_reconcile_task", "runway_resolve_charge", "runway_save_key", "runway_spend", "runway_status",
    "runway_unresolved", "runway_voice_create", "runway_voice_preview",
    "save_adult_prompt_option", "save_settings", "scenes_delete", "scenes_list", "scenes_save",
    "script_analyze", "ui_test_report", "vault_apply", "vault_delete", "vault_list", "vault_save",
    "verify_outputs", "voice_profile_save", "voice_profiles_list", "voiceover_decide", "voiceover_import",
    "voiceover_list",
)


class JsBridge:
    """Explicit allowlist facade over DesktopAPI (no public attributes, only listed methods)."""

    __slots__ = ("_api",)

    def __init__(self, api: Any) -> None:
        missing = [name for name in JS_API_METHODS if not callable(getattr(api, name, None))]
        if missing:
            raise AttributeError(f"Desktop API is missing bridged operations: {', '.join(missing)}")
        self._api = api


def _proxy(name: str) -> Callable[..., Any]:
    def call(self: JsBridge, *args: Any) -> Any:
        return getattr(self._api, name)(*args)

    call.__name__ = name
    call.__qualname__ = f"JsBridge.{name}"
    call.__doc__ = f"Forwards to DesktopAPI.{name}."
    return call


for _name in JS_API_METHODS:
    setattr(JsBridge, _name, _proxy(_name))
del _name
