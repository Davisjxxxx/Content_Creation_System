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
    # The simple Studio app (Make / My stuff / Settings) calls exactly these.
    "cancel_queue_item", "character_profiles", "choose_files", "comfyui_start", "comfyui_stop", "engine_status", "forge_go",
    "forge_price", "forge_state", "get_state", "open_output_dir",
    "open_path", "pick_local_images", "runway_check", "runway_save_key", "save_settings",
    "script_control", "script_select",
    "studio_avatar_delete", "studio_avatars", "studio_delete", "studio_go", "studio_image", "studio_library",
    "studio_pick", "studio_presenter_add", "studio_price", "studio_state",
    "studio_talker_status", "studio_thumb", "studio_video", "ui_test_report",
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
