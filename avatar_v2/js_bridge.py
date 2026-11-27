"""The only object handed to pywebview as ``js_api``.

pywebview exposes every public method of the ``js_api`` object to page JavaScript, and it also
recurses into public non-callable attributes. Passing ``DesktopAPI`` directly therefore exposed
its internals (the render queue, the avatar library and their ``Path`` objects, for example
``queue.work_dir.unlink``). This facade exposes exactly the operations the desktop UI calls,
listed explicitly below, holds the real API in a private slot that pywebview skips, and
enforces page-boundary authority on settings and file arguments (Astra RC-03).

When the UI needs a new operation, add it to ``JS_API_METHODS``; the test suite checks that the
UI's calls and this list stay in sync.
"""

from __future__ import annotations

import threading
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

# Settings keys that page JavaScript is permitted to modify via JsBridge.save_settings().
# Privileged engine paths/executables/args (comfyui_dir, comfyui_python, comfyui_args,
# comfyui_url, workflow_*, archive_dir, etc.) are NEVER writable from page JS!
JS_UI_EDITABLE_SETTINGS: frozenset[str] = frozenset(
    {
        "runway_job_limit",
        "runway_batch_limit",
        "runway_daily_limit",
        "runway_monthly_limit",
        "studio_look",
        "studio_extra",
        "gpu_autoclean",
        "gpu_threshold",
        "gpu_check_interval_s",
        "local_vlm_inspection",
        "local_auto_correct",
        "offline_mode",
    }
)

_BRIDGE_CTX = threading.local()


def in_js_bridge() -> bool:
    """True while executing a call dispatched from the page-facing JsBridge."""
    return bool(getattr(_BRIDGE_CTX, "active", False))


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
        if name == "save_settings":
            raw = args[0] if args else {}
            if not isinstance(raw, dict):
                return {"ok": False, "error": "Settings payload must be a JSON object."}
            disallowed = sorted(str(k) for k in raw.keys() if str(k) not in JS_UI_EDITABLE_SETTINGS)
            if disallowed:
                return {
                    "ok": False,
                    "error": (
                        "Privileged engine or path settings cannot be changed from the page bridge: "
                        + ", ".join(disallowed)
                    ),
                }
        prev = getattr(_BRIDGE_CTX, "active", False)
        _BRIDGE_CTX.active = True
        try:
            return getattr(self._api, name)(*args)
        finally:
            _BRIDGE_CTX.active = prev

    call.__name__ = name
    call.__qualname__ = f"JsBridge.{name}"
    call.__doc__ = f"Forwards to DesktopAPI.{name}."
    return call


for _name in JS_API_METHODS:
    setattr(JsBridge, _name, _proxy(_name))
del _name
