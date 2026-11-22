#!/usr/bin/env bash
# Avatar V2 desktop launcher — cwd-independent.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The system Python is 3.14 and its GTK bindings (gi) exist only for 3.14, so .venv314 is the
# current environment. .venv (Python 3.12) is kept as a rollback for tests and the CLI:
#   AVATAR_V2_PYTHON="$APP_DIR/.venv/bin/python" scripts/launch_avatar_v2.sh
if [ -n "${AVATAR_V2_PYTHON:-}" ]; then
    PYTHON_BIN="$AVATAR_V2_PYTHON"
elif [ -x "$APP_DIR/.venv314/bin/python" ]; then
    PYTHON_BIN="$APP_DIR/.venv314/bin/python"
else
    PYTHON_BIN="$APP_DIR/.venv/bin/python"
fi

cd "$APP_DIR"

# A freshly built package waits in dist/AvatarV2.next while the app is open, and is switched in on
# the next launch. The build it replaces is renamed (never deleted) so it stays available for rollback.
NEXT_BUILD="$APP_DIR/dist/AvatarV2.next"
if [ -x "$NEXT_BUILD/AvatarV2" ] && ! pgrep -f "$APP_DIR/dist/AvatarV2/AvatarV2" >/dev/null 2>&1; then
    if [ -d "$APP_DIR/dist/AvatarV2" ]; then
        mv "$APP_DIR/dist/AvatarV2" "$APP_DIR/dist/AvatarV2-prev-$(date +%Y%m%d-%H%M%S)"
    fi
    mv "$NEXT_BUILD" "$APP_DIR/dist/AvatarV2"
fi

# Prefer the tested PyInstaller build when it exists, unless AVATAR_V2_FROM_SOURCE=1.
# The previous build is preserved under dist/ for rollback (see docs/RECOVERY_AND_ROLLBACK.md).
if [ "${AVATAR_V2_FROM_SOURCE:-0}" != "1" ] && [ -x "$APP_DIR/dist/AvatarV2/AvatarV2" ]; then
    exec "$APP_DIR/dist/AvatarV2/AvatarV2" "$@"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Avatar V2 environment not found at $PYTHON_BIN" >&2
    echo "Create it with: python3.14 -m venv --system-site-packages .venv314 && .venv314/bin/pip install -e '.[desktop,dev]'" >&2
    exit 1
fi

exec "$PYTHON_BIN" -m avatar_v2.desktop "$@"
