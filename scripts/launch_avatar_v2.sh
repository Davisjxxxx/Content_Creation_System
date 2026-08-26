#!/usr/bin/env bash
# Avatar V2 desktop launcher — cwd-independent.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AVATAR_V2_PYTHON:-$APP_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Avatar V2 venv not found at $PYTHON_BIN" >&2
    echo "Create it with: python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e '.[desktop]'" >&2
    exit 1
fi

cd "$APP_DIR"

# Prefer the tested PyInstaller build when it exists; fall back to the project venv.
if [ -x "$APP_DIR/dist/AvatarV2/AvatarV2" ]; then
    exec "$APP_DIR/dist/AvatarV2/AvatarV2" "$@"
fi

exec "$PYTHON_BIN" -m avatar_v2.desktop "$@"
