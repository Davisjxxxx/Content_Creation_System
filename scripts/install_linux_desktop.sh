#!/usr/bin/env bash
# Install the Avatar V2 desktop launcher (.desktop entry + icon) for the current user.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ICON_DIR="$HOME/.local/share/icons/hicolor"
APPS_DIR="$HOME/.local/share/applications"

mkdir -p "$ICON_DIR/128x128/apps" "$ICON_DIR/256x256/apps" "$ICON_DIR/512x512/apps" "$APPS_DIR"

for size in 128 256 512; do
    cp -f "$APP_DIR/assets/icons/avatar_v2_${size}.png" "$ICON_DIR/${size}x${size}/apps/avatar-v2.png"
done

sed "s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/scripts/avatar-v2.desktop.in" > "$APPS_DIR/avatar-v2.desktop"
chmod +x "$APP_DIR/scripts/launch_avatar_v2.sh"

update-desktop-database "$APPS_DIR" 2>/dev/null || true

# Desktop shortcut when the desktop directory exists.
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
    cp -f "$APPS_DIR/avatar-v2.desktop" "$DESKTOP_DIR/avatar-v2.desktop"
    chmod +x "$DESKTOP_DIR/avatar-v2.desktop"
    gio set "$DESKTOP_DIR/avatar-v2.desktop" metadata::trusted true 2>/dev/null || true
    echo "Desktop shortcut: $DESKTOP_DIR/avatar-v2.desktop"
fi

echo "Launcher installed: $APPS_DIR/avatar-v2.desktop"
echo "Icon installed: $ICON_DIR"
