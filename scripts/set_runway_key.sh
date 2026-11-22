#!/usr/bin/env bash
# Save (or replace) the Runway API key for Avatar V2.
# The key is read with hidden input and handed to the app's secret store over stdin, so it is never
# echoed, logged, kept in shell history, or visible on a command line. It is stored outside the
# repository: the OS keyring when one is available, otherwise ~/.avatar_v2/secrets/runway_api_key (mode 600).
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AVATAR_V2_PYTHON:-$APP_DIR/.venv314/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$APP_DIR/.venv/bin/python"
cd "$APP_DIR"

finish() {
    echo
    read -rp "Press Enter to close this window." _ || true
}
trap finish EXIT

echo "Avatar V2 · save Runway API key"
echo "Stored outside the repository; never shown, logged, or written to settings."
echo
read -rsp "Paste your Runway API key and press Enter (input is hidden): " RUNWAY_KEY
echo
if [ -z "${RUNWAY_KEY//[[:space:]]/}" ]; then
    echo "No key entered; nothing was saved."
    exit 0
fi

if ! printf '%s' "$RUNWAY_KEY" | "$PYTHON_BIN" -c '
import sys
from avatar_v2.secret_store import SecretStoreError, save_secret, secret_status
try:
    where = save_secret("runway", sys.stdin.read())
except SecretStoreError as exc:
    print("Not saved: " + str(exc))
    sys.exit(1)
status = secret_status("runway")
print("Saved to the " + where + " store. Key configured: " + str(status["configured"]) + " (active source: " + str(status["source"]) + ").")
if status["source"] == "environment":
    print("Note: RUNWAYML_API_SECRET is set in this environment and takes precedence over the saved key.")
'; then
    unset RUNWAY_KEY
    exit 0
fi
unset RUNWAY_KEY

echo
read -rp "Check the key with Runway now? It is free and only shows your credit balance. [y/N] " CHECK || CHECK=""
if [[ "${CHECK:-}" =~ ^[Yy]$ ]]; then
    "$PYTHON_BIN" -c '
from avatar_v2.providers import runway as r
try:
    with r.client_from_store(timeout_s=30) as client:
        org = client.organization()
    print("Key accepted by Runway. Credit balance: " + str(org.get("creditBalance")) + " credits.")
except r.RunwayError as exc:
    print("Check failed: " + str(exc))
' || true
fi

echo
echo "Next: open Avatar V2 → Advanced → Runway cloud provider, tick Enable and the upload consent, then Save."
echo "Nothing is generated or billed until you confirm a job's upload list and cost."
