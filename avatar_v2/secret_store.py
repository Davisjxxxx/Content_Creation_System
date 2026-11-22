"""Local credential storage for optional cloud providers.

Read order for a provider key:

1. The provider's environment variable. For Runway this is ``RUNWAYML_API_SECRET``, the
   variable Runway's official SDKs read.
2. The operating-system keyring, only when the optional ``keyring`` package is installed
   and has a usable backend.
3. A user-only file under ``~/.avatar_v2/secrets/`` (directory 0700, file 0600).

Keys are never written to ``desktop.json``, never returned to the UI, and never logged.
Status calls report only whether a key exists and where it came from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

KEYRING_SERVICE = "avatar-v2"
SECRETS_DIR = Path.home() / ".avatar_v2" / "secrets"

# provider id -> (environment variable, storage name)
PROVIDER_SECRETS: dict[str, tuple[str, str]] = {
    "runway": ("RUNWAYML_API_SECRET", "runway_api_key"),
}


class SecretStoreError(ValueError):
    """Raised for invalid secrets or unknown providers. Messages never contain the secret."""


def _provider(provider: str) -> tuple[str, str]:
    try:
        return PROVIDER_SECRETS[provider]
    except KeyError as exc:
        raise SecretStoreError(f"Unknown credential provider '{provider}'.") from exc


def _keyring() -> Any | None:
    """Return the keyring module when it is installed with a real backend, else None."""
    try:
        import keyring  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    module = type(backend).__module__.lower()
    if "fail" in module or "null" in module or type(backend).__name__ in {"FailKeyring", "NullKeyring"}:
        return None
    return keyring


def _validate(secret: str) -> str:
    value = str(secret or "").strip()
    if not value:
        raise SecretStoreError("The API key is empty.")
    if any(character.isspace() for character in value):
        raise SecretStoreError("The API key must not contain spaces or line breaks.")
    if len(value) < 16 or len(value) > 512:
        raise SecretStoreError("The API key length is not plausible (expected 16-512 characters).")
    return value


def _file_path(name: str) -> Path:
    return SECRETS_DIR / name


def _remove_file(name: str) -> bool:
    try:
        _file_path(name).unlink()
        return True
    except FileNotFoundError:
        return False


def get_secret(provider: str) -> tuple[str | None, str | None]:
    """Return ``(secret, source)`` where source is environment, keyring or file."""
    env_var, name = _provider(provider)
    from_env = os.environ.get(env_var, "").strip()
    if from_env:
        return from_env, "environment"

    keyring = _keyring()
    if keyring is not None:
        try:
            stored = keyring.get_password(KEYRING_SERVICE, name)
        except Exception:
            stored = None
        if stored:
            return stored, "keyring"

    try:
        stored = _file_path(name).read_text(encoding="utf-8").strip()
    except OSError:
        stored = ""
    if stored:
        return stored, "file"
    return None, None


def save_secret(provider: str, secret: str) -> str:
    """Persist a key outside the repository and return where it was stored."""
    value = _validate(secret)
    _env_var, name = _provider(provider)

    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.set_password(KEYRING_SERVICE, name, value)
            _remove_file(name)
            return "keyring"
        except Exception:
            pass  # fall back to the user-only file

    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(SECRETS_DIR, 0o700)
    except OSError:
        pass
    path = _file_path(name)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return "file"


def clear_secret(provider: str) -> bool:
    """Remove a stored key from the keyring and the file store. Environment keys are untouched."""
    _env_var, name = _provider(provider)
    removed = False
    keyring = _keyring()
    if keyring is not None:
        try:
            if keyring.get_password(KEYRING_SERVICE, name):
                keyring.delete_password(KEYRING_SERVICE, name)
                removed = True
        except Exception:
            pass
    return _remove_file(name) or removed


def secret_status(provider: str) -> dict[str, Any]:
    """Describe key availability without revealing any part of the key."""
    env_var, _name = _provider(provider)
    secret, source = get_secret(provider)
    return {
        "configured": bool(secret),
        "source": source,
        "env_var": env_var,
        "keyring_available": _keyring() is not None,
    }
