"""Host-side entry point for the Native Avatar V2 Python SDK (Sprint 4, Section 6)."""

from __future__ import annotations

from .sdk_client import (
    SDK_CONTRACT_VERSION,
    AvatarV2Client,
    GenerationResult,
    RunLimitReached,
    RunStopped,
    SDKError,
    connect,
)

__all__ = [
    "SDK_CONTRACT_VERSION",
    "AvatarV2Client",
    "GenerationResult",
    "RunLimitReached",
    "RunStopped",
    "SDKError",
    "connect",
]
