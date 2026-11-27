import os
from pathlib import Path

import pytest


@pytest.fixture
def measured_local_gpu(monkeypatch):
    """Supply measured headroom to renderer doubles in tests that require successful admission."""
    commands = Path(__file__).parent / "fixtures" / "rc5" / "telemetry-bin"
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
