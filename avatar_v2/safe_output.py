"""Create media privately and publish complete files atomically without replacing a name."""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable


def private_output_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".avatar-output-", dir=parent))


def publish_output(target: Path, produce: Callable[[Path], None], *, rename_on_collision: bool = True) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".avatar-output-", dir=target.parent) as reserved:
        staged = Path(reserved) / target.name
        produce(staged)
        if staged.is_symlink() or not staged.is_file() or staged.stat().st_size == 0:
            raise OSError(f"Output producer did not create a regular nonempty file: {target.name}")
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate = target
        while True:
            try:
                os.link(staged, candidate, follow_symlinks=False)
                directory = os.open(candidate.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return candidate
            except FileExistsError:
                if not rename_on_collision:
                    raise
                candidate = target.with_name(f"{target.stem}_{uuid.uuid4().hex}{target.suffix}")


def publish_text(target: Path, text: str) -> Path:
    return publish_output(target, lambda staged: staged.write_text(text, encoding="utf-8"))


def render_output(command: list[str], target: Path, *, timeout: int = 600) -> Path:
    """The subprocess sees only a fresh name inside an exclusive private directory."""
    def render(staged: Path) -> None:
        subprocess.run([command[0], "-nostdin", "-n", *command[1:], str(staged)],
                       check=True, capture_output=True, timeout=timeout)
        from .media_validation import validate_media
        validate_media(staged)
    return publish_output(target, render)
