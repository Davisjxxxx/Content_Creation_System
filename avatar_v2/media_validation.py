"""Validate the actual bytes used for a terminal media receipt."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageOps
from .integrity import sha256_file

IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP",
                 ".bmp": "BMP", ".tiff": "TIFF", ".tif": "TIFF", ".gif": "GIF"}
CONTAINERS = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".webm": "webm", ".mkv": "matroska",
              ".avi": "avi", ".wav": "wav", ".mp3": "mp3", ".m4a": "m4a", ".ogg": "ogg", ".flac": "flac"}


def validate_media(path: Path) -> dict:
    suffix = path.suffix.lower()
    before = path.stat()
    if not path.is_file() or before.st_size == 0:
        raise ValueError(f"Missing or empty output: {path.name}")
    if suffix in IMAGE_FORMATS:
        with Image.open(path) as image:
            if image.format != IMAGE_FORMATS[suffix]:
                raise ValueError(f"Image format disagrees with filename: {path.name}")
            image.verify()
        with Image.open(path) as image:
            frame_count = getattr(image, "n_frames", 1)
            for index in range(frame_count):
                image.seek(index)
                image.load()
            image.seek(0)
            width, height = ImageOps.exif_transpose(image).size
            facts = dict(kind="image", width=width, height=height, frames=frame_count, format=image.format)
    elif suffix in CONTAINERS:
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
                                "-of", "json", str(path)], capture_output=True, check=True, timeout=600)
        if probe.stderr.strip():
            raise ValueError(f"Media decoder reported errors: {path.name}")
        data = json.loads(probe.stdout)
        container = data["format"]
        if CONTAINERS[suffix] not in container["format_name"].split(","):
            raise ValueError(f"Unexpected container: {path.name}")
        kind = "audio" if suffix in {".wav", ".mp3", ".m4a", ".ogg", ".flac"} else "video"
        stream = next((s for s in data["streams"] if s.get("codec_type") == kind), None)
        if not stream or int(stream.get("nb_read_frames") or 0) < 1:
            raise ValueError(f"No decodable {kind} stream: {path.name}")
        duration = float(container.get("duration") or 0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration: {path.name}")
        facts = dict(kind=kind, duration_s=duration, format=container["format_name"])
        if kind == "video":
            frames = int(stream["nb_read_frames"])
            declared = stream.get("nb_frames")
            if declared and declared != "N/A" and int(declared) != frames:
                raise ValueError(f"Decoded frame count differs from container: {path.name}")
            width, height = int(stream["width"]), int(stream["height"])
            if min(width, height) <= 0:
                raise ValueError(f"Invalid dimensions: {path.name}")
            facts.update(width=width, height=height, frames=frames, fps=round(frames/duration, 3))
    else:
        raise ValueError(f"Unsupported output media: {path.name}")
    digest = sha256_file(path)
    after = path.stat()
    if not digest or (before.st_dev, before.st_ino, before.st_size, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_ctime_ns
    ):
        raise ValueError(f"Output changed during validation: {path.name}")
    return {**facts, "path": str(path), "exists": True, "size_bytes": after.st_size, "sha256": digest}
