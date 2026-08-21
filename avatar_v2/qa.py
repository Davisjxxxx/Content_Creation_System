from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def ffprobe_video(path: str | Path) -> dict[str, Any]:
    video = Path(path)
    if not video.exists():
        raise FileNotFoundError(video)
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required for video QA; install ffmpeg first")

    cmd = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise RuntimeError("no video stream found")

    return {
        "path": str(video),
        "codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "duration_s": float(data.get("format", {}).get("duration", 0) or 0),
        "size_bytes": int(data.get("format", {}).get("size", 0) or 0),
    }
