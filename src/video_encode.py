"""Make OpenCV MP4 files playable in web browsers (H.264 + faststart)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ffmpeg_path() -> str | None:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return None


def make_browser_playable(video_path: Path) -> Path:
    """
    Re-encode to H.264/yuv420p so HTML5 <video> can read duration and play.
    Overwrites the file when conversion succeeds.
    """
    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return video_path

    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return video_path

    temp_path = video_path.with_name(video_path.stem + "_web.mp4")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        return video_path

    video_path.unlink(missing_ok=True)
    temp_path.rename(video_path)
    return video_path
