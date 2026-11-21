"""Re-encode OpenCV MP4 files so browsers can play them."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _ffmpeg_path() -> str | None:
    """Find ffmpeg on the system or from imageio-ffmpeg."""
    # First, check if ffmpeg is installed on the operating system.
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        # If not, fall back to the imageio-ffmpeg bundled version.
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return None


def make_browser_playable(video_path: Path) -> Path:
    """Convert the file to H.264 so the result page video player works.

    OpenCV often writes mp4v, which many browsers refuse. This step overwrites
    the file with a browser-friendly copy when ffmpeg is available.
    """
    video_path = Path(video_path)
    # If the file is missing or empty, just return it as-is.
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return video_path

    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        return video_path

    # Create a temporary path for the browser-friendly video.
    temp_path = video_path.with_name(video_path.stem + "_web.mp4")
    
    # The FFmpeg command: convert to H.264 video, optimize for web playback, and remove audio.
    command = [
        ffmpeg,
        "-y",                  # Overwrite output files without asking
        "-i", str(video_path), # Input file
        "-c:v", "libx264",     # Use the widely supported H.264 codec
        "-preset", "ultrafast",# Process as fast as possible
        "-crf", "28",          # Set compression quality (28 is good for web)
        "-pix_fmt", "yuv420p", # Ensure compatibility with most web browsers
        "-movflags", "+faststart", # Allow the video to start playing before it's fully downloaded
        "-an",                 # Remove audio (not needed for this video)
        str(temp_path),        # Output to the temporary file
    ]
    
    # Run the FFmpeg command.
    completed = subprocess.run(command, capture_output=True, text=True)
    
    # If FFmpeg failed, or didn't create the file, delete the temp file and return the original.
    if completed.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        return video_path

    # Replace the original OpenCV video with the new browser-friendly video.
    video_path.unlink(missing_ok=True)
    temp_path.rename(video_path)
    return video_path