"""Shared FFmpeg helpers; no GUI or model dependencies at import time."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction


ENCODERS = {"h264": "libx264", "hevc": "libx265", "av1": "libaom-av1",
            "h264(NVENC)": "h264_nvenc", "hevc(NVENC)": "hevc_nvenc"}
FORMATS = {"avi": {"h264", "h264(NVENC)"},
           "mov": set(ENCODERS) - {"av1"}, "mkv": set(ENCODERS),
           "mp4": set(ENCODERS), "webm": {"av1"},
           "flv": {"h264", "h264(NVENC)"},
           "hls": set(ENCODERS) - {"av1"}, "dash": set(ENCODERS),
           "srt": set(ENCODERS) - {"av1"}}
# Historical values describe input protocols, not file extensions.
ALIASES = {"m3u8": "hls", "http": "hls", "https": "hls",
           "rtmp": "flv", "rtsp": "srt"}
EXTENSIONS = {"hls": "m3u8", "dash": "mpd", "srt": "ts"}


def check_interrupt():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def executable(name):
    value = os.environ.get("ZZX_" + name.upper(), name)
    found = shutil.which(value)
    if not found:
        raise RuntimeError(f"{name} not found. Install FFmpeg with ffprobe and add it to PATH, "
                           f"or set ZZX_{name.upper()} to the executable path.")
    return found


def run_process(command, timeout=None):
    """Poll cancellation without pipe deadlocks or unbounded RAM for FFmpeg logs."""
    check_interrupt()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            started = time.monotonic()
            try:
                while process.poll() is None:
                    check_interrupt()
                    if timeout is not None and time.monotonic() - started > timeout:
                        raise RuntimeError("Media operation timed out; check the input stream.")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if process.returncode:
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 8000))
                detail = stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Media command failed ({process.returncode}): {detail}")
            stdout.seek(0)
            return stdout.read().decode("utf-8", errors="replace")


def validate_format(video_format, codec):
    video_format = ALIASES.get(video_format, video_format)
    if codec not in FORMATS.get(video_format, set()):
        raise ValueError(f"Unsupported format/codec: {video_format}/{codec}. "
                         "Use MP4/MKV with H.264; WebM requires AV1. MSS is no longer supported.")
    return video_format


def encoding_options(codec, quality):
    if codec not in ENCODERS or not 5 <= int(quality) <= 40:
        raise ValueError("Invalid codec or quality (expected 5–40).")
    options = ["-c:v", ENCODERS[codec], "-pix_fmt", "yuv420p"]
    if "NVENC" in codec:
        options += ["-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    else:
        options += ["-crf", str(quality)]
        if codec == "av1":
            options += ["-b:v", "0", "-cpu-used", "6"]
        else:
            options += ["-preset", "medium"]
    return options


def output_file(directory, filename, video_format, default):
    if not directory.strip():
        try:
            import folder_paths
            directory = folder_paths.get_output_directory()
        except ImportError:
            directory = str(Path.cwd() / "output")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    filename = filename.strip() or default
    if any(c in filename for c in '/\\:%') or filename in {".", ".."}:
        raise ValueError("output_filename must be a simple filename (no path or %).")
    stem = Path(filename).stem
    if video_format in {"hls", "dash"}:
        stem += "_" + video_format
    extension = EXTENSIONS.get(video_format, video_format)
    for counter in range(100000):
        path = directory / f"{stem}_{counter:04d}.{extension}"
        try:
            # Reserve atomically, including the namespace for segmented outputs.
            with path.open("xb"):
                pass
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many outputs with this filename.")


def mux_options(video_format, path):
    if video_format == "hls":
        return ["-f", "hls", "-hls_time", "10", "-hls_playlist_type", "vod",
                "-hls_segment_filename", str(path.with_name(path.stem + "_%05d.ts"))]
    if video_format == "dash":
        return ["-f", "dash", "-seg_duration", "10", "-init_seg_name",
                path.stem + "_init_$RepresentationID$.m4s", "-media_seg_name",
                path.stem + "_$RepresentationID$_$Number%05d$.m4s"]
    if video_format == "srt":
        return ["-f", "mpegts"]
    if video_format in {"mp4", "mov"}:
        return ["-movflags", "+faststart"]
    return []


def remove_failed_output(path, video_format):
    path.unlink(missing_ok=True)
    if video_format in {"hls", "dash"}:
        # Only remove files in the namespace reserved by this execution.
        for candidate in path.parent.iterdir():
            if candidate.name.startswith(path.stem + "_") and candidate.suffix in {".ts", ".m4s", ".tmp"}:
                candidate.unlink(missing_ok=True)


def probe(path):
    raw = run_process([executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
                       "-show_streams", "-show_format", "-of", "json", str(path)], timeout=30)
    data = json.loads(raw)
    if not data.get("streams"):
        raise ValueError("Input contains no video stream.")
    stream = data["streams"][0]
    def number(value):
        try:
            result = float(Fraction(str(value)))
            return result if math.isfinite(result) and result >= 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    fps = number(stream.get("avg_frame_rate")) or number(stream.get("r_frame_rate"))
    duration = number(stream.get("duration")) or number(data.get("format", {}).get("duration"))
    count = int(number(stream.get("nb_frames"))) or round(duration * fps)
    return {"fps": fps, "frame_count": count, "duration": duration,
            "width": int(stream["width"]), "height": int(stream["height"])}
