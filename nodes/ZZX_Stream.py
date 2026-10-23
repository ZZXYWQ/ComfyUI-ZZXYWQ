import datetime
import time
from . import video_utils as media


class StreamRecorder:
    @classmethod
    def INPUT_TYPES(s):
        now = datetime.datetime.now()
        default_start_time = now.strftime("%H/%M/%S")  # Example default time
        return {
            "required": {
                "stream_url": ("STRING", {"multiline": False, "default": ""}),
                "use_local_time": (["true", "false"], {"default": "false"}),
                "start_time": ("STRING", {"multiline": False, "default": default_start_time}),  # Example default time
                "record_hours": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1, "display": "slider"}),
                "record_minutes": ("INT", {"default": 1, "min": 0, "max": 59, "step": 1, "display": "slider"}),
                "record_seconds": ("INT", {"default": 0, "min": 0, "max": 59, "step": 1, "display": "slider"}),
                "output_filename": ("STRING", {"multiline": False, "default": ""}),
                "video_format": (["avi", "mov", "mkv", "mp4", "hls", "dash", "srt", "flv", "webm", "rtmp", "rtsp", "m3u8", "http", "https"], {"default": "mp4"}),
                "codec": (["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"], {"default": "h264"}),
                "video_quality": ("INT", {"default": 23, "min": 5, "max": 40, "step": 1, "display": "slider"}),
                "use_cuda": (["true", "false"], {"default": "false"}),  # Add CUDA option
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_filename",)
    FUNCTION = "record_stream"

    CATEGORY = "ZZX/Stream"

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_format_support(self, video_format, codec):
        return media.validate_format(video_format, codec)

    def record_stream(self, stream_url, use_local_time, start_time, record_hours, record_minutes, record_seconds, output_filename, video_format, codec, video_quality, use_cuda, output_path):
        video_format = self.check_format_support(video_format, codec)
        if not stream_url.strip():
            raise ValueError("Stream URL is required.")
        hours, minutes, seconds = int(record_hours), int(record_minutes), int(record_seconds)
        # Accept legacy workflows whose seconds default was 60.
        if not (0 <= hours <= 12 and 0 <= minutes <= 59 and 0 <= seconds <= 60):
            raise ValueError("Invalid recording duration.")
        duration = hours * 3600 + minutes * 60 + seconds
        if duration <= 0:
            raise ValueError("Recording duration must be greater than zero.")
        command = [media.executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin"]
        if use_cuda == "true":
            command += ["-hwaccel", "cuda"]
        command += ["-rw_timeout", "15000000", "-i", stream_url.strip(), "-t", str(duration),
                    "-map", "0:v:0", "-map", "0:a:0?"]
        command += media.encoding_options(codec, video_quality)
        command += ["-c:a", "libopus" if video_format == "webm" else "aac", "-b:a", "128k"]
        if use_local_time == "true":
            try:
                parts = [int(value) for value in start_time.replace(":", "/").split("/")]
                if len(parts) != 3:
                    raise ValueError
                now = datetime.datetime.now()
                start = now.replace(hour=parts[0], minute=parts[1], second=parts[2], microsecond=0)
            except ValueError as exc:
                raise ValueError("start_time must be HH/MM/SS or HH:MM:SS.") from exc
            if start < now:
                raise ValueError("Scheduled time has already passed today (ComfyUI server local time).")
            while datetime.datetime.now() < start:
                media.check_interrupt()
                time.sleep(0.2)
        target = media.output_file(output_path, output_filename, video_format, "recording")
        try:
            command += media.mux_options(video_format, target) + ["-y", target.as_posix()]
            media.run_process(command, timeout=duration + 60)
            if target.stat().st_size == 0:
                raise RuntimeError("Recording produced no video data.")
        except BaseException:
            media.remove_failed_output(target, video_format)
            raise
        return (str(target),)


NODE_CLASS_MAPPINGS = {"StreamRecorder": StreamRecorder}
NODE_DISPLAY_NAME_MAPPINGS = {"StreamRecorder": "Stream Recorder"}
