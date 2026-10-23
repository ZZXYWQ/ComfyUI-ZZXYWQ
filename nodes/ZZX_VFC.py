from pathlib import Path
from . import video_utils as media


class VideoFormatConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False, "default": ""}),
                "output_enabled": (["true", "false"],),
                "output_filename": ("STRING", {"multiline": False, "default": ""}),
                "video_format": (["avi", "mov", "mkv", "mp4"],{"default": "mp4"}),
                "codec": (["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],{"default": "h264"}),
                "video_quality": ("INT", {"default": 23, "min": 5, "max": 40, "step": 1, "display": "slider"}),
                "frame_rate": (["8", "15", "24", "25", "30", "50", "59", "60", "120"],{"default": "25"}),
                "opencl_acceleration": (["enable", "disable"], {"default": "disable"}),
                "video_width": (["272", "300", "320", "360", "400", "450", "480", "512", "540", "600", "640", "720", "800", "960", "1080", "1280", "1440", "1536", "1920", "2560"], {"default": "1280"}),
                "video_height": (["272", "300", "320", "360", "400", "450", "480", "512", "540", "600", "640", "720", "800", "960", "1080", "1280", "1440", "1536", "1920", "2560"], {"default": "720"}),
                "scaling_filter": (["bilinear", "bicubic", "neighbor", "area", "bicublin", "lanczos"],{"default": "bicubic"}),
                "processing_method": (["fill", "crop"],),
                "audio_codec": (["copy", "mp3", "aac"],{"default": "aac"}),
                "bit_rate": (["96", "128", "192"],{"default": "192"}),
                "audio_channels": (["original", "mono", "stereo"],{"default": "stereo"}),
                "sample_rate": (["44100", "48000"],{"default": "48000"}),
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "VHS_VIDEOINFO")
    RETURN_NAMES = ("output_filename", "video_info")
    FUNCTION = "process_video"

    CATEGORY = "ZZX/Video"

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Saving is an explicit side effect on each queued execution.
        return float("nan")

    def process_video(self, video_path, output_enabled, output_filename, video_format, codec, video_quality, frame_rate, opencl_acceleration, video_width, video_height, scaling_filter, processing_method, audio_codec, bit_rate, audio_channels, sample_rate, output_path):
        if output_enabled == "false":
            return ("", {})
        source = Path(video_path.strip()).expanduser()
        if not video_path.strip() or not source.is_file():
            raise ValueError("video_path must be an existing video file on the ComfyUI server.")
        source = source.resolve()
        video_format = media.validate_format(video_format, codec)
        width, height, fps = int(video_width), int(video_height), float(frame_rate)
        if width <= 0 or height <= 0 or width % 2 or height % 2 or not 0 < fps <= 120:
            raise ValueError("Dimensions must be positive even numbers; frame rate must be 0–120.")
        if scaling_filter not in {"bilinear", "bicubic", "neighbor", "area", "bicublin", "lanczos"}:
            raise ValueError("Invalid scaling filter.")
        if processing_method not in {"fill", "crop"}:
            raise ValueError("processing_method must be fill or crop.")
        if audio_codec not in {"copy", "aac", "mp3"} or audio_channels not in {"original", "mono", "stereo"}:
            raise ValueError("Invalid audio settings.")
        if int(bit_rate) not in {96, 128, 192} or int(sample_rate) not in {44100, 48000}:
            raise ValueError("Invalid audio bitrate or sample rate.")
        if opencl_acceleration == "enable":
            import logging
            logging.warning("ZZX: legacy OpenCL option is ignored; select an NVENC codec for GPU encoding.")
        source_info = media.probe(source)
        scale = f"scale={width}:{height}:flags={scaling_filter}"
        if processing_method == "crop":
            scale += f":force_original_aspect_ratio=increase:force_divisible_by=2,crop={width}:{height}"
        scale += ",setsar=1"
        command = [media.executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                   "-vf", scale, "-r", str(frame_rate)]
        command += media.encoding_options(codec, video_quality)
        command += ["-c:a", {"mp3": "libmp3lame"}.get(audio_codec, audio_codec)]
        if audio_codec != "copy":
            command += ["-b:a", f"{bit_rate}k", "-ar", str(sample_rate)]
            if audio_channels != "original":
                command += ["-ac", "1" if audio_channels == "mono" else "2"]
        target = media.output_file(output_path, output_filename, video_format, source.stem)
        try:
            command += media.mux_options(video_format, target) + ["-y", target.as_posix()]
            media.run_process(command)
            loaded_info = media.probe(target)
        except BaseException:
            media.remove_failed_output(target, video_format)
            raise
        info = {"source_" + key: value for key, value in source_info.items()}
        info.update({"loaded_" + key: value for key, value in loaded_info.items()})
        return (str(target), info)


NODE_CLASS_MAPPINGS = {"VideoFormatConverter": VideoFormatConverter}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoFormatConverter": "Video Format Converter"}
