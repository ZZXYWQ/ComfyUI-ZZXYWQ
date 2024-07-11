import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog
import subprocess
import cv2

class StreamRecorder:

    def __init__(self):
        self.input_url = ""
        self.output_path = ""

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
                "record_minutes": ("INT", {"default": 0, "min": 0, "max": 59, "step": 1, "display": "slider"}),
                "record_seconds": ("INT", {"default": 60, "min": 0, "max": 59, "step": 1, "display": "slider"}),
                "output_filename": ("STRING", {"multiline": False, "default": ""}),
                "video_format": (["avi", "mov", "mkv", "mp4", "hls", "dash", "mss", "srt", "flv", "webm", "rtmp", "rtsp", "m3u8", "http", "https"], {"default": "mp4"}),
                "codec": (["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"], {"default": "h264(NVENC)"}),
                "video_quality": ("INT", {"default": 10, "min": 5, "max": 40, "step": 1, "display": "slider"}),
                "use_cuda": (["true", "false"], {"default": "true"}),  # Add CUDA option
                "output_path": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_filename",)
    FUNCTION = "record_stream"

    CATEGORY = "ZZX/Stream"

    def select_output_file(self):
        root = tk.Tk()
        root.withdraw()
        self.output_path = filedialog.asksaveasfilename(title="Select output video file")
        return self.output_path

    def get_unique_filename(self, output_path, output_filename, video_format):
        base_name, ext = os.path.splitext(output_filename)
        counter = 0
        while True:
            new_filename = f"{base_name}_{counter:04d}.{video_format}"
            full_path = os.path.join(output_path, new_filename)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    def calculate_bitrate(self, video_quality):
        min_quality = 1
        max_quality = 40
        min_bitrate = 100  # kbps for quality=40
        max_bitrate = 10000  # kbps for quality=1
        return int((max_bitrate - min_bitrate) / (max_quality - min_quality) * (video_quality - min_quality) + min_bitrate)

    def check_format_support(self, video_format, codec):
        format_support = {
            "avi": ["h264", "h264(NVENC)"],
            "mov": ["h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "mkv": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "mp4": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "hls": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "dash": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "mss": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "srt": ["h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "flv": ["h264", "h264(NVENC)"],
            "webm": ["av1", "h264", "h264(NVENC)"],
            "rtmp": ["h264", "h264(NVENC)"],
            "rtsp": ["h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "m3u8": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "http": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"],
            "https": ["av1", "h264", "h264(NVENC)", "hevc", "hevc(NVENC)"]
        }

        if codec not in format_support.get(video_format, []):
            raise ValueError(
                "请选择正确编码模式:\n"
                "AVI (avi): h264\n"
                "MOV (mov): h264, hevc\n"
                "MKV (mkv): av1, h264, hevc\n"
                "MP4 (mp4): av1, h264, hevc\n"
                "HLS (hls): av1, h264, hevc\n"
                "DASH (dash): av1, h264, hevc\n"
                "MSS (mss): av1, h264, hevc\n"
                "SRT (srt): h264, hevc\n"
                "FLV (flv): h264\n"
                "WebM (webm): av1, h264\n"
                "RTMP (rtmp): h264\n"
                "RTSP (rtsp): h264, hevc\n"
                "M3U8 (m3u8): av1, h264, hevc\n"
                "HTTP (http): av1, h264, hevc\n"
                "HTTPS (https): av1, h264, hevc"
            )

    def record_stream(self, stream_url, use_local_time, start_time, record_hours, record_minutes, record_seconds, output_filename, video_format, codec, video_quality, use_cuda, output_path):
        self.check_format_support(video_format, codec)

        if not stream_url:
            raise ValueError("Stream URL is required")

        if not output_path:
            output_path = self.select_output_file()

        # Ensure the output path has the correct file extension
        if not output_path.endswith("/"):
            output_path += "/"

        # Ensure the output filename has the correct format extension
        if not output_filename.endswith(f".{video_format}"):
            output_filename += f".{video_format}"

        # Get a unique filename
        output_full_path = self.get_unique_filename(output_path, output_filename, video_format)

        # Calculate bitrate based on video_quality
        bitrate = self.calculate_bitrate(video_quality)

        # Map codec to correct FFmpeg encoder names
        codec_map = {
            "h264(NVENC)": "h264_nvenc",
            "hevc(NVENC)": "hevc_nvenc",
            "hevc": "libx265",
            "av1": "libaom-av1"
        }
        if codec in codec_map:
            codec = codec_map[codec]

        # Calculate the total record duration in seconds
        total_duration = record_hours * 3600 + record_minutes * 60 + record_seconds

        # Handle local time scheduling
        if use_local_time == "true":
            now = datetime.datetime.now()
            start_time_parts = list(map(int, start_time.split('/')))
            start_dt = now.replace(hour=start_time_parts[0], minute=start_time_parts[1], second=start_time_parts[2], microsecond=0)
            
            if start_dt < now:
                raise ValueError("输入有误，请重新输入时间")

            while datetime.datetime.now() < start_dt:
                time.sleep(1)

        # Construct the FFmpeg command for recording the stream
        cmd = ["ffmpeg"]
        
        if use_cuda == "true":
            cmd.extend(["-hwaccel", "cuda"])

        cmd.extend([
            "-i", stream_url,
            "-t", str(int(total_duration)),
            "-c:v", codec,
            "-b:v", f"{bitrate}k",
            "-c:a", "copy",  # Copy the audio codec from the source
            "-y", output_full_path
        ])

        if video_format == "hls":
            cmd.extend(["-f", "hls", "-hls_time", "10", "-hls_playlist_type", "vod", "-hls_segment_filename", os.path.join(output_path, "segment_%03d.ts")])
        elif video_format == "dash":
            cmd.extend(["-f", "dash", "-seg_duration", "10", "-init_seg_name", "init.m4s", "-media_seg_name", "segment_%03d.m4s"])
        elif video_format == "mss":
            cmd.extend(["-f", "hds", "-hls_time", "10", "-hls_playlist_type", "vod", "-hls_segment_filename", os.path.join(output_path, "segment_%03d.f4m")])
        elif video_format == "srt":
            cmd.extend(["-f", "mpegts"])
        elif video_format == "flv":
            cmd.extend(["-f", "flv"])
        elif video_format == "webm":
            cmd.extend(["-f", "webm"])
        elif video_format == "rtmp":
            cmd.extend(["-f", "flv"])  # RTMP streams are usually in FLV format
        elif video_format == "rtsp":
            cmd.extend(["-f", "rtsp"])
        elif video_format == "m3u8":
            cmd.extend(["-f", "hls"])
        elif video_format == "http" or video_format == "https":
            cmd.extend(["-f", "hls"])

        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg command: {' '.join(cmd)}")
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

        return (output_full_path,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "StreamRecorder": StreamRecorder
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamRecorder": "Stream Recorder"
}
