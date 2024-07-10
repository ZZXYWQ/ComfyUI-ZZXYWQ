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
                "video_format": (["avi", "mov", "mkv", "mp4"], {"default": "mp4"}),
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

    def record_stream(self, stream_url, use_local_time, start_time, record_hours, record_minutes, record_seconds, output_filename, video_format, codec, video_quality, use_cuda, output_path):
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
