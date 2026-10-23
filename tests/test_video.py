import functools
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodes import video_utils as media
from nodes.ZZX_VFC import VideoFormatConverter
from nodes.ZZX_Stream import StreamRecorder


def convert(source, directory, **changes):
    arguments = dict(video_path=str(source), output_enabled="true", output_filename="result",
                     video_format="mp4", codec="h264", video_quality=23, frame_rate="24",
                     opencl_acceleration="disable", video_width="320", video_height="320",
                     scaling_filter="bicubic", processing_method="crop", audio_codec="aac",
                     bit_rate="128", audio_channels="original", sample_rate="48000",
                     output_path=str(directory))
    arguments.update(changes)
    return VideoFormatConverter().process_video(**arguments)


def record(source, directory, **changes):
    arguments = dict(stream_url=str(source), use_local_time="false", start_time="00/00/00",
                     record_hours=0, record_minutes=0, record_seconds=1, output_filename="stream",
                     video_format="mp4", codec="h264", video_quality=25, use_cuda="false",
                     output_path=str(directory))
    arguments.update(changes)
    return StreamRecorder().record_stream(**arguments)[0]


class ContractTests(unittest.TestCase):
    def test_disabled_has_two_outputs_without_io(self):
        with patch.object(media, "probe", side_effect=AssertionError):
            self.assertEqual(convert("", "", output_enabled="false"), ("", {}))

    def test_defaults_and_output_nodes(self):
        for cls in (StreamRecorder, VideoFormatConverter):
            self.assertTrue(cls.OUTPUT_NODE)
            self.assertNotEqual(cls.IS_CHANGED(), cls.IS_CHANGED())
            for kind, *config in cls.INPUT_TYPES()["required"].values():
                if kind == "INT" and config:
                    self.assertLessEqual(config[0]["default"], config[0]["max"])
        self.assertEqual(media.validate_format("m3u8", "h264"), "hls")
        for fmt, codec in [("webm", "h264"), ("mss", "h264"), ("hls", "av1")]:
            with self.assertRaises(ValueError):
                media.validate_format(fmt, codec)

    def test_validation_and_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            first = media.output_file(directory, "test.mp4", "mp4", "unused")
            second = media.output_file(directory, "test.mp4", "mp4", "unused")
            self.assertNotEqual(first, second)
            for name in ("../escape", "sub/file", "%05d"):
                with self.assertRaises(ValueError):
                    media.output_file(directory, name, "mp4", "unused")
        with self.assertRaises(ValueError):
            record("example", "", record_seconds=0)
        with self.assertRaises(ValueError):
            record("example", "", use_local_time="true", start_time="bad")

    def test_missing_binary_message(self):
        with patch.dict(os.environ, {"ZZX_FFMPEG": "nonexistent-zzx-ffmpeg"}):
            with self.assertRaisesRegex(RuntimeError, "ZZX_FFMPEG"):
                media.executable("ffmpeg")

    def test_unknown_probe_values(self):
        data = {"streams": [{"avg_frame_rate": "0/0", "r_frame_rate": "0/0",
                              "duration": "N/A", "width": 20, "height": 10}]}
        with patch.object(media, "executable", return_value="ffprobe"), patch.object(media, "run_process", return_value=json.dumps(data)):
            info = media.probe("unused")
        self.assertEqual(info["fps"], 0)
        self.assertEqual(info["duration"], 0)

    def test_cancellation_reaps_process(self):
        processes = []
        original = subprocess.Popen
        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(media.subprocess, "Popen", side_effect=launch), patch.object(media, "check_interrupt", side_effect=[None, InterruptedError("cancel")]):
            with self.assertRaises(InterruptedError):
                media.run_process([sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertIsNotNone(processes[0].poll())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
class RealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.source = cls.directory / "source video.mp4"
        media.run_process([media.executable("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                           "testsrc2=size=320x180:rate=24", "-f", "lavfi", "-i",
                           "sine=frequency=440:sample_rate=44100", "-t", "1", "-c:v",
                           "libx264", "-c:a", "aac", str(cls.source)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_convert_audio_copy_and_original_channels(self):
        for audio, method in [("copy", "crop"), ("aac", "fill"), ("mp3", "fill")]:
            with self.subTest(audio=audio):
                path, info = convert(self.source, self.directory, audio_codec=audio, processing_method=method)
                self.assertTrue(Path(path).is_file())
                self.assertEqual(info["source_width"], 320)
                self.assertEqual(info["source_height"], 180)
                self.assertEqual(info["loaded_height"], 320)
                self.assertEqual(info["loaded_frame_count"], 24)
                self.assertEqual(info["loaded_fps"], 24)

    def test_no_audio(self):
        silent = self.directory / "silent.mp4"
        media.run_process([media.executable("ffmpeg"), "-v", "error", "-i", str(self.source),
                           "-an", "-c:v", "copy", str(silent)])
        path, info = convert(silent, self.directory)
        self.assertGreater(Path(path).stat().st_size, 0)
        self.assertEqual(info["loaded_frame_count"], 24)

    def test_local_record_formats(self):
        for fmt in ("mp4", "mov", "mkv", "avi", "flv", "srt", "hls", "dash", "webm"):
            with self.subTest(format=fmt):
                path = record(self.source, self.directory, video_format=fmt,
                              codec="av1" if fmt == "webm" else "h264")
                self.assertGreater(media.probe(path)["duration"], 0)
        # A second HLS recording must not overwrite the first one's segments.
        existing = {p: p.read_bytes() for p in self.directory.glob("stream_*.ts")}
        record(self.source, self.directory, video_format="hls")
        for path, content in existing.items():
            self.assertEqual(path.read_bytes(), content)

    def test_http_hls_record(self):
        playlist = record(self.source, self.directory, video_format="hls")
        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        handler = functools.partial(QuietHandler, directory=str(self.directory))
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/{Path(playlist).name}"
                output = record(url, self.directory)
                self.assertGreater(media.probe(output)["duration"], 0)
            finally:
                server.shutdown()
                thread.join()

    def test_failed_record_cleanup(self):
        directory = self.directory / "failure"
        with self.assertRaises(RuntimeError):
            record(self.directory / "missing.mp4", directory)
        self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
