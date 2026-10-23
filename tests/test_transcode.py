import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent))
engine = importlib.import_module(root.name + '.nodes.transcode')
nodes = importlib.import_module(root.name + '.nodes.ZZX_Transcode')
media = engine.media


class SettingsTests(unittest.TestCase):
    def test_precedence_and_unknown_keys(self):
        value = engine.settings('720p H.264', {'width': 640}, '{"height":360}')
        self.assertEqual((value['width'], value['height']), (640, 360))
        for raw in ('[]', '{"unexpected":1}', '{"width":-2}', '{"playback_speed":0}', '{"width":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                engine.settings(preset_json=raw)

    def test_extra_options_cannot_add_outputs(self):
        for raw in ('{"-i":"input"}', '{"-y":"file"}', '{"-vf":"movie=other"}', '[]'):
            with self.assertRaises(ValueError):
                engine.extra_options(raw)
        self.assertEqual(engine.extra_options('{"-g":48}'), ['-g', '48'])

    def test_audio_copy_rejects_filters(self):
        s = engine.settings(custom={'audio_encoder': 'copy', 'volume_db': 1})
        with self.assertRaisesRegex(ValueError, 'Audio copy'):
            engine.audio_options(s, [{}], 'mkv')

    def test_preset_file_roundtrip_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            name = str(Path(folder) / 'custom.json')
            result = nodes.ZZX_TranscodePreset().process('save', name, '{"quality":19}')['result']
            second = nodes.ZZX_TranscodePreset().process('save', name, '{"quality":29}')['result']
            self.assertNotEqual(result[2], second[2])
            loaded = nodes.ZZX_TranscodePreset().process('load', result[2], '{}')['result']
            self.assertEqual(loaded[0]['quality'], 19)

    def test_legacy_interfaces_unchanged(self):
        old = importlib.import_module(root.name + '.nodes.ZZX_VFC').VideoFormatConverter
        self.assertEqual(old.RETURN_TYPES, ('STRING', 'VHS_VIDEOINFO'))
        self.assertEqual(len(old.INPUT_TYPES()['required']), 17)

    def test_auto_fallback(self):
        with patch.object(engine, 'hardware_works', return_value=False), patch.object(engine, 'compiled_encoders', return_value={'libx264'}):
            self.assertEqual(engine.resolve_encoder('auto_h264', 'ffmpeg'), 'libx264')


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools required')
class TranscodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.source = cls.folder / 'source.mkv'
        cls.subtitle = cls.folder / "字幕 ' test.srt"
        cls.subtitle.write_text('1\n00:00:00,000 --> 00:00:01,800\nZZX Subtitle Test\n', encoding='utf-8')
        media.run_process([media.executable('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x96:rate=24',
                           '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-f', 'lavfi', '-i',
                           'sine=frequency=880:sample_rate=48000', '-i', str(cls.subtitle), '-map', '0:v',
                           '-map', '1:a', '-map', '2:a', '-map', '3:s', '-t', '2', '-c:v', 'libx264',
                           '-c:a', 'aac', '-c:s', 'srt', '-metadata:s:a:0', 'language=eng',
                           '-metadata:s:a:1', 'language=jpn', str(cls.source)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_media(self, preset='Custom', **changes):
        return engine.transcode(str(self.source), engine.settings(preset, changes), str(self.folder), 'converted')

    def test_resize_modes_rotation_crop(self):
        for mode, expected in [('fit', (160, 96)), ('pad', (160, 160)), ('crop', (160, 160)), ('stretch', (160, 160))]:
            with self.subTest(mode=mode):
                _, info, _ = self.run_media(width=160, height=160, resize_mode=mode)
                self.assertEqual((info['loaded_width'], info['loaded_height']), expected)
        _, info, _ = self.run_media(rotation='90', crop_left=8, crop_right=8)
        self.assertEqual((info['loaded_width'], info['loaded_height']), (96, 144))

    def test_trim_speed_filters(self):
        _, info, report = self.run_media(start_seconds=.25, duration_seconds=1, playback_speed=2,
                                        frame_rate=24, flip='both', denoise=1, sharpen=.5,
                                        brightness=.05, saturation=.8, deinterlace=True)
        self.assertLess(info['loaded_duration'], .7)
        self.assertGreater(info['loaded_duration'], .3)
        self.assertTrue(any(s['codec_type'] == 'audio' for s in report['output_info']['streams']))

    def test_remux_multitrack_subtitles(self):
        _, _, report = self.run_media('Remux MKV - no re-encode')
        streams = report['output_info']['streams']
        self.assertEqual([x['codec_name'] for x in streams], ['h264', 'aac', 'aac', 'subrip'])

    def test_audio_extract_and_processing(self):
        for preset, codec in [('Extract MP3', 'mp3'), ('Extract FLAC', 'flac')]:
            _, info, report = self.run_media(preset, audio_tracks='1')
            self.assertEqual(info, {})
            self.assertEqual(report['output_info']['streams'][0]['codec_name'], codec)
        _, _, report = self.run_media(audio_tracks='all', volume_db=-3, normalize_audio=True, channels='mono', sample_rate=48000)
        audios = [x for x in report['output_info']['streams'] if x['codec_type'] == 'audio']
        self.assertEqual(len(audios), 2)
        self.assertTrue(all(x['channels'] == 1 and x['sample_rate'] == '48000' for x in audios))

    def test_subtitle_embed_and_burn(self):
        for external in ('', str(self.subtitle)):
            with self.subTest(external=external):
                _, _, report = self.run_media(subtitle_mode='embed', subtitle_file=external, start_seconds=.25, duration_seconds=1)
                self.assertTrue(any(x['codec_name'] == 'mov_text' for x in report['output_info']['streams']))
                path, _, _ = self.run_media(subtitle_mode='burn', subtitle_file=external, start_seconds=.25, duration_seconds=1)
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_rate_modes(self):
        for mode in ('vbr', 'cbr', 'two_pass'):
            with self.subTest(mode=mode):
                _, info, report = self.run_media(rate_control=mode, video_bitrate_kbps=300)
                self.assertGreater(info['loaded_frame_count'], 0)
                self.assertEqual(len(report['commands']), 2 if mode == 'two_pass' else 1)

    def test_encoders_and_formats(self):
        pairs = [('mov', 'prores_ks'), ('mkv', 'ffv1'), ('avi', 'mpeg4'), ('mpeg', 'mpeg2video'),
                 ('ts', 'libx264'), ('flv', 'libx264'), ('webm', 'libvpx-vp9'), ('ogv', 'libtheora'),
                 ('gif', 'gif'), ('hls', 'libx264'), ('dash', 'libx264')]
        for container, encoder in pairs:
            with self.subTest(container=container):
                if encoder not in engine.compiled_encoders(media.executable('ffmpeg')):
                    continue
                path, info, _ = self.run_media(container=container, video_encoder=encoder, duration_seconds=.5)
                self.assertGreater(Path(path).stat().st_size, 0)
                self.assertEqual(info['loaded_width'], 160)

    def test_copy_cannot_resize(self):
        with self.assertRaisesRegex(ValueError, 'Video copy'):
            self.run_media(video_encoder='copy', width=320)

    def test_dotted_filename_and_metadata_strip(self):
        path, _, report = engine.transcode(str(self.source), engine.settings(custom={'metadata': 'strip'}),
                                          str(self.folder), 'clip.final')
        self.assertTrue(Path(path).name.startswith('clip.final_'))
        for stream in report['output_info']['streams']:
            self.assertNotIn(stream.get('tags', {}).get('language'), {'eng', 'jpn'})

    def test_failed_encoder_removes_output(self):
        dest = self.folder / 'failure'
        with self.assertRaises(RuntimeError):
            engine.transcode(str(self.source), engine.settings(custom={'profile': 'not-a-real-profile'}), str(dest), 'bad')
        self.assertEqual(list(dest.iterdir()), [])

    def test_batch_partial_failure_and_subfolders(self):
        nested = self.folder / 'inputs' / 'sub'
        nested.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, nested / 'good.mkv')
        (nested / 'broken.mkv').write_bytes(b'broken')
        dest = self.folder / 'batch'
        result = nodes.ZZX_VideoBatchConvert().convert(str(nested.parent), '', 'mkv', True, 'Custom',
                  str(dest), '_converted', True, True, 10)['result']
        report = json.loads(result[1])
        self.assertEqual((report['succeeded'], report['failed']), (1, 1))
        self.assertEqual(Path(result[0][0]).parent, dest / 'sub')


if __name__ == '__main__':
    unittest.main()
