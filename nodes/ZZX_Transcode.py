import json
from pathlib import Path

from . import transcode as engine
from . import video_utils as media


def text(default='', multiline=False):
    return ('STRING', {'default': default, 'multiline': multiline})


def integer(default, minimum=0, maximum=8192):
    return ('INT', {'default': default, 'min': minimum, 'max': maximum})


def number(default, minimum=0, maximum=100000, step=.1):
    return ('FLOAT', {'default': default, 'min': minimum, 'max': maximum, 'step': step})


def choice(values, default):
    return (values, {'default': default})


def render_result(values, report):
    return {'ui': {'text': [json.dumps(report, ensure_ascii=False, indent=2)]}, 'result': values}


class ZZX_VideoSettings:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'build'
    RETURN_TYPES = ('ZZX_TRANSCODE_SETTINGS', 'STRING')
    RETURN_NAMES = ('settings', 'preset_json')

    @classmethod
    def INPUT_TYPES(cls):
        schema = {'required': {
            'container': choice(engine.CONTAINERS, 'mp4'),
            'video_encoder': choice(engine.VIDEO_ENCODERS, 'libx264'),
            'rate_control': choice(['quality', 'vbr', 'cbr', 'two_pass'], 'quality'),
            'quality': integer(23, 0, 63), 'video_bitrate_kbps': integer(4000, 1, 1000000),
            'encoding_speed': choice(['fast', 'balanced', 'slow'], 'balanced'),
        }, 'optional': {
            'width': integer(0), 'height': integer(0),
            'resize_mode': choice(['fit', 'pad', 'crop', 'stretch'], 'fit'),
            'scaling_filter': choice(['lanczos', 'bicubic', 'bilinear', 'neighbor'], 'lanczos'),
            'frame_rate': number(0, 0, 240, .001),
            'start_seconds': number(0), 'duration_seconds': number(0),
            'playback_speed': number(1, .25, 4, .05),
            'rotation': choice(['none', '90', '180', '270'], 'none'),
            'flip': choice(['none', 'horizontal', 'vertical', 'both'], 'none'),
            'crop_left': integer(0), 'crop_top': integer(0), 'crop_right': integer(0), 'crop_bottom': integer(0),
            'deinterlace': ('BOOLEAN', {'default': False}),
            'denoise': number(0, 0, 10), 'sharpen': number(0, -2, 5),
            'brightness': number(0, -1, 1, .01), 'contrast': number(1, 0, 3, .01),
            'saturation': number(1, 0, 3, .01),
            'pixel_format': choice(['auto', 'yuv420p', 'yuv420p10le', 'yuv422p10le', 'yuv444p', 'yuv444p10le'], 'auto'),
            'profile': text(), 'video_track': integer(0, 0, 100),
            'metadata': choice(['keep', 'strip'], 'keep'),
            'extra_options_json': text('{}', True),
            'audio_subtitles': ('ZZX_AUDIO_SETTINGS',),
        }}
        common = {'width', 'height', 'resize_mode', 'frame_rate', 'start_seconds', 'duration_seconds', 'playback_speed'}
        for key, spec in schema['optional'].items():
            if key not in common and len(spec) > 1:
                spec[1]['advanced'] = True
        return schema

    def build(self, audio_subtitles=None, **kwargs):
        result = engine.settings(custom={**kwargs, **(audio_subtitles or {})})
        return result, json.dumps(result, ensure_ascii=False, indent=2)


class ZZX_AudioSubtitleSettings:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'build'
    RETURN_TYPES = ('ZZX_AUDIO_SETTINGS',)
    RETURN_NAMES = ('audio_subtitles',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio_encoder': choice(engine.AUDIO_ENCODERS, 'auto'),
            'audio_tracks': text('first'), 'audio_bitrate_kbps': integer(192, 8, 1536),
            'sample_rate': integer(0, 0, 192000),
            'channels': choice(['original', 'mono', 'stereo', '5.1'], 'original'),
            'volume_db': number(0, -60, 30), 'normalize_audio': ('BOOLEAN', {'default': False}),
            'subtitle_mode': choice(['none', 'copy', 'embed', 'burn'], 'none'),
            'subtitle_track': integer(0, 0, 100), 'subtitle_file': text(),
            'subtitle_language': text('und'),
        }}

    def build(self, **kwargs):
        engine.settings(custom=kwargs)
        return (kwargs,)


class ZZX_VideoConvertAdvanced:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'convert'
    RETURN_TYPES = ('STRING', 'VHS_VIDEOINFO', 'STRING')
    RETURN_NAMES = ('output_path', 'video_info', 'report_json')
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video_path': text(), 'preset': choice(list(engine.PRESETS), 'MP4 H.264 - compatible'),
                             'output_path': text(), 'output_filename': text()},
                'optional': {'settings': ('ZZX_TRANSCODE_SETTINGS',), 'preset_json': text('', True)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def convert(self, video_path, preset, output_path, output_filename, settings=None, preset_json=''):
        config = engine.settings(preset, settings, preset_json)
        path, info, report = engine.transcode(video_path, config, output_path, output_filename)
        return render_result((path, info, json.dumps(report, ensure_ascii=False)),
                             {'output': path, 'video_encoder': report['video_encoder'],
                              'audio_encoder': report['audio_encoder'], 'video_info': info})


class ZZX_VideoBatchConvert:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'convert'
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('output_paths', 'report_json')
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'input_directory': text(), 'file_list': text('', True),
            'extensions': text('mp4,mkv,mov,avi,webm,ts,mts,m2ts,m4v,wmv,flv,mpg,mpeg,mp3,wav,flac,m4a,ogg,opus'),
            'recursive': ('BOOLEAN', {'default': False}),
            'preset': choice(list(engine.PRESETS), 'MP4 H.264 - compatible'),
            'output_path': text(), 'filename_suffix': text('_converted'),
            'preserve_subfolders': ('BOOLEAN', {'default': True}),
            'continue_on_error': ('BOOLEAN', {'default': True}), 'max_files': integer(100, 1, 10000),
        }, 'optional': {'settings': ('ZZX_TRANSCODE_SETTINGS',), 'preset_json': text('', True)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def convert(self, input_directory, file_list, extensions, recursive, preset, output_path,
                filename_suffix, preserve_subfolders, continue_on_error, max_files, settings=None, preset_json=''):
        config = engine.settings(preset, settings, preset_json)
        directory = Path(input_directory).expanduser().resolve() if input_directory.strip() else None
        if file_list.strip():
            files = [Path(line.strip().strip('"')).expanduser().resolve() for line in file_list.splitlines() if line.strip()]
        elif directory and directory.is_dir():
            suffixes = {'.' + x.strip().lower().lstrip('.') for x in extensions.split(',') if x.strip()}
            output_root = Path(output_path).expanduser().resolve() if output_path.strip() else None
            files = sorted(p.resolve() for p in (directory.rglob('*') if recursive else directory.iterdir())
                           if p.is_file() and p.suffix.lower() in suffixes
                           and not (output_root and output_root != directory and p.resolve().is_relative_to(output_root)))
        else:
            raise ValueError('Provide an input directory or a newline-separated list of files.')
        files = list(dict.fromkeys(files))
        if not files:
            raise ValueError('No matching input files.')
        if len(files) > int(max_files):
            raise ValueError(f'Found {len(files)} files; increase max_files ({max_files}) to process them.')
        if any(c in filename_suffix for c in '/\\:%'):
            raise ValueError('filename_suffix cannot contain a path or %.')
        reports, outputs = [], []
        for index, source in enumerate(files):
            media.check_interrupt()
            destination = output_path
            if preserve_subfolders and directory and source.is_relative_to(directory) and output_path.strip():
                destination = str(Path(output_path) / source.relative_to(directory).parent)
            try:
                path, _, report = engine.transcode(str(source), config, destination, source.stem + filename_suffix)
                outputs.append(path)
                reports.append({'source': str(source), 'output': path, 'status': 'success',
                                'video_encoder': report['video_encoder']})
            except (ValueError, RuntimeError, OSError) as exc:
                reports.append({'source': str(source), 'status': 'error', 'error': str(exc)})
                if not continue_on_error:
                    raise RuntimeError(f'Batch stopped at {source}. {len(outputs)} earlier outputs were kept. {exc}') from exc
            try:
                from comfy.utils import ProgressBar
            except ImportError:
                pass
            else:
                ProgressBar(len(files)).update_absolute(index + 1)
        summary = {'total': len(files), 'succeeded': len(outputs), 'failed': len(files) - len(outputs), 'files': reports}
        return render_result((outputs, json.dumps(summary, ensure_ascii=False)), summary)


class ZZX_TranscodePreset:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'process'
    RETURN_TYPES = ('ZZX_TRANSCODE_SETTINGS', 'STRING', 'STRING')
    RETURN_NAMES = ('settings', 'preset_json', 'saved_path')
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'mode': choice(['load', 'save'], 'load'), 'preset_file': text(),
                             'preset_json': text('{}', True)},
                'optional': {'settings': ('ZZX_TRANSCODE_SETTINGS',)}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def process(self, mode, preset_file, preset_json, settings=None):
        if mode not in {'load', 'save'} or not preset_file.strip():
            raise ValueError('Select load/save and provide a JSON preset file path.')
        file = Path(preset_file).expanduser().resolve()
        if file.suffix.lower() != '.json':
            raise ValueError('Preset filename must end in .json.')
        if mode == 'load':
            if file.stat().st_size > 1024 * 1024:
                raise ValueError('Preset exceeds 1 MiB.')
            result = engine.settings(preset_json=file.read_text(encoding='utf-8-sig'))
        else:
            result = engine.settings(custom=settings, preset_json=preset_json)
            file = media.output_file(str(file.parent), file.name, 'json', 'preset')
            try:
                file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            except BaseException:
                file.unlink(missing_ok=True)
                raise
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        return render_result((result, encoded, str(file)), {'preset_file': str(file), 'settings': result})


class ZZX_MediaInfo:
    CATEGORY = 'ZZX/Video Advanced'
    FUNCTION = 'inspect'
    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('media_json', 'encoders_json')
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'video_path': text(), 'test_hardware': ('BOOLEAN', {'default': False}),
                             'refresh': ('BOOLEAN', {'default': False})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def inspect(self, video_path, test_hardware, refresh):
        info = {}
        if video_path.strip():
            path = Path(video_path).expanduser().resolve()
            if not path.is_file():
                raise ValueError('Media file does not exist.')
            info = engine.inspect_file(path)
        if test_hardware:
            support = engine.capabilities(refresh)
        else:
            if refresh:
                engine.compiled_encoders.cache_clear()
            support = {'compiled_encoders': sorted(engine.compiled_encoders(media.executable('ffmpeg'))),
                       'note': 'Hardware has not been tested; enable test_hardware for a real one-frame test.'}
        return render_result((json.dumps(info, ensure_ascii=False), json.dumps(support, ensure_ascii=False)),
                             {'media': info, 'encoders': support})


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (ZZX_VideoSettings, ZZX_AudioSubtitleSettings,
    ZZX_VideoConvertAdvanced, ZZX_VideoBatchConvert, ZZX_TranscodePreset, ZZX_MediaInfo)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'ZZX_VideoSettings': 'ZZX Video Settings / 视频设置',
    'ZZX_AudioSubtitleSettings': 'ZZX Audio & Subtitles / 音频字幕',
    'ZZX_VideoConvertAdvanced': 'ZZX Advanced Converter / 高级转码',
    'ZZX_VideoBatchConvert': 'ZZX Batch Converter / 批量转码',
    'ZZX_TranscodePreset': 'ZZX Preset File / 预设文件',
    'ZZX_MediaInfo': 'ZZX Media Info / 媒体与编码器信息',
}
