"""Advanced transcode engine. Existing video nodes keep their original interfaces."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from functools import lru_cache

from . import video_utils as media


VIDEO_ENCODERS = ['auto_h264', 'auto_hevc', 'auto_av1', 'copy', 'libx264', 'libx265',
                  'libsvtav1', 'libaom-av1', 'h264_nvenc', 'hevc_nvenc', 'av1_nvenc',
                  'h264_qsv', 'hevc_qsv', 'av1_qsv', 'h264_amf', 'hevc_amf', 'av1_amf',
                  'libvpx-vp9', 'libvpx', 'prores_ks', 'ffv1', 'mpeg4', 'mpeg2video',
                  'wmv2', 'libtheora', 'gif', 'none']
AUDIO_ENCODERS = ['auto', 'copy', 'none', 'aac', 'libmp3lame', 'libopus', 'libvorbis',
                  'flac', 'alac', 'pcm_s16le', 'pcm_s24le', 'ac3', 'eac3']
CONTAINERS = ['mp4', 'mkv', 'mov', 'avi', 'webm', 'ts', 'mpeg', 'flv', 'ogv', 'gif',
              'hls', 'dash', 'mp3', 'm4a', 'aac', 'wav', 'flac', 'opus', 'ogg']
AUDIO_FORMATS = {'mp3', 'm4a', 'aac', 'wav', 'flac', 'opus', 'ogg'}
DEFAULTS = dict(container='mp4', video_encoder='libx264', rate_control='quality', quality=23,
                video_bitrate_kbps=4000, encoding_speed='balanced', pixel_format='auto',
                profile='', width=0, height=0, resize_mode='fit', scaling_filter='lanczos',
                frame_rate=0.0, rotation='none', flip='none', crop_left=0, crop_top=0,
                crop_right=0, crop_bottom=0, deinterlace=False, denoise=0.0, sharpen=0.0,
                brightness=0.0, contrast=1.0, saturation=1.0, start_seconds=0.0,
                duration_seconds=0.0, playback_speed=1.0, video_track=0, metadata='keep',
                audio_encoder='auto', audio_tracks='first', audio_bitrate_kbps=192,
                sample_rate=0, channels='original', volume_db=0.0, normalize_audio=False,
                subtitle_mode='none', subtitle_track=0, subtitle_file='', subtitle_language='und',
                extra_options_json='{}')
PRESETS = {
    'Custom': {},
    'MP4 H.264 - compatible': {},
    'MP4 H.264 - GPU auto': {'video_encoder': 'auto_h264'},
    'HEVC - smaller file': {'video_encoder': 'libx265', 'quality': 26},
    'AV1 - smaller file': {'video_encoder': 'libsvtav1', 'quality': 30},
    'WebM VP9': {'container': 'webm', 'video_encoder': 'libvpx-vp9', 'quality': 30, 'audio_encoder': 'libopus'},
    '1080p H.264': {'width': 1920, 'height': 1080, 'resize_mode': 'pad'},
    '720p H.264': {'width': 1280, 'height': 720, 'resize_mode': 'pad'},
    'Vertical 1080x1920': {'width': 1080, 'height': 1920, 'resize_mode': 'crop'},
    'ProRes editing': {'container': 'mov', 'video_encoder': 'prores_ks', 'profile': '3', 'audio_encoder': 'pcm_s24le'},
    'Lossless archive': {'container': 'mkv', 'video_encoder': 'ffv1', 'audio_encoder': 'flac'},
    'Remux MKV - no re-encode': {'container': 'mkv', 'video_encoder': 'copy', 'audio_encoder': 'copy', 'audio_tracks': 'all', 'subtitle_mode': 'copy'},
    'Extract MP3': {'container': 'mp3', 'video_encoder': 'none', 'audio_encoder': 'libmp3lame'},
    'Extract FLAC': {'container': 'flac', 'video_encoder': 'none', 'audio_encoder': 'flac'},
    'GIF 480px': {'container': 'gif', 'video_encoder': 'gif', 'width': 480, 'frame_rate': 12, 'audio_encoder': 'none'},
}


def inspect_file(path):
    return json.loads(media.run_process([media.executable('ffprobe'), '-v', 'error', '-show_streams',
                                        '-show_format', '-of', 'json', str(path)], timeout=30))


@lru_cache(maxsize=4)
def compiled_encoders(binary):
    result = media.run_process([binary, '-hide_banner', '-encoders'], timeout=20)
    return set(re.findall(r'^\s*[VAS][A-Z.]{5}\s+(\S+)', result, re.MULTILINE))


@lru_cache(maxsize=32)
def hardware_works(binary, encoder):
    if encoder not in compiled_encoders(binary):
        return False
    try:
        media.run_process([binary, '-v', 'error', '-f', 'lavfi', '-i', 'color=size=256x256:rate=30',
                           '-frames:v', '1', '-c:v', encoder, '-f', 'null', '-'], timeout=12)
        return True
    except RuntimeError:
        return False


def capabilities(refresh=False):
    if refresh:
        compiled_encoders.cache_clear()
        hardware_works.cache_clear()
    binary = media.executable('ffmpeg')
    available = compiled_encoders(binary)
    hardware = {e: hardware_works(binary, e) for e in VIDEO_ENCODERS
                if e.endswith(('_nvenc', '_qsv', '_amf'))}
    return {'ffmpeg': binary, 'compiled_video_encoders': [e for e in VIDEO_ENCODERS if e in available],
            'compiled_audio_encoders': [e for e in AUDIO_ENCODERS if e in available],
            'hardware_smoke_test': hardware,
            'note': 'A successful 8-bit test does not guarantee every profile, resolution or pixel format.'}


def resolve_encoder(name, binary):
    if name.startswith('auto_'):
        family = name[5:]
        for candidate in (family + '_nvenc', family + '_qsv', family + '_amf'):
            if hardware_works(binary, candidate):
                return candidate
        choices = {'h264': ['libx264'], 'hevc': ['libx265'], 'av1': ['libsvtav1', 'libaom-av1']}[family]
        name = next((e for e in choices if e in compiled_encoders(binary)), choices[0])
    if name not in {'copy', 'none'} and name not in compiled_encoders(binary):
        raise ValueError(f'FFmpeg does not include encoder {name}. Check Media Info / encoder capabilities.')
    return name


def settings(preset='Custom', custom=None, preset_json=''):
    if preset not in PRESETS:
        raise ValueError('Unknown preset.')
    result = {**DEFAULTS, **PRESETS[preset]}
    for supplied in (custom or {}, json.loads(preset_json) if preset_json.strip() else {}):
        if not isinstance(supplied, dict):
            raise ValueError('Preset must be a JSON object.')
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown preset settings: {sorted(unknown)}')
        result.update(supplied)
    validate(result)
    return result


def validate(s):
    enums = {'container': CONTAINERS, 'video_encoder': VIDEO_ENCODERS, 'audio_encoder': AUDIO_ENCODERS,
             'rate_control': ['quality', 'vbr', 'cbr', 'two_pass'], 'encoding_speed': ['fast', 'balanced', 'slow'],
             'pixel_format': ['auto', 'yuv420p', 'yuv420p10le', 'yuv422p10le', 'yuv444p', 'yuv444p10le'],
             'resize_mode': ['fit', 'pad', 'crop', 'stretch'], 'scaling_filter': ['lanczos', 'bicubic', 'bilinear', 'neighbor'],
             'rotation': ['none', '90', '180', '270'], 'flip': ['none', 'horizontal', 'vertical', 'both'],
             'channels': ['original', 'mono', 'stereo', '5.1'], 'metadata': ['keep', 'strip'],
             'subtitle_mode': ['none', 'copy', 'embed', 'burn']}
    for key, choices in enums.items():
        if s[key] not in choices:
            raise ValueError(f'Invalid {key}: {s[key]}')
    for key, default in DEFAULTS.items():
        value = s[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean.')
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number.')
            if isinstance(default, int) and int(value) != value:
                raise ValueError(f'{key} must be an integer.')
            if isinstance(default, int):
                s[key] = int(value)
        elif not isinstance(value, str):
            raise ValueError(f'{key} must be text.')
    for key in ['width', 'height', 'crop_left', 'crop_top', 'crop_right', 'crop_bottom',
                'start_seconds', 'duration_seconds', 'video_track', 'subtitle_track', 'frame_rate', 'sample_rate']:
        if s[key] < 0:
            raise ValueError(f'{key} cannot be negative.')
    if not .25 <= s['playback_speed'] <= 4:
        raise ValueError('Playback speed must be 0.25–4.')
    if not 0 <= s['quality'] <= 63 or s['video_bitrate_kbps'] <= 0 or s['audio_bitrate_kbps'] <= 0:
        raise ValueError('Quality must be 0–63; bitrates must be positive.')
    if s['subtitle_mode'] in {'copy', 'embed'} and s['playback_speed'] != 1:
        raise ValueError('For speed changes, burn subtitles or disable them to keep timing correct.')
    extra_options(s['extra_options_json'])


def extra_options(value):
    # Codec tuning only: no extra inputs, outputs, maps, paths or filter graphs.
    allowed = {'-g', '-bf', '-refs', '-threads', '-tune', '-level:v', '-max_muxing_queue_size',
               '-compression_level', '-qmin', '-qmax', '-aq-mode', '-aq-strength'}
    obj = json.loads(value)
    if not isinstance(obj, dict) or set(obj) - allowed:
        raise ValueError(f'extra_options_json must contain only: {sorted(allowed)}')
    args = []
    for key, item in obj.items():
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise ValueError('Extra option values must be strings or numbers.')
        args += [key, str(item)]
    return args


def video_options(s, encoder):
    if encoder == 'copy':
        return ['-c:v', 'copy']
    mode, q, speed = s['rate_control'], str(s['quality']), s['encoding_speed']
    bitrate = str(int(s['video_bitrate_kbps'])) + 'k'
    args = ['-c:v', encoder]
    fixed = encoder in {'prores_ks', 'ffv1', 'gif'}
    if fixed and mode != 'quality':
        raise ValueError(f'{encoder} uses its native quality/profile settings; choose quality mode.')
    if mode == 'two_pass' and encoder not in {'libx264', 'libvpx-vp9', 'mpeg4', 'mpeg2video'}:
        raise ValueError('Two-pass is supported for libx264, libvpx-vp9, mpeg4 and mpeg2video.')
    if not fixed:
        if mode == 'quality':
            if encoder.endswith('_nvenc'):
                args += ['-rc', 'vbr', '-cq', q, '-b:v', '0']
            elif encoder.endswith('_qsv'):
                args += ['-global_quality', q]
            elif encoder.endswith('_amf'):
                args += ['-rc', 'cqp', '-qp_i', q, '-qp_p', q]
            elif encoder in {'libx264', 'libx265', 'libsvtav1', 'libaom-av1', 'libvpx-vp9', 'libvpx'}:
                args += ['-crf', q]
                if encoder in {'libaom-av1', 'libvpx-vp9'}:
                    args += ['-b:v', '0']
                elif encoder == 'libvpx':
                    args += ['-b:v', bitrate]
            else:
                args += ['-q:v', str(max(1, min(31, round(s['quality'] / 2))))]
        else:
            args += ['-b:v', bitrate]
            if mode == 'cbr':
                args += ['-minrate', bitrate, '-maxrate', bitrate, '-bufsize', str(int(s['video_bitrate_kbps'] * 2)) + 'k']
                if encoder.endswith('_nvenc'):
                    args += ['-rc', 'cbr']
                elif encoder.endswith('_amf'):
                    args += ['-rc', 'cbr']
                elif encoder == 'libx264':
                    args += ['-x264-params', 'nal-hrd=cbr:filler=1']
            elif encoder.endswith('_nvenc'):
                args += ['-rc', 'vbr']
            elif encoder.endswith('_amf'):
                args += ['-rc', 'vbr_peak', '-maxrate', bitrate]
    if encoder in {'libx264', 'libx265'}:
        args += ['-preset', {'fast': 'veryfast', 'balanced': 'medium', 'slow': 'slow'}[speed]]
    elif encoder == 'libsvtav1':
        args += ['-preset', {'fast': '10', 'balanced': '8', 'slow': '5'}[speed]]
    elif encoder in {'libaom-av1', 'libvpx', 'libvpx-vp9'}:
        args += ['-cpu-used', {'fast': '6', 'balanced': '4', 'slow': '2'}[speed]]
    elif encoder.endswith('_nvenc'):
        args += ['-preset', {'fast': 'p1', 'balanced': 'p4', 'slow': 'p7'}[speed]]
    elif encoder.endswith('_qsv'):
        args += ['-preset', {'fast': 'veryfast', 'balanced': 'medium', 'slow': 'veryslow'}[speed]]
    elif encoder.endswith('_amf'):
        args += ['-quality', {'fast': 'speed', 'balanced': 'balanced', 'slow': 'quality'}[speed]]
    if s['profile']:
        args += ['-profile:v', s['profile']]
    pixel = s['pixel_format']
    if pixel == 'auto':
        pixel = 'yuv422p10le' if encoder == 'prores_ks' else 'yuv420p'
    if encoder != 'gif' and not (encoder == 'ffv1' and s['pixel_format'] == 'auto'):
        args += ['-pix_fmt', pixel]
    return args


def video_filters(s, stream, subtitle_name):
    filters = []
    if s['deinterlace']:
        filters += ['bwdif=mode=send_frame:parity=auto:deint=all']
    left, top, right, bottom = [int(s[k]) for k in ('crop_left', 'crop_top', 'crop_right', 'crop_bottom')]
    if any((left, top, right, bottom)):
        if left + right >= stream['width'] or top + bottom >= stream['height']:
            raise ValueError('Crop margins remove the entire image.')
        filters += [f'crop=iw-{left+right}:ih-{top+bottom}:{left}:{top}']
    if s['rotation'] != 'none':
        filters += {'90': ['transpose=1'], '180': ['hflip', 'vflip'], '270': ['transpose=2']}[s['rotation']]
    if s['flip'] in {'horizontal', 'both'}:
        filters += ['hflip']
    if s['flip'] in {'vertical', 'both'}:
        filters += ['vflip']
    if s['denoise']:
        filters += [f'hqdn3d={s["denoise"]}']
    if s['sharpen']:
        filters += [f'unsharp=5:5:{s["sharpen"]}:5:5:0']
    if (s['brightness'], s['contrast'], s['saturation']) != (0, 1, 1):
        filters += [f'eq=brightness={s["brightness"]}:contrast={s["contrast"]}:saturation={s["saturation"]}']
    w, h = int(s['width']), int(s['height'])
    if w or h:
        if (w and w % 2) or (h and h % 2):
            raise ValueError('Use even target width and height for video encoding.')
        scale = f'scale={w or -2}:{h or -2}:flags={s["scaling_filter"]}'
        if w and h and s['resize_mode'] != 'stretch':
            increase = s['resize_mode'] == 'crop'
            scale += ':force_original_aspect_ratio=' + ('increase' if increase else 'decrease') + ':force_divisible_by=2'
            if s['resize_mode'] == 'pad':
                scale += f',pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black'
            elif increase:
                scale += f',crop={w}:{h}'
        filters += [scale, 'setsar=1']
    if subtitle_name:
        # Work in a temporary cwd so Unicode/Windows paths need no filter escaping.
        filters += [f'setpts=PTS+{s["start_seconds"]}/TB', f'subtitles=filename={subtitle_name}', 'setpts=PTS-STARTPTS']
    if s['playback_speed'] != 1:
        filters += [f'setpts=(PTS-STARTPTS)/{s["playback_speed"]}']
    if s['frame_rate']:
        filters += [f'fps={s["frame_rate"]}']
    return filters


def audio_options(s, streams, container):
    encoder = s['audio_encoder']
    if encoder == 'none' or container == 'gif':
        return ['-an'], None
    track = s['audio_tracks'].strip()
    if track == 'first':
        indices = [0] if streams else []
    elif track == 'all':
        indices = list(range(len(streams)))
    else:
        try:
            indices = list(dict.fromkeys(int(x.strip()) for x in track.split(',')))
        except ValueError as exc:
            raise ValueError('audio_tracks must be first, all, or zero-based indices such as 0,2.') from exc
        if any(i < 0 or i >= len(streams) for i in indices):
            raise ValueError(f'Audio track not found; source has {len(streams)} audio tracks.')
    if not indices:
        if container in AUDIO_FORMATS:
            raise ValueError('Audio extraction requires an input audio stream.')
        return ['-an'], None
    if container in AUDIO_FORMATS and len(indices) != 1:
        raise ValueError('Select one audio track for audio-only output.')
    if encoder == 'auto':
        encoder = {'webm': 'libopus', 'opus': 'libopus', 'ogg': 'libvorbis', 'ogv': 'libvorbis',
                   'mp3': 'libmp3lame', 'flac': 'flac', 'wav': 'pcm_s16le', 'mpeg': 'ac3'}.get(container, 'aac')
    filters = []
    if s['volume_db']:
        filters += [f'volume={s["volume_db"]}dB']
    if s['normalize_audio']:
        filters += ['loudnorm=I=-16:TP=-1.5:LRA=11']
    speed = s['playback_speed']
    if speed != 1:
        while speed > 2:
            filters += ['atempo=2']
            speed /= 2
        while speed < .5:
            filters += ['atempo=0.5']
            speed /= .5
        filters += [f'atempo={speed}']
    if encoder == 'copy' and (filters or s['sample_rate'] or s['channels'] != 'original'):
        raise ValueError('Audio copy cannot change volume, normalization, speed, sample rate or channels.')
    args = []
    for index in indices:
        args += ['-map', f'0:a:{index}']
    args += ['-c:a', encoder]
    if encoder != 'copy':
        if not encoder.startswith('pcm_') and encoder not in {'flac', 'alac'}:
            args += ['-b:a', str(int(s['audio_bitrate_kbps'])) + 'k']
        if filters:
            args += ['-af', ','.join(filters)]
        if s['sample_rate']:
            args += ['-ar', str(int(s['sample_rate']))]
        if s['channels'] != 'original':
            args += ['-ac', {'mono': '1', 'stereo': '2', '5.1': '6'}[s['channels']]]
    return args, encoder


def transcode(source, s, output_directory='', output_name=''):
    validate(s)
    source = Path(source.strip()).expanduser().resolve()
    if not source.is_file():
        raise ValueError('Input must be an existing media file on the ComfyUI server.')
    binary = media.executable('ffmpeg')
    source_info = inspect_file(source)
    videos = [x for x in source_info['streams'] if x['codec_type'] == 'video' and not x.get('disposition', {}).get('attached_pic')]
    audios = [x for x in source_info['streams'] if x['codec_type'] == 'audio']
    subtitles = [x for x in source_info['streams'] if x['codec_type'] == 'subtitle']
    container = s['container']
    audio_only = container in AUDIO_FORMATS
    encoder = 'none' if audio_only else resolve_encoder(s['video_encoder'], binary)
    if container == 'gif' and encoder != 'gif':
        raise ValueError('GIF output requires the gif encoder (use the GIF preset).')
    if encoder == 'none' and not audio_only:
        raise ValueError('Choose an audio-only container when video_encoder is none.')
    if encoder == 'copy' and s['rate_control'] == 'two_pass':
        raise ValueError('Video copy cannot use two-pass encoding.')
    if not audio_only and s['video_track'] >= len(videos):
        raise ValueError('Selected video track does not exist.')
    if container == 'webm' and encoder not in {'libvpx', 'libvpx-vp9', 'libsvtav1', 'libaom-av1', 'av1_nvenc', 'av1_qsv', 'av1_amf', 'copy'}:
        raise ValueError('WebM requires VP8, VP9 or AV1.')
    if encoder == 'prores_ks' and container not in {'mov', 'mkv'}:
        raise ValueError('ProRes requires MOV or MKV.')
    if encoder == 'ffv1' and container not in {'mkv', 'avi'}:
        raise ValueError('FFV1 requires MKV or AVI.')
    audio_args, audio_encoder = audio_options(s, audios, container)
    if audio_encoder:
        resolve_encoder(audio_encoder, binary)
    if container == 'webm' and audio_encoder not in {None, 'copy', 'libopus', 'libvorbis'}:
        raise ValueError('WebM audio requires Opus or Vorbis.')
    if audio_only and not audio_encoder:
        raise ValueError('Audio-only output cannot disable audio.')
    subtitle_mode = s['subtitle_mode']
    if audio_only and subtitle_mode != 'none':
        raise ValueError('Audio-only output does not support subtitles.')
    if subtitle_mode in {'copy', 'embed'} and container not in {'mkv', 'mp4', 'mov'}:
        raise ValueError('Subtitle muxing supports MKV, MP4 and MOV; use burn for other video containers.')
    if subtitle_mode == 'copy' and s['subtitle_file'].strip():
        raise ValueError('Use embed or burn to read an external subtitle file.')
    external = None
    if subtitle_mode in {'burn', 'embed'} and s['subtitle_file'].strip():
        external = Path(s['subtitle_file']).expanduser().resolve()
        if not external.is_file() or external.suffix.lower() not in {'.srt', '.ass', '.ssa', '.vtt'}:
            raise ValueError('Subtitle file must be an existing SRT, ASS, SSA or VTT file.')
    if subtitle_mode in {'burn', 'embed'} and not external and s['subtitle_track'] >= len(subtitles):
        raise ValueError('Selected subtitle track does not exist.')
    commands = []
    target = None
    with tempfile.TemporaryDirectory(prefix='zzx_transcode_') as scratch:
        subtitle_name = None
        if subtitle_mode == 'burn':
            if external:
                subtitle_name = 'subtitle' + external.suffix.lower()
                shutil.copyfile(external, Path(scratch) / subtitle_name)
            else:
                subtitle_name = 'subtitle.ass'
                extract = [binary, '-v', 'error', '-nostdin', '-i', str(source), '-map', f'0:s:{s["subtitle_track"]}',
                           '-c:s', 'ass', '-y', subtitle_name]
                media.run_process(extract, cwd=scratch)
        prefix = [binary, '-hide_banner', '-loglevel', 'error', '-nostdin']
        if s['start_seconds']:
            prefix += ['-ss', str(s['start_seconds'])]
        prefix += ['-i', str(source)]
        if subtitle_mode == 'embed' and external:
            if s['start_seconds']:
                prefix += ['-ss', str(s['start_seconds'])]
            prefix += ['-i', str(external)]
        video_args = []
        if not audio_only:
            stream = videos[int(s['video_track'])]
            filters = video_filters(s, stream, subtitle_name)
            if encoder == 'copy' and (filters or s['pixel_format'] != 'auto' or s['profile']):
                raise ValueError('Video copy cannot apply picture filters, frame rate, speed, profile or pixel format. Choose an encoder.')
            video_args = ['-map', f'0:{stream["index"]}'] + video_options(s, encoder)
            if encoder != 'copy':
                # Even dimensions also handle odd-sized source media for 4:2:0 encoders.
                if encoder not in {'ffv1', 'gif'}:
                    filters += ['pad=ceil(iw/2)*2:ceil(ih/2)*2']
                if encoder == 'gif':
                    filters += ['split[a][b];[a]palettegen[p];[b][p]paletteuse']
                if filters:
                    video_args += ['-vf', ','.join(filters)]
        else:
            video_args += ['-vn']
        output_args = []
        if subtitle_mode == 'copy':
            output_args += ['-map', '0:s?', '-c:s', 'copy']
        elif subtitle_mode == 'embed':
            output_args += ['-map', '1:s:0' if external else f'0:s:{s["subtitle_track"]}',
                            '-c:s', 'mov_text' if container in {'mp4', 'mov'} else 'srt',
                            '-metadata:s:s:0', 'language=' + s['subtitle_language']]
        else:
            output_args += ['-sn']
        output_args += ['-map_metadata', '0' if s['metadata'] == 'keep' else '-1',
                        '-map_chapters', '0' if s['metadata'] == 'keep' and not s['start_seconds'] and not s['duration_seconds'] and s['playback_speed'] == 1 else '-1']
        if s['metadata'] == 'strip':
            output_args += ['-map_metadata:s', '-1']
        if s['duration_seconds']:
            output_args += ['-t', str(s['duration_seconds'] / s['playback_speed'])]
        output_args += extra_options(s['extra_options_json'])
        # All presets are validated before reserving a target filename.
        label = output_name.strip() or source.stem + '_converted'
        extension = '.' + media.EXTENSIONS.get(container, container)
        if not label.lower().endswith(extension):
            label += extension
        target = media.output_file(output_directory, label, container, source.stem + '_converted')
        try:
            mux = media.mux_options(container, target)
            if container == 'ts':
                mux += ['-f', 'mpegts']
            if s['rate_control'] == 'two_pass' and not audio_only:
                passlog = str(Path(scratch) / 'pass')
                first = prefix + video_args + ['-an', '-sn', '-pass', '1', '-passlogfile', passlog]
                if s['duration_seconds']:
                    first += ['-t', str(s['duration_seconds'] / s['playback_speed'])]
                first += ['-f', 'null', '-']
                commands.append(first)
                media.run_process(first, cwd=scratch)
                video_args += ['-pass', '2', '-passlogfile', passlog]
            command = prefix + video_args + audio_args + output_args + mux + ['-y', target.as_posix()]
            commands.append(command)
            media.run_process(command, cwd=scratch)
            output_info = inspect_file(target)
            info = {}
            if not audio_only:
                before, after = media.probe(source, str(stream['index'])), media.probe(target)
                info = {**{'source_' + k: v for k, v in before.items()}, **{'loaded_' + k: v for k, v in after.items()}}
        except BaseException:
            media.remove_failed_output(target, container)
            raise
    report = {'source': str(source), 'output': str(target), 'video_encoder': encoder,
              'audio_encoder': audio_encoder, 'settings': s, 'commands': commands,
              'source_info': source_info, 'output_info': output_info}
    return str(target), info, report
