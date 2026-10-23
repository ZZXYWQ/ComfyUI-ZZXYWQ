"""Run against an idle, dedicated local ComfyUI instance (includes interrupt tests)."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve() / datetime.datetime.now().strftime('advanced_%Y%m%d_%H%M%S')
    out.mkdir(parents=True)

    def api(path, payload=None):
        request = urllib.request.Request(args.url.rstrip('/') + path,
                    data=None if payload is None else json.dumps(payload).encode(),
                    headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
            return json.loads(data) if data else None

    queue = api('/queue')
    if queue['queue_running'] or queue['queue_pending']:
        raise RuntimeError('Use an idle dedicated server; tests issue /interrupt.')
    info = api('/object_info')
    report = {'system': api('/system_stats'), 'cases': [], 'output_directory': str(out)}

    def node(kind, **changes):
        inputs = {}
        for key, spec in info[kind]['input']['required'].items():
            config = spec[1] if len(spec) > 1 else {}
            inputs[key] = config.get('default', spec[0][0] if isinstance(spec[0], list) else '')
        inputs.update(changes)
        return {'class_type': kind, 'inputs': inputs}

    def wait(prompt_id):
        for _ in range(900):
            data = api('/history/' + prompt_id)
            if prompt_id in data:
                return data[prompt_id]
            time.sleep(.2)
        raise TimeoutError(prompt_id)

    def submit(name, graph, expected='success'):
        (out / (name + '.api.json')).write_text(json.dumps(graph, indent=2), encoding='utf-8')
        prompt_id = api('/prompt', {'prompt': graph})['prompt_id']
        history = wait(prompt_id)
        (out / (name + '.history.json')).write_text(json.dumps(history, indent=2), encoding='utf-8')
        assert history['status']['status_str'] == expected, history['status']
        report['cases'].append({'name': name, 'prompt_id': prompt_id, 'status': expected})
        print(name, expected, flush=True)
        return history

    source = out / 'source.mkv'
    subtitle = out / "subtitle ' space.srt"
    subtitle.write_text('1\n00:00:00,000 --> 00:00:01,800\nZZX Advanced Subtitle\n', encoding='utf-8')
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=24',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-f', 'lavfi', '-i',
                    'sine=frequency=880:sample_rate=48000', '-i', str(subtitle), '-map', '0:v',
                    '-map', '1:a', '-map', '2:a', '-map', '3:s', '-t', '2', '-c:v', 'libx264',
                    '-c:a', 'aac', '-c:s', 'srt', str(source)], check=True)

    def converter(name, **changes):
        return node('ZZX_VideoConvertAdvanced', video_path=str(source), output_path=str(out),
                    output_filename=name, **changes)

    history = submit('capabilities', {'1': node('ZZX_MediaInfo', video_path=str(source), test_hardware=True, refresh=True)})
    report['capabilities'] = json.loads(history['outputs']['1']['text'][0])['encoders']
    for name, preset, custom in [
        ('h264', 'MP4 H.264 - compatible', {}),
        ('auto_h264', 'MP4 H.264 - GPU auto', {}),
        ('auto_av1', 'Custom', {'video_encoder': 'auto_av1'}),
        ('hevc_10bit', 'Custom', {'video_encoder': 'hevc_nvenc', 'pixel_format': 'yuv420p10le'}),
        ('svt_av1', 'AV1 - smaller file', {}),
        ('webm', 'WebM VP9', {}),
        ('prores', 'ProRes editing', {}),
        ('lossless', 'Lossless archive', {}),
        ('remux', 'Remux MKV - no re-encode', {}),
        ('gif', 'GIF 480px', {}),
        ('audio', 'Extract FLAC', {'audio_tracks': '1'}),
        ('two_pass', 'Custom', {'rate_control': 'two_pass', 'video_bitrate_kbps': 400}),
        ('embed', 'Custom', {'subtitle_mode': 'embed', 'subtitle_file': str(subtitle)}),
    ]:
        submit(name, {'1': converter(name, preset=preset, preset_json=json.dumps(custom))})

    audio = node('ZZX_AudioSubtitleSettings', subtitle_mode='burn', subtitle_file=str(subtitle),
                 audio_tracks='all', volume_db=-3, normalize_audio=True, sample_rate=48000)
    settings = node('ZZX_VideoSettings', width=320, height=320, resize_mode='pad', start_seconds=.25,
                    duration_seconds=1, playback_speed=1.25, audio_subtitles=['1', 0])
    convert = converter('filters', preset='Custom', settings=['2', 0])
    loader = node('VHS_LoadVideoPath', video=['3', 0], frame_load_cap=1, format='None')
    submit('settings_burn_preview', {'1': audio, '2': settings, '3': convert, '4': loader,
                                    '5': node('PreviewImage', images=['4', 0])})

    legacy = node('VideoFormatConverter', video_path=str(source), output_path=str(out), output_filename='legacy',
                  video_width='320', video_height='320', frame_rate='24')
    from_legacy = converter('from_legacy', preset='Custom')
    from_legacy['inputs']['video_path'] = ['1', 0]
    submit('legacy_to_advanced', {'1': legacy, '2': from_legacy})

    history = submit('save_preset', {'1': node('ZZX_TranscodePreset', mode='save', preset_file=str(out / 'saved.json'),
                                               preset_json='{"width":320,"height":320,"resize_mode":"pad"}')})
    saved = json.loads(history['outputs']['1']['text'][0])['preset_file']
    submit('load_preset', {'1': node('ZZX_TranscodePreset', mode='load', preset_file=saved),
                            '2': converter('loaded', preset='Custom', settings=['1', 0])})
    batch = node('ZZX_VideoBatchConvert', file_list=str(source) + '\n' + str(out / 'missing.mkv'),
                 preset='MP4 H.264 - compatible', output_path=str(out / 'batch'))
    history = submit('batch_partial_failure', {'1': batch})
    summary = json.loads(history['outputs']['1']['text'][0])
    assert (summary['succeeded'], summary['failed']) == (1, 1), summary

    submit('invalid_copy_filters', {'1': converter('invalid', preset='Custom',
           preset_json='{"video_encoder":"copy","width":320}')}, expected='error')
    assert not list(out.glob('invalid_*.mp4'))

    # Run a deliberately expensive software encode, then interrupt our dedicated server.
    import psutil
    def ffmpeg_pids():
        return {p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() in {'ffmpeg', 'ffmpeg.exe'}}
    before = ffmpeg_pids()
    prompt_id = api('/prompt', {'prompt': {'1': converter('cancelled', preset='Custom', preset_json=json.dumps({
        'video_encoder': 'libaom-av1', 'encoding_speed': 'slow', 'width': 1920, 'height': 1080, 'frame_rate': 120}))}})['prompt_id']
    for _ in range(100):
        active = ffmpeg_pids() - before
        if active:
            break
        time.sleep(.1)
    assert active, 'Expected active FFmpeg child'
    time.sleep(.3)
    api('/interrupt', {})
    history = wait(prompt_id)
    assert any(x[0] == 'execution_interrupted' for x in history['status']['messages']), history
    assert not ffmpeg_pids() - before
    assert not list(out.glob('cancelled_*.mp4'))
    report['cases'].append({'name': 'cancel_active_encode', 'status': 'interrupted', 'prompt_id': prompt_id})
    report['files'] = []
    for file in out.iterdir():
        if file.suffix in {'.mp4', '.mkv', '.mov', '.webm', '.gif', '.flac'}:
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(file)],
                                    check=True, capture_output=True, text=True, encoding='utf-8')
            report['files'].append({'path': str(file), 'probe': json.loads(probe.stdout)})
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT:', out / 'report.json', flush=True)


if __name__ == '__main__':
    main()
