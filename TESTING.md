# Paints-Undo validation — 2026-10-23

## Actual model inference

Hardware: Windows, NVIDIA RTX 5000 Ada (32 GB class).

Both environments generated all six requested model frames successfully:

| Environment | Python | PyTorch | Diffusers | Transformers | Result |
| --- | --- | --- | --- | --- | --- |
| A | 3.13.15 | 2.8.0+cu129 | 0.35.1 | 4.56.1 | 8 RGB frames including endpoints |
| B | 3.12.14 | 2.11.0+cu128 | 0.41.0 | 5.20.2 | 8 RGB frames including endpoints |

Input: bundled `Paints-UNDO/imgs/1.jpg`; automatic WD14 MOAT tags;
operations `400,600,800,900,950,999`; seed 12345; 30 sampling steps;
CFG 3; longest side 640. Output: float32 CPU tensor `[8,640,512,3]`,
all finite, range 0–1. Visually inspected both contact sheets: background,
shading and colors diminish, followed by line drawing and fewer strokes.
The last white frame is an explicit generated endpoint, not a model prediction.

Environment A's contact sheet is included in `workflows/PaintsUndo-sequence-example.jpg`.

## Current ComfyUI integration

- ComfyUI **0.42.0**, commit `bc5f880dbdd43e426077f42ccec53bb577ffee9f`.
- Frontend **1.57.1**, environment A above.
- Loaded the extension through ComfyUI's actual custom-node loader.
- `/object_info/ZZX_PaintsUndoSequence` returned the expected schema.
- Submitted `workflows/paints_undo_api.json` through `/prompt`.
- History reported `execution_success`, `completed: true`; SaveImage saved eight images.
- Legacy `ZZX_PaintsUndo` also completed via `/prompt`, saving exactly one 512×640 RGB image (12.65 seconds).
- Workflow duration: **37.83 seconds** on this machine with weights cached.

This is actual GPU inference and server-side workflow execution, not only mocked tests.
The UI workflow JSON is supplied separately from the API JSON.

## Installed ComfyUI Studio verification

The actual Studio installation was also tested on 2026-10-23:
`D:\ComfyUI_Studio\ComfyUI_windows_portable` (not the separate validation checkout).
ComfyUI 0.41.0, Python 3.13.14, PyTorch 2.14.0+cu130,
Diffusers 0.41.0, Transformers 5.17.0.

An initially incomplete local `models/paints_undo/single_frame` directory caused
missing-weight errors. Existing complete model files from the earlier validation
cache were checked against the model repository's SHA256 manifest and copied to
the Studio model directory; no redownload or inference-code change was needed.

- Replayed the user's previously failing single-frame workflow on Studio's
  running server: `execution_success`, 11.13 seconds.
- Ran the installed sequence/video workflow: `execution_success`, 27.57 seconds,
  eight 512×640 PNG frames plus a four-second, 2 FPS MP4 preview.
- No private user input or generated user image is included in this repository.

The plugin resolves models relative to the running ComfyUI installation. Updating
this GitHub repository does not copy weights or dependencies into other installs.

## Regression tests

`python -m unittest discover -s tests -v`: **5 passed** in both environments.
These lightweight tests use fake models to verify invalid operation/input rejection,
lazy node construction, RGB batch layout, chronological operation order, shared
seed behavior, and model offload after failure. They do not evaluate model quality.

## Windows console regression (2026-10-24)

A Studio session with Manager enabled failed in Transformers' `Loading weights`
progress bar: the wrapped stderr stream raised `OSError(22, 'Invalid argument')`.
CLIP loading now disables that progress display temporarily and restores both
Transformers and Hugging Face Hub progress settings in `finally`, including when
loading fails. Weight-loading errors still propagate normally.

The six regression tests cover a stderr stream that raises this exact error,
all four combinations of the two progress settings, and restoration after errors.
The actual CLIP checkpoint (115,972,608 parameters) also loaded successfully
with stderr forced to raise OSError(22). Full real-model inference using Studio
Python and Studio weights produced eight finite RGB frames `[8,640,512,3]`.
This validation ran in a fresh process; the existing UI server was not restarted.

Restart ComfyUI after updating the node: an already-running server retains the
old imported Python code. If other nodes also report console/logging errors,
restart from the Studio launcher with a valid console or log-file redirection.

## Limits

CUDA was tested. CPU fallback exists but full CPU inference was not benchmarked.
AMD, Apple MPS, Linux and macOS were not tested. Other dependency combinations
inside the declared ranges are not individually certified. The model simulates
drawing history and may redraw details; strict pixelwise monotonic removal is
not guaranteed. Multi-frame video interpolation remains in the upstream reference
source and is not part of this lightweight ComfyUI node.


# Video utilities v1.2 validation — 2026-10-23

Windows, Python 3.12.14, FFmpeg 2026-02-23 (Gyan full build).
`python -m unittest discover -s tests -v`: **16 passed**, including the five
existing Paints-Undo regressions. Video-only tests need no model dependencies:
`python -m unittest discover -s tests -p test_video.py -v`.

Real media tests generate a one-second H.264/AAC test clip and verify:

- MP4 conversion with crop/fill, copied audio, AAC and MP3; original channel mode.
- Silent input; output dimensions, frame count, frame rate and VHS metadata.
- Recording to MP4, MOV, MKV, AVI, FLV, MPEG-TS, HLS, DASH and AV1/Opus WebM.
- Recording an HLS playlist served by a local HTTP server into MP4.
- Repeated HLS output preserves existing segment contents.
- Missing input removes the reserved incomplete recording.

Contract tests cover disabled output tuple length, valid UI defaults, legacy
format aliases, invalid formats/durations/schedules, unique output filenames,
missing executable diagnostics, unknown probe rates and cancellation of a real
sleeping child process (child termination confirmed).

Additional actual hardware checks passed: HEVC and AV1 software conversion,
H.264 and HEVC NVENC conversion (each returned 24 frames at 320×320), and CUDA
decode + H.264 NVENC recording on the available NVIDIA GPU.

A standalone Linux CI job now runs the video tests using system FFmpeg. Its
remote result is separate from the local Windows results above. Public RTMP/RTSP endpoints, long recordings and all GPU/driver combinations
were not tested. HLS and DASH playback were checked with ffprobe, not a browser player.


## ComfyUI Studio live API integration

Validated in the user's installed portable environment on 2026-10-23:
ComfyUI 0.41.0, frontend 1.55.17, embedded Python 3.13.14,
PyTorch 2.14.0+cu130, NVIDIA RTX 5000 Ada. The installed ZZX extension was
updated from 1.1.0 to 1.2.0 after backing up changed files.

All 16 regressions passed under the embedded Python. The test imports were
fixed to use the extension's package name, avoiding a collision with ComfyUI's
own `nodes.py` in portable Python's search path.

An independent localhost server on port 8189 used the installed ComfyUI and
Python, loading ZZX and VideoHelperSuite with assets disabled to avoid the
active server's database lock. Actual `/prompt` and `/history` executions verified:

- CPU conversion and re-queuing the same graph creates separate outputs.
- CUDA/NVENC recording → HEVC NVENC conversion → VHS video loading → PreviewImage.
- Converter `VHS_VIDEOINFO` → VHS_VideoInfo → EmptyImage → PreviewImage.
- HLS, DASH, WebM and HTTP HLS recording; produced files verified with ffprobe.
- Missing input reports an execution error and leaves no partial output.
- Interrupting a scheduled recording exits without creating a file.
- Interrupting active FFmpeg waiting on UDP reports `execution_interrupted`,
  terminates the observed child PID, and removes partial output.

All **11 API cases** met their expected success/error/interruption outcomes.
The pre-existing 8188 service was left running; it needs a restart to load the
updated modules. This checks server execution and preview generation, not manual
browser interaction or compatibility with every other installed extension.


# Advanced transcode v1.3 — 2026-10-24

Validated in `D:/ComfyUI_Studio/ComfyUI_windows_portable` using ComfyUI 0.41.0,
frontend 1.55.17, embedded Python 3.13.14, PyTorch 2.14.0+cu130 and RTX 5000 Ada.
Existing installed files were backed up before updating. The existing server
on port 8188 was not stopped. A dedicated server on 8189 loaded ZZX and VHS;
assets were disabled only on that test server to avoid the shared database lock.

## Regression and real media

**33 tests passed** under the installed embedded Python: 17 advanced-transcode,
11 original video and 5 Paints-Undo regression tests. New tests cover:

- Preset precedence, JSON validation, safe extra encoding options, preset save/load.
- Legacy input/output contracts; automatic encoder fallback.
- Crop/fit/pad/stretch, rotation, trim, synchronized speed and picture filters.
- Multiple audio tracks, extraction, loudness, channel count and sample rate.
- Embedded/external subtitle muxing and burn-in with spaces, Unicode and apostrophes in paths.
- VBR, CBR and two-pass output; ProRes, FFV1, MPEG-4, MPEG-2, VP9, Theora,
  GIF, MPEG-TS, HLS and DASH, verified by ffprobe.
- Batch success/failure reporting and preserved subfolders, metadata stripping,
  dotted filenames and failure cleanup.

## Live queue and frontend

`scripts/test_transcode_api.py` completed **21 API scenarios** through actual
`/prompt` and `/history`: capabilities, H.264, automatic H.264/AV1, HEVC NVENC
10-bit, SVT-AV1, VP9 WebM, ProRes, FFV1, remux, GIF, FLAC extraction, two-pass,
subtitle embedding, settings → subtitle burn-in → VHS preview, legacy → advanced
conversion, preset save/load, partial batch failure, expected invalid-copy error,
and active-encoding cancellation. Cancellation confirmed the child FFmpeg PID
exited and the partial output was deleted.

The browser also imported the shipped advanced workflow and executed it through
Run. Confirmed the advanced input toggle and the populated result textarea in
frontend 1.55.17. Fixed a custom DOM widget incompatibility by using the standard
Comfy multiline string widget. A decoded subtitle frame was visually inspected:
text is visible inside the padded 320×320 image after seek and speed change.

Hardware smoke tests successfully encoded H.264, HEVC and AV1 with NVENC. QSV and
AMF are compiled into this FFmpeg but are unavailable on this machine; no Intel
or AMD hardware compatibility claim is made. The detector uses a 256×256 frame
because the tested NVENC driver rejects smaller dimensions.

The API harness requires a dedicated idle server, VHS and the tested NVIDIA
encoders. It writes real outputs and issues `/interrupt`; do not point it at a
server running other jobs. Example:

```sh
python scripts/test_transcode_api.py --url http://127.0.0.1:8189 --output /path/to/test-output
```

Linux CI includes both video test modules; the results above are Windows results.
Not every codec/container/profile combination is tested. HDR tone mapping,
bitmap subtitle OCR, custom subtitle attachment fonts, long-duration workloads
and exact CFR/CBR behavior across all encoders are outside this validation.
