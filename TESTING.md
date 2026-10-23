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

## Regression tests

`python -m unittest discover -s tests -v`: **5 passed** in both environments.
These lightweight tests use fake models to verify invalid operation/input rejection,
lazy node construction, RGB batch layout, chronological operation order, shared
seed behavior, and model offload after failure. They do not evaluate model quality.

## Limits

CUDA was tested. CPU fallback exists but full CPU inference was not benchmarked.
AMD, Apple MPS, Linux and macOS were not tested. Other dependency combinations
inside the declared ranges are not individually certified. The model simulates
drawing history and may redraw details; strict pixelwise monotonic removal is
not guaranteed. Multi-frame video interpolation remains in the upstream reference
source and is not part of this lightweight ComfyUI node.
