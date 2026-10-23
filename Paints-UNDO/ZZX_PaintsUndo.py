"""ComfyUI adapters for the upstream Paints-Undo single-frame model."""
import torch
from .undo_pipeline import UndoPipeline, parse_operations


class ZZX_PaintsUndo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "Prompt": ("STRING", {"default": "", "multiline": True}),
            "undo_steps": ("INT", {"default": 500, "min": 0, "max": 999}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "process_image"
    CATEGORY = "ZZX/PaintsUndo"

    @torch.inference_mode()
    def process_image(self, image, Prompt, undo_steps, seed):
        return UndoPipeline().run(image, Prompt, [undo_steps], seed)


class ZZX_PaintsUndoSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "operations": ("STRING", {"default": "400,600,800,900,950,999"}),
            "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
            "sampling_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0}),
            "resolution": ("INT", {"default": 640, "min": 256, "max": 1024, "step": 64}),
            "include_original": ("BOOLEAN", {"default": True}),
            "include_blank": ("BOOLEAN", {"default": True}),
            "negative_prompt": ("STRING", {"default": "lowres, bad anatomy, bad hands, cropped, worst quality", "multiline": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "prompt")
    FUNCTION = "generate"
    CATEGORY = "ZZX/PaintsUndo"

    @torch.inference_mode()
    def generate(self, image, prompt, operations, seed, sampling_steps=50,
                 cfg=3.0, resolution=640, include_original=True,
                 include_blank=True, negative_prompt=""):
        return UndoPipeline().run(image, prompt, parse_operations(operations), seed,
                                  sampling_steps, cfg, resolution, include_original,
                                  include_blank, negative_prompt)


NODE_CLASS_MAPPINGS = {
    "ZZX_PaintsUndo": ZZX_PaintsUndo,
    "ZZX_PaintsUndoSequence": ZZX_PaintsUndoSequence,
}
