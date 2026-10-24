"""Lazy inference path: no Gradio, video model or import-time GPU allocation."""
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def parse_operations(value):
    try:
        values = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise ValueError("operations must be comma-separated integers, e.g. 400,600,800,999") from exc
    if not values or any(v < 0 or v > 999 for v in values):
        raise ValueError("Each operation must be between 0 and 999.")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("operations must increase strictly toward blank canvas.")
    return values


def model_directory():
    if os.environ.get("PAINTS_UNDO_MODEL_DIR"):
        return Path(os.environ["PAINTS_UNDO_MODEL_DIR"])
    try:
        import folder_paths
        return Path(folder_paths.models_dir) / "paints_undo"
    except ImportError:
        return Path(__file__).parent / "hf_download"


def prepare_image(image, resolution):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError("PaintsUndo expects exactly one RGB image [1,H,W,3].")
    if not torch.isfinite(image).all():
        raise ValueError("Input image contains NaN or infinity.")
    pixels = (image[0].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    pil = Image.fromarray(pixels)
    scale = resolution / max(pil.size)
    size = tuple(max(64, round(v * scale / 64) * 64) for v in pil.size)
    return pil.resize(size, Image.Resampling.LANCZOS)


def load_clip_text_encoder(source, **options):
    from transformers import CLIPTextModel
    from transformers.utils import logging as transformer_logging
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    # Windows background launches can leave Manager's stderr handle invalid.
    # Weight loading must not depend on rendering a console progress bar.
    enabled = transformer_logging.is_progress_bar_enabled()
    hub_disabled = are_progress_bars_disabled()
    transformer_logging.disable_progress_bar()
    try:
        return CLIPTextModel.from_pretrained(source, subfolder="text_encoder", **options)
    finally:
        if enabled:
            transformer_logging.enable_progress_bar()
        if hub_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()


class UndoPipeline:
    def load(self):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor2_0
        from transformers import CLIPTokenizer
        from .diffusers_helper.cat_cond import unet_add_concat_conds
        from .diffusers_helper.code_cond import unet_add_coded_conds
        from .diffusers_helper.k_diffusion import KDiffusionSampler

        class ModifiedUNet(UNet2DConditionModel):
            @classmethod
            def from_config(cls, *args, **kwargs):
                model = super().from_config(*args, **kwargs)
                unet_add_concat_conds(model, new_channels=4)
                unet_add_coded_conds(model, added_number_count=1)
                return model

        directory = model_directory()
        source = str(directory / "single_frame") if (directory / "single_frame").is_dir() else "lllyasviel/paints_undo_single_frame"
        options = {"cache_dir": str(directory)}
        self.tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", **options)
        self.text_encoder = load_clip_text_encoder(source, **options).to(self.dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(source, subfolder="vae", **options).float().eval()
        self.unet = ModifiedUNet.from_pretrained(source, subfolder="unet", low_cpu_mem_usage=False, **options).to(self.dtype).eval()
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())
        self.vae.enable_slicing()
        self.sampler = KDiffusionSampler(self.unet, timesteps=1000,
                                        linear_start=0.00085, linear_end=0.020, linear=True)

    def encode_prompt(self, text):
        ids = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=self.tokenizer.model_max_length,
                             return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids).last_hidden_state

    @torch.inference_mode()
    def run(self, image, prompt, operations, seed, sampling_steps=50, cfg=3.0,
            resolution=640, include_original=False, include_blank=False,
            negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality"):
        operations = parse_operations(",".join(str(v) for v in operations))
        pil = prepare_image(image, resolution)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        original = torch.from_numpy(np.array(pil)).float().unsqueeze(0) / 255
        frames = [original] if include_original else []
        prompt = prompt.strip()
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            check_interrupt = mm.throw_exception_if_processing_interrupted
        except ImportError:
            check_interrupt = lambda: None
        try:
            check_interrupt()
            if not prompt:
                from .wd14tagger import default_interrogator
                prompt = default_interrogator(pil, cache_dir=model_directory() / "tagger")
            self.load()
            self.vae.to(self.device)
            condition = self.vae.encode(original.movedim(-1, 1).to(self.device) * 2 - 1).latent_dist.mode()
            condition = (condition * self.vae.config.scaling_factor).to(self.dtype)
            self.vae.cpu()
            self.text_encoder.to(self.device)
            positive, negative = self.encode_prompt(prompt), self.encode_prompt(negative_prompt)
            self.text_encoder.cpu()
            for operation in operations:
                check_interrupt()
                if operation == 0:
                    frames.append(original.clone())
                    continue
                self.unet.to(self.device)
                # Reuse noise across keyframes, as in upstream's same_noise_in_batch.
                rng = torch.Generator(device=self.device).manual_seed(seed)
                def progress(items):
                    for item in items:
                        check_interrupt()
                        yield item
                latent = self.sampler(
                    initial_latent=torch.zeros_like(condition), strength=1.0,
                    num_inference_steps=sampling_steps, guidance_scale=cfg,
                    batch_size=1, generator=rng, prompt_embeds=positive,
                    negative_prompt_embeds=negative,
                    cross_attention_kwargs={"concat_conds": condition,
                                            "coded_conds": torch.tensor([operation], device=self.device)},
                    progress_tqdm=progress,
                )
                self.unet.cpu()
                self.vae.to(self.device)
                decoded = self.vae.decode(latent.float() / self.vae.config.scaling_factor).sample
                if not torch.isfinite(decoded).all():
                    raise RuntimeError("VAE produced non-finite pixels; no invalid frame was returned.")
                frames.append((decoded / 2 + 0.5).clamp(0, 1).movedim(1, -1).float().cpu())
                self.vae.cpu()
            if include_blank:
                frames.append(torch.ones_like(original))
            return torch.cat(frames, dim=0), prompt
        finally:
            for name in ("unet", "vae", "text_encoder"):
                model = getattr(self, name, None)
                if model is not None:
                    model.cpu()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
