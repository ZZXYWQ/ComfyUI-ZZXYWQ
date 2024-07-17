import os
import subprocess
import torch
import numpy as np
from PIL import Image

# 导入必要的模块
from .memory_management import load_models_to_gpu, unload_all_models
from .wd14tagger import default_interrogator
from .diffusers_helper.k_diffusion import KDiffusionSampler
from .diffusers_helper.cat_cond import unet_add_concat_conds
from .diffusers_helper.code_cond import unet_add_coded_conds
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m

class ZZX_PaintsUndo:
    def __init__(self):
        self.model_name = 'lllyasviel/paints_undo_single_frame'
        self.custom_nodes_path = os.path.join(os.path.dirname(__file__), 'custom_nodes', 'ComfyUI-ZZXYWQ')
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.k_sampler = None
        self.download_files()
        self.initialize_models()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": ""}),
                "undo_steps": ("INT", {"default": 5, "min": 1, "max": 999, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "value_slider": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "ZZX/PaintsUndo"

    def download_files(self):
        if not os.path.exists(self.custom_nodes_path):
            os.makedirs(self.custom_nodes_path)
        if not os.listdir(self.custom_nodes_path):
            print("Downloading Paints-UNDO files...")
            subprocess.run(['git', 'clone', 'https://github.com/lllyasviel/Paints-UNDO.git', self.custom_nodes_path])
        else:
            print("Paints-UNDO files already exist.")

    def initialize_models(self):
        os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
        dtype = torch.float16

        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name, subfolder="text_encoder").to(dtype)
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(dtype)
        self.unet = ModifiedUNet.from_pretrained(self.model_name, subfolder="unet").to(dtype)

        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

        self.k_sampler = KDiffusionSampler(
            self.unet,
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.020,
            linear=True
        )

        unload_all_models([self.vae, self.text_encoder, self.unet])

    def process_image(self, image, prompt, undo_steps, seed, value_slider):
        print("Starting process_image method")
        pil_image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
    
        if not prompt:
            prompt = default_interrogator(pil_image)
    
        print(f"Input image shape: {np.array(pil_image).shape}")
        print(f"Prompt: {prompt}")
        print(f"Undo steps: {undo_steps}")
        print(f"Seed: {seed}")
        print(f"Value slider: {value_slider}")
    
        results = self.paints_undo_process(pil_image, prompt, undo_steps, seed, value_slider)
        
        # 选择中间的图像
        mid_index = len(results) // 2
        selected_image = results[mid_index]
        
        print(f"Selected image shape: {selected_image.shape}")
    
        # 转换为 ComfyUI 期望的格式：[C, H, W] 的 torch.Tensor，值范围 0-1
        output_image = torch.from_numpy(selected_image).float().permute(2, 0, 1) / 255.0
    
        print(f"Final output image shape: {output_image.shape}")
        print(f"Final output image min/max values: {output_image.min()}, {output_image.max()}")
    
        return (output_image,)

    def paints_undo_process(self, image, prompt, undo_steps, seed, value_slider):
        print("Starting paints_undo_process method")
        load_models_to_gpu([self.vae, self.text_encoder, self.unet])

        dtype = self.unet.dtype

        image = np.array(image)
        concat_conds = torch.from_numpy(image).unsqueeze(0).to(self.vae.device, dtype=dtype) / 127.5 - 1.0
        concat_conds = self.vae.encode(concat_conds.permute(0, 3, 1, 2)).latent_dist.mode() * self.vae.config.scaling_factor

        print(f"Concat_conds shape: {concat_conds.shape}")

        conds = self.encode_prompt(prompt)
        unconds = self.encode_prompt("")

        generator = torch.Generator(device=self.unet.device).manual_seed(seed)

        fs = torch.tensor([undo_steps], device=self.unet.device, dtype=torch.long)
        latents = self.k_sampler(
            initial_latent=torch.zeros_like(concat_conds),
            strength=0.8,
            num_inference_steps=30,
            guidance_scale=value_slider * 10,
            batch_size=3,  # 输出三张图
            generator=generator,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            cross_attention_kwargs={'concat_conds': concat_conds, 'coded_conds': fs},
        )

        print(f"Latents shape after sampling: {latents.shape}")

        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        print(f"Images shape after VAE decode: {images.shape}")
        print(f"Images min/max values: {images.min()}, {images.max()}")
        
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        unload_all_models([self.vae, self.text_encoder, self.unet])

        final_images = [(img * 255).astype(np.uint8) for img in images]
        for i, img in enumerate(final_images):
            print(f"Final image {i+1} shape: {img.shape}")
            print(f"Final image {i+1} min/max values: {img.min()}, {img.max()}")
        
        return final_images

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        return prompt_embeds

NODE_CLASS_MAPPINGS = {
    "ZZX_PaintsUndo": ZZX_PaintsUndo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZZX_PaintsUndo": "ZZX Paints Undo Node"
}
