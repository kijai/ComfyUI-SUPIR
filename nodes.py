import os
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
import comfy.utils
import comfy.model_management as mm
import folder_paths
from nodes import ImageScaleBy
from nodes import ImageScale
import torch.cuda
from .sgm.util import instantiate_from_config
from .SUPIR.util import convert_dtype, load_state_dict
import open_clip
from contextlib import contextmanager

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextConfig,

)
script_directory = os.path.dirname(os.path.abspath(__file__))

def dummy_build_vision_tower(*args, **kwargs):
    # Monkey patch the CLIP class before you create an instance.
    return None

@contextmanager
def patch_build_vision_tower():
    original_build_vision_tower = open_clip.model._build_vision_tower
    open_clip.model._build_vision_tower = dummy_build_vision_tower

    try:
        yield
    finally:
        open_clip.model._build_vision_tower = original_build_vision_tower

def build_text_model_from_openai_state_dict(
        state_dict: dict,
        cast_dtype=torch.float16,
    ):

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = None
    text_cfg = open_clip.CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )

    with patch_build_vision_tower():
        model = open_clip.CLIP(
            embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=True,
            cast_dtype=cast_dtype,
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

class SUPIR_Upscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "supir_model": (folder_paths.get_filename_list("checkpoints"),),
            "sdxl_model": (folder_paths.get_filename_list("checkpoints"),),
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "resize_method": (s.upscale_methods, {"default": "lanczos"}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "steps": ("INT", {"default": 45, "min": 3, "max": 4096, "step": 1}),
            "restoration_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 6.0, "step": 1.0}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0, "max": 100, "step": 0.01}),
            "a_prompt": ("STRING", {"multiline": True, "default": "high quality, detailed", }),
            "n_prompt": ("STRING", {"multiline": True, "default": "bad quality, blurry, messy", }),
            "s_churn": ("INT", {"default": 5, "min": 0, "max": 40, "step": 1}),
            "s_noise": ("FLOAT", {"default": 1.003, "min": 1.0, "max": 1.1, "step": 0.001}),
            "control_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.05}),
            "cfg_scale_start": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.05}),
            "control_scale_start": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
            "color_fix_type": (
                [
                    'None',
                    'AdaIn',
                    'Wavelet',
                ], {
                    "default": 'Wavelet'
                }),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "use_tiled_vae": ("BOOLEAN", {"default": True}),
            "encoder_tile_size_pixels": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "decoder_tile_size_latent": ("INT", {"default": 64, "min": 32, "max": 8192, "step": 64}),
        },
            "optional": {
                "captions": ("STRING", {"forceInput": True, "multiline": False, "default": "", }),
                "diffusion_dtype": (
                    [
                        'fp16',
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
                "encoder_dtype": (
                    [
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "use_tiled_sampling": ("BOOLEAN", {"default": False}),
                "sampler_tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "sampler_tile_stride": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
                "fp8_unet": ("BOOLEAN", {"default": False}),
                "fp8_vae": ("BOOLEAN", {"default": False}),
                "sampler": (
                    [
                        'RestoreDPMPP2MSampler',
                        'RestoreEDMSampler',
                    ], {
                        "default": 'RestoreEDMSampler'
                    }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "SUPIR"

    def process(self, steps, image, color_fix_type, seed, scale_by, cfg_scale, resize_method, s_churn, s_noise,
                encoder_tile_size_pixels, decoder_tile_size_latent,
                control_scale, cfg_scale_start, control_scale_start, restoration_scale, keep_model_loaded,
                a_prompt, n_prompt, sdxl_model, supir_model, use_tiled_vae, use_tiled_sampling=False, sampler_tile_size=128, sampler_tile_stride=64, captions="", diffusion_dtype="auto",
                encoder_dtype="auto", batch_size=1, fp8_unet=False, fp8_vae=False, sampler="RestoreEDMSampler"):
        device = mm.get_torch_device()
        mm.unload_all_models()

        SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)
        SDXL_MODEL_PATH = folder_paths.get_full_path("checkpoints", sdxl_model)

        config_path = os.path.join(script_directory, "options/SUPIR_v0.yaml")
        config_path_tiled = os.path.join(script_directory, "options/SUPIR_v0_tiled.yaml")
        clip_config_path = os.path.join(script_directory, "configs/clip_vit_config.json")
        tokenizer_path = os.path.join(script_directory, "configs/tokenizer")

        custom_config = {
            'sdxl_model': sdxl_model,
            'diffusion_dtype': diffusion_dtype,
            'encoder_dtype': encoder_dtype,
            'use_tiled_vae': use_tiled_vae,
            'supir_model': supir_model,
            'use_tiled_sampling': use_tiled_sampling,
            'fp8_unet': fp8_unet,
            'fp8_vae': fp8_vae,
            'sampler': sampler
        }

        if diffusion_dtype == 'auto':
            try:
                if mm.should_use_fp16():
                    print("Diffusion using fp16")
                    dtype = torch.float16
                    model_dtype = 'fp16'
                if mm.should_use_bf16():
                    print("Diffusion using bf16")
                    dtype = torch.bfloat16
                    model_dtype = 'bf16'
                else:
                    print("Diffusion using using fp32")
                    dtype = torch.float32
                    model_dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI too old, can't autodecet properly. Set your dtypes manually.")
        else:
            print(f"Diffusion using using {diffusion_dtype}")
            dtype = convert_dtype(diffusion_dtype)
            model_dtype = diffusion_dtype

        if encoder_dtype == 'auto':
            try:
                if mm.should_use_bf16():
                    print("Encoder using bf16")
                    vae_dtype = 'bf16'
                else:
                    print("Encoder using using fp32")
                    vae_dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI too old, can't autodetect properly. Set your dtypes manually.")
        else:
            vae_dtype = encoder_dtype
            print(f"Encoder using using {vae_dtype}")

        if not hasattr(self, "model") or self.model is None or self.current_config != custom_config:
            self.current_config = custom_config
            self.model = None
            
            mm.soft_empty_cache()
            
            if use_tiled_sampling:
                config = OmegaConf.load(config_path_tiled)
                config.model.params.sampler_config.params.tile_size = sampler_tile_size // 8
                config.model.params.sampler_config.params.tile_stride = sampler_tile_stride // 8
                config.model.params.sampler_config.target = f".sgm.modules.diffusionmodules.sampling.Tiled{sampler}"
                print("Using tiled sampling")
            else:
                config = OmegaConf.load(config_path)
                config.model.params.sampler_config.target = f".sgm.modules.diffusionmodules.sampling.{sampler}"
                print("Using non-tiled sampling")

            if mm.XFORMERS_IS_AVAILABLE:
                config.model.params.control_stage_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers" 
                
            config.model.params.ae_dtype = vae_dtype
            config.model.params.diffusion_dtype = model_dtype
            
            self.model = instantiate_from_config(config.model).cpu()

            try:
                print(f'Attempting to load SUPIR model: [{SUPIR_MODEL_PATH}]')
                supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
                
            except:
                raise Exception("Failed to load SUPIR model")
            try:
                print(f"Attempting to load SDXL model: [{SDXL_MODEL_PATH}]")
                sdxl_state_dict = load_state_dict(SDXL_MODEL_PATH)
            except:
                raise Exception("Failed to load SDXL model")
            self.model.load_state_dict(supir_state_dict, strict=False)
            self.model.load_state_dict(sdxl_state_dict, strict=False)

            del supir_state_dict

            #first clip model from SDXL checkpoint
            try:
                print("Loading first clip model from SDXL checkpoint")
                
                replace_prefix = {}
                replace_prefix["conditioner.embedders.0.transformer."] = ""
    
                sd = comfy.utils.state_dict_prefix_replace(sdxl_state_dict, replace_prefix, filter_keys=False)
                clip_text_config = CLIPTextConfig.from_pretrained(clip_config_path)
                self.model.conditioner.embedders[0].tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
                self.model.conditioner.embedders[0].transformer = CLIPTextModel(clip_text_config)
                self.model.conditioner.embedders[0].transformer.load_state_dict(sd, strict=False)
                self.model.conditioner.embedders[0].eval()
                for param in self.model.conditioner.embedders[0].parameters():
                    param.requires_grad = False
            except:
                raise Exception("Failed to load first clip model from SDXL checkpoint")
            
            del sdxl_state_dict

            #second clip model from SDXL checkpoint
            try:
                print("Loading second clip model from SDXL checkpoint")
                replace_prefix2 = {}
                replace_prefix2["conditioner.embedders.1.model."] = ""
                sd = comfy.utils.state_dict_prefix_replace(sd, replace_prefix2, filter_keys=True)                
                clip_g = build_text_model_from_openai_state_dict(sd, cast_dtype=dtype)
                self.model.conditioner.embedders[1].model = clip_g
            except:
                raise Exception("Failed to load second clip model from SDXL checkpoint")
        
            del sd, clip_g
            mm.soft_empty_cache()

            self.model.to(dtype)

            #only unets and/or vae to fp8 
            if fp8_unet:
                self.model.model.to(torch.float8_e4m3fn)
            if fp8_vae:
                self.model.first_stage_model.to(torch.float8_e4m3fn)

            if use_tiled_vae:
                self.model.init_tile_vae(encoder_tile_size=encoder_tile_size_pixels, decoder_tile_size=decoder_tile_size_latent)
        
        upscaled_image, = ImageScaleBy.upscale(self, image, resize_method, scale_by)
        B, H, W, C = upscaled_image.shape
        new_height = H if H % 64 == 0 else ((H // 64) + 1) * 64
        new_width = W if W % 64 == 0 else ((W // 64) + 1) * 64
        upscaled_image = upscaled_image.permute(0, 3, 1, 2)
        resized_image = F.interpolate(upscaled_image, size=(new_height, new_width), mode='bicubic', align_corners=False)
        resized_image = resized_image.to(device)
        
        captions_list = []
        captions_list.append(captions)
        print("captions: ", captions_list)

        use_linear_CFG = cfg_scale_start > 0
        use_linear_control_scale = control_scale_start > 0
        out = []
        pbar = comfy.utils.ProgressBar(B)

        batched_images = [resized_image[i:i + batch_size] for i in
                          range(0, len(resized_image), batch_size)]
        captions_list = captions_list * resized_image.shape[0]
        batched_captions = [captions_list[i:i + batch_size] for i in range(0, len(captions_list), batch_size)]

        mm.soft_empty_cache()
        i = 1
        for imgs, caps in zip(batched_images, batched_captions):
            try:
                samples = self.model.batchify_sample(imgs, caps, num_steps=steps,
                                                     restoration_scale=restoration_scale, s_churn=s_churn,
                                                     s_noise=s_noise, cfg_scale=cfg_scale, control_scale=control_scale,
                                                     seed=seed,
                                                     num_samples=1, p_p=a_prompt, n_p=n_prompt,
                                                     color_fix_type=color_fix_type,
                                                     use_linear_CFG=use_linear_CFG,
                                                     use_linear_control_scale=use_linear_control_scale,
                                                     cfg_scale_start=cfg_scale_start,
                                                     control_scale_start=control_scale_start)
            except torch.cuda.OutOfMemoryError as e:
                mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
                self.model = None
                mm.soft_empty_cache()
                print("It's likely that too large of an image or batch_size for SUPIR was used,"
                      " and it has devoured all of the memory it had reserved, you may need to restart ComfyUI. Make sure you are using tiled_vae, "
                      " you can also try using fp8 for reduced memory usage if your system supports it.")
                raise e

            out.append(samples.squeeze(0).cpu())
            print("Sampled ", i * len(imgs), " out of ", B)
            i = i + 1
            pbar.update(1)
        if not keep_model_loaded:
            self.model = None
            mm.soft_empty_cache()

        if len(out[0].shape) == 4:
            out_stacked = torch.cat(out, dim=0).cpu().to(torch.float32).permute(0, 2, 3, 1)
        else:
            out_stacked = torch.stack(out, dim=0).cpu().to(torch.float32).permute(0, 2, 3, 1)
            
        final_image, = ImageScale.upscale(self, out_stacked, resize_method, W, H, crop="disabled")

        return (final_image,)

NODE_CLASS_MAPPINGS = {
    "SUPIR_Upscale": SUPIR_Upscale
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SUPIR_Upscale": "SUPIR_Upscale"
}