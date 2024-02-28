import os
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from omegaconf import OmegaConf

import comfy.model_management
import folder_paths
from nodes import ImageScaleBy
from nodes import ImageScale
import torch.cuda
from .SUPIR.models.SUPIR_model import SUPIRModel
from PIL import Image
from .sgm.util import instantiate_from_config
script_directory = os.path.dirname(os.path.abspath(__file__))

class SUPIR_Upscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "supir_model": (folder_paths.get_filename_list("checkpoints"), ),
            "sdxl_model": (folder_paths.get_filename_list("checkpoints"), ),
            "image": ("IMAGE", ),
            "resize_method": (s.upscale_methods, {"default": "lanczos"}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "steps": ("INT", {"default": 45, "min": 3, "max": 4096, "step": 1}),
            "cfg_scale": ("FLOAT", {"default": 7.5,"min": 0, "max": 20, "step": 0.01}),
            "a_prompt": ("STRING", {"multiline": True, "default": "high quality",}),
            "n_prompt": ("STRING", {"multiline": True, "default": "illustration",}),

            "min_size": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
          
            "color_fix_type": (
            [   
                'None',
                'AdaIn',
                'Wavelet',
            ], {
               "default": 'adain'
            }),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "SUPIR"

    def process(self, steps, image, color_fix_type, seed, scale_by, min_size, cfg_scale, resize_method, 
                a_prompt, n_prompt, sdxl_model,  supir_model, keep_model_loaded):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        comfy.model_management.unload_all_models()
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)
        SDXL_MODEL_PATH = folder_paths.get_full_path("checkpoints", sdxl_model)
        
        config_path = os.path.join(script_directory, "options/SUPIR_v0.yaml")
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32
        if not hasattr(self, "model") or self.model is None:
            
            config = OmegaConf.load(config_path)
            self.model = instantiate_from_config(config.model).cpu()
            from .SUPIR.util import load_state_dict
            supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
            sdxl_state_dict = load_state_dict(SDXL_MODEL_PATH)
            self.model.load_state_dict(supir_state_dict, strict=False)
            self.model.load_state_dict(sdxl_state_dict, strict=False)
            self.model.to(device).to(dtype)
   
        autocast_condition = dtype == torch.float16 or torch.bfloat16 and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image, = ImageScaleBy.upscale(self, image, resize_method, scale_by)
        
            # Assuming 'image' is a PyTorch tensor with shape [B, H, W, C] and you want to resize it.
            B, H, W, C = image.shape

            # Calculate the new height and width, rounding down to the nearest multiple of 64.
            new_height = H // 64 * 64
            new_width = W // 64 * 64

            # Reorder to [B, C, H, W] before using interpolate.
            image = image.permute(0, 3, 1, 2).contiguous()

            # Resize the image tensor.
            resized_image = F.interpolate(image, size=(new_height, new_width), mode='bicubic', align_corners=False)
            
            captions = ['']
            print(captions)

            # # step 3: Diffusion Process
            samples = self.model.batchify_sample(resized_image, captions, num_steps=steps, restoration_scale= -1, s_churn=5,
                                            s_noise=1.003, cfg_scale=cfg_scale, control_scale= 1, seed=seed,
                                            num_samples=1, p_p=a_prompt, n_p=n_prompt, color_fix_type=color_fix_type,
                                            use_linear_CFG=False, use_linear_control_scale=False,
                                            cfg_scale_start=1.0, control_scale_start=0)
            # save
            print(samples.shape)
            samples = samples.permute(0, 2, 3, 1).cpu()
            
            return(samples,)
    
NODE_CLASS_MAPPINGS = {
    "SUPIR_Upscale": SUPIR_Upscale,
    "SUPIR_Upscale": SUPIR_Upscale
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SUPIR_Upscale": "SUPIR_Upscale",
    "SUPIR_Upscale": "SUPIR_Upscale"
}