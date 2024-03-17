from .nodes import SUPIR_Upscale
from .nodes_v2 import SUPIR_sample, SUPIR_model_loader, SUPIR_first_stage, SUPIR_encode, SUPIR_decode, SUPIR_conditioner, SUPIR_tiles

NODE_CLASS_MAPPINGS = {
    "SUPIR_Upscale": SUPIR_Upscale,
    "SUPIR_sample": SUPIR_sample,
    "SUPIR_model_loader": SUPIR_model_loader,
    "SUPIR_first_stage": SUPIR_first_stage,
    "SUPIR_encode": SUPIR_encode,
    "SUPIR_decode": SUPIR_decode,
    "SUPIR_conditioner": SUPIR_conditioner,
    "SUPIR_tiles": SUPIR_tiles
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SUPIR_Upscale": "SUPIR Upscale",
    "SUPIR_sample": "SUPIR Sampler",
    "SUPIR_model_loader": "SUPIR Model Loader",
    "SUPIR_first_stage": "SUPIR First Stage (Denoiser)",
    "SUPIR_encode": "SUPIR Encode",
    "SUPIR_decode": "SUPIR Decode",
    "SUPIR_conditioner": "SUPIR Conditioner",
    "SUPIR_tiles": "SUPIR Tiles"
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]