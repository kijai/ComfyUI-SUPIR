import os
import torch
from omegaconf import OmegaConf
import comfy.utils
import comfy.model_management as mm
import folder_paths
import torch.cuda
import torch.nn.functional as F
from .sgm.util import instantiate_from_config
from .SUPIR.util import convert_dtype, load_state_dict
from .sgm.modules.distributions.distributions import DiagonalGaussianDistribution
import open_clip
from contextlib import contextmanager, nullcontext
import gc

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

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
        device,
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
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            model = open_clip.CLIP(
                embed_dim,
                vision_cfg=vision_cfg,
                text_cfg=text_cfg,
                quick_gelu=True,
                cast_dtype=cast_dtype,
            )
    if is_accelerate_available:
        for key in state_dict:
            set_module_tensor_to_device(model, key, device=device, value=state_dict[key])
    else:
        model.load_state_dict(state_dict, strict=False)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

class SUPIR_encode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_VAE": ("SUPIRVAE",),
            "image": ("IMAGE",),
            "use_tiled_vae": ("BOOLEAN", {"default": True}),
            "encoder_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "encoder_dtype": (
                    [
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "SUPIR"

    def encode(self, SUPIR_VAE, image, encoder_dtype, use_tiled_vae, encoder_tile_size):
        device = mm.get_torch_device()
        mm.unload_all_models()
        if encoder_dtype == 'auto':
            try:
                if mm.should_use_bf16():
                    print("Encoder using bf16")
                    vae_dtype = 'bf16'
                else:
                    print("Encoder using fp32")
                    vae_dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
        else:
            vae_dtype = encoder_dtype
            print(f"Encoder using {vae_dtype}")

        dtype = convert_dtype(vae_dtype)

        image = image.permute(0, 3, 1, 2)
        B, C, H, W = image.shape
        downscale_ratio = 32
        orig_H, orig_W = H, W
        if W % downscale_ratio != 0:
            W = W - (W % downscale_ratio)
        if H % downscale_ratio != 0:
            H = H - (H % downscale_ratio)
        if orig_H % downscale_ratio != 0 or orig_W % downscale_ratio != 0:
            image = F.interpolate(image, size=(H, W), mode="bicubic")
        resized_image = image.to(device)        
        
        if use_tiled_vae:
            from .SUPIR.utils.tilevae import VAEHook
            # Store the `original_forward` only if it hasn't been stored already
            if not hasattr(SUPIR_VAE.encoder, 'original_forward'):
                SUPIR_VAE.encoder.original_forward = SUPIR_VAE.encoder.forward
            SUPIR_VAE.encoder.forward = VAEHook(
                SUPIR_VAE.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
        else:
            # Only assign `original_forward` back if it exists
            if hasattr(SUPIR_VAE.encoder, 'original_forward'):
                SUPIR_VAE.encoder.forward = SUPIR_VAE.encoder.original_forward
        
        pbar = comfy.utils.ProgressBar(B)
        out = []
        for img in resized_image:

            SUPIR_VAE.to(dtype).to(device)

            autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
            with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():

                z = SUPIR_VAE.encode(img.unsqueeze(0))
                z = z * 0.13025
                out.append(z)
                pbar.update(1)

        if len(out[0].shape) == 4:
            samples_out_stacked = torch.cat(out, dim=0)
        else:
            samples_out_stacked = torch.stack(out, dim=0)
        return ({"samples":samples_out_stacked, "original_size": [orig_H, orig_W]},)

class SUPIR_decode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_VAE": ("SUPIRVAE",),
            "latents": ("LATENT",),
            "use_tiled_vae": ("BOOLEAN", {"default": True}),
            "decoder_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "SUPIR"

    def decode(self, SUPIR_VAE, latents, use_tiled_vae, decoder_tile_size):
        device = mm.get_torch_device()
        mm.unload_all_models()
        samples = latents["samples"]
        
        B, H, W, C = samples.shape
                
        pbar = comfy.utils.ProgressBar(B)
       
        if mm.should_use_bf16():
            print("Decoder using bf16")
            dtype = torch.bfloat16
        else:
            print("Decoder using fp32")
            dtype = torch.float32
        print("SUPIR decoder using", dtype)
           
        SUPIR_VAE.to(dtype).to(device)
        samples = samples.to(device)

        if use_tiled_vae:
            from .SUPIR.utils.tilevae import VAEHook
            # Store the `original_forward` only if it hasn't been stored already
            if not hasattr(SUPIR_VAE.decoder, 'original_forward'):
                SUPIR_VAE.decoder.original_forward = SUPIR_VAE.decoder.forward
            SUPIR_VAE.decoder.forward = VAEHook(
                SUPIR_VAE.decoder, decoder_tile_size // 8, is_decoder=True, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
        else:
            # Only assign `original_forward` back if it exists
            if hasattr(SUPIR_VAE.decoder, 'original_forward'):
                SUPIR_VAE.decoder.forward = SUPIR_VAE.decoder.original_forward

        out = []
        for sample in samples:
            autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
            with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
                sample = 1.0 / 0.13025 * sample
                decoded_image = SUPIR_VAE.decode(sample.unsqueeze(0))
                out.append(decoded_image)
                pbar.update(1)

        decoded_out= torch.cat(out, dim=0).float()

        if "original_size" in latents and latents["original_size"] is not None:
            orig_H, orig_W = latents["original_size"]
            if decoded_out.shape[2] != orig_H or decoded_out.shape[3] != orig_W:
                print("Restoring original dimensions: ", orig_W,"x",orig_H)
                decoded_out = F.interpolate(decoded_out, size=(orig_H, orig_W), mode="bicubic")

        decoded_out = torch.clip(decoded_out, 0, 1)
        decoded_out = decoded_out.cpu().to(torch.float32).permute(0, 2, 3, 1)
        

        return (decoded_out,)
        
class SUPIR_first_stage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_VAE": ("SUPIRVAE",),
            "image": ("IMAGE",),
            "use_tiled_vae": ("BOOLEAN", {"default": True}),
            "encoder_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "decoder_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "encoder_dtype": (
                    [
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            }
        }

    RETURN_TYPES = ("SUPIRVAE", "IMAGE", "LATENT",)
    RETURN_NAMES = ("SUPIR_VAE", "denoised_image", "denoised_latents",)
    FUNCTION = "process"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
SUPIR "first stage" processing.
Encodes and decodes the image using SUPIR's "denoise_encoder", purpose  
is to fix compression artifacts and such, ends up blurring the image often  
which is expected. Can be replaced with any other denoiser/blur or not used at all.
"""

    def process(self, SUPIR_VAE, image, encoder_dtype, use_tiled_vae, encoder_tile_size, decoder_tile_size):
        device = mm.get_torch_device()
        mm.unload_all_models()
        if encoder_dtype == 'auto':
            try:
                
                if mm.should_use_bf16():
                    print("Encoder using bf16")
                    vae_dtype = 'bf16'
                else:
                    print("Encoder using fp32")
                    vae_dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
        else:
            vae_dtype = encoder_dtype
            print(f"Encoder using {vae_dtype}")

        dtype = convert_dtype(vae_dtype)

        if use_tiled_vae:
            from .SUPIR.utils.tilevae import VAEHook
            # Store the `original_forward` only if it hasn't been stored already
            if not hasattr(SUPIR_VAE.encoder, 'original_forward'):
                SUPIR_VAE.denoise_encoder.original_forward = SUPIR_VAE.denoise_encoder.forward
                SUPIR_VAE.decoder.original_forward = SUPIR_VAE.decoder.forward
                     
            SUPIR_VAE.denoise_encoder.forward = VAEHook(
                SUPIR_VAE.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
            
            SUPIR_VAE.decoder.forward = VAEHook(
                SUPIR_VAE.decoder, decoder_tile_size // 8, is_decoder=True, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
        else:
            # Only assign `original_forward` back if it exists
            if hasattr(SUPIR_VAE.denoise_encoder, 'original_forward'):
                SUPIR_VAE.denoise_encoder.forward = SUPIR_VAE.denoise_encoder.original_forward
                SUPIR_VAE.decoder.forward = SUPIR_VAE.decoder.original_forward

        image = image.permute(0, 3, 1, 2)
        B, C, H, W = image.shape
        downscale_ratio = 32
        orig_H, orig_W = H, W
        if W % downscale_ratio != 0:
            W = W - (W % downscale_ratio)
        if H % downscale_ratio != 0:
            H = H - (H % downscale_ratio)
        if orig_H % downscale_ratio != 0 or orig_W % downscale_ratio != 0:
            image = F.interpolate(image, size=(H, W), mode="bicubic")
        resized_image = image.to(device)
        
        pbar = comfy.utils.ProgressBar(B)
        out = []
        out_samples = []
        for img in resized_image:

            SUPIR_VAE.to(dtype).to(device)

            autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
            with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
                
                h = SUPIR_VAE.denoise_encoder(img.unsqueeze(0))
                moments = SUPIR_VAE.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                sample = posterior.sample()
                decoded_images = SUPIR_VAE.decode(sample).float()

                out.append(decoded_images.cpu())
                out_samples.append(sample.cpu() * 0.13025)
                pbar.update(1)


        out_stacked = torch.cat(out, dim=0).to(torch.float32).permute(0, 2, 3, 1)
        out_samples_stacked = torch.cat(out_samples, dim=0)
        original_size = [orig_H, orig_W]
        return (SUPIR_VAE, out_stacked, {"samples": out_samples_stacked, "original_size": original_size},)

class SUPIR_sample:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_model": ("SUPIRMODEL",),
            "latents": ("LATENT",),
            "positive": ("SUPIR_cond_pos",),
            "negative": ("SUPIR_cond_neg",),
            "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "steps": ("INT", {"default": 45, "min": 3, "max": 4096, "step": 1}),
            "cfg_scale_start": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "cfg_scale_end": ("FLOAT", {"default": 4.0, "min": 0, "max": 100.0, "step": 0.01}),
            "EDM_s_churn": ("INT", {"default": 5, "min": 0, "max": 40, "step": 1}),
            "s_noise": ("FLOAT", {"default": 1.003, "min": 1.0, "max": 1.1, "step": 0.001}),
            "DPMPP_eta": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01}),
            "control_scale_start": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01}),
            "control_scale_end": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.01}),
            "restore_cfg": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.01}),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            "sampler": (
                    [
                        'RestoreDPMPP2MSampler',
                        'RestoreEDMSampler',
                        'TiledRestoreDPMPP2MSampler',
                        'TiledRestoreEDMSampler',
                    ], {
                        "default": 'RestoreEDMSampler'
                    }),
        },
            "optional": {
                "sampler_tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "sampler_tile_stride": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
- **latent:**
Latent to sample from, when using SUPIR latent this is just for the noise shape,  
it's actually not used otherwise here. Identical to feeding this comfy empty latent.  
If fed anything else it's used as it is, no noise is added.  
- **cfg:**
Linearly scaled CFG is always used, first step will use the cfg_scale_start value,  
and that is interpolated to the cfg_scale_end value at last step.  
To disable scaling set these values to be the same.  
- **EDM_s_churn:**
controls the rate of adaptation of the diffusion process to changes in noise levels  
over time. Has no effect with DPMPP samplers.  
- **s_noise:**
This parameter directly controls the amount of noise added to the image at each  
step of the diffusion process.  
- **DPMPP_eta:**
Scaling factor that influences the diffusion process by adjusting how the denoising  
process adapts to changes in noise levels over time.
No effect with EDM samplers.  
- **control_scale:**
The strenght of the SUPIR control model, scales linearly from start to end.  
Lower values allow more freedom from the input image.  
- **restore_cfg:**
Controls the degree of restoration towards the original image during the diffusion   
process. It allows for dome fine-tuning of the process.  
- **samplers:**
EDM samplers need lots of steps but generally have better quality.  
DPMPP samplers work well with lower steps, good for lightning models.  
Tiled samplers enable tiled diffusion process, this is very slow but allows higher  
resolutions to be used by saving VRAM.  Tile size should be chosen so the image  
is evenly tiled.  Tile stride affects the overlap of the tiles.  Check the  
SUPIR Tiles -node for preview to understand how the image is tiled.

"""

    def sample(self, SUPIR_model, latents, steps, seed, cfg_scale_end, EDM_s_churn, s_noise, positive, negative,
                cfg_scale_start, control_scale_start, control_scale_end, restore_cfg, keep_model_loaded, DPMPP_eta,
                sampler, sampler_tile_size=1024, sampler_tile_stride=512):
        
        torch.manual_seed(seed)
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()

        self.sampler_config = {
            'target': f'.sgm.modules.diffusionmodules.sampling.{sampler}',
            'params': {
                'num_steps': steps,
                'restore_cfg': restore_cfg,
                's_churn': EDM_s_churn,
                's_noise': s_noise,
                'discretization_config': {
                    'target': '.sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization'
                },
                'guider_config': {
                    'target': '.sgm.modules.diffusionmodules.guiders.LinearCFG',
                    'params': {
                        'scale': cfg_scale_start,
                        'scale_min': cfg_scale_end
                    }
                }
            }
        }
        if 'Tiled' in sampler:
            self.sampler_config['params']['tile_size'] = sampler_tile_size // 8
            self.sampler_config['params']['tile_stride'] = sampler_tile_stride // 8
        if 'DPMPP' in sampler:
            self.sampler_config['params']['eta'] = DPMPP_eta
            self.sampler_config['params']['restore_cfg'] = -1
        if not hasattr (self,'sampler') or self.sampler_config != self.current_sampler_config: 
            self.sampler = instantiate_from_config(self.sampler_config)
            self.current_sampler_config = self.sampler_config
 
        print("sampler_config: ", self.sampler_config)
        
        SUPIR_model.denoiser.to(device)
        SUPIR_model.model.diffusion_model.to(device)
        SUPIR_model.model.control_model.to(device)
        
        use_linear_control_scale = control_scale_start != control_scale_end

        denoiser = lambda input, sigma, c, control_scale: SUPIR_model.denoiser(SUPIR_model.model, input, sigma, c, control_scale)

        original_size = positive['original_size']
        positive = positive['cond']
        negative = negative['uncond']
        samples = latents["samples"]
        samples = samples.to(device)
        #print("positives: ", len(positive))
        #print("negatives: ", len(negative))
        out = []
        pbar = comfy.utils.ProgressBar(samples.shape[0])
        for i, sample in enumerate(samples):
            try:
                if 'original_size' in latents:
                    print("Using random noise")
                    noised_z = torch.randn_like(sample.unsqueeze(0), device=samples.device)
                else:
                    print("Using latent from input")
                    noised_z = torch.randn_like(sample.unsqueeze(0), device=samples.device)
                    noised_z += sample.unsqueeze(0)
                if len(positive) != len(samples):
                    print("Tiled sampling")
                    _samples = self.sampler(denoiser, noised_z, cond=positive, uc=negative, x_center=sample.unsqueeze(0), control_scale=control_scale_end,
                                    use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)
                else:
                    #print("positives[i]: ", len(positive[i]))
                    #print("negatives[i]: ", len(negative[i]))
                    _samples = self.sampler(denoiser, noised_z, cond=positive[i], uc=negative[i], x_center=sample.unsqueeze(0), control_scale=control_scale_end,
                                            use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)

                
            except torch.cuda.OutOfMemoryError as e:
                mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
                SUPIR_model = None
                mm.soft_empty_cache()
                print("It's likely that too large of an image or batch_size for SUPIR was used,"
                      " and it has devoured all of the memory it had reserved, you may need to restart ComfyUI. Make sure you are using tiled_vae, "
                      " you can also try using fp8 for reduced memory usage if your system supports it.")
                raise e
            out.append(_samples)
            print("Sampled ", i+1, " of ", samples.shape[0])
            pbar.update(1)

        if not keep_model_loaded:
            SUPIR_model.denoiser.to('cpu')
            SUPIR_model.model.diffusion_model.to('cpu')
            SUPIR_model.model.control_model.to('cpu')
            mm.soft_empty_cache()

        if len(out[0].shape) == 4:
            samples_out_stacked = torch.cat(out, dim=0)
        else:
            samples_out_stacked = torch.stack(out, dim=0)

        if original_size is None:
            samples_out_stacked = samples_out_stacked / 0.13025

        return ({"samples":samples_out_stacked, "original_size": original_size},)

class SUPIR_conditioner:
    # @classmethod
    # def IS_CHANGED(s):
    #     return ""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_model": ("SUPIRMODEL",),
            "latents": ("LATENT",),
            "positive_prompt": ("STRING", {"multiline": True, "default": "high quality, detailed", }),
            "negative_prompt": ("STRING", {"multiline": True, "default": "bad quality, blurry, messy", }),
        },
            "optional": {
                "captions": ("STRING", {"forceInput": True, "multiline": False, "default": "", }),
            }
        }

    RETURN_TYPES = ("SUPIR_cond_pos", "SUPIR_cond_neg",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "condition"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Creates the conditioning for the sampler.  
Caption input is optional, when it receives a single caption, it's added to the positive prompt.
    
If a list of caption is given for single input image, the captions need to match the number of tiles,  
refer to the SUPIR Tiles node.  
  
If a list of captions is given and it matches the incoming image batch, each image uses corresponding caption.
"""

    def condition(self, SUPIR_model, latents, positive_prompt, negative_prompt, captions=""):
        
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        if "original_size" in latents:
            original_size = latents["original_size"]
            samples = latents["samples"]
        else:
            original_size = None
            samples = latents["samples"] * 0.13025
        
        N, H, W, C = samples.shape
        import copy

        if not isinstance(captions, list):
            captions_list = []
            captions_list.append([captions])
            captions_list = captions_list * N
        else:
            captions_list = captions

        print("captions: ", captions_list)
      
        SUPIR_model.conditioner.to(device)
        samples = samples.to(device)

        uc = []
        pbar = comfy.utils.ProgressBar(N)
        autocast_condition = (SUPIR_model.model.dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=SUPIR_model.model.dtype) if autocast_condition else nullcontext():
            if N != len(captions_list): #Tiled captioning
                print("Tiled captioning")
                c = []
                uc = []
                for i, caption in enumerate(captions_list):
                    cond = {}
                    cond['original_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['crop_coords_top_left'] = torch.tensor([[0, 0]]).to(device)
                    cond['target_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['aesthetic_score'] = torch.tensor([[9.0]]).to(device)
                    cond['control'] = samples[0].unsqueeze(0)

                    uncond = copy.deepcopy(cond)
                    uncond['txt'] = [negative_prompt]
                    
                    cond['txt'] = [''.join([caption[0], positive_prompt])]
                    if i == 0:
                        _c, uc = SUPIR_model.conditioner.get_unconditional_conditioning(cond, uncond)
                    else:
                        _c, _ = SUPIR_model.conditioner.get_unconditional_conditioning(cond, None)
    
                    c.append(_c)
                    pbar.update(1)
            else: #batch captioning
                print("Batch captioning")
                c = []
                uc = []
                for i, sample in enumerate(samples):
                    
                    cond = {}
                    cond['original_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['crop_coords_top_left'] = torch.tensor([[0, 0]]).to(device)
                    cond['target_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['aesthetic_score'] = torch.tensor([[9.0]]).to(device)
                    cond['control'] = sample.unsqueeze(0)

                    uncond = copy.deepcopy(cond)
                    uncond['txt'] = [negative_prompt]
                    cond['txt'] = [''.join([captions_list[i][0], positive_prompt])]
                    _c, _uc = SUPIR_model.conditioner.get_unconditional_conditioning(cond, uncond)    
                    c.append(_c)
                    uc.append(_uc)
                    
                    pbar.update(1)

            
        SUPIR_model.conditioner.to('cpu')

        if "original_size" in latents:
            original_size = latents["original_size"]
        else:
            original_size = None
                
        return ({"cond": c, "original_size":original_size}, {"uncond": uc},)
    
class SUPIR_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "supir_model": (folder_paths.get_filename_list("checkpoints"),),
            "sdxl_model": (folder_paths.get_filename_list("checkpoints"),),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            "diffusion_dtype": (
                    [
                        'fp16',
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
        }

    RETURN_TYPES = ("SUPIRMODEL", "SUPIRVAE")
    RETURN_NAMES = ("SUPIR_model","SUPIR_VAE",)
    FUNCTION = "process"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Old loader, not recommended to be used.  
Loads the SUPIR model and the selected SDXL model and merges them.
"""

    def process(self, supir_model, sdxl_model, diffusion_dtype, fp8_unet):
        device = mm.get_torch_device()
        mm.unload_all_models()

        SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)
        SDXL_MODEL_PATH = folder_paths.get_full_path("checkpoints", sdxl_model)

        config_path = os.path.join(script_directory, "options/SUPIR_v0.yaml")
        clip_config_path = os.path.join(script_directory, "configs/clip_vit_config.json")
        tokenizer_path = os.path.join(script_directory, "configs/tokenizer")

        custom_config = {
            'sdxl_model': sdxl_model,
            'diffusion_dtype': diffusion_dtype,
            'supir_model': supir_model,
            'fp8_unet': fp8_unet,
        }

        if diffusion_dtype == 'auto':
            try:
                if mm.should_use_fp16():
                    print("Diffusion using fp16")
                    dtype = torch.float16
                    model_dtype = 'fp16'
                elif mm.should_use_bf16():
                    print("Diffusion using bf16")
                    dtype = torch.bfloat16
                    model_dtype = 'bf16'
                else:
                    print("Diffusion using fp32")
                    dtype = torch.float32
                    model_dtype = 'fp32'
            except:
                raise AttributeError("ComfyUI version too old, can't autodetect properly. Set your dtypes manually.")
        else:
            print(f"Diffusion using {diffusion_dtype}")
            dtype = convert_dtype(diffusion_dtype)
            model_dtype = diffusion_dtype
        
        if not hasattr(self, "model") or self.model is None or self.current_config != custom_config:
            self.current_config = custom_config
            self.model = None
            
            mm.soft_empty_cache()
            
            config = OmegaConf.load(config_path)
           
            if mm.XFORMERS_IS_AVAILABLE:
                print("Using XFORMERS")
                config.model.params.control_stage_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers" 
                
            config.model.params.diffusion_dtype = model_dtype
            config.model.target = ".SUPIR.models.SUPIR_model_v2.SUPIRModel"
            pbar = comfy.utils.ProgressBar(5)

            self.model = instantiate_from_config(config.model).cpu()
            self.model.model.dtype = dtype
            pbar.update(1)
            try:
                print(f"Attempting to load SDXL model: [{SDXL_MODEL_PATH}]")
                sdxl_state_dict = load_state_dict(SDXL_MODEL_PATH)
                self.model.load_state_dict(sdxl_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                pbar.update(1)
            except:
                raise Exception("Failed to load SDXL model")
            
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
                self.model.conditioner.embedders[0].to(dtype)
                for param in self.model.conditioner.embedders[0].parameters():
                    param.requires_grad = False
                pbar.update(1)
            except:
                raise Exception("Failed to load first clip model from SDXL checkpoint")
            
            del sdxl_state_dict

            #second clip model from SDXL checkpoint
            try:
                print("Loading second clip model from SDXL checkpoint")
                replace_prefix2 = {}
                replace_prefix2["conditioner.embedders.1.model."] = ""
                sd = comfy.utils.state_dict_prefix_replace(sd, replace_prefix2, filter_keys=True)                
                clip_g = build_text_model_from_openai_state_dict(sd, device, cast_dtype=dtype)
                self.model.conditioner.embedders[1].model = clip_g
                self.model.conditioner.embedders[1].to(dtype)
                pbar.update(1)
            except:
                raise Exception("Failed to load second clip model from SDXL checkpoint")
        
            del sd, clip_g

            try:
                print(f'Attempting to load SUPIR model: [{SUPIR_MODEL_PATH}]')
                supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
                self.model.load_state_dict(supir_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                del supir_state_dict
                pbar.update(1)
            except:
                raise Exception("Failed to load SUPIR model")
            mm.soft_empty_cache()

        return (self.model, self.model.first_stage_model,)

class SUPIR_model_loader_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model" :("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "supir_model": (folder_paths.get_filename_list("checkpoints"),),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            "diffusion_dtype": (
                    [
                        'fp16',
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
            "optional": {
                "high_vram": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SUPIRMODEL", "SUPIRVAE")
    RETURN_NAMES = ("SUPIR_model","SUPIR_VAE",)
    FUNCTION = "process"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Loads the SUPIR model and merges it with the SDXL model.  

Diffusion type should be kept on auto, unless you have issues loading the model.  
fp8_unet casts the unet weights to torch.float8_e4m3fn, which saves a lot of VRAM but has slight quality impact.  
high_vram: uses Accelerate to load weights to GPU, slightly faster model loading.
"""

    def process(self, supir_model, diffusion_dtype, fp8_unet, model, clip, vae, high_vram=False):
        if high_vram:
            device = mm.get_torch_device()
        else:
            device = mm.unet_offload_device()
        print("Loading weights to: ", device)
        mm.unload_all_models()

        SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)

        config_path = os.path.join(script_directory, "options/SUPIR_v0.yaml")
        clip_config_path = os.path.join(script_directory, "configs/clip_vit_config.json")
        tokenizer_path = os.path.join(script_directory, "configs/tokenizer")

        custom_config = {
            'diffusion_dtype': diffusion_dtype,
            'supir_model': supir_model,
            'fp8_unet': fp8_unet,
            'model': model,
            "clip": clip,
            "vae": vae
        }

        if diffusion_dtype == 'auto':
            try:
                if mm.should_use_fp16():
                    print("Diffusion using fp16")
                    dtype = torch.float16
                elif mm.should_use_bf16():
                    print("Diffusion using bf16")
                    dtype = torch.bfloat16
                else:
                    print("Diffusion using fp32")
                    dtype = torch.float32
            except:
                raise AttributeError("ComfyUI version too old, can't autodecet properly. Set your dtypes manually.")
        else:
            print(f"Diffusion using {diffusion_dtype}")
            dtype = convert_dtype(diffusion_dtype)
        
        if not hasattr(self, "model") or self.model is None or self.current_config != custom_config:
            self.current_config = custom_config
            self.model = None
            
            mm.soft_empty_cache()
            
            config = OmegaConf.load(config_path)
            if mm.XFORMERS_IS_AVAILABLE:
                print("Using XFORMERS")
                config.model.params.control_stage_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers" 
                
            config.model.target = ".SUPIR.models.SUPIR_model_v2.SUPIRModel"
            pbar = comfy.utils.ProgressBar(5)

            #with (init_empty_weights() if is_accelerate_available else nullcontext()):
            self.model = instantiate_from_config(config.model).cpu()
            self.model.model.dtype = dtype
            pbar.update(1)
            try:
                print(f"Attempting to load SDXL model from node inputs")
                mm.load_model_gpu(model)
                sdxl_state_dict = model.model.state_dict_for_saving(None, vae.get_sd(), None)
                if is_accelerate_available:
                    for key in sdxl_state_dict:
                        set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=sdxl_state_dict[key])
                else:
                    self.model.load_state_dict(sdxl_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                del sdxl_state_dict
                pbar.update(1)
            except:
                raise Exception("Failed to load SDXL model")            
            gc.collect()
            mm.soft_empty_cache()
            #first clip model from SDXL checkpoint
            try:
                print("Loading first clip model from SDXL checkpoint")
                clip_sd = None
                clip_model = clip.load_model()
                mm.load_model_gpu(clip_model)
                clip_sd = clip.get_sd()
                clip_sd = model.model.model_config.process_clip_state_dict_for_saving(clip_sd)

                replace_prefix = {}
                replace_prefix["conditioner.embedders.0.transformer."] = ""
    
                clip_l_sd = comfy.utils.state_dict_prefix_replace(clip_sd, replace_prefix, filter_keys=True)
                clip_text_config = CLIPTextConfig.from_pretrained(clip_config_path)
                self.model.conditioner.embedders[0].tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
                with (init_empty_weights() if is_accelerate_available else nullcontext()):
                    self.model.conditioner.embedders[0].transformer = CLIPTextModel(clip_text_config)
                if is_accelerate_available:
                    for key in clip_l_sd:
                        set_module_tensor_to_device(self.model.conditioner.embedders[0].transformer, key, device=device, dtype=dtype, value=clip_l_sd[key])
                else:
                    self.model.conditioner.embedders[0].transformer.load_state_dict(clip_l_sd, strict=False)
                self.model.conditioner.embedders[0].eval()
                for param in self.model.conditioner.embedders[0].parameters():
                    param.requires_grad = False
                self.model.conditioner.embedders[0].to(dtype)
                del clip_l_sd
                pbar.update(1)
            except:
                raise Exception("Failed to load first clip model from SDXL checkpoint")
            gc.collect()
            mm.soft_empty_cache()
            #second clip model from SDXL checkpoint
            try:
                print("Loading second clip model from SDXL checkpoint")
                replace_prefix2 = {}
                replace_prefix2["conditioner.embedders.1.model."] = ""
                clip_g_sd = comfy.utils.state_dict_prefix_replace(clip_sd, replace_prefix2, filter_keys=True)             
                clip_g = build_text_model_from_openai_state_dict(clip_g_sd, device, cast_dtype=dtype)
                self.model.conditioner.embedders[1].model = clip_g
                self.model.conditioner.embedders[1].model.to(dtype)
                del clip_g_sd
                pbar.update(1)
            except:
                raise Exception("Failed to load second clip model from SDXL checkpoint")
            
            try:
                print(f'Attempting to load SUPIR model: [{SUPIR_MODEL_PATH}]')
                supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
                if "Q" not in supir_model or not is_accelerate_available: #I don't know why this doesn't work with the Q model. 
                    for key in supir_state_dict:
                        set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=supir_state_dict[key])
                else:
                    self.model.load_state_dict(supir_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                del supir_state_dict
                pbar.update(1)
            except:
                raise Exception("Failed to load SUPIR model")
            mm.soft_empty_cache()

        return (self.model, self.model.first_stage_model,)
    
class SUPIR_model_loader_v2_clip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model" :("MODEL",),
            "clip_l": ("CLIP",),
            "clip_g": ("CLIP",),
            "vae": ("VAE",),
            "supir_model": (folder_paths.get_filename_list("checkpoints"),),
            "fp8_unet": ("BOOLEAN", {"default": False}),
            "diffusion_dtype": (
                    [
                        'fp16',
                        'bf16',
                        'fp32',
                        'auto'
                    ], {
                        "default": 'auto'
                    }),
            },
            "optional": {
                "high_vram": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SUPIRMODEL", "SUPIRVAE")
    RETURN_NAMES = ("SUPIR_model","SUPIR_VAE",)
    FUNCTION = "process"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Loads the SUPIR model and merges it with the SDXL model.  

Diffusion type should be kept on auto, unless you have issues loading the model.  
fp8_unet casts the unet weights to torch.float8_e4m3fn, which saves a lot of VRAM but has slight quality impact.  
high_vram: uses Accelerate to load weights to GPU, slightly faster model loading.
"""

    def process(self, supir_model, diffusion_dtype, fp8_unet, model, clip_l, clip_g, vae, high_vram=False):
        if high_vram:
            device = mm.get_torch_device()
        else:
            device = mm.unet_offload_device()
        print("Loading weights to: ", device)
        mm.unload_all_models()

        SUPIR_MODEL_PATH = folder_paths.get_full_path("checkpoints", supir_model)

        config_path = os.path.join(script_directory, "options/SUPIR_v0.yaml")
        clip_config_path = os.path.join(script_directory, "configs/clip_vit_config.json")
        tokenizer_path = os.path.join(script_directory, "configs/tokenizer")

        custom_config = {
            'diffusion_dtype': diffusion_dtype,
            'supir_model': supir_model,
            'fp8_unet': fp8_unet,
            'model': model,
            "clip": clip_l,
            "clip_g": clip_g,
            "vae": vae
        }

        if diffusion_dtype == 'auto':
            try:
                if mm.should_use_fp16():
                    print("Diffusion using fp16")
                    dtype = torch.float16
                elif mm.should_use_bf16():
                    print("Diffusion using bf16")
                    dtype = torch.bfloat16
                else:
                    print("Diffusion using fp32")
                    dtype = torch.float32
            except:
                raise AttributeError("ComfyUI version too old, can't autodecet properly. Set your dtypes manually.")
        else:
            print(f"Diffusion using {diffusion_dtype}")
            dtype = convert_dtype(diffusion_dtype)
        
        if not hasattr(self, "model") or self.model is None or self.current_config != custom_config:
            self.current_config = custom_config
            self.model = None
            
            mm.soft_empty_cache()
            
            config = OmegaConf.load(config_path)
            if mm.XFORMERS_IS_AVAILABLE:
                print("Using XFORMERS")
                config.model.params.control_stage_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.network_config.params.spatial_transformer_attn_type = "softmax-xformers"
                config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla-xformers" 
                
            config.model.target = ".SUPIR.models.SUPIR_model_v2.SUPIRModel"
            pbar = comfy.utils.ProgressBar(5)

            #with (init_empty_weights() if is_accelerate_available else nullcontext()):
            self.model = instantiate_from_config(config.model).cpu()
            self.model.model.dtype = dtype
            pbar.update(1)
            try:
                print(f"Attempting to load SDXL model from node inputs")
                mm.load_model_gpu(model)
                sdxl_state_dict = model.model.state_dict_for_saving(None, vae.get_sd(), None)
                if is_accelerate_available:
                    for key in sdxl_state_dict:
                        set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=sdxl_state_dict[key])
                else:
                    self.model.load_state_dict(sdxl_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                del sdxl_state_dict
                pbar.update(1)
            except:
                raise Exception("Failed to load SDXL model")            
            gc.collect()
            mm.soft_empty_cache()
            #first clip model from SDXL checkpoint
            try:
                print("Loading first clip model from SDXL checkpoint")
                clip_l_sd = None
                clip_l_model = clip_l.load_model()
                mm.load_model_gpu(clip_l_model)
                clip_l_sd = clip_l.get_sd()
                clip_l_sd = model.model.model_config.process_clip_state_dict_for_saving(clip_l_sd)

                replace_prefix = {}
                replace_prefix["conditioner.embedders.0.transformer."] = ""
    
                clip_l_sd = comfy.utils.state_dict_prefix_replace(clip_l_sd, replace_prefix, filter_keys=True)
                clip_text_config = CLIPTextConfig.from_pretrained(clip_config_path)
                self.model.conditioner.embedders[0].tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
                with (init_empty_weights() if is_accelerate_available else nullcontext()):
                    self.model.conditioner.embedders[0].transformer = CLIPTextModel(clip_text_config)
                if is_accelerate_available:
                    for key in clip_l_sd:
                        set_module_tensor_to_device(self.model.conditioner.embedders[0].transformer, key, device=device, dtype=dtype, value=clip_l_sd[key])
                else:
                    self.model.conditioner.embedders[0].transformer.load_state_dict(clip_l_sd, strict=False)
                self.model.conditioner.embedders[0].eval()
                for param in self.model.conditioner.embedders[0].parameters():
                    param.requires_grad = False
                self.model.conditioner.embedders[0].to(dtype)
                del clip_l_sd
                pbar.update(1)
            except:
                raise Exception("Failed to load first clip model from SDXL checkpoint")
            gc.collect()
            mm.soft_empty_cache()
            #second clip model from SDXL checkpoint
            try:
                print("Loading second clip model from SDXL checkpoint")
                clip_g_sd = None
                clip_g_model = clip_g.load_model()
                mm.load_model_gpu(clip_g_model)
                clip_g_sd = clip_g.get_sd()
                clip_g_sd = model.model.model_config.process_clip_state_dict_for_saving(clip_g_sd)

                replace_prefix2 = {}
                replace_prefix2["conditioner.embedders.1.model."] = ""
                clip_g_sd = comfy.utils.state_dict_prefix_replace(clip_g_sd, replace_prefix2, filter_keys=True)             
                clip_g = build_text_model_from_openai_state_dict(clip_g_sd, device, cast_dtype=dtype)
                self.model.conditioner.embedders[1].model = clip_g
                self.model.conditioner.embedders[1].model.to(dtype)
                del clip_g_sd
                pbar.update(1)
            except:
                raise Exception("Failed to load second clip model from SDXL checkpoint")
            
            try:
                print(f'Attempting to load SUPIR model: [{SUPIR_MODEL_PATH}]')
                supir_state_dict = load_state_dict(SUPIR_MODEL_PATH)
                if "Q" not in supir_model or not is_accelerate_available: #I don't know why this doesn't work with the Q model. 
                    for key in supir_state_dict:
                        set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=supir_state_dict[key])
                else:
                    self.model.load_state_dict(supir_state_dict, strict=False)
                if fp8_unet:
                    self.model.model.to(torch.float8_e4m3fn)
                else:
                    self.model.model.to(dtype)
                del supir_state_dict
                pbar.update(1)
            except:
                raise Exception("Failed to load SUPIR model")
            mm.soft_empty_cache()

        return (self.model, self.model.first_stage_model,)
    
class SUPIR_tiles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            "tile_stride": ("INT", {"default": 256, "min": 64, "max": 8192, "step": 64}),
          
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("image_tiles", "tile_size", "tile_stride",)
    FUNCTION = "tile"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Tiles the image with same function as the Tiled samplers use.  
Useful for previewing the tiling and generating captions per tile (WIP feature)
"""

    def tile(self, image, tile_size, tile_stride):

        def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
            hi_list = list(range(0, h - tile_size + 1, tile_stride))
            if (h - tile_size) % tile_stride != 0:
                hi_list.append(h - tile_size)

            wi_list = list(range(0, w - tile_size + 1, tile_stride))
            if (w - tile_size) % tile_stride != 0:
                wi_list.append(w - tile_size)

            coords = []
            for hi in hi_list:
                for wi in wi_list:
                    coords.append((hi, hi + tile_size, wi, wi + tile_size))
            return coords

        image = image.permute(0, 3, 1, 2)
        _, _, h, w = image.shape

        tiles_iterator = _sliding_windows(h, w, tile_size, tile_stride)

        tiles = []
        for hi, hi_end, wi, wi_end in tiles_iterator:
            tile = image[:, :, hi:hi_end, wi:wi_end]
            
            tiles.append(tile)
        out = torch.cat(tiles, dim=0).to(torch.float32).permute(0, 2, 3, 1)
        print(out.shape)
        print("len(tiles): ", len(tiles))
        
        return (out, tile_size, tile_stride,)
