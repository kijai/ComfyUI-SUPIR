# ComfyUI SUPIR upscaler wrapper node
# UPDATE3:
Pruned models in safetensors format now available here:
https://huggingface.co/Kijai/SUPIR_pruned/tree/main
# UPDATE2:
![image](https://github.com/kijai/ComfyUI-SUPIR/assets/40791699/65baec3e-cb4a-4eec-8d45-2b08157b1e86)
Added a better way to load the SDXL model, which also allows using LoRAs. The old node will remain for now to not break old workflows, and it is dubbed Legacy along with the single node, as I do not want to maintain those.

# UPDATE:

As I have learned a lot with this project, I have now separated the single node to multiple nodes that make more sense to use in ComfyUI, and makes it clearer how SUPIR works. This is still a wrapper, though the whole thing has deviated from the original with much wider hardware support, more efficient model loading, far less memory usage and more sampler options. Here's a quick example (workflow is included) of using a Ligntning model, quality suffers then but it's very fast and I recommend starting with it as faster sampling makes it a lot easier to learn what the settings do.

Under the hood SUPIR is SDXL img2img pipeline, the biggest custom part being their ControlNet. What they call "first stage" is a denoising process using their special "denoise encoder" VAE. This is not to be confused with the Gradio demo's "first stage" that's labeled as such for the Llava preprocessing, the Gradio "Stage2" still runs the denoising process anyway. This can be fully skipped with the nodes, or replaced with any other preprocessing node such as a model upscaler or anything you want.

https://github.com/kijai/ComfyUI-SUPIR/assets/40791699/5cae2a24-d425-462c-b89d-df7dcf01595c



# Installing
Either manager and install from git, or clone this repo to custom_nodes and run:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-SUPIR\requirements.txt`

Pytorch version should be pretty new too, latest stable (2.2.1) works.

`xformers` is automatically detected and enabled if found, but it's not necessary, in some cases it can be a bit faster though:

`pip install -U xformers --no-dependencies`  (for portable `python_embeded\python.exe -m pip install -U xformers --no-dependencies` )

Get the SUPIR model(s) from the original links below, they are loaded from the normal `ComfyUI/models/checkpoints` -folder
In addition you need an SDXL model, they are loaded from the same folder.

I have not included llava in this, but you can input any captions to the node and thus use anything you want to generate them, or just don't, seems to work great even without.

Memory requirements are directly related to the input image resolution, the "scale_by" in the node simply scales the input, you can leave it at 1.0 and size your input with any other node as well. In my testing I was able to run 512x512 to 1024x1024 with a 10GB 3080 GPU, and other tests on 24GB GPU to up 3072x3072. System RAM requirements are also hefty, don't know numbers but I would guess under 32GB is going to have issues, tested with 64GB.

## Updates: 
- fp8 seems to work fine for the unet, I was able to do 512p to 2048 with under 10GB VRAM used. For the VAE it seems to cause artifacts, I recommend using tiled_vae instead.
- CLIP models are no longer needed separately, instead they are loaded from your selected SDXL checkpoint
______
Mirror for the models: https://huggingface.co/camenduru/SUPIR/tree/main

# Tests
Video upscale test (currently the node does frames one by one from input batch):

Original: https://github.com/kijai/ComfyUI-SUPIR/assets/40791699/33621520-a429-4155-aa3a-ac5cd15bda56

Upscaled 3x: https://github.com/kijai/ComfyUI-SUPIR/assets/40791699/d6c60e0a-11c3-496d-82c6-a724758a131a

Image upscale from 3x from 512p:
https://github.com/kijai/ComfyUI-SUPIR/assets/40791699/545ddce4-8324-45cb-a545-6d1f527d8750



-------------------------------------------


Original repo:
https://github.com/Fanghua-Yu/SUPIR

#### Models we provided:
* `SUPIR-v0Q`: [Baidu Netdisk](https://pan.baidu.com/s/1lnefCZhBTeDWijqbj1jIyw?pwd=pjq6), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)
    
    Default training settings with paper. High generalization and high image quality in most cases.

* `SUPIR-v0F`: [Baidu Netdisk](https://pan.baidu.com/s/1AECN8NjiVuE3hvO8o-Ua6A?pwd=k2uz), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)

    Training with light degradation settings. Stage1 encoder of `SUPIR-v0F` remains more details when facing light degradations.


## BibTeX
    @misc{yu2024scaling,
      title={Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild}, 
      author={Fanghua Yu and Jinjin Gu and Zheyuan Li and Jinfan Hu and Xiangtao Kong and Xintao Wang and Jingwen He and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2401.13627},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

---

## 📧 Contact
If you have any question, please email `fanghuayu96@gmail.com`.

---
## Non-Commercial Use Only Declaration
The SUPIR ("Software") is made available for use, reproduction, and distribution strictly for non-commercial purposes. For the purposes of this declaration, "non-commercial" is defined as not primarily intended for or directed towards commercial advantage or monetary compensation.

By using, reproducing, or distributing the Software, you agree to abide by this restriction and not to use the Software for any commercial purposes without obtaining prior written permission from Dr. Jinjin Gu.

This declaration does not in any way limit the rights under any open source license that may apply to the Software; it solely adds a condition that the Software shall not be used for commercial purposes.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For inquiries or to obtain permission for commercial use, please contact Dr. Jinjin Gu (hellojasongt@gmail.com).
