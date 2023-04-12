Clone `https://github.com/AUTOMATIC1111/stable-diffusion-webui`

Edit `webui-user.sh` and set `install_dir="/workspace"`.

Example launch:

```sh
CUDA_VISIBLE_DEVICES=0 python3 launch.py --xformers --nowebui --ckpt models/Stable-diffusion/DreamShaper_5_beta2_BakedVae.safetensors
```
