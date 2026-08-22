# Machine Manifest — JD-ROG-STRIX (Avatar V2 known-good stack)

Frozen 2026-08-22 after the first proven end-to-end desktop render gate.
This stack demonstrably renders (AnimateDiff SD1.5/SDXL sweeps, all PASS).

## Machine

- Host: JD-ROG-STRIX, Ubuntu 24.04.4 LTS (Noble), kernel 7.0.0-28-generic
- Session: X11 / GNOME (ubuntu:GNOME)
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, **8188 MiB** (8 GB), UUID GPU-7a726e7d-931e-1eb8-fe43-731728ab25ec
- Driver: **580.173.02** · CUDA compatibility reported by nvidia-smi: **13.0**
- RAM: 62 GiB · Swap: 8 GiB · Disk: 158 GB free on /
- ffmpeg/ffprobe 6.1.1

## Python environments

| Venv | Path | Python | Key packages |
|---|---|---|---|
| App venv | `/home/jd/avatar_project/.venv` | 3.12.3 (system, `--system-site-packages`) | pywebview 5.4, pydantic 2.13.4, httpx 0.28.1, typer 0.27.1, pytest 8.4.2, PyInstaller 6.22.2, Pillow 11.3.0 |
| ComfyUI venv | `/home/jd/AvatarForge/venv` | 3.11.15 | **torch 2.13.0+cu130**, torchvision 2.24.0, transformers 4.53.1, opencv-python-headless 5.0.0, av 16.x, imageio-ffmpeg |

## ComfyUI service

- Location: `/home/jd/AvatarForge/comfyui`
- Core: **v0.33.0** @ commit `7dde5617` ("Increase trellis2 memory factor a bit. (#15796)")
- Start (managed by Avatar V2 or manually):
  `cd /home/jd/AvatarForge/comfyui && venv/bin/python main.py --listen 127.0.0.1 --port 8188`
- Run flags for 8 GB VRAM: `--fp8_e4m3fn-unet` (on-the-fly fp8 UNet weights)
- Endpoints verified: /system_stats, /object_info (1071 nodes), /prompt, /history, /interrupt, /free, /upload/image

### Custom nodes (git SHAs)

| Node pack | SHA |
|---|---|
| AnimateDiff-Evolved (`custom_nodes/AnimateDiff-Evolved`, also `ComfyUI-AnimateDiff-Evolved`) | `9257651` Merge pull request #579 from Kos |
| VideoHelperSuite (`custom_nodes/ComfyUI-VideoHelperSuite`, also `VideoHelperSuite`) | `4ee72c0` fix(metadata): stop double-stringifying |

## Installed models (sha256)

ComfyUI `models/` at `/home/jd/AvatarForge/comfyui/models/`:

| File | Size | sha256 |
|---|---|---|
| checkpoints/sd_xl_base_1.0.safetensors | 6.9 GB | `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b` |
| checkpoints/Deliberate_v6.safetensors | ~2.1 GB | `bcce73a08e95a4d4a3332875342792a50ab9bf9f07fdedcab49fd425fac173be` |
| checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors | ~7.0 GB | `c9e3e68f89b8e38689e1097d4be4573cf308de4e3fd044c64ca697bdb4aa8bca` |
| checkpoints/v1-5-pruned-emaonly.safetensors | ~4.3 GB | `6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa` |
| animatediff_models/mm_sd_v15_v2.ckpt | 1.8 GB | `69ed0f5fef82b110aca51bcab73b21104242bc65d6ab4b8b2a2a94d31cad1bf0` |
| animatediff_models/mm_sdxl_v10_beta.ckpt | 950 MB | `fa4950a062e892fca50d4c441fcd6130d1ad68a621a0404d155be17580072978` |
| vae/sdxl_vae.safetensors | — | `235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1` |
| vae/vae-ft-mse-840000-ema-pruned.safetensors | 334 MB | `735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75` |

Checkpoints `Deliberate_v6`, `Juggernaut-XL_v9`, `v1-5-pruned-emaonly` and `sdxl_vae` are symlinks into the preserved legacy tree at `/home/jd/avatar_project_legacy/`.

## Wan 2.2 additions (2026-08-22, in progress)

Downloaded from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (split_files):

| File | Target dir | sha256 |
|---|---|---|
| wan2.2_fun_control_5B_bf16.safetensors (10.0 GB) | diffusion_models/ | pending download |
| umt5_xxl_fp8_e4m3fn_scaled.safetensors (6.74 GB) | text_encoders/ | pending download |
| wan2.2_vae.safetensors (1.41 GB) | vae/ | pending download |
| wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors (1.23 GB) | loras/ | pending download |
| wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors (1.23 GB) | loras/ | pending download |

## Proven render envelope (2026-08-21, RTX 4070 Laptop 8 GB)

| Config | Peak VRAM | Result |
|---|---|---|
| AnimateDiff SD1.5 512²×16f×8s | 6161 MB (75%) | PASS |
| AnimateDiff SD1.5 768²×16f×8s | 5404 MB (66%) | PASS |
| AnimateDiff SD1.5 864×480×24f×8s | 5200 MB (64%) | PASS |
| AnimateDiff SDXL 512²×16f×8s | 6579 MB (80%) | PASS |
| ComfyUI /free manual cleanup | 2358 → 1025 MB | PASS |
