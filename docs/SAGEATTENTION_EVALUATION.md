# Isolated Attention Backend & SageAttention Compatibility Evaluation

**Evaluation Date:** 2026-11-25  
**Environment Audited (Read-Only — Production Venv Unmodified):** `/home/jd/AvatarForge/venv`

## Verified Hardware & Software Facts

| Component | Verified Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (`sm_89`, Ada Lovelace, Compute Capability `8.9`) |
| Driver | `580.178.04` |
| ComfyUI Venv Python | `3.12.3` |
| PyTorch | `2.14.1+cu130` (CUDA `13.0`, cuDNN `92400`) |
| Triton | `3.8.0` (installed) |
| `comfy-kitchen` | `0.2.42` (installed) — **`int8_attention_is_available() == True`**, **`sol_attn_is_available() == True`** |
| Host `nvcc` | `/usr/local/cuda-12.1/bin/nvcc` (**CUDA 12.1 — mismatch with PyTorch `cu130`!**) |
| `sageattention` | Not installed |
| `flash_attn` / `xformers` / `spargeattn` | Not installed |

## Compatibility Conclusions

1. **PyTorch SDPA (`Using pytorch attention`):** Currently active, stable, and lossless on `sm_89`. Remains the verified baseline.
2. **SageAttention 2.x (CUDA C++ Kernels):** **Blocked from source compilation** in the current host configuration because host `nvcc` is CUDA `12.1` while `/home/jd/AvatarForge/venv` runs PyTorch `2.14.1+cu130` (CUDA `13.0`). Installing SageAttention 2.x requires either a matching prebuilt `cp312` + `cu130` wheel or installing CUDA 13.0 toolkit headers before building—never attempted against the production venv.
3. **SageAttention 1.0.6 (Pure-Python + Triton Wheel):** PyPI hosts `sageattention-1.0.6-py3-none-any.whl`, which uses Triton kernels (compatible with installed `triton 3.8.0` and `sm_89`) and requires zero `nvcc` compilation. ComfyUI `0.46.0` natively supports `--use-sage-attention` in `comfy/ldm/modules/attention.py:28-38, 715-745` with automatic fallback to `attention_pytorch` when `low_precision_attention=False` or masks are unsupported.
4. **`comfy-kitchen 0.2.42` Native INT8 / SOL Attention:** Already installed in `/home/jd/AvatarForge/venv` and verified available (`True`) on RTX 4070 `sm_89`. ComfyUI `0.46.0` activates it automatically for checkpoints carrying `comfy_kitchen_int8` / `comfy_kitchen_sol` metadata.

## Sprint 2 Validation Gate (Before Any Production Activation)

- Do **not** pass `--use-sage-attention` in production launch args yet.
- In Sprint 2 (once GPU is idle), test `sageattention==1.0.6` in a cloned/isolated test venv (or verified prebuilt SageAttention 2 `cu130` wheel if available) against Wan 2.2 A14B and FLUX.2 9B over ≥ 3 repeated trials, verifying:
  - Zero NaN/black frames or facial detail degradation (fixed-seed SSIM/PSNR + visual review),
  - Measurable latency speedup (target ≥ 15% on Wan 2.2 14B) and peak VRAM parity/reduction,
  - Clean fallback on unsupported mask/head-dim paths.
- Run `scripts/eval_attention_compat.py` at any time for a non-destructive re-check.
