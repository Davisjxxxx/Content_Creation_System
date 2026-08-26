#!/usr/bin/env bash
# Download MiniMax H3 weights into the ComfyUI model dirs.
# Requires: `hf auth login` as a user who has ACCEPTED the MiniMax H3 license
# on the model page (https://huggingface.co/Comfy-Org/MiniMax_H3_ComfyUI_Repackaged).
set -euo pipefail

HF_REPO="Comfy-Org/MiniMax_H3_ComfyUI_Repackaged"
MODELS_DIR="/home/jd/AvatarForge/comfyui/models"

mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/clip"

hf download "$HF_REPO" --include "split_files/diffusion_models/*" --local-dir "$MODELS_DIR/diffusion_models/h3_repo"
hf download "$HF_REPO" --include "split_files/text_encoders/*" --local-dir "$MODELS_DIR/text_encoders/h3_repo"
hf download "$HF_REPO" --include "split_files/vae/*" --local-dir "$MODELS_DIR/vae/h3_repo"
hf download "$HF_REPO" --include "split_files/clip/*" --local-dir "$MODELS_DIR/clip/h3_repo" || true

# Symlink downloaded files into the flat model dirs ComfyUI scans.
find "$MODELS_DIR/diffusion_models/h3_repo" -name "*.safetensors" -exec ln -sf {} "$MODELS_DIR/diffusion_models/" \;
find "$MODELS_DIR/text_encoders/h3_repo" -name "*.safetensors" -exec ln -sf {} "$MODELS_DIR/text_encoders/" \;
find "$MODELS_DIR/vae/h3_repo" -name "*.safetensors" -exec ln -sf {} "$MODELS_DIR/vae/" \;
find "$MODELS_DIR/clip/h3_repo" -name "*.safetensors" -exec ln -sf {} "$MODELS_DIR/clip/" \; || true

echo "H3 files staged. Restart ComfyUI to rescan."
