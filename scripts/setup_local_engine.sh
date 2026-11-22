#!/usr/bin/env bash
# Set up Avatar V2's local engine: ComfyUI + Wan 2.2 14B image-to-video, sized for an 8 GB NVIDIA GPU.
#   - Wan 2.2 I2V A14B as GGUF Q4_K_S (high- and low-noise experts), loaded through ComfyUI-GGUF
#   - the official 4-step Lightning LoRAs (lightx2v), the UMT5-XXL fp8 text encoder, the Wan 2.1 VAE
# Safe to re-run: finished files are verified and skipped, and partial downloads resume. Every model
# is checked against its published SHA-256. The script stops before any download that would leave
# less than AVATAR_V2_MIN_FREE_GB free. Nothing outside AVATAR_V2_ENGINE_DIR (default ~/AvatarForge)
# is changed. Total: about 27 GB of models plus about 7 GB for ComfyUI and PyTorch.
set -euo pipefail

ENGINE_DIR="${AVATAR_V2_ENGINE_DIR:-$HOME/AvatarForge}"
COMFY="$ENGINE_DIR/comfyui"
VENV="$ENGINE_DIR/venv"
BASE_PYTHON="${AVATAR_V2_ENGINE_PYTHON:-python3.12}"
TORCH_INDEX="${AVATAR_V2_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_PINS="${AVATAR_V2_TORCH_PINS:-torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0}"
MIN_FREE_GB="${AVATAR_V2_MIN_FREE_GB:-15}"
HF="https://huggingface.co"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
free_gb() { df -Pk "$ENGINE_DIR" | awk 'NR==2 {printf "%d", $4/1048576}'; }

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v curl >/dev/null || { echo "curl is required"; exit 1; }
mkdir -p "$ENGINE_DIR"
say "Engine folder: $ENGINE_DIR ($(free_gb) GB free)"

if [ ! -f "$COMFY/main.py" ]; then
    say "Cloning ComfyUI"
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
if [ ! -d "$COMFY/custom_nodes/ComfyUI-GGUF" ]; then
    say "Cloning ComfyUI-GGUF"
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "$COMFY/custom_nodes/ComfyUI-GGUF"
fi
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating the Python environment ($BASE_PYTHON)"
    uv venv --python "$BASE_PYTHON" "$VENV"
fi
say "Installing PyTorch with CUDA ($TORCH_PINS)"
# shellcheck disable=SC2086
uv pip install --python "$VENV/bin/python" $TORCH_PINS --index-url "$TORCH_INDEX"
say "Installing ComfyUI and ComfyUI-GGUF requirements"
uv pip install --python "$VENV/bin/python" \
    -r "$COMFY/requirements.txt" -r "$COMFY/custom_nodes/ComfyUI-GGUF/requirements.txt"

fetch() {  # fetch <models subfolder> <file name> <url> <bytes> <sha256>
    local dir="$COMFY/models/$1" name="$2" url="$3" bytes="$4" sha="$5"
    local dest="$dir/$name" part="$dir/$name.part"
    mkdir -p "$dir"
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ]; then
        say "Already present: $name"
        return
    fi
    local have=0
    [ -f "$part" ] && have=$(stat -c %s "$part")
    if [ "$have" != "$bytes" ]; then
        local need_gb=$(( (bytes - have) / 1073741824 + 1 ))
        if [ $(( $(free_gb) - need_gb )) -lt "$MIN_FREE_GB" ]; then
            say "STOP: $name needs about ${need_gb} GB and would leave less than ${MIN_FREE_GB} GB free."
            exit 3
        fi
        say "Downloading $name ($(( bytes / 1048576 )) MB)"
        curl -L --fail --retry 5 --retry-delay 5 -C - -o "$part" "$url"
    fi
    say "Verifying $name"
    if ! echo "$sha  $part" | sha256sum -c --quiet -; then
        say "Checksum mismatch for $name; the bad download was removed. Run the script again."
        rm -f "$part"
        exit 4
    fi
    mv "$part" "$dest"
}

fetch diffusion_models Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf \
    "$HF/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf" \
    8748151296 8fc5f726ccf1214e43c6a599bdef1e63daf7fab06e71fb74081416c4b9ce200b
fetch diffusion_models Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf \
    "$HF/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf" \
    8748151296 54e2313ef7fbb169e72ab1d7cecdc90a04db01097c86f9f39097b8caacea50fb
fetch text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    "$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    6735906897 c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
fetch vae wan_2.1_vae.safetensors \
    "$HF/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    253815318 2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b
fetch loras wan2.2_i2v_lightning_4step_high.safetensors \
    "$HF/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
    1226977424 d176c808d6fc461999b68e321efcb7501b20b8c3797523ed0df14f7d1deff11e
fetch loras wan2.2_i2v_lightning_4step_low.safetensors \
    "$HF/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" \
    1226977424 024f21de095bc8fad9809ded3e9e49a2e170dcf27075da8145ba7d60d8aab7f9

mkdir -p "$ENGINE_DIR/reference"
curl -sL --fail -o "$ENGINE_DIR/reference/wan22_i2v_lightning_native_comfy.json" \
    "$HF/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-NativeComfy.json" \
    || say "Reference workflow download skipped (optional)."

say "Checking CUDA from the engine's PyTorch"
"$VENV/bin/python" - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} · CUDA runtime {torch.version.cuda} · GPU available: {ok}")
if ok:
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name(0)} · {total / 2**30:.1f} GB VRAM ({free / 2**30:.1f} GB free now)")
PY
touch "$ENGINE_DIR/.avatar_v2_engine_ready"
say "Local engine ready ($(free_gb) GB free). Avatar V2 starts ComfyUI automatically when it opens."
