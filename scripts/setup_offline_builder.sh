#!/usr/bin/env bash
# Set up the Offline Builder's image model: FLUX.2 [klein] 4B for Avatar V2's "Free builder".
#   - FLUX.2 klein 4B as GGUF Q5_K_M (Apache 2.0, 4-step distilled) through ComfyUI-GGUF
#   - Qwen3-4B text encoder as GGUF Q5_K_M (loaded on CPU to keep VRAM for the UNet)
#   - the FLUX.2 VAE (168 MB)
# Total: about 6 GB. The video side (Wan 2.2 14B GGUF) is installed by scripts/setup_local_engine.sh.
#
# Safe to re-run: finished files are verified and skipped, and partial downloads resume. Every model
# is checked against its published SHA-256. The script stops before any download that would leave
# less than AVATAR_V2_MIN_FREE_GB free. Nothing outside AVATAR_V2_ENGINE_DIR (default ~/AvatarForge)
# is changed.
set -euo pipefail

ENGINE_DIR="${AVATAR_V2_ENGINE_DIR:-$HOME/AvatarForge}"
COMFY="$ENGINE_DIR/comfyui"
MIN_FREE_GB="${AVATAR_V2_MIN_FREE_GB:-15}"
HF="https://huggingface.co"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
free_gb() { df -Pk "$ENGINE_DIR" | awk 'NR==2 {printf "%d", $4/1048576}'; }

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
if [ ! -f "$COMFY/main.py" ]; then
    echo "ComfyUI is not installed yet. Run scripts/setup_local_engine.sh first." >&2
    exit 2
fi
mkdir -p "$ENGINE_DIR"
say "Engine folder: $ENGINE_DIR ($(free_gb) GB free)"

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

# FLUX.2 [klein] 4B distilled UNet, GGUF Q5_K_M (unsloth dynamic quant).
fetch diffusion_models flux-2-klein-4b-Q5_K_M.gguf \
    "$HF/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q5_K_M.gguf" \
    3073368640 58c01c75fee2272eadb127f53e26dd05a5a8e7f812e37c36fa44603301f91e54

# Qwen3-4B text encoder, GGUF Q5_K_M (runs on CPU; the UNet keeps the VRAM).
fetch text_encoders Qwen3-4B-Q5_K_M.gguf \
    "$HF/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q5_K_M.gguf" \
    2889513184 aca596860e8cb40af6539e3f2ea40df305f42515deac56d49c08d39a02e6533f

# Qwen3-8B text encoder, GGUF Q4_K_M: the 9B Base model's encoder (the 4B's Qwen3-4B
# conditioning does not fit the 9B's joint attention).
fetch text_encoders Qwen3-8B-Q4_K_M.gguf \
    "$HF/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" \
    5027783488 d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785

# FLUX.2 VAE (saved under the name the offline builder expects).
fetch vae flux2-vae.safetensors \
    "$HF/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors" \
    168120878 ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04

# Wan 2.2 TI2V 5B Turbo (Apache 2.0): the faster 4-step video option for scenes.
fetch diffusion_models Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf \
    "$HF/Kiijoku/Wan2.2-TI2V-5B-Turbo-GGUF/resolve/main/Wan2_2-TI2V-5B-Turbo-Q5_K_M.gguf" \
    3815414496 fb39b4a9fc545c26b04cfcc1a0897a83494fd48cc9291bec856665f045e2cc45

# The 48-channel Wan 2.2 VAE the 5B Turbo decodes with (the A14B GGUF is packed for the
# 16-channel Wan 2.1 VAE instead).
fetch vae Wan2.2_VAE.safetensors \
    "$HF/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors" \
    1409400960 e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156

# FLUX.2 [klein] 9B Base, GGUF Q4_K_M (optional quality tier). NOTE: this model is under the
# FLUX Non-Commercial License, so it is for testing and research — the 4B above is the
# commercial option.
if [ "${AVATAR_V2_FORGE_9B:-1}" != "0" ]; then
    fetch diffusion_models flux-2-klein-base-9b-Q4_K_M.gguf \
        "$HF/unsloth/FLUX.2-klein-base-9B-GGUF/resolve/main/flux-2-klein-base-9b-Q4_K_M.gguf" \
        5908912416 e251d3b2dcd7a2b954754763252a388f0c8fcb992fc5b243d041aa97a3c81668
else
    say "Skipping the 9B model (AVATAR_V2_FORGE_9B=0)."
fi

say "Offline Builder models are ready (FLUX.2 klein 4B/9B + Wan 5B Turbo)."
