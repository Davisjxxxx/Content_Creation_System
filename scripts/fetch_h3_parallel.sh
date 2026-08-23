#!/usr/bin/env bash
# Parallel range downloader for large HF files (4 connections per file).
set -euo pipefail

TOKEN=$(cat /home/jd/.cache/huggingface/token)
BASE="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
DEST="/home/jd/AvatarForge/comfyui/models"
PARTS=4
TMP=/tmp/h3_parts
mkdir -p "$TMP"

fetch() {
    local url="$1" out="$2"
    local size
    size=$(curl -sIL -H "Authorization: Bearer $TOKEN" "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
    [ -n "$size" ] || { echo "FAIL size $url"; exit 1; }
    local chunk=$(( (size + PARTS - 1) / PARTS ))
    local pids=()
    for i in $(seq 0 $((PARTS - 1))); do
        local start=$(( i * chunk ))
        local end=$(( (i + 1) * chunk - 1 ))
        [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        [ "$start" -ge "$size" ] && continue
        curl -sL -H "Authorization: Bearer $TOKEN" -r "${start}-${end}" -o "$TMP/part_$i" "$url" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    rm -f "$out"
    for i in $(seq 0 $((PARTS - 1))); do
        [ -f "$TMP/part_$i" ] && cat "$TMP/part_$i" >> "$out"
    done
    rm -f "$TMP"/part_*
    local got
    got=$(stat -c%s "$out")
    echo "fetched $out : $got / $size bytes"
    [ "$got" -eq "$size" ] || { echo "SIZE MISMATCH $out"; exit 1; }
}

fetch "$BASE/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$DEST/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
fetch "$BASE/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$DEST/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
fetch "$BASE/vae/minimax_h3_video_vae_fp16.safetensors" "$DEST/vae/minimax_h3_video_vae_fp16.safetensors"
fetch "$BASE/vae/minimax_h3_audio_vae_fp32.safetensors" "$DEST/vae/minimax_h3_audio_vae_fp32.safetensors"
fetch "$BASE/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" "$DEST/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
fetch "$BASE/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" "$DEST/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
echo H3_DOWNLOADS_DONE
