#!/usr/bin/env bash
# Parallel range downloader: 4 connections per file, stall-detecting retries,
# completion guard. No mid-transfer resume (curl -C - + -r corrupts offsets).
set -u

TOKEN=$(cat /home/jd/.cache/huggingface/token)
BASE="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
DEST="/home/jd/AvatarForge/comfyui/models"
PARTS=4
TMP=/tmp/h3_parts
mkdir -p "$TMP"

CURL_OPTS=(--connect-timeout 20 --speed-limit 200000 --speed-time 25 --retry 8 --retry-delay 5)

fetch() {
    local url="$1" out="$2" stem="$3"
    local size
    size=$(curl -sIL -H "Authorization: Bearer $TOKEN" "$url" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
    [ -n "$size" ] || { echo "FAIL size $url"; exit 1; }

    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -eq "$size" ]; then
        echo "skip (complete) $out"
        return
    fi
    rm -f "$out"

    local chunk=$(( (size + PARTS - 1) / PARTS ))
    local pids=()
    for i in $(seq 0 $((PARTS - 1))); do
        local start=$(( i * chunk ))
        local end=$(( (i + 1) * chunk - 1 ))
        [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        [ "$start" -ge "$size" ] && break
        local want=$(( end - start + 1 ))
        (
            local part="$TMP/${stem}_part_$i"
            while true; do
                rm -f "$part"
                curl -sL "${CURL_OPTS[@]}" -H "Authorization: Bearer $TOKEN" \
                    -r "${start}-${end}" -o "$part" "$url" || true
                local got=0
                [ -f "$part" ] && got=$(stat -c%s "$part")
                if [ "$got" -eq "$want" ]; then
                    break
                fi
                echo "part $stem/$i incomplete ($got/$want), retrying"
                sleep 2
            done
        ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done

    for i in $(seq 0 $((PARTS - 1))); do
        [ -f "$TMP/${stem}_part_$i" ] && cat "$TMP/${stem}_part_$i" >> "$out"
    done
    rm -f "$TMP"/${stem}_part_*
    local got
    got=$(stat -c%s "$out")
    echo "fetched $out : $got / $size bytes"
    [ "$got" -eq "$size" ] || { echo "SIZE MISMATCH $out"; exit 1; }
}

fetch "$BASE/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$DEST/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" fl2va
fetch "$BASE/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$DEST/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" qwen
fetch "$BASE/vae/minimax_h3_video_vae_fp16.safetensors" "$DEST/vae/minimax_h3_video_vae_fp16.safetensors" vvae
fetch "$BASE/vae/minimax_h3_audio_vae_fp32.safetensors" "$DEST/vae/minimax_h3_audio_vae_fp32.safetensors" avae
fetch "$BASE/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" "$DEST/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" lora4
fetch "$BASE/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" "$DEST/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" lora8
echo H3_DOWNLOADS_DONE
