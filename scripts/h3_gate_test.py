"""AV2-H3-GATE — MiniMax H3 ladder on the 8 GB RTX 4070 Laptop.

Usage: .venv/bin/python scripts/h3_gate_test.py [--gate A|B|C|D|E|F]
A = FL2VA smallest (352x608, 8 steps, no turbo)   B = FL2VA 480x864, 8 steps
C = FL2VA 480x864 + turbo 4-step LoRA             D = FL2VA low-memory-quality profile
E = Ref2VA one identity reference                 F = Ref2VA identity + motion
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from avatar_v2.desktop import DesktopAPI

IDENTITY_FRONT = str(Path.home() / ".avatar_v2/test_assets/identity/identity_front_00001_.png")
MOTION_WALK = str(Path.home() / ".avatar_v2/test_assets/motion_832x480.mp4")

PROMPT = (
    "A woman with deep auburn hair in loose waves looks toward the camera and gives a subtle "
    "relaxed smile. Natural micro-movements, realistic breathing, photorealistic, neutral studio background."
)
NEGATIVE = "blurry, distorted, warped anatomy, flicker, low quality, distorted eyes, crossed eyes"


def payload(label: str, seed: int, *, engine: str, width: int, height: int, duration_s: float,
            steps: int, ref_image: str | None, motion: str | None, lora: str | None = None) -> dict:
    preset = "vertical_fast" if height <= 608 else "vertical_4070"
    return {
        "runtime_profile": "auto",
        "h3_preset": preset,
        "label": label,
        "avatar": {
            "avatar_id": "h3-gate", "display_name": "H3 Gate Avatar",
            "subject_kind": "synthetic", "age_verified_18_plus": True, "consent_confirmed": True,
            "identity_refs": [ref_image] if ref_image else [],
            "body_refs": [], "hair_refs": [], "wardrobe_refs": [],
            "persistent_features": [],
        },
        "shot": {
            "shot_id": label[:24], "content_class": "general",
            "user_prompt": PROMPT, "negative_prompt": NEGATIVE,
            "duration_s": duration_s, "seed": seed, "engine_preference": engine,
            "width": width, "height": height, "fps": 24,
            "camera": {"framing": "medium portrait", "movement": "subtle handheld"},
            "wind": {"direction": "none", "speed_mps": 0, "gust_variation_pct": 0},
            "wardrobe": "", "environment": "neutral studio background",
            "references": {
                "init_image": None, "last_frame": None,
                "motion_video": motion, "scene_image": None, "audio": None,
                "extra_images": [], "extra_videos": [], "extra_audios": [],
            },
        },
        "advanced": {
            "STEPS": steps, "CFG": 1.0,
            "OUTPUT_PREFIX": label.replace("-", "_"),
            "H3_DIFFUSION_MODEL": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
            if engine == "h3_fl2va" else "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "H3_TEXT_ENCODER": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "H3_VAE": "minimax_h3_video_vae_fp16.safetensors",
            "H3_AUDIO_VAE": "minimax_h3_audio_vae_fp32.safetensors",
            "INIT_IMAGE": ref_image,
        },
    }


def run(api: DesktopAPI, p: dict) -> None:
    result = api.queue_render(p)
    if not result.get("ok"):
        print(f"[{p['label']}] REJECTED: {result.get('error')}")
        return
    qid = result["queue_id"]
    while True:
        item = next((i for i in api.queue_state()["items"] if i["id"] == qid), None)
        if item and item["status"] in ("done", "error", "cancelled"):
            peak = (item.get("gpu_trace") or {}).get("peak") or []
            peak_s = " / ".join(f"{x['used_mb']}MB ({x['usage']*100:.0f}%)" for x in peak)
            print(f"[{item['label']}] {item['status']} elapsed={item['elapsed_s']:.0f}s "
                  f"peak_vram={peak_s} frames={item['frames']} res={item['resolution']}")
            if item.get("error_message"):
                print(f"    {item['error_message']}")
            if item.get("error"):
                print(f"    detail: {str(item['error'])[:400]}")
            return
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", default="A", help="gates to run, e.g. A, AB, ABCDEF")
    args = parser.parse_args()
    api = DesktopAPI()

    if "A" in args.gate:
        print("GATE A: FL2VA smallest (352x608, 2s, 8 steps)")
        run(api, payload("h3-gate-a-fl2va-min", 94001, engine="h3_fl2va", width=352, height=608,
                         duration_s=2, steps=8, ref_image=None, motion=None))
    if "B" in args.gate:
        print("GATE B: FL2VA 480x864, 2s, 8 steps")
        run(api, payload("h3-gate-b-fl2va-4070", 94002, engine="h3_fl2va", width=480, height=864,
                         duration_s=2, steps=8, ref_image=None, motion=None))
    if "C" in args.gate:
        print("GATE C: FL2VA 480x864 + turbo 4-step LoRA")
        p = payload("h3-gate-c-fl2va-turbo", 94003, engine="h3_fl2va", width=480, height=864,
                    duration_s=2, steps=4, ref_image=None, motion=None)
        p["advanced"]["H3_LORA"] = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
        run(api, p)
    if "D" in args.gate:
        print("GATE D: FL2VA low-memory-quality profile (480x864, 16 steps, forced cleanup)")
        p = payload("h3-gate-d-fl2va-lmq", 94004, engine="h3_fl2va", width=480, height=864,
                    duration_s=2, steps=16, ref_image=None, motion=None)
        p["runtime_profile"] = "low_memory_quality"
        run(api, p)
    if "E" in args.gate:
        print("GATE E: Ref2VA one identity reference (352x608, 2s, 8 steps)")
        run(api, payload("h3-gate-e-ref2va-id", 94005, engine="h3_ref2va", width=352, height=608,
                         duration_s=2, steps=8, ref_image=IDENTITY_FRONT, motion=None))
    if "F" in args.gate:
        print("GATE F: Ref2VA identity + motion (352x608, 2s, 8 steps)")
        run(api, payload("h3-gate-f-ref2va-idmotion", 94006, engine="h3_ref2va", width=352, height=608,
                         duration_s=2, steps=8, ref_image=IDENTITY_FRONT, motion=MOTION_WALK))

    api.queue.shutdown()


if __name__ == "__main__":
    main()
