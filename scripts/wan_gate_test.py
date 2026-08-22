"""AV2-RENDER-GATE-001 — Wan 2.2 fun-control gate tests on the 8 GB RTX 4070 Laptop.

Usage: .venv/bin/python scripts/wan_gate_test.py [--gate A|B|C|D]
Gates: A = smallest ref-image-only render, B = identity+motion transfer,
       C = envelope push (832x480 / 49f / 20 steps), D = 4-step LoRA probe (recorded, not used for fun_control).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from avatar_v2.desktop import DesktopAPI

IDENTITY_FRONT = str(Path.home() / ".avatar_v2/test_assets/identity/identity_front_00001_.png")
IDENTITY_34 = str(Path.home() / ".avatar_v2/test_assets/identity/identity_34_left_00001_.png")
MOTION_WALK = str(Path.home() / ".avatar_v2/test_assets/motion_walk.mp4")

PROMPT = (
    "The woman walks toward the camera with natural weight transfer, arms swinging loosely, "
    "looking forward with a subtle relaxed smile. Photorealistic, documentary style, plain studio background."
)
NEGATIVE = "blurry, distorted, warped anatomy, extra fingers, flicker, low quality, watermark, text"


def payload(label: str, seed: int, *, width: int, height: int, frames: int, steps: int,
            ref_image: str | None, motion: str | None, duration_s: float) -> dict:
    return {
        "runtime_profile": "auto",
        "label": label,
        "avatar": {
            "avatar_id": "wan-gate", "display_name": "Wan Gate Avatar",
            "subject_kind": "synthetic", "age_verified_18_plus": True, "consent_confirmed": True,
            "identity_refs": [ref_image] if ref_image else [],
            "body_refs": [], "hair_refs": [], "wardrobe_refs": [],
            "persistent_features": [],
        },
        "shot": {
            "shot_id": label[:24], "content_class": "general",
            "user_prompt": PROMPT, "negative_prompt": NEGATIVE,
            "duration_s": duration_s, "seed": seed, "engine_preference": "wan22",
            "width": width, "height": height, "fps": 16,
            "camera": {"framing": "full body", "movement": "static tripod"},
            "wind": {"direction": "none", "speed_mps": 0, "gust_variation_pct": 0},
            "wardrobe": "", "environment": "plain studio background",
            "references": {
                "init_image": None, "last_frame": None,
                "motion_video": motion, "scene_image": None, "audio": None,
                "extra_images": [], "extra_videos": [], "extra_audios": [],
            },
        },
        "advanced": {
            "WAN_DIFFUSION_MODEL": "wan2.2_fun_control_5B_bf16.safetensors",
            "WAN_TEXT_ENCODER": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "WAN_VAE": "wan2.2_vae.safetensors",
            "WEIGHT_DTYPE": "default",
            "SAMPLER_NAME": "uni_pc",
            "SCHEDULER": "simple",
            "STEPS": steps, "CFG": 1.0,
            "OUTPUT_PREFIX": label.replace("-", "_"),
            "INIT_IMAGE": ref_image,
        },
    }


def run(api: DesktopAPI, p: dict) -> dict:
    result = api.queue_render(p)
    if not result.get("ok"):
        return {"label": p["label"], "status": "rejected", "error": result.get("error")}
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
                print(f"    detail: {str(item['error'])[:300]}")
            return item
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", default="AB", help="gates to run, e.g. A, AB, ABC")
    args = parser.parse_args()
    api = DesktopAPI()

    if "A" in args.gate:
        print("GATE A: smallest ref-image-only render (480x272, 33f, 8 steps)")
        run(api, payload("wan-gate-a-min", 91001, width=480, height=272, frames=33, steps=8,
                         ref_image=IDENTITY_FRONT, motion=None, duration_s=2))

    if "B" in args.gate:
        print("GATE B: identity + motion transfer (512x512, 33f, 12 steps)")
        run(api, payload("wan-gate-b-idmotion", 91002, width=512, height=512, frames=33, steps=12,
                         ref_image=IDENTITY_FRONT, motion=MOTION_WALK, duration_s=2))

    if "C" in args.gate:
        print("GATE C: envelope push (832x480, 49f, 20 steps)")
        run(api, payload("wan-gate-c-envelope", 91003, width=832, height=480, frames=49, steps=20,
                         ref_image=IDENTITY_FRONT, motion=MOTION_WALK, duration_s=3))

    api.queue.shutdown()


if __name__ == "__main__":
    main()
