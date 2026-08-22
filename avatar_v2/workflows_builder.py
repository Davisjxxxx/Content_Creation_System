from __future__ import annotations

from typing import Any

# Dynamic API-format workflow builders for the renderers installed in the
# current ComfyUI core (0.33+): MiniMax H3 Ref2VA / FL2VA and Wan 2.2 FLF2V.
#
# These graphs were written from the live /object_info schemas of this
# ComfyUI build. Static placeholder JSON cannot express a variable number of
# H3 reference slots, so the builders add exactly the LoadImage / LoadVideo /
# LoadAudio nodes the job actually uses. Values come from the STAGED asset
# map (files already uploaded/copied into ComfyUI input).


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _link(node_id: int | str, output: int = 0) -> list[Any]:
    return [str(node_id), output]


def _stage_entry(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _value(mapping: dict[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key)
    return value if value not in (None, "") else default


WAN_FPS = 16
WAN_FRAME_STEP = 4
WAN_FRAME_OFFSET = 1
WAN_MAX_FRAMES = 481
WAN_CANVAS_MULTIPLE = 32


def wan_frame_count(duration_s: float) -> int:
    """Snap a duration to a valid Wan 2.2 frame count (4k+1 grid, 16 fps)."""
    requested = max(WAN_FRAME_OFFSET, round(float(duration_s) * WAN_FPS))
    snapped = requested + (WAN_FRAME_OFFSET - (requested % WAN_FRAME_STEP)) % WAN_FRAME_STEP
    return min(snapped, WAN_MAX_FRAMES)


def snap_wan_canvas(width: int, height: int) -> tuple[int, int]:
    """Snap to multiples of 32 — Wan 2.2 RoPE fails on odd latent dims otherwise."""
    width = max(WAN_CANVAS_MULTIPLE, round(width / WAN_CANVAS_MULTIPLE) * WAN_CANVAS_MULTIPLE)
    height = max(WAN_CANVAS_MULTIPLE, round(height / WAN_CANVAS_MULTIPLE) * WAN_CANVAS_MULTIPLE)
    return width, height


def build_h3_common(
    mapping: dict[str, Any],
    task_node: str,
    task_inputs: dict[str, Any],
    task_slots: dict[str, list[str]],
) -> dict[str, Any]:
    """Shared H3 scaffold: loaders, sigma shift, conditioning, sampler, decode."""
    graph: dict[str, Any] = {}
    next_id = 1

    graph[str(next_id)] = _node(
        "UNETLoader",
        unet_name=_value(mapping, "H3_DIFFUSION_MODEL", "minimax_h3.safetensors"),
        weight_dtype=_value(mapping, "WEIGHT_DTYPE", "default"),
    )
    unet_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CLIPLoader",
        clip_name=_value(mapping, "H3_TEXT_ENCODER", "qwen3vl_32b.safetensors"),
        type="minimax",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAELoader",
        vae_name=_value(mapping, "H3_VAE", "minimax_h3_vae.safetensors"),
    )
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAELoader",
        vae_name=_value(mapping, "H3_AUDIO_VAE", "minimax_h3_audio_vae.safetensors"),
    )
    audio_vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "MiniMaxH3SigmaShift",
        model=_link(unet_id),
        shift_video=12.0,
        shift_audio=3.0,
    )
    model_id = next_id
    next_id += 1

    # Reference / frame input slots.
    slot_links: dict[str, list[list[Any]]] = {key: [] for key in task_slots}
    for key, slots in task_slots.items():
        for slot in slots:
            value = _stage_entry(mapping, slot)
            if value is None:
                continue
            if key == "images":
                node_id = str(next_id)
                graph[node_id] = _node("LoadImage", image=value)
                slot_links[key].append(_link(node_id))
                next_id += 1
            elif key == "videos":
                node_id = str(next_id)
                graph[node_id] = _node(
                    "VHS_LoadVideoPath",
                    video=f"input/{value}",
                    force_rate=0,
                    custom_width=0,
                    custom_height=0,
                    frame_load_cap=0,
                    skip_first_frames=0,
                    select_every_nth=1,
                )
                slot_links[key].append(_link(node_id))
                next_id += 1
            elif key == "audios":
                node_id = str(next_id)
                graph[node_id] = _node("VHS_LoadAudio", audio_file=f"input/{value}")
                slot_links[key].append(_link(node_id))
                next_id += 1

    inputs: dict[str, Any] = {
        "clip": _link(clip_id),
        "vae": _link(vae_id),
        "audio_vae": _link(audio_vae_id),
        "prompt": _value(mapping, "PROMPT", ""),
        "width": int(_value(mapping, "WIDTH", 480)),
        "height": int(_value(mapping, "HEIGHT", 864)),
        "length": int(_value(mapping, "FRAMES", 125)),
        **task_inputs,
    }
    if slot_links.get("images"):
        inputs["ref_images"] = slot_links["images"]
    if slot_links.get("videos"):
        inputs["ref_videos"] = slot_links["videos"]
    if slot_links.get("audios"):
        inputs["ref_audios"] = slot_links["audios"]

    graph[str(next_id)] = _node(task_node, **inputs)
    task_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("ConditioningZeroOut", conditioning=_link(task_id, 0))
    negative_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "KSampler",
        model=_link(model_id),
        seed=int(_value(mapping, "SEED", 41001)),
        steps=int(_value(mapping, "STEPS", 20)),
        cfg=float(_value(mapping, "CFG", 1.0)),
        sampler_name=_value(mapping, "SAMPLER_NAME", "res_multistep"),
        scheduler=_value(mapping, "SCHEDULER", "simple"),
        positive=_link(task_id, 0),
        negative=_link(negative_id),
        latent_image=_link(task_id, 1),
        denoise=1.0,
    )
    sample_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAEDecode", samples=_link(sample_id), vae=_link(vae_id))
    decode_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAEDecodeAudio", samples=_link(sample_id), vae=_link(audio_vae_id)
    )
    audio_decode_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VHS_VideoCombine",
        images=_link(decode_id),
        frame_rate=float(_value(mapping, "FPS", 24)),
        loop_count=0,
        filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2"),
        format="video/h264-mp4",
        pingpong=False,
        save_output=True,
    )
    video_out_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "SaveAudio",
        audio=_link(audio_decode_id),
        filename_prefix="audio/" + str(_value(mapping, "OUTPUT_PREFIX", "avatar_v2")),
    )
    audio_out_id = next_id
    next_id += 1

    return graph


def build_h3_ref2va_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    task_inputs: dict[str, Any] = {
        "ref_image_size": "match",
    }
    slots = {
        "images": [f"H3_PICTURE_{i}" for i in range(1, 10)],
        "videos": [f"H3_VIDEO_{i}" for i in range(1, 4)],
        "audios": [f"H3_AUDIO_{i}" for i in range(1, 4)],
    }
    return build_h3_common(mapping, "MiniMaxH3ReferenceToVideo", task_inputs, slots)


def build_h3_fl2va_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    first = _stage_entry(mapping, "INIT_IMAGE")
    last = _stage_entry(mapping, "LAST_FRAME")
    extra_nodes: dict[str, dict[str, Any]] = {}
    if first:
        extra_nodes["first"] = _node("LoadImage", image=first)
    if last:
        extra_nodes["last"] = _node("LoadImage", image=last)

    graph = build_h3_common(mapping, "MiniMaxH3ImageToVideo", {}, {})

    task_node_id = None
    for node_id, node in graph.items():
        if node["class_type"] == "MiniMaxH3ImageToVideo":
            task_node_id = node_id
            break
    if task_node_id is None:
        raise RuntimeError("MiniMaxH3ImageToVideo node missing from built graph")

    if extra_nodes.get("first"):
        new_id = str(int(task_node_id) + 100)
        graph[new_id] = extra_nodes["first"]
        graph[task_node_id]["inputs"]["first_frame"] = _link(new_id)
    if extra_nodes.get("last"):
        new_id = str(int(task_node_id) + 101)
        graph[new_id] = extra_nodes["last"]
        graph[task_node_id]["inputs"]["last_frame"] = _link(new_id)
    return graph


def build_wan_flf2v_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    """Wan 2.2 first/last-frame-to-video graph (WanFirstLastFrameToVideo)."""
    graph: dict[str, Any] = {}
    next_id = 1

    graph[str(next_id)] = _node(
        "UNETLoader",
        unet_name=_value(mapping, "WAN_DIFFUSION_MODEL", "wan2.2_fun_i2v.safetensors"),
        weight_dtype=_value(mapping, "WEIGHT_DTYPE", "default"),
    )
    unet_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CLIPLoader",
        clip_name=_value(mapping, "WAN_TEXT_ENCODER", "umt5_xxl.safetensors"),
        type="wan",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAELoader", vae_name=_value(mapping, "WAN_VAE", "wan2.2_vae.safetensors"))
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "PROMPT", ""), clip=_link(clip_id))
    pos_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "NEGATIVE_PROMPT", ""), clip=_link(clip_id))
    neg_id = next_id
    next_id += 1

    first = _stage_entry(mapping, "INIT_IMAGE")
    last = _stage_entry(mapping, "LAST_FRAME")
    task_inputs: dict[str, Any] = {
        "positive": _link(pos_id),
        "negative": _link(neg_id),
        "vae": _link(vae_id),
        "width": int(_value(mapping, "WIDTH", 864)),
        "height": int(_value(mapping, "HEIGHT", 480)),
        "length": int(_value(mapping, "FRAMES", 125)),
        "batch_size": 1,
    }
    if first:
        graph[str(next_id)] = _node("LoadImage", image=first)
        task_inputs["start_image"] = _link(next_id)
        next_id += 1
    if last:
        graph[str(next_id)] = _node("LoadImage", image=last)
        task_inputs["end_image"] = _link(next_id)
        next_id += 1

    graph[str(next_id)] = _node("WanFirstLastFrameToVideo", **task_inputs)
    task_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "KSampler",
        model=_link(unet_id),
        seed=int(_value(mapping, "SEED", 41001)),
        steps=int(_value(mapping, "STEPS", 20)),
        cfg=float(_value(mapping, "CFG", 1.0)),
        sampler_name=_value(mapping, "SAMPLER_NAME", "uni_pc"),
        scheduler=_value(mapping, "SCHEDULER", "simple"),
        positive=_link(task_id, 0),
        negative=_link(task_id, 1),
        latent_image=_link(task_id, 2),
        denoise=1.0,
    )
    sample_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAEDecode", samples=_link(sample_id), vae=_link(vae_id))
    decode_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VHS_VideoCombine",
        images=_link(decode_id),
        frame_rate=float(_value(mapping, "FPS", 24)),
        loop_count=0,
        filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2"),
        format="video/h264-mp4",
        pingpong=False,
        save_output=True,
    )
    video_out_id = next_id
    next_id += 1

    return graph


def build_wan_funcontrol_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    """Wan 2.2 Fun-Control graph: ref_image = identity, control_video = motion."""
    graph: dict[str, Any] = {}
    next_id = 1

    graph[str(next_id)] = _node(
        "UNETLoader",
        unet_name=_value(mapping, "WAN_DIFFUSION_MODEL", "wan2.2_fun_control_5B_bf16.safetensors"),
        weight_dtype=_value(mapping, "WEIGHT_DTYPE", "default"),
    )
    unet_id = next_id
    next_id += 1

    lora = _stage_entry(mapping, "WAN_LORA")
    if lora:
        graph[str(next_id)] = _node(
            "LoraLoaderModelOnly",
            model=_link(unet_id),
            lora_name=lora,
            strength_model=float(_value(mapping, "WAN_LORA_STRENGTH", 1.0)),
        )
        unet_id = next_id
        next_id += 1

    graph[str(next_id)] = _node(
        "CLIPLoader",
        clip_name=_value(mapping, "WAN_TEXT_ENCODER", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        type="wan",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAELoader", vae_name=_value(mapping, "WAN_VAE", "wan2.2_vae.safetensors"))
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "PROMPT", ""), clip=_link(clip_id))
    pos_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "NEGATIVE_PROMPT", ""), clip=_link(clip_id))
    neg_id = next_id
    next_id += 1

    task_inputs: dict[str, Any] = {
        "positive": _link(pos_id),
        "negative": _link(neg_id),
        "vae": _link(vae_id),
        "width": int(_value(mapping, "WIDTH", 832)),
        "height": int(_value(mapping, "HEIGHT", 480)),
        "length": int(_value(mapping, "FRAMES", 81)),
        "batch_size": 1,
    }

    ref_image = _stage_entry(mapping, "INIT_IMAGE") or _stage_entry(mapping, "IDENTITY_REF_1")
    if ref_image:
        graph[str(next_id)] = _node("LoadImage", image=ref_image)
        task_inputs["ref_image"] = _link(next_id)
        next_id += 1

    control_video = _stage_entry(mapping, "MOTION_VIDEO")
    if control_video:
        graph[str(next_id)] = _node(
            "VHS_LoadVideoPath",
            video=f"input/{control_video}",
            force_rate=0,
            custom_width=0,
            custom_height=0,
            frame_load_cap=0,
            skip_first_frames=0,
            select_every_nth=1,
        )
        task_inputs["control_video"] = _link(next_id, 0)
        next_id += 1

    graph[str(next_id)] = _node("Wan22FunControlToVideo", **task_inputs)
    task_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "KSampler",
        model=_link(unet_id),
        seed=int(_value(mapping, "SEED", 41001)),
        steps=int(_value(mapping, "STEPS", 20)),
        cfg=float(_value(mapping, "CFG", 1.0)),
        sampler_name=_value(mapping, "SAMPLER_NAME", "uni_pc"),
        scheduler=_value(mapping, "SCHEDULER", "simple"),
        positive=_link(task_id, 0),
        negative=_link(task_id, 1),
        latent_image=_link(task_id, 2),
        denoise=1.0,
    )
    sample_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAEDecode", samples=_link(sample_id), vae=_link(vae_id))
    decode_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VHS_VideoCombine",
        images=_link(decode_id),
        frame_rate=float(_value(mapping, "FPS", 16)),
        loop_count=0,
        filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2"),
        format="video/h264-mp4",
        pingpong=False,
        save_output=True,
    )
    next_id += 1

    return graph


BUILTIN_BUILDERS = {
    "h3_ref2va": build_h3_ref2va_workflow,
    "h3_fl2va": build_h3_fl2va_workflow,
    "wan22": build_wan_flf2v_workflow,
    "wan22_funcontrol": build_wan_funcontrol_workflow,
}
