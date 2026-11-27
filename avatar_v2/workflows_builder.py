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

# Pixel threshold above which FLUX.2 still decoding automatically switches to VAEDecodeTiled
# (standard 896x1280 = 1,146,880 px stays on baseline VAEDecode unless TILED_VAE is requested).
TILED_VAE_STILL_PIXEL_THRESHOLD = 1024 * 1280


def tiled_vae_decode_node(
    samples_link: list[Any],
    vae_link: list[Any],
    *,
    tile_size: int = 512,
    overlap: int = 64,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
) -> dict[str, Any]:
    """Schema-verified ComfyUI 0.46 VAEDecodeTiled node (nodes.py:348)."""
    tile_size = max(64, min(int(tile_size), 4096))
    overlap = max(0, min(int(overlap), tile_size // 4))
    temporal_size = max(8, min(int(temporal_size), 4096))
    temporal_overlap = max(4, min(int(temporal_overlap), temporal_size // 2))
    return _node(
        "VAEDecodeTiled",
        samples=samples_link,
        vae=vae_link,
        tile_size=tile_size,
        overlap=overlap,
        temporal_size=temporal_size,
        temporal_overlap=temporal_overlap,
    )


def with_tiled_vae(
    graph: dict[str, Any],
    *,
    tile_size: int = 512,
    overlap: int = 64,
    temporal_size: int = 64,
    temporal_overlap: int = 8,
) -> dict[str, Any]:
    """Return a copy of a ComfyUI API graph with every VAEDecode node upgraded to VAEDecodeTiled."""
    updated: dict[str, Any] = {}
    for node_id, node in graph.items():
        if isinstance(node, dict) and node.get("class_type") == "VAEDecode":
            inputs = node.get("inputs") or {}
            tiled = tiled_vae_decode_node(
                inputs["samples"],
                inputs["vae"],
                tile_size=tile_size,
                overlap=overlap,
                temporal_size=temporal_size,
                temporal_overlap=temporal_overlap,
            )
            if "_meta" in node:
                tiled["_meta"] = dict(node["_meta"])
            updated[node_id] = tiled
        else:
            updated[node_id] = node
    return updated


def with_wan_end_frame(graph: dict[str, Any], end_image_name: str) -> dict[str, Any]:
    """Upgrade a WanImageToVideo graph to WanFirstLastFrameToVideo with an explicit end keyframe.

    Verified against live ComfyUI 0.46 ``/object_info/WanFirstLastFrameToVideo`` schema:
    identical required inputs and ``(positive, negative, latent)`` outputs plus optional ``end_image``.
    Enables FLUX.2-conditioned front -> back or angle-change keyframes to guide Wan 2.2 turns.
    """
    end_clean = str(end_image_name or "").strip()
    if not end_clean:
        return dict(graph)
    updated: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in graph.items()}
    numeric_ids = [int(k) for k in updated if str(k).isdigit()]
    end_node_id = str(max(numeric_ids, default=100) + 1)
    upgraded = False
    for node_id, node in list(updated.items()):
        if isinstance(node, dict) and node.get("class_type") == "WanImageToVideo":
            inputs = dict(node.get("inputs") or {})
            inputs["end_image"] = _link(end_node_id)
            updated[node_id] = {**node, "class_type": "WanFirstLastFrameToVideo", "inputs": inputs}
            upgraded = True
    if upgraded:
        updated[end_node_id] = _node("LoadImage", image=end_clean)
    return updated


def without_wan_lightning(
    graph: dict[str, Any],
    *,
    steps: int = 12,
    cfg: float = 3.5,
) -> dict[str, Any]:
    """Return a copy of a Wan 2.2 A14B graph bypassing Lightning LoRAs for standard CFG>1 sampling.

    Used in Sprint 9 controlled video benchmarks comparing 4-step/6-step Lightning acceleration
    against standard higher-step base sampling on identical starter frames.
    """
    updated: dict[str, Any] = {}
    lora_bypass: dict[str, list[Any]] = {}
    for node_id, node in graph.items():
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly":
            model_link = (node.get("inputs") or {}).get("model")
            if isinstance(model_link, list):
                lora_bypass[str(node_id)] = model_link
                continue
        updated[node_id] = dict(node) if isinstance(node, dict) else node

    half = max(1, int(steps) // 2)
    sampler_idx = 0
    for node_id, node in list(updated.items()):
        if not isinstance(node, dict):
            continue
        inputs = dict(node.get("inputs") or {})
        if isinstance(inputs.get("model"), list) and str(inputs["model"][0]) in lora_bypass:
            inputs["model"] = lora_bypass[str(inputs["model"][0])]
        if node.get("class_type") == "KSamplerAdvanced":
            inputs["steps"] = int(steps)
            inputs["cfg"] = float(cfg)
            if sampler_idx == 0:
                inputs["start_at_step"] = 0
                inputs["end_at_step"] = half
            else:
                inputs["start_at_step"] = half
                inputs["end_at_step"] = int(steps)
            sampler_idx += 1
        updated[node_id] = {**node, "inputs": inputs}
    return updated


def build_flux2_masked_edit_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    """FLUX.2 genuine mask-conditioned localized editing + pixel-preserving ImageCompositeMasked.

    Uses verified ComfyUI 0.46 core nodes (SetLatentNoiseMask + DifferentialDiffusion +
    ReferenceLatent + ImageToMask + ImageCompositeMasked) with installed FLUX.2 9B/4B GGUF.
    Guarantees 0.00% pixel alteration outside the feathered edit mask.
    """
    source_image = _stage_entry(mapping, "SOURCE_IMAGE") or _stage_entry(mapping, "REF_IMAGE_1")
    mask_image = _stage_entry(mapping, "MASK_IMAGE")
    if not source_image or not mask_image:
        raise RuntimeError("build_flux2_masked_edit_workflow requires SOURCE_IMAGE and MASK_IMAGE.")

    graph: dict[str, Any] = {}
    next_id = 1

    graph[str(next_id)] = _node("UnetLoaderGGUF", unet_name=_value(mapping, "FORGE_UNET", "flux-2-klein-base-9b-Q4_K_M.gguf"))
    unet_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("DifferentialDiffusion", model=_link(unet_id), strength=float(_value(mapping, "DIFF_STRENGTH", 1.0)))
    model_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CLIPLoaderGGUF",
        clip_name=_value(mapping, "FORGE_TEXT_ENCODER", "Qwen3-8B-Q4_K_M.gguf"),
        type="flux2",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAELoader", vae_name=_value(mapping, "FORGE_VAE", "flux2-vae.safetensors"))
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "PROMPT", ""), clip=_link(clip_id))
    pos_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("CLIPTextEncode", text=_value(mapping, "NEGATIVE_PROMPT", ""), clip=_link(clip_id))
    neg_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("LoadImage", image=source_image)
    src_img_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("LoadImage", image=mask_image)
    mask_img_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("ImageToMask", image=_link(mask_img_id), channel="red")
    mask_id = next_id
    next_id += 1

    # Encode full source image into FLUX.2 latent and condition via ReferenceLatent for context coherence.
    graph[str(next_id)] = _node("VAEEncode", pixels=_link(src_img_id), vae=_link(vae_id))
    src_lat_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("ReferenceLatent", conditioning=_link(pos_id), latent=_link(src_lat_id))
    pos_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("ReferenceLatent", conditioning=_link(neg_id), latent=_link(src_lat_id))
    neg_id = next_id
    next_id += 1

    # Attach noise mask so diffusion updates only inside the masked latent region.
    graph[str(next_id)] = _node("SetLatentNoiseMask", samples=_link(src_lat_id), mask=_link(mask_id))
    masked_lat_id = next_id
    next_id += 1

    width = int(_value(mapping, "WIDTH", 896))
    height = int(_value(mapping, "HEIGHT", 1280))
    graph[str(next_id)] = _node("Flux2Scheduler", steps=int(_value(mapping, "STEPS", 28)), width=width, height=height)
    sigmas_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("RandomNoise", noise_seed=int(_value(mapping, "SEED", 41001)))
    noise_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("KSamplerSelect", sampler_name=_value(mapping, "SAMPLER_NAME", "euler"))
    sampler_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CFGGuider",
        model=_link(model_id),
        positive=_link(pos_id),
        negative=_link(neg_id),
        cfg=float(_value(mapping, "CFG", 3.5)),
    )
    guider_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "SamplerCustomAdvanced",
        noise=_link(noise_id),
        guider=_link(guider_id),
        sampler=_link(sampler_id),
        sigmas=_link(sigmas_id),
        latent_image=_link(masked_lat_id),
    )
    sample_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("VAEDecode", samples=_link(sample_id), vae=_link(vae_id))
    decode_id = next_id
    next_id += 1

    # Composite ONLY the feathered masked region back onto the original source image (0.00% outside-mask drift!).
    graph[str(next_id)] = _node(
        "ImageCompositeMasked",
        destination=_link(src_img_id),
        source=_link(decode_id),
        x=0,
        y=0,
        resize_source=True,
        mask=_link(mask_id),
    )
    comp_id = next_id
    next_id += 1

    graph[str(next_id)] = _node("SaveImage", images=_link(comp_id), filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2/masked_edit"))
    return graph


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
        unet_name=_value(mapping, "H3_DIFFUSION_MODEL", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        weight_dtype=_value(mapping, "WEIGHT_DTYPE", "default"),
    )
    unet_id = next_id
    next_id += 1

    h3_lora = _stage_entry(mapping, "H3_LORA")
    if h3_lora:
        graph[str(next_id)] = _node(
            "LoraLoaderModelOnly",
            model=_link(unet_id),
            lora_name=h3_lora,
            strength_model=float(_value(mapping, "H3_LORA_STRENGTH", 1.0)),
        )
        unet_id = next_id
        next_id += 1

    graph[str(next_id)] = _node(
        "CLIPLoader",
        clip_name=_value(mapping, "H3_TEXT_ENCODER", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
        type="minimax",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAELoader",
        vae_name=_value(mapping, "H3_VAE", "minimax_h3_video_vae_fp16.safetensors"),
    )
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAELoader",
        vae_name=_value(mapping, "H3_AUDIO_VAE", "minimax_h3_audio_vae_fp32.safetensors"),
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
        "prompt": _value(mapping, "PROMPT", ""),
        "width": int(_value(mapping, "WIDTH", 480)),
        "height": int(_value(mapping, "HEIGHT", 864)),
        "length": int(_value(mapping, "FRAMES", 125)),
        **task_inputs,
    }
    # Only Ref2VA's execute() accepts audio_vae; FL2VA carries audio purely
    # through the AV latent (decoded later by VAEDecodeAudio).
    if task_node == "MiniMaxH3ReferenceToVideo":
        inputs["audio_vae"] = _link(audio_vae_id)
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

    if bool(mapping.get("TILED_VAE")) or (
        int(_value(mapping, "WIDTH", 0)) * int(_value(mapping, "HEIGHT", 0)) > TILED_VAE_STILL_PIXEL_THRESHOLD
    ):
        graph[str(next_id)] = tiled_vae_decode_node(
            _link(sample_id),
            _link(vae_id),
            tile_size=int(_value(mapping, "VAE_TILE_SIZE", 512)),
            overlap=int(_value(mapping, "VAE_OVERLAP", 64)),
            temporal_size=int(_value(mapping, "VAE_TEMPORAL_SIZE", 64)),
            temporal_overlap=int(_value(mapping, "VAE_TEMPORAL_OVERLAP", 8)),
        )
    else:
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
    next_id += 1

    return graph


def build_h3_ref2va_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    task_inputs: dict[str, Any] = {
        "ref_image_size": _value(mapping, "H3_REF_IMAGE_SIZE", "max"),
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

    if bool(mapping.get("TILED_VAE")) or (
        int(_value(mapping, "WIDTH", 0)) * int(_value(mapping, "HEIGHT", 0)) > TILED_VAE_STILL_PIXEL_THRESHOLD
    ):
        graph[str(next_id)] = tiled_vae_decode_node(
            _link(sample_id),
            _link(vae_id),
            tile_size=int(_value(mapping, "VAE_TILE_SIZE", 512)),
            overlap=int(_value(mapping, "VAE_OVERLAP", 64)),
            temporal_size=int(_value(mapping, "VAE_TEMPORAL_SIZE", 64)),
            temporal_overlap=int(_value(mapping, "VAE_TEMPORAL_OVERLAP", 8)),
        )
    else:
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

    if bool(mapping.get("TILED_VAE")) or (
        int(_value(mapping, "WIDTH", 0)) * int(_value(mapping, "HEIGHT", 0)) > TILED_VAE_STILL_PIXEL_THRESHOLD
    ):
        graph[str(next_id)] = tiled_vae_decode_node(
            _link(sample_id),
            _link(vae_id),
            tile_size=int(_value(mapping, "VAE_TILE_SIZE", 512)),
            overlap=int(_value(mapping, "VAE_OVERLAP", 64)),
            temporal_size=int(_value(mapping, "VAE_TEMPORAL_SIZE", 64)),
            temporal_overlap=int(_value(mapping, "VAE_TEMPORAL_OVERLAP", 8)),
        )
    else:
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


def build_flux2_txt2img_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    """FLUX.2 [klein] 4B text-to-image graph (ComfyUI 0.46 native Flux2 pipeline).

    Node wiring follows the community-verified 8 GB workflow for the unsloth GGUF
    (UnetLoaderGGUF + Qwen3 CLIP on CPU + CFGGuider + Flux2Scheduler). FLUX.2 uses
    an SD3-style latent, so the empty latent comes from EmptySD3LatentImage and the
    sampler runs through SamplerCustomAdvanced. Canvas dimensions must be multiples
    of 16 (FLUX.2 VAE latent).
    """
    graph: dict[str, Any] = {}
    next_id = 1

    graph[str(next_id)] = _node(
        "UnetLoaderGGUF",
        unet_name=_value(mapping, "FORGE_UNET", "flux-2-klein-4b-Q5_K_M.gguf"),
    )
    unet_id = next_id
    next_id += 1

    # The Qwen3 text encoder is a GGUF, so it loads through ComfyUI-GGUF's CLIPLoaderGGUF.
    # In this ComfyUI build (0.46) text encoders run on the CPU by default (no --gpu-only
    # in the app's launch args), which keeps the 8 GB card free for the UNet.
    graph[str(next_id)] = _node(
        "CLIPLoaderGGUF",
        clip_name=_value(mapping, "FORGE_TEXT_ENCODER", "Qwen3-4B-Q5_K_M.gguf"),
        type="flux2",
    )
    clip_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VAELoader",
        vae_name=_value(mapping, "FORGE_VAE", "flux2-vae.safetensors"),
    )
    vae_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CLIPTextEncode",
        text=_value(mapping, "PROMPT", ""),
        clip=_link(clip_id),
    )
    pos_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CLIPTextEncode",
        text=_value(mapping, "NEGATIVE_PROMPT", ""),
        clip=_link(clip_id),
    )
    neg_id = next_id
    next_id += 1

    # Optional FLUX.2 native multi-reference conditioning via chained ReferenceLatent nodes
    # (verified against live ComfyUI 0.46 /object_info/ReferenceLatent & ImageScaleToTotalPixels).
    # Adaptive pixel budget prevents quadratic attention latency spikes when 2-4 refs are chained
    # (empirically calibrated on RTX 4070 8 GB: 0 refs=227s, 1x1.0MP=401s, 2x1.0MP=607s).
    active_ref_files = [
        ref_file
        for ref_slot in (f"REF_IMAGE_{i}" for i in range(1, 5))
        if (ref_file := _stage_entry(mapping, ref_slot))
    ]
    if mapping.get("REF_MEGAPIXELS") not in (None, ""):
        ref_megapixels = float(mapping["REF_MEGAPIXELS"])
    elif len(active_ref_files) <= 1:
        ref_megapixels = 1.0
    elif len(active_ref_files) == 2:
        ref_megapixels = 0.65
    else:
        ref_megapixels = 0.5
    for ref_file in active_ref_files:
        graph[str(next_id)] = _node("LoadImage", image=ref_file)
        load_ref_id = next_id
        next_id += 1

        graph[str(next_id)] = _node(
            "ImageScaleToTotalPixels",
            image=_link(load_ref_id),
            upscale_method="lanczos",
            megapixels=ref_megapixels,
            resolution_steps=16,
        )
        scaled_ref_id = next_id
        next_id += 1

        graph[str(next_id)] = _node(
            "VAEEncode",
            pixels=_link(scaled_ref_id),
            vae=_link(vae_id),
        )
        ref_lat_id = next_id
        next_id += 1

        graph[str(next_id)] = _node(
            "ReferenceLatent",
            conditioning=_link(pos_id),
            latent=_link(ref_lat_id),
        )
        pos_id = next_id
        next_id += 1

        graph[str(next_id)] = _node(
            "ReferenceLatent",
            conditioning=_link(neg_id),
            latent=_link(ref_lat_id),
        )
        neg_id = next_id
        next_id += 1

    graph[str(next_id)] = _node(
        "EmptySD3LatentImage",
        width=int(_value(mapping, "WIDTH", 896)),
        height=int(_value(mapping, "HEIGHT", 1280)),
        batch_size=1,
    )
    latent_id = next_id
    next_id += 1

    width = int(_value(mapping, "WIDTH", 896))
    height = int(_value(mapping, "HEIGHT", 1280))
    graph[str(next_id)] = _node(
        "Flux2Scheduler",
        steps=int(_value(mapping, "STEPS", 8)),
        width=width,
        height=height,
    )
    sigmas_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "RandomNoise",
        noise_seed=int(_value(mapping, "SEED", 41001)),
    )
    noise_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "KSamplerSelect",
        sampler_name=_value(mapping, "SAMPLER_NAME", "euler"),
    )
    sampler_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "CFGGuider",
        model=_link(unet_id),
        positive=_link(pos_id),
        negative=_link(neg_id),
        cfg=float(_value(mapping, "CFG", 1.2)),
    )
    guider_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "SamplerCustomAdvanced",
        noise=_link(noise_id),
        guider=_link(guider_id),
        sampler=_link(sampler_id),
        sigmas=_link(sigmas_id),
        latent_image=_link(latent_id),
    )
    sample_id = next_id
    next_id += 1

    if bool(mapping.get("TILED_VAE")) or (
        int(_value(mapping, "WIDTH", 0)) * int(_value(mapping, "HEIGHT", 0)) > TILED_VAE_STILL_PIXEL_THRESHOLD
    ):
        graph[str(next_id)] = tiled_vae_decode_node(
            _link(sample_id),
            _link(vae_id),
            tile_size=int(_value(mapping, "VAE_TILE_SIZE", 512)),
            overlap=int(_value(mapping, "VAE_OVERLAP", 64)),
            temporal_size=int(_value(mapping, "VAE_TEMPORAL_SIZE", 64)),
            temporal_overlap=int(_value(mapping, "VAE_TEMPORAL_OVERLAP", 8)),
        )
    else:
        graph[str(next_id)] = _node("VAEDecode", samples=_link(sample_id), vae=_link(vae_id))
    decode_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "SaveImage",
        images=_link(decode_id),
        filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2/forge"),
    )
    next_id += 1

    return graph


def build_upscale_workflow(mapping: dict[str, Any]) -> dict[str, Any]:
    """Upscale an existing MP4 frame-by-frame through an ESRGAN-style model."""
    graph: dict[str, Any] = {}
    next_id = 1

    source = _stage_entry(mapping, "SOURCE_VIDEO")
    if source is None:
        raise RuntimeError("SOURCE_VIDEO asset is required for the upscale workflow")

    graph[str(next_id)] = _node(
        "VHS_LoadVideoPath",
        video=f"input/{source}",
        force_rate=0,
        custom_width=0,
        custom_height=0,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
    )
    load_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "UpscaleModelLoader",
        model_name=_value(mapping, "UPSCALE_MODEL", "4x-UltraSharp.pth"),
    )
    model_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "ImageUpscaleWithModel",
        upscale_model=_link(model_id),
        image=_link(load_id, 0),
    )
    upscale_id = next_id
    next_id += 1

    graph[str(next_id)] = _node(
        "VHS_VideoCombine",
        images=_link(upscale_id),
        frame_rate=float(_value(mapping, "FPS", 16)),
        loop_count=0,
        filename_prefix=_value(mapping, "OUTPUT_PREFIX", "avatar_v2_upscale"),
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
    "flux2_txt2img": build_flux2_txt2img_workflow,
    "upscale": build_upscale_workflow,
}
