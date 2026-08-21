# ComfyUI workflow integration

Avatar V2 deliberately does not hard-code Wan/LTX custom-node graphs. Export the exact workflow you have proven in ComfyUI using **Save (API Format)**, then replace the values you want Avatar V2 to control with placeholders.

Supported placeholders:

- `${PROMPT}`
- `${NEGATIVE_PROMPT}`
- `${INIT_IMAGE}`
- `${LAST_FRAME}`
- `${MOTION_VIDEO}`
- `${SCENE_IMAGE}`
- `${AUDIO}`
- `${WIDTH}`
- `${HEIGHT}`
- `${FPS}`
- `${FRAMES}`

Example node fragment:

```json
{
  "12": {
    "class_type": "LoadImage",
    "inputs": {"image": "${INIT_IMAGE}"}
  },
  "44": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "${PROMPT}", "clip": ["10", 1]}
  }
}
```

When `input_dir` is configured, local reference files are copied into the ComfyUI input directory and placeholders resolve to the staged filenames. This also works for motion-video or audio loader nodes that read files from the ComfyUI input directory.

Use separate workflow exports for the exact model stack you want to benchmark, for example:

- `wan22_i2v_api.json`
- `wan22_animate_api.json`
- `ltx25_i2v_api.json`
- `ltx25_first_last_api.json`

The renderer router records the preferred engine in the job manifest, but V2 intentionally lets the supplied workflow remain authoritative. This avoids silently changing checkpoints, LoRAs, samplers or custom-node versions.
