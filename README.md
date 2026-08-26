# Avatar V2 — H3 Desktop Studio

Avatar V2 is a local-first desktop application for reference-driven synthetic-human video production. It separates identity, body, hair, wardrobe, motion, scene, audio, camera and renderer configuration rather than asking one video model to invent everything from a single prompt.

The desktop branch is **MiniMax H3-first** with Wan 2.2 as the local fallback and bundled AnimateDiff lanes for older GPUs. It communicates with ComfyUI over HTTP and keeps ComfyUI workflow graphs authoritative, so model checkpoints, LoRAs, samplers and nodes can evolve independently of the application.

```
AVATAR IDENTITY VAULT + BODY/HAIR/WARDROBE/MOTION/SCENE/AUDIO REFS
        ↓
REFERENCE COMPOSER  →  H3-NATIVE PROMPT COMPILER
        ↓
RUNTIME PROFILE RESOLVER  →  MODEL / WORKFLOW ROUTER
        ↓
COMFYUI PROVIDER  →  GPU MEMORY MANAGER  →  RENDER QUEUE
        ↓
OUTPUT / QA / HISTORY  →  DESKTOP APPLICATION
```

## What the desktop app exposes

- **Studio** — renderer lanes (MiniMax H3 Ref2VA, H3 FL2VA, Wan 2.2, Custom ComfyUI), runtime profiles, timed motion prompting, wind physics, H3 reference-role bindings, best-of-N and Profile Sweep. Adult content classes also expose an opt-in prompt builder with preloaded roles, environments, actions, toys/props, moves and positions; it remains gated by the existing verified-adult consent policy.
- **Avatar Vault + Profile Designer** — persistent identity/body/hair/wardrobe/anatomy roles, source-image pools, an explicit adult apparent-age anchor, appearance manifests (body type, freckles/moles, scars, tattoos and anatomy details), a provenance-checked local anatomy-guide library, and complete front/rear/side/top/bottom/detail coverage. Each generated or imported candidate is accepted explicitly and saved as an iteration.
- **Motion Library** — movement-only video assets with categories (walk_confident, turn_and_smile, …). Motion references transfer choreography, never identity.
- **Scene Library** — local scene references with location/lighting/time-of-day/camera metadata.
- **History / Runs** — every queued run retains engine, profile, model, workflow, seed, canvas, frames, steps, CFG, GPU, peak VRAM, elapsed time, cleanup events, outputs and the exact error.
- **Settings** — ComfyUI URL/folders/service control, workflow files, H3 license gate, GPU auto-clean threshold/interval, preferred output directory. Persisted under `~/.avatar_v2/`.

All libraries, runs, and settings live under `~/.avatar_v2/`. Nothing is uploaded anywhere.

The Profile Designer can always build exact slot prompts and import/accept candidate images. **Run Next Missing** queues one reviewable iteration; **Build All Missing Views** is the faster fill-only path. **Build Complete Validation Set** rebuilds every applicable view, including accepted coverage, with the primary identity image anchored and the entire same-avatar cache, accepted profile views, and selected appearance examples automatically rotated through H3's per-render reference capacity. Large caches create additional validation candidates so every image participates. Existing accepted images remain saved until the user explicitly replaces them. Per-slot **Generate** buttons remain available. The selected apparent age is injected into both profile turns and later Studio renders as a stable face/skin/body-maturity anchor. H3 performs one slow transition to the requested angle, holds the target view, and Avatar V2 extracts the final frame as a PNG candidate. Body regions absent from the source still have to be inferred, so generated views remain candidates that require visual acceptance.

The optional **Hybrid guided Ref2VA** route keeps identity slots on FL2VA and uses selected guide-only anatomy images for body/anatomy slots. It never treats guides as avatar identity: compiled H3 roles restrict them to clinical topology and explicitly reject transfer of face, age, ethnicity, skin, body identity, marks, tattoos, hair, or clothing. Guide records remain local file references and require provenance, adult-subject confirmation, usage-rights confirmation, and consent/model-release confirmation for real-person photos. Guided Ref2VA forces the INT8 low-memory profile, 352×608 canvas, match-sized references, and reduced steps, but remains experimental—not promised to fit—on an 8 GB GPU.

The separate **Identity + body refs · Ref2VA** route connects the selected primary source as the facial identity anchor and binds up to eight additional images per render, prioritizing the reference matching the requested body angle. Complete validation rotates through the full saved pool across its runs, so the per-render H3 limit does not silently exclude later images. Identity coverage remains on FL2VA unless a ready face-shape or skin option is selected; those identity slots use Ref2VA while retaining the primary source as the identity anchor. Body and anatomy coverage uses Ref2VA with visible bound-versus-available reference counts in each affected slot. This route also forces the INT8 low-memory profile, 352×608 canvas, match-sized references, and reduced steps, and remains experimental on an 8 GB GPU.

For simpler setup, **Quick Profile Builder** accepts one local reference-cache folder and inventories still images, GIFs, videos, and audio. When the bundled local Ollama route is available, `qwen3-vl:2b-instruct` inspects every still image and assigns face, front, rear, side, top, low, torso, pelvis, hands, or feet roles with a confidence threshold; descriptive filenames are only a fallback. Analysis is restricted to the loopback Ollama service, and the vision model is unloaded afterward to return VRAM to H3. Still images can become identity/body candidates; GIF and video remain motion examples; audio remains voice/timing input. Low-confidence and duplicate candidates are listed for review instead of being silently trusted. The detailed per-angle fields, anatomy references, prompts, and renderer controls remain available under Advanced panels.

Previously authorized anatomy references are presented separately under **Use saved anatomy references**, where they can be attached to the current avatar without recreating labels or provenance. The collapsed **Add a new anatomy reference** form is only for authoring a new reusable reference record.

The local **Appearance Component Catalog** groups two to eight authorized examples into reusable selectors for face shape, body type, height, proportions, torso, buttocks, pelvis, hands, feet, or skin. Each option records provenance, adult/rights confirmation, and real-person consent or release. It becomes selectable only after its configured minimum example count is met. Face-shape and skin examples can condition identity Ref2VA slots; the remaining selected component examples condition body/anatomy slots for the named morphology only. The primary source remains the facial identity anchor. This is reference conditioning, not model training or dataset ingestion.

## Runtime profiles

The app does not assume one configuration will work everywhere. Use the **Runtime Profile** controls to benchmark the same shot under different setups:

| Profile | Purpose |
|---|---|
| **Auto** | Detect VRAM and installed models; prefer INT8 on 12–16 GB cards |
| **H3 Quality** | Prefer an installed non-INT8 H3 checkpoint when practical |
| **H3 INT8 · 12 GB** | Force the pruned INT8 H3 checkpoint |
| **H3 Low Memory** | INT8 + smallest practical H3 canvas + reduced steps |
| **H3 Low Memory · Quality** | INT8 + 4070-class canvas + reduced steps |
| **Wan 2.2 Fallback** | Skip H3 and use the configured Wan workflow |

**Profile Sweep** queues Auto, INT8, Low Memory and Low Memory · Quality with different deterministic seeds. Each queue record retains the exact profile, model, canvas, seed, timing, VRAM peak and result/error so the working configuration can be determined empirically.

### Important 12 GB H3 note

The H3 community UI we reviewed reports that FL2VA text/image-to-video works on a 12 GB RTX 3060 around 864×480, while Ref2VA remained roughly 1 GB over the card's available VRAM even after reducing resolution, steps, frame count and reference size. That makes the low-memory Ref2VA profiles genuine experiments, not a promise that Ref2VA will fit on 12 GB. If they still fail, Avatar V2 can move the shot to Wan locally or a separately configured larger/hosted renderer.

## H3-native rules

Avatar V2 enforces H3's practical generation constraints:

- 24 fps.
- Dimensions from multiple-of-32 presets.
- Valid frame counts snapped to the `17k+5` grid.
- Single H3 clip duration capped at 15 seconds.
- Consumer presets including `480×864` vertical and `864×480` landscape.
- ComfyUI runtime model discovery rather than hard-coded filenames.
- Optional last-frame chaining infrastructure for longer sequences.

## Local H3 license gate

Avatar V2 **does not download MiniMax H3 weights automatically**. Local H3 execution is disabled by default. The user/admin must explicitly confirm in Settings that local open-weight use is permitted by the current MiniMax H3 license or covered by a separate written MiniMax license.

If local H3 is not enabled, H3 queue requests are refused with that exact reason (they are not silently rerouted), and the app can use the configured Wan local fallback instead. Hosted/API H3 can be added as a separate explicit provider rather than silently sending local prompts or avatar assets off-machine.

## Hard content boundaries

Adult workflows are supported only when every represented subject is an age-verified consenting adult or a synthetic adult persona with explicit adult provenance. Unknown provenance fails closed. Do not use the system for minors, non-consensual sexual content, or sexualized impersonation/deepfakes without the represented adult's consent.

## GPU memory auto-clean (mandatory architecture)

`avatar_v2/gpu_memory.py` is the production-safe evolution of the user's GPU auto-cleaner concept. It:

- reads all visible NVIDIA GPUs through `nvidia-smi`;
- records VRAM used/total and utilization fraction;
- samples VRAM during each render and records per-GPU peak usage;
- defaults to an 85% cleanup threshold;
- forces pre-render cleanup for `int8_12gb`, `low_memory`, and `low_memory_quality` profiles;
- requests post-render cleanup after failed/cancelled renders and low-memory runs;
- requests ComfyUI cleanup through `POST /free` with `unload_models=true` and `free_memory=true`;
- records before/after memory snapshots and cleanup errors in queue state;
- **never automatically kills arbitrary Python processes**. The emergency termination dialog in the GPU panel requires explicitly selected PIDs and confirmation.

Environment controls:

```text
AVATAR_V2_GPU_AUTOCLEAN=1
AVATAR_V2_GPU_THRESHOLD=0.85
AVATAR_V2_GPU_CHECK_INTERVAL=5
```

The ComfyUI `/free` path matters because `torch.cuda.empty_cache()` in the Avatar V2 controller process cannot clear allocator/cache memory owned by the separate ComfyUI process. See `docs/GPU_MEMORY_MANAGER.md` and `docs/CUDA_ERROR_999_QUICK_REFERENCE.txt`.

## ComfyUI integration

Avatar V2 talks to ComfyUI at `http://127.0.0.1:8188` by default and supports:

- `GET /system_stats`, `GET /object_info` (model/node discovery)
- `POST /prompt`, `GET /history/{prompt_id}` (queueing and polling)
- `POST /interrupt`, `POST /free` (cancel and memory release)
- HTTP upload staging (`POST /upload/image`) or direct input-folder staging

The Settings screen can start/stop a locally managed ComfyUI (only the process Avatar V2 itself started, bound to 127.0.0.1 — never 0.0.0.0).

### Bundled workflows

- `workflows/animatediff_sd15_api.json` and `workflows/animatediff_sdxl_api.json` — validated AnimateDiff T2V graphs (AnimateDiff-Evolved + VideoHelperSuite) selectable in the Custom lane.
- Built-in dynamic workflow builders for **MiniMax H3 Ref2VA**, **MiniMax H3 FL2VA**, and **Wan 2.2 first/last-frame-to-video**, constructed from live ComfyUI schemas. Profile Designer candidates use H3 FL2VA and automatic final-frame extraction. When you configure your own exported API-format workflow in Settings, that file wins.

See `workflows/README.md` for the placeholder contract.

### Renderer availability on this machine (as deployed)

| Renderer | Status |
|---|---|
| **Wan 2.2 fun-control (5B)** | **READY** — identity+motion renders verified at 832×480/960×544/704²/768², 8–20 steps, 33–81 frames, peak VRAM 87–94%. Canvas dims must be multiples of 32 (app snaps automatically). Motion references must be generated at the exact target resolution with neutral color balance and matching framing — mismatched motion sources cause pixelation, color cast, and identity drift. |
| Custom ComfyUI / AnimateDiff SD 1.5 | READY — real renders verified |
| Custom ComfyUI / AnimateDiff SDXL | READY — 512²×16f verified (80% peak VRAM) |
| MiniMax H3 Ref2VA / FL2VA | PARTIALLY READY — native ComfyUI nodes present, workflows structurally validated; BLOCKED on weights + license gate |

## Install for development (Linux)

Prerequisites: Ubuntu desktop (X11 or Wayland), Python 3.11+, ComfyUI, `ffmpeg`/`ffprobe`, GTK3 + WebKit2GTK (for pywebview).

```bash
git clone https://github.com/Davisjxxxx/Content_Creation_System.git
cd Content_Creation_System
git checkout avatar-v2-h3-desktop
python3 -m venv --system-site-packages .venv   # system gi is used by pywebview on Linux
source .venv/bin/activate
pip install -e '.[desktop,dev]'
avatar-v2-desktop
```

## Linux desktop launcher

```bash
./scripts/install_linux_desktop.sh
```

This installs:

- `~/.local/share/applications/avatar-v2.desktop` (launcher: **Avatar V2**)
- icons under `~/.local/share/icons/hicolor/`
- a desktop shortcut when `~/Desktop` exists

The launcher prefers the PyInstaller build at `dist/AvatarV2/AvatarV2` when present and otherwise runs the project venv. It is working-directory independent.

### Build the packaged app (optional)

```bash
python -m PyInstaller --noconfirm --clean scripts/avatar_v2.spec
```

Produces `dist/AvatarV2/`. The venv launcher remains a supported fallback if a future pywebview/GTK packaging change breaks the bundle.

## CLI remains available

```bash
avatar-v2 validate examples/avatar.synthetic.yaml examples/shot.realism.yaml
avatar-v2 plan examples/avatar.synthetic.yaml examples/shot.realism.yaml -o build/job.json
avatar-v2 plan-batch examples/avatar.synthetic.yaml examples/shot.realism.yaml -n 4
avatar-v2 doctor examples/provider.comfyui.yaml
avatar-v2 qa output.mp4
```

## Developer commands

```bash
python -m compileall -q avatar_v2 scripts
pytest -q
```

## Privacy

Local-first by design: no analytics, no telemetry, no automatic prompt or avatar uploading, no public network binding, no secrets in source, and no hosted model provider enabled implicitly. Private files live in ignored directories (`assets/private/`, `runs/`) and model weights/media are gitignored.

## Troubleshooting

- **ComfyUI unreachable** — use Settings → Start ComfyUI (requires `comfyui_dir` + `comfyui_python`), or start it manually and press Check in Studio.
- **CUDA out of memory** — smaller canvas / fewer frames / Low Memory profile. The error message includes the classification.
- **CUDA error 999 while nvidia-smi works** — see `docs/CUDA_ERROR_999_QUICK_REFERENCE.txt` (privileged fixes are never executed automatically).
- **Render failed** — every run in History retains the exact underlying exception plus a human-readable diagnosis; the app never reduces failures to "Render failed."
