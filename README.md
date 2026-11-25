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

## The app (start here)

Avatar V2 is now one simple screen with three parts in the sidebar:

- **Make**: "What do you want to make?" Six recipes, each a short list of numbered steps with one
  pink button that shows the price before you click.
  1. **Her look**: pick Natural, Polished or Original, optionally say what to change, and make
     1, 2 or 4 options (8 credits each). Click any option for the full-size viewer, then **Use this one**.
  2. **Photos of her**: the same face from new angles and expressions; keep the ones that look like her.
  3. **Voiceover**: pick her voice, hear a 1-credit sample, and type the script.
  4. **Talking video**: she says your words, lip-synced. Long scripts become several videos
     automatically, and the first video sets up her Runway talking avatar.
  5. **Act it out**: film yourself for 3-30 s; she copies your face and moves, and your voice is removed.
  6. **Moving clip**: turn her photo into a 5-second clip **free on your GPU**, or a 5-10 s clip on
     Runway (Gen-4.5 or the cheaper Turbo).
- **Free builder** (⚡ in the sidebar): build offline, prompt-first, no credits. One prompt box can
  make an **avatar photo** (FLUX.2 klein 4B on the GPU) or a **video scene** — the long prompt is
  split into shots, rendered through Wan 2.2, and stitched into one video. Start from a picture or
  from words alone (the start photo is made first, then the scene follows automatically). Photo
  model dropdown: FLUX.2 klein 4B (Apache 2.0) or 9B Base (non-commercial). Video model dropdown:
  Wan A14B + Lightning or the faster 5B Turbo. A VRAM mode can move text encoders to the CPU for
  the heavy jobs. Every job forces GPU-memory cleanup before and after the render. Details:
  [`docs/OFFLINE_BUILDER.md`](docs/OFFLINE_BUILDER.md).
- **My stuff**: every photo, video and voiceover as a thumbnail. Click to see it full size (Use /
  Keep / Toss, arrow keys, actual-size zoom, "Open in its own window").
- **Settings**: Runway key and balance, the GPU engine (start/stop), optional spending limits, the
  output folder and an activity log.

The old advanced screens (Create with every renderer and option, Build Avatar, Body/Motion/Scene
libraries, Presenter tools, the results table and the old settings) were removed on 2026-11-22
because they didn't run on this machine or conflicted with the recipes. Their backend code is
still in the package, and the screens are in git history (`798d34f`). Code: `avatar_v2/ui/index.html`
(the page) and `avatar_v2/desktop_studio.py` (the recipes).

### Your GPU engine (local, free)

`scripts/setup_local_engine.sh` installs ComfyUI with **Wan 2.2 14B image-to-video** (GGUF Q4_K_S
plus the 4-step Lightning LoRAs), sized for an 8 GB NVIDIA GPU, into `~/AvatarForge` (about 34 GB).
`scripts/setup_offline_builder.sh` adds the Free builder's image models (FLUX.2 klein 4B/9B and
the Wan 5B Turbo, about 16 GB more). The app starts ComfyUI when it opens and stops it when it
closes. It powers the free "Moving clip" option and the whole Free builder; everything else uses
Runway, because nothing else runs well enough on 8 GB. Details and measured speed:
[`docs/LOCAL_ENGINE.md`](docs/LOCAL_ENGINE.md) and [`docs/OFFLINE_BUILDER.md`](docs/OFFLINE_BUILDER.md).

## What the desktop app exposes

- **Studio** — renderer lanes (MiniMax H3 Ref2VA, H3 FL2VA, Wan 2.2, Custom ComfyUI), runtime profiles, timed motion prompting, wind physics, H3 reference-role bindings, best-of-N and Profile Sweep. Adult content classes also expose an opt-in prompt builder with preloaded roles, environments, actions, toys/props, moves and positions; it remains gated by the existing verified-adult consent policy.
- **Avatar Vault + Profile Designer** — persistent identity/body/hair/wardrobe/anatomy roles, source-image pools, an explicit adult apparent-age anchor, appearance manifests (body type, freckles/moles, scars, tattoos and anatomy details), a provenance-checked local anatomy-guide library, and complete front/rear/side/top/bottom/detail coverage. Each generated or imported candidate is accepted explicitly and saved as an iteration.
- **Motion Library** — movement-only video assets with categories (walk_confident, turn_and_smile, …). Motion references transfer choreography, never identity.
- **Scene Library** — local scene references with location/lighting/time-of-day/camera metadata.
- **History / Runs** — every queued run retains engine, profile, model, workflow, seed, canvas, frames, steps, CFG, GPU, peak VRAM, elapsed time, cleanup events, outputs and the exact error.
- **Settings** — ComfyUI URL/folders/service control, workflow files, H3 license gate, GPU auto-clean threshold/interval, preferred output directory. Persisted under `~/.avatar_v2/`.

All libraries, runs, and settings live under `~/.avatar_v2/`. Nothing is uploaded anywhere.

The Profile Designer can always build exact slot prompts and import/accept candidate images. **Run Next Missing** queues one reviewable face/body iteration; **Build All Missing Views** is the faster fill-only path. **Build Complete Validation Set** rebuilds applicable face and body views. It schedules at least one high-quality candidate per supported slot and then schedules each remaining compatible, evidence-gated reference once, so a lower-ranked image is not silently ignored. Every generated job binds one exact source through FL2VA; the app never blends the dump into one Ref2VA identity. Tests on visually conflicting reference sets showed that pooling can destroy identity and body consistency even when every file is technically bound. Anatomy-detail sources are preserved directly in the profile instead of being regenerated by FL2VA: frame-by-frame testing showed that transforming a seated or pitched anatomy source into a standing studio view invents and deforms geometry during the video transition. Missing anatomy evidence therefore stays missing until the user supplies a matching source image. Images held by the evidence gates are named in the confirmation and excluded from generation. Existing accepted images remain saved until the user explicitly replaces them. Explicit experimental Ref2VA modes remain available. The selected apparent age is injected into generated profile turns and later Studio renders as a stable face/skin/body-maturity anchor. H3 performs one slow transition to a requested face/body angle, holds the target view, and Avatar V2 extracts the final frame as a PNG candidate. Body regions absent from the source still have to be inferred, so generated views remain candidates that require visual acceptance.

The optional **Hybrid guided Ref2VA** route keeps identity slots on FL2VA and uses selected guide-only anatomy images for body/anatomy slots. It never treats guides as avatar identity: compiled H3 roles restrict them to clinical topology and explicitly reject transfer of face, age, ethnicity, skin, body identity, marks, tattoos, hair, or clothing. Guide records remain local file references and require provenance, adult-subject confirmation, usage-rights confirmation, and consent/model-release confirmation for real-person photos. Guided Ref2VA forces the INT8 low-memory profile, 352×608 canvas, match-sized references, and reduced steps, but remains experimental—not promised to fit—on an 8 GB GPU.

The separate **Identity + body refs · Ref2VA** route connects the selected primary source as the facial identity anchor and binds up to eight additional images per render, prioritizing the reference matching the requested body angle. Complete validation rotates through the full saved pool across its runs, so the per-render H3 limit does not silently exclude later images. Identity coverage remains on FL2VA unless a ready face-shape or skin option is selected; those identity slots use Ref2VA while retaining the primary source as the identity anchor. Body and anatomy coverage uses Ref2VA with visible bound-versus-available reference counts in each affected slot. This route also forces the INT8 low-memory profile, 352×608 canvas, match-sized references, and reduced steps, and remains experimental on an 8 GB GPU.

For simpler setup, **Quick Profile Builder** accepts either explicitly selected still images or one local reference-cache folder. Folder mode inventories still images, GIFs, videos, and audio; file mode analyzes only the still images the user selected. During the queue-idle analysis phase, the locally installed `qwen3-vl:8b-instruct` model is the recommended 8 GB default because its stricter identity and framing decisions prevent false coverage from becoming bad generations. The selected model inspects every selected still image and assigns every independently usable face, front, rear, side, top, low, torso/chest, pelvis/buttocks, hands, or feet role—not merely one dominant label per file. A region must be complete and clearly visible before it can constrain that slot. It also applies source-quality and non-identifying same-subject consistency gates: heavy occlusion, incomplete faces, low-confidence views, and visually conflicting subjects are held out of automatic generation. Phone UI and watermarks are reported as crop warnings but do not double-penalize otherwise clear, usable face or body geometry. The comparison never names or recognizes a person and remains restricted to the loopback Ollama service; the model is unloaded before H3 uses the GPU. A confirmed single-avatar cache may admit a usable body-only view whose face cannot be compared, but a visible face mismatch remains held.

After analysis, **AI-sorted reference galleries** show local thumbnails for every compatible image under each target region. The user can inspect the grouping, choose which image should be tried first, remove a wrong assignment, or add known face/body/anatomy images directly to a region. Galleries and preferred choices are saved with the avatar. The build confirmation reports how many eligible images were analyzed and scheduled, which were held, and any byte-identical duplicates. GIF and video remain motion examples; audio remains voice/timing input. The detailed per-angle fields, reusable anatomy references, prompts, and renderer controls remain available under Advanced panels.

Each avatar save also writes `~/.avatar_v2/library/manifests/<avatar-id>.avatar_manifest.json`. This structured manifest records analyzed image roles and confidence, preferred sources, identity/body anchors, source-protected anatomy slots, appearance constraints, and pending/accepted/rejected candidate history. After H3 is unloaded, the same local Qwen Vision service compares each extracted face/body candidate with the exact source that produced it and reports an advisory pass/review/reject result in Queue and History. It never auto-accepts a candidate: History provides explicit accept/reject actions, and generated anatomy cannot be promoted to source evidence.

The model selector also exposes locally installed 2B Q4, 2B Q8, 4B, and Qwen2.5-VL experiments. Coverage is not treated as correctness: on a ten-file failure reproduction, 2B Q8 admitted nine files and invented a full-body assignment, while 8B correctly found no full-body source and held six files whose visible identity conflicted with the anchor. The smaller models remain available for speed comparisons, but Quick Build defaults to the stricter 8B result.

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

> **Status 2026-11-21:** the table above describes the earlier deployment. The ComfyUI install
> at `/home/jd/AvatarForge` and its model weights are no longer on this machine, so every local
> lane is unavailable until ComfyUI is reinstalled. See
> [`docs/AVATAR_V2_CAPABILITY_AUDIT.md`](docs/AVATAR_V2_CAPABILITY_AUDIT.md).

## Runway cloud provider

The recipes send jobs to Runway's developer API:
photoreal presenter stills (text-only or anchored by tagged identity references), consistent
character variations, image-to-video (`gen4.5`, `gen4_turbo`, hosted `h3_max`), Act-Two
performance transfer, lip-synced talking-presenter videos from a script or narration
(`gwm1_avatars` with a custom avatar), and still upscaling.

Runway works as soon as an API key is saved (OS keyring, a user-only file, or
`RUNWAYML_API_SECRET`; never `desktop.json`; `scripts/set_runway_key.sh` saves one with hidden
input). There is no enable switch, upload-consent box or per-job confirmation: the price is shown
before you click, and jobs start on one click. Spending limits (per job, per batch, daily, monthly)
are optional and off unless you set them in Settings. It is never used as an
automatic fallback for local rendering, and the app never resubmits a paid job on its own. Cloud
jobs stay limited to general content and to avatars declared synthetic or consenting adults, so a
real person's likeness is never uploaded without their consent, and the presenter is always an adult:
prompts that describe someone under 18 are refused before anything is sent or billed.
Setup, costs and the presenter workflow: [`docs/RUNWAY_PROVIDER.md`](docs/RUNWAY_PROVIDER.md).

## Presenter production (BRAND Instructor)

The recipes turn Avatar V2 into a repeatable pipeline for the fictional presenter
`BRAND_INSTRUCTOR_PHOTOREAL_V1`:

- master-identity candidates you pick in a full-size viewer;
- nine reference slots whose chosen photos are recorded by hash (a different face is flagged, not blocked);
- narration that is imported, generated or designed (no voice cloning) and usable right away;
- talking-head and performance-transfer paths, where the performer's audio is never uploaded;
- long talking-video scripts split into parts automatically;
- a cost ledger (charged and unconfirmed spend, optional limits) and verified output hashes.

Settings and the recommended first render: [`docs/BRAND_INSTRUCTOR.md`](docs/BRAND_INSTRUCTOR.md).

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

The venv must use the same Python version as the system's GTK bindings (`gi`). On this machine,
after the OS moved to Python 3.14, that is `.venv314`:

```bash
python3.14 -m venv --system-site-packages .venv314
.venv314/bin/pip install -e '.[desktop,dev]'
.venv314/bin/python -m pytest -q
scripts/launch_avatar_v2.sh   # prefers dist/AvatarV2, then .venv314; AVATAR_V2_FROM_SOURCE=1 forces source
```

The older Python 3.12 `.venv` is kept as a rollback for the CLI and tests. Rollback paths are listed
in [`docs/RECOVERY_AND_ROLLBACK.md`](docs/RECOVERY_AND_ROLLBACK.md).

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
