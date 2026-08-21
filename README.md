# Avatar V2 — H3 Desktop Studio

Avatar V2 is a local-first desktop application and CLI for reference-driven synthetic-human video production. It separates identity, body, hair, wardrobe, motion, scene, audio, camera and renderer configuration rather than asking one video model to invent everything from a single prompt.

The desktop branch is **MiniMax H3-first** with Wan 2.2 as the local fallback. It communicates with ComfyUI over HTTP and keeps ComfyUI workflow graphs authoritative, so model checkpoints, LoRAs, samplers and nodes can evolve independently of the application.

## Desktop experience

The Windows app uses a black Web3-style interface with a glowing pink/red heart identity. It exposes:

- MiniMax H3 Ref2VA and FL2VA lanes.
- Wan 2.2 and custom ComfyUI workflow lanes.
- Separate identity/body/hair/wardrobe reference packs.
- Motion video, scene, first/last frame and audio references.
- H3 positional reference binding (`<Picture N>`, `<Video N>`, `<Audio N>`).
- Timed motion prompting and physical hair/fabric/wind direction.
- Adult-capable content classes with hard age/consent/provenance boundaries.
- ComfyUI model/node discovery.
- Runtime profiles for trying different memory/quality setups.
- Sequential render queue with cancellation.
- Profile Sweep for empirical 4070 testing.
- Local job/run records under `~/.avatar_v2`.

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

**Profile Sweep** queues Auto, INT8, Low Memory and Low Memory · Quality with different deterministic seeds. Each queue record retains the exact profile, model, canvas, seed, timing and result/error so the working configuration can be determined experimentally.

### Important 12 GB H3 note

The H3 community UI we reviewed reports that FL2VA text/image-to-video works on a 12 GB RTX 3060 around 864×480, while Ref2VA remained roughly 1 GB over the card's available VRAM even after reducing resolution, steps, frame count and reference size. That makes the low-memory Ref2VA profiles genuine experiments, not a promise that Ref2VA will fit on 12 GB. If they still fail, Avatar V2 can move the shot to Wan locally or a separately configured larger/hosted renderer.

## H3-native rules incorporated into V2

Avatar V2 now handles H3's practical generation constraints:

- 24 fps.
- Dimensions from multiple-of-32 presets.
- Valid frame counts snapped to the `17k+5` grid.
- Single H3 clip duration capped at 15 seconds.
- Consumer presets including `480×864` vertical and `864×480` landscape.
- ComfyUI runtime model discovery rather than hard-coded filenames.
- Optional last-frame chaining infrastructure for longer sequences.

## Local H3 license gate

Avatar V2 **does not download MiniMax H3 weights automatically**. Local H3 execution is disabled by default. The user/admin must explicitly confirm in Settings that local open-weight use is permitted by the current MiniMax H3 license or covered by a separate written MiniMax license.

This is important because the current H3 Community License includes territorial restrictions. If local H3 is not enabled, the app can use the configured Wan local fallback instead. Hosted/API H3 can be added as a separate explicit provider rather than silently sending local prompts or avatar assets off-machine.

## Hard content boundaries

Adult workflows are supported only when every represented subject is an age-verified consenting adult or a synthetic adult persona with explicit adult provenance. Unknown provenance fails closed. Do not use the system for minors, non-consensual sexual content, or sexualized impersonation/deepfakes without the represented adult's consent.

## Install for development

Prerequisites:

- Windows 10/11 or Linux desktop
- Python 3.11+
- ComfyUI
- `ffmpeg` / `ffprobe`
- WebView2 Runtime on Windows (normally already present)
- Renderer checkpoints/workflows you are authorized to use

```bash
git clone https://github.com/Davisjxxxx/Content_Creation_System.git
cd Content_Creation_System
git checkout avatar-v2-h3-desktop
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[desktop,dev]"
avatar-v2-desktop
```

Linux/macOS shell:

```bash
source .venv/bin/activate
pip install -e '.[desktop,dev]'
avatar-v2-desktop
```

## Build the physical Windows desktop app

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

The build script:

1. creates/uses `.venv`,
2. installs desktop + build dependencies,
3. generates the black/pink/red glowing-heart app icon,
4. packages the UI and Python backend with PyInstaller,
5. produces:

```text
dist\AvatarV2\AvatarV2.exe
```

The generated `.exe` is the physical desktop launcher. ComfyUI remains a separate local renderer service so you can update GPU/model stacks without rebuilding the entire app.

## Connect ComfyUI

Open **Settings** in Avatar V2 and configure:

- ComfyUI base URL, normally `http://127.0.0.1:8188`
- optional ComfyUI input folder
- ComfyUI output folder
- H3 Ref2VA API-format workflow
- H3 FL2VA API-format workflow
- Wan 2.2 API-format workflow

If the input directory is omitted, Avatar V2 can stage inputs through ComfyUI's HTTP upload endpoint. This also allows a separate ComfyUI machine on your LAN without giving that machine access to this GitHub repository.

See [`workflows/README.md`](workflows/README.md) for supported placeholders and H3 bindings.

## CLI remains available

The V2 CLI from the previous branch remains intact:

```bash
avatar-v2 validate examples/avatar.synthetic.yaml examples/shot.realism.yaml
avatar-v2 plan examples/avatar.synthetic.yaml examples/shot.realism.yaml -o build/job.json
avatar-v2 plan-batch examples/avatar.synthetic.yaml examples/shot.realism.yaml -n 4
pytest
```

## Repository map

```text
avatar_v2/
  desktop.py             pywebview desktop backend
  ui/index.html          Web3/neon heart desktop UI
  h3_runtime.py          H3 frame/canvas constraints + presets
  h3_access.py           model discovery and license-aware recommendation
  runtime_profiles.py    Auto / quality / INT8 / low-memory profiles
  render_queue.py        sequential render queue + chaining seam
  composer.py            timed prompt + reference binding compiler
  models.py              avatar/shot/job contracts
  policy.py              adult-capable hard boundaries
  providers/comfyui.py   ComfyUI HTTP client, discovery, upload/render
  cli.py                 CLI entry point
scripts/
  build_windows.ps1      one-command Windows app package
  make_icon.py           glowing-heart .ico generator
  desktop_entry.py       PyInstaller entry point
workflows/
  README.md              workflow placeholder contract
```

## Design rule

The application should remain **renderer modular**. H3 is currently the primary multimodal experiment, not a permanent dependency. Better engines can be added behind the same identity/reference/job contracts without rebuilding the avatar library or production workflow.
