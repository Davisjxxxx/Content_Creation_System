# Claude Code / DeepSeek deployment prompt

Run this prompt from the repository root while checked out to `avatar-v2-h3-desktop`.

```text
You are the implementation and deployment engineer for Avatar V2. Work directly in the current Git repository and complete the deployment on this machine. Do not redesign the project from scratch. Inspect the existing branch first and preserve the architecture, content boundaries, local-first privacy model, H3 license gate, runtime-profile system, and Web3/neon-heart visual identity.

GOAL
Deliver a working physical Windows desktop application named "Avatar V2" from the existing `avatar-v2-h3-desktop` branch. It must launch from an .exe on the desktop, connect to local ComfyUI, expose MiniMax H3 Ref2VA / FL2VA plus Wan 2.2 fallback, and let me toggle runtime profiles to empirically find what works on my GPU.

NON-NEGOTIABLE PRODUCT DESIGN
- This is a real desktop app, not a browser tab and not a Gradio demo.
- Keep the black Web3-style visual system already implemented.
- Preserve the glowing pink/red heart brand mark and icon.
- Keep the interface clean, premium, restrained and production-oriented.
- Preserve separate identity/body/hair/wardrobe/motion/scene/audio references.
- Preserve H3 positional reference binding and timed motion prompting.
- Preserve adult-capable classes only for verified consenting adults or synthetic adult personas; do not weaken the hard boundaries around minors, non-consent, or non-consensual sexualized impersonation.
- Do not add analytics, telemetry, public sharing, or silent remote uploads.

SOURCE OF TRUTH
1. Read README.md.
2. Read workflows/README.md.
3. Read avatar_v2/desktop.py, runtime_profiles.py, h3_runtime.py, h3_access.py, composer.py, render_queue.py, providers/comfyui.py, policy.py, and ui/index.html.
4. Read and run the tests before changing implementation.
5. Use the existing code whenever it is correct; make focused fixes only where execution proves they are needed.

FIRST: AUDIT THE MACHINE
Run and record:
- git status / branch / HEAD
- python --version and py launcher versions
- nvidia-smi, GPU name, VRAM, driver version
- total system RAM
- free disk space on the system drive and the drive holding ComfyUI/model files
- ffmpeg and ffprobe availability
- whether Microsoft Edge WebView2 Runtime is installed
- where ComfyUI is installed, if discoverable
- whether ComfyUI is already running on 127.0.0.1:8188

Do not assume the machine has 32 GB VRAM. This project is specifically intended to test practical configurations on an RTX 4070-class 12 GB GPU.

H3 RUNTIME PROFILES
Preserve and verify these UI choices:
- Auto
- H3 Quality
- H3 INT8 · 12 GB
- H3 Low Memory
- H3 Low Memory · Quality
- Wan 2.2 Fallback

The point is empirical testing, not pretending one configuration is guaranteed to fit. Each run must retain runtime profile, exact diffusion model filename, canvas, valid H3 frame count, steps, seed, elapsed time, outputs, and renderer error if it fails.

H3 TECHNICAL RULES
Preserve the H3-native constraints already encoded in the repo:
- 24 fps
- multiple-of-32 canvas presets
- valid frame count on the 17k+5 grid
- 15-second single-clip practical cap
- 480x864 vertical and 864x480 landscape as primary 12 GB-class presets
- smaller 352x608 / 608x352 low-memory presets

Treat Ref2VA and FL2VA separately. Existing community testing indicates FL2VA can fit 12 GB significantly more readily than Ref2VA. Low-memory Ref2VA is an experiment and may still OOM due to fixed overhead. Surface that truth in logs rather than hiding it.

H3 MODEL / LICENSE RULE
Do NOT automatically download MiniMax H3 weights.
Do NOT bypass the application's local_h3_authorized gate.
If local H3 open-weight use is not explicitly authorized in app settings, leave local H3 execution disabled and keep Wan 2.2 as the local fallback. Hosted H3 may be added only as a separate explicit provider with clear user consent and no silent data transfer.
If local H3 is explicitly authorized and compatible H3 weights are already present, discover them through ComfyUI instead of hard-coding paths.

COMFYUI INTEGRATION
1. Start or connect to ComfyUI locally.
2. Use the ComfyUI HTTP API for health, node/model discovery, uploads, queueing, polling and interrupt.
3. Confirm the official MiniMax H3 nodes are available in this ComfyUI build before calling H3 ready.
4. Inspect `/object_info` and model lists and report exact H3 diffusion/text-encoder/VAE files ComfyUI can see.
5. The app must not require direct filesystem access to ComfyUI/input when HTTP upload works.
6. If a real API-format H3 Ref2VA or FL2VA workflow is not yet present in this repository, create/export one from the installed ComfyUI version rather than inventing stale node schemas. Bind the fields Avatar V2 should control using placeholders documented in workflows/README.md, including `${PROMPT}`, `${WIDTH}`, `${HEIGHT}`, `${FRAMES}`, `${SEED}`, `${STEPS}`, `${CFG}`, `${H3_DIFFUSION_MODEL}`, `${H3_PICTURE_N}`, `${H3_VIDEO_N}`, and `${H3_AUDIO_N}` as applicable.
7. Keep separate workflow files for Ref2VA, FL2VA and Wan.

TEST ORDER
Do not start with the hardest reference workload.
Gate tests in this order:
A. Unit tests and compile checks.
B. Launch desktop UI with no renderer.
C. Connect to ComfyUI and run Doctor/model discovery.
D. H3 FL2VA dry run with placeholders only.
E. H3 FL2VA smallest real render if locally authorized and weights exist.
F. H3 INT8 · 12 GB at the recommended 12 GB canvas.
G. H3 Low Memory at the fast canvas.
H. H3 Ref2VA with one identity reference only.
I. Ref2VA with identity + motion reference.
J. Add scene/audio only after earlier gates succeed.
K. Run a Profile Sweep and compare actual failures/speed/output.
L. Verify Wan fallback works if its workflow/models are installed.

For any CUDA OOM or Windows memory error, preserve the exact error and diagnose whether the bottleneck is VRAM, system RAM, pagefile, or disk space. Do not blindly retry indefinitely. Record the smallest configuration that passes and the largest configuration that fails.

DESKTOP BUILD
Use the existing scripts/build_windows.ps1 path. Fix it only if actual execution exposes a problem.
Required output:
`dist\AvatarV2\AvatarV2.exe`

Verify:
- no console window opens with the app
- packaged `avatar_v2/ui/index.html` loads correctly
- the generated glowing-heart icon is used
- file/folder pickers work
- Settings persist under the user's home directory
- the app launches without requiring the source tree as cwd
- ComfyUI can remain a separately updateable local service

Then create a Windows desktop shortcut named `Avatar V2` pointing to the built executable and using the generated icon. Do not delete or overwrite unrelated desktop files.

QUALITY / SECURITY CHECKS
- Run `python -m compileall -q avatar_v2 scripts`.
- Run all pytest tests.
- Review for accidental secrets, hard-coded tokens, remote analytics, public bind addresses, or GitHub credentials. None should be required.
- Ensure private avatar assets, videos and render outputs remain ignored by Git.
- Do not commit model weights.
- Do not commit generated private media.
- Do not expose ComfyUI outside localhost unless explicitly requested.

ACCEPTANCE CRITERIA
Do not call the deployment complete until all applicable items below are evidenced:
1. Tests pass.
2. Desktop app launches.
3. Web3 black / pink-red heart UI renders correctly.
4. Settings persist.
5. ComfyUI Doctor works or returns a precise actionable failure.
6. Runtime-profile toggle works.
7. Profile Sweep queues distinguishable jobs and records per-profile outcomes.
8. Dry-run substitution works for configured workflow(s).
9. At least one real renderer path is executed if compatible authorized models/workflows are present.
10. Physical Windows .exe exists.
11. Desktop shortcut exists and opens the app.
12. No private repo token/model weight/private media has been added to Git.

VERSION CONTROL
- Stay on `avatar-v2-h3-desktop` unless there is a compelling reason to create a child branch.
- Do not merge to main.
- Do not rewrite history.
- Commit focused fixes with meaningful messages only after tests pass.
- At completion, report starting SHA, ending SHA, changed files, test results, app build path, shortcut path, ComfyUI/H3 detection, exact runtime profiles tested, which configurations passed/failed, and any remaining blockers.

Begin now by inspecting the repo and machine state. Do not ask me to manually perform steps you can perform from the CLI. Only stop for a genuinely external requirement such as a license authorization choice, a missing credential for an explicitly requested hosted provider, or hardware/model files that are not present and cannot legally/safely be obtained automatically.
```
