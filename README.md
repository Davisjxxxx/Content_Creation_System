# Content Creation System — Avatar V2 CLI

Avatar V2 is a reference-driven synthetic-human video orchestration CLI. It separates identity, motion, scene, camera, wardrobe and physical-realism direction instead of asking one model to invent the entire clip from a single prompt.

The first vertical slice is deliberately **renderer-agnostic** and uses ComfyUI API-format workflows. That means you can benchmark Wan 2.2, LTX, custom LoRA stacks, motion-transfer workflows, or another ComfyUI-compatible graph without rewriting the CLI.

## What V2 does now

- Persistent avatar manifest with multiple identity/body/hair/wardrobe references.
- Shot manifest with timed motion beats, camera, wind, wardrobe and environment controls.
- Renderer routing hint (`wan22`, `ltx25`, `seedance`, `custom`) while keeping your supplied workflow authoritative.
- Generic ComfyUI workflow placeholder injection, including multi-reference bindings.
- Deterministic seeds and best-of-N batch planning.
- Local asset staging into the ComfyUI input directory.
- Queue/poll/output discovery through the ComfyUI HTTP API.
- Structural video QA with `ffprobe`.
- Adult-capable content classes with hard age/consent/provenance boundaries rather than a blanket NSFW disable switch.
- Unit tests for routing, prompting, policy and workflow substitution.

## Hard boundaries

Adult workflows are permitted only when every represented subject is an age-verified consenting adult or a synthetic adult persona with explicit adult provenance. Unknown provenance fails closed. Do not use this system for minors, non-consensual sexual content, or sexualized impersonation/deepfakes without the represented adult's consent.

## Install

Prerequisites:

- Python 3.11+
- `ffmpeg` / `ffprobe`
- ComfyUI for actual local rendering
- The checkpoints/custom nodes required by the particular Wan/LTX/etc. workflow you choose

```bash
git clone https://github.com/Davisjxxxx/Content_Creation_System.git
cd Content_Creation_System
git checkout avatar-v2-cli
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Fast CLI test without a GPU or real assets

The shipped example paths are intentionally placeholders. Planning and workflow dry-runs do not require those files to exist.

```bash
avatar-v2 validate examples/avatar.synthetic.yaml examples/shot.realism.yaml
avatar-v2 plan examples/avatar.synthetic.yaml examples/shot.realism.yaml -o build/job.json
avatar-v2 plan-batch examples/avatar.synthetic.yaml examples/shot.realism.yaml -n 4
avatar-v2 render build/job.json examples/provider.comfyui.yaml \
  --workflow workflows/mock_placeholder_workflow.json \
  --dry-run
pytest
```

The resolved workflow will be written to `build/resolved_workflow.json`; inspect it to confirm prompts, timed motion instructions, reference bindings, resolution, FPS, frame count and seed substitution.

## Connect ComfyUI for an actual render

Edit `examples/provider.comfyui.yaml` with the real absolute paths to your ComfyUI `input` and `output` directories.

```bash
avatar-v2 doctor examples/provider.comfyui.yaml
```

In ComfyUI, load the exact Wan/LTX workflow you want, export it using **Save (API Format)**, and replace the values V2 should control with placeholders described in [`workflows/README.md`](workflows/README.md).

Resolve your real workflow before sending it to ComfyUI:

```bash
avatar-v2 render build/job.json examples/provider.comfyui.yaml \
  --workflow /path/to/wan22_i2v_api.json \
  --dry-run
```

Then run the actual generation:

```bash
avatar-v2 render build/job.json examples/provider.comfyui.yaml \
  --workflow /path/to/wan22_i2v_api.json
```

Inspect an output video:

```bash
avatar-v2 qa /path/to/output.mp4
```

## Recommended V2 test sequence

1. **Identity lock** — use one avatar and a simple 4–6 second portrait movement.
2. **Motion-reference test** — same avatar/start frame, add a reference video.
3. **Hair/wind test** — same scene with 0, 3 and 6 m/s wind descriptions.
4. **Wardrobe continuity test** — walking/turning clip with loose outerwear.
5. **First/last-frame test** — run the same identity through an LTX first/last-frame workflow.
6. **Best-of-N test** — render four seeds and retain only clips passing identity/anatomy/temporal QA.
7. Only after those pass, add more complex scenes, multiple people, speech/audio and heavier LoRA stacks.

## Repository map

```text
avatar_v2/
  cli.py                 CLI entry point
  composer.py            prompt + reference job composer
  models.py              manifests/contracts
  policy.py              adult-capable hard-boundary policy
  qa.py                  ffprobe structural QA
  router.py              renderer selection hints
  providers/comfyui.py   generic ComfyUI API adapter
examples/                 starter manifests
workflows/                placeholder docs + dry-run fixture
tests/                    unit tests
```

V2 intentionally does not lock the project to one checkpoint or LoRA. The workflow file remains the source of truth for the actual renderer graph, sampler, scheduler, model, LoRA weights and custom nodes so those can be iterated independently of the orchestration system.
