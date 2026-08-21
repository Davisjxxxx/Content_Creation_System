# Avatar V2 implementation instructions

When using Claude Code or another CLI coding agent in this repository, preserve the existing Avatar V2 architecture and inspect the implementation before replacing anything.

## GPU memory management is mandatory

The desktop branch includes `avatar_v2/gpu_memory.py`, derived from the user's GPU auto-cleaner concept. It is part of the production design, not optional cleanup code.

Before declaring a Windows/GPU deployment complete:

1. Read `docs/GPU_MEMORY_MANAGER.md`.
2. Verify `nvidia-smi` memory detection on the target machine.
3. Verify automatic pre/post render cleanup through ComfyUI's `POST /free` endpoint.
4. Confirm `int8_12gb`, `low_memory`, and `low_memory_quality` force a clean-memory preflight.
5. Run a profile sweep and preserve per-job cleanup events and peak VRAM traces.
6. Do not replace the safe manager with the broad process-killing behavior in `tools/gpu_cleaner_legacy.py`.
7. Never automatically kill arbitrary Python GPU processes. Emergency termination must require explicit approved PIDs.
8. Preserve environment controls `AVATAR_V2_GPU_AUTOCLEAN`, `AVATAR_V2_GPU_THRESHOLD`, and `AVATAR_V2_GPU_CHECK_INTERVAL`.
9. If CUDA/PyTorch reports error 999 while `nvidia-smi` still works, consult `docs/CUDA_ERROR_999_QUICK_REFERENCE.txt`, but do not automatically execute privileged driver/group/reboot commands.

The intended default threshold is 85% VRAM. Automatic cleanup should be observable in logs/queue state and should never silently destroy an unrelated process.

## Deployment source

Also follow `docs/CLAUDE_DEEPSEEK_DEPLOY_PROMPT.md` for the broader desktop deployment, H3 runtime-profile tests, ComfyUI integration, packaging, and acceptance gates.
