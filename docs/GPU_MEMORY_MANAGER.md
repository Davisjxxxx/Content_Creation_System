# GPU Memory Manager

Avatar V2 includes a production-safe GPU memory manager derived from the user-provided `gpu_cleaner.py` concept.

## Why the original script is not invoked automatically

The uploaded reference script monitors `nvidia-smi`, calls local Python/PyTorch/TensorFlow cache cleanup, and in aggressive mode terminates any Python GPU process using at least 1 GB of VRAM. That last behavior is too broad for the desktop app because it could kill ComfyUI, a concurrent renderer, or unrelated Python work.

Avatar V2 therefore keeps process termination out of the automatic path.

## Active behavior

`avatar_v2/gpu_memory.py`:

- reads all visible NVIDIA GPUs through `nvidia-smi`;
- records VRAM used/total and utilization fraction;
- samples VRAM during each render and records per-GPU peak usage;
- defaults to an 85% cleanup threshold;
- runs a pre-render cleanup when the threshold is exceeded;
- forces pre-render cleanup for `int8_12gb`, `low_memory`, and `low_memory_quality` profiles;
- requests post-render cleanup after failed/cancelled renders and low-memory runs;
- requests ComfyUI cleanup through `POST /free` with `unload_models=true` and `free_memory=true`;
- records before/after memory snapshots and cleanup errors in queue state;
- never automatically kills arbitrary Python processes.

The ComfyUI `/free` path matters because `torch.cuda.empty_cache()` in the Avatar V2 controller process cannot clear allocator/cache memory owned by the separate ComfyUI process.

## Environment controls

```text
AVATAR_V2_GPU_AUTOCLEAN=1
AVATAR_V2_GPU_THRESHOLD=0.85
AVATAR_V2_GPU_CHECK_INTERVAL=5
```

Set `AVATAR_V2_GPU_AUTOCLEAN=0` to disable automatic cleanup without changing code.

## Emergency process termination

`GPUMemoryManager.emergency_terminate()` accepts only explicitly approved PIDs. It does not discover-and-kill every Python process. A future UI emergency action must show the candidate GPU processes and require the user to select/confirm the PID(s) before termination.

## Profile-sweep data

Every queued render now retains:

- runtime profile;
- seed;
- renderer result/error;
- cleanup events;
- before/after VRAM snapshots;
- sampled peak VRAM per GPU;
- elapsed time.

This makes H3 full/INT8/low-memory profile comparisons evidence-driven on the target GPU.

## Legacy reference

The original uploaded auto-cleaner is preserved separately under `tools/gpu_cleaner_legacy.py` for reference/manual use. It is intentionally not imported by the desktop app.
