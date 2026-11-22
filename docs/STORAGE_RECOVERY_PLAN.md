# Storage recovery plan (local rendering deferred)

Measured 2026-11-22, read-only. **Nothing was deleted, cleaned or downloaded.** Every action below is
a recommendation for the owner to decide; commands are examples to run deliberately, not automation.

## Current state

| Item | Size |
|---|---|
| Root filesystem `/dev/nvme0n1p2` | 915 GB, **811 GB used, 58 GB free (94%)** |
| `/home/jd` | 543 GB (about 268 GB is outside home: system, `/usr`, `/var`, snaps) |
| Ollama models (`/usr/share/ollama/.ollama/models`) | 78 GB |
| `~/.cache` | 52 GB: Hugging Face 24, uv 10, nova-starbound 4, Chrome 3 |
| `/home/jd/avatar_project_legacy` (restored here; must be kept) | 61 GB: `webui` 37, Forge 8, `webui_cuda_isolated` 7, `training_dataset` 4, `models` 3 |
| `~/miniconda3` | 22 GB (package cache 6 GB) |
| `~/avatar_studio` | 29 GB |
| `~/.codex-BACKUP-20261019-*` (two copies) | 16 GB |
| Other large project folders | `learning-evidence-os` 55, `Desktop` 33, `voicebox` 27, `Downloads` 18, `VoiceStudio` 18 GB |
| Second drive `nvme1n1` (954 GB) | NTFS `OS` partition, 926 GB (Windows, not mounted); `RESTORE` partition 26 GB (8 GB free) |
| Added by the recovery pass | `.venv314` 74 MB, new `dist/AvatarV2` 346 MB, `build/py314` 47 MB; previous package kept (364 MB) |

## What local rendering needs again

The earlier local install (ComfyUI, its venv, and the H3/Wan/Qwen weights) is gone. Restoring it
needs roughly:

| Component | Approx. size |
|---|---|
| ComfyUI and its Python/torch environment | 10-15 GB |
| H3 INT8 set: FL2VA and Ref2VA (about 21 GB each), Qwen3-VL text encoder (about 15 GB), video and audio VAEs, turbo LoRAs | about 62-65 GB |
| Wan 2.2 Fun-Control 5B (diffusion model, text encoder, VAE) | about 10-12 GB |
| Working headroom (outputs, staging, swap growth) | about 20 GB |
| **Total** | **about 105-110 GB free** before reinstalling; 58 GB is free today |

H3 weights are license-gated. Re-download them only with explicit authorization, and only after the
space exists.

## Options, safest first

1. **Add storage instead of deleting.** A dedicated internal or external SSD (1 TB or more), mounted at
   for example `/mnt/models`, holding ComfyUI's `models/` directory, is the cleanest fix. The Windows
   `OS` partition on the second NVMe is an operating-system volume; avoid putting Linux model storage
   there unless you deliberately repartition.
2. **Reclaim regenerable caches** (no project data). Together these come to about 16-40 GB:
   - `uv cache clean` (about 10 GB);
   - `conda clean --all` (about 6 GB);
   - Chrome's cache (about 3 GB);
   - selectively prune `~/.cache/huggingface` with `huggingface-cli delete-cache`, which lets you
     choose models and shows sizes first (up to 24 GB).
3. **Review Ollama models** (`ollama list`). Avatar V2 uses `qwen3-vl:8b-instruct`, and
   `qwen3-vl:2b-instruct` as the fast option. Candidates to review include `mythomax-local` and
   `mythomax-l2` (7.9 GB each, possibly duplicates), and models other projects may need
   (`deepseek-r1:8b`, `llama3.1:8b`, `qwen2.5-coder:7b`). Remove one with `ollama rm <name>` only
   after confirming no project uses it.
4. **Deduplicate identical checkpoints.** Three copies of the SD 1.5 checkpoint have identical size
   (4,265,146,304 bytes): `~/avatar_studio/models/sd_v15.safetensors`,
   `~/avatar_studio/core/webui/models/Stable-diffusion/sd_v15.safetensors` and the legacy
   `v1-5-pruned-emaonly.safetensors`. Confirm they match with `sha256sum`, then replace extras with
   hard links (`ln -f`) to keep every path working (saves about 8.5 GB). Do not remove the legacy
   copy.
5. **Archive, then remove, old backups** you no longer need, after verifying the archive: the two
   October `.codex-BACKUP-*` copies (16 GB) and `~/Backups` (11 GB).
6. **The preserved package** `dist/AvatarV2-20260903-py312` (364 MB) no longer launches on this OS
   (see `RECOVERY_AND_ROLLBACK.md`), so it is an archive only. Keep it until the new build has been
   used for a while.

Options 1 and 2 alone can restore the about 110 GB needed without touching any project or legacy
data.
