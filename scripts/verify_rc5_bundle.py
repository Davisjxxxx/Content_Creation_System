"""Verify an existing onedir candidate against immutable Git blobs. Run with the build interpreter."""
import argparse
import hashlib
import json
import marshal
import os
from pathlib import Path
import subprocess

from PyInstaller.archive.readers import CArchiveReader


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle_identity(root):
    digest = hashlib.sha256()
    entries = 0
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for name in [*dirs, *sorted(files)]:
            path = Path(directory, name)
            if path.is_symlink():
                kind, value = "L", os.readlink(path)
            elif path.is_dir():
                kind, value = "D", ""
            else:
                kind, value = "F", digest_file(path)
            digest.update(f"{kind}\0{path.relative_to(root)}\0{value}\n".encode())
            entries += 1
    return entries, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("commit")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "--verify", args.commit + "^{commit}"], cwd=repo, text=True).strip()
    def blob(path):
        return subprocess.check_output(["git", "cat-file", "blob", f"{commit}:{path}"], cwd=repo)
    exe = args.bundle / "AvatarV2"
    archive = CArchiveReader(str(exe))
    pyz = archive.open_embedded_archive("PYZ.pyz")
    codes = {name: pyz.extract(name) for name in pyz.toc if name == "avatar_v2" or name.startswith("avatar_v2.")}
    codes["desktop_entry"] = marshal.loads(archive.extract("desktop_entry"))
    mismatches = []
    for name, code in sorted(codes.items()):
        path = "scripts/desktop_entry.py" if name == "desktop_entry" else code.co_filename
        if code != compile(blob(path), code.co_filename, "exec", dont_inherit=True, optimize=0):
            mismatches.append(name)
    required = {"avatar_v2." + name for name in (
        "media_validation", "safe_output", "render_queue", "ownership", "inspection_contract", "script_runner",
        "providers.comfyui", "providers.runway", "desktop", "desktop_studio", "ollama_client", "gpu_memory",
        "reference_import", "reference_picker", "workflow_router", "correction_engine", "voiceover", "script_broker",
        "video_inspector", "visual_inspector", "realism_qa")}
    missing = sorted(required - codes.keys())
    assets = ["avatar_v2/ui/index.html", "avatar_v2/data/prompt_libraries.json", "avatar_v2/sdk_client.py",
              "workflows/animatediff_sd15_api.json", "workflows/animatediff_sdxl_api.json",
              "workflows/wan22_i2v_14b_gguf_api.json", "workflows/wan22_i2v_5b_turbo_gguf_api.json"]
    asset_digests = {}
    for name in assets:
        checksum = digest_file(args.bundle / "_internal" / name)
        asset_digests[name] = checksum
        if checksum != hashlib.sha256(blob(name)).hexdigest():
            mismatches.append(name)
    count, tree = bundle_identity(args.bundle)
    print(json.dumps(dict(commit=commit, executable_sha256=digest_file(exe), bundle_sha256=tree,
                          tree_entries=count, application_code_objects=len(codes), mismatches=mismatches,
                          required_modules_missing=missing, assets=asset_digests), indent=2))
    return bool(mismatches or missing)


if __name__ == "__main__":
    raise SystemExit(main())
