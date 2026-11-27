# Avatar V2 RC3 — Release Readiness & Rollback Report

## Explicit Disposition

**`RC3_READY_FOR_INDEPENDENT_RECHECK`**

### Basis for Disposition

All **23 original Astra RC1 findings** — including both P0 failures (`RC-01`, `RC-02`) and all 15 remaining partial/open items identified in Astra's independent RC2 recheck (`c63fb9f`) — have been remediated at the underlying invariant level on dedicated branch `rc3-astra-remediation` without expanding product scope:

- **6 / 23 findings:** *Independently confirmed fixed* by Astra at `c63fb9f` (`RC-05`, `RC-08`, `RC-20`, `RC-21`, `RC-22` with RC3 active-job balance polish, `RC-23`).
- **17 / 23 findings:** *Corrected but awaiting independent verification* (`RC-01`, `RC-02`, `RC-03`, `RC-04`, `RC-06`, `RC-07`, `RC-09`, `RC-10`, `RC-11`, `RC-12`, `RC-13`, `RC-14`, `RC-15`, `RC-16`, `RC-17`, `RC-18`, `RC-19`).
- **0** *Partially corrected*, **0** *Open*, **0** *Not reproducible*.
- **Automated & Adversarial Suite:** **`469 / 469`** passing under a clean isolated `HOME` (`0` failed, `0` skipped, zero implicit `/home/jd/AvatarForge` dependency).
- **Isolated RC3 Bundle (`dist/AvatarV2.rc3`):** **`58 / 58`** `avatar_v2.*` PYZ bytecode correspondence, **`7 / 7`** bundled data/workflow SHA-256 matches, and packaged `AVATAR_V2_UITEST=1` passed from `/tmp` with **`0` JS errors**.

Per governance rules, this candidate is **not** promoted, merged, deployed, or authorized for owner testing against original media; independent recheck remains the next external gate.

---

## Candidate & Provenance Identity

- **Remediation Branch:** `rc3-astra-remediation`
- **Branched From (RC2 Source Commit, preserved):** `87ce485e7879f1ea085596dece66a95d9c60b9ff` (`rc1-astra-remediation`)
- **Original RC1 Source Commit (preserved):** `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12` (`sprint-1-gpu-memory-foundation`)
- **Independent Audit References (preserved):**
  - RC1 Audit: `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` (`audit/astra-rc1-independent`)
  - RC2 Recheck: `c63fb9fb7d4598db2a122b16b4c3a03feb1a2316` (`audit/astra-rc2-recheck`)
- **RC3 Remediation Commits Packaged in `dist/AvatarV2.rc3`:**
  - `f4716a0` — Gates 1 & 2: Durable managed-asset provenance (`ManagedAssetRegistry`), endpoint-bound renderer ownership (`verify_endpoint_ownership`), atomic restart `pidfd` signalling, and complete `JsBridge` candidate path authority (`RC-01`, `RC-02`, `RC-03`)
  - `d4127c3` — Gate 3: Durable pre-submission persistence (`phase="submitted"` + `submission_token`), fail-closed file-locked script budgets, strict terminal ComfyUI `wait()`, single-writer lock held until `_worker` exits, and `expected_type` JSON schema validation (`RC-10`, `RC-11`, `RC-16`, `RC-18`)
  - `0e9fe3f` — Gate 4: Explicit paid/local video engine selection, locked reference enforcement & truthful Wan `verification_record`, multi-count & multi-sentence requirement extraction, non-fallback per-output UI/backend verdicts, end-to-end `asset_fingerprint` preservation, durable `.pins.json` with finish-time release, and removal of automatic job-transition balance fetch (`RC-04`, `RC-06`, `RC-07`, `RC-09`, `RC-19`, `RC-22`)
  - `0c03089` — Gate 5: Model-matched Wan 5B vs 14B correction workflow & placeholders, exact resolved-graph receipt extraction + still `frames=1` + `ffprobe` telemetry, bidirectional Ollama↔ComfyUI VRAM handoff & verified pre-VLM unload, fail-closed broker capabilities when `comfyui_dir` is missing, recursive `/proc` descendant tree + aggregate RSS limits, and `AVATAR_V2_RC3=1` launcher support (`RC-12`, `RC-13`, `RC-14`, `RC-15`, `RC-17`)
  - `4472e75` — Test portability fix removing implicit `~/AvatarForge` dependency under clean isolated `HOME`
- **Candidate Bundle Path:** `dist/AvatarV2.rc3`
- **Executable SHA-256 (`dist/AvatarV2.rc3/AvatarV2`):** `cb3fad889b41b10856ff1dfacf9111b7589905943f89d554d33566d59cf4f48e`
- **Deterministic Onedir Tree Digest (`dist/AvatarV2.rc3/`, `42,708` entries):** `473c80d2a1d8f66b8142ba803435f0d8adc01ef40f35bda20f95fccef2fc8540`

### Exact Reproducible Tree-Digest Verification Recipe

To independently reproduce the exact `473c80d2a1d8f66b8142ba803435f0d8adc01ef40f35bda20f95fccef2fc8540` digest for `dist/AvatarV2.rc3`, hash sorted `(rel_posix_path, type, value)` records serialized as `f"{type}\0{rel}\0{val}\n"` (`"D"` for directory with `val=""`, `"L"` for symlink with `val=os.readlink(p)`, `"F"` for regular file with `val=sha256_hex(bytes)`):

```python
import hashlib, os, pathlib

bundle = pathlib.Path("dist/AvatarV2.rc3")
entries = []
for root, dirs, files in os.walk(bundle):
    dirs.sort()
    files.sort()
    for name in dirs:
        p = pathlib.Path(root) / name
        rel = p.relative_to(bundle).as_posix()
        entries.append((rel, "L", os.readlink(p)) if p.is_symlink() else (rel, "D", ""))
    for name in files:
        p = pathlib.Path(root) / name
        rel = p.relative_to(bundle).as_posix()
        if p.is_symlink():
            entries.append((rel, "L", os.readlink(p)))
        else:
            fh = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    fh.update(chunk)
            entries.append((rel, "F", fh.hexdigest()))
entries.sort(key=lambda x: x[0])
h = hashlib.sha256()
for rel, typ, val in entries:
    h.update(f"{typ}\0{rel}\0{val}\n".encode("utf-8"))
print(h.hexdigest(), len(entries))
```

---

## Launch & Rollback Instructions (Non-Destructive)

Production (`dist/AvatarV2`), RC1 (`dist/AvatarV2.next`), RC2 (`dist/AvatarV2.rc2`), and owner data (`~/.avatar_v2`) were **never modified or promoted**.

### Launch RC3 in Isolated Evaluation Mode

```bash
AVATAR_V2_HOME=/tmp/avatar_v2_rc3_eval AVATAR_V2_RC3=1 /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

Or via explicit candidate directory:

```bash
AVATAR_V2_HOME=/tmp/avatar_v2_rc3_eval AVATAR_V2_CANDIDATE_DIR=/home/jd/avatar_project/dist/AvatarV2.rc3 /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

### Rollback / Continue Using Production Unchanged

Production remains the default target of `scripts/launch_avatar_v2.sh` (`dist/AvatarV2/AvatarV2`, SHA-256 `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893`). To remove RC3 cleanly without affecting production, RC1, or RC2:

```bash
rm -rf /home/jd/avatar_project/dist/AvatarV2.rc3
```

---

## Explicitly Deferred / Out-of-Scope Capabilities (Preserved)

- **Sprint 11 / new product scope:** Not started.
- **Autonomous mask detection:** Masked editing (`builtin:flux2_masked_edit`) requires an explicit user- or harness-supplied mask with recorded provenance.
- **Wan 2.2 FunControl / ControlNet / OpenPose / DWPose:** Deferred and excluded from executable script model capabilities.
- **`A1111CompatibilityAdapter`:** Explicitly `source_only_optional` (`RC-21`); not exposed as a packaged desktop service.
- **Secondary NTFS drive migration (`/dev/nvme1n1`):** Untouched and out of scope.

---

## Related Audit Artifacts

- Full 23-Finding RC3 Tracking Matrix: [RC3_REMEDIATION_MATRIX.md](file:///home/jd/avatar_project/docs/audit/RC3_REMEDIATION_MATRIX.md)
- RC3 Targeted Acceptance & Adversarial Evidence: [RC3_TARGETED_ACCEPTANCE.md](file:///home/jd/avatar_project/docs/audit/RC3_TARGETED_ACCEPTANCE.md)
- Prior Preserved Audit Artifacts: [RC1_REMEDIATION_MATRIX.md](file:///home/jd/avatar_project/docs/audit/RC1_REMEDIATION_MATRIX.md), [RC2_TARGETED_ACCEPTANCE.md](file:///home/jd/avatar_project/docs/audit/RC2_TARGETED_ACCEPTANCE.md), [RC2_RELEASE_READINESS.md](file:///home/jd/avatar_project/docs/audit/RC2_RELEASE_READINESS.md)
