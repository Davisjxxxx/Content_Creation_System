# Avatar V2 RC2 — Release Readiness & Rollback Report

## Explicit Disposition

**`RC2_READY_FOR_TARGETED_INDEPENDENT_RECHECK`**

### Basis for Disposition

All **23 independently confirmed Astra RC1 findings** (`2 P0`, `16 P1`, `4 P2`, `1 P3`) have been remediated on isolated branch `rc1-astra-remediation`, verified by **447 / 447 passing automated & adversarial tests**, validated in the freshly packaged isolated **`dist/AvatarV2.rc2`** bundle (`58 / 58` PYZ bytecode correspondence, `33 / 33` packaged UI self-test checks with `0` JS errors), and confirmed via targeted live RTX 4070 acceptance trials (`owned_private` process lifecycle, FLUX.2 ground-truth receipts, live `builtin:flux2_masked_edit` with `0 / 796,438` exact-black-mask drift, and live ComfyUI → `qwen3-vl:8b-instruct` → verified `status="released"` VRAM handoff).

Per project governance, `RC2_READY_FOR_TARGETED_INDEPENDENT_RECHECK` is declared rather than self-promoting to production: an independent targeted Astra recheck and explicit owner promotion authorization remain the next external gates.

---

## Candidate & Provenance Identity

- **Remediation Branch:** `rc1-astra-remediation`
- **Branched From (RC1 Source Commit, preserved):** `2bec33fc9032fc90bf655ca7ffbc3e822ba87e12`
- **Independent Astra Audit Reference (preserved):** `5d53b2acf5c5b83d8c44de74f4babee6419cdff0` (`audit/astra-rc1-independent`)
- **Remediation Commits Included in RC2 Binary:**
  - `b25807a` — Gate 1: Process ownership (`pidfd` + `start_ticks`/`boot_id`), managed media ownership/detach, JsBridge authority (`RC-01`, `RC-02`, `RC-03`)
  - `f5bb251` — Gate 2: Provider/budget protection, zero-egress startup, durable script budgets, capability alignment, mandatory `prlimit` & bounded pipes/connections/descendants (`RC-04`, `RC-11`, `RC-15`, `RC-17`, `RC-22`)
  - `678930a` — Gate 3: Lifecycle phases, QA/correction cancellation, post-submit `recovery_required`, partial-error rejection, traversal guard, atomic JSON + corrupt quarantine + ENOSPC worker survival + single-writer `flock` (`RC-10`, `RC-16`, `RC-18`)
  - `09607c6` — Gate 4: Opt-in Forge QA/auto-correct UI controls, expanded counts/spatial/typography negative prompt, no single-VLM self-corroboration, `VideoInspector` eligibility, `CorrectionCoordinator` `compute_inspection_status()`, candidate binding & per-output verdicts (`RC-05`, `RC-07`, `RC-08`, `RC-09`)
  - `5960405` — Gate 5: Reference cardinality/multi-subject/missing starter guards, end-frame UI + record, profile SHA-256 tamper detection, staged ref pinning, `flux2_masked_edit` registration/staging/EXIF, ground-truth receipts, fail-closed Ollama VRAM status, time-bucketed swap-rate anchors, `AVATAR_V2_GPU_*` env precedence, `RC-21` & `RC-23` (`RC-06`, `RC-12`, `RC-13`, `RC-14`, `RC-19`, `RC-20`, `RC-21`, `RC-23`)
  - `5401b27` — Gate 6: Isolated `AVATAR_V2_RC2=1` / `AVATAR_V2_CANDIDATE_DIR` launcher support
- **Candidate Bundle Path:** `dist/AvatarV2.rc2`
- **Executable SHA-256 (`dist/AvatarV2.rc2/AvatarV2`):** `5eb0a9a43a08f10d433dd5829320092af47584ea898aaa675430d74ab70cfd1c`
- **Onedir Tree-Content Digest (`dist/AvatarV2.rc2/`, 40,782 entries):** `a875cb3d1d41c21e8055058d56705ac98964ee67e296d030cf45d9da794b3f08`
- **PYZ Bytecode Correspondence:** `58 / 58` `avatar_v2.*` modules exact normalized bytecode match; `7 / 7` bundled data/workflow files exact SHA-256 match.

---

## Launch & Rollback Instructions (Non-Destructive)

Production (`dist/AvatarV2`) and owner data (`~/.avatar_v2`) were **never modified or promoted**.

### Launch RC2 in Isolated Evaluation Mode

```bash
AVATAR_V2_HOME=/tmp/avatar_v2_rc2_eval AVATAR_V2_RC2=1 /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

Or directly via an explicit candidate directory:

```bash
AVATAR_V2_HOME=/tmp/avatar_v2_rc2_eval AVATAR_V2_CANDIDATE_DIR=/home/jd/avatar_project/dist/AvatarV2.rc2 /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

### Rollback / Continue Using Production Unchanged

Production remains the default target of `scripts/launch_avatar_v2.sh` (`dist/AvatarV2/AvatarV2`, SHA-256 `8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893`) with `~/.avatar_v2/desktop.json` untouched (`5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043`). To remove RC2 cleanly without affecting production or RC1:

```bash
rm -rf /home/jd/avatar_project/dist/AvatarV2.rc2
```

---

## Explicitly Deferred / Out-of-Scope Capabilities (Preserved)

- **Sprint 11 / new product features:** Not started (strict RC1 corrective scope maintained).
- **Autonomous mask detection:** Masked editing (`builtin:flux2_masked_edit`) requires an explicit user- or harness-supplied mask with recorded provenance.
- **Wan 2.2 FunControl / ControlNet / OpenPose / DWPose:** Deferred and excluded from executable script model capabilities.
- **`A1111CompatibilityAdapter`:** Explicitly marked `source_only_optional` (`RC-21`); not exposed as a packaged desktop UI service.
- **Secondary NTFS drive migration (`/dev/nvme1n1`):** Untouched and out of scope.

---

## Related Audit Artifacts

- Full 23-Finding Tracking Matrix: [RC1_REMEDIATION_MATRIX.md](file:///home/jd/avatar_project/docs/audit/RC1_REMEDIATION_MATRIX.md)
- Targeted Acceptance & Live GPU Evidence: [RC2_TARGETED_ACCEPTANCE.md](file:///home/jd/avatar_project/docs/audit/RC2_TARGETED_ACCEPTANCE.md)
- Original Independent Astra Audit (preserved in `/home/jd/avatar_rc1_independent_audit` @ `5d53b2acf5c5b83d8c44de74f4babee6419cdff0`)
