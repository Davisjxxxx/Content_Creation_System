# Avatar V2 RC4 — Release Readiness & Independent Recheck Handoff

## Disposition

**`RC4_READY_FOR_INDEPENDENT_RECHECK`**

> **Strict Gate Reminder:** RC4 is **NOT** approved for promotion, merge, production deployment, or owner testing against original media until an independent Astra recheck validates the packaged `dist/AvatarV2.rc4` candidate and issues a release clearance.

---

## 1. Candidate Identity & Provenance

| Field | Value |
|---|---|
| **Remediation Branch** | `rc4-astra-remediation` |
| **Exact Base Commit (RC3 Packaged Build)** | `4472e7522f13b684738327b6dd1c6694c563279c` |
| **P0 Invariant Commit (`F01`, `F02`)** | `21cb0303fcb86b1e946b75925384e29b996de69d` |
| **P1/P2 Invariant Commit (`F03`–`F13`, `AVATAR_V2_RC4=1`)** | `4dee3e57b5071cf3a61317688f8ff086dc903ff3` |
| **Packaged Executable** | `dist/AvatarV2.rc4/AvatarV2` |
| **Executable SHA-256** | `a63423b57203ea547d9fcde49e4bafec6e92030b47d69e0077f50a1169a6ed0b` |
| **Deterministic Bundle Tree Digest** | `8cefc6da46adde6a16074ab70fb9b42747360752f436a8fa4bf838467c5343ac` (`42,708` entries) |
| **Tree Digest Recipe** | Sorted `os.walk` over `dist/AvatarV2.rc4`, updating SHA-256 with `f"{typ}\0{rel}\0{val}\n"` (`L`=symlink target, `D`=`""`, `F`=file SHA-256 hex) |
| **PYZ Bytecode Match** | `58 / 58` archived `avatar_v2.*` modules + `desktop_entry` (`100%` exact match to `4dee3e5`) |
| **Bundled Asset Match** | `7 / 7` exact SHA-256 match |
| **Clean-HOME Pytest Suite** | `490 / 490` passed, `0` failed, `0` skipped |
| **Packaged WebKit2GTK UI Self-Test** | `4/4` views, `7/7` recipes, all checks PASS, `js_errors: []` |

---

## 2. Isolated Launch & Non-Destructive Rollback

### Launch Isolated RC4 Candidate (Zero Production / Owner-Data Side Effects)

```bash
cd /tmp && \
env HOME=/tmp/avatar_v2_rc4_os_home AVATAR_V2_HOME=/tmp/avatar_v2_rc4_data AVATAR_V2_RC4=1 \
  /home/jd/avatar_project/scripts/launch_avatar_v2.sh
```

### Non-Destructive Rollback (Already Preserved Untouched)
- **Production bundle:** `dist/AvatarV2/AvatarV2` (`8e71ad4d409f38c69f813ada968ccca1bf17832e7325b4eb5303e130002d4893`) — launched by default when `AVATAR_V2_RC4` is unset.
- **Prior candidates preserved:** `dist/AvatarV2.next` (RC1), `dist/AvatarV2.rc2` (RC2), `dist/AvatarV2.rc3` (RC3).
- **Owner config preserved:** `/home/jd/.avatar_v2/desktop.json` (`5dd11ff5d7dffc72a8542c9a67461aaf20501016c7cad1e13c4b256af7fbf043`).
- To discard RC4 cleanly without touching production: remove `dist/AvatarV2.rc4` and `/tmp/avatar_v2_rc4_*`.

---

## 3. Scope Freeze & Explicitly Deferred Items

- **Preserved intact:** Approved frontend (Automatic, Guided, Script modes; My Stuff & Browse Local Files; optional multi-angle subject refs; Character Profiles; generation history & media viewer; collapsed inspection/correction details; progress & cancellation controls), working FLUX.2 9B & Wan 2.2 14B workflows, installed models, `GPUMemoryManager` with env precedence (`AVATAR_V2_GPU_AUTOCLEAN`, `AVATAR_V2_GPU_THRESHOLD`, `AVATAR_V2_GPU_CHECK_INTERVAL`), and all prior audit branches/commits.
- **Explicitly out of scope / deferred:**
  - Secondary NTFS drive (`/dev/nvme1n1`) mounting, formatting, or storage migration (untouched).
  - Additional generation models, specialist vision frameworks, or Sprint 11 features.
  - Kernel-level container/VM isolation for Script mode (documented explicitly under the single-OS-user trust boundary in `docs/audit/RC4_INVARIANT_DESIGN.md`).
  - Live GPU end-to-end render re-benchmarking against ComfyUI/Ollama (left for independent verification with isolated test media; unit/adversarial tests verified all GPU handoff, VRAM telemetry, and cleanup boundaries without disturbing owner models or media).

---

## 4. Audit Companion Documents

- [RC4_INVARIANT_DESIGN.md](RC4_INVARIANT_DESIGN.md)
- [RC4_23_FINDING_MATRIX.md](RC4_23_FINDING_MATRIX.md)
- [RC4_ADVERSARIAL_ACCEPTANCE.md](RC4_ADVERSARIAL_ACCEPTANCE.md)
