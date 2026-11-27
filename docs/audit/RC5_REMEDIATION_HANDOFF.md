# Avatar V2 RC5 implementation handoff

**RC5_READY_FOR_INDEPENDENT_RECHECK**

The isolated candidate is ready for Astra’s independent recheck. This is the remediation team’s result. Controlled owner acceptance and production activation remain pending.

## Immutable target

| Item | Verified value |
|---|---|
| Repository and branch | `Davisjxxxx/Content_Creation_System` · `rc5-architectural-remediation` |
| RC4 packaged base | `4dee3e57b5071cf3a61317688f8ff086dc903ff3` |
| RC4 independent audit | `de6df8a590de5cb432f9087b8870c770cec3912d` |
| RC5 build source | `7fcfa4a8c9ec93b1c2b92a9753354a258af3bce0` |
| Local candidate | `/home/jd/avatar_rc5/dist/AvatarV2.rc5/AvatarV2` |
| Executable SHA-256 | `d9f22e7e171f3fb91dd2dec5113de0c9d7fb2a7b19623d8c1d4a19f189f9c420` |
| Full bundle SHA-256 | `1017a8b36e954adfe18f275f5684017c5ebc028438ba6d3ac382600831a2f031` · 42,710 entries |
| Source correspondence | 61 code objects, including the entry point, matched fresh compilations of the Git blobs. Seven bundled assets matched byte for byte. |

Built with the existing Python 3.14.4 environment and PyInstaller 6.22.3 on x86-64 Linux. The build used the checked-in spec, fresh cache and temporary work/dist directories. The new candidate lives in the separate remediation worktree. The unchanged frontend HTML was bundled directly. Hash verification ran before and after packaged tests. The executable and bundle digest use the same serialization as RC4. [Integrity receipt](rc5_evidence/bundle_identity.json)

## What the tests established

| Gate | Observation on the final revision |
|---|---|
| Ownership | With incomplete Ollama telemetry the unchanged voiceover case made zero submissions. With healthy telemetry the exact desktop wrote a valid eight-frame sibling and left the occupied original intact. Direct and standalone construction denied global operations. A request lacking established-peer evidence produced zero HTTP operations, and the authorized control produced one. |
| Queue and correction | In a two-boot exact-executable run, a refused item’s provisional file survived a failed rollback. A different job was accepted once, then its real inspection checkpoint directory became unwritable. Both launches showed error/recovery_required for the accepted ID. Together they made one renderer submission, zero for the refused item, and preserved the media. Duplicate IDs produced one completion. Recovery-state cancellation made zero interrupt calls in the archived regression. Candidate-checkpoint failure stopped reinspection. |
| Media and inspection | The binary completed a real PNG and rejected a one-byte PNG. The Wan graph’s requested, submitted and observed values remained distinct. WebKit showed “Not inspected” for both changed bytes and missing requirement binding. A source regression also replaced an image during inspection. |
| Resource telemetry | Incomplete residency refused dispatch, including the corrective path with VLM off. Fresh measured device headroom was required for renderer dispatch. Removing the synthetic device readings, even with healthy empty Ollama residency, yielded zero submissions. An empty GPU sample never verified release. One missing descendant RSS sample invalidated the aggregate. The archived manager ran the bundled SDK through real Bubblewrap with host Python, then stopped a second run when required telemetry disappeared. |
| References and historical controls | A staged ending survived movement of the source. Missing staged bytes and a legacy unbound ending each refused continuation. Failed unpin was observable and recovery retained active pins. All thirteen RC4 entries and twenty-three original IDs are reconciled in the matrix. |

## Evidence grades and limits

- **Source gates.** 499 tests passed in disposable network, PID and filesystem namespaces. The five cases requiring their own Bubblewrap namespace passed separately. All 504 distinct cases passed. Existing success fixtures use valid small media. The original RC4 fixture bodies retain their hashes. [Source tests](rc5_evidence/source_tests.json) · [RC4 baseline](rc5_evidence/rc4_baseline.json)
- **Separate-process verification.** Private renderer doubles and fresh homes supplied deterministic media and simulated storage faults. The three healthy-residency controls change only the Ollama status fixture, with the same synthetic GPU readings in both comparisons, so that FFmpeg, terminal decoding and duplicate-ID handling actually execute. The original combined unpin fixture now stops at its injected I/O error. Its remaining checks were run separately. [Source observations](rc5_evidence/source_observations.json)
- **Exact executable.** The release launcher ran outside the repository with private state, a read-only bundle, no GPU device and no external network. Durable pending jobs drove actual renderer requests, FFmpeg, decoding and history. Four views and seven recipes opened with zero reported JavaScript errors. [Packaged observations](rc5_evidence/packaged_observations.json)
- **Archived code.** 34 applicable targeted tests passed using modules extracted from the executable. The CLI render test passed from source. The desktop spec omits that standalone CLI. The fault-injection and instrumented WebKit results use an external controller with archived code and the shipped HTML. The outer GUI run reported the engine absent and nested Bubblewrap unavailable, as expected for that fixture. A broader archive-only test import also encountered source-only SDK and optional A1111 modules. The separate host-side SDK smoke used the bundled SDK file and `/usr/bin/python3`.
- **Script startup limitation.** An additional archived-manager run with a long private `TMPDIR` raised `OSError: AF_UNIX path too long` at broker bind, before starting the script. The RC4 base uses the same socket path construction. The identical control under a short private temporary directory executed the SDK and then stopped the telemetry-loss case. A short temporary-directory path is required for this recheck.
- **Build diagnostics.** PyInstaller reported missing optional GTK imports `gi._gi_cairo` and `gi._option`, and could not collect a setuptools-vendored packaging licenses module. GTK Broadway exercised the shipped UI with software rendering. No claim is made about hardware-accelerated rendering, native-library supplier provenance, fresh model quality, long GPU sessions, physical power loss or a hostile process with the owner’s OS privileges.
- **Preservation.** The production and RC1–RC4 executable hashes and owner settings matched preflight. The preserved RC4 tree reproduced its published digest. The release checkout retained its prior branch, HEAD and two pre-existing untracked directories. The review used synthetic media, private loopback services and local compute. [Preservation receipt](rc5_evidence/preservation.json)

## Recheck inputs

The [entry-point inventory](RC5_ENTRY_POINTS.md) and [remediation matrix](RC5_REMEDIATION_MATRIX.md) identify the shared boundaries and additional callers. Preliminary validation exposed an ambiguous-ID cancellation and an assumed 6,500 MiB GPU reading. Both have negative regression cases on the final build. Only the build identified above is submitted.

From the remediation worktree, the existing test interpreter can verify the archive against Git. The second command replays the physical storage fault in two launches. It mounts a disposable HOME and leaves the fixture receipts in the printed temporary directory.

```bash
/home/jd/avatar_project/.venv314/bin/python scripts/verify_rc5_bundle.py dist/AvatarV2.rc5 7fcfa4a8c9ec93b1c2b92a9753354a258af3bce0
bash docs/audit/rc5_evidence/replay_queue.sh dist/AvatarV2.rc5 /home/jd/avatar_project/.venv314
```
