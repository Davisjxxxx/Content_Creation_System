# RC5 remediation and prebuild gates

The baseline is packaged source `4dee3e57b5071cf3a61317688f8ff086dc903ff3`. RC4 independent commit `de6df8a` supplies the findings and original fixture observations. The separate audit worktrees retain RC1, RC2 and RC3. This is implementation-team validation. Independent acceptance remains pending.

## RC4 findings

| Finding | Control and additional path | Source and separate-process result |
|---|---|---|
| F01 renderer ownership | `ComfyUIProvider._post_global` denies missing authorization, uses a literal loopback endpoint with proxies and redirects disabled, and rechecks before request headers. The desktop authorizer verifies the established peer socket belongs to its recorded PID. | Direct and standalone providers denied both global requests. Revoking peer evidence produced zero HTTP requests. The authorized connected-peer control produced one request. Desktop shutdown, restart and automatic cleanup are covered by ownership regressions. |
| F02 media ownership | `safe_output.publish_output` gives the producer a private directory on the target filesystem, flushes the completed file, then publishes by no-replace link. FFmpeg receives a fresh name and no-overwrite mode. Archive, staging, downloads and provenance use exclusive publication. | RC4 guard intercepted an overwrite attempt. On RC5 the exact fixture stops at its incomplete telemetry. With complete telemetry the desktop adds audio, preserves the occupied original and probes the new eight-frame video. Late collision and partial producer tests preserved prior contents. |
| F03 deferred references | Continuation requires the staged ending and digest. Provider staging checks the bound reference. | The staged ending survived a source move. Missing staging and legacy pathname-only continuation admitted zero new jobs. |
| F04 compound criteria | Retained deterministic interpretation and workflow contract. | Two counts and both object states survived the source fixture. Visual compliance is a separate question. |
| F05 inspection binding | `bound_inspection_status` checks the candidate bytes, complete contract, live reference fingerprints, mandatory assessments and eligibility. `present_job` derives history and queue labels, including nested summaries. | WebKit displayed “Not inspected” for replaced output and missing contract. Positive complete evidence and the uncertain/alternate controls are covered by focused regressions. |
| F06 durable admission and acknowledgements | Provisional item checkpoints require a matching committed SQLite admission row. One transaction admits a batch. The worker and offline producer use the same directory lock. Per-endpoint claim records associate an acknowledgement with one job and attempt. | Refused admission with surviving provisional files restored zero executable jobs. A second ID collision yielded one completion and one recovery receipt. The broker only refunds definitive rejection. |
| F07 terminal history | Terminal history wins over stale active state. Queued retry uses a job checkpoint. Nonterminal history writes are refused. | Stale-checkpoint and concurrent-writer regressions passed. An offline writer was refused while the reopened worker held the directory. |
| F08 correction durability | Base and corrective dispatch have separate durable markers and acknowledgements. Recovery takes precedence in terminal finalization. A failed checkpoint stops subsequent corrective work. | Initial inspection and candidate checkpoint faults both returned error/recovery_required. The acknowledged correction remained identifiable beside the base ID. Cancellation preserved the original and issued a scoped interrupt. |
| F09 measured receipts | Retained separate request, resolved graph and observed media. Media receipts now require successful decoding. | The Wan scene requested 720×1280/80, submitted 480×640/81 and observed 160×96/8. The two sampler nodes agreed with the recorded seed. |
| F10 resource handoff | Normal and corrective ComfyUI dispatch check Ollama residency. Incomplete model entries refuse admission. Measured cleanup requires matching complete device samples, a drop and usage below threshold. | Empty/malformed responses, a missing VRAM field and timeout refused. Valid empty residency admitted. A cleanup acknowledgement without samples recorded no verified release. No live GPU load test. |
| F11 valid media | `validate_media` verifies image format, decodes all image frames, counts decodable video frames, checks stream, dimensions and duration, and fingerprints the observed file. The worker validates exported and corrected results before completion. | A valid PNG completed. A one-byte PNG failed with zero observed media. Wrong-format, truncated and missing output controls failed. A valid MP4 produced measured dimensions and frame count. |
| F12 sandbox telemetry | The descendant census requires usable status information for every observed member. The resource guard closes admission on missing census evidence. | One unreadable member in a two-member inventory returned incomplete, zero admissible RSS. Host-side Bubblewrap regressions exercised actual containment under disposable settings. |
| F13 reference pins | Persist failed pin release visibly and restore the in-memory pin map on failure. Reconcile inactive job pins when constructing the restored manager. | Injected unpin failure raised. Reopening and reconciling removed the finished job and retained the live job’s pin. Original source remained present. |

## Original RC1 findings

The original RC1 numbering governs this table. “Retained” means the scoped RC4 closure is carried forward with source regression coverage. It does not assert a new live hardware or paid-service test.

| Original | RC5 disposition and evidence scope |
|---|---|
| RC-01 renderer/process identity | Remediated under F01. Existing PID/boot/listener and pidfd controls retained. |
| RC-02 original media | Remediated under F02, including occupied voiceover, archive, linked names and provenance sidecar. |
| RC-03 external bridge paths | F03 and picker regressions. Legacy continuation requires reselection. |
| RC-04 unintended paid fallback | Retained. Explicit-engine and mocked provider cases, with zero paid calls. |
| RC-05 shipped QA controls | Retained. Opt-in payload and existing frontend regressions. |
| RC-06 locked references | F03. Unsupported locked routes refuse. Start/end graph control retained. |
| RC-07 mandatory criteria | F04. Tested compound counts and states. |
| RC-08 honest acceptance status | F05. Missing evidence and ineligible mandatory passes cannot certify a candidate. |
| RC-09 exact candidate evidence | F05/F08. Candidate-specific results and reinspection faults. |
| RC-10 cancellation and replay | F06–F08. Admission, post-submit recovery, scoped cancellation. |
| RC-11 host budgets | F06/F08. Ledger write failure, concurrent cap and definitive-refund regressions. |
| RC-12 correction dispatch | Retained 5B and explicit-mask workflow regressions. No autonomous masks. |
| RC-13 measured receipts | F09. Actual graph seed and independently different fixture output metadata. |
| RC-14 GPU handoff | F10. Synthetic residency and release evidence. |
| RC-15 broker capability truth | Retained installed-component and empty-inventory regressions. |
| RC-16 terminal output | F11. Terminal status alone cannot certify a deliverable. |
| RC-17 sandbox resource bounds | F12. Partial RSS and live containment tests. |
| RC-18 writer and state consistency | F06–F08. Provisional records, writer exclusion, retry and terminal history. |
| RC-19 durable reference ownership | F03/F13. Staged digest, restart and pin release controls. |
| RC-20 swap and environment | Retained polling-interval/environment regressions. No induced paging or thermal run. |
| RC-21 optional A1111 adapter | Prior source-only scope retained. No additional adapter capability claimed. |
| RC-22 offline startup | WebKit with configured-key double made zero account calls at startup and when pending jobs fell to zero. |
| RC-23 benchmark evidence | Retained exact-mask regression and published tree serialization. No new visual score or benchmark claim. |

## Additional callers found during implementation

The source CLI now submits through DesktopAPI’s durable RenderQueue. A rejected CLI admission dispatches zero requests. CLI plans, resolved workflows, generated QA reports, reviewed QA reports, and optional mask exports use no-replace publication. Normal and corrective ComfyUI dispatch now use the same Ollama residency check, including corrections with VLM disabled. Inspection takes the output and contract fingerprints before examining the file and checks them again when deriving a verdict. An injected image replacement during inspection invalidates the result.

The state directory contains application-owned configuration, locks, ledgers, job records and derived indexes that legitimately change. The renderer and FFmpeg see private or unique media destinations. Private script snapshots stay under an exclusively created temporary root. Packaged media subprocesses, the CLI and the offline visual evaluation fixture writer were traced. Administrative build, install and promotion utilities were inventoried separately and were not invoked as runtime controls.

A further F06 cancellation test on preliminary source `957db52` recorded one scoped interrupt call when the job was already in recovery with an ambiguous identifier. The final revision returns refusal before dispatch, stores an unconfirmed acknowledgement as diagnostic information, and exposes a terminal queue state only after its history persistence attempt. The regression requires zero interrupt calls. The preliminary binary is excluded from the handoff.
