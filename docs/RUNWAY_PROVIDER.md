# Runway cloud provider

Avatar V2 sends the jobs you start to Runway's developer API. Runway is ready as soon as an API
key is saved; there is no enable switch, upload-consent box or confirmation pop-up. Every job
shows its price before you click (Pro tools → Create also lists what will be uploaded), it starts
on one click, and its real cost is recorded afterwards. It never runs as a fallback for local
rendering, and the app never resubmits a paid job on its own.

- Code: `avatar_v2/providers/runway.py` (adapter), `avatar_v2/desktop_studio.py` (Studio), `avatar_v2/presenter.py` (prompt compiler),
  `avatar_v2/secret_store.py` (API key storage), `avatar_v2/cloud_ledger.py` (cost ledger), the
  `runway_*` / `queue_runway` methods in `avatar_v2/desktop.py`, and the presenter APIs in
  `avatar_v2/desktop_presenter.py`. The BRAND profile is documented in
  [`BRAND_INSTRUCTOR.md`](BRAND_INSTRUCTOR.md).
- API reference used: <https://docs.dev.runwayml.com/api>, with `X-Runway-Version: 2024-11-06`.
  Request schemas, prices, input limits and failure codes were verified on **2026-11-21**.
- Tests: `tests/test_runway_provider.py`, `tests/test_runway_desktop.py` and
  `tests/test_presenter_pipeline.py`. They use mocked HTTP and locally generated media; nothing in the
  test suite contacts Runway or spends credits.

## Setup

1. Create an API key in the Runway developer portal and buy credits there
   (1 credit = $0.01).
2. Save the key: paste it into **Studio → Connect Runway** (or Pro tools → Settings → Runway), or
   run `scripts/set_runway_key.sh`, which reads it with hidden input. It is stored in the OS keyring
   when the optional `keyring` package is installed, otherwise in `~/.avatar_v2/secrets/runway_api_key`
   (directory `0700`, file `0600`). Alternatively, set `RUNWAYML_API_SECRET`, which takes precedence.
   The key is never written to `desktop.json`, never sent to the UI, and never logged.
3. That's it. Studio shows your credit balance (`GET /v1/organization`, free) and every step is ready.
   Pro tools → Create → **Runway** renderer card exposes every model and option.

## What stops a job, and what doesn't

Changed on 2026-11-22 at the owner's request: Runway jobs have no built-in stops besides a saved key.

| Control | Behavior |
|---|---|
| API key | The only requirement. Without one, jobs explain how to add it. |
| Price | Shown on Studio buttons and in Pro tools → Create → **Estimate** before you start a job. No confirmation pop-up, no "estimate must match" check. |
| Spending limits | Optional and **off by default** (0 = no limit): `runway_job_limit`, `runway_batch_limit`, `runway_daily_limit`, `runway_monthly_limit`, read only from saved settings. When you set a per-job limit and Runway's own `estimatedCost` comes back above it, the task is cancelled immediately. The old caps (`runway_max_credits_per_*`, `runway_*_credit_cap`) are ignored. |
| Unconfirmed charges | Jobs whose charge Runway never confirmed are listed in Presenter tools for your records and counted in spend, but never block new jobs. They are never resubmitted automatically. |
| Identity check | When a vault profile has a chosen master photo (`reference_hashes["front"]`), a different primary identity or Act-Two character adds a warning ("her face may come out different") instead of refusing the job. The job spec and receipt record the identity hashes. |
| Narration | Any voiceover in the library (generated or imported) can drive a talking head or replace video audio; there is no approval step. Imports take an optional note. |
| Performer audio | Act-Two driving footage is re-muxed without its audio track before upload, so the performer's voice never reaches Runway. The output can be muted or given a voiceover locally (`output_audio`). |
| Content rule (kept) | Cloud jobs accept the `general` content class only (Runway's usage policy); adult classes stay on local renderers. The subject must be declared `synthetic`, or a consenting adult with confirmed consent, so a real person's likeness is never uploaded without consent. The BRAND presenter is synthetic, so this never gets in her way. |
| What can be uploaded | Identity references, one wardrobe and one scene reference, first/last frames, the performance video (Act-Two), narration audio, or the still being upscaled. Body, anatomy and guide references are never sent. |
| Metadata | Images carrying EXIF are re-encoded without it before upload (ICC color profile kept). |
| Key exposure | The key is sent only to `https://api.dev.runwayml.com`. Presigned upload URLs and output download URLs are contacted by a separate client that carries no credentials. |
| Receipts | Each queue item keeps a cloud receipt: model, endpoint, task id, local and Runway estimates, credits charged, upload list (file name, size, hash prefix, transfer method), failure code. Inline image data is redacted. |
| Outputs | Downloaded right after success (Runway URLs expire in 24-48 h) to `~/.avatar_v2/runs/runway/<task-id>/`, then archived like local renders. |

Local GPU cleanup does not apply to cloud jobs. The queue records a `skipped` memory event for
them instead of calling ComfyUI `/free`. Local ComfyUI jobs keep the mandatory forced
pre-render cleanup for `int8_12gb`, `low_memory` and `low_memory_quality`.

## Supported tasks and models

| Task (UI) | Endpoint | Models | Price (credits) |
|---|---|---|---|
| Presenter still | `POST /v1/text_to_image` | `gen4_image` (0-3 tagged refs), `gen4_image_turbo` (1-3 refs), `gemini_image3.1_flash` (up to 14 refs, 5 marked human; 1 or 4 outputs) | 5 (720p) / 8 (1080p); 2; 5 / 7 / 11 / 16 per image by size |
| Consistent variations | `POST /v1/text_to_image` × N | as above | N × still price |
| Animate approved still | `POST /v1/image_to_video` | `gen4.5`, `gen4_turbo` (2-10 s), `h3_max` (hosted MiniMax H3, 5-15 s, optional last frame) | 12/s; 5/s; 5/s at 480p, 8/s at 768p |
| Talking presenter (lip-synced) | `POST /v1/avatar_videos` | `gwm1_avatars` with your custom avatar; script (TTS) or narration audio | 1 per 3 s of video + 1 per 50 script characters, minimum 2 |
| Performance transfer | `POST /v1/character_performance` | `act_two` (3-30 s performance video) | 5/s |
| Upscale approved still | `POST /v1/image_upscale` | `magnific_precision_upscaler_v2` | 25 (150 if the output exceeds 4096 px) |
| Presenter avatar setup | `POST/GET/DELETE /v1/avatars` | custom avatar from the primary identity still plus a preset voice | not listed as billable |
| Narration voiceover | `POST /v1/text_to_speech` | `eleven_v4` (2,500 characters, `[pause]`, speed, neighbouring-segment context), `eleven_v3` (5,000), `eleven_multilingual_v2` (1,000) | 5 per 1,000 characters; 1 per 50; 1 per 50 |
| Designed voice | `POST /v1/voices/preview`, `POST /v1/voices` (`from.type = "text"` only) | `eleven_ttv_v3`, `eleven_multilingual_ttv_v2` | not listed |

Voice cloning (`from.type = "audio"`) exists in Runway's API but is deliberately not offered:
narration comes from authorized recordings, preset voices, or voices designed from a description.

## Reliability and accounting

- **Restart reconciliation.** The task id is persisted durably (atomic write plus fsync) the moment
  Runway accepts a task. If the app restarts mid-job, the item is re-queued to **poll the existing
  task**, never to create a duplicate.
- **Uncertain submissions.** The intent to send is recorded durably before the create request leaves.
  A lost response, a 5xx, a success without an id, or a crash after that point is recorded as
  **uncertain**: never called unbilled, never resubmitted automatically, and its estimate is counted as
  unconfirmed spend (it does not block new jobs). Settle it from Presenter tools → Providers & budget
  (Not billed, Billed, or reconcile from a dashboard task id). Only a job stopped before sending is
  reported as unbilled. The full state table is in [`RECOVERY_AND_ROLLBACK.md`](RECOVERY_AND_ROLLBACK.md).
- **Cost ledger.** `~/.avatar_v2/cloud_ledger.jsonl` (append-only, fsynced) records Runway's estimate
  at creation and the credits actually charged at completion. A charge counts as actual only when
  Runway reports it; failed, cancelled, timed-out or cost-less tasks keep their estimate as
  *unconfirmed* until resolved. **Presenter tools → Providers & budget** shows charged, unconfirmed and
  queued spend.
- **Output integrity.** Every finished job (local or cloud) records the SHA-256 of each output.
  **Presenter → Output integrity** re-verifies them. Runway receipts also hash each raw download.
- **You pick.** Slot jobs, seeded master-candidate batches, and manual imports register outputs as
  pending candidates in the vault profile; you keep or toss them (one click, with a full-size viewer).
- **Lessons.** Talking-presenter segments are hashed by their inputs, and only new, changed,
  failed, output-missing or charge-unconfirmed segments are re-queued when you click **Queue changed
  segments**. A stale result never overwrites a newer render.

Runway lists many more models (Veo, Kling, Seedance, Wan 3, Gemini Omni, and others). Only models
whose request schemas were verified are registered. Add others deliberately in
`MODELS` / `estimate_credits` / `build_request`, with tests, rather than passing arbitrary model ids
through.

## Recommended presenter workflow

1. **Design**: with no identity reference set, a *Presenter still* is pure text-to-image. The
   compiler writes the subject from the avatar's declared sex and apparent age
   (for example "a woman in her mid-20s"). Generate candidates and pick one.
2. **Approve**: save the chosen still as the avatar's primary identity in **Build Avatar**, with
   subject type *synthetic*. That is the provenance the cloud gate requires.
3. **Variations**: *Consistent variations* reuses the same tagged `@presenter` reference and
   seed across presets (studio headshot, warm smile, teaching pose, desk host, whiteboard,
   left/right profile). Add a wardrobe reference (`@outfit`) and a scene (`@location`) to lock
   the brand look.
4. **Voice and lip sync**: under *Talking presenter*, create a Runway avatar from the primary
   identity still with a preset voice. Then generate lesson segments from scripts of up to
   5,000 characters each, or from recorded narration.
5. **Motion**: *Animate approved still* for B-roll and gestures (`gen4.5` for fidelity, `h3_max`
   for first-to-last keyframe control). *Performance transfer* maps a recorded delivery onto the
   presenter.

Prompts are compiled within each model's limit, measured in UTF-16 units as Runway counts them.
Low-priority sections are trimmed first, and sections that mention a tagged reference are never
trimmed. The estimate shows the compiled prompt and anything that was trimmed.

## Failure handling

| `failureCode` | Meaning | Retry? |
|---|---|---|
| `SAFETY.INPUT.*`, `SAFETY.OUTPUT.*`, `INPUT_PREPROCESSING.SAFETY.TEXT` | Moderation rejected an input, output or the prompt | No; revise first. Credits are not refunded for `SAFETY.INPUT.*`. |
| `INTERNAL.BAD_OUTPUT.*` | Output discarded for quality (often logos, watermarks or text in inputs, or prompts asking for text) | After fixing the inputs |
| `ASSET.INVALID` | An input's dimensions, duration or format is unusable | No; replace the input |
| `INPUT_PREPROCESSING.INTERNAL`, `INTERNAL`, none | Transient internal failure | Yes, after a delay |
| `THIRD_PARTY.UNAVAILABLE` | Upstream model provider unavailable | Later |

Polling follows Runway's guidance: at least 5 s between polls, with jitter and exponential backoff
on errors. A rate-limited (429) task creation is retried, because no task was created. Server
errors on creation are **not** retried, to avoid paying for a duplicate task. Cancel in the queue
deletes the remote task. A timeout also cancels it.

## Not yet verified live

No real Runway call has been made from this machine: there is no API key configured and no
credits. Before relying on the provider, run one cheap end-to-end job, for example a
`gen4_image_turbo` still at 2 credits with the primary identity attached, and confirm the
download, receipt and cost.

## Publishing note

The presenter is designed to be indistinguishable from a real person. Major platforms require
labels on realistic synthetic media, and the EU AI Act's transparency obligations (Article 50)
apply from 2 August 2026. Plan an "AI-generated presenter" disclosure for published course
material.
