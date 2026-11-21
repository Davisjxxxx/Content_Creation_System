# Avatar Studio Product Design

## Product promise

Avatar Studio turns an unlabeled folder of authorized adult or synthetic reference media into a reusable avatar, then lets the user create an image or video with that avatar using one prompt.

The default experience has four verbs:

1. **Build an avatar** from a set of images.
2. **Refine the avatar** with exact face, body, skin, or anatomy references.
3. **Create** an image or video with a prompt.
4. **Review results** and keep, reject, compare, or reuse them.

Model names, reference-slot wiring, seeds, samplers, and memory controls are Advanced settings. They are never required to complete the normal workflow.

## Design principles

- **One intake, not many upload boxes.** The user selects an image set or folder once. Local vision analysis classifies view geometry, quality, and same-subject consistency.
- **Evidence before inference.** Clearly visible source anatomy is saved as source evidence. Missing anatomy stays missing instead of being presented as known truth.
- **Use every eligible reference.** The build plan rotates through the full evidence-gated pool. A per-render model limit must never silently discard later files.
- **Show the receipt.** Before generation, show selected, analyzed, eligible, held, duplicate, scheduled, and omitted counts in plain language.
- **Progressive disclosure.** Normal mode shows the next useful action. Technical controls and individual mappings live under Advanced.
- **Still first, motion second.** Establish and validate identity/body stills before spending time on animation.
- **Local-first and recoverable.** References, manifests, candidate decisions, settings, and results remain local. Removing a library record does not delete its source file.

## Information architecture

### Create

The everyday screen. The user chooses a saved avatar, types a prompt, chooses **Image**, **GIF**, or **Video**, and selects **Fast**, **Balanced**, or **Highest quality**. The app chooses the engine and memory plan.

Advanced exposes H3 Ref2VA, H3 FL2VA, Wan, custom ComfyUI workflows, runtime profiles, shot controls, reference roles, seeds, and model diagnostics.

### Build Avatar

The default builder is a single vertical flow:

1. Name the avatar.
2. Choose synthetic or consenting adult, sex anatomy, apparent adult age, and anatomy/clothing coverage.
3. Confirm the selected files show one authorized avatar.
4. Select images or a folder.
5. Click **Analyze and build avatar**.

The app then performs these stages and displays them as a progress timeline:

`Inventory → classify → identity gate → sort references → save manifest → queue validation → review candidates`

The user can pause after classification to inspect the AI-sorted galleries. A full build uses all eligible images, including images that already support accepted views, so the current run revalidates the complete set.

### Body Library

A reusable catalog for authorized appearance components. Categories include face shape, skin, body type, height impression, proportions, torso, buttocks, pelvis, hands, and feet.

Each option contains:

- a human-readable label;
- multiple authorized example images;
- the specific region the option may influence;
- provenance, rights, and consent/release status;
- maturity state based on evidence count and diversity;
- a preferred example and optional exclusions.

Options influence only their named morphology. They never replace the avatar's facial identity, age, skin tone, marks, hair, or unrelated anatomy. The primary identity source remains pinned.

### Results

Results open newest-first and support search, type filters, status filters, favorite/strong-candidate state, and grouping by avatar/build. Every result shows the exact source reference used, engine, model, seed, settings, elapsed time, reference receipt, and candidate-QA decision.

Missing or moved files are visibly marked and cannot behave like active Open buttons. Valid files open in the system viewer. A single **Open results folder** action supports arrow-key browsing outside the app.

### Advanced

Contains model/service setup, GPU controls, technical renderer selection, Motion Library, Scene Library, prompt-component editing, and raw per-angle reference mappings.

## Media rules

- **Still image:** identity, body, anatomy, clothing, scene, opening frame, or ending frame evidence.
- **GIF:** motion/choreography example. Extracted stills may become candidates only after explicit review.
- **Video:** motion, timing, camera path, and physics reference. It does not become identity evidence automatically.
- **Audio:** voice, timing, emotion, or sound reference. It never affects appearance.
- **Output image:** best for profile validation, appearance refinement, and scene keyframes.
- **Output GIF:** quick movement preview with lower fidelity expectations.
- **Output video:** final motion work after the avatar still profile is accepted.

The UI explains these rules beside the media choice rather than expecting the user to know engine terminology.

## Reference ingestion and gating

1. Inventory supported media and hash file content so duplicate paths do not inflate evidence.
2. Run local vision classification for framing, facing, camera pitch, pose, visible regions, face visibility, source quality, and crop/UI/watermark warnings.
3. Never ask vision analysis to infer name, identity, sex, ethnicity, or age. Sex anatomy and adult status come from the user.
4. Select the strongest clear face as the identity anchor.
5. Compare other visible faces with the anchor. Hold conflicts and uncertainty for review.
6. Allow body-only images when the user confirms a single-avatar set, while still holding any visible face mismatch.
7. Assign every independently visible usable region, not just one dominant label.
8. Rank compatible references per target slot and save the full ranked gallery.
9. Generate a build plan that gives each supported view at least one candidate and schedules every remaining compatible eligible image once.
10. Preserve direct anatomy evidence without pose-conversion regeneration.

## Rendering architecture

### Juggernaut still lane

Juggernaut XL remains the photorealistic still-image specialist. The legacy scripts are treated as a recipe library, not launched as competing monoliths:

- `continuous_generator_juggernaut.py` supplies the current prompt and realism recipe.
- the dated and backup generators supply quality-profile variants and comparison baselines;
- `juggernaut_hybrid_test.py` supplies the tested hybrid/ADetailer recipe;
- `juggernaut_clean.py` supplies the clean baseline.

Shared settings are normalized into named presets. The production app calls one adapter and records which recipe contributed each setting. Obsolete scripts remain historical evidence and are never all executed for one request.

Juggernaut is used for high-quality still anchors, clean profile candidates, inpainting, anatomy repair with explicit masks, and final still refinement. It does not replace H3's multi-view/motion role.

### H3 lane

H3 FL2VA handles one exact source per profile-turn job. H3 Ref2VA handles explicit multi-reference experiments and selected component conditioning. The app respects H3's picture limit, reports bound versus available files, prioritizes the requested view, and rotates the full pool across validation jobs.

### Local analysis lane

Qwen Vision is a classifier and candidate reviewer, not the generator and not the source of sex or identity. It unloads before H3/Juggernaut generation. The 8B quantized model is the default accuracy route; smaller models are opt-in speed experiments.

### Optional GGUF lane

GGUF is a quantized model format, not a universal quality upgrade. ComfyUI-GGUF can reduce memory for supported transformer/DiT models such as Flux and quantized text encoders. It should be installed only with a tested workflow and model. It does not make the existing 20 GB H3 weights or an SDXL checkpoint automatically fit, and it must not displace a known-good H3/Juggernaut route without an A/B test.

## 8 GB GPU scheduler

The RTX 4070 Laptop GPU is treated as one exclusive generation device. The machine's system RAM is used for model offload, analysis cache, thumbnails, manifests, and queue state.

- Only one GPU-heavy phase runs at a time.
- Vision analysis unloads before generation.
- High-quality work uses a two-pass flow: conservative base render, then targeted upscale/detail work.
- ComfyUI receives explicit unload/free-memory requests between incompatible model families.
- Manual Free GPU works even when automatic cleanup is disabled and reports before/after memory plus released memory.
- Automatic cleanup is threshold-driven and does not unload between every job when a compatible model can be reused.
- Model family switches are visible in the queue so a user understands reload delays.
- The scheduler refuses combinations that are known not to fit instead of repeatedly attempting silent OOM recovery.

Current local facts at the time of this design: RTX 4070 Laptop GPU with 8188 MiB VRAM, 62 GiB system RAM, H3 FL2VA/Ref2VA INT8 weights, H3 FP8 Ref2VA, a quantized H3 Qwen text encoder, Wan 2.2 5B, and Juggernaut XL v9 in the legacy A1111 model store.

## Quality strategy

There is no trustworthy single numeric realism score. Quality is evaluated by reproducible comparisons:

- identity similarity to the exact source;
- body-proportion similarity;
- requested-view correctness;
- landmark and anatomy plausibility;
- artifact rate for hands, feet, face, joints, and occlusions;
- temporal consistency for motion;
- source receipt completeness;
- user accept/reject rate.

Every model or preset experiment reuses the same source, prompt, seed where supported, canvas, and output count. The app retains both winners and failures. Yesterday's strong runs remain pinned as baselines rather than being lost in a global History list.

## Adult-content boundary

The app supports synthetic adults and consenting adults whose images the user is authorized to use. Adult status, consent/authorization, and sex anatomy are explicit profile facts and are never inferred by Qwen.

The normal UI asks for this once per avatar and saves it, instead of making the user repeat the declaration for every render. Adult generation remains blocked for unknown status, missing authorization, or apparent age below 18. There is no upper apparent-age limit.

## Product benchmark takeaways

Current creator products converge on plain-language prompting, fast style choices, pose/composition controls, remixing, persistent characters, and one-click image-to-video. The useful lesson is the interaction model, not their branding or proprietary implementation.

Avatar Studio should go further in the area that matters for this local tool: source receipts, multi-image evidence, exact per-region galleries, local privacy, repeatable quality comparisons, and honest missing-evidence states.

## Delivery sequence

1. Simplify labels, navigation, typography, and the one-button Quick Builder while keeping Advanced controls available.
2. Add a first-class Body Library shortcut and visual component previews.
3. Add the Juggernaut adapter and named still-quality recipes.
4. Add automatic mode/model routing for Image, GIF, and Video.
5. Add grouped Results, favorites, comparisons, and source-receipt inspection.
6. Add masked regional refinement and A/B quality evaluation.
7. Evaluate a GGUF/Flux lane only after installing a supported node/model pair and measuring it against Juggernaut and H3 on the same references.

## Acceptance criteria

- A first-time user can build an avatar from an unlabeled folder without opening Advanced.
- A first-time user can create an image or video from a saved avatar with one prompt.
- Every selected file has a visible disposition: used, direct evidence, held, duplicate, unsupported, or missing.
- No reference is silently omitted due to a model input cap or same-basename staging collision.
- Accepted views can be revalidated in a complete run and are not skipped unless the user explicitly chooses missing-only.
- Anatomy candidates cannot overwrite direct source evidence automatically.
- Results open reliably, and moved/deleted outputs are visibly marked.
- The normal UI has readable 16 px controls, visible keyboard focus, and high-contrast dropdowns.
- A model change is called better only after a saved A/B comparison, not from one attractive output.
