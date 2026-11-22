# BRAND_INSTRUCTOR_PHOTOREAL_V1

The reusable fictional presenter profile. It is defined in code in
`avatar_v2/brand_instructor.py` and installed into the Avatar Vault with
**Presenter → Load BRAND Instructor**, or the same button on the Runway panel. Installing creates
the profile and its voice profile if they are missing. It never generates, uploads, or overwrites
approved references.

> Prompt instructions alone cannot guarantee anatomical or temporal realism. Every generated
> reference is a candidate that needs manual approval. Output is production-ready only after the
> realism QA suite and a named human reviewer have passed a 20-30 s instructional segment.

## Identity

| Field | Value |
|---|---|
| Vault id | `brand_instructor_photoreal_v1`. Vault ids are lowercase-only; the display name is `BRAND_INSTRUCTOR_PHOTOREAL_V1`. |
| Subject | Fictional, synthetic, female, apparent age 26. Not based on any real person. |
| Content | General only. `consent_confirmed` is false and coverage is `clothed`, so adult content classes stay locked for this profile, locally and in the cloud. |
| Persistent features | Natural brunette hair with realistic individual strands; expressive brown eyes; subtle natural facial asymmetry; authentic skin texture with visible pores; subtle natural makeup; natural lip shape. |

**Full identity prompt** (for models with long prompt limits):

> A fictional 26-year-old woman who looks attractive, intelligent, approachable, natural and
> professional. Natural brunette hair with realistic individual strands and a few soft flyaways;
> expressive brown eyes; believable facial proportions with subtle, realistic asymmetry; authentic
> unretouched skin texture with visible pores and fine natural variation; subtle natural makeup;
> natural lip shape. Professional smart-casual clothing. Chest-up framing, soft natural studio
> lighting, neutral background, direct eye contact with the camera, calm and confident
> presentation.

**Compact identity prompt**: the Create-screen default, sized so the `@presenter` binding, a
slot direction and the realism guards fit inside Gen-4 Image's 1,000-character limit:

> Fictional 26-year-old woman: attractive, intelligent, approachable, natural and professional.
> Natural brunette hair with realistic strands, expressive brown eyes, believable proportions with
> subtle natural asymmetry, authentic skin texture with visible pores, subtle makeup, natural lips,
> smart-casual professional clothing. Soft natural studio light, neutral background, direct eye
> contact, calm and confident.

**Realism guards.** Runway's image and video models have no negative-prompt field, so the
exclusions are restated positively and take priority over camera and feature text when a prompt is
trimmed:

> Skin keeps its real texture with a natural matte sheen rather than an airbrushed finish; the face
> keeps subtle natural asymmetry; features stay true to life in size and proportion; each eye shows
> one soft, natural catchlight from the key light.

**Exclusions.** Used verbatim as the negative prompt where a provider supports one (local
SDXL/Juggernaut and Wan through ComfyUI):

`excessive skin smoothing, airbrushed skin, artificial perfect symmetry, glossy plastic skin, waxy skin,
exaggerated features, oversized eyes, overfilled lips, artificial eye reflections, duplicated catchlights,
glassy doll eyes, CGI look, beauty filter, over-sharpening halos, extra fingers, fused fingers, text, logo,
watermark`

## Reference slots

All slots are reviewable. Generated or imported images arrive as **pending** candidates in
**Presenter → Review & approval queue**. Accepting one records its SHA-256 in the profile's
`reference_hashes`. Accepting the straight-on portrait sets the **master identity**: cloud jobs for
this profile then refuse a primary identity whose hash differs, so the character never changes
silently when switching providers.

| Slot id | Label | Direction |
|---|---|---|
| `identity_front` | Straight-on portrait (master) | Chest-up, square to camera, relaxed neutral expression, direct eye contact |
| `identity_three_quarter_left` | Three-quarter left | Face about 45° toward her left, eyes to camera |
| `identity_three_quarter_right` | Three-quarter right | Face about 45° toward her right, eyes to camera |
| `identity_profile_left` | Left profile | True left profile, same lighting and background |
| `identity_profile_right` | Right profile | True right profile, same lighting and background |
| `presenter_expression_neutral` | Neutral expression | Calm neutral expression, lips relaxed and closed |
| `presenter_expression_smile` | Natural smile | Gentle smile reaching the eyes, teeth only slightly visible |
| `presenter_expression_speaking` | Speaking expression | Mid-sentence, mouth naturally open on a vowel |
| `presenter_upper_body` | Upper-body instructional | Waist-up, open-hand explanatory gesture with natural fingers |

The first five reuse the Avatar Designer's identity slots. The last four are presenter slots that
do not change the H3 designer's coverage report.

## Voice

Voice identity is stored separately from visual identity (`brand_instructor_voice_v1`):

| Setting | Value |
|---|---|
| Model | `eleven_v4` (Runway text-to-speech; supports `[pause]`, speed, and continuity across segments) |
| Voice | **Not preset.** Audition Runway preset voices or design one, then save it. |
| Design prompt | Warm, clear, confident female presenter voice in her mid-twenties with a clear, neutral accent, a calm medium pace, friendly and professional delivery, natural breaths and no vocal fry. |
| Speed / stability / similarity / style | 0.95 / 0.6 / 0.75 / 0.15 |
| Language | `en` |
| Target pace | 140-160 words per minute |

Narration can be imported (with an optional note), generated with a preset voice, or generated with
a voice designed from a text description. Avatar V2 offers no voice cloning. Any narration in the
library can drive a talking head or replace video audio right away.

## Recommended first render workflow

The quickest route is **Studio** (the start screen), which runs the steps below with simple choices.
Paid steps need only a saved Runway key; each button shows its price before you click. Credit figures
are local estimates from Runway's published prices (1 credit = $0.01).

Studio's **Look** picker sets the identity description used for new master options: **Natural**
(default: little makeup, no eyeliner, matte skin, natural brows and lip colour, unhighlighted
brunette hair), **Polished** (soft professional makeup), or **Original** (the text above). These
are `LOOKS` in `avatar_v2/brand_instructor.py`. The first live master (2026-11-22) came out glossier
and more made-up than the brief, which is why Natural is the default.

1. **Load BRAND Instructor** (or use Studio step 1). Then queue **4 master candidates**: Gen-4 Image,
   `1080:1440`, fresh seeds, slot *Straight-on portrait*, no identity reference.
   About 32 credits.
2. Open each candidate in the full-size viewer (**Make → Her look**) and pick exactly one as the master.
3. Re-load the preset, which now carries the master, and queue **Consistent variations** for the other
   eight slots, about 8 credits each. Review and accept or reject each one.
4. Under **Talking presenter**, create a Runway avatar from the master still with your chosen
   voice.
5. Prepare narration: import a recording, or queue a **Narration** job (about 5 credits per
   1,000 characters with Eleven v4).
6. Render a **20-30 s talking-head segment** from that narration
   (about 1 credit per 3 s of video).
7. Run **Realism QA** on that segment and complete the human review. Its READY / NOT READY result
   is a guide for your own decision; it does not block anything.
8. Optional: **Performance transfer** (Act-Two) from your recorded delivery. Your footage's audio is
   removed before upload, and your voiceover is muxed locally.
