# OmniVoice for warm gratitude narration

Research checked 9-10 September 2026 against upstream `k2-fsa/OmniVoice` master and the published paper. The running Colab may contain an earlier version. Conclusions about performance below are proposals to audition, not claims that this particular voice has passed a listening test.

## Verified capabilities and limits

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Cloning follows the reference's delivery. Matching `instruct` attributes can reinforce it; conflicting instructions usually lose to the reference. | [Official tips](https://github.com/k2-fsa/OmniVoice/blob/master/docs/tips.md) | The reference must already sound warm, intimate, and clearly articulated. Text instructions cannot reliably transform a flat recording. |
| Instructions use a fixed attribute vocabulary. Categories cover gender, age, pitch, accent, dialect, and the style `whisper`. There is no documented gratitude, storyteller, ASMR, emphasis, or emotion-strength instruction. | [Voice-design guide](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md) | Do not send free-form acting directions. Leave instructions empty for natural intimate narration; use `whisper` only for a deliberate whisper audition. |
| The demo recommends a 3–10-second reference and supports an optional exact transcript. Its current cloning endpoint accepts `instruct`, speed, steps, guidance, denoise, and processing flags. | [Official demo](https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/cli/demo.py) | Reuse the same reference and transcript for consistency. Verify the deployed endpoint schema before uploading anything. |
| English pronunciation can be overridden with bracketed CMU phonemes, including lexical stress digits. Supported nonverbal tags include `[sigh]` and `[laughter]`. Numeric normalization is recommended. | [Official README](https://github.com/k2-fsa/OmniVoice/blob/master/README.md#non-verbal--pronunciation-control) | Maintain explicit pronunciation replacements for difficult names or homographs. Lexical stress is not a command for dramatic sentence emphasis. Avoid adding stock sighs throughout gratitude stories. |
| Defaults are 32 decoding steps and guidance 2.0. Speed below 1 lengthens speech; fixed duration overrides speed. Automatic long-text chunking targets 15 seconds after a 30-second threshold. | [Generation parameters](https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md) | Use modest speed variation between complete thought units, preserving enough sentence context. Do not force every sentence into a fixed duration. |

The current model source includes instruction tokens alongside reference audio: the claim that cloning always ignores `instruct` is false for this version. It estimates target duration from reference text and audio, making transcript alignment consequential. Output postprocessing reduces long internal silences to about 500 ms; disabling that setting preserves them. Separate edge padding/fades remain active. The demo currently creates the clone prompt without forwarding its preprocessing checkbox, so that checkbox cannot be assumed effective without fixing the Colab wrapper. [Model source](https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/models/omnivoice.py), [demo source](https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/cli/demo.py).

## What the paper establishes

OmniVoice predicts acoustic tokens through iterative unmasking. Its evaluated decoding baseline uses 32 steps and guidance 2.0, with random mask-position selection and deterministic token-class selection. These are generation settings, not emotional expressivity knobs. In Appendix B, increasing to 64 steps makes small, mixed changes: Seed-TTS English WER is 1.60 at 64 steps versus 1.53 at 32. The prompt-denoising ablation improves predicted naturalness while reducing speaker similarity slightly. These benchmarks establish intelligibility and similarity; they do not demonstrate gratitude-story emotion or ASMR quality. [OmniVoice paper, sections 3.4, 4.3 and Appendix B](https://arxiv.org/html/2604.00688v1).

## Implementation strategy and audition criteria

These are engineering/editorial recommendations inferred from the capabilities above:

1. Start with `b1.1.wav`. Its previously measured 7.67-second duration fits the recommendation, but duration alone does not establish its expressiveness. Prefer one speaker, clean consonants, a gentle change of pitch, and a complete thought. Provide the exact words spoken; never invent a transcript.
2. Keep source extraction and quote handling separate from the spoken script. Preserve meaning, names, and quotations. Expose pronunciation edits and performance planning for review before inference.
3. Organize narration into sentence or paragraph thought units. Use a calm baseline with slightly more forward movement in story development and more space after a realization or quotation. Avoid mechanically slowing every sentence or inserting an ellipsis after every phrase.
4. Use native speed control, editorial pauses, and reference delivery. Keep intentional breaths and pauses with output postprocessing disabled. Do not manufacture expression with pitch shifting, spectral matching, voice restoration, or aggressive loudness flattening.
5. Audition a short passage containing a name, narrative action, a quotation, and a reflective ending. Compare 32 and 48 steps plus guidance 2 versus 3, baseline and modestly slower speed, then choose by listening at matched volume. Test denoising off only when the reference is clean and the baseline loses desired breath detail.
6. Check every final segment for omitted/repeated words, pronunciation, clipped consonants, flat delivery, abrupt joins, and identity drift. Signal checks detect silence or clipping, but cannot certify emotional quality. Keep original segment WAVs and exact settings so a weak segment can be regenerated independently.

Warm intimate narration does not require continuous whispering. Gentle voiced delivery generally leaves more room for clear enunciation and vocal contrast; treat that as the preferred artistic target for this project, subject to listening.


The v2 implementation restores engine-independent mastering as optional finishing, with automatic mastering as the default and spectral matching opt-in. This can improve listening level and tone, not create emotional acting. See [the proposal-by-proposal assessment and AWS review](omni-v2-review.md).
