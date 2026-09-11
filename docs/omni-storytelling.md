# Gratitude-story production workflow

The artistic target is warm, intimate, clearly voiced storytelling with gentle expression. Constant whispering is not the default. The research basis and model limits are in [the source review](omni-source-research.md).

## Start with the performance

Use one continuous 3-10 second reference with the delivery you want: one listener, a complete thought, relaxed pace, clear word endings and a gentle pitch change. Avoid a montage of samples. A quiet, flat reference can produce a recognizably similar but still flat clone. A different instruction cannot reliably repair that performance.

The current local reference is `b1.1.wav` (7.67 seconds). Its exact transcript is stored in local settings and `.env`. The preparation stage keeps the continuous clip and converts it to mono 24 kHz. It does not splice together ranked segments. Reference-derived spectral matching is a separate opt-in finishing stage. For a longer recording, select a continuous excerpt; when supplying an exact transcript, cut the file first so the transcript matches exactly.

## Review the script before synthesis

Run `python generate_narration.py --plan-only`. Inspect `output/stories/story-plan.json`. It shows original wording, spoken input, source block indexes, voice routing, pauses and native pace. The canonical extractor handles Ghost content, including headings, lists and quotations. Review this plan when importing a full public webpage: navigation and unrelated page content should not become narration.

Whole sentences are packed within paragraphs, targeting up to 420 characters. Long sentences split at a substantial clause or safe word boundary. Names, decimals, common abbreviations and phoneme strings are protected. There is no automatic paraphrasing, emotional stage direction, forced uppercase emphasis or invented personal detail.

`preserve` retains quotes with the same narrator. `exclude` removes quote blocks. `two_voice` uses the second reference for quote words and keeps a named attribution with the narrator. The old rotating attribution verbs were retired.

## Pronunciation and enunciation

Use a JSON lexicon for mispronounced words. Replacements match whole words, longest first, in one pass. Values can be spoken spellings or bracketed CMU ARPAbet. An example is `{"gratitude": "[G R AE1 T AH0 T UW2 D]"}`. Use `--pronunciation pronunciation.example.json` to try the supplied demonstration.

Confirm a person's name before adding a phonetic spelling; none has been invented for this story. Expand an ambiguous acronym, number, date or currency amount through an explicit lexicon entry and review the resulting plan. The upgraded GPU/Colab runtime enables numeric normalization and records normalized text; the stock demo cannot expose this option. Review dates, years and identifiers because their intended readings can be ambiguous.

ARPAbet stress digits control pronunciation within a word. They do not guarantee dramatic emphasis on that word in a sentence. For emotional emphasis, start with a suitable reference, preserve sentence context, create space around a complete thought, and audition a fresh take. Do not put unsupported `[emphasis]`, `[whisper]` or SSML in the script. The model has a separate limited `whisper` voice attribute, which is not a default for this project.

## Shape pacing without dragging the voice

The `warm` preset normally uses native speed 1.0, headings 0.98, quotations and the final narrative paragraph 0.97. These are deliberately modest editorial settings, not measured emotion controls. `natural` uses 1.0 throughout. `--speed` multiplies the native values; there is no post-generation time stretch.

Use `--directions directions.example.json` to override selected source blocks. Keys are zero-based block indexes from the plan. `pace` accepts 0.8-1.2; `pause_after_ms` accepts 0-2000. A useful artistic pattern is a steady opening, forward movement through events, and a little space after a realization. Adjust specific blocks instead of slowing every line. A final pause has no effect after the story ends. Optional `before` / `after` fields accept a supported tag name such as `sigh` or `surprise-ah`, without brackets. Tags affect only the spoken input, not the recorded source wording; none are inserted by default.

The renderer keeps all generated audio, including internal breaths and silence. It only fills a shortfall in near-silent space between chunks; it never chops off a quiet consonant to force a gap. Colab output postprocessing is disabled to avoid removing its long internal pauses. Existing model edge fades can still be present.

## Audition and select

Run `python story_audition.py`. The browser page compares the same narrative passage, quotation and closing in natural/32/guidance 2, warm/32/2, warm/48/2, warm/48/3, warm/64/2 and warm/64/3 configurations, matched to a common loudness using attenuation only. More decoding steps are not an emotion setting; choose by listening. The stock demo does not expose seeds. The upgraded runtime uses controlled per-take seeds; hardware/runtime differences can still affect results.

Listen for:

- Names and medical terms read correctly; consonants and endings remain audible.
- Every intended word present once, without added or repeated phrases.
- Gentle changes in pitch and stress; no monotonous rhythm or exaggerated acting.
- A consistent narrator, comfortable pauses, and no audible cut at a join.
- An emotionally respectful delivery for the difficult parts of the story.

The page supports per-candidate listening notes. Signal checks flag silence, clipping and suspicious speed. The upgraded runtime additionally compares output ASR against intended text and detects reference-text leakage. `--verify-text required` rejects excessive disagreement; the default `auto` flags disagreement for review. ASR can mishear names and nonverbal sounds; it cannot certify every word or assess gratitude and emotion.

## Fix a weak phrase, then deliver

Each final HTML player has phrase navigation. From the same output directory and settings, `python generate_narration.py --take 1 --retake-chunks 2,5` generates new audio for those one-based phrase numbers and assembles another complete file. Unselected phrases reuse baseline take 0. Include the full set of desired retakes in subsequent commands; outputs are numbered and previous renders are retained.

The cache includes model endpoint/revision, exact text, prepared reference fingerprint, transcript, native controls and take. Bump `OMNIVOICE_MODEL_REVISION` after changing the model behind an unchanged endpoint. Keep this value in the CLI `.env` or worker process environment. Cache data is bounded at 128 MiB for normal entries.

Finishing reuses the original `narration_mastering.py` unchanged. Default `--mastering auto` applies measured loudness/peak control and conservative processing when needed. `natural` and `warm_story` are alternative presets. `--mastering gain` retains a single constant gain targeting -19 LUFS subject to -1.5 dBTP; `--mastering off` bypasses finishing. `--spectral-matching` enables separate reference-derived correction with a mastering profile. Processing cannot supply missing vocal expression. Raw cached speech remains reusable when changing any finishing setting.

Use `python narration_mastering.py path/to/story.wav --compare-dir output/new-mastering-review` to audition mastering without any new remote inference. The existing full-story comparison is `output/omni-mastering-review/index.html`.

## Deployment considerations

The local browser pages are output/audition players, not an always-running editor. The existing AWS Studio and content contracts remain; AWS has not been redeployed. The supplied `colab_runtime.py` and `omni_native.py` expose model identity, reusable clone prompts, manual transcripts, per-request seeds, normalization and output ASR. See `OmniVoice-Narration-Colab.ipynb` and [the v2 infrastructure review](omni-v2-review.md) for activation and remaining GPU validation. The public demo reloads the reference prompt for each call and may disconnect. Keep long jobs out of a short Lambda timeout, and benchmark throughput before moving production traffic.

The old implementation is recoverable from `output/migration-backups/before-omnivoice-rebuild-20260909-154911.zip`. It is no longer on the active generation path.
