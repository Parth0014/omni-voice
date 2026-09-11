# OmniVoice gratitude narration

A fresh OmniVoice narration pipeline for warm, intimate gratitude stories. Speech runs on your Colab endpoint; extraction, planning, caching and the browser player run locally. The local client needs no Torch or GPU. Restored engine-independent mastering runs after raw speech assembly; reference spectral matching is optional.

## Run locally

Double-click **Run-Local-Test.cmd**. It reads `local-test.settings.json`, generates the full story, and opens a browser audio player. The current local settings use `sample.html`, your `b1.1.wav` reference and its exact transcript.

For a new installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Python 3.10 through 3.14 is supported. The repository's isolated `.venv` is preferred by the launcher. You can also create the `omnivoice` Conda environment from `environment.yml`.

Direct commands:

```powershell
.\.venv\Scripts\python.exe generate_narration.py --plan-only
.\.venv\Scripts\python.exe story_audition.py
.\.venv\Scripts\python.exe generate_narration.py
```

The CLI reads `.env`; the one-click launcher reads `local-test.settings.json`. Keep the Colab session running. Both send the text and prepared reference audio to the configured public endpoint. Update `OMNIVOICE_URL` in `.env` (or `endpoint` in local settings) when Colab changes its share URL.

The audition command creates six matched-volume examples: natural/32/guidance 2, warm/32/2, warm/48/2, warm/48/3, warm/64/2, warm/64/3. It does not choose a winner automatically. Each final generation saves a WAV, a JSON report and an HTML player with phrase navigation. The CLI defaults to `output/stories`; one-click runs use separate folders under `output/local-tests`.

## How the delivery works

The warm preset uses coherent sentence/paragraph phrasing, modest native speed changes around quotes and endings, and short editorial pauses. It preserves the source wording and natural breaths. Default `--mastering auto` applies measured finishing. Other profiles are `natural`, `warm_story`, `gain` (constant gain only) and `off`. Add `--spectral-matching` only when auditioning reference-derived tonal correction.

Your reference performance is the main style guide. OmniVoice supports pronunciation phonemes and limited voice attributes; it does not provide a general emotional acting prompt or reliable word-by-word dramatic emphasis. See [research and evidence](docs/omni-source-research.md) and the [storytelling workflow](docs/omni-storytelling.md).

Quote modes remain:

- `preserve`: same narrator for the story and quotes.
- `exclude`: omit quotation blocks.
- `two_voice`: second reference for quote words; attribution stays with the narrator.

Use `--quote-mode two_voice --quote-reference path/to/quote.wav` for a second speaker. Reference clips should contain 3-10 seconds of clean, expressive speech. Supply their exact transcript with `--reference-text` / `--quote-reference-text` or the local settings fields.

## Pronunciation and retakes

Create a JSON pronunciation file mapping exact words to their spoken form or CMU phonemes. The provided `pronunciation.example.json` is only an example, not an enabled story dictionary. Names require confirmed pronunciation.

```powershell
.\.venv\Scripts\python.exe generate_narration.py --pronunciation pronunciation.example.json --plan-only
.\.venv\Scripts\python.exe generate_narration.py --take 1 --retake-chunks 2,5
```

Selective retakes require the same output/cache directory and all other settings as the original. Unselected chunks reuse take 0; selected chunks use the given take number. On native/upgraded runtimes, each take also derives a stable request seed. The stock demo only supports requesting a fresh stochastic render. Cross-hardware reproducibility is not guaranteed. For a second selective edit, include every chunk whose nonzero take should remain, or keep the previous final output as your reference.

## AWS

The worker still calls `generate_narration.run_pipeline` with the existing HTML, reference, quote and speed arguments. Canonical extraction, Studio, S3 output, idempotency and job status contracts remain. The worker image now contains the remote OmniVoice client instead of a local synthesis model. Rebuild the image and configure `OMNIVOICE_URL` to activate this implementation in AWS; no AWS deployment has been performed.

Exact transcripts are required by default. AWS uses a mounted SHA-256 keyed voice-reference manifest, preserving the existing job schema; see `voice-transcripts.example.json`. Unregistered references fail clearly. Do not reuse a global transcript with unrelated voices.

For permanent GPU inference, use `Dockerfile.gpu`, `gpu_worker.py` and the ECS EC2 task template under `aws/omni-voice-gpu`. Start on a dedicated test FIFO: the current Lambda consumer is active. The GPU consumer preserves the existing SQS/S3 handler and renews visibility while processing. No GPU image has been deployed or tested on real GPU hardware yet. See [the infrastructure and implementation review](docs/omni-v2-review.md).

## Upgraded Colab runtime

Open [OmniVoice-Narration-Colab.ipynb](OmniVoice-Narration-Colab.ipynb) in a GPU Colab, upload its three listed Python files and run the cells. Set its printed URL locally, then run:

```powershell
.\.venv\Scripts\python.exe generate_narration.py --normalize-text --verify-text required
```

This runtime adds numeral normalization, seeded takes, reusable clone prompts, disabled internal splitting and output ASR. The stock demo lacks these controls and reports its output as unverified. Explicitly required unsupported controls fail before synthesis. ASR checks reject reference-text leakage and retry once; required mode also rejects excessive transcript disagreement. ASR cannot judge emotion and can mishear names.

## Verification and migration

Run `python -m pytest` and `python -m ruff check .` in a suitable development environment. The dev extra includes `boto3` for AWS contract tests.

Retired engine, enhancement and experiment files were removed after a recovery snapshot was saved under `output/migration-backups`. Existing audio, source content, AWS infrastructure and unrelated Studio edits were retained. Mastering and spectral matching were restored unchanged from the snapshot. Use `--help` for current engine options; obsolete PocketTTS backend and enhancement flags are retired.
