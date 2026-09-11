# OmniVoice migration review

Reviewed 10 September 2026. Target: warm, intimate English gratitude stories with clear speech and gentle expression. These changes are implemented locally on `dev/omni-voice`; no AWS cutover or new GPU inference is claimed.

## Assessment of the proposed improvements

| Proposal | Assessment and implementation |
| --- | --- |
| Keep block-aware chunking | Correct. `storytelling.py` carries the philosophy into the new planner: separate source blocks, pack complete sentences, split oversized clauses safely and retain original text. |
| Keep extraction and three quote modes | Correct. Canonical Ghost normalization and worker extraction remain. Preserve/exclude/two_voice are tested; named attribution stays with the narrator. The engine knows nothing about Ghost or quotes. |
| Reuse mastering | Correct. `narration_mastering.py` and `spectral_matching.py` were restored unchanged. Finishing occurs after synthesis and can change without invalidating raw takes. Automatic mastering is the default; reference-derived spectral matching is optional and needs listening review. |
| Use 32-48 steps | Adopted. Recent prior runs already used 32. Baseline is 32; audition 48. More steps are not an emotion control. |
| Raise guidance toward 3 | A reasonable experiment, not a guaranteed improvement. Baseline 2; compare warm/48/2 against warm/48/3 with the same passage and seed where supported. |
| Hard reference constraint 3-10 seconds | Upstream recommends this range; it is not a universal hard model limit. This application enforces it and never silently cuts a longer exact-transcript recording. One speaker, clean speech and matching language still need reference review. |
| Manual reference transcript | Required by default. Local settings contain the supplied exact `b1.1.wav` transcript. AWS resolves transcripts by original file SHA-256 rather than applying a global transcript to unrelated uploads. |
| Detect reference text in output | Added output-ASR checks and one retry with a different seed. Reference-exclusive six-word sequences are rejected. Required mode also rejects WER above 0.20. This is a heuristic, not proof that every word is correct. |
| Normalize numerals | Native/upgraded runtime installs `omnivoice[tn]` and calls `normalize_text=True`. Reports record normalized input. Ambiguous identifiers, dates and names still need pronunciation review. |
| Nonverbal tags | Supported as explicit block directions, such as `{"before":"sigh"}`. No routine sigh/laughter injection; those sounds can feel inappropriate in gratitude stories. Original wording remains visible. |
| Instruct with reference audio | Supported with documented attributes. Matching attributes may help; conflicting attributes usually lose to the reference. No invented free-form emotional acting prompt. |
| Disable internal splitting | Native/upgraded adapter bounds requests and raises the internal threshold so the upstream splitter is not selected. The stock demo cannot expose this control. Setting `audio_chunk_duration=0` alone is insufficient in the inspected source. |
| Diffusion cache needs rework | Correct about identity, but diffusion does not make caching invalid. Keys cover the complete request/model/reference/take/verification policy, not just voice identity. Audio has a checksum; rejected takes are not cached. |
| GPU-backed inference | Correct for the intended runtime. Added a serial GPU worker using the existing SQS/S3 handler, a GPU image and an ECS EC2 task template. Colab remains the temporary GPU host. |

Primary evidence: [generation parameters](https://github.com/k2-fsa/OmniVoice/blob/master/docs/generation-parameters.md), [reference/instruction tips](https://github.com/k2-fsa/OmniVoice/blob/master/docs/tips.md), [voice attributes](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md), [pronunciation and tags](https://github.com/k2-fsa/OmniVoice#non-verbal--pronunciation-control), [model source](https://github.com/k2-fsa/OmniVoice/blob/master/omnivoice/models/omnivoice.py). The reference-text issue is a [closed user report](https://github.com/k2-fsa/OmniVoice/issues/21), not evidence of a verified current fix. The [paper's Appendix B](https://arxiv.org/html/2604.00688v1) reports mixed changes from additional steps, not a storytelling benchmark.

## Existing infrastructure

The repository's flow is Ghost HTML -> canonical narration blocks -> Studio/manager immutable S3 input and DynamoDB state -> FIFO job queue -> worker -> immutable S3 WAV -> status queue/Studio. Publishing is separate. The worker transports existing quote modes and V2 tempo to `run_pipeline`; those schemas remain unchanged.

Read-only AWS inspection found `pocket-tts-dev` active in `us-east-1`: an x86_64 image Lambda with **10,240 MB memory and a 900-second timeout**. Its jobs FIFO has an **enabled** Lambda mapping, batch size 1 and maximum concurrency 6. Older repository notes describing 8 GB or a disabled consumer are stale. No AWS configuration was changed.

```mermaid
flowchart LR
  A[Ghost HTML / Studio job] --> B[Canonical extraction]
  B --> C[Quote routing and story plan]
  C --> D[Speech engine contract]
  D --> E[Colab Gradio adapter]
  D --> F[Native GPU adapter]
  E --> G[Raw take cache and word checks]
  F --> G
  G --> H[Pauses and assembly]
  H --> I[Optional mastering]
  I --> J[WAV / report / player or S3 status]
```

The native adapter caches up to eight clone prompts, serializes model/RNG access, resolves the requested model revision to a local Hugging Face snapshot and records runtime identity. Raw caching is separate from mastering. Seeds improve controlled comparisons; identical output across different GPUs/dependency versions is not promised.

## Colab activation

1. Open `OmniVoice-Narration-Colab.ipynb` in Colab and select a GPU runtime.
2. Run installation, upload `omni_voice.py`, `omni_native.py`, `colab_runtime.py`, then run the final cell.
3. Put the printed public URL in local settings or `.env`.
4. Generate with `--normalize-text --verify-text required`. Initial model/ASR downloads can be slow; raise `--remote-timeout` for cold starts if necessary.

The adapter discovers capabilities. Stock `/_clone_fn` remains usable for testing but lacks seeded requests, explicit normalization/split control and output ASR. Reports disclose these limits. Unsupported explicitly required controls fail before synthesis.

## GPU AWS staging

Use `Dockerfile.gpu` and `aws/omni-voice-gpu/task-definition.example.json` as a staging starting point, not a completed deployment. ECS GPU tasks require suitable EC2 GPU hosts and reservations; see [AWS GPU task documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html).

- Create a dedicated test FIFO and DLQ. Do not point the GPU consumer at the active Lambda queue. The runner checks Lambda mappings and refuses that configuration.
- Pin a model commit, build the CUDA image, record its digest and mount persistent model/raw/report storage. The template reserves one GPU and 12 GiB host memory; fit and throughput must be measured on the selected hardware.
- Mount a JSON manifest based on `voice-transcripts.example.json`. Keys are SHA-256 hashes of original WAV objects downloaded from S3; values contain exact text and `language: "English"`. Register narration and quote uploads separately. Studio does not yet collect manual transcripts; this currently requires maintaining the manifest.
- Grant existing scoped S3/status permissions plus receive/delete/change-visibility/get-attributes for the test queue and permission to list Lambda event-source mappings. Provision its log group and ECS execution permissions. Preserve immutable-object access rules.
- Run `python gpu_worker.py --queue-url <test-url>` for the read-only ownership check. Add `--run` to start the staging consumer. It preloads the model, receives one job at a time, renews visibility and acknowledges only completed processing with a valid lease. Failed jobs remain for retry/DLQ handling.
- Submit an existing V2-format test job with registered references. Verify WAV upload, status updates, retries, restart and lease-loss behavior. JSON QA reports remain under `/cache/reports`; S3 currently receives the existing WAV contract only.
- Benchmark generation time, ASR time, VRAM, cold starts and long-story failure rates before choosing capacity or changing the live consumer. ECS shutdown can interrupt long jobs after its grace period; exercise immutable-output retries on real infrastructure.

## Verification and limits

The complete local suite covers extraction, immutable worker contracts, quote/tempo behavior, chunking, raw caching, mastering and mocked Gradio/native/SQS adapters. Real FFmpeg mastering produced `output/omni-mastering-review/index.html`, reusing the prior 197-second narration. Automatic mastering measured approximately -19.02 LUFS versus -21.51 LUFS beforehand. This is a finishing comparison, not new v2 model inference or an emotional-quality assessment.

The GPU image and notebook have not been executed on a GPU. Live v2 auditions were blocked by automatic approval review because the configured public destination differs from the originally authorized URL. No voice or text was uploaded by that attempt. Approval for the current endpoint is needed before those listening tests can run.

Remaining acceptance work: real upgraded-runtime inference, human listening to four candidate deliveries, pronunciation review for story names, and staging AWS end-to-end validation. No automatic metric can certify warmth, appropriate emotion or sentence emphasis.


## Subsequent 64-step local-only request

The user authorized the configured Colab destination and requested 64-step comparisons without AWS or S3 changes. The adapter now accepts 32-64 steps and the audition set includes warm/64/guidance 2 and warm/64/guidance 3. Local launcher settings use 64 steps, guidance 2 and automatic mastering. The connection attempt could not fetch the Gradio configuration from the configured URL; a fresh running Colab URL is required. No new inference or AWS/S3 writes occurred.
