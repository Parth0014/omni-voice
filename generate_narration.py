"""OmniVoice-only gratitude narration: canonical text, directed phrasing, native speech."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf

from omni_voice import DEFAULT_URL, validate_settings
from speech_engine import create_engine
from story_audio import SAMPLE_RATE, assemble, file_hash, normalize_delivery, prepare_reference, signal_report
from storytelling import build_story_plan
from transcript_quality import check_transcript
from worker_document import extract_worker_blocks

BASE_DIR = Path(__file__).resolve().parent
ENGINE_VERSION = "omnivoice-story-v2"


def load_runtime_environment(base_dir=BASE_DIR, environ=None):
    values = {}
    path = Path(base_dir) / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.lstrip().startswith("#"):
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ if environ is None else environ)
    return values


def build_runtime_config(base_dir, environment):
    def path(key, default=None):
        value = environment.get(key) or default
        if value is None:
            return None
        result = Path(value).expanduser()
        return str(result if result.is_absolute() else Path(base_dir) / result)
    return {
        "post_html_file": path("NARRATION_POST_HTML_FILE", "sample.html"),
        "narration_reference_audio": path("NARRATION_REFERENCE_AUDIO", "voice_samples/reference.wav"),
        "quote_reference_audio": path("NARRATION_QUOTE_REFERENCE_AUDIO"),
        "quote_mode": environment.get("NARRATION_QUOTE_MODE", "preserve"),
        "speed": float(environment.get("NARRATION_SPEED", "1.0")),
        "delivery": environment.get("NARRATION_DELIVERY", "warm"),
        "endpoint": environment.get("OMNIVOICE_URL", DEFAULT_URL),
        "output_dir": str(Path(base_dir) / "output" / "stories"),
    }


def write_json(path, payload):
    path = Path(path)
    descriptor, temp = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _audio_cache(directory, payload, generate):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    audio_path, manifest_path = directory / f"{key}.wav", directory / f"{key}.json"
    if audio_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            audio, rate = sf.read(audio_path, dtype="float32")
            if rate != SAMPLE_RATE or manifest.get("payload") != payload or manifest["audio_sha256"] != file_hash(audio_path):
                raise ValueError("Cached audio fingerprint mismatch")
            signal_report(audio, payload["text"])
            return audio, True, manifest.get("metadata", {})
        except (ValueError, KeyError, OSError, RuntimeError):
            pass
    audio, metadata = generate()
    signal_report(audio, payload["text"])
    descriptor, temp = tempfile.mkstemp(suffix=".wav", dir=directory)
    os.close(descriptor)
    try:
        sf.write(temp, audio, SAMPLE_RATE, subtype="FLOAT")
        os.replace(temp, audio_path)
        write_json(manifest_path, {"payload": payload, "audio_sha256": file_hash(audio_path), "metadata": metadata})
    finally:
        Path(temp).unlink(missing_ok=True)
    # This directory contains only our cache entries. Prune oldest complete audio
    # entries after use; retained in-memory audio remains available for assembly.
    entries = sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in entries)
    for entry in entries:
        if total <= 128 * 1024 * 1024:
            break
        if entry == audio_path:
            continue
        try:
            total -= entry.stat().st_size
            entry.unlink()
            entry.with_suffix(".json").unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    return audio, False, metadata


def _write_player(path, wav, report):
    rows = "".join(
        f'<tr><td>{i + 1}</td><td>{html.escape(c["role"])}</td>'
        f'<td><button onclick="playAt({c["start_seconds"]:.3f})">Listen</button></td>'
        f'<td>{html.escape(c["original_text"])}</td></tr>'
        for i, c in enumerate(report["chunks"])
    )
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in report["warnings"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Gratitude narration</title>
<style>body{{background:#f5f1e8;color:#253b36;max-width:1000px;margin:40px auto;padding:24px;font:17px/1.6 system-ui}}audio{{width:100%}}td{{padding:12px;border-bottom:1px solid #d3dbd5;vertical-align:top}}button,a{{color:#245c4b}}button{{padding:8px;cursor:pointer}}small{{color:#566}}</style>
<h1>Your gratitude story</h1><p>{report['duration_seconds'] / 60:.2f} minutes ? {len(report['chunks'])} phrases ? {html.escape(report['delivery'])} delivery</p>
<audio id="player" controls src="{html.escape(wav.name)}"></audio>
<p><a href="{html.escape(wav.name)}" download>Download audio</a> ? <a href="{html.escape(wav.with_suffix('.json').name)}">Generation report</a></p>
<p>Listen for correct names, clear consonants, natural emphasis and a consistent voice. Signal checks cannot judge storytelling.</p>
<details><summary>Review notes</summary><ul>{warnings}</ul></details>
<table><thead><tr><th>Phrase</th><th>Voice</th><th>Play</th><th>Script</th></tr></thead><tbody>{rows}</tbody></table>
<script>function playAt(t){{let p=document.getElementById('player');p.currentTime=t;p.play();}}</script></html>"""
    Path(path).write_text(page, encoding="utf-8")


def run_pipeline(post_html_file, narration_reference_audio, quote_reference_audio=None,
                 quote_mode="preserve", output_dir=None, speed=1.0, *, delivery="warm",
                 reference_text="", quote_reference_text="", reference_start_seconds=None,
                 quote_reference_start_seconds=None, endpoint=None, steps=32, guidance=2.0,
                 denoise=True, instruct="", pronunciation=None, directions=None,
                 max_chunks=None, take=0, normalize=True, plan_only=False, retake_chunks=None, remote_timeout=None, model_revision=None,
                 engine=None, engine_instance=None, normalize_text=None, verify_text=None,
                 allow_auto_reference=False, mastering=None, spectral_matching=False, reference_manifest=None):
    """Return a complete local WAV; AWS transport and quote contracts remain intact.

    Warm delivery is an editorial pacing preset, not an emotion prompt. The
    reference performance supplies timbre and much of the prosody. The backend is
    swappable; optional mastering is applied only after raw speech is assembled.
    """
    if isinstance(speed, bool) or not math.isfinite(float(speed)) or not .5 <= float(speed) <= 1.5:
        raise ValueError("speed must be a native OmniVoice multiplier between 0.5 and 1.5")
    if isinstance(take, bool) or not isinstance(take, int) or take < 0:
        raise ValueError("take must be a non-negative integer; it selects a new cache namespace, not a model seed")
    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    if not isinstance(normalize, bool) or not isinstance(denoise, bool):
        raise ValueError("normalize and denoise must be booleans")
    if not isinstance(reference_text, str) or not isinstance(quote_reference_text, str):
        raise ValueError("Reference transcripts must be strings")
    verify_text = verify_text or os.environ.get("NARRATION_VERIFY_TEXT", "auto")
    if verify_text not in {"auto", "required", "off"}:
        raise ValueError("verify_text must be auto, required, or off")
    if normalize_text is not None and not isinstance(normalize_text, bool):
        raise ValueError("normalize_text must be a boolean or None")
    if not isinstance(spectral_matching, bool):
        raise ValueError("spectral_matching must be boolean")
    mastering = mastering or ("auto" if normalize else "off")
    if mastering not in {"gain", "off", "auto", "natural", "warm_story"}:
        raise ValueError("Unsupported mastering profile")
    if spectral_matching and mastering in {"gain", "off"}:
        raise ValueError("Spectral matching requires auto, natural or warm_story mastering")
    validate_settings(steps=steps, guidance=guidance, speed=float(speed), instruct=instruct, denoise=denoise)
    post_html = Path(post_html_file).read_text(encoding="utf-8-sig")
    extracted_blocks = extract_worker_blocks(post_html)
    plan = build_story_plan(extracted_blocks, quote_mode=quote_mode, pronunciation=pronunciation,
                            directions=directions, delivery=delivery)
    if not plan:
        raise ValueError("No narration remains after extraction and quote mode")
    if max_chunks is not None:
        plan = plan[:max_chunks]
    for item in plan:
        item["native_speed"] = float(speed) * item["speed_factor"]
        if not .5 <= item["native_speed"] <= 1.5:
            raise ValueError("Combined story direction and speed exceed OmniVoice's 0.5?1.5 range")
    if retake_chunks is not None:
        if (not isinstance(retake_chunks, list) or not retake_chunks or take < 1
                or any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= len(plan)
                       for i in retake_chunks)):
            raise ValueError("retake_chunks must contain valid one-based chunk numbers and requires take >= 1")
    roles = set(item["role"] for item in plan)
    if "quote" in roles and quote_mode == "two_voice" and not quote_reference_audio:
        raise ValueError("two_voice requires a separate quote reference recording")
    output_dir = Path(output_dir or BASE_DIR / "output" / "stories").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "story-plan.json"
    write_json(plan_path, {"engine": ENGINE_VERSION, "delivery": delivery, "chunks": plan})
    if plan_only:
        print(f"Prepared {len(plan)} phrases without inference: {plan_path}")
        return str(plan_path)
    cache_dir = Path(os.environ.get("OMNIVOICE_CACHE_DIR", str(output_dir / "_cache"))).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    references, reference_reports, transcripts = {}, {}, {}
    for role in sorted(roles):
        separate_quote = role == "quote" and quote_mode == "two_voice"
        source = quote_reference_audio if separate_quote else narration_reference_audio
        transcript = quote_reference_text if separate_quote else reference_text
        if not transcript:
            registry_path = reference_manifest or os.environ.get("OMNIVOICE_REFERENCE_MANIFEST")
            if registry_path:
                catalog = json.loads(Path(registry_path).read_text(encoding="utf-8-sig"))
                entry = catalog.get(file_hash(source), {})
                if entry.get("language", "English") != "English":
                    raise ValueError("The reference manifest language must match this English narration")
                transcript = entry.get("text", "")
        if not isinstance(transcript, str) or (not transcript.strip() and not allow_auto_reference):
            raise ValueError("Supply the exact reference transcript or a SHA-256 keyed reference manifest")
        start = quote_reference_start_seconds if separate_quote else reference_start_seconds
        references[role], reference_reports[role] = prepare_reference(
            source, output_dir / "_references", start=start, transcript=transcript)
        transcripts[role] = transcript
    if "quote" in roles and "narration" in roles and quote_mode == "two_voice":
        if reference_reports["quote"]["anchor_sha256"] == reference_reports["narration"]["anchor_sha256"]:
            raise ValueError("two_voice requires distinct selected reference recordings")
    client = engine_instance or create_engine(engine, endpoint=endpoint, timeout=remote_timeout, revision=model_revision)
    capabilities = getattr(client, "capabilities", {})
    use_normalization = capabilities.get("normalize_text", False) if normalize_text is None else normalize_text
    if use_normalization and not capabilities.get("normalize_text"):
        raise ValueError("This endpoint does not expose normalize_text; deploy the supplied Colab wrapper")
    can_verify = bool(capabilities.get("transcribe")) and verify_text != "off"
    if verify_text == "required" and not can_verify:
        raise ValueError("Required transcript verification needs the upgraded GPU/Colab runtime")
    started = time.monotonic()
    clips, chunks = [], []
    print(f"OmniVoice ? {delivery} ? {len(plan)} phrases ? quotes: {quote_mode}", flush=True)
    for index, item in enumerate(plan):
        role = item["role"]
        settings = dict(reference_text=transcripts[role], instruct=instruct, language="English",
                        steps=steps, guidance=guidance, speed=item["native_speed"],
                        denoise=denoise, preprocess=True, postprocess=False)
        selected_take = take if retake_chunks is None or index + 1 in retake_chunks else 0
        if capabilities.get("normalize_text"):
            settings["normalize_text"] = use_normalization
        if capabilities.get("seed"):
            seed_material = f"{item['text']}|{selected_take}|{reference_reports[role]['anchor_sha256']}"
            settings["seed"] = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % (2**31)
        payload = {"engine": ENGINE_VERSION, "model": client.identity, "text": item["text"],
                   "reference_sha256": reference_reports[role]["anchor_sha256"],
                   "settings": settings, "take": selected_take, "verification": verify_text, "retry_policy": "two-attempts-v2"}

        def generate(item=item, role=role, settings=settings):
            rejected = []
            for attempt in range(2):
                request = dict(settings)
                if "seed" in request:
                    request["seed"] = (request["seed"] + attempt * 1000003) % (2**31)
                audio = client.synthesize(item["text"], references[role], **request)
                metadata = dict(getattr(client, "last_metadata", {}))
                metadata["verification"] = {"status": "unverified", "reason": "ASR unavailable or disabled"}
                try:
                    signal_report(audio, item["text"])
                    if can_verify:
                        actual = client.transcribe(audio)
                        expected = metadata.get("normalized_text", item["original_text"])
                        # ARPAbet spells sounds, not words. Use source wording for
                        # ASR comparison when a pronunciation dictionary was applied.
                        if "[" in expected:
                            expected = item["original_text"]
                        verification = check_transcript(expected, actual, transcripts[role])
                        metadata["verification"] = verification
                        if verification["status"] == "reject" or (verify_text == "required" and verification["status"] == "review"):
                            raise ValueError(f"Transcript check rejected: {verification}")
                    metadata["rejected_attempts"] = rejected
                    return audio, metadata
                except ValueError as exc:
                    rejected.append(str(exc))
                    print(f"Take rejected (attempt {attempt + 1}): {exc}", flush=True)
            raise RuntimeError(f"Speech failed quality checks twice: {rejected}")

        clip, hit, generation_metadata = _audio_cache(cache_dir, payload, generate)
        quality = signal_report(clip, item["original_text"])
        clips.append(clip)
        chunks.append({**item, "cache_hit": hit, "quality": quality, "settings": settings, "generation": generation_metadata})
        print(f"{index + 1}/{len(plan)} {role}: {quality['duration_seconds']:.1f}s"
              + (" (cached)" if hit else ""), flush=True)
    joined, timeline = assemble(clips, plan)
    if mastering in {"gain", "off"}:
        joined, finishing = normalize_delivery(joined, enabled=mastering == "gain")
    else:
        from narration_mastering import master_audio

        spectral_reports = None
        if spectral_matching:
            from spectral_matching import match_roles

            spans = [(round(t["start_seconds"] * SAMPLE_RATE), round(t["end_seconds"] * SAMPLE_RATE), p["role"])
                     for t, p in zip(timeline, plan)]
            joined, spectral_reports = match_roles(joined, SAMPLE_RATE, spans, references)
        joined, finishing = master_audio(joined, SAMPLE_RATE, mastering, temp_dir=output_dir,
                                         spectral_reports=spectral_reports)
        finishing["method"] = "mastering"
        finishing["estimated_output_lufs"] = finishing["after"]["input_i"]
    for chunk, times in zip(chunks, timeline):
        chunk.update(times)
    warnings = list(dict.fromkeys(
        [warning for r in reference_reports.values() for warning in r["warnings"]]
        + [warning for c in chunks for warning in c["quality"]["warnings"]]
        + (["This endpoint exposes no seed control; take numbers select new cache entries."] if not capabilities.get("seed") else [])
        + (["Output words were not ASR-verified; use the upgraded GPU runtime for reference-leak checks."] if not can_verify else [])
        + (["The public demo cannot disable internal splitting or normalize numerals; explicit pronunciation entries remain available."]
           if not capabilities.get("disable_internal_chunking") else [])
        + ["ASR review requested for phrase " + str(i + 1) for i, c in enumerate(chunks)
           if c["generation"].get("verification", {}).get("status") == "review"]
        + ["Emotion and sentence emphasis require a listening review; decoding settings do not guarantee acting quality."]
    ))
    report = {"engine": ENGINE_VERSION, "model": client.identity, "delivery": delivery,
              "quote_mode": quote_mode, "speed": float(speed), "take": take,
              "duration_seconds": len(joined) / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
              "source_sha256": file_hash(post_html_file), "reference_anchors": reference_reports,
              "chunks": chunks, "finishing": finishing, "warnings": warnings,
              "generation_seconds": time.monotonic() - started}
    # Reserve a fresh destination only after every phrase is ready.
    number = 1
    while True:
        output = output_dir / f"narration_{number:04d}.wav"
        try:
            descriptor = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            number += 1
    descriptor, temp = tempfile.mkstemp(suffix=".wav", dir=output_dir)
    os.close(descriptor)
    try:
        sf.write(temp, joined, SAMPLE_RATE, subtype="PCM_16")
        delivered, rate = sf.read(temp, dtype="float32")
        signal_report(delivered)
        if rate != SAMPLE_RATE or len(delivered) != len(joined):
            raise RuntimeError("Final WAV validation failed")
        report["output_sha256"] = file_hash(temp)
        write_json(output.with_suffix(".json"), report)
        os.replace(temp, output)
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix(".json").unlink(missing_ok=True)
        raise
    finally:
        Path(temp).unlink(missing_ok=True)
    report_directory = os.environ.get("OMNIVOICE_REPORT_DIR")
    if report_directory:
        retained_reports = Path(report_directory)
        retained_reports.mkdir(parents=True, exist_ok=True)
        write_json(retained_reports / (report["output_sha256"] + ".json"), report)
    _write_player(output.with_suffix(".html"), output, report)
    print(f"Ready: {output} ({report['duration_seconds'] / 60:.2f} minutes)", flush=True)
    return str(output)


def main(argv=None):
    environment = load_runtime_environment()
    config = build_runtime_config(BASE_DIR, environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=config["post_html_file"])
    parser.add_argument("--reference", default=config["narration_reference_audio"])
    parser.add_argument("--quote-reference", default=config["quote_reference_audio"])
    parser.add_argument("--quote-mode", choices=("preserve", "exclude", "two_voice"), default=config["quote_mode"])
    parser.add_argument("--speed", type=float, default=config["speed"])
    parser.add_argument("--delivery", choices=("warm", "natural"), default=config["delivery"])
    parser.add_argument("--endpoint", default=config["endpoint"])
    parser.add_argument("--engine", choices=("gradio", "native"), default=environment.get("NARRATION_ENGINE", "gradio"))
    parser.add_argument("--local-asr-model", type=Path, help="Downloaded faster-whisper model directory for local word checks")
    parser.add_argument("--normalize-text", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verify-text", choices=("auto", "required", "off"), default=environment.get("NARRATION_VERIFY_TEXT", "auto"))
    parser.add_argument("--allow-auto-reference", action="store_true")
    parser.add_argument("--reference-manifest", default=environment.get("OMNIVOICE_REFERENCE_MANIFEST"))
    parser.add_argument("--mastering", choices=("gain", "off", "auto", "natural", "warm_story"))
    parser.add_argument("--spectral-matching", action="store_true")
    parser.add_argument("--remote-timeout", type=float, default=float(environment.get("OMNIVOICE_TIMEOUT_SECONDS", "180")))
    parser.add_argument("--model-revision", default=environment.get("OMNIVOICE_MODEL_REVISION"))
    parser.add_argument("--reference-text", default=environment.get("OMNIVOICE_REFERENCE_TEXT", ""))
    parser.add_argument("--quote-reference-text", default=environment.get("OMNIVOICE_QUOTE_REFERENCE_TEXT", ""))
    parser.add_argument("--reference-start", type=float)
    parser.add_argument("--quote-reference-start", type=float)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pronunciation", type=Path, help="JSON mapping exact words to spoken forms or [CMU PHONEMES]")
    parser.add_argument("--directions", type=Path, help="JSON block-index mapping to pace/pause_after_ms")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--take", type=int, default=0)
    parser.add_argument("--retake-chunks", help="Comma-separated one-based chunks to regenerate with --take")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output-dir", default=config["output_dir"])
    args = parser.parse_args(argv)
    options = vars(args)
    local_asr_model = options.pop("local_asr_model")
    if local_asr_model is not None and not options["plan_only"]:
        from local_transcription import LocalTranscriptionEngine

        backend = create_engine(options["engine"], endpoint=options["endpoint"],
                                timeout=options["remote_timeout"], revision=options["model_revision"])
        options["engine_instance"] = LocalTranscriptionEngine(backend, model_path=local_asr_model)
    if options["retake_chunks"] is not None:
        options["retake_chunks"] = [int(value) for value in options["retake_chunks"].split(",")]
    options["post_html_file"] = options.pop("html")
    options["narration_reference_audio"] = options.pop("reference")
    options["quote_reference_audio"] = options.pop("quote_reference")
    options["reference_start_seconds"] = options.pop("reference_start")
    options["quote_reference_start_seconds"] = options.pop("quote_reference_start")
    options["normalize"] = not options.pop("no_normalize")
    for key in ("pronunciation", "directions"):
        if options[key] is not None:
            options[key] = json.loads(options[key].read_text(encoding="utf-8-sig"))
    return run_pipeline(**options)


if __name__ == "__main__":
    main()
