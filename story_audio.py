"""Reference validation, breath-preserving assembly and gain-only delivery audio."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 24000


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    if samples.shape[1] > 2:
        raise ValueError("Use a mono or stereo reference recording")
    # Preserve a single channel if stereo channels cancel each other.
    mono = samples.mean(axis=1)
    loudest = samples[:, np.argmax(np.mean(samples ** 2, axis=0))]
    if np.mean(mono ** 2) < np.mean(loudest ** 2) * 0.25:
        mono = loudest
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    return np.asarray(mono, dtype=np.float32)


def signal_report(samples, text=""):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Generated speech must be nonempty finite mono audio")
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    duration = len(samples) / SAMPLE_RATE
    if peak < 1e-5 or rms < 1e-6 or duration < 0.15:
        raise ValueError("Generated speech is silent or too short")
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    warnings = []
    words = len(re.findall(r"\b[\w']+\b", text))
    wpm = words * 60 / duration if words else None
    if clipped > 0.001:
        warnings.append("Clipping detected; audition this take and regenerate if distorted.")
    if wpm is not None and words >= 12 and not 85 <= wpm <= 220:
        warnings.append("Unusual speaking rate; check for omissions, repetition or long silence by listening.")
    return {"duration_seconds": duration, "peak": peak, "rms": rms,
            "clipped_fraction": clipped, "words_per_minute_estimate": wpm, "warnings": warnings}


def prepare_reference(source, directory, *, start=None, transcript=""):
    source = Path(source).resolve(strict=True)
    audio = read_audio(source)
    duration = len(audio) / SAMPLE_RATE
    if start is not None and (isinstance(start, bool) or not math.isfinite(start) or start < 0):
        raise ValueError("Reference start must be finite and non-negative")
    if duration > 10.01:
        raise ValueError("Supply an already cut 3-10 second reference file; automatic truncation is disabled")
    start = 0.0 if start is None else float(start)
    segment = audio[round(start * SAMPLE_RATE):round((start + 10) * SAMPLE_RATE)]
    if len(segment) < 3 * SAMPLE_RATE:
        raise ValueError("Select a reference segment containing 3–10 seconds of clear speech")
    if transcript and (start > 0 or duration > 10.01):
        raise ValueError("With an exact transcript, supply an already cut 3–10 second reference file")
    report = signal_report(segment)
    report.update({"source_sha256": file_hash(source), "start_seconds": start,
                   "source_duration_seconds": duration, "transcript_supplied": bool(transcript)})
    if duration > 10.01:
        report["warnings"].append("Used a continuous 10-second reference segment; audition its ending.")
    if not transcript:
        report["warnings"].append("Reference transcript is automatic on Colab; supply a verified transcript for repeatability.")
    key = hashlib.sha256(segment.tobytes()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"reference_{key}.wav"
    # FLOAT retains a consistent prepared reference and avoids PCM clipping.
    if not path.exists():
        sf.write(path, segment, SAMPLE_RATE, subtype="FLOAT")
    report["anchor_sha256"] = file_hash(path)
    return path, report


def edge_quiet_samples(audio, *, tail=False):
    """Only near-digital silence counts toward a pause; quiet breaths stay intact."""
    active = np.flatnonzero(np.abs(audio) > 0.0001)
    if not active.size:
        return len(audio)
    return len(audio) - 1 - int(active[-1]) if tail else int(active[0])


def assemble(clips, plan):
    if not clips or len(clips) != len(plan):
        raise ValueError("Every planned chunk must have audio before assembly")
    pieces, timeline, offset = [], [], 0
    for index, (clip, item) in enumerate(zip(clips, plan)):
        signal_report(clip)
        timeline.append({"start_seconds": offset / SAMPLE_RATE,
                         "end_seconds": (offset + len(clip)) / SAMPLE_RATE})
        pieces.append(clip)
        offset += len(clip)
        if index + 1 < len(clips):
            natural = edge_quiet_samples(clip, tail=True) + edge_quiet_samples(clips[index + 1])
            added = max(0, round(item["pause_after_ms"] * SAMPLE_RATE / 1000) - natural)
            if added:
                pieces.append(np.zeros(added, dtype=np.float32))
                offset += added
    return np.concatenate(pieces), timeline


def normalize_delivery(audio, *, enabled=True):
    """Measure LUFS/true peak, then apply one constant gain; never compress or EQ."""
    if not enabled:
        return audio, {"method": "none", "gain_db": 0.0}
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="omni-loudness-") as temp:
        path = Path(temp) / "input.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="FLOAT")
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path),
            "-af", "loudnorm=I=-19:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-",
        ], capture_output=True, text=True, timeout=120, check=True)
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if match is None:
        raise RuntimeError("Could not measure narration loudness")
    measured = json.loads(match.group())
    loudness, true_peak = float(measured["input_i"]), float(measured["input_tp"])
    if not math.isfinite(loudness) or not math.isfinite(true_peak):
        raise ValueError("Narration is too short or too quiet to measure loudness; use --no-normalize for previews")
    gain = min(-19 - loudness, -1.5 - true_peak)
    return (audio * (10 ** (gain / 20))).astype(np.float32), {
        "method": "constant_gain", "gain_db": gain,
        "input_lufs": loudness, "input_true_peak_dbtp": true_peak,
        "estimated_output_lufs": loudness + gain,
        "estimated_output_true_peak_dbtp": true_peak + gain,
        "target_lufs": -19, "target_limited_by_peak": gain < -19 - loudness - 0.01,
    }
