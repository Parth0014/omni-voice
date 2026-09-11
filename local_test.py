"""One-click OmniVoice generation with a local result player and readable failures."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "quote_mode": "preserve", "delivery": "warm", "speed": 1.0,
    "reference_text": "", "quote_reference_text": "", "steps": 32,
    "guidance": 2.0, "instruct": "", "take": 0, "normalize": True, "mastering": "auto",
}
OPTION_FLAGS = {
    "html": "--html", "reference": "--reference", "quote_reference": "--quote-reference",
    "quote_mode": "--quote-mode", "delivery": "--delivery", "speed": "--speed",
    "reference_text": "--reference-text", "quote_reference_text": "--quote-reference-text",
    "reference_start": "--reference-start", "quote_reference_start": "--quote-reference-start",
    "endpoint": "--endpoint", "steps": "--steps", "guidance": "--guidance", "instruct": "--instruct",
    "pronunciation": "--pronunciation", "directions": "--directions", "max_chunks": "--max-chunks",
    "take": "--take", "engine": "--engine", "mastering": "--mastering",
    "local_asr_model": "--local-asr-model",
    "verify_text": "--verify-text", "reference_manifest": "--reference-manifest",
}


def load_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict) or set(settings) - (set(OPTION_FLAGS) | {"normalize"}):
        raise ValueError("Settings must use the documented OmniVoice keys in local-test.settings.json")
    settings = {**DEFAULTS, **settings}
    for field in ("html", "reference"):
        if not isinstance(settings.get(field), str) or not settings[field].strip():
            raise ValueError(f"Set {field} in local-test.settings.json")
    for field in ("html", "reference", "quote_reference", "pronunciation", "directions"):
        value = settings.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a file path or null")
        resolved = Path(value).expanduser()
        resolved = resolved if resolved.is_absolute() else ROOT / resolved
        if not resolved.is_file():
            raise FileNotFoundError(f"{field} file does not exist: {resolved}")
        settings[field] = str(resolved.resolve())
    if settings["quote_mode"] not in {"preserve", "exclude", "two_voice"}:
        raise ValueError("quote_mode must be preserve, exclude, or two_voice")
    if settings["quote_mode"] == "two_voice" and not settings.get("quote_reference"):
        raise ValueError("two_voice requires a quote_reference file")
    if settings["delivery"] not in {"warm", "natural"}:
        raise ValueError("delivery must be warm or natural")
    for field in ("reference_text", "quote_reference_text", "instruct"):
        if not isinstance(settings[field], str):
            raise ValueError(f"{field} must be text; use an empty string for automatic/default behavior")
    for field in ("speed", "guidance", "reference_start", "quote_reference_start"):
        value = settings.get(field)
        if value is None and field.endswith("start"):
            continue
        minimum = 0 if field.endswith("start") else 0.000001
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < minimum):
            kind = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{field} must be a finite {kind} number")
    for field in ("steps", "max_chunks", "take"):
        value = settings.get(field)
        if value is None and field == "max_chunks":
            continue
        minimum = 0 if field == "take" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer of at least {minimum}")
    if settings["mastering"] not in {"off", "gain", "auto", "natural", "warm_story"}:
        raise ValueError("Invalid mastering profile")
    if not isinstance(settings["normalize"], bool):
        raise ValueError("normalize must be true or false")
    endpoint = settings.get("endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, str):
            raise ValueError("endpoint must be an HTTPS URL or null")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("endpoint must be an HTTPS URL without credentials")
    return settings


def generation_command(settings, directory):
    command = [sys.executable, "-u", str(ROOT / "generate_narration.py"), "--output-dir", str(directory)]
    for field, flag in OPTION_FLAGS.items():
        if field == "mastering" and not settings.get("normalize", True):
            command.extend([flag, "off"])
        elif settings.get(field) is not None:
            command.extend([flag, str(settings[field])])
    if not settings.get("normalize", True):
        command.append("--no-normalize")
    return command


def failure_hint(log):
    lowered = log.lower()
    if any(marker in lowered for marker in ("timed out", "timeout", "connection", "could not fetch", "404")):
        return ("The OmniVoice endpoint could not complete this request. Keep Colab running, "
                "confirm its current public URL, and update endpoint in local-test.settings.json "
                "or OMNIVOICE_URL in .env. The full error is in run.log.")
    if "ModuleNotFoundError" in log:
        return 'A dependency is missing. Run python -m pip install -e ".[dev]" in your Python environment.'
    return "Generation did not complete. See run.log in this result folder for the detailed error."


def run(settings_path, *, check=False, open_result=True):
    settings = load_settings(settings_path)
    if check:
        print("Local settings and input paths are valid. No generation or network request was made.")
        print("During generation, narration text and reference audio are sent to your OmniVoice endpoint.")
        return 0
    output_root = ROOT / "output" / "local-tests"
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=output_root))
    (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.setdefault("OMNIVOICE_CACHE_DIR", str(output_root / "_cache"))
    # A null selection in this settings file must override a stale optional .env value.
    for key in ("NARRATION_REFERENCE_START", "NARRATION_QUOTE_REFERENCE_START", "NARRATION_QUOTE_REFERENCE_AUDIO"):
        environment[key] = ""
    environment.pop("AWS_LAMBDA_FUNCTION_NAME", None)
    print("Generating with OmniVoice; text and reference audio are sent to the configured endpoint.")
    print(f"Results: {directory}\n", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            generation_command(settings, directory), cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        message = failure_hint((directory / "run.log").read_text(encoding="utf-8"))
        (directory / "FAILED.txt").write_text(message, encoding="utf-8")
        print(f"\n{message}\nLog: {directory / 'run.log'}")
        if open_result and os.name == "nt":
            os.startfile(directory)
        return code
    outputs = sorted(directory.glob("narration_*.wav"))
    if len(outputs) != 1 or not outputs[0].with_suffix(".json").is_file():
        raise RuntimeError(f"Generation returned without a complete WAV/report pair. Inspect {directory}")
    wav = outputs[0]
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>OmniVoice narration</title>
<style>body{{max-width:760px;margin:60px auto;padding:24px;font:18px system-ui;line-height:1.6}}
audio{{width:100%}}a{{color:#236645}}</style></head><body>
<h1>Your narration is ready</h1><audio controls src="{html.escape(wav.name)}"></audio>
<p><a href="{html.escape(wav.name)}">Download WAV</a> &middot;
<a href="{html.escape(wav.with_suffix('.json').name)}">Generation report</a> &middot;
<a href="run.log">Run log</a></p>
<p>Delivery: {html.escape(settings['delivery'])}. Listen for clear names, natural emphasis,
and consistent voice before using the finished narration.</p></body></html>'''
    player = directory / "index.html"
    detailed_player = wav.with_suffix(".html")
    player.write_text(detailed_player.read_text(encoding="utf-8") if detailed_player.exists() else page, encoding="utf-8")
    print(f"\nReady: {wav}\nPlayer: {player}")
    if open_result and os.name == "nt":
        os.startfile(player)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "local-test.settings.json")
    parser.add_argument("--check", action="store_true", help="Validate local settings and paths without generation")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(args.settings, check=args.check, open_result=not args.no_open)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Local test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())