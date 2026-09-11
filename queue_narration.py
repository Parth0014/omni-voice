"""Generate one narration with several configurations for side-by-side listening."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from generate_narration import build_runtime_config, load_runtime_environment, run_pipeline, write_json


VARIANT_FIELDS = {
    "delivery", "speed", "steps", "guidance", "instruct", "denoise", "mastering",
    "spectral_matching", "normalize_text", "verify_text", "take", "max_chunks",
    "pronunciation", "directions", "reference_start_seconds", "quote_reference_start_seconds",
}


def identifier(value: object, used: set[str]) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value).strip()).strip("-_").lower() or "variant"
    text = text[:60].rstrip("-_")
    candidate = text
    counter = 2
    while candidate in used:
        candidate = f"{text[:54].rstrip('-_')}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def load_variants(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("variants")
    if not isinstance(payload, list) or not payload:
        raise ValueError('Variants JSON must contain a non-empty array or a {"variants": [...]} object.')
    variants = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Variant {index} must be a JSON object.")
        unknown = set(item) - VARIANT_FIELDS - {"id", "name"}
        if unknown:
            raise ValueError(f"Variant {index} has unsupported fields: {', '.join(sorted(unknown))}")
        variant = dict(item)
        variant["name"] = variant.get("name") or variant.get("id") or f"variant-{index:02d}"
        variants.append(variant)
    return variants


def read_json_value(value, label):
    if value is None or isinstance(value, dict):
        return value
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def report_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def reference_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.wav"))
        if not files:
            raise ValueError(f"No WAV references found in {path}")
        return files
    raise ValueError(f"Reference file or folder not found: {path}")


def main(argv=None):
    environment = load_runtime_environment()
    defaults = build_runtime_config(Path(__file__).resolve().parent, environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True, help="JSON array of configuration overrides")
    parser.add_argument("--html", default=defaults["post_html_file"])
    parser.add_argument("--reference", default=defaults["narration_reference_audio"])
    parser.add_argument("--reference-text", default=environment.get("OMNIVOICE_REFERENCE_TEXT", ""))
    parser.add_argument("--output-dir", type=Path, default=Path("output/narration-auditions"))
    parser.add_argument("--quote-reference")
    parser.add_argument("--quote-mode", choices=("preserve", "exclude", "two_voice"), default=defaults["quote_mode"])
    parser.add_argument("--quote-reference-text", default=environment.get("OMNIVOICE_QUOTE_REFERENCE_TEXT", ""))
    parser.add_argument("--endpoint", default=defaults["endpoint"])
    parser.add_argument("--engine", choices=("gradio", "native"), default=environment.get("NARRATION_ENGINE", "gradio"))
    parser.add_argument("--remote-timeout", type=float, default=float(environment.get("OMNIVOICE_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--model-revision", default=environment.get("OMNIVOICE_MODEL_REVISION"))
    parser.add_argument("--verify-text", choices=("auto", "required", "off"), default=environment.get("NARRATION_VERIFY_TEXT", "auto"))
    parser.add_argument("--allow-auto-reference", action="store_true")
    parser.add_argument("--reference-manifest")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    html_file = Path(args.html).resolve()
    reference = Path(args.reference).resolve()
    references = reference_files(reference)
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    variants = load_variants(args.variants)
    index = {"source": str(html_file), "references": [str(path) for path in references], "variants": []}
    shared = {
        "post_html_file": str(html_file),
        "quote_reference_audio": str(Path(args.quote_reference).resolve()) if args.quote_reference else None,
        "quote_mode": args.quote_mode, "reference_text": args.reference_text,
        "quote_reference_text": args.quote_reference_text, "endpoint": args.endpoint,
        "engine": args.engine, "remote_timeout": args.remote_timeout,
        "model_revision": args.model_revision, "verify_text": args.verify_text,
        "allow_auto_reference": args.allow_auto_reference, "reference_manifest": args.reference_manifest,
    }

    total = len(references) * len(variants)
    completed = 0
    for reference_number, reference_path in enumerate(references, 1):
        reference_id = identifier(reference_path.stem, set())
        used = set()
        for variant_number, variant in enumerate(variants, 1):
            name = variant["name"]
            variant_id = identifier(name, used)
            config = {**shared, "narration_reference_audio": str(reference_path),
                      **{key: value for key, value in variant.items() if key not in {"id", "name"}}}
            if len(references) > 1 and not args.reference_text:
                config["allow_auto_reference"] = True
            for key in ("pronunciation", "directions"):
                config[key] = read_json_value(config.get(key), key)
            destination = output_root / reference_id / variant_id
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "configuration.json", config)
            print(f"[{completed + 1}/{total}] {reference_path.name} / {name} -> {destination}", flush=True)
            try:
                wav = run_pipeline(output_dir=destination, **config)
                report = Path(wav).with_suffix(".json")
                index["variants"].append({
                    "reference": reference_path.name, "reference_id": reference_id,
                    "id": variant_id, "name": name, "state": "complete",
                    "audio": report_path(Path(wav), root),
                    "report": report_path(report, root),
                    "player": report_path(report.with_suffix(".html"), root),
                    "configuration_sha256": hashlib.sha256(
                        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest(),
                })
            except Exception as exc:
                index["variants"].append({"reference": reference_path.name, "reference_id": reference_id,
                                           "id": variant_id, "name": name, "state": "failed", "error": str(exc)})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            completed += 1
            write_json(output_root / "index.json", index)

    write_json(output_root / "index.json", index)
    failed = [item for item in index["variants"] if item["state"] == "failed"]
    print(f"Finished: {len(index['variants']) - len(failed)} complete, {len(failed)} failed")
    print(f"Compare using: {output_root / 'index.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())