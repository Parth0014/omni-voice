"""Create a short, volume-matched listening comparison before rendering a story."""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from generate_narration import BASE_DIR, build_runtime_config, load_runtime_environment, run_pipeline
from worker_document import extract_worker_blocks

VARIANTS = (
    ("natural-32-cfg2", "natural", 32, 2.0),
    ("warm-32-cfg2", "warm", 32, 2.0),
    ("warm-48-cfg2", "warm", 48, 2.0),
    ("warm-48-cfg3", "warm", 48, 3.0),
    ("warm-64-cfg2", "warm", 64, 2.0),
    ("warm-64-cfg3", "warm", 64, 3.0),
)


def excerpt_html(source):
    blocks = extract_worker_blocks(Path(source).read_text(encoding="utf-8-sig"))
    paragraphs = [i for i, b in enumerate(blocks) if b["type"] == "paragraph"
                  and not b["text"].lower().startswith(("trigger warning", "content warning"))]
    quotes = [i for i, b in enumerate(blocks) if b["type"] == "quote"]
    # The same action, quotation and reflective ending appear in every candidate.
    selected = set(paragraphs[:1] + quotes[:1] + paragraphs[-1:])
    if not selected:
        selected = set(range(min(3, len(blocks))))
    parts = []
    for index in sorted(selected):
        block = blocks[index]
        tag = "blockquote" if block["type"] == "quote" else "p"
        parts.append(f"<{tag}>{html.escape(block['text'])}</{tag}>")
    if not parts:
        raise ValueError("No spoken content is available for audition")
    return "\n".join(parts)


def create_audition(source, reference, *, output_dir=None, reference_text="", endpoint=None,
                    pronunciation=None, take=0):
    root = Path(output_dir or BASE_DIR / "output" / "auditions").resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="story-", dir=root))
    excerpt = directory / "excerpt.html"
    excerpt.write_text(excerpt_html(source), encoding="utf-8")
    results = []
    for name, delivery, steps, guidance in VARIANTS:
        result = {"name": name, "delivery": delivery, "steps": steps, "guidance": guidance}
        try:
            audio = Path(run_pipeline(
                excerpt, reference, output_dir=directory / name, delivery=delivery,
                steps=steps, guidance=guidance, mastering="gain", reference_text=reference_text, endpoint=endpoint,
                pronunciation=pronunciation, take=take,
            ))
            report = json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))
            result.update(audio_path=str(audio), report=report, status="complete")
        except Exception as exc:
            result.update(status="failed", error=str(exc))
        results.append(result)
        (directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    complete = [r for r in results if r["status"] == "complete"]
    if not complete:
        raise RuntimeError(f"All auditions failed; inspect {directory / 'results.json'}")
    # Attenuate to the quietest candidate. No limiter or compression changes delivery.
    common_lufs = min(r["report"]["finishing"]["estimated_output_lufs"] for r in complete)
    cards = []
    for result in results:
        label = html.escape(result["name"])
        if result["status"] != "complete":
            cards.append(f"<article><h2>{label}</h2><p>{html.escape(result['error'])}</p></article>")
            continue
        path = Path(result["audio_path"])
        audio, rate = sf.read(path, dtype="float32")
        gain = common_lufs - result["report"]["finishing"]["estimated_output_lufs"]
        listen = path.with_name("listen.wav")
        sf.write(listen, np.asarray(audio * 10 ** (gain / 20), dtype=np.float32), rate, subtype="PCM_16")
        url = html.escape(listen.relative_to(directory).as_posix())
        full = ["python", "generate_narration.py", "--html", str(Path(source).resolve()),
                "--reference", str(Path(reference).resolve()), "--delivery", result["delivery"],
                "--steps", str(result["steps"]), "--guidance", str(result["guidance"])]
        command = shlex.join(full) if os.name != "nt" else __import__("subprocess").list2cmdline(full)
        cards.append(f'<article><h2>{label}</h2><audio controls src="{url}"></audio>'
                     f'<p><a href="{url}" download>Download audition</a></p>'
                     '<label>Your listening notes<textarea placeholder="Names, consonants, expression, warmth, voice consistency"></textarea></label>'
                     f'<details><summary>Full-generation starting command</summary><pre>{html.escape(command)}</pre>'
                     '<p>Also carry over your endpoint and any reference transcript or pronunciation file.</p></details></article>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Choose your story delivery</title>
<style>body{{max-width:1050px;margin:40px auto;padding:24px;background:#f5f1e8;color:#253b36;font:17px/1.6 system-ui}}article{{background:white;border-radius:16px;padding:24px;margin:20px 0}}audio,textarea{{width:100%}}textarea{{min-height:90px}}pre{{white-space:pre-wrap}}a{{color:#245c4b}}</style>
<h1>Choose your story delivery</h1><p>The same excerpt and reference voice, played at a matched volume.
Compare 32/48/64 steps and guidance 2/3; none is a promise of more emotion.</p>
<p>Judge pronunciation, natural emphasis, emotional warmth, consistency, and clean joins.
The reference performance is the main style guide. These notes stay in this browser.</p>
{''.join(cards)}
<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{{if(a!==b)b.pause()}})));
document.querySelectorAll('textarea').forEach((a,i)=>{{try{{a.value=localStorage.getItem(location.href+i)||'';a.oninput=()=>localStorage.setItem(location.href+i,a.value)}}catch(e){{}}}});</script></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")
    print(f"Audition player: {directory / 'index.html'}")
    return directory


def main(argv=None):
    environment = load_runtime_environment()
    config = build_runtime_config(BASE_DIR, environment)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", default=config["post_html_file"])
    parser.add_argument("--reference", default=config["narration_reference_audio"])
    parser.add_argument("--reference-text", default=environment.get("OMNIVOICE_REFERENCE_TEXT", ""))
    parser.add_argument("--endpoint", default=config["endpoint"])
    parser.add_argument("--pronunciation", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument("--take", type=int, default=0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    pronunciation = json.loads(args.pronunciation.read_text()) if args.pronunciation else None
    directory = create_audition(args.html, args.reference, reference_text=args.reference_text,
                               endpoint=args.endpoint, pronunciation=pronunciation,
                               output_dir=args.output_dir, take=args.take)
    if os.name == "nt" and not args.no_open:
        os.startfile(directory / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
