"""Run the extended narration API on a Colab GPU; independent of Ghost and AWS."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf


def handlers(engine, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def narrate(text, ref_aud, ref_text, options):
        options = dict(options or {})
        allowed = {"instruct", "language", "steps", "guidance", "speed", "denoise", "preprocess",
                   "postprocess", "normalize_text", "seed"}
        if set(options) - allowed:
            raise ValueError("Unsupported synthesis options")
        audio = engine.synthesize(text, ref_aud, reference_text=ref_text, **options)
        descriptor, path = tempfile.mkstemp(prefix="take-", suffix=".wav", dir=output_dir)
        os.close(descriptor)
        sf.write(path, audio, engine.sample_rate, subtype="FLOAT")
        for previous in output_dir.glob("take-*.wav"):
            if previous.stat().st_mtime < time.time() - 3600:
                previous.unlink(missing_ok=True)
        return path, dict(engine.last_metadata)

    def transcribe_audio(path):
        audio, rate = sf.read(path, dtype="float32")
        if rate != engine.sample_rate or audio.ndim != 1:
            raise ValueError("Expected mono 24 kHz narration audio")
        return engine.transcribe(audio)

    def capabilities():
        return {"protocol": "omni-narration-v1", "capabilities": engine.capabilities, "identity": engine.identity}

    return narrate, transcribe_audio, capabilities


def build_demo(engine):
    import gradio as gr

    narrate, transcribe_audio, capabilities = handlers(engine, Path(tempfile.gettempdir()) / "omni-narration-api")
    with gr.Blocks() as demo:
        gr.Markdown("# OmniVoice narration runtime\nUse the local story/audition launcher with this share URL.")
        text = gr.Textbox(label="Text")
        reference = gr.Audio(type="filepath", label="Reference: 3–10 seconds")
        transcript = gr.Textbox(label="Exact reference transcript")
        options = gr.JSON(value={"steps": 32, "guidance": 2.0, "normalize_text": True, "seed": 1234})
        audio = gr.Audio(type="filepath")
        metadata = gr.JSON()
        gr.Button("Generate").click(narrate, [text, reference, transcript, options], [audio, metadata], api_name="narrate")
        gr.Button("Check transcript").click(transcribe_audio, [audio], [gr.Textbox()], api_name="transcribe_audio")
        gr.Button("Runtime details").click(capabilities, [], [gr.JSON()], api_name="capabilities")
    return demo.queue(default_concurrency_limit=1, max_size=16)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="HF model commit; resolves and prints current commit if omitted")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    from huggingface_hub import model_info

    from omni_native import NativeOmniVoice

    revision = args.revision or model_info("k2-fsa/OmniVoice").sha
    print(f"Pinned model revision: {revision}", flush=True)
    engine = NativeOmniVoice(revision=revision)
    build_demo(engine).launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
