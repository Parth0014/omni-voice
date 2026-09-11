import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from omni_native import NativeOmniVoice


def test_prompt_reuse_exact_transcript_and_disabled_internal_split(tmp_path, monkeypatch):
    reference = tmp_path / "voice.wav"
    sf.write(reference, np.full(4 * 24000, .1), 24000)
    prompts, requests, seeds = [], [], []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=seeds.append))

    def create(**kwargs):
        prompts.append(kwargs)
        return object()

    def generate(**kwargs):
        requests.append(kwargs)
        return [np.full(24000, .1, dtype=np.float32)]

    model = SimpleNamespace(sampling_rate=24000, create_voice_clone_prompt=create, generate=generate)
    engine = NativeOmniVoice(model=model, revision="test-commit")
    for text in ("First story thought.", "Second story thought."):
        engine.synthesize(text, reference, reference_text="The exact reference words.",
                          preprocess=False, normalize_text=False, seed=88, steps=48, guidance=3.0)
    assert len(prompts) == 1
    assert prompts[0]["preprocess_prompt"] is False
    assert prompts[0]["ref_text"] == "The exact reference words."
    assert requests[0]["audio_chunk_threshold"] == 86400
    assert requests[0]["num_step"] == 48 and requests[0]["guidance_scale"] == 3
    assert requests[0]["postprocess_output"] is False
    assert seeds == [88, 88]
    assert engine.last_metadata["prompt_cache_hit"] is True


def test_manual_reference_is_required_before_model_call():
    engine = NativeOmniVoice(model=SimpleNamespace(sampling_rate=24000), revision="test")
    with pytest.raises(ValueError, match="manual"):
        engine.synthesize("Target words.", "not-read.wav", reference_text="")
