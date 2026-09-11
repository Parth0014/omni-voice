from types import SimpleNamespace

import numpy as np

from local_transcription import LocalTranscriptionEngine


def test_local_asr_resamples_and_does_not_prompt_with_expected_words():
    requests = []

    def transcribe(audio, **kwargs):
        requests.append((audio, kwargs))
        return iter([SimpleNamespace(text=" Actual spoken words. ")]), None

    remote = SimpleNamespace(sample_rate=24000, identity={"endpoint": "test"},
                              capabilities={"seed": False}, last_metadata={"text": "original"})
    wrapper = LocalTranscriptionEngine(remote, model_path="snapshot", model=SimpleNamespace(transcribe=transcribe))
    assert wrapper.transcribe(np.full(24000, .1)) == "Actual spoken words."
    assert requests[0][0].shape == (16000,)
    assert "initial_prompt" not in requests[0][1]
    assert wrapper.capabilities["transcribe"] is True
    assert wrapper.capabilities["seed"] is False
    assert wrapper.last_metadata == remote.last_metadata
