from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from speaker_similarity import score_checkpoints


def test_same_audio_same_score_and_common_windows(tmp_path):
    source = tmp_path / "voice.wav"
    samples = np.sin(np.arange(16000 * 4) * 0.08).astype("float32") * 0.1
    sf.write(source, samples, 16000)
    encoder = SimpleNamespace(embed_utterance=lambda wave: np.array([1.0, float(np.std(wave))]))
    report = score_checkpoints(source, {"raw": source, "spectral": source}, encoder=encoder,
                               preprocess=lambda wave, source_sr: wave)
    assert report["scores"]["raw"]["windows"] == report["scores"]["spectral"]["windows"]
    assert report["window_starts"] == [0]


def test_duration_mismatch_rejected(tmp_path):
    source, shortened = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(source, np.ones(16000 * 4) * 0.1, 16000)
    sf.write(shortened, np.ones(16000 * 3) * 0.1, 16000)
    with pytest.raises(ValueError, match="durations differ"):
        score_checkpoints(source, {"raw": source, "spectral": shortened},
                          encoder=SimpleNamespace(embed_utterance=lambda wave: np.ones(2)),
                          preprocess=lambda wave, source_sr: wave)
