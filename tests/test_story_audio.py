import numpy as np
import pytest
import soundfile as sf

from story_audio import SAMPLE_RATE, assemble, normalize_delivery, prepare_reference, signal_report


def test_assembly_preserves_breath_and_internal_pause():
    clip = np.r_[np.full(1000, .1), np.zeros(12000), np.full(1000, .0005)].astype("float32")
    plan = [{"pause_after_ms": 300}] * 2
    output, timeline = assemble([clip, clip], plan)
    assert np.array_equal(output[:len(clip)], clip)
    assert np.array_equal(output[-len(clip):], clip)
    assert timeline[1]["start_seconds"] == pytest.approx((len(clip) + 7200) / SAMPLE_RATE)


def test_exact_reference_transcript_rejects_automatic_cut(tmp_path):
    reference = tmp_path / "long.wav"
    sf.write(reference, np.full(12 * SAMPLE_RATE, .1), SAMPLE_RATE)
    with pytest.raises(ValueError, match="already cut"):
        prepare_reference(reference, tmp_path / "prepared", transcript="Exact words.")


@pytest.mark.parametrize("audio", [np.zeros(24000), np.full(24000, np.nan), np.ones(10)])
def test_broken_audio_rejected(audio):
    with pytest.raises(ValueError):
        signal_report(audio)


def test_gain_only_preserves_relative_dynamics(tmp_path, monkeypatch):
    monkeypatch.setenv("TMP", str(tmp_path))
    wave = (.1 * np.sin(np.arange(SAMPLE_RATE * 5) * 2 * np.pi * 220 / SAMPLE_RATE)).astype("float32")
    wave[:SAMPLE_RATE] *= .4
    processed, report = normalize_delivery(wave)
    assert report["method"] == "constant_gain"
    assert np.allclose(processed, wave * 10 ** (report["gain_db"] / 20), atol=1e-7)
