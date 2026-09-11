from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from omni_voice import OmniVoiceClient, validate_settings


def test_native_controls_and_audio_conversion(tmp_path):
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.column_stack([np.full(48000, .1), np.full(48000, .2)]), 48000)
    calls = []

    def submit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(result=lambda timeout: (str(audio), "Done."))

    client = OmniVoiceClient(client=SimpleNamespace(submit=submit))
    samples = client.synthesize("Hello.", audio, reference_text="Reference words.", speed=.97, steps=48)
    assert samples.shape == (24000,)
    assert samples.dtype == np.float32
    assert calls[0]["sp"] == .97
    assert calls[0]["ref_text"] == "Reference words."
    assert calls[0]["po"] is False
    assert calls[0]["ns"] == 48
    assert calls[0]["du"] is None


@pytest.mark.parametrize("settings", [
    {"steps": True}, {"steps": 65}, {"speed": float("nan")}, {"guidance": -1},
    {"instruct": "warm intimate storytelling"}, {"instruct": "male, female"}, {"denoise": "false"},
])
def test_unsupported_controls_fail(settings):
    with pytest.raises(ValueError):
        validate_settings(**settings)


def test_model_error_is_not_silently_retried(tmp_path):
    reference = tmp_path / "ref.wav"
    reference.touch()
    job = SimpleNamespace(result=lambda timeout: (None, "Error: CUDA out of memory"))
    client = OmniVoiceClient(client=SimpleNamespace(submit=lambda **kw: job))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        client.synthesize("Hello.", reference)


def test_timeout_cancels(tmp_path):
    reference = tmp_path / "ref.wav"
    reference.touch()
    cancelled = []

    def result(timeout):
        raise TimeoutError()

    job = SimpleNamespace(result=result, cancel=lambda: cancelled.append(True))
    client = OmniVoiceClient(client=SimpleNamespace(submit=lambda **kw: job))
    with pytest.raises(RuntimeError, match="timed out"):
        client.synthesize("Hello.", reference)
    assert cancelled == [True]


def test_revision_is_in_model_identity():
    a = OmniVoiceClient(client=object(), revision="one")
    b = OmniVoiceClient(client=object(), revision="two")
    assert a.identity != b.identity


def test_64_step_quality_candidate_is_supported():
    validate_settings(steps=64, guidance=2.0)
    validate_settings(steps=64, guidance=3.0)


def test_upgraded_runtime_discovery_and_request_controls(tmp_path, monkeypatch):
    import gradio_client

    reference = tmp_path / "ref.wav"
    sf.write(reference, np.full(4 * 24000, .1), 24000)
    requests = []

    def submit(*args, **kwargs):
        requests.append((args, kwargs))
        return SimpleNamespace(result=lambda timeout: (str(reference), {"seed": 87, "normalized_text": "Twelve gifts."}))

    remote = SimpleNamespace(
        view_api=lambda **kw: {"named_endpoints": {"/narrate": {}, "/transcribe_audio": {}, "/capabilities": {}}},
        predict=lambda **kw: {"protocol": "omni-narration-v1", "identity": {"revision": "actual-commit"},
                              "capabilities": {"seed": True, "normalize_text": True, "transcribe": True}},
        submit=submit,
    )
    monkeypatch.setattr(gradio_client, "Client", lambda *args, **kw: remote)
    client = OmniVoiceClient(endpoint="https://test.invalid")
    client.synthesize("12 gifts.", reference, reference_text="Exact reference words.",
                      normalize_text=True, seed=87, steps=48, guidance=3)
    args, kwargs = requests[0]
    assert kwargs["api_name"] == "/narrate"
    assert args[0] == "12 gifts." and args[2] == "Exact reference words."
    assert args[3]["normalize_text"] is True and args[3]["seed"] == 87
    assert client.identity["remote_model"]["revision"] == "actual-commit"
    assert client.last_metadata["normalized_text"] == "Twelve gifts."
