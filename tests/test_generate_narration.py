import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import generate_narration as narration


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.html"
    source.write_text('<p>She opened the door.</p><blockquote>Thank you for being here.</blockquote><p>She felt at home.</p>')
    reference = tmp_path / "ref.wav"
    wave = (.12 * np.sin(np.arange(4 * 24000) * 2 * np.pi * 200 / 24000)).astype("float32")
    sf.write(reference, wave, 24000)
    quote = tmp_path / "quote.wav"
    sf.write(quote, wave * .8, 24000)
    registry = tmp_path / "references.json"
    registry.write_text(json.dumps({narration.file_hash(reference): {"text": "Exact reference words."},
                                    narration.file_hash(quote): {"text": "Exact second reference words."}}))
    monkeypatch.setenv("OMNIVOICE_REFERENCE_MANIFEST", str(registry))
    calls = []

    def synthesize(text, reference, **settings):
        calls.append((text, reference, settings))
        return wave[:24000]

    client = SimpleNamespace(identity={"test": "omni"}, synthesize=synthesize)
    monkeypatch.setattr(narration, "create_engine", lambda *args, **kwargs: client)
    monkeypatch.delenv("OMNIVOICE_CACHE_DIR", raising=False)
    return source, reference, quote, calls, tmp_path / "output", client


@pytest.mark.parametrize("mode,count", [("preserve", 3), ("exclude", 2), ("two_voice", 3)])
def test_modes_native_pace_reports_and_cache(fixture, mode, count):
    source, ref, quote, calls, output, _ = fixture
    kw = dict(quote_mode=mode, quote_reference_audio=quote, output_dir=output, normalize=False)
    first = narration.run_pipeline(source, ref, **kw)
    report = json.loads(Path(first).with_suffix(".json").read_text())
    assert len(calls) == count
    assert report["model"] == {"test": "omni"}
    assert all(c[2]["postprocess"] is False for c in calls)
    assert calls[-1][2]["speed"] == .97
    if mode == "exclude":
        assert all("Thank you" not in c[0] for c in calls)
    if mode == "two_voice":
        assert calls[0][1] != calls[1][1]
    assert Path(first).with_suffix(".html").is_file()
    second = narration.run_pipeline(source, ref, **kw)
    assert second != first
    assert len(calls) == count
    assert all(c["cache_hit"] for c in json.loads(Path(second).with_suffix(".json").read_text())["chunks"])


def test_selective_retake_only_regenerates_selected_chunk(fixture):
    source, ref, _, calls, output, _ = fixture
    narration.run_pipeline(source, ref, output_dir=output, normalize=False)
    result = narration.run_pipeline(source, ref, output_dir=output, normalize=False, take=1, retake_chunks=[2])
    assert len(calls) == 4
    assert "Thank you" in calls[-1][0]
    report = json.loads(Path(result).with_suffix(".json").read_text())
    assert [c["cache_hit"] for c in report["chunks"]] == [True, False, True]


def test_corrupt_cache_is_regenerated(fixture):
    source, ref, _, calls, output, _ = fixture
    narration.run_pipeline(source, ref, output_dir=output, normalize=False)
    next((output / "_cache").glob("*.wav")).write_bytes(b"broken")
    narration.run_pipeline(source, ref, output_dir=output, normalize=False)
    assert len(calls) == 4


def test_failed_synthesis_publishes_no_final_audio(fixture):
    source, ref, _, _, output, client = fixture
    client.synthesize = lambda *a, **k: np.zeros(24000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="twice"):
        narration.run_pipeline(source, ref, output_dir=output, normalize=False)
    assert not list(output.glob("narration_*.wav"))


def test_plan_only_does_not_need_reference_or_remote(fixture, monkeypatch):
    source, _, _, _, output, _ = fixture
    monkeypatch.setattr(narration, "create_engine", lambda *args, **kwargs: pytest.fail("Network attempted"))
    result = narration.run_pipeline(source, "missing.wav", output_dir=output, plan_only=True)
    assert len(json.loads(Path(result).read_text())["chunks"]) == 3
    assert not (output / "_cache").exists()


@pytest.mark.parametrize("settings", [{"speed": True}, {"steps": 100}, {"delivery": "fake"},
                                      {"instruct": "be emotional"}, {"max_chunks": 0}])
def test_invalid_settings_before_inference(fixture, settings):
    source, ref, _, calls, output, _ = fixture
    with pytest.raises(ValueError):
        narration.run_pipeline(source, ref, output_dir=output, **settings)
    assert not calls


def test_required_verification_fails_before_synthesis_on_legacy_demo(fixture):
    source, ref, _, calls, output, _ = fixture
    with pytest.raises(ValueError, match="Required transcript"):
        narration.run_pipeline(source, ref, output_dir=output, verify_text="required")
    assert not calls


def test_reference_leak_retries_without_caching_or_delivering(fixture):
    source, ref, _, calls, output, client = fixture
    client.capabilities = {"transcribe": True, "seed": True}
    reference_text = "The most beautiful moments in life rarely ask for anything."
    client.transcribe = lambda audio: reference_text
    with pytest.raises(RuntimeError, match="twice"):
        narration.run_pipeline(source, ref, output_dir=output, reference_text=reference_text,
                               verify_text="required", normalize=False)
    assert len(calls) == 2
    assert calls[0][2]["seed"] != calls[1][2]["seed"]
    assert not list((output / "_cache").glob("*.wav"))
    assert not list(output.glob("narration_*.wav"))


def test_verified_raw_cache_survives_mastering_changes(fixture, monkeypatch):
    source, ref, _, calls, output, client = fixture
    client.capabilities = {"transcribe": True, "seed": True}
    client.transcribe = lambda audio: calls[-1][0]
    first = narration.run_pipeline(source, ref, output_dir=output, verify_text="required", normalize=False)
    import narration_mastering

    mastered = []

    def master(audio, rate, profile, **kwargs):
        mastered.append(profile)
        return audio, {"after": {"input_i": -19.0}}

    monkeypatch.setattr(narration_mastering, "master_audio", master)
    second = narration.run_pipeline(source, ref, output_dir=output, verify_text="required", mastering="warm_story")
    assert first != second and len(calls) == 3 and mastered == ["warm_story"]
    report = json.loads(Path(second).with_suffix(".json").read_text())
    assert all(chunk["cache_hit"] for chunk in report["chunks"])
    assert all(chunk["generation"]["verification"]["status"] == "pass" for chunk in report["chunks"])
