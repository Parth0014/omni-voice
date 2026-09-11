import json
from pathlib import Path

import numpy as np
import soundfile as sf

import story_audition


def test_excerpt_keeps_story_quote_and_closing_without_warning(tmp_path):
    source = tmp_path / "source.html"
    source.write_text('<p>Trigger warning: illness.</p><p>She opened the door.</p>'
                      '<blockquote>Thank you.</blockquote><p>She felt at home.</p>')
    excerpt = story_audition.excerpt_html(source)
    assert "Trigger warning" not in excerpt
    assert "She opened the door." in excerpt and "Thank you." in excerpt and "She felt at home." in excerpt


def test_comparison_uses_same_reference_and_transcript_and_writes_player(tmp_path, monkeypatch):
    source = tmp_path / "source.html"
    source.write_text('<p>A story.</p><blockquote>A quote.</blockquote>')
    reference = tmp_path / "ref.wav"
    reference.touch()
    calls = []

    def generate(excerpt, ref, **settings):
        calls.append((Path(excerpt).read_text(), ref, settings))
        output = Path(settings["output_dir"])
        output.mkdir()
        wav = output / "narration_0001.wav"
        sf.write(wav, np.full(24000, .1), 24000)
        wav.with_suffix(".json").write_text(json.dumps({"finishing": {"estimated_output_lufs": -19}}))
        return str(wav)

    monkeypatch.setattr(story_audition, "run_pipeline", generate)
    directory = story_audition.create_audition(source, reference, output_dir=tmp_path / "auditions",
                                              reference_text="Exact words.")
    assert len(calls) == 6
    assert {(c[2]["steps"], c[2]["guidance"]) for c in calls} >= {(64, 2.0), (64, 3.0)}
    assert len({c[0] for c in calls}) == 1
    assert all(c[1] == reference and c[2]["reference_text"] == "Exact words." for c in calls)
    assert (directory / "index.html").is_file()
    assert len(list(directory.glob("*/listen.wav"))) == 6
