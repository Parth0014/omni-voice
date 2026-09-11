import json

import local_test


def setup_runner(tmp_path, monkeypatch, generator):
    monkeypatch.setattr(local_test, "ROOT", tmp_path)
    (tmp_path / "sample.html").write_text("<p>Test narration</p>")
    (tmp_path / "voice.wav").write_bytes(b"reference fixture")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"html": "sample.html", "reference": "voice.wav", "delivery": "warm"}))
    (tmp_path / "generate_narration.py").write_text(generator)
    return settings


def test_success_creates_local_result_and_player(tmp_path, monkeypatch):
    settings = setup_runner(tmp_path, monkeypatch, '''import sys
from pathlib import Path
assert sys.argv[sys.argv.index('--delivery') + 1] == 'warm'\nassert sys.argv[sys.argv.index('--mastering') + 1] == 'auto'
output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
(output / 'narration_0001.wav').write_bytes(b'test output')
(output / 'narration_0001.json').write_text('{}')
print('Test generation completed')
''')
    assert local_test.run(settings, open_result=False) == 0
    result = next((tmp_path / "output/local-tests").iterdir())
    assert (result / "index.html").is_file()
    assert "Test generation completed" in (result / "run.log").read_text()


def test_endpoint_failure_retains_log_and_explanation(tmp_path, monkeypatch):
    settings = setup_runner(tmp_path, monkeypatch, '''import sys
print('OmniVoice connection timed out')
sys.exit(1)
''')
    assert local_test.run(settings, open_result=False) == 1
    result = next((tmp_path / "output/local-tests").iterdir())
    assert "Colab" in (result / "FAILED.txt").read_text()
    assert not (result / "index.html").exists()


def test_check_does_not_launch_generation_or_create_output(tmp_path, monkeypatch):
    settings = setup_runner(tmp_path, monkeypatch, "raise RuntimeError('Must not execute')")
    assert local_test.run(settings, check=True, open_result=False) == 0
    assert not (tmp_path / "output").exists()
