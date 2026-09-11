from types import SimpleNamespace

import numpy as np
import soundfile as sf

from colab_runtime import handlers


def test_extended_api_preserves_exact_text_and_exposes_capabilities(tmp_path):
    requests = []

    def synthesize(text, reference, **settings):
        requests.append((text, reference, settings))
        return np.full(24000, .1, dtype=np.float32)

    engine = SimpleNamespace(synthesize=synthesize, sample_rate=24000, last_metadata={"seed": 4},
                             identity={"revision": "pinned"}, capabilities={"normalize_text": True},
                             transcribe=lambda audio: "Target words.")
    narrate, transcribe, capabilities = handlers(engine, tmp_path)
    path, metadata = narrate("Target words.", "reference.wav", "Reference words.", {"normalize_text": True})
    assert requests[0][0] == "Target words."
    assert requests[0][2]["reference_text"] == "Reference words."
    assert sf.info(path).samplerate == 24000
    assert metadata == {"seed": 4}
    assert transcribe(path) == "Target words."
    assert capabilities()["protocol"] == "omni-narration-v1"
