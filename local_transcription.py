"""Optional CPU transcription checks for speech engines without output ASR."""

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class LocalTranscriptionEngine:
    def __init__(self, engine, *, model_path, model=None):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4,
                                 local_files_only=True)
        self.engine = engine
        self.model = model
        self.sample_rate = engine.sample_rate
        self.capabilities = {**engine.capabilities, "transcribe": True}
        self.identity = {**engine.identity, "local_asr": {"model": Path(model_path).name,
                         "decoder": "faster-whisper-small.en-int8-v1", "beam_size": 5}}

    @property
    def last_metadata(self):
        return self.engine.last_metadata

    def synthesize(self, *args, **kwargs):
        return self.engine.synthesize(*args, **kwargs)

    def transcribe(self, audio):
        # Faster Whisper expects arrays at 16 kHz; never pass a 24 kHz array as-is.
        import math

        divisor = math.gcd(self.sample_rate, 16000)
        samples = resample_poly(np.asarray(audio, dtype=np.float32),
                                16000 // divisor, self.sample_rate // divisor)
        segments, _ = self.model.transcribe(samples, language="en", beam_size=5,
                                            condition_on_previous_text=False, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()
