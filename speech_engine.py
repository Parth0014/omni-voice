"""Application-independent speech contract and explicit backend selection."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

import numpy as np


class SpeechEngine(Protocol):
    identity: dict
    capabilities: dict
    sample_rate: int

    def synthesize(self, text: str, reference_path, **settings) -> np.ndarray: ...


@lru_cache(maxsize=1)
def _native(model, revision, device):
    from omni_native import NativeOmniVoice

    return NativeOmniVoice(model_id=model, revision=revision, device=device)


def create_engine(name=None, *, endpoint=None, timeout=None, revision=None):
    name = name or os.environ.get("NARRATION_ENGINE", "gradio")
    if name == "gradio":
        from omni_voice import OmniVoiceClient

        return OmniVoiceClient(endpoint, timeout=timeout, revision=revision)
    if name == "native":
        return _native(os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice"),
                       revision or os.environ.get("OMNIVOICE_MODEL_REVISION"),
                       os.environ.get("OMNIVOICE_DEVICE", "cuda:0"))
    raise ValueError("Unknown speech engine; select gradio or native")
