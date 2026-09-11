"""Validated OmniVoice Gradio client; no local TTS model or Torch dependency."""

from __future__ import annotations

import math
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_URL = "https://755352763c878e5bc1.gradio.live"
ATTRIBUTES = {
    "male", "female", "child", "teenager", "young adult", "middle-aged", "elderly",
    "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch", "whisper",
    *{f"{accent} accent" for accent in (
        "american", "british", "australian", "canadian", "indian", "chinese", "korean",
        "japanese", "portuguese", "russian")},
}
PARAMETERS = {"text", "lang", "ref_aud", "ref_text", "instruct", "ns", "gs", "dn", "sp", "du", "pp", "po"}


def validate_settings(*, steps=32, guidance=2.0, speed=1.0, instruct="", denoise=True,
                      preprocess=True, postprocess=False, **kwargs):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 32 <= steps <= 64:
        raise ValueError("steps must be an integer between 32 and 64")
    for name, value, low, high in (("guidance", guidance, 0, 4), ("speed", speed, .5, 1.5)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if not isinstance(instruct, str):
        raise ValueError("instruct must be a string")
    selected = [part.strip().lower() for part in instruct.split(",") if part.strip()]
    if any(part not in ATTRIBUTES for part in selected):
        raise ValueError("Use only documented English voice attributes; free-form emotion directions are unsupported")
    groups = [
        {"male", "female"}, {"child", "teenager", "young adult", "middle-aged", "elderly"},
        {a for a in ATTRIBUTES if a.endswith("pitch")}, {a for a in ATTRIBUTES if a.endswith("accent")},
    ]
    if any(len(set(selected) & group) > 1 for group in groups):
        raise ValueError("Use only one instruction per voice attribute category")
    if not all(isinstance(value, bool) for value in (denoise, preprocess, postprocess)):
        raise ValueError("Processing flags must be booleans")


class OmniVoiceClient:
    sample_rate = 24000
    capabilities = {"seed": False, "normalize_text": False, "disable_internal_chunking": False,
                    "transcribe": False, "reference_prompt_cache": False}

    def __init__(self, endpoint=None, *, timeout=None, revision=None, client=None):
        endpoint = (endpoint or os.environ.get("OMNIVOICE_URL") or DEFAULT_URL).rstrip("/")
        parsed = urlparse(endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError("OmniVoice endpoint must be an HTTPS URL without credentials, query or fragment")
        self.timeout = float(timeout if timeout is not None else os.environ.get("OMNIVOICE_TIMEOUT_SECONDS", "180"))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("OmniVoice timeout must be finite and positive")
        try:
            client_version = version("gradio-client")
        except PackageNotFoundError:
            client_version = "unknown"
        self.identity = {"backend": "omni_voice", "endpoint": endpoint, "api_name": "/_clone_fn",
                         "adapter_version": "story-v2", "gradio_client_version": client_version,
                         "revision": revision or os.environ.get("OMNIVOICE_MODEL_REVISION", "colab-test-v1"),
                         "seed_supported": False, "free_form_emotion_supported": False}
        self.extended = False
        self.last_metadata = {}
        self.capabilities = dict(type(self).capabilities)
        if client is None:
            from gradio_client import Client

            try:
                client = Client(endpoint, verbose=False, httpx_kwargs={"timeout": self.timeout})
                schema = client.view_api(return_format="dict", print_info=False)
                endpoints = schema["named_endpoints"]
                self.extended = "/narrate" in endpoints and "/transcribe_audio" in endpoints
                if self.extended:
                    metadata = client.predict(api_name="/capabilities")
                    if not isinstance(metadata, dict) or metadata.get("protocol") != "omni-narration-v1":
                        raise ValueError("Unrecognized extended narration API")
                    self.capabilities.update(metadata["capabilities"])
                    self.identity.update(remote_model=metadata["identity"], api_name="/narrate",
                                         seed_supported=self.capabilities.get("seed", False))
                else:
                    parameters = endpoints["/_clone_fn"]["parameters"]
                    if not PARAMETERS <= {p["parameter_name"] for p in parameters}:
                        raise ValueError("The Colab clone API has incompatible parameters")
            except Exception as exc:
                raise RuntimeError(f"Cannot connect to the OmniVoice clone API; check Colab and its URL: {exc}") from exc
        self.client = client

    def synthesize(self, text, reference_path, *, reference_text="", instruct="", language="English",
                   steps=32, guidance=2.0, speed=1.0, denoise=True, preprocess=True, postprocess=False,
                   normalize_text=False, seed=None):
        validate_settings(steps=steps, guidance=guidance, speed=speed, instruct=instruct,
                          denoise=denoise, preprocess=preprocess, postprocess=postprocess)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text to synthesize cannot be empty")
        if not isinstance(reference_text, str):
            raise ValueError("Reference transcript must be text")
        reference_path = Path(reference_path).resolve(strict=True)
        from gradio_client import handle_file

        if (normalize_text or seed is not None) and not self.extended:
            raise ValueError("This demo cannot normalize text or set seeds; use the supplied Colab wrapper")
        if self.extended:
            options = dict(instruct=instruct, language=language, steps=steps, guidance=guidance, speed=speed,
                           denoise=denoise, preprocess=preprocess, postprocess=postprocess,
                           normalize_text=normalize_text, seed=1234 if seed is None else seed)
            job = self.client.submit(text, handle_file(str(reference_path)), reference_text, options,
                                     api_name="/narrate")
        else:
            job = self.client.submit(
                text=text, lang=language, ref_aud=handle_file(str(reference_path)), ref_text=reference_text,
                instruct=instruct, ns=steps, gs=guidance, dn=denoise, sp=speed, du=None,
                pp=preprocess, po=postprocess, api_name="/_clone_fn",
            )
        try:
            result = job.result(timeout=self.timeout)
        except TimeoutError as exc:
            job.cancel()
            raise RuntimeError("OmniVoice inference timed out; check the Colab session and public URL") from exc
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("OmniVoice returned an unexpected result; expected audio and status")
        path, status = result
        self.last_metadata = status if isinstance(status, dict) else {"normalized_text": text}
        if not isinstance(path, (str, os.PathLike)) or not Path(path).is_file():
            raise RuntimeError(f"OmniVoice returned no audio: {status}")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate <= 0 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("OmniVoice returned invalid or empty audio")
        samples = samples.mean(axis=1)
        if rate != self.sample_rate:
            divisor = math.gcd(rate, self.sample_rate)
            samples = resample_poly(samples, self.sample_rate // divisor, rate // divisor)
        return np.asarray(samples, dtype=np.float32)

    def transcribe(self, audio):
        if not self.capabilities["transcribe"]:
            raise RuntimeError("The current Gradio demo has no output transcription API")
        import tempfile

        from gradio_client import handle_file

        with tempfile.TemporaryDirectory(prefix="omni-asr-") as directory:
            path = Path(directory) / "speech.wav"
            sf.write(path, audio, self.sample_rate, subtype="FLOAT")
            job = self.client.submit(handle_file(str(path)), api_name="/transcribe_audio")
            try:
                return job.result(timeout=self.timeout)
            except TimeoutError as exc:
                job.cancel()
                raise RuntimeError("Output transcription timed out") from exc
