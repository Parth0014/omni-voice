"""GPU-only OmniVoice adapter, reusable by Colab and the AWS SQS worker."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from importlib.metadata import version
from pathlib import Path
from threading import RLock

import numpy as np
import soundfile as sf

from omni_voice import validate_settings


class NativeOmniVoice:
    sample_rate = 24000
    capabilities = {"seed": True, "normalize_text": True, "disable_internal_chunking": True,
                    "transcribe": True, "reference_prompt_cache": True}

    def __init__(self, model_id="k2-fsa/OmniVoice", revision=None, device="cuda:0", *, model=None):
        self._lock = RLock()
        self._prompts = OrderedDict()
        self.last_metadata = {}
        if model is None:
            import torch
            from huggingface_hub import snapshot_download
            from omnivoice import OmniVoice

            if not device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("Native OmniVoice requires an NVIDIA GPU; use gradio for remote inference")
            if not revision:
                raise ValueError("Pin OMNIVOICE_MODEL_REVISION to an actual Hugging Face commit for native generation")
            # Upstream's resolver does not reliably forward HF revision. Resolve
            # a pinned snapshot explicitly and load the concrete local directory.
            location = snapshot_download(model_id, revision=revision)
            model = OmniVoice.from_pretrained(location, device_map=device, dtype=torch.float16, load_asr=False)
            revision = Path(location).name
            package = version("omnivoice")
            device_name = torch.cuda.get_device_name(device)
        else:
            package, device_name = "injected-test-model", device
        self.model = model
        self.sample_rate = int(model.sampling_rate)
        if self.sample_rate != 24000:
            raise ValueError("The narration engine contract requires a 24 kHz OmniVoice checkpoint")
        self.identity = {"backend": "omni_voice_native", "model": model_id, "revision": revision,
                         "omnivoice_version": package, "device": device_name, "adapter_version": "native-v1",
                         "audio_chunk_threshold": 86400.0, "seed_supported": True,
                         "tokenizer_revision": getattr(getattr(getattr(model, "audio_tokenizer", None), "config", None), "_commit_hash", None)}

    def synthesize(self, text, reference_path, *, reference_text="", instruct="", language="English",
                   steps=32, guidance=2.0, speed=1.0, denoise=True, preprocess=True, postprocess=False,
                   normalize_text=True, seed=1234):
        validate_settings(steps=steps, guidance=guidance, speed=speed, instruct=instruct,
                          denoise=denoise, preprocess=preprocess, postprocess=postprocess)
        if not reference_text.strip():
            raise ValueError("Native narration requires an exact manual reference transcript")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2147483647]")
        if not isinstance(normalize_text, bool):
            raise ValueError("normalize_text must be a boolean")
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            raise ValueError("Send one controlled nonempty chunk of at most 600 characters")
        reference = Path(reference_path)
        info = sf.info(reference)
        if not 3 <= info.duration <= 10.01:
            raise ValueError("Reference audio must contain 3-10 seconds")
        payload = {"audio": hashlib.sha256(reference.read_bytes()).hexdigest(), "text": reference_text,
                   "language": language, "preprocess": preprocess, "model": self.identity}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        # Model RNG, prompt reuse and ASR share one process/GPU. No concurrent
        # mutation of global Torch RNG or model buffers within this adapter.
        with self._lock:
            prompt_hit = key in self._prompts
            if not prompt_hit:
                self._prompts[key] = self.model.create_voice_clone_prompt(
                    ref_audio=str(reference), ref_text=reference_text, preprocess_prompt=preprocess)
            self._prompts.move_to_end(key)
            while len(self._prompts) > 8:
                self._prompts.popitem(last=False)
            import torch

            normalized = text
            if normalize_text:
                from omnivoice.utils.text import normalize_text as normalize

                normalized = normalize(text, language=language)
            if len(normalized) > 600:
                raise ValueError("Normalized chunk exceeds 600 characters; shorten the planned chunk")
            torch.manual_seed(seed)
            # A finite high threshold avoids the upstream split path. Setting
            # audio_chunk_duration=0 alone does NOT disable it in v0.2.1 source.
            audio = self.model.generate(
                text=text, language=language, voice_clone_prompt=self._prompts[key], instruct=instruct or None,
                num_step=steps, guidance_scale=guidance, speed=speed, denoise=denoise,
                preprocess_prompt=preprocess, postprocess_output=postprocess, normalize_text=normalize_text,
                audio_chunk_threshold=86400.0, pad_duration=.04, fade_duration=.01,
            )[0]
            self.last_metadata = {"seed": seed, "normalized_text": normalized, "prompt_cache_hit": prompt_hit,
                                  "internal_chunking_disabled": True, "normalize_text": normalize_text}
            return np.asarray(audio, dtype=np.float32).reshape(-1)

    def transcribe(self, audio):
        with self._lock:
            if getattr(self.model, "_asr_pipe", None) is None:
                self.model.load_asr_model(device=os.environ.get("OMNIVOICE_ASR_DEVICE", "cpu"))
            return self.model.transcribe((audio, self.sample_rate))
