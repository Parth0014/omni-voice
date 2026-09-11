"""Local OmniVoice Studio. Serves only localhost; generation uses the configured Colab."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

from generate_narration import build_runtime_config, load_runtime_environment, write_json
from omni_voice import OmniVoiceClient, validate_settings
from storytelling import build_story_plan
from worker_document import extract_worker_blocks

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "output" / "studio"
ASSETS = ROOT / "studio_web"
LOCK = threading.Lock()
JOBS = {}
ACTIVE = None
FIELDS = {"endpoint", "passage", "format", "reference", "reference_text", "quote_reference",
          "quote_reference_text", "quote_mode", "delivery", "steps", "guidance", "speed",
          "instruct", "denoise", "mastering", "spectral_matching", "normalize_text",
          "verification", "take", "timeout", "pronunciation", "directions"}


def endpoint_url(value):
    parsed = urlparse(value.strip())
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("Enter an HTTPS Colab URL without credentials, query or fragment.")
    return value.strip().rstrip("/")


def reference_path(token):
    if not re.fullmatch(r"[a-f0-9]{64}\.wav", token or ""):
        raise ValueError("Choose a reference recording.")
    path = DATA / "references" / token
    if not path.is_file():
        raise ValueError("Reference recording is missing. Upload it again.")
    return path


def save_reference(data):
    if len(data) > 30 * 1024 * 1024:
        raise ValueError("Reference files must be smaller than 30 MB.")
    info = sf.info(io.BytesIO(data))
    if not 3 <= info.duration <= 10.01 or info.channels not in (1, 2):
        raise ValueError("Use a mono/stereo recording of 3-10 seconds.")
    # Store a real WAV regardless of the uploaded container.
    wave, rate = sf.read(io.BytesIO(data), dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, wave, rate, format="WAV", subtype="FLOAT")
    content = buffer.getvalue()
    token = hashlib.sha256(content).hexdigest() + ".wav"
    path = DATA / "references" / token
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return {"id": token, "seconds": round(info.duration, 2), "url": "/reference/" + token}


def source_html(passage, kind):
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("Enter a passage first.")
    if len(passage) > 150000:
        raise ValueError("Keep passages below 150,000 characters.")
    if kind == "html":
        return passage
    if kind != "text":
        raise ValueError("Unknown passage format.")
    parts = []
    for block in re.split(r"\n\s*\n", passage.strip()):
        lines = block.strip().splitlines()
        quote = all(line.lstrip().startswith(">") for line in lines)
        text = " ".join(line.lstrip()[1:].strip() if quote else line.strip() for line in lines)
        tag = "blockquote" if quote else "p"
        parts.append(f"<{tag}>{html.escape(text)}</{tag}>")
    return "\n".join(parts)


def dictionary(value, label):
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object.")
    return value


def defaults():
    env = load_runtime_environment()
    cli = build_runtime_config(ROOT, env)
    local = {}
    settings = ROOT / "local-test.settings.json"
    if settings.exists():
        local = json.loads(settings.read_text())
    result = dict(endpoint=local.get("endpoint") or cli["endpoint"], passage="", format="text",
                  reference="", reference_text=local.get("reference_text", ""),
                  quote_reference="", quote_reference_text="", quote_mode="preserve",
                  delivery="warm", steps=64, guidance=2.0, speed=1.0, instruct="",
                  denoise=True, mastering="auto", spectral_matching=False,
                  normalize_text="auto", verification="local", take=0, timeout=600,
                  pronunciation="{}", directions="{}")
    reference = Path(local.get("reference") or cli["narration_reference_audio"])
    if reference.is_file():
        try:
            result["reference"] = save_reference(reference.read_bytes())["id"]
        except (ValueError, RuntimeError):
            pass
    saved = DATA / "settings.json"
    if saved.exists():
        result.update({k: v for k, v in json.loads(saved.read_text()).items() if k in FIELDS})
    return result


def local_model():
    directory = ROOT / "output" / "asr-models" / "models--Systran--faster-whisper-small.en" / "snapshots"
    choices = sorted(directory.glob("*/model.bin"))
    if not choices:
        raise ValueError("Local word-checking model not found. Choose Endpoint checks or Off.")
    return choices[0].parent


def prepare(config, plan_only=False):
    c = {k: v for k, v in config.items() if k in FIELDS}
    c["endpoint"] = endpoint_url(c.get("endpoint", ""))
    validate_settings(steps=c.get("steps"), guidance=c.get("guidance"), speed=c.get("speed"),
                      instruct=c.get("instruct", ""), denoise=c.get("denoise"))
    if c.get("mastering") not in {"auto", "natural", "warm_story", "gain", "off"}:
        raise ValueError("Choose a mastering profile.")
    if c.get("verification") not in {"local", "required", "auto", "off"}:
        raise ValueError("Choose a word-checking mode.")
    if c.get("normalize_text") not in {"auto", "on", "off"}:
        raise ValueError("Choose a text-normalization mode.")
    if not isinstance(c.get("spectral_matching"), bool):
        raise ValueError("Spectral matching must be on or off.")
    if c["spectral_matching"] and c["mastering"] in {"gain", "off"}:
        raise ValueError("Spectral matching requires an automatic, natural or warm-story master.")
    if isinstance(c.get("take"), bool) or not isinstance(c.get("take"), int) or c["take"] < 0:
        raise ValueError("Take must be a non-negative whole number.")
    if not isinstance(c.get("timeout"), (int, float)) or not 30 <= c["timeout"] <= 1800:
        raise ValueError("Timeout must be between 30 and 1800 seconds.")
    source = source_html(c.get("passage"), c.get("format"))
    pronunciations = dictionary(c.get("pronunciation", "{}"), "Pronunciation")
    directions = dictionary(c.get("directions", "{}"), "Directions")
    plan = build_story_plan(extract_worker_blocks(source), quote_mode=c.get("quote_mode"),
                            delivery=c.get("delivery"), pronunciation=pronunciations, directions=directions)
    if not plan:
        raise ValueError("No spoken content remains after extraction and quote handling.")
    if any(not .5 <= item["speed_factor"] * c["speed"] <= 1.5 for item in plan):
        raise ValueError("Combined speed and paragraph directions exceed the model's 0.5-1.5 range.")
    if not plan_only:
        reference_path(c.get("reference"))
        if not isinstance(c.get("reference_text"), str) or not c["reference_text"].strip():
            raise ValueError("Enter the exact words spoken in your reference recording.")
        if c["quote_mode"] == "two_voice":
            reference_path(c.get("quote_reference"))
            if not isinstance(c.get("quote_reference_text"), str) or not c["quote_reference_text"].strip():
                raise ValueError("Enter the exact transcript for the quote voice.")
            if c["quote_reference"] == c["reference"]:
                raise ValueError("Two voices require distinct reference recordings.")
        if c["verification"] == "local":
            local_model()
    return c, source, plan, pronunciations, directions


def _run_job(job_id, c, source, plan, pronunciations, directions):
    directory = DATA / "jobs" / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "passage.html").write_text(source, encoding="utf-8")
    write_json(directory / "settings.json", c)
    command = [sys.executable, "-u", str(ROOT / "generate_narration.py"),
               "--html", str(directory / "passage.html"), "--reference", str(reference_path(c["reference"])),
               "--reference-text", c["reference_text"], "--engine", "gradio", "--endpoint", c["endpoint"],
               "--output-dir", str(directory)]
    for key, flag in {"quote_mode": "--quote-mode", "delivery": "--delivery", "steps": "--steps",
                      "guidance": "--guidance", "speed": "--speed", "instruct": "--instruct",
                      "mastering": "--mastering", "take": "--take", "timeout": "--remote-timeout"}.items():
        command.extend([flag, str(c[key])])
    if c["quote_mode"] == "two_voice":
        command.extend(["--quote-reference", str(reference_path(c["quote_reference"])),
                        "--quote-reference-text", c["quote_reference_text"]])
    if not c["denoise"]:
        command.append("--no-denoise")
    if c["spectral_matching"]:
        command.append("--spectral-matching")
    if c["normalize_text"] != "auto":
        command.append("--normalize-text" if c["normalize_text"] == "on" else "--no-normalize-text")
    if c["verification"] == "local":
        command.extend(["--local-asr-model", str(local_model()), "--verify-text", "required"])
    else:
        command.extend(["--verify-text", c["verification"]])
    for key, data in (("pronunciation", pronunciations), ("directions", directions)):
        path = directory / (key + ".json")
        write_json(path, data)
        command.extend(["--" + key, str(path)])
    environment = dict(os.environ)
    # Each request is fully specified and cannot inherit a previous CLI's voice or output settings.
    for key in list(environment):
        if key.startswith(("NARRATION_", "OMNIVOICE_")):
            environment.pop(key)
    environment["OMNIVOICE_CACHE_DIR"] = str(DATA / "cache")
    environment["PYTHONIOENCODING"] = "utf-8"
    global ACTIVE
    try:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                line = line.rstrip()
                with LOCK:
                    job = JOBS[job_id]
                    job["log"].append(line)
                    job["log"] = job["log"][-150:]
                    match = re.match(r"(\d+)/(\d+) ", line)
                    if match:
                        job["completed"] = int(match[1])
                with (directory / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            code = process.wait()
        if code:
            raise RuntimeError("Generation failed. See the run log below for the reason.")
        wav = sorted(directory.glob("narration_*.wav"))[-1]
        report = json.loads(wav.with_suffix(".json").read_text())
        summary = {"seconds": report["duration_seconds"], "warnings": report["warnings"],
                   "word_checks": [chunk["generation"]["verification"]["status"] for chunk in report["chunks"]],
                   "audio": f"/result/{job_id}/{wav.name}",
                   "report": f"/result/{job_id}/{wav.with_suffix('.json').name}",
                   "player": f"/result/{job_id}/{wav.with_suffix('.html').name}"}
        with LOCK:
            JOBS[job_id].update(state="complete", result=summary)
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(state="failed", error=str(exc))
    finally:
        with LOCK:
            write_json(directory / "job.json", JOBS[job_id])
            ACTIVE = None


def run_job(job_id, *args):
    global ACTIVE
    try:
        _run_job(job_id, *args)
    except Exception as exc:
        with LOCK:
            JOBS[job_id].update(state="failed", error=str(exc))
            ACTIVE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def allowed(self):
        port = self.server.server_port
        if self.headers.get("Host") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def send(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def file(self, path):
        if not path.is_file():
            return self.send({"error": "File not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed():
            return self.send({"error": "Local access only"}, 403)
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self.file(ASSETS / "index.html")
            if path in {"/app.js", "/styles.css"}:
                return self.file(ASSETS / path[1:])
            if path == "/api/config":
                return self.send(defaults())
            if path == "/api/sample":
                return self.send({"passage": (ROOT / "sample.html").read_text(encoding="utf-8"), "format": "html"})
            if path == "/api/jobs":
                with LOCK:
                    jobs = list(JOBS.values())
                for entry in sorted((DATA / "jobs").glob("*/job.json"), reverse=True)[:20]:
                    data = json.loads(entry.read_text())
                    if not any(j["id"] == data["id"] for j in jobs):
                        jobs.append(data)
                return self.send(sorted(jobs, key=lambda j: j["id"], reverse=True)[:20])
            if path.startswith("/api/job/"):
                job_id = path.split("/")[-1]
                with LOCK:
                    job = JOBS.get(job_id)
                return self.send(job or {"error": "Job not found"}, 200 if job else 404)
            if path.startswith("/reference/"):
                return self.file(reference_path(path.split("/")[-1]))
            if path.startswith("/result/"):
                parts = path.split("/")
                if len(parts) != 4 or not re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{8}", parts[2]):
                    raise ValueError("Invalid result")
                if not re.fullmatch(r"narration_\d{4}\.(wav|json|html)", parts[3]):
                    raise ValueError("Invalid artifact")
                return self.file(DATA / "jobs" / parts[2] / parts[3])
            return self.send({"error": "Not found"}, 404)
        except (ValueError, OSError) as exc:
            self.send({"error": str(exc)}, 400)

    def do_POST(self):
        if not self.allowed() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send({"error": "Local JSON requests only"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 42 * 1024 * 1024:
                raise ValueError("Request is too large or empty.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if self.path == "/api/reference":
                return self.send(save_reference(base64.b64decode(body["data"], validate=True)))
            if self.path == "/api/config":
                unknown = set(body) - FIELDS
                if unknown:
                    raise ValueError("Unknown settings.")
                body["endpoint"] = endpoint_url(body.get("endpoint", ""))
                write_json(DATA / "settings.json", body)
                return self.send({"saved": True})
            if self.path == "/api/connect":
                client = OmniVoiceClient(endpoint_url(body.get("endpoint", "")), timeout=30)
                return self.send({"identity": client.identity, "capabilities": client.capabilities})
            if self.path in {"/api/plan", "/api/generate"}:
                prepared = prepare(body, plan_only=self.path == "/api/plan")
                if self.path == "/api/plan":
                    return self.send({"chunks": prepared[2]})
                global ACTIVE
                with LOCK:
                    if ACTIVE:
                        return self.send({"error": "A narration is already running. Wait for it to finish."}, 409)
                    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
                    ACTIVE = job_id
                    JOBS[job_id] = {"id": job_id, "state": "running", "completed": 0,
                                    "total": len(prepared[2]), "log": [], "endpoint": prepared[0]["endpoint"]}
                thread = threading.Thread(target=run_job, args=(job_id, *prepared), daemon=True)
                thread.start()
                return self.send({"id": job_id}, 202)
            return self.send({"error": "Not found"}, 404)
        except Exception as exc:
            self.send({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"OmniVoice Studio: {url}\nOutputs: {DATA}\nKeep this window open while generating.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
