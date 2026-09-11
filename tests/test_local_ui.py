import base64
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import soundfile as sf

import local_ui


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(local_ui, "DATA", tmp_path)
    monkeypatch.setattr(local_ui, "ACTIVE", None)
    monkeypatch.setattr(local_ui, "JOBS", {})
    server = local_ui.ThreadingHTTPServer(("127.0.0.1", 0), local_ui.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()
    server.server_close()


def request(url, data=None, headers=None):
    body = None if data is None else json.dumps(data).encode()
    return urlopen(Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})}))


def config():
    return dict(endpoint="https://example.gradio.live", passage="A quiet morning.\n\n> Thank you.",
                format="text", steps=64, guidance=2.0, speed=1.0, instruct="", denoise=True,
                quote_mode="preserve", delivery="warm", mastering="auto", spectral_matching=False,
                normalize_text="auto", verification="off", take=0, timeout=600,
                pronunciation="{}", directions="{}", reference="", reference_text="Exact words.")


def test_plan_offline_and_quote_routing(studio):
    url, _ = studio
    c = config()
    c["quote_mode"] = "two_voice"
    with request(url + "/api/plan", c) as response:
        plan = json.load(response)["chunks"]
    assert [chunk["role"] for chunk in plan] == ["narration", "quote"]
    assert "Thank you." == plan[1]["text"]


def test_reference_upload_duration_and_playback(studio):
    url, _ = studio
    buffer = io.BytesIO()
    sf.write(buffer, np.full(4 * 24000, .1), 24000, format="WAV")
    with request(url + "/api/reference", {"data": base64.b64encode(buffer.getvalue()).decode()}) as response:
        ref = json.load(response)
    with request(url + ref["url"]) as response:
        assert sf.info(io.BytesIO(response.read())).duration == 4
    tiny = io.BytesIO()
    sf.write(tiny, np.ones(24000), 24000, format="WAV")
    with pytest.raises(HTTPError) as exc:
        request(url + "/api/reference", {"data": base64.b64encode(tiny.getvalue()).decode()})
    assert exc.value.code == 400


def test_global_endpoint_persists_and_blocks_cross_origin_writes(studio):
    url, folder = studio
    with request(url + "/api/config", config()) as response:
        assert json.load(response)["saved"]
    assert json.loads((folder / "settings.json").read_text())["endpoint"] == config()["endpoint"]
    with pytest.raises(HTTPError) as exc:
        request(url + "/api/config", config(), {"Origin": "https://unrelated.example"})
    assert exc.value.code == 403


def test_generation_needs_exact_reference_before_start(studio):
    url, _ = studio
    with pytest.raises(HTTPError) as exc:
        request(url + "/api/generate", config())
    assert exc.value.code == 400
    assert local_ui.ACTIVE is None


def test_reference_and_result_paths_are_confined(studio):
    url, _ = studio
    for path in ["/reference/secret.env", "/result/../.env"]:
        with pytest.raises(HTTPError) as exc:
            request(url + path)
        assert exc.value.code == 400


def test_runner_clears_busy_state_on_setup_failure(studio, monkeypatch):
    _, _folder = studio
    local_ui.ACTIVE = "job"
    local_ui.JOBS["job"] = {"id": "job", "state": "running", "log": []}
    monkeypatch.setattr(local_ui, "_run_job", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    local_ui.run_job("job", {}, "", [], {}, {})
    assert local_ui.ACTIVE is None
    assert local_ui.JOBS["job"]["state"] == "failed"
