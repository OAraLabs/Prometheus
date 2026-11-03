"""The terminal chat, the anatomy scan and the doctor's Inference rows ask a keyed llama.cpp server with its key.

Follow-up to the provider's optional key: the provider sent it, but the probes around it did not, so a server started
with ``--api-key`` showed as unreachable in ``oara doctor`` and the anatomy scan, and the terminal chat could not read
its model or chat template. Each now sends the configured key as a bearer token. The key is found where the daemon
finds it: the environment, then the env file (``oara doctor`` and the terminal chat do not load the env file into
the environment). A named variable that is set nowhere is reported by name, never as "not responding".

The stand-in is a real HTTP server on loopback that answers only with the key, like ``llama-server --api-key``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

KEY_ENV = "LLAMA_CPP_PROBE_TEST_KEY"
KEY = "probe-test-" + "k3y" * 6

PROPS = {
    "default_generation_settings": {"n_ctx": 32768},
    "modalities": {"vision": True},
    "total_slots": 2,
    "chat_template": "{{ messages }}",
}


class _Handler(BaseHTTPRequestHandler):
    seen: list[tuple[str, str | None]]

    def log_message(self, *args) -> None:       # quiet
        pass

    def do_GET(self) -> None:  # noqa: N802
        auth = self.headers.get("Authorization")
        self.server.seen.append((self.path, auth))  # type: ignore[attr-defined]
        if auth != f"Bearer {KEY}":
            self._send(401, {"error": {"message": "Invalid API Key"}})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": "keyed-model-Q4_K_M.gguf"}]})
        elif self.path == "/props":
            self._send(200, PROPS)
        elif self.path == "/slots":
            self._send(200, [])
        else:
            self._send(404, {})

    def _send(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def keyed():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """An empty env file of our own, so the developer's real one is never read."""
    path = tmp_path / "env"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(KEY_ENV, raising=False)
    return path


def model(url: str, **extra) -> dict:
    return {"provider": "llama_cpp", "base_url": url, "api_key_env": KEY_ENV, **extra}


# ── the shared lookup ────────────────────────────────────────────────────────

def test_the_header_comes_from_the_environment(env_file, monkeypatch):
    from prometheus.providers.registry import local_server_headers

    monkeypatch.setenv(KEY_ENV, KEY)
    assert local_server_headers(model("http://x")) == {"Authorization": f"Bearer {KEY}"}


def test_the_header_comes_from_the_env_file_when_the_environment_has_none(env_file):
    from prometheus.providers.registry import local_server_headers

    env_file.write_text(f"{KEY_ENV}={KEY}\n", encoding="utf-8")
    assert local_server_headers(model("http://x")) == {"Authorization": f"Bearer {KEY}"}


def test_no_key_configured_is_no_header(env_file):
    from prometheus.providers.registry import local_server_headers

    assert local_server_headers({"provider": "llama_cpp", "base_url": "http://x"}) == {}
    assert local_server_headers({"provider": "llama_cpp", "api_key_env": ""}) == {}
    assert local_server_headers({"provider": "ollama", "api_key_env": KEY_ENV}) == {}


def test_a_named_variable_set_nowhere_is_an_error_naming_it(env_file):
    from prometheus.providers.registry import local_server_headers

    with pytest.raises(ValueError, match=KEY_ENV):
        local_server_headers(model("http://x"))


# ── oara doctor's Inference row ──────────────────────────────────────────────

def test_oara_doctor_reaches_a_keyed_server(keyed, env_file, monkeypatch):
    from prometheus.cli.doctor import check_inference

    server, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    reach, found = check_inference({"model": model(url)})
    assert reach.status == "ok", reach.message
    assert found.status == "ok", found.message
    assert all(auth == f"Bearer {KEY}" for _, auth in server.seen)


def test_oara_doctor_names_an_unset_key_variable(keyed, env_file):
    from prometheus.cli.doctor import check_inference

    server, url = keyed
    reach, _ = check_inference({"model": model(url)})
    assert reach.status == "error"
    assert KEY_ENV in reach.message
    assert "not responding" not in reach.message
    assert server.seen == []


async def test_the_doctor_inference_row_reaches_a_keyed_server(keyed, env_file, monkeypatch):
    from prometheus.infra.anatomy import AnatomyState
    from prometheus.infra.doctor import Doctor

    _, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    state = AnatomyState(scanned_at="now")
    state.inference_engine = "llama_cpp"
    state.inference_url = url
    check = await Doctor({"model": model(url)})._check_inference(state)
    assert check.status == "ok", check.message


# ── the anatomy scan ─────────────────────────────────────────────────────────

async def test_the_anatomy_scan_reads_a_keyed_servers_model(keyed, env_file, monkeypatch):
    from prometheus.infra.anatomy import AnatomyScanner, AnatomyState

    server, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    scanner = AnatomyScanner.from_config({"model": model(url)})
    state = AnatomyState(scanned_at="now")
    await scanner._detect_model_llama_cpp(state)
    assert state.model_name == "keyed-model-Q4_K_M.gguf"
    assert state.vision_enabled is True
    assert "multi_slot" in state.inference_features
    assert all(auth == f"Bearer {KEY}" for _, auth in server.seen)


# ── the terminal chat ────────────────────────────────────────────────────────

def test_the_terminal_chat_reads_a_keyed_servers_model(keyed, env_file, monkeypatch):
    from prometheus.__main__ import _detect_served_model

    server, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    assert _detect_served_model(model(url)) == "keyed-model-Q4_K_M.gguf"
    assert all(auth == f"Bearer {KEY}" for _, auth in server.seen)


def test_the_terminal_chat_reads_a_keyed_servers_chat_template(keyed, env_file, monkeypatch):
    from prometheus.__main__ import _detect_tool_template_or_none

    server, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    template = _detect_tool_template_or_none(model(url))
    assert template is not None
    assert ("/props", f"Bearer {KEY}") in server.seen


def test_the_key_is_not_in_any_doctor_message(keyed, env_file, monkeypatch, caplog):
    from prometheus.cli.doctor import check_inference

    _, url = keyed
    monkeypatch.setenv(KEY_ENV, KEY)
    reach, found = check_inference({"model": model(url)})
    assert KEY not in reach.message + found.message
    assert not any(KEY in record.getMessage() for record in caplog.records)
