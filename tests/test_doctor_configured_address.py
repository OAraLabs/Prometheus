"""Doctor's Inference row probes the address the config names, at the path that server answers.

The boot check and the chat /doctor built their scanner with ``ollama_url=model.fallback_url`` (default
``http://localhost:11434``) and took ``model.base_url`` only for llama.cpp, so a configured Ollama at another address
was reported "not responding at http://localhost:11434" while it answered; LM Studio and vLLM were asked
``/api/tags`` at 11434, and a cloud primary was "not responding" at 11434 too. The CLI check defaulted every local
provider to :8080 and doubled ``/v1`` for a base that already ends in it.

Every case builds the scanner the way the daemon does (``AnatomyScanner.from_config``) against a stand-in that answers
only the path it should be asked.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def server_answering(path: str):
    """A stand-in that answers 200 only at *path* (404 elsewhere) and records every path it was asked."""
    seen: list[str] = []
    body = json.dumps({"object": "list", "data": [{"id": "some-model"}], "models": [{"name": "some-model"}]}).encode()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            self.send_response(200 if self.path == path else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args, **kwargs):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    finally:
        srv.shutdown()
        srv.server_close()


def inference_row(config: dict):
    from prometheus.infra.anatomy import AnatomyScanner
    from prometheus.infra.doctor import Doctor

    scanner = AnatomyScanner.from_config(config)
    state = asyncio.run(scanner.quick_scan())
    return asyncio.run(Doctor(config)._check_inference(state))


class TestTheBootAndChatDoctor:
    def test_ollama_at_its_configured_address(self):
        with server_answering("/api/tags") as (url, seen):
            row = inference_row({"model": {"provider": "ollama", "base_url": url, "model": "qwen3.5:0.8b"}})
        assert row.status == "ok", row
        assert url.split("//")[1] in row.message and "11434" not in row.message, row.message
        assert "/api/tags" in seen

    def test_base_url_wins_over_a_dead_fallback_url(self):
        with server_answering("/api/tags") as (url, _seen):
            row = inference_row({"model": {"provider": "ollama", "base_url": url,
                                           "fallback_url": f"http://127.0.0.1:{_free_port()}"}})
        assert row.status == "ok", row

    def test_llama_cpp_as_before(self):
        with server_answering("/v1/models") as (url, seen):
            row = inference_row({"model": {"provider": "llama_cpp", "base_url": url}})
        assert row.status == "ok" and "/v1/models" in seen, row

    def test_lm_studio_with_a_v1_base_is_asked_v1_models_once(self):
        with server_answering("/v1/models") as (url, seen):
            row = inference_row({"model": {"provider": "lm_studio", "base_url": f"{url}/v1"}})
        assert row.status == "ok", (row, seen)
        assert "/v1/v1/models" not in seen and "/api/tags" not in seen, seen

    def test_vllm_from_its_base_url_variable(self, monkeypatch):
        with server_answering("/v1/models") as (url, _seen):
            monkeypatch.setenv("VLLM_BASE_URL", f"{url}/v1")
            row = inference_row({"model": {"provider": "vllm"}})
        assert row.status == "ok", row

    def test_a_cloud_primary_is_not_probed_at_localhost(self):
        row = inference_row({"model": {"provider": "anthropic", "model": "claude-sonnet-5"}})
        assert row.status != "error" and "11434" not in row.message, row

    def test_the_daemon_and_the_cli_build_the_scanner_from_the_config(self):
        """One construction point, so the boot path cannot drift back to fallback_url."""
        import prometheus.cli.doctor as cli_doctor
        import prometheus.daemon as daemon_mod

        for module in (daemon_mod, cli_doctor):
            source = inspect.getsource(module)
            assert "AnatomyScanner.from_config(" in source, module.__name__
            assert 'ollama_url=model_config.get("fallback_url"' not in source
            assert 'ollama_url=model_cfg.get("fallback_url"' not in source


class TestTheCliCheck:
    def test_ollama_with_no_base_url_is_looked_for_at_its_own_port(self):
        from prometheus.cli.doctor import check_inference

        reach, _model = check_inference({"model": {"provider": "ollama"}}, timeout=0.5)
        assert "11434" in reach.message and "8080" not in reach.message, reach.message

    def test_lm_studio_with_a_v1_base_is_asked_v1_models_once(self):
        from prometheus.cli.doctor import check_inference

        with server_answering("/v1/models") as (url, seen):
            reach, _model = check_inference({"model": {"provider": "lm_studio", "base_url": f"{url}/v1"}}, timeout=2)
        assert reach.status == "ok", (reach, seen)
        assert "/v1/v1/models" not in seen, seen
