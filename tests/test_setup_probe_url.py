"""``oara setup --probe-url`` — the flag the FIRSTLIGHT harness surfaced.

``setup --noninteractive`` probes four fixed localhost ports. On a machine
whose inference server listens anywhere else — and in CI, always — there is
nothing to detect, so the noninteractive path exits without writing a config
and the README's first ten minutes cannot complete. ``--probe-url`` points
the probe at ONE user-supplied URL through :func:`remote_server_candidates`,
the same "what counts as an inference server" definition the interactive
remote prompt and the pairing wizard already use.

Both directions are tested (Standing-Principles §2c): a reachable URL is
ADMITTED (config written, pointing at it), and an unreachable one is REFUSED
(no config, clean no-crash exit) — plus the flag's existence on the parser,
so the CLI surface and the threading cannot drift apart silently.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prometheus.cli.init import run_init
from prometheus.cli.setup import add_setup_arguments


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps(
                {"object": "list", "data": [{"id": "probe-url-model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


@pytest.fixture
def model_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_setup_arguments(p)
    return p


def test_probe_url_flag_exists_on_the_setup_parser():
    args = _parser().parse_args(["--noninteractive", "--probe-url", "http://x:1"])
    assert args.probe_url == "http://x:1"
    # And its default is None — absent flag must not change the classic path.
    assert _parser().parse_args(["--noninteractive"]).probe_url is None


def test_probe_url_is_admitted_and_the_config_points_at_it(model_server, tmp_path):
    config = run_init(
        noninteractive=True, target_dir=tmp_path, timeout=2.0,
        probe_url=model_server,
    )
    assert config is not None, "a reachable --probe-url server must be detected"
    assert config["model"]["base_url"] == model_server
    assert config["model"]["provider"] == "llama_cpp"
    written = (tmp_path / "prometheus.yaml").read_text(encoding="utf-8")
    assert model_server in written

    # Far side of the flag: the model name came from the server's own
    # /v1/models answer, not from a hardcoded default.
    assert config["model"]["model"] == "probe-url-model"


def test_unreachable_probe_url_is_refused_without_a_config(tmp_path):
    # A port from the ephemeral range with nothing listening.
    config = run_init(
        noninteractive=True, target_dir=tmp_path, timeout=0.5,
        probe_url="http://127.0.0.1:9",  # discard port — never an HTTP server
    )
    assert config is None
    assert not (tmp_path / "prometheus.yaml").exists(), (
        "no config may be written when the probed URL is not an inference "
        "server — a config known broken is the dead-end rule violated"
    )


def test_explicit_candidates_still_win_over_probe_url(model_server, tmp_path):
    # The tests-only injection point keeps priority, so existing wizard tests
    # cannot be silently rerouted by a stray probe_url.
    config = run_init(
        noninteractive=True, target_dir=tmp_path, timeout=2.0,
        candidates=[{"name": "injected", "url": model_server,
                     "models_path": "/v1/models", "provider": "llama_cpp"}],
        probe_url="http://127.0.0.1:9",
    )
    assert config is not None
    assert config["model"]["base_url"] == model_server
