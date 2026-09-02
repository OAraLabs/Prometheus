"""FIRSTLIGHT FL-2 — a fresh install must advertise the shipped tool set.

The stranger-walk found that ``oara setup`` wrote NO ``tools:``
section, so ``DynamicToolLoader`` got an empty ``always_loaded`` and — with
deferred mode ``auto`` resolving ON for every local provider — a fresh
install advertised nothing. The turn in the walk worked only because the
scripted stub calls a tool regardless, surfacing as the loop's ``Lucky
guess`` allowance; a real model would have sat toolless.

Three pins, per Standing-Principles §2d (assert what the CONSUMER receives):

* the FAR-SIDE guard — the config setup writes, pushed through the REAL
  ``DynamicToolLoader``, must advertise exactly what the shipped template
  advertises;
* the DRIFT guard — ``SHIPPED_ALWAYS_LOADED`` (hardcoded because pip
  installs don't carry the template file) must equal the template's list;
* the WALK's GAP-1 — the no-server message must name ``--probe-url``, so a
  user whose server is elsewhere is not told to install one they have.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from prometheus.cli.init import SHIPPED_ALWAYS_LOADED, run_init
from tests.support.advertisement import (
    TEMPLATE_CONFIG,
    advertised_names,
    build_registry,
)


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps(
            {"object": "list", "data": [{"id": "fl2-model"}]}
        ).encode()
        self.send_response(200 if self.path.rstrip("/") == "/v1/models" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def model_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def test_generated_config_advertises_exactly_the_template_set(model_server, tmp_path):
    """Far side: the yaml setup writes, through the real loader, equals the
    shipped default's advertised set — not "a tools key exists"."""
    from prometheus.context.dynamic_tools import DynamicToolLoader

    config = run_init(noninteractive=True, target_dir=tmp_path, timeout=2.0,
                      probe_url=model_server)
    assert config is not None
    written = yaml.safe_load((tmp_path / "prometheus.yaml").read_text())
    deferred = (written.get("tools") or {}).get("deferred_loading")
    assert deferred, "setup wrote no tools.deferred_loading section (FL-2)"

    loader = DynamicToolLoader(build_registry(), deferred)
    got = {s.get("name") for s in loader.schemas_for_run(True)}
    assert got == advertised_names(TEMPLATE_CONFIG), (
        "a fresh install and the shipped template advertise different sets"
    )


def test_shipped_list_matches_the_template(tmp_path):
    """Drift guard for the hardcoded list (pip installs lack the template)."""
    template = yaml.safe_load(TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    template_list = template["tools"]["deferred_loading"]["always_loaded"]
    assert list(SHIPPED_ALWAYS_LOADED) == list(template_list), (
        "SHIPPED_ALWAYS_LOADED in cli/init.py drifted from "
        "config/prometheus.yaml.default — update BOTH or neither"
    )


def test_no_server_message_names_probe_url(tmp_path, capsys):
    """GAP-1: the failure message is where the server-elsewhere user is
    looking; --help is not. Refusal direction stays refusal (no config).

    Hermetic via a dead probe_url — ``candidates=[]`` is falsy and falls
    back to the REAL well-known ports, which on a developer box can detect
    a live server (observed: this test first wrote a config pointing at the
    box's actual ollama)."""
    config = run_init(noninteractive=True, target_dir=tmp_path, timeout=0.5,
                      probe_url="http://127.0.0.1:9")
    assert config is None
    out = capsys.readouterr().out
    assert "--probe-url" in out, (
        "the no-server message must name --probe-url, not only tell the "
        "user to install a server they may already have"
    )
