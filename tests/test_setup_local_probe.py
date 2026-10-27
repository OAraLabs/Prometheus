"""Setup's look and probe give a reason (Beacon's docs/ONBOARDING-LOCAL-FIRST.md §4.1, P1). Written before the code: red.

Stand-in apps on ephemeral ports, pointed at through ``local_probe.KNOWN_PLACES`` (never the real 11434 or 1234: real apps and
other sessions use them). Pins:

- the app is identified by what it answers, not its port (an Ollama also answers /v1/models: it was labelled llama_cpp);
- per AI: loaded, can chat, window; only usable ones are usable; one reason when none is, in a fixed order;
- redirects not followed, answers capped, bodies never echoed, probes in parallel;
- a typed computer: its forms, refused forms, a name that does not resolve;
- the routes: detect's new ``local`` beside the unchanged ``backends``, ``POST /api/setup/probe``, configure's refusals and the
  window it writes, ``features`` on status;
- MIN_WINDOW is set only from a recorded measurement (§4.3).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.api_token import TOKEN_ENV_VAR  # noqa: E402
from prometheus.providers import local_probe  # noqa: E402
from prometheus.web.setup_server import PairingState, SetupModeState, create_setup_app  # noqa: E402

CODE = "042999"
MIN = 16384
#: A string a stand-in puts in a field nobody parses: it must never come back.
MARKER = "STANDIN-BODY-MUST-NOT-ECHO-7f3a"


# ---------------------------------------------------------------------------
# Stand-in apps
# ---------------------------------------------------------------------------


def _handler(routes: dict[str, object], *, delay: float = 0.0, status: int = 200, redirect: str | None = None,
             seen: list[str] | None = None, raw: bytes | None = None):
    """A stand-in answering ``routes`` (path → JSON, or a callable of the POST body), or everything with ``status``/``raw``."""

    class H(BaseHTTPRequestHandler):
        def _answer(self, body: object | None) -> None:
            if seen is not None:
                seen.append(f"{self.command} {self.path}")
            if delay:
                time.sleep(delay)
            if redirect is not None:
                self.send_response(302)
                self.send_header("Location", redirect + self.path)
                self.end_headers()
                return
            if raw is not None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if body is None or status != 200:
                self.send_response(status if status != 200 else 404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._answer(routes.get(self.path.split("?")[0]))

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            sent = json.loads(self.rfile.read(length) or b"{}")
            route = routes.get(self.path)
            self._answer(route(sent) if callable(route) else route)

        def log_message(self, *args, **kwargs):
            pass

    return H


def ollama(models: list[tuple[str, list[str], int]], loaded: list[str] = ()) -> dict[str, object]:
    """Ollama: (name, capabilities, trained window) per model; ``loaded`` are in /api/ps. It also answers /v1/models."""
    by_name = {m[0]: m for m in models}
    return {
        "/api/version": {"version": "0.23.0"},
        "/api/tags": {"models": [{"name": n, "model": n, "license": MARKER} for n, _, _ in models]},
        "/api/ps": {"models": [{"name": n, "context_length": by_name[n][2]} for n in loaded]},
        "/api/show": lambda body: {"capabilities": by_name[body.get("model") or body.get("name")][1],
                                   "model_info": {"general.architecture": "qwen3",
                                                  "qwen3.context_length": by_name[body.get("model") or body.get("name")][2]}},
        "/v1/models": {"object": "list", "data": [{"id": n, "object": "model"} for n, _, _ in models]},
    }


def lm_studio(models: list[dict[str, object]]) -> dict[str, object]:
    return {
        "/api/v0/models": {"object": "list", "data": models},
        "/v1/models": {"object": "list", "data": [{"id": m["id"], "object": "model"} for m in models]},
    }


LLAMA_CPP = {
    "/props": {"default_generation_settings": {"n_ctx": 32768}, "model_path": "/models/qwen3-27b.gguf", "total_slots": 1},
    "/v1/models": {"object": "list", "data": [{"id": "qwen3-27b.gguf", "object": "model"}]},
    "/health": {"status": "ok"},
}
VLLM = {
    "/version": {"version": "0.11.0"},
    "/v1/models": {"object": "list", "data": [{"id": "Qwen/Qwen3-8B", "object": "model", "max_model_len": 32768}]},
}
QWEN_LOADED = ollama([("qwen3:8b", ["completion", "tools"], 40960), ("nomic-embed-text:latest", ["embedding"], 8192)],
                     loaded=["qwen3:8b"])


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def places(monkeypatch):
    """Point this machine's known places at stand-ins: places({"ollama": port, ...}); a missing app gets a closed port."""

    monkeypatch.setattr(local_probe, "THIS_MACHINE", "127.0.0.1")
    monkeypatch.setattr(local_probe, "MIN_WINDOW", MIN)

    def point(ports: dict[str, int]) -> None:
        monkeypatch.setattr(local_probe, "KNOWN_PLACES",
                            [{"app": app, "port": ports.get(app) or _free_port()} for app in ("llama_cpp", "ollama", "lm_studio", "vllm")])

    return point


def usable(answer) -> list[tuple[str, str]]:
    return [(f.app, m.id) for f in answer.found for m in f.models if m.usable]


# ---------------------------------------------------------------------------
# What a probe finds
# ---------------------------------------------------------------------------


class TestIdentify:
    def test_ollama_is_ollama_though_it_answers_v1_models_too(self, places):
        with serve(_handler(QWEN_LOADED)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert [f.app for f in a.found] == ["ollama"]
        assert a.found[0].provider == "ollama"

    @pytest.mark.parametrize("routes,app", [
        (lm_studio([{"id": "qwen3-14b", "type": "llm", "state": "loaded", "loaded_context_length": 32768}]), "lm_studio"),
        (LLAMA_CPP, "llama_cpp"),
        (VLLM, "vllm"),
        ({"/v1/models": {"object": "list", "data": [{"id": "some-ai", "object": "model"}]}}, "openai_compatible"),
    ])
    def test_each_app_by_what_it_answers(self, places, routes, app):
        with serve(_handler(routes)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert [f.app for f in a.found] == [app]
        assert a.found[0].base_url == f"http://127.0.0.1:{port}"

    def test_versions_where_the_app_says(self, places):
        with serve(_handler(QWEN_LOADED)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert a.found and a.found[0].version == "0.23.0"


class TestPerModel:
    def test_ollama_loaded_chat_and_window(self, places):
        with serve(_handler(QWEN_LOADED)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        models = {m.id: m for m in a.found[0].models}
        assert models["qwen3:8b"].loaded is True and models["qwen3:8b"].chat is True
        assert models["qwen3:8b"].window == 40960 and models["qwen3:8b"].window_source == "trained"
        assert models["nomic-embed-text:latest"].chat is False and models["nomic-embed-text:latest"].usable is False
        assert models["nomic-embed-text:latest"].problem == "no_chat"
        assert usable(a) == [("ollama", "qwen3:8b")] and a.reason is None

    def test_llama_cpp_window_is_the_served_one(self, places):
        with serve(_handler(LLAMA_CPP)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        m = a.found[0].models[0]
        assert (m.window, m.window_source, m.loaded) == (32768, "served", True)

    def test_vllm_window_is_max_model_len(self, places):
        with serve(_handler(VLLM)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert (a.found[0].models[0].window, a.found[0].models[0].window_source) == (32768, "served")

    def test_an_lm_studio_embedding_model_is_not_a_chat(self, places):
        routes = lm_studio([{"id": "qwen3-14b", "type": "llm", "state": "loaded", "loaded_context_length": 32768},
                            {"id": "text-embedding-nomic", "type": "embeddings", "state": "not-loaded"}])
        with serve(_handler(routes)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert usable(a) == [("lm_studio", "qwen3-14b")]


class TestReasons:
    @pytest.mark.parametrize("routes,reason", [
        (ollama([]), "no_models"),
        (ollama([("nomic-embed-text:latest", ["embedding"], 8192)]), "no_chat"),
        (lm_studio([{"id": "qwen3-14b", "type": "llm", "state": "not-loaded", "max_context_length": 131072}]), "none_loaded"),
        (lm_studio([{"id": "gemma-3-12b", "type": "llm", "state": "loaded", "loaded_context_length": 4096}]), "short_context"),
    ])
    def test_a_running_app_with_nothing_usable_says_why(self, places, routes, reason):
        with serve(_handler(routes)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert usable(a) == []
        assert a.reason == reason
        assert a.detail  # Prometheus's own sentence

    def test_short_context_is_against_min_window_and_says_it(self, places):
        routes = lm_studio([{"id": "gemma-3-12b", "type": "llm", "state": "loaded", "loaded_context_length": MIN}])
        with serve(_handler(routes)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert usable(a) == [("lm_studio", "gemma-3-12b")], "exactly the floor is enough"
        assert a.min_window == MIN

    def test_html_is_not_ai(self, places):
        with serve(_handler({}, raw=b"<!doctype html><html>dashboard</html>")) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert (a.found, a.reason) == ([], "not_ai")

    @pytest.mark.parametrize("status", [401, 403])
    def test_a_locked_app_needs_a_key(self, places, status):
        with serve(_handler({}, status=status)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert (a.found, a.reason) == ([], "needs_key")

    def test_nothing_listening_is_refused(self, places):
        a = local_probe.probe_target(f"127.0.0.1:{_free_port()}")
        assert (a.found, a.reason) == ([], "refused")

    def test_an_app_that_never_answers_is_no_answer(self, places):
        with serve(_handler(QWEN_LOADED, delay=5.0)) as port:
            t0 = time.monotonic()
            a = local_probe.probe_target(f"127.0.0.1:{port}", timeout=0.5)
            took = time.monotonic() - t0
        assert (a.found, a.reason) == ([], "no_answer")
        assert took < 3.0

    def test_a_name_that_does_not_resolve(self, places):
        a = local_probe.probe_target("no-such-computer.invalid")
        assert (a.found, a.reason) == ([], "name_not_found")

    def test_a_bad_target_is_bad_target(self, places):
        assert local_probe.probe_target("user:pass@studio-mac").reason == "bad_target"


class TestWhatAProbeMayDo:
    def test_a_redirect_is_not_followed(self, places):
        seen: list[str] = []
        with serve(_handler(QWEN_LOADED, seen=seen)) as real, \
                serve(_handler({}, redirect=f"http://127.0.0.1:{real}")) as port:
            direct = local_probe.probe_target(f"127.0.0.1:{real}")
            asked = len(seen)
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert [f.app for f in direct.found] == ["ollama"], "control: the target itself is found when asked directly"
        assert a.found == [] and len(seen) == asked, "the redirect's target was asked nothing"

    def test_an_answer_over_the_cap_is_not_read_as_an_app(self, places):
        small = {"/v1/models": {"object": "list", "data": [{"id": "x" * 1024, "object": "model"}] * 4}}
        big = {"/v1/models": {"object": "list", "data": [{"id": "x" * 1024, "object": "model"}] * 400}}
        with serve(_handler(small)) as s_port, serve(_handler(big)) as b_port:
            control = local_probe.probe_target(f"127.0.0.1:{s_port}")
            a = local_probe.probe_target(f"127.0.0.1:{b_port}")
        assert [f.app for f in control.found] == ["openai_compatible"], "control: the same shape, under the cap, is an app"
        assert a.found == []

    def test_no_body_is_echoed(self, places):
        with serve(_handler(QWEN_LOADED)) as port:
            a = local_probe.probe_target(f"127.0.0.1:{port}")
        assert a.found, "the stand-in was found"
        assert MARKER not in json.dumps(a.as_dict())

    def test_this_machine_is_looked_at_in_parallel(self, places):
        slow = _handler(QWEN_LOADED, delay=3.0)
        with serve(slow) as a1, serve(slow) as a2, serve(slow) as a3, serve(slow) as a4:
            places({"llama_cpp": a1, "ollama": a2, "lm_studio": a3, "vllm": a4})
            t0 = time.monotonic()
            a = local_probe.look_here(timeout=1.0)
            took = time.monotonic() - t0
        assert a.reason == "no_answer", "every place was asked, and each timed out"
        assert took < 2.0, f"four places took {took:.1f}s: one after another, not in parallel"

    def test_detect_s_old_answer_asks_the_same_places_unchanged_by_default(self):
        """backends and local describe the same ports, and with the defaults backends asks exactly what it always did."""
        from prometheus.cli.init import KNOWN_LOCAL_SERVERS

        assert local_probe.legacy_candidates() == KNOWN_LOCAL_SERVERS

    def test_look_here_finds_each_app_where_it_is(self, places):
        with serve(_handler(QWEN_LOADED)) as o, serve(_handler(LLAMA_CPP)) as lc:
            places({"ollama": o, "llama_cpp": lc})
            a = local_probe.look_here()
        assert sorted(f.app for f in a.found) == ["llama_cpp", "ollama"]
        assert a.target is None and a.reason is None

    def test_a_bare_name_tries_every_known_port_on_that_computer(self, places):
        with serve(_handler(QWEN_LOADED)) as o:
            places({"ollama": o})
            a = local_probe.probe_target("127.0.0.1")
        assert [f.base_url for f in a.found] == [f"http://127.0.0.1:{o}"]


class TestTargets:
    @pytest.mark.parametrize("text,want", [
        ("studio-mac", (None, "studio-mac", None)),
        ("Studio-Mac.local", (None, "studio-mac.local", None)),
        ("192.0.2.20:11434", (None, "192.0.2.20", 11434)),
        ("[fd00::20]:1234", (None, "fd00::20", 1234)),
        ("http://studio-mac.local:1234/", ("http", "studio-mac.local", 1234)),
    ])
    def test_accepted(self, text, want):
        assert local_probe.parse_target(text) == want

    @pytest.mark.parametrize("text", [
        "", "studio mac", "user:pass@studio-mac", "http://user@studio-mac", "studio-mac/v1/models", "studio-mac?x=1",
        "file:///etc/hosts", "ftp://studio-mac", "studio-mac:99999", "studio-mac:0", "a" * 254, "studio\nmac",
    ])
    def test_refused(self, text):
        assert local_probe.parse_target(text) is None


class TestMinWindow:
    def test_it_is_set_from_a_recorded_measurement(self):
        m = local_probe.MIN_WINDOW_MEASURED
        assert isinstance(m, dict), "MIN_WINDOW is set only after measuring round 0 (Beacon spec §4.3)"
        assert m.get("commit") and m.get("method")
        round0 = m.get("round0") or {}
        assert set(round0) >= {"llama_cpp", "ollama"} and all(isinstance(v, int) and v > 0 for v in round0.values())
        need = max(round0.values()) + 4096 + 4096
        want = -(-need // 4096) * 4096 + 4096  # up to the next 4,096, plus 4,096 of headroom
        assert local_probe.MIN_WINDOW == want

    @pytest.mark.parametrize("round0,floor", [(8253, 24576), (8192, 20480), (8193, 24576), (1, 16384)])
    def test_the_rule(self, round0, floor):
        assert local_probe.min_window_for(round0) == floor


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    import os

    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    yield path
    os.environ.pop(TOKEN_ENV_VAR, None)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    confdir = tmp_path / "confdir"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(confdir))
    monkeypatch.chdir(tmp_path)
    return confdir


def client_and_auth():
    app = create_setup_app(PairingState(code=CODE), api_port=8123, ws_port=8124, state=SetupModeState())
    client = TestClient(app)
    token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
    return client, {"Authorization": f"Bearer {token}"}


class TestRoutes:
    def test_status_says_it_can_probe(self, env_file, config_dir):
        client, _ = client_and_auth()
        assert "local_probe" in client.get("/api/setup/status").json().get("features", [])

    def test_detect_adds_local_and_keeps_backends(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        with serve(_handler(QWEN_LOADED)) as o:
            places({"ollama": o})
            body = client.get("/api/setup/detect", headers=auth).json()
        assert body["backends"] and all(isinstance(m, str) for b in body["backends"] for m in b["models"]), "unchanged for the old wizard"
        local = body["local"]
        assert local["target"] is None and local["reason"] is None and local["min_window"] == MIN
        assert [(f["app"], [m["id"] for m in f["models"] if m["usable"]]) for f in local["found"]] == [("ollama", ["qwen3:8b"])]

    def test_detect_with_nothing_running_says_refused(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        places({})
        local = client.get("/api/setup/detect", headers=auth).json()["local"]
        assert (local["found"], local["reason"]) == ([], "refused")

    def test_probe_needs_the_setup_credential(self, env_file, config_dir):
        app = create_setup_app(PairingState(code=CODE), api_port=8123, ws_port=8124, state=SetupModeState())
        resp = TestClient(app).post("/api/setup/probe", json={"target": "studio-mac"})
        assert resp.status_code == 401

    def test_probe_answers_the_shape(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        places({})
        with serve(_handler(lm_studio([{"id": "gemma-3-12b", "type": "llm", "state": "loaded", "loaded_context_length": 4096}]))) as p:
            resp = client.post("/api/setup/probe", json={"target": f"127.0.0.1:{p}"}, headers=auth)
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"target", "found", "reason", "detail", "min_window"}
        assert body["reason"] == "short_context"
        assert body["found"][0]["models"][0] == {"id": "gemma-3-12b", "loaded": True, "chat": True, "window": 4096,
                                                 "window_source": "loaded", "usable": False, "problem": "short_context"}
        assert MARKER not in resp.text

    def test_probe_refuses_a_bad_target(self, env_file, config_dir):
        client, auth = client_and_auth()
        resp = client.post("/api/setup/probe", json={"target": "user:pass@studio-mac"}, headers=auth)
        assert resp.status_code == 400 and resp.json()["error"] == "bad_target"


class TestConfigure:
    def _configure(self, client, auth, provider, url, model):
        return client.post("/api/setup/configure", json={"provider": provider, "base_url": url, "model": model}, headers=auth)

    def test_a_usable_ollama_choice_is_written(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        with serve(_handler(QWEN_LOADED)) as o:
            resp = self._configure(client, auth, "ollama", f"http://127.0.0.1:{o}", "qwen3:8b")
        assert resp.status_code == 200, resp.text
        cfg = yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert (cfg["model"]["provider"], cfg["model"]["model"]) == ("ollama", "qwen3:8b")

    def test_a_model_the_app_does_not_serve_is_refused(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        with serve(_handler(QWEN_LOADED)) as o:
            resp = self._configure(client, auth, "ollama", f"http://127.0.0.1:{o}", "llama3:70b")
        assert resp.status_code == 400 and resp.json()["error"] == "model_not_served"
        assert not (config_dir / "prometheus.yaml").exists()

    def test_an_unusable_choice_is_refused_with_the_reason(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        routes = lm_studio([{"id": "qwen3-14b", "type": "llm", "state": "not-loaded", "max_context_length": 131072}])
        with serve(_handler(routes)) as p:
            resp = self._configure(client, auth, "lm_studio", f"http://127.0.0.1:{p}", "qwen3-14b")
        assert resp.status_code == 400
        assert (resp.json()["error"], resp.json().get("reason")) == ("backend_unreachable", "none_loaded")
        assert not (config_dir / "prometheus.yaml").exists()

    def test_the_measured_window_is_the_budget_when_smaller(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        routes = lm_studio([{"id": "qwen3-14b", "type": "llm", "state": "loaded", "loaded_context_length": 20000}])
        with serve(_handler(routes)) as p:
            resp = self._configure(client, auth, "lm_studio", f"http://127.0.0.1:{p}", "qwen3-14b")
        assert resp.status_code == 200, resp.text
        cfg = yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["context"]["effective_limit"] == 20000

    def test_openai_compatible_is_a_provider_setup_takes(self, env_file, config_dir, places):
        client, auth = client_and_auth()
        with serve(_handler({"/v1/models": {"object": "list", "data": [{"id": "some-ai", "object": "model"}]}})) as p:
            resp = self._configure(client, auth, "openai_compatible", f"http://127.0.0.1:{p}", "some-ai")
        assert resp.status_code == 200, resp.text

    def test_an_old_client_naming_llama_cpp_for_a_plain_openai_answer_still_works(self, env_file, config_dir, places):
        """The old wizard labels every typed address llama_cpp. An answer that is only /v1/models is still accepted for it."""
        client, auth = client_and_auth()
        with serve(_handler({"/v1/models": {"object": "list", "data": [{"id": "gemma4-26b", "object": "model"}]}})) as p:
            resp = self._configure(client, auth, "llama_cpp", f"http://127.0.0.1:{p}", "gemma4-26b")
        assert resp.status_code == 200, resp.text

    def test_the_refusal_codes_are_in_the_catalog(self):
        from prometheus.web.setup_errors import CATALOG

        assert {"bad_target", "model_not_served"} <= set(CATALOG)
