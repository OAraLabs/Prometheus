"""Setup-mode remote-drivable setup API (Onboarding Phase 2, Part A).

Covers the authed ``/api/setup/detect|configure|complete`` surface and
the setup→real-daemon in-process flip:

- auth: every mutation 401s unauthed / pre-pair; works with the paired token
- detect: fake JSON backend found via ``?base_url=``; HTML dashboard rejected
- configure: writes the SAME yaml the CLI wizard's fast path writes
  (model/web sections byte-compared), generates identity, idempotent
- status.configured transitions false → true
- complete: 409 before configure; 200 + restart flag after
- subprocess: no-config daemon → pair → configure → complete → the REAL
  daemon boots in the SAME process and answers /api/status with the token
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.api_token import TOKEN_ENV_VAR  # noqa: E402
from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.web.setup_server import (  # noqa: E402
    SETUP_COMPLETE,
    PairingState,
    SetupModeState,
    create_setup_app,
)

CODE = "042999"


# ---------------------------------------------------------------------------
# Fixtures — isolated env file + config dir + fake inference backends
# ---------------------------------------------------------------------------


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Isolate the env file + token env var — never the real machine."""
    import os

    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    yield path
    os.environ.pop(TOKEN_ENV_VAR, None)


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Isolate the config dir; cwd moved off the repo (no repo-local config).

    The directory is NOT created here — the no-state rule says only an
    explicit configure call may create it.
    """
    confdir = tmp_path / "confdir"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(confdir))
    monkeypatch.chdir(tmp_path)
    return confdir


class _FakeLlamaCppHandler(BaseHTTPRequestHandler):
    """OpenAI-shaped /v1/models payload (same shape as test_cli_init)."""

    def do_GET(self):  # noqa: N802 — required name
        if self.path == "/v1/models":
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": "gemma4-26b", "object": "model"},
                    {"id": "qwen3.5-32b", "object": "model"},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args, **kwargs):  # silence test output
        pass


class _HtmlDashboardHandler(BaseHTTPRequestHandler):
    """200 + HTML on every path — must NOT count as an inference server."""

    def do_GET(self):  # noqa: N802
        body = b"<!doctype html><html><body>dashboard</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _serve(handler_cls):
    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def make_client(
    state: SetupModeState | None = None, on_complete=None,
    api_port: int = 8123, ws_port: int = 8124,
):
    pairing = PairingState(code=CODE)
    state = state or SetupModeState()
    app = create_setup_app(
        pairing, api_port=api_port, ws_port=ws_port, state=state,
        on_complete=on_complete,
    )
    return TestClient(app), state


def pair(client) -> dict[str, str]:
    """Pair and return the Authorization header dict."""
    token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth — every mutation requires the paired token
# ---------------------------------------------------------------------------


class TestSetupMutationAuth:
    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/setup/detect"),
        ("POST", "/api/setup/configure"),
        ("POST", "/api/setup/complete"),
    ])
    def test_pre_pair_mutations_are_401_not_paired(
        self, env_file, config_dir, method, path,
    ):
        client, _ = make_client()
        resp = client.request(method, path)
        assert resp.status_code == 401
        assert resp.json()["error"] == "not_paired"
        # And nothing was created (no-state rule).
        assert not config_dir.exists()

    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/setup/detect"),
        ("POST", "/api/setup/configure"),
        ("POST", "/api/setup/complete"),
    ])
    def test_wrong_token_after_pair_is_401(
        self, env_file, config_dir, method, path,
    ):
        client, _ = make_client()
        pair(client)
        resp = client.request(
            method, path, headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"

    def test_status_stays_open_it_is_the_discovery_probe(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        assert client.get("/api/setup/status").status_code == 200
        pair(client)
        assert client.get("/api/setup/status").status_code == 200


# ---------------------------------------------------------------------------
# GET /api/setup/detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_probes_one_custom_base_url(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = client.get(
                "/api/setup/detect", params={"base_url": url}, headers=headers,
            )
        assert resp.status_code == 200
        backends = resp.json()["backends"]
        assert len(backends) == 1
        b = backends[0]
        assert b["provider"] == "llama_cpp"
        assert b["base_url"] == url
        assert "gemma4-26b" in b["models"]
        assert b["latency_ms"] > 0

    def test_html_dashboard_is_rejected(self, env_file, config_dir):
        """The Phase 0 JSON-shape hardening applies over the wire too."""
        client, _ = make_client()
        headers = pair(client)
        with _serve(_HtmlDashboardHandler) as url:
            resp = client.get(
                "/api/setup/detect", params={"base_url": url}, headers=headers,
            )
        assert resp.status_code == 200
        assert resp.json()["backends"] == []

    def test_unreachable_base_url_is_empty_not_error(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        resp = client.get(
            "/api/setup/detect",
            params={"base_url": "http://127.0.0.1:9"},  # discard port — closed
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["backends"] == []

    def test_non_http_base_url_is_400(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        resp = client.get(
            "/api/setup/detect", params={"base_url": "gopher://oops"},
            headers=headers,
        )
        assert resp.status_code == 400
        assert resp.json()["error"] == "bad_base_url"


# ---------------------------------------------------------------------------
# POST /api/setup/configure
# ---------------------------------------------------------------------------


class TestConfigure:
    def _configure(self, client, headers, url, **extra):
        body = {"provider": "llama_cpp", "base_url": url,
                "model": "gemma4-26b", **extra}
        return client.post("/api/setup/configure", json=body, headers=headers)

    def test_writes_the_same_yaml_as_the_cli_wizard(
        self, env_file, config_dir, tmp_path,
    ):
        """The API and `prometheus setup --fast` share ONE writer — the
        model/web sections must be byte-identical."""
        from prometheus.cli.init import run_init

        # Default ports here: in production setup mode serves on the
        # default 8005/8010 unless overridden, so the written web section
        # is identical to the CLI wizard's (nothing binds in TestClient).
        client, _ = make_client(api_port=8005, ws_port=8010)
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(client, headers, url)
            assert resp.status_code == 200

            cli_dir = tmp_path / "cli-out"
            run_init(
                noninteractive=True, target_dir=cli_dir,
                candidates=[{"name": "llama.cpp", "url": url,
                             "models_path": "/v1/models",
                             "provider": "llama_cpp"}],
            )

        api_cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        cli_cfg = yaml.safe_load(
            (cli_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        for section in ("model", "web"):
            api_bytes = yaml.safe_dump(api_cfg[section], sort_keys=False)
            cli_bytes = yaml.safe_dump(cli_cfg[section], sort_keys=False)
            assert api_bytes == cli_bytes, f"{section} section diverged"
        assert api_cfg["web"] == {"enabled": True, "api_port": 8005,
                                  "ws_port": 8010}

    def test_summary_shape_and_no_token_leak(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(
                client, headers, url,
                agent_name="TestAgent", persona="calm and precise",
                telegram_token="123456:SECRET-TELEGRAM",
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["configured"] is True
        assert body["provider"] == "llama_cpp"
        assert body["model"] == "gemma4-26b"
        assert body["agent_name"] == "TestAgent"
        assert body["telegram_token_saved"] is True
        assert body["web"]["enabled"] is True
        assert "SECRET-TELEGRAM" not in resp.text
        assert TOKEN_ENV_VAR not in resp.text
        # Telegram token persisted to the env file, not the yaml.
        assert parse_env_file(env_file)["PROMETHEUS_TELEGRAM_TOKEN"] == \
            "123456:SECRET-TELEGRAM"
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["gateway"]["telegram_enabled"] is True
        assert "SECRET-TELEGRAM" not in \
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8")

    def test_identity_generated_with_agent_name_and_persona(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(
                client, headers, url,
                agent_name="TestAgent", persona="dry wit, no filler",
            )
        assert resp.status_code == 200
        soul = (config_dir / "SOUL.md").read_text(encoding="utf-8")
        assert "You are **TestAgent**" in soul
        assert "## Persona" in soul
        assert "dry wit, no filler" in soul
        assert (config_dir / "AGENTS.md").is_file()
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["system"]["name"] == "TestAgent"

    def test_no_identity_without_agent_name(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(client, headers, url)
        assert resp.status_code == 200
        assert resp.json()["identity"] is None
        assert not (config_dir / "SOUL.md").exists()

    def test_unreachable_backend_is_rejected_nothing_written(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        resp = self._configure(client, headers, "http://127.0.0.1:9")
        assert resp.status_code == 400
        assert resp.json()["error"] == "backend_unreachable"
        assert not (config_dir / "prometheus.yaml").exists()

    def test_html_backend_is_rejected(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_HtmlDashboardHandler) as url:
            resp = self._configure(client, headers, url)
        assert resp.status_code == 400
        assert resp.json()["error"] == "backend_unreachable"

    def test_missing_fields_400(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        resp = client.post(
            "/api/setup/configure", json={"provider": "ollama"},
            headers=headers,
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "base_url" in detail and "model" in detail

    def test_idempotent_re_configure_overwrites_cleanly(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            assert self._configure(
                client, headers, url, agent_name="FirstName",
            ).status_code == 200
            assert self._configure(
                client, headers, url, model="qwen3.5-32b",
                agent_name="SecondName",
            ).status_code == 200
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["model"]["model"] == "qwen3.5-32b"
        assert cfg["system"]["name"] == "SecondName"
        soul = (config_dir / "SOUL.md").read_text(encoding="utf-8")
        assert "SecondName" in soul and "FirstName" not in soul
        # No backup litter from the overwrite.
        assert list(config_dir.glob("prometheus.yaml.backup-*")) == []


# ---------------------------------------------------------------------------
# SPRINT G3 — gateway fields on POST /api/setup/configure
# ---------------------------------------------------------------------------


class TestConfigureGateways:
    def _configure(self, client, headers, url, **extra):
        body = {"provider": "llama_cpp", "base_url": url,
                "model": "gemma4-26b", **extra}
        return client.post("/api/setup/configure", json=body, headers=headers)

    def test_all_three_gateways_written(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(
                client, headers, url,
                telegram_token="123456:SECRET-TG",
                slack_bot_token="xoxb-SECRET-BOT",
                slack_app_token="xapp-SECRET-APP",
                slack_channels=["C0123", "C0456"],
                discord_token="SECRET-DISCORD",
                discord_guild_ids=[123456789012345678],
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["gateways_enabled"] == ["telegram", "slack", "discord"]
        assert body["telegram_token_saved"] is True

        # Env file carries all four token vars…
        env = parse_env_file(env_file)
        assert env["PROMETHEUS_TELEGRAM_TOKEN"] == "123456:SECRET-TG"
        assert env["PROMETHEUS_SLACK_BOT_TOKEN"] == "xoxb-SECRET-BOT"
        assert env["PROMETHEUS_SLACK_APP_TOKEN"] == "xapp-SECRET-APP"
        assert env["PROMETHEUS_DISCORD_TOKEN"] == "SECRET-DISCORD"

        # …the yaml has the enabled blocks + whitelists (never tokens)…
        cfg_text = (config_dir / "prometheus.yaml").read_text(encoding="utf-8")
        cfg = yaml.safe_load(cfg_text)
        assert cfg["gateway"]["telegram_enabled"] is True
        assert cfg["gateway"]["slack"]["enabled"] is True
        assert cfg["gateway"]["slack"]["allowed_channels"] == ["C0123", "C0456"]
        assert cfg["gateway"]["discord"]["enabled"] is True
        assert cfg["gateway"]["discord"]["guild_ids"] == [123456789012345678]
        for secret in ("SECRET-TG", "SECRET-BOT", "SECRET-APP", "SECRET-DISCORD"):
            assert secret not in cfg_text

        # …and the response never echoes a token.
        for secret in ("SECRET-TG", "SECRET-BOT", "SECRET-APP", "SECRET-DISCORD"):
            assert secret not in resp.text

    @pytest.mark.parametrize("half", [
        {"slack_bot_token": "xoxb-only"},
        {"slack_app_token": "xapp-only"},
    ])
    def test_slack_half_token_is_400_nothing_written(
        self, env_file, config_dir, half,
    ):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(client, headers, url, **half)
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"] == "slack_token_pair_incomplete"
        assert "BOTH" in body["detail"]
        assert not (config_dir / "prometheus.yaml").exists()
        assert "PROMETHEUS_SLACK" not in (
            env_file.read_text(encoding="utf-8") if env_file.exists() else "")

    def test_discord_guild_ids_accept_strings_and_ints(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(
                client, headers, url,
                discord_token="tok", discord_guild_ids=["111", 222],
            )
        assert resp.status_code == 200
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["gateway"]["discord"]["guild_ids"] == [111, 222]

    def test_bad_discord_guild_ids_is_400(self, env_file, config_dir):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(
                client, headers, url,
                discord_token="tok", discord_guild_ids=["not-a-guild"],
            )
        assert resp.status_code == 400
        assert resp.json()["error"] == "bad_discord_guild_ids"

    def test_no_gateways_means_empty_list_and_disabled_blocks(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            resp = self._configure(client, headers, url)
        assert resp.status_code == 200
        body = resp.json()
        assert body["gateways_enabled"] == []
        assert body["telegram_token_saved"] is False
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["gateway"]["telegram_enabled"] is False
        assert cfg["gateway"]["slack"]["enabled"] is False
        assert cfg["gateway"]["discord"]["enabled"] is False

    def test_idempotent_re_post_drops_removed_gateways(
        self, env_file, config_dir,
    ):
        """A re-POST is the WHOLE answer: gateways omitted the second time
        are disabled in the freshly written yaml (env-file tokens remain,
        but they are inert without the enabled flag)."""
        client, _ = make_client()
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            assert self._configure(
                client, headers, url,
                discord_token="tok-1", discord_guild_ids=[1],
            ).status_code == 200
            resp = self._configure(client, headers, url,
                                   telegram_token="123:tg")
        assert resp.status_code == 200
        assert resp.json()["gateways_enabled"] == ["telegram"]
        cfg = yaml.safe_load(
            (config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
        assert cfg["gateway"]["telegram_enabled"] is True
        assert cfg["gateway"]["discord"]["enabled"] is False
        # No backup litter from the overwrite (idempotency contract).
        assert list(config_dir.glob("prometheus.yaml.backup-*")) == []


# ---------------------------------------------------------------------------
# status.configured + POST /api/setup/complete
# ---------------------------------------------------------------------------


class TestConfiguredTransitionsAndComplete:
    def test_status_configured_flips_after_configure(
        self, env_file, config_dir,
    ):
        client, _ = make_client()
        headers = pair(client)
        assert client.get("/api/setup/status").json()["configured"] is False
        with _serve(_FakeLlamaCppHandler) as url:
            client.post("/api/setup/configure", json={
                "provider": "llama_cpp", "base_url": url,
                "model": "gemma4-26b",
            }, headers=headers)
        assert client.get("/api/setup/status").json()["configured"] is True

    def test_complete_before_configure_is_409(self, env_file, config_dir):
        client, state = make_client()
        headers = pair(client)
        resp = client.post("/api/setup/complete", headers=headers)
        assert resp.status_code == 409
        assert resp.json()["error"] == "not_configured"
        assert state.restart_requested is False

    def test_complete_verifies_parses_and_requests_restart(
        self, env_file, config_dir,
    ):
        stopped: list[bool] = []
        state = SetupModeState()
        client, state = make_client(state=state, on_complete=lambda: stopped.append(True))
        headers = pair(client)
        with _serve(_FakeLlamaCppHandler) as url:
            client.post("/api/setup/configure", json={
                "provider": "llama_cpp", "base_url": url,
                "model": "gemma4-26b",
            }, headers=headers)
        resp = client.post("/api/setup/complete", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["restarting"] is True
        assert state.restart_requested is True
        # The stop callback ran AFTER the response (background task).
        assert stopped == [True]

    def test_complete_with_garbage_config_is_409(self, env_file, config_dir):
        client, state = make_client()
        headers = pair(client)
        config_dir.mkdir(parents=True)
        (config_dir / "prometheus.yaml").write_text(
            ":\nnot yaml: [unclosed", encoding="utf-8")
        resp = client.post("/api/setup/complete", headers=headers)
        assert resp.status_code == 409
        assert state.restart_requested is False


# ---------------------------------------------------------------------------
# The in-process flip — subprocess: setup mode → real daemon, one process
# ---------------------------------------------------------------------------


class TestSetupCompleteFallthrough:
    """`prometheus daemon` with no config pairs, configures, completes —
    and the SAME process then serves the real daemon."""

    def test_full_flow_boots_real_daemon_in_process(self, tmp_path):
        import os
        import re
        import subprocess
        import sys
        import time
        import urllib.error
        import urllib.request

        confdir = tmp_path / "confdir"
        api_port = _free_port()
        ws_port = _free_port()
        env = dict(os.environ)
        env.update({
            "PROMETHEUS_CONFIG_DIR": str(confdir),
            "PROMETHEUS_ENV_FILE": str(tmp_path / "env"),
            "PROMETHEUS_WEB_API_PORT": str(api_port),
            "PROMETHEUS_WEB_WS_PORT": str(ws_port),
        })
        env.pop("PROMETHEUS_API_TOKEN", None)

        out_path = tmp_path / "daemon.out"

        def api(path, *, method="GET", payload=None, token=None, timeout=5):
            req = urllib.request.Request(
                f"http://127.0.0.1:{api_port}{path}", method=method,
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            data = None
            if payload is not None:
                data = json.dumps(payload).encode()
                req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, data, timeout=timeout) as r:
                    return r.status, json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode() or "{}")

        with _serve(_FakeLlamaCppHandler) as backend_url, \
                out_path.open("wb") as out:
            proc = subprocess.Popen(
                [sys.executable, "-m", "prometheus.daemon"],
                stdout=out, stderr=subprocess.STDOUT,
                cwd=str(tmp_path), env=env,
            )
            try:
                # 1. Banner → pairing code.
                code = None
                deadline = time.time() + 30
                while time.time() < deadline and code is None:
                    m = re.search(
                        r"^\s{4}(\d{6})\s*$",
                        out_path.read_text(errors="replace"), re.M,
                    )
                    if m:
                        code = m.group(1)
                    else:
                        time.sleep(0.3)
                assert code, f"no pairing code in output:\n{out_path.read_text(errors='replace')}"

                # The banner prints before uvicorn binds — wait for the
                # setup server to actually answer.
                deadline = time.time() + 30
                up = False
                while time.time() < deadline:
                    try:
                        status, _body = api("/api/setup/status", timeout=2)
                        if status == 200:
                            up = True
                            break
                    except (urllib.error.URLError, ConnectionError, OSError,
                            TimeoutError):
                        time.sleep(0.3)
                assert up, "setup-mode server never came up"

                # No state was created just by booting setup mode.
                assert not confdir.exists()

                # 2. Pair → token.
                status, body = api("/api/setup/pair", method="POST",
                                   payload={"code": code})
                assert status == 200, body
                token = body["token"]

                # 3. Unauthed configure → 401 (acceptance item 5).
                status, body = api("/api/setup/configure", method="POST",
                                   payload={})
                assert status == 401

                # 4. Detect the fake backend, then configure + complete.
                status, body = api(
                    f"/api/setup/detect?base_url={backend_url}", token=token)
                assert status == 200 and len(body["backends"]) == 1
                status, body = api("/api/setup/configure", method="POST",
                                   token=token, payload={
                                       "provider": "llama_cpp",
                                       "base_url": backend_url,
                                       "model": "gemma4-26b",
                                       "agent_name": "TestAgent",
                                   })
                assert status == 200, body
                status, body = api("/api/setup/status")
                assert body["configured"] is True
                status, body = api("/api/setup/complete", method="POST",
                                   token=token)
                assert status == 200 and body["restarting"] is True

                # 5. The SAME process now boots the real daemon. Poll
                # /api/status with the paired token until it answers 200
                # (setup mode answered 403 on that path).
                deadline = time.time() + 120
                real_up = False
                while time.time() < deadline:
                    assert proc.poll() is None, (
                        "daemon exited instead of falling through:\n"
                        + out_path.read_text(errors="replace")
                    )
                    try:
                        status, body = api("/api/status", token=token,
                                           timeout=3)
                    except (urllib.error.URLError, ConnectionError, OSError,
                            TimeoutError, json.JSONDecodeError):
                        time.sleep(1.0)
                        continue
                    if status == 200:
                        real_up = True
                        break
                    time.sleep(1.0)
                assert real_up, (
                    "real daemon never answered /api/status 200:\n"
                    + out_path.read_text(errors="replace")[-4000:]
                )

                # 6. The artifacts the wizard promised.
                cfg = yaml.safe_load(
                    (confdir / "prometheus.yaml").read_text(encoding="utf-8"))
                assert cfg["web"]["enabled"] is True
                assert cfg["web"]["api_port"] == api_port
                assert cfg["model"]["model"] == "gemma4-26b"
                soul = (confdir / "SOUL.md").read_text(encoding="utf-8")
                assert "TestAgent" in soul
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Sentinel constant — daemon.main's fallthrough contract
# ---------------------------------------------------------------------------


def test_setup_complete_sentinel_is_stable():
    """daemon.main compares against this value — it is a wire-ish contract."""
    assert SETUP_COMPLETE == "setup-complete"
