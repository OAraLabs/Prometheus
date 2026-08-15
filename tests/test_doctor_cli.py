"""``prometheus doctor`` extended checks (Phase 0, item 4).

Server up/down is exercised against a real ephemeral HTTP server (same
pattern as test_cli_init); everything filesystem-shaped is confined to
tmp dirs via PROMETHEUS_CONFIG_DIR / PROMETHEUS_ENV_FILE.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml

from prometheus.cli.doctor import (
    check_config,
    check_dirs_writable,
    check_gateways,
    check_inference,
    check_coding_sandbox,
    check_token,
    check_trajectory_export,
    check_web_port,
    check_whisper,
    render_report,
    run_doctor_command,
)


class _ModelsHandler(BaseHTTPRequestHandler):
    payload: dict = {"data": [{"id": "test-model-7b"}]}

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        pass


class _EmptyModelsHandler(_ModelsHandler):
    payload = {"data": []}


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
        yield f"http://127.0.0.1:{port}", port
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "envfile"))
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)
    monkeypatch.delenv("PROMETHEUS_LOGS_DIR", raising=False)
    monkeypatch.delenv("PROMETHEUS_WORKSPACE_DIR", raising=False)
    return tmp_path


class TestCheckConfig:
    def test_missing_config_is_error(self, isolated_dirs):
        check, config = check_config(str(isolated_dirs / "nope.yaml"))
        assert check.status == "error"
        assert config == {}
        assert "prometheus setup" in (check.fix or "")

    def test_valid_config_parses(self, isolated_dirs):
        path = isolated_dirs / "ok.yaml"
        path.write_text("model:\n  provider: ollama\n", encoding="utf-8")
        check, config = check_config(str(path))
        assert check.status == "ok"
        assert config["model"]["provider"] == "ollama"

    def test_broken_yaml_is_error(self, isolated_dirs):
        path = isolated_dirs / "broken.yaml"
        path.write_text("model: [unclosed\n", encoding="utf-8")
        check, config = check_config(str(path))
        assert check.status == "error"
        assert config == {}


class TestCheckInference:
    def test_server_up_model_detected(self):
        with _serve(_ModelsHandler) as (url, _port):
            reach, model = check_inference(
                {"model": {"provider": "llama_cpp", "base_url": url}},
                timeout=3.0,
            )
        assert reach.status == "ok"
        assert model.status == "ok"
        assert "test-model-7b" in model.message

    def test_server_up_no_model_is_error(self):
        with _serve(_EmptyModelsHandler) as (url, _port):
            reach, model = check_inference(
                {"model": {"provider": "llama_cpp", "base_url": url}},
                timeout=3.0,
            )
        assert reach.status == "ok"
        assert model.status == "error"

    def test_server_down_is_error(self):
        url = f"http://127.0.0.1:{_free_port()}"
        reach, model = check_inference(
            {"model": {"provider": "llama_cpp", "base_url": url}},
            timeout=0.3,
        )
        assert reach.status == "error"
        assert model.status == "error"

    def test_cloud_provider_checks_key(self, monkeypatch):
        monkeypatch.delenv("TEST_DOCTOR_KEY", raising=False)
        cfg = {"model": {"provider": "anthropic",
                         "api_key_env": "TEST_DOCTOR_KEY", "model": "m"}}
        reach, _model = check_inference(cfg)
        assert reach.status == "error"
        monkeypatch.setenv("TEST_DOCTOR_KEY", "k")
        reach, model = check_inference(cfg)
        assert reach.status == "ok"
        assert model.status == "ok"


class TestCheckWebPort:
    def test_disabled_is_warning(self):
        assert check_web_port({"web": {"enabled": False}}).status == "warning"

    def test_free_port_is_ok(self):
        cfg = {"web": {"enabled": True, "api_port": _free_port()}}
        check = check_web_port(cfg)
        assert check.status == "ok"
        assert "free" in check.message

    def test_foreign_listener_is_error(self):
        # A plain TCP listener that never speaks HTTP — occupied port,
        # not Prometheus.
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            check = check_web_port(
                {"web": {"enabled": True, "api_port": port}}, timeout=0.5,
            )
        finally:
            srv.close()
        assert check.status == "error"


class TestCheckTokenAndDirs:
    def test_token_unset_warns(self, isolated_dirs):
        assert check_token({}).status == "warning"

    def test_token_set_ok(self, isolated_dirs, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "s3cr3t-value-xyz")
        check = check_token({})
        assert check.status == "ok"
        assert "s3cr3t-value-xyz" not in check.message  # never leak the token

    def test_dirs_writable(self, isolated_dirs):
        assert check_dirs_writable().status == "ok"


class TestCheckGateways:
    """SPRINT G3: per-gateway doctor lines (Telegram / Slack / Discord)."""

    def _by_name(self, config):
        return {c.name: c for c in check_gateways(config)}

    def test_all_disabled_render_info_lines(self, isolated_dirs, monkeypatch):
        for var in ("PROMETHEUS_TELEGRAM_TOKEN", "PROMETHEUS_SLACK_BOT_TOKEN",
                    "PROMETHEUS_SLACK_APP_TOKEN", "PROMETHEUS_DISCORD_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        checks = self._by_name({"gateway": {}})
        assert set(checks) == {
            "Telegram gateway", "Slack gateway", "Discord gateway"}
        for c in checks.values():
            assert c.status == "info"
            assert c.message == "not enabled"

    def test_telegram_enabled_without_token_is_error(
        self, isolated_dirs, monkeypatch,
    ):
        monkeypatch.delenv("PROMETHEUS_TELEGRAM_TOKEN", raising=False)
        c = self._by_name({"gateway": {"telegram_enabled": True}})[
            "Telegram gateway"]
        assert c.status == "error"
        assert "no bot token" in c.message
        assert "BotFather" in (c.fix or "")

    def test_telegram_token_from_env_file_counts(
        self, isolated_dirs, monkeypatch,
    ):
        monkeypatch.delenv("PROMETHEUS_TELEGRAM_TOKEN", raising=False)
        (isolated_dirs / "envfile").write_text(
            "PROMETHEUS_TELEGRAM_TOKEN=123:abc\n", encoding="utf-8")
        c = self._by_name({"gateway": {"telegram_enabled": True}})[
            "Telegram gateway"]
        # python-telegram-bot is a core dependency of the test env → ok.
        assert c.status == "ok"
        assert "token present" in c.message

    def test_slack_checks_both_tokens(self, isolated_dirs, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("PROMETHEUS_SLACK_APP_TOKEN", raising=False)
        # Only the bot token → error naming the missing app token.
        c = self._by_name({"gateway": {"slack": {
            "enabled": True, "bot_token": "xoxb-x"}}})["Slack gateway"]
        assert c.status == "error"
        assert "app token" in c.message
        assert "bot token" not in c.message
        # Neither → both named.
        c = self._by_name({"gateway": {"slack_enabled": True}})["Slack gateway"]
        assert c.status == "error"
        assert "bot token" in c.message and "app token" in c.message

    def test_slack_flat_keys_win_and_ok_with_library(
        self, isolated_dirs, monkeypatch,
    ):
        pytest.importorskip("slack_bolt")
        c = self._by_name({"gateway": {
            "slack_enabled": True,
            "slack_bot_token": "xoxb-x", "slack_app_token": "xapp-x",
        }})["Slack gateway"]
        assert c.status == "ok"
        assert "slack-bolt installed" in c.message

    def test_discord_enabled_without_token_is_error(
        self, isolated_dirs, monkeypatch,
    ):
        monkeypatch.delenv("PROMETHEUS_DISCORD_TOKEN", raising=False)
        c = self._by_name({"gateway": {"discord": {"enabled": True}}})[
            "Discord gateway"]
        assert c.status == "error"
        assert "no bot token" in c.message
        assert "PROMETHEUS_DISCORD_TOKEN" in (c.fix or "")

    def test_discord_enabled_with_token_and_library(
        self, isolated_dirs, monkeypatch,
    ):
        pytest.importorskip("discord")
        monkeypatch.setenv("PROMETHEUS_DISCORD_TOKEN", "tok")
        c = self._by_name({"gateway": {"discord": {"enabled": True}}})[
            "Discord gateway"]
        assert c.status == "ok"
        assert "discord.py installed" in c.message

    def test_enabled_without_library_suggests_extra(
        self, isolated_dirs, monkeypatch,
    ):
        monkeypatch.setenv("PROMETHEUS_DISCORD_TOKEN", "tok")
        monkeypatch.setattr(
            "prometheus.cli.doctor._library_installed", lambda mod: False)
        c = self._by_name({"gateway": {"discord": {"enabled": True}}})[
            "Discord gateway"]
        assert c.status == "error"
        assert "not installed" in c.message
        assert "oara-prometheus[discord]" in (c.fix or "")


class TestCheckWhisper:
    def test_voice_disabled_skips(self):
        assert check_whisper({}).status == "info"

    def test_voice_enabled_engine_missing_is_error(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.tools.builtin.whisper_stt._detect_whisper_engine",
            lambda: None,
        )
        check = check_whisper({"whisper": {"enabled": True}})
        assert check.status == "error"
        assert "oara-prometheus[voice]" in (check.fix or "")

    def test_voice_enabled_engine_present_is_ok(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.tools.builtin.whisper_stt._detect_whisper_engine",
            lambda: "faster-whisper",
        )
        assert check_whisper({"whisper": {"enabled": True}}).status == "ok"


class TestDoctorCommand:
    def test_healthy_config_exits_zero(self, isolated_dirs, capsys):
        with _serve(_ModelsHandler) as (url, _port):
            cfg_path = isolated_dirs / "prometheus.yaml"
            cfg_path.write_text(yaml.safe_dump({
                "model": {"provider": "llama_cpp", "base_url": url},
                "web": {"enabled": True, "api_port": _free_port()},
            }), encoding="utf-8")
            rc = run_doctor_command(argparse.Namespace(
                config=str(cfg_path), no_scan=True, timeout=3.0,
            ))
        out = capsys.readouterr().out
        assert rc == 0
        assert "✓" in out
        assert "RESULT" in out

    def test_unreachable_server_exits_nonzero(self, isolated_dirs, capsys):
        cfg_path = isolated_dirs / "prometheus.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "model": {"provider": "llama_cpp",
                      "base_url": f"http://127.0.0.1:{_free_port()}"},
            "web": {"enabled": True, "api_port": _free_port()},
        }), encoding="utf-8")
        rc = run_doctor_command(argparse.Namespace(
            config=str(cfg_path), no_scan=True, timeout=0.3,
        ))
        out = capsys.readouterr().out
        assert rc == 1
        assert "✗" in out

    def test_missing_config_exits_nonzero(self, isolated_dirs):
        rc = run_doctor_command(argparse.Namespace(
            config=str(isolated_dirs / "absent.yaml"), no_scan=True, timeout=0.3,
        ))
        assert rc == 1

    def test_report_includes_fix_lines(self, isolated_dirs, capsys):
        from prometheus.infra.doctor import DiagnosticCheck
        report = render_report([
            DiagnosticCheck(name="Thing", category="platform",
                            status="error", message="broken", fix="do the fix"),
        ])
        assert "✗ Thing: broken" in report
        assert "fix: do the fix" in report


class TestTrajectoryExportCheck:
    """`doctor` must distinguish off-on-purpose from off-by-accident.

    Nothing breaks when golden-trace export is disabled — the traces simply
    stop accumulating, and the cost only shows up much later as an absent
    training corpus. Before this row the two states were indistinguishable
    from outside, separated only by a yaml comment nobody greps.
    """

    def _seed(self, tmp_path, n_golden: int):
        """A telemetry DB with *n_golden* golden rows, at the real path."""
        import os
        from prometheus.telemetry.tracker import ToolCallTelemetry

        home = tmp_path / "home"
        (home / ".prometheus").mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(home)
        tel = ToolCallTelemetry(db_path=home / ".prometheus" / "telemetry.db")
        for i in range(n_golden):
            tel.record(
                model="m", tool_name="bash", success=True, retries=0,
                raw_model_output="prose",
                parsed_tool_call='{"name":"bash","input":{}}',
                provider="anthropic", session_id=f"s{i}",
            )
        return home

    @pytest.fixture(autouse=True)
    def _restore_home(self):
        import os

        original = os.environ.get("HOME")
        yield
        if original is not None:
            os.environ["HOME"] = original

    def test_disabled_with_stranded_traces_warns(self, tmp_path):
        """The forgotten-flag case: traces piling up, nothing exporting."""
        home = self._seed(tmp_path, 5)
        check = check_trajectory_export({
            "trajectory_export": {
                "enabled": False,
                "output_dir": str(home / ".prometheus" / "trajectories"),
            }
        })
        assert check.status == "warning"
        assert "DISABLED" in check.message
        assert "5 golden trace" in check.message
        assert check.fix

    def test_disabled_with_nothing_stranded_is_ok(self, tmp_path):
        """The deliberate case: off, and nothing is being lost by it. The
        COUNT is what separates this from the row above — the flag alone
        cannot, which is the whole reason this check reports a consequence."""
        home = self._seed(tmp_path, 0)
        check = check_trajectory_export({
            "trajectory_export": {
                "enabled": False,
                "output_dir": str(home / ".prometheus" / "trajectories"),
            }
        })
        assert check.status == "ok"
        assert "DISABLED" in check.message

    def test_enabled_reports_files_and_backlog(self, tmp_path):
        home = self._seed(tmp_path, 3)
        out_dir = home / ".prometheus" / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "golden_traces_1_1.jsonl").write_text("{}\n")
        check = check_trajectory_export({
            "trajectory_export": {"enabled": True, "output_dir": str(out_dir)}
        })
        assert check.status == "ok"
        assert "enabled" in check.message
        assert "1 export file" in check.message

    def test_watermark_is_respected(self, tmp_path):
        """Already-exported traces are not counted as stranded."""
        from prometheus.sentinel.golden_trace_exporter import WATERMARK_FILENAME

        home = self._seed(tmp_path, 4)
        out_dir = home / ".prometheus" / "trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / WATERMARK_FILENAME).write_text(
            json.dumps({"last_rowid": 4})
        )
        check = check_trajectory_export({
            "trajectory_export": {
                "enabled": False, "output_dir": str(out_dir),
            }
        })
        assert check.status == "ok", "exported traces counted as stranded"

    def test_missing_db_does_not_crash_the_doctor(self, tmp_path):
        """Diagnostics degrade; they never raise."""
        import os

        os.environ["HOME"] = str(tmp_path / "empty")
        check = check_trajectory_export({"trajectory_export": {"enabled": False}})
        assert check.status in ("ok", "warning")
        assert "unknown" in check.message

    def test_absent_config_defaults_to_enabled(self, tmp_path):
        self._seed(tmp_path, 0)
        assert check_trajectory_export({}).status == "ok"


class TestCodingSandboxCheck:
    """`doctor` must answer "which backend, and can it start?" BEFORE a run.

    create_sandbox() raises rather than degrading when a backend is
    unavailable — right behaviour, but it raises when a coding run starts, so
    an operator who selects bwrap or docker discovers the problem mid-task.
    Both carry a system dependency nothing else announces.
    """

    def _check(self, **coding):
        return check_coding_sandbox({"coding": coding})

    def test_process_backend_states_its_limit(self):
        """'available' is not 'contains'. ProcessSandbox starts anywhere and
        a one-line shell redirect leaves it — saying so is the point."""
        c = self._check(enabled=True, sandbox_type="process")
        assert c.status == "ok"
        assert "process" in c.message
        assert "redirect" in c.message

    def test_default_backend_is_process(self):
        assert "process" in check_coding_sandbox({}).message

    def test_unknown_backend_is_an_error(self):
        c = self._check(enabled=True, sandbox_type="nonsense")
        assert c.status == "error"
        assert c.fix and "process" in c.fix

    def test_unavailable_backend_errors_when_coding_is_on(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.coding.sandbox.docker_available", lambda: False
        )
        c = self._check(enabled=True, sandbox_type="docker")
        assert c.status == "error"
        assert "NOT available" in c.message
        assert "will fail to start" in c.message

    def test_severity_ignores_the_inert_coding_enabled_flag(self, monkeypatch):
        """`coding.enabled` is documented and defaults to false, but NOTHING
        reads it — run_coding_task() builds a sandbox regardless. Gating this
        row on it would report "nothing invokes this yet" about a backend one
        command away from failing."""
        monkeypatch.setattr(
            "prometheus.coding.sandbox.docker_available", lambda: False
        )
        for flag in (True, False):
            c = self._check(enabled=flag, sandbox_type="docker")
            assert c.status == "error", f"enabled={flag} changed the verdict"
            assert "will fail to start" in c.message

    def test_available_docker_is_ok(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.coding.sandbox.docker_available", lambda: True
        )
        c = self._check(enabled=True, sandbox_type="docker")
        assert c.status == "ok"
        assert "reachable" in c.message

    def test_bwrap_missing_binary_names_the_package(self, monkeypatch):
        """The whole onboarding gap: bubblewrap is not a Python dependency,
        so the fix has to say what to install."""
        monkeypatch.setattr(
            "prometheus.coding.sandbox.BwrapSandbox.is_available",
            staticmethod(lambda: False),
        )
        c = self._check(enabled=True, sandbox_type="bwrap")
        assert c.status == "error"
        assert "bubblewrap" in c.message

    def test_bwrap_present_but_blocked_reports_the_host_reason(self, monkeypatch):
        """Installed is not working. When the kernel refuses the namespace,
        the self-check's own detail must reach the operator rather than a
        generic 'unavailable'."""
        from prometheus.coding.sandbox import BwrapSelfCheck

        monkeypatch.setattr(
            "prometheus.coding.sandbox.BwrapSandbox.is_available",
            staticmethod(lambda: True),
        )
        monkeypatch.setattr(
            "prometheus.coding.sandbox.BwrapSandbox.self_check",
            staticmethod(lambda: BwrapSelfCheck(False, "uid map denied", False, False)),
        )
        c = self._check(enabled=True, sandbox_type="bwrap")
        assert c.status == "error"
        assert "uid map denied" in c.message

    def test_backend_probe_failure_never_raises(self, monkeypatch):
        """Diagnostics degrade; they never crash the doctor."""
        def _boom():
            raise RuntimeError("docker exploded")

        monkeypatch.setattr(
            "prometheus.coding.sandbox.docker_available", _boom
        )
        c = self._check(enabled=True, sandbox_type="docker")
        assert c.status == "error"
        assert "could not be checked" in c.message
