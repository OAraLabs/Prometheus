"""`oara search setup`: SearXNG in Docker, JSON on, and web_search.searxng_url written.

Nothing here runs Docker or touches the network. The setup logic is driven
through ``FakeDocker`` (the same methods as ``search.Docker``), and the HTTP
readiness probe through an ``httpx.MockTransport``. The real ``Docker``
wrapper is exercised once, against a fake ``docker`` executable on PATH.

What is pinned:

* no Docker (or a Docker whose daemon does not answer) explains and changes
  nothing: no config write, no backup, no settings file;
* a fresh run writes a settings.yml with JSON on, the limiter off and a random
  secret_key, starts searxng/searxng with --restart unless-stopped on the
  chosen port, waits until a test query answers JSON, backs the config up and
  writes web_search.searxng_url, leaving every comment and other key alone;
* a re-run finds the container and changes nothing;
* --dry-run says what it would do and does none of it;
* JSON disabled, or no answer in time, leaves the config alone.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
import yaml

from prometheus.cli import search
from prometheus.tools.builtin import web_search_backends as wsb

CONFIG_TEXT = textwrap.dedent("""\
    # my config — hand-written comments must survive
    model:
      provider: llama_cpp   # the 4090
      base_url: http://localhost:8080

    web_search:
      backends: [searxng, brave, duckduckgo]
      searxng_url: ""       # set by oara search setup

    security:
      trust_level: 2
    """)

SEARX_JSON = json.dumps({"query": "test", "results": [], "unresponsive_engines": []})


class FakeDocker:
    """Stands in for ``search.Docker``; records every call."""

    def __init__(
        self, *, version: str | None = "27.1.1", error: str = "",
        container: dict[str, Any] | None = None, run_error: str = "",
    ) -> None:
        self.version = version
        self.error = error
        self.container = container
        self.run_error = run_error
        self.calls: list[tuple[str, ...]] = []

    def daemon_version(self) -> tuple[str | None, str]:
        self.calls.append(("info",))
        return self.version, self.error

    def inspect(self, name: str) -> dict[str, Any] | None:
        self.calls.append(("inspect", name))
        return self.container

    def run_detached(self, argv: list[str]) -> str:
        self.calls.append(("run", *argv))
        if self.run_error:
            raise search.DockerError(self.run_error)
        return "c0ffee"

    def start(self, name: str) -> None:
        self.calls.append(("start", name))

    @property
    def changing_calls(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[0] in ("run", "start")]


def _container(*, running: bool = True, port: str = "8888",
               image: str = "searxng/searxng") -> dict[str, Any]:
    return {
        "Name": "/searxng",
        "Config": {"Image": image},
        "State": {"Running": running, "Status": "running" if running else "exited"},
        "HostConfig": {"PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": port}]}},
    }


def _probe(*answers: Callable[[httpx.Request], httpx.Response]) -> tuple[httpx.MockTransport, list]:
    """Each request takes the next answer; the last one repeats."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = answers[min(len(seen) - 1, len(answers) - 1)]
        return answer(request)

    return httpx.MockTransport(handler), seen


def _json_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=SEARX_JSON, headers={"content-type": "application/json"})


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _forbidden(request: httpx.Request) -> httpx.Response:
    return httpx.Response(403, text="<title>403 Forbidden</title>")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "prometheus.yaml"
    path.write_text(CONFIG_TEXT)
    return path


def _setup(
    config: Path, docker: FakeDocker | None, transport: httpx.MockTransport | None = None,
    **opts: Any,
) -> tuple[int, str]:
    lines: list[str] = []
    clock = FakeClock()
    options = search.SetupOptions(
        config_path=str(config), settings_dir=config.parent / "searxng", **opts,
    )
    code = search.run_setup(
        options, docker=docker, transport=transport or _probe(_json_ok)[0],
        out=lines.append, sleep=clock.sleep, clock=clock.monotonic,
    )
    return code, "\n".join(lines)


def _backups(config: Path) -> list[Path]:
    return sorted(p for p in config.parent.iterdir() if ".bak" in p.name)


# ---------------------------------------------------------------------------
# No Docker: explain, change nothing
# ---------------------------------------------------------------------------


class TestNoDocker:
    def test_no_docker_binary_explains_and_changes_nothing(self, config: Path) -> None:
        code, out = _setup(config, None)
        assert code == 1
        assert "Docker" in out and "not installed" in out
        assert "searxng_url" in out  # the manual route
        assert "Nothing was changed" in out
        assert config.read_text() == CONFIG_TEXT
        assert _backups(config) == []
        assert not (config.parent / "searxng").exists()

    def test_a_docker_whose_daemon_is_down_changes_nothing(self, config: Path) -> None:
        docker = FakeDocker(version=None, error="Cannot connect to the Docker daemon")
        code, out = _setup(config, docker)
        assert code == 1
        assert "Cannot connect to the Docker daemon" in out
        assert "Nothing was changed" in out
        assert docker.changing_calls == []
        assert config.read_text() == CONFIG_TEXT
        assert not (config.parent / "searxng").exists()

    def test_no_config_file_is_refused_before_docker_is_touched(self, tmp_path: Path) -> None:
        docker = FakeDocker()
        code, out = _setup(tmp_path / "absent.yaml", docker)
        assert code == 1
        assert "oara setup" in out
        assert docker.calls == []
        assert not (tmp_path / "searxng").exists()


# ---------------------------------------------------------------------------
# A fresh run
# ---------------------------------------------------------------------------


class TestFreshRun:
    def test_it_starts_searxng_waits_for_json_and_writes_the_url(self, config: Path) -> None:
        docker = FakeDocker(container=None)
        transport, seen = _probe(_refused, _refused, _json_ok)
        code, out = _setup(config, docker, transport)
        assert code == 0, out

        run = next(c for c in docker.calls if c[0] == "run")
        argv = list(run[1:])
        settings_dir = config.parent / "searxng"
        assert argv == [
            "--name", "searxng", "--restart", "unless-stopped",
            "-p", "127.0.0.1:8888:8080",
            "-e", "FORCE_OWNERSHIP=false",
            "-v", f"{settings_dir}:/etc/searxng",
            "searxng/searxng",
        ]

        # Waited until a test query answered JSON.
        assert len(seen) == 3
        assert seen[-1].url.params["format"] == "json"
        assert str(seen[-1].url).startswith("http://127.0.0.1:8888/search")

        written = yaml.safe_load(config.read_text())
        assert written["web_search"]["searxng_url"] == "http://127.0.0.1:8888"
        assert "web_search.searxng_url" in out

    def test_the_generated_settings_turn_json_on_and_the_limiter_off(self, config: Path) -> None:
        _setup(config, FakeDocker(container=None))
        settings_path = config.parent / "searxng" / "settings.yml"
        settings = yaml.safe_load(settings_path.read_text())
        assert settings["use_default_settings"] is True
        assert settings["search"]["formats"] == ["html", "json"]
        assert settings["server"]["limiter"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", settings["server"]["secret_key"])
        # The container's user must be able to read it.
        assert stat.S_IMODE(settings_path.stat().st_mode) & stat.S_IROTH

    def test_every_run_gets_its_own_secret(self, tmp_path: Path) -> None:
        secrets = []
        for n in range(2):
            d = tmp_path / str(n)
            d.mkdir()
            (d / "prometheus.yaml").write_text(CONFIG_TEXT)
            _setup(d / "prometheus.yaml", FakeDocker(container=None))
            secrets.append(yaml.safe_load(
                (d / "searxng" / "settings.yml").read_text())["server"]["secret_key"])
        assert secrets[0] != secrets[1]

    def test_the_config_is_backed_up_first_and_keeps_its_comments(self, config: Path) -> None:
        _setup(config, FakeDocker(container=None))
        [backup] = _backups(config)
        assert backup.read_text() == CONFIG_TEXT
        text = config.read_text()
        assert "# my config — hand-written comments must survive" in text
        assert "provider: llama_cpp   # the 4090" in text
        before, after = yaml.safe_load(CONFIG_TEXT), yaml.safe_load(text)
        before["web_search"]["searxng_url"] = "http://127.0.0.1:8888"
        assert after == before

    def test_the_port_is_configurable(self, config: Path) -> None:
        docker = FakeDocker(container=None)
        transport, seen = _probe(_json_ok)
        code, _out = _setup(config, docker, transport, port=9999)
        assert code == 0
        run = next(c for c in docker.calls if c[0] == "run")
        assert "127.0.0.1:9999:8080" in run
        assert seen[0].url.port == 9999
        assert yaml.safe_load(config.read_text())["web_search"]["searxng_url"] == (
            "http://127.0.0.1:9999"
        )

    def test_a_failed_docker_run_leaves_the_config_alone(self, config: Path) -> None:
        docker = FakeDocker(container=None, run_error="port is already allocated")
        code, out = _setup(config, docker)
        assert code == 1
        assert "port is already allocated" in out
        assert config.read_text() == CONFIG_TEXT
        assert _backups(config) == []


# ---------------------------------------------------------------------------
# Re-running is safe
# ---------------------------------------------------------------------------


class TestRerun:
    def test_a_second_run_finds_the_container_and_changes_nothing(self, config: Path) -> None:
        docker = FakeDocker(container=None)
        assert _setup(config, docker)[0] == 0
        settings = (config.parent / "searxng" / "settings.yml").read_text()
        text_after_first = config.read_text()
        backups_after_first = _backups(config)

        again = FakeDocker(container=_container(running=True))
        code, out = _setup(config, again)
        assert code == 0, out
        assert again.changing_calls == []
        assert (config.parent / "searxng" / "settings.yml").read_text() == settings
        assert config.read_text() == text_after_first
        assert _backups(config) == backups_after_first
        assert "already" in out

    def test_a_stopped_container_is_started_not_recreated(self, config: Path) -> None:
        docker = FakeDocker(container=_container(running=False))
        code, out = _setup(config, docker)
        assert code == 0, out
        assert docker.changing_calls == [("start", "searxng")]

    def test_a_running_container_on_another_port_is_used_on_its_port(self, config: Path) -> None:
        docker = FakeDocker(container=_container(running=True, port="8899"))
        transport, seen = _probe(_json_ok)
        code, out = _setup(config, docker, transport)
        assert code == 0, out
        assert seen[0].url.port == 8899
        assert yaml.safe_load(config.read_text())["web_search"]["searxng_url"] == (
            "http://127.0.0.1:8899"
        )

    def test_a_container_of_that_name_running_something_else_is_refused(self, config: Path) -> None:
        docker = FakeDocker(container=_container(image="nginx:latest"))
        code, out = _setup(config, docker)
        assert code == 1
        assert "nginx" in out and "--name" in out
        assert docker.changing_calls == []
        assert config.read_text() == CONFIG_TEXT


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_it_says_what_it_would_do_and_does_none_of_it(self, config: Path) -> None:
        docker = FakeDocker(container=None)
        transport, seen = _probe(_json_ok)
        code, out = _setup(config, docker, transport, dry_run=True)
        assert code == 0, out
        assert docker.changing_calls == []
        assert seen == []
        assert not (config.parent / "searxng").exists()
        assert config.read_text() == CONFIG_TEXT
        assert _backups(config) == []
        assert "docker run -d --name searxng --restart unless-stopped" in out
        assert "settings.yml" in out
        assert 'searxng_url: "http://127.0.0.1:8888"' in out
        assert "dry run" in out.lower()

    def test_a_dry_run_without_docker_still_says_so(self, config: Path) -> None:
        code, out = _setup(config, None, dry_run=True)
        assert code == 1
        assert "not installed" in out


# ---------------------------------------------------------------------------
# It does not write a URL that does not answer JSON
# ---------------------------------------------------------------------------


class TestReadiness:
    def test_json_disabled_is_said_and_the_config_is_left_alone(self, config: Path) -> None:
        docker = FakeDocker(container=_container(running=True))
        code, out = _setup(config, docker, _probe(_forbidden)[0])
        assert code == 1
        assert wsb.SEARXNG_JSON_DISABLED in out
        assert config.read_text() == CONFIG_TEXT
        assert _backups(config) == []

    def test_no_answer_in_time_is_a_failure(self, config: Path) -> None:
        docker = FakeDocker(container=None)
        code, out = _setup(config, docker, _probe(_refused)[0], timeout=10.0)
        assert code == 1
        assert "did not answer" in out
        assert "docker logs searxng" in out
        assert config.read_text() == CONFIG_TEXT


# ---------------------------------------------------------------------------
# Editing the config: one key, nothing else
# ---------------------------------------------------------------------------


URL = "http://127.0.0.1:8888"


class TestSetSearxngUrl:
    def _check(self, before: str, after: str) -> None:
        old, new = yaml.safe_load(before) or {}, yaml.safe_load(after)
        section = old.get("web_search") or {}
        old["web_search"] = {**section, "searxng_url": URL}
        assert new == old

    def test_an_existing_key_is_replaced_in_place(self) -> None:
        after = search.set_searxng_url(CONFIG_TEXT, URL)
        self._check(CONFIG_TEXT, after)
        assert '  searxng_url: "http://127.0.0.1:8888"       # set by oara search setup' in after

    def test_a_section_without_the_key_gets_it(self) -> None:
        before = "web_search:\n  backends: [brave, duckduckgo]\nother: 1\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)

    def test_no_section_is_appended(self) -> None:
        before = "model:\n  provider: llama_cpp\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)
        assert after.startswith(before)

    def test_a_flow_style_section_is_handled(self) -> None:
        before = "web_search: {backends: [searxng, duckduckgo]}\nmodel: {provider: x}\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)

    def test_a_null_section_is_handled(self) -> None:
        before = "web_search: null   # off for now\nmodel:\n  provider: x\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)
        assert "# off for now" in after

    def test_an_empty_section_is_handled(self) -> None:
        before = "web_search:\nmodel:\n  provider: x\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)

    def test_a_key_nested_elsewhere_is_not_mistaken_for_ours(self) -> None:
        before = "other:\n  searxng_url: keep-me\nweb_search:\n  backends: [duckduckgo]\n"
        after = search.set_searxng_url(before, URL)
        self._check(before, after)
        assert "searxng_url: keep-me" in after


# ---------------------------------------------------------------------------
# The real Docker wrapper, against a fake `docker` on PATH
# ---------------------------------------------------------------------------


FAKE_DOCKER = """\
#!{python}
import json, sys
from pathlib import Path
log = Path({log!r})
with log.open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if args[:1] == ["info"]:
    print("27.1.1")
elif args[:2] == ["container", "inspect"]:
    if args[2] == "searxng":
        print(json.dumps([{{"Name": "/searxng", "Config": {{"Image": "searxng/searxng"}},
                           "State": {{"Running": True}}, "HostConfig": {{}}}}]))
    else:
        sys.stderr.write("Error: No such container: " + args[2] + "\\n")
        sys.exit(1)
elif args[:2] == ["run", "-d"]:
    print("c0ffee" * 10)
elif args[:1] == ["start"]:
    print(args[1])
else:
    sys.stderr.write("unexpected\\n")
    sys.exit(2)
"""


class TestTheRealWrapper:
    @pytest.fixture
    def fake_docker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "docker.log"
        script = bindir / "docker"
        script.write_text(FAKE_DOCKER.format(python=sys.executable, log=str(log)))
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
        return log

    def _log(self, log: Path) -> list[list[str]]:
        return [json.loads(line) for line in log.read_text().splitlines()]

    def test_it_finds_docker_on_path_and_speaks_its_cli(self, fake_docker: Path) -> None:
        docker = search.Docker.find()
        assert docker is not None
        assert docker.daemon_version() == ("27.1.1", "")
        assert docker.inspect("searxng")["Config"]["Image"] == "searxng/searxng"
        assert docker.inspect("other") is None
        assert docker.run_detached(["--name", "x", "img"]) == "c0ffee" * 10
        docker.start("x")
        assert self._log(fake_docker) == [
            ["info", "--format", "{{.ServerVersion}}"],
            ["container", "inspect", "searxng"],
            ["container", "inspect", "other"],
            ["run", "-d", "--name", "x", "img"],
            ["start", "x"],
        ]

    def test_no_docker_on_path_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        assert search.Docker.find() is None

    def test_oara_search_setup_dry_run_end_to_end(
        self, fake_docker: Path, config: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from prometheus.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "oara", "--config", str(config), "search", "setup", "--dry-run", "--port", "8901",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "127.0.0.1:8901" in out
        assert config.read_text() == CONFIG_TEXT
        assert all(entry[0] in ("info", "container") for entry in self._log(fake_docker))


class TestTheProbeIsBounded:
    def test_a_trickling_answer_cannot_stall_the_wait(
        self, config: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """httpx's timeouts are per read, so a server trickling bytes never trips
        them; each probe has a wall-clock limit, and the wait still ends."""
        import asyncio
        import threading

        async def drip():
            while True:
                await asyncio.sleep(0.02)
                yield b" "

        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=drip()))
        monkeypatch.setattr(search, "PROBE_TIMEOUT", 0.2)
        outcome: list[tuple[int, str]] = []
        worker = threading.Thread(
            target=lambda: outcome.append(
                _setup(config, FakeDocker(container=_container()), transport, timeout=4.0)
            ),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive(), "the readiness wait never ended"
        code, out = outcome[0]
        assert code == 1
        assert "did not answer" in out and "took longer than" in out
        assert config.read_text() == CONFIG_TEXT
