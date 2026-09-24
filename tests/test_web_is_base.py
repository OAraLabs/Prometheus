"""The web API is a base dependency — and a missing web stack is a message,
not a traceback.

WHY THIS EXISTS
---------------
Until 0.9.2 fastapi, uvicorn and websockets were the `web` extra. A plain
`pip install oara-prometheus` — and the Homebrew formula, which is built from
the base list — therefore could not run `oara daemon` at all: not the real
daemon, and not setup mode, the no-config path that exists so Beacon can pair
with a fresh install. The docs said "keep [full]", but a stranger who typed
the package name got a daemon that died on start.

It died with a traceback, not the message written for exactly that case.
setup_server guards its fastapi import so the daemon's setup-mode gate can
still say "install the web stack" — but #292 added a StrictQueryRoute import
ABOVE the guard, strict_query imports fastapi at module level, and the gate
raised ModuleNotFoundError before the guard was ever reached, config or no
config.

So this file pins:

  * the DECLARATION — fastapi, uvicorn (plain, not [standard]) and websockets
    are base requirements, and `[web]` still lists them so every existing
    `pip install 'oara-prometheus[web]'` keeps working;
  * the MESSAGE — with the web stack made unimportable in a child process
    (the dev environment has it, so absence is simulated with a meta-path
    finder, as in test_voice_is_an_extra), a real `oara daemon` with no
    config names what is missing and exits 1, with no traceback.

Setup mode was not the only thing in the way. Once configured, the daemon
reads the MCP server store before deciding whether MCP is configured — an
import written as "lazy + guarded" — but `prometheus.mcp`'s `__init__`
imported the optional `mcp` SDK eagerly, so reading the store needed the
SDK and a plain install died there too. Hence:

  * the PLAIN BOOT — every extra-only package made unimportable (derived
    from pyproject, so it tracks the extras as they change), a configured
    `oara daemon` must come up and answer on its API port.

The real-install half — a built wheel with no extras, an isolated HOME, and
`oara daemon` booting into setup mode (no config) or running the whole
first-run flow (configured) — needs PyPI, so it runs in CI as the firstlight
`setup-mode` / `base` and `local` / `base` legs (scripts/firstlight_harness.py),
not here.
"""

from __future__ import annotations

import json
import os
import signal
import site
import socket
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tests.test_daemon_shutdown import _free_port, _ModelsHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

WEB_DISTS = {"fastapi", "uvicorn", "websockets"}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _requirements(specs: list[str]) -> dict[str, Requirement]:
    return {canonicalize_name(Requirement(s).name): Requirement(s) for s in specs}


# ── the declaration ─────────────────────────────────────────────────────


def test_the_web_stack_is_in_the_base_dependencies():
    base = _requirements(_project()["dependencies"])
    missing = WEB_DISTS - base.keys()
    assert not missing, (
        f"{sorted(missing)} missing from the BASE dependencies — a plain "
        f"`pip install oara-prometheus` (and the Homebrew formula) then "
        f"cannot start `oara daemon`, not even into setup mode."
    )


def test_base_uvicorn_is_plain_not_standard():
    """uvicorn[standard] drags in uvloop/httptools/watchfiles native builds;
    the daemon needs none of them (its WS bridge runs on `websockets`)."""
    uvicorn = _requirements(_project()["dependencies"])["uvicorn"]
    assert not uvicorn.extras, f"base uvicorn carries extras {uvicorn.extras}"


def test_the_web_extra_still_lists_the_web_stack():
    """`pip install 'oara-prometheus[web]'` is in docs, scripts and muscle
    memory; an extra that vanished or thinned out would break them."""
    web = _requirements(_project()["optional-dependencies"]["web"])
    missing = WEB_DISTS - web.keys()
    assert not missing, f"`web` extra no longer lists {sorted(missing)}"


# ── the message ─────────────────────────────────────────────────────────

_CHILD = textwrap.dedent('''\
    import importlib.abc, json, os, runpy, sys

    BLOCKED = set(json.loads(os.environ["BLOCKED"]))

    class Absent(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}",
                                          name=name)
            return None

    sys.meta_path.insert(0, Absent())

    # The finder must bite, or the assertions downstream are vacuous.
    for probe in BLOCKED:
        try:
            __import__(probe)
        except ModuleNotFoundError:
            continue
        sys.exit(f"BLOCKER INERT: {probe} imported")

    if sys.argv[1] == "mcp-store":
        # What the daemon's boot and the /api/mcp/servers routes import.
        import prometheus.mcp.bootstrap
        import prometheus.mcp.store
        import prometheus.mcp as pkg
        try:
            pkg.McpRuntime
            lazy = None
        except ModuleNotFoundError as exc:
            lazy = str(exc)
        print(json.dumps({"store_imported": True, "runtime_error": lazy}))
    else:
        sys.argv = ["oara", "daemon"]
        runpy.run_module("prometheus", run_name="__main__", alter_sys=True)
''')


def _child_env(blocked: set[str], home: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "HOME": str(home),  # no ~/.prometheus → no config → setup mode
        "BLOCKED": json.dumps(sorted(blocked)),
        # If the guard ever failed OPEN and setup mode really started, it
        # must not grab a port a live daemon on this host is using.
        "PROMETHEUS_WEB_API_PORT": str(_free_port()),
        # A foreign HOME moves user site-packages; see test_daemon_shutdown.
        "PYTHONUSERBASE": site.getuserbase(),
    }
    for key in ("VIRTUAL_ENV", "PYTHONPATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _child_script(tmp_path: Path) -> Path:
    script = tmp_path / "absent.py"
    script.write_text(_CHILD, encoding="utf-8")
    return script


def _run_daemon_without(blocked: set[str], tmp_path: Path) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir()
    return subprocess.run(
        [sys.executable, str(_child_script(tmp_path)), "daemon"],
        env=_child_env(blocked, home), cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize(
    ("blocked", "named"),
    [
        # The whole web stack gone — the shape of a pre-0.9.2 plain install.
        ({"fastapi", "starlette", "uvicorn", "websockets"},
         "fastapi and uvicorn are not installed"),
        # fastapi present, uvicorn not: without the check, setup mode would
        # print its banner and then die importing uvicorn to serve it.
        ({"uvicorn"}, "uvicorn is not installed"),
    ],
    ids=["whole-web-stack", "uvicorn-only"],
)
def test_no_config_and_no_web_stack_is_a_message_not_a_traceback(
        tmp_path, blocked, named):
    proc = _run_daemon_without(blocked, tmp_path)
    out = proc.stdout + proc.stderr
    assert "BLOCKER INERT" not in out, out
    assert "Traceback" not in out, (
        f"`oara daemon` with no config and no {sorted(blocked)} crashed "
        f"instead of explaining itself:\n{out[-3000:]}"
    )
    assert proc.returncode == 1, (proc.returncode, out[-2000:])
    assert "setup mode needs the web API" in out, out[-2000:]
    assert named in out, out[-2000:]
    assert "pip install 'oara-prometheus[web]'" in out, out[-2000:]
    assert "PROMETHEUS IS IN SETUP MODE" not in out, (
        "setup mode printed its banner — it must refuse BEFORE promising a "
        "pairing code it cannot serve"
    )


# ── the plain boot ──────────────────────────────────────────────────────

# Import names for extras whose distribution name is not its import name;
# anything else imports as its canonical name with "-" -> "_".
_IMPORT_NAMES = {
    "arize-phoenix": ["phoenix"],
    "discord-py": ["discord"],
    "opentelemetry-api": ["opentelemetry"],
    "opentelemetry-sdk": ["opentelemetry"],
    "piper-tts": ["piper"],
    "python-multipart": ["multipart", "python_multipart"],
}


def _extra_only_imports() -> set[str]:
    """Import names of every package an extra adds beyond the base list.
    `dev` is test tooling, not a product extra."""
    project = _project()
    base = _requirements(project["dependencies"]).keys()
    dists = {
        name
        for extra, specs in project["optional-dependencies"].items()
        if extra != "dev"
        for name in _requirements(specs)
    } - base
    return {m for d in dists for m in _IMPORT_NAMES.get(d, [d.replace("-", "_")])}


def test_the_extra_only_set_is_what_it_should_be():
    """Guard the guard: a derivation that came back empty, or had swallowed
    the web stack, would make the plain-boot test prove nothing."""
    blocked = _extra_only_imports()
    assert {"mcp", "anthropic", "playwright", "slack_bolt", "discord",
            "piper"} <= blocked
    assert not blocked & {"fastapi", "uvicorn", "websockets"}


def test_the_mcp_store_imports_without_the_mcp_sdk(tmp_path):
    """The store is SDK-free; importing it must not import the SDK. Until
    0.9.2 `prometheus.mcp.__init__` did, eagerly."""
    home = tmp_path / "home"
    home.mkdir()
    proc = subprocess.run(
        [sys.executable, str(_child_script(tmp_path)), "mcp-store"],
        env=_child_env({"mcp"}, home), cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["store_imported"] is True
    # The public name still resolves — lazily, and only then needs the SDK.
    assert result["runtime_error"] and "mcp" in result["runtime_error"]


def test_a_plain_install_boots_the_configured_daemon(tmp_path):
    """A REAL configured `oara daemon` with every extra-only package
    unimportable must come up and answer on its API port — and, on a fresh
    HOME, refuse a bare request (it minted a token). Same shape as
    test_voice_is_an_extra's boot test: loopback stub model, isolated HOME."""
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        api_port, ws_port = _free_port(), _free_port()
        home = tmp_path / "home"
        (home / ".prometheus").mkdir(parents=True)
        (home / ".prometheus" / "prometheus.yaml").write_text(textwrap.dedent(f"""\
            model:
              provider: llama_cpp
              base_url: http://127.0.0.1:{stub.server_port}
              model: fl1-model
            gateway:
              telegram_enabled: false
            web:
              enabled: true
              api_port: {api_port}
              ws_port: {ws_port}
            """), encoding="utf-8")
        log_path = tmp_path / "daemon.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, str(_child_script(tmp_path)), "daemon"],
                cwd=tmp_path, env=_child_env(_extra_only_imports(), home),
                stdout=log, stderr=subprocess.STDOUT,
            )
        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                if proc.poll() is not None:
                    pytest.fail(
                        f"daemon exited rc={proc.returncode} with every "
                        f"extra-only package absent:\n"
                        f"{log_path.read_text()[-3000:]}")
                try:
                    with socket.create_connection(("127.0.0.1", api_port),
                                                  timeout=1):
                        break
                except OSError:
                    time.sleep(0.5)
            else:
                pytest.fail("daemon API port never opened within 90s:\n"
                            + log_path.read_text()[-3000:])
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{api_port}/api/status", timeout=10)
                code = 200
            except urllib.error.HTTPError as exc:
                code = exc.code
            assert code == 401, (
                f"bare /api/status answered {code}; a fresh install mints a "
                f"token and must refuse it")
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
    finally:
        stub.shutdown()
