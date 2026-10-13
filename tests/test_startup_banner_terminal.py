"""The startup banner in the real processes, on a real (pseudo) terminal.

test_startup_banner.py pins the text and the rules on fake streams. This is the other half: the
actual ``oara`` and ``oara daemon`` processes, started with an isolated HOME, a stub model server
and free ports (nothing here touches the machine's own daemon, config or ports), with stdout on
a terminal, on a pipe, or with ``--quiet``, ``NO_COLOR`` or ``OARA_NO_BANNER`` set. What a person
would see is what is asserted: the port the daemon was told to use, the model the server reported,
and none of the directories or addresses the process knows.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import site
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pty = pytest.importorskip("pty")

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "scripts" / "firstlight_stub_model.py"
# A llama.cpp server may name its model by the file it loaded, directory and all.
SERVED = "/srv/private-models/fl1-model.gguf"
TAGLINE = "a fire you own doesn't go out"
SANGUINE = "\x1b[38;2;184;71;58m"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ModelsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: ANN002
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": SERVED}]}).encode()
        self.send_response(200 if "/v1/models" in self.path else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def models_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def stub_model(tmp_path):
    """The FIRSTLIGHT stub: answers a chat turn, which `--once` needs."""
    port = _free_port()
    with (tmp_path / "stub.log").open("wb") as log:
        stub = subprocess.Popen([sys.executable, str(STUB), "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 15
    while time.time() < deadline and stub.poll() is None:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    if stub.poll() is not None:
        pytest.fail(f"stub model did not start: {(tmp_path / 'stub.log').read_text(errors='replace')[-1500:]}")
    try:
        yield port
    finally:
        stub.terminate()
        stub.wait(timeout=5)


def _home(tmp_path: Path, model_port: int, *, api_port: int = 0, ws_port: int = 0) -> Path:
    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    web = f"web:\n  enabled: true\n  api_port: {api_port}\n  ws_port: {ws_port}\n" if api_port else ""
    (home / ".prometheus" / "prometheus.yaml").write_text(
        textwrap.dedent(f"""\
            model:
              provider: llama_cpp
              base_url: http://127.0.0.1:{model_port}
              model: fl1-model
            gateway:
              telegram_enabled: false
            tools:
              deferred_loading:
                enabled: auto
                always_loaded: [bash, read_file]
            """) + web,
        encoding="utf-8",
    )
    return home


def _env(home: Path, **extra: str) -> dict[str, str]:
    """A from-scratch environment: nothing of the caller's (a live config dir, NO_COLOR, a token) leaks in."""
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "TERM": "xterm-256color",
        # A foreign HOME moves the user site-packages; the isolation wanted here is of config, not of packages.
        "PYTHONUSERBASE": site.getuserbase(),
    }
    for passthrough in ("PYTHONPATH", "VIRTUAL_ENV"):
        if passthrough in os.environ:
            env[passthrough] = os.environ[passthrough]
    env.update(extra)
    return env


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _on_pty(argv, *, env, cwd, stdin_text=None, until=None, timeout=90) -> str:
    """Run *argv* with stdout (and stderr) on a pseudo-terminal; return what it wrote.

    Reads until *until* appears (the process is then stopped) or the process ends. A thread does the
    reading: select() refuses a descriptor numbered 1024 or more, which a pseudo-terminal opened late in
    a long suite is.
    """
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        argv, cwd=cwd, env=env, close_fds=True,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=slave, stderr=subprocess.STDOUT,
    )
    os.close(slave)
    if stdin_text is not None:
        proc.stdin.write(stdin_text.encode())
        proc.stdin.close()

    arrived: queue.Queue[bytes] = queue.Queue()  # b"" marks the end

    def pump() -> None:
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:  # EIO: the child closed its end
                data = b""
            arrived.put(data)
            if not data:
                return

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    chunks: list[bytes] = []
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                data = arrived.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if not data:
                break
            chunks.append(data)
            if until and until in b"".join(chunks).decode("utf-8", "replace"):
                break
        else:
            pytest.fail(f"timed out after {timeout}s; output so far:\n{b''.join(chunks).decode('utf-8', 'replace')[-2000:]}")
    finally:
        # Keep reading while it shuts down: a terminal nobody reads fills up, and the process then
        # blocks writing its last log lines until it is killed.
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        stopped = time.time() + 20
        while time.time() < stopped:
            try:
                if not arrived.get(timeout=0.25):
                    break
            except queue.Empty:
                if proc.poll() is not None:
                    break
        _stop(proc)
        # The reader gets EIO once the child is gone; closing the descriptor under it first could
        # leave it reading whatever reuses the number.
        reader.join(timeout=5)
        if not reader.is_alive():
            os.close(master)
    return b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n")


def _daemon(tmp_path, models_server, *flags: str, **env_extra: str) -> tuple[str, int]:
    api_port, ws_port = _free_port(), _free_port()
    home = _home(tmp_path, models_server.server_port, api_port=api_port, ws_port=ws_port)
    out = _on_pty(
        [sys.executable, "-m", "prometheus", "--config", str(home / ".prometheus" / "prometheus.yaml"), "daemon", *flags],
        env=_env(home, **env_extra), cwd=tmp_path, until="Prometheus daemon running",
    )
    assert "Prometheus daemon running" in out, out[-2000:]
    return out, api_port


def _plain(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _banner_of(text: str) -> str:
    """Just the banner: the wordmark's first row through the footer line."""
    start = text.index("┏━┓")
    footer = text.index("OAra Labs · prometheus", start)
    return text[start : text.index("\n", footer) + 1]


def test_the_daemon_on_a_terminal_prints_the_banner_with_what_is_true(tmp_path, models_server):
    out, api_port = _daemon(tmp_path, models_server)
    assert out.count(TAGLINE) == 1
    shown = _plain(out)
    assert f"  ◉ daemon   listening on :{api_port}\n" in shown
    assert "  ◉ model    fl1-model.gguf · llama.cpp\n" in shown
    assert "  ◉ beacon   oara.ai/beacon\n" in shown
    assert "  OAra Labs · prometheus " in shown
    assert SANGUINE in out
    # After the boot logging, before "daemon running": the banner is the last thing a person reads.
    assert out.index(TAGLINE) < out.index("Prometheus daemon running")
    assert out.index("Web bridge started") < out.index(TAGLINE)


def test_the_banner_names_no_directory_or_address(tmp_path, models_server):
    out, _ = _daemon(tmp_path, models_server)
    banner_text = _plain(_banner_of(out))
    for leaked in ("/srv", "private-models", str(tmp_path), "127.0.0.1", "0.0.0.0", "localhost", "http", "token"):
        assert leaked not in banner_text, leaked


def test_the_daemon_with_no_color_prints_the_banner_plain(tmp_path, models_server):
    out, api_port = _daemon(tmp_path, models_server, NO_COLOR="1")
    banner_text = _banner_of(out)
    assert "\x1b" not in banner_text
    assert f"  ◉ daemon   listening on :{api_port}\n" in banner_text


@pytest.mark.parametrize("how", ["flag", "short flag", "env"])
def test_the_daemon_prints_no_banner_when_told_not_to(tmp_path, models_server, how):
    flags = {"flag": ["--quiet"], "short flag": ["-q"], "env": []}[how]
    env = {"OARA_NO_BANNER": "1"} if how == "env" else {}
    out, _ = _daemon(tmp_path, models_server, *flags, **env)
    assert TAGLINE not in out and "┏━┓" not in out and "oara.ai/beacon" not in out


def test_the_daemon_off_a_terminal_prints_no_banner(tmp_path, models_server):
    """systemd, a pipe, a file, CI: stdout is not a terminal."""
    api_port, ws_port = _free_port(), _free_port()
    home = _home(tmp_path, models_server.server_port, api_port=api_port, ws_port=ws_port)
    log = tmp_path / "daemon.out"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "prometheus", "--config", str(home / ".prometheus" / "prometheus.yaml"), "daemon"],
            cwd=tmp_path, env=_env(home), stdout=fh, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.time() + 90
            while time.time() < deadline and "Prometheus daemon running" not in log.read_text(errors="replace"):
                assert proc.poll() is None, log.read_text(errors="replace")[-2000:]
                time.sleep(0.5)
        finally:
            _stop(proc)
    text = log.read_text(errors="replace")
    assert "Prometheus daemon running" in text
    assert TAGLINE not in text and "┏━┓" not in text and "\x1b[" not in text


def test_interactive_on_a_terminal_prints_the_banner_without_a_daemon_line(tmp_path, models_server):
    home = _home(tmp_path, models_server.server_port)
    out = _on_pty(
        [sys.executable, "-m", "prometheus", "--config", str(home / ".prometheus" / "prometheus.yaml")],
        env=_env(home), cwd=tmp_path, stdin_text="exit\n", until="Goodbye",
    )
    shown = _plain(out)
    assert TAGLINE in shown
    assert "  ◉ model    fl1-model.gguf · llama.cpp\n" in shown
    assert shown.index(TAGLINE) < shown.index("interactive mode")
    banner_text = _banner_of(shown)
    assert "daemon" not in banner_text  # no daemon is listening: no line
    assert "/srv" not in banner_text and "private-models" not in banner_text


def test_interactive_quiet_prints_no_banner(tmp_path, models_server):
    home = _home(tmp_path, models_server.server_port)
    out = _on_pty(
        [sys.executable, "-m", "prometheus", "--quiet", "--config", str(home / ".prometheus" / "prometheus.yaml")],
        env=_env(home), cwd=tmp_path, stdin_text="exit\n", until="Goodbye",
    )
    assert "interactive mode" in out and TAGLINE not in out


def test_once_prints_no_banner_even_on_a_terminal(tmp_path, stub_model):
    home = _home(tmp_path, stub_model)
    out = _on_pty(
        [sys.executable, "-m", "prometheus", "--config", str(home / ".prometheus" / "prometheus.yaml"), "--once", "list files"],
        env=_env(home), cwd=tmp_path,
    )
    assert "FIRSTLIGHT-COMPLETE" in out, out[-1500:]
    assert TAGLINE not in out and "┏━┓" not in out and "oara.ai/beacon" not in out


@pytest.mark.parametrize("name", ["oara", "prometheus"])
def test_version_is_one_plain_line_on_a_pipe_and_on_a_terminal(tmp_path, name):
    from prometheus.version import package_version

    wanted = f"◉ prometheus {package_version()} · OAra Labs\n"
    script = f"import sys; sys.argv[0] = {name!r}; from prometheus.__main__ import main; main()"
    piped = subprocess.run(
        [sys.executable, "-c", script, "--version"], cwd=tmp_path, env=_env(tmp_path),
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert (piped.returncode, piped.stdout, piped.stderr) == (0, wanted, "")
    on_terminal = _on_pty([sys.executable, "-c", script, "--version"], env=_env(tmp_path), cwd=tmp_path)
    assert on_terminal == wanted  # no colour even on a terminal: scripts read this line


def test_version_does_not_fail_on_an_ascii_stdout(tmp_path):
    env = _env(tmp_path, PYTHONIOENCODING="ascii")
    done = subprocess.run(
        [sys.executable, "-m", "prometheus", "--version"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("prometheus ") and done.stdout.endswith(" - OAra Labs\n")


def test_waiting_for_output_does_not_depend_on_the_descriptor_number(tmp_path):
    """The full suite leaves enough descriptors open that a pseudo-terminal's lands past 1023, where select() refuses it."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 1300
    if hard != resource.RLIM_INFINITY and hard < needed:
        pytest.skip(f"cannot raise the descriptor limit to {needed}")
    held: list[int] = []
    try:
        if soft != resource.RLIM_INFINITY and soft < needed:
            resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
        while not held or held[-1] < 1100:
            held.append(os.open(os.devnull, os.O_RDONLY))
        out = _on_pty([sys.executable, "-c", "print('ok')"], env=_env(tmp_path), cwd=tmp_path)
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert out.strip() == "ok"
