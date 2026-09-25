"""An isolated daemon: its own HOME, data dir, config and ports.

Never the live daemon. Isolation is environmental, as in FIRSTLIGHT: the
process gets a minimal, non-inherited environment with HOME pointed into the
harness root, so the operator's ``~/.prometheus``, ``~/.config/prometheus/env``
and exported keys cannot reach it — and nothing it writes can reach them.

The root path is FIXED (``/tmp/prometheus-parity`` by default). The daemon
writes absolute paths into the prompts it sends — the working directory, the
outbox, a workspace — and a random temp path would make every request differ
between runs. A fixed path keeps them identical without a normalization rule
(the cheaper kind of stability: nothing is hidden). A lock file refuses two
concurrent runs on the same root.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/tmp/prometheus-parity")
# A fixed, obviously-synthetic bearer token, rendered into the config at run
# time (the traces carry {{API_TOKEN}}, never a value). Short on purpose: the
# repo's pre-commit hook blocks 32+ char values bound to a token-named key.
# The daemon listens on harness-chosen ports and lives for one scenario.
API_TOKEN = "parity-test-token"


ROOT_MARKER = ".parity-root"


class HarnessError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RootLock:
    def __init__(self, root: Path) -> None:
        root.parent.mkdir(parents=True, exist_ok=True)
        self.path = root.parent / f".{root.name}.lock"
        self.fh = None

    def __enter__(self) -> "RootLock":
        self.fh = self.path.open("w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HarnessError(
                f"another parity run holds {self.path} — the root is a fixed path, "
                f"so two runs would share one data dir")
        return self

    def __exit__(self, *exc) -> None:
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


def http(method: str, url: str, body: Any = None, timeout: float = 30.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        code = exc.code
    try:
        return code, json.loads(raw)
    except json.JSONDecodeError:
        return code, raw


class Instance:
    """One daemon process for one scenario."""

    def __init__(self, root: Path, src_root: Path, config_text: str) -> None:
        self.root = root
        self.src_root = src_root
        self.home = root / "home"
        self.cfg_dir = self.home / ".prometheus"
        self.cwd = root / "cwd"
        self.config_path = self.cfg_dir / "prometheus.yaml"
        self.config_text = config_text
        self.api_port = free_port()
        self.ws_port = free_port()
        self.proc: subprocess.Popen | None = None
        self.log_path = root.parent / f"{root.name}.daemon.log"
        self.boot_seconds: float | None = None

    # -- filesystem -------------------------------------------------------
    def prepare(self) -> None:
        # The repo-local config/prometheus.yaml is searched BEFORE --config by
        # the subsystems that call config_search_paths(None) (LCM compaction
        # among them). A developer's live config there would leak into the run.
        repo_cfg = self.src_root / "config" / "prometheus.yaml"
        if repo_cfg.exists():
            raise HarnessError(
                f"{repo_cfg} exists. Subsystems that search for their own config "
                f"read it before --config, so the replay would run on the live "
                f"config. Run the harness from a worktree or a clean checkout.")
        if self.root.exists():
            # --root is a CLI argument and this is an rmtree: wipe only a tree
            # this harness created (it carries the marker), never an arbitrary
            # directory someone pointed us at.
            if not (self.root / ROOT_MARKER).is_file():
                raise HarnessError(
                    f"{self.root} exists and is not a parity root (no {ROOT_MARKER}); "
                    f"refusing to delete it. Remove it yourself or pick another --root.")
            shutil.rmtree(self.root)
        for d in (self.cfg_dir, self.cwd, self.home / ".config"):
            d.mkdir(parents=True, exist_ok=True)
        (self.root / ROOT_MARKER).write_text("parity harness root — safe to delete\n")
        self.config_path.write_text(
            self.config_text
            .replace("{{API_PORT}}", str(self.api_port))
            .replace("{{WS_PORT}}", str(self.ws_port)),
            encoding="utf-8")

    def env(self) -> dict[str, str]:
        venv_bin = str(Path(sys.executable).parent)
        return {
            "HOME": str(self.home),
            "PATH": f"{venv_bin}:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(self.src_root / "src"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "USER": "parity",
            "LOGNAME": "parity",
            "SHELL": "/bin/bash",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            # Set iteration order is part of what a turn produces wherever a
            # set is serialised; pinning the seed pins it on both sides.
            "PYTHONHASHSEED": "0",
            # SHELL, TZ, USER and the locale above are PINNED rather than
            # normalized: the daemon reports $SHELL in every system prompt,
            # and a pinned input needs no rule that could hide anything.
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    # -- process ----------------------------------------------------------
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def start(self, timeout: float = 180.0) -> None:
        log = self.log_path.open("w", encoding="utf-8")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "prometheus", "--config", str(self.config_path), "daemon"],
            cwd=self.cwd, env=self.env(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise HarnessError(f"daemon exited rc={self.proc.returncode} during boot "
                                   f"(log: {self.log_path})")
            try:
                code, body = http("GET", f"{self.base}/api/status", timeout=2)
                if code == 200 and isinstance(body, dict):
                    self.boot_seconds = time.monotonic() - t0
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.25)
        raise HarnessError(f"daemon did not answer /api/status within {timeout}s "
                           f"(log: {self.log_path})")

    def stop(self, timeout: float = 30.0) -> str:
        if self.proc is None or self.proc.poll() is not None:
            return "not running"
        self.proc.send_signal(signal.SIGTERM)
        try:
            rc = self.proc.wait(timeout=timeout)
            note = f"clean exit rc={rc}"
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=15)
            note = f"SIGKILL after {timeout}s"
        # Children the daemon spawned into its own session (coding runs) die
        # with the group; nothing may outlive the scenario.
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc = None
        return note

    def rss_kb(self) -> dict[str, int]:
        """VmRSS / VmHWM of the daemon process, in kB (Linux /proc)."""
        if self.proc is None:
            return {}
        out: dict[str, int] = {}
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                key, _, val = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    out[key] = int(val.split()[0])
        except OSError:
            pass
        return out
