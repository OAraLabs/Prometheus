"""``oara search setup`` — run SearXNG in Docker and point web_search at it.

What it does, in order, stopping at the first thing it cannot do:

1. Finds the config file the daemon reads. None, or an unreadable one: it says
   so (``oara setup`` makes one) and changes nothing.
2. Checks for Docker and that its daemon answers. No Docker: it explains, names
   the manual route, and changes nothing.
3. Looks for a container of that name (default ``searxng``). Running: reused.
   Stopped: started. Running something other than searxng/searxng: refused.
   None: it writes ``<config dir>/searxng/settings.yml`` (JSON output on, the
   limiter off, a random secret_key) unless one is already there, and runs
   ``searxng/searxng`` with ``--restart unless-stopped`` on 127.0.0.1:<port>.
4. Waits until a test query answers JSON — through the same function
   web_search uses, so "answers JSON" means the same thing to both.
5. Backs the config up (``.bak``, ``.bak.1``, … never overwriting one) and sets
   ``web_search.searxng_url``. One key; every comment and every other key stays
   as it was, which is checked by re-parsing before anything is written.

``--dry-run`` runs the read-only checks (config, ``docker info``, ``docker
container inspect``) and prints the rest without doing it. Re-running is safe:
a running container is reused and a URL that is already set is left alone.

Why ``FORCE_OWNERSHIP=false``: run as root, the image's entrypoint
``chown -R``s the mounted ``/etc/searxng`` to its own user, which on Linux would
leave a directory under the Prometheus config dir owned by a uid the operator
does not have. The settings file is written world-readable instead, so the
container's user can read it without owning it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

DEFAULT_PORT = 8888
DEFAULT_NAME = "searxng"
DEFAULT_TIMEOUT = 90.0
IMAGE = "searxng/searxng"
CONTAINER_PORT = 8080
POLL_INTERVAL = 2.0
PROBE_TIMEOUT = 5.0
DOCS = "docs/guide/web-search.md"


class DockerError(Exception):
    """A docker command failed; the message is its stderr."""


class ConfigEditError(Exception):
    """The config could not be edited without changing more than one key."""


# ---------------------------------------------------------------------------
# Docker, through its CLI
# ---------------------------------------------------------------------------


class Docker:
    """The few ``docker`` commands setup needs. Tests pass a fake with the same methods."""

    def __init__(self, binary: str) -> None:
        self.binary = binary

    @classmethod
    def find(cls) -> Docker | None:
        path = shutil.which("docker")
        return cls(path) if path else None

    def _run(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args], capture_output=True, text=True,
            timeout=timeout, check=False,
        )

    def daemon_version(self) -> tuple[str | None, str]:
        """(server version, "") when the daemon answers, else (None, why not)."""
        try:
            proc = self._run(["info", "--format", "{{.ServerVersion}}"], 30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        version = proc.stdout.strip()
        if proc.returncode != 0 or not version:
            detail = (proc.stderr or proc.stdout).strip()
            return None, detail or f"`docker info` exited {proc.returncode}"
        return version, ""

    def inspect(self, name: str) -> dict[str, Any] | None:
        """The container's inspect record, or None when there is none of that name."""
        try:
            proc = self._run(["container", "inspect", name], 30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerError(f"{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            if "no such" in detail.lower():
                return None
            raise DockerError(detail or f"`docker container inspect` exited {proc.returncode}")
        try:
            records = json.loads(proc.stdout)
        except ValueError as exc:
            raise DockerError("`docker container inspect` printed something that is not JSON") from exc
        return records[0] if isinstance(records, list) and records else None

    def run_detached(self, argv: list[str]) -> str:
        """``docker run -d <argv>``; returns the container id. May pull the image first."""
        try:
            proc = self._run(["run", "-d", *argv], 900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerError(f"{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            raise DockerError((proc.stderr or proc.stdout).strip() or f"`docker run` exited {proc.returncode}")
        lines = proc.stdout.strip().splitlines()
        return lines[-1] if lines else ""

    def start(self, name: str) -> None:
        try:
            proc = self._run(["start", name], 60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerError(f"{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            raise DockerError((proc.stderr or proc.stdout).strip() or f"`docker start` exited {proc.returncode}")


# ---------------------------------------------------------------------------
# The setup
# ---------------------------------------------------------------------------


@dataclass
class SetupOptions:
    config_path: str | None = None
    port: int = DEFAULT_PORT
    name: str = DEFAULT_NAME
    dry_run: bool = False
    timeout: float = DEFAULT_TIMEOUT
    #: Where settings.yml goes. Default: ``<config dir>/searxng``.
    settings_dir: Path | None = None


def run_setup(
    opts: SetupOptions,
    *,
    docker: Docker | None,
    transport: httpx.AsyncBaseTransport | None = None,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the setup; returns an exit code. ``docker`` None means no Docker was found."""
    from prometheus.config.defaults import resolve_config_path

    dry = opts.dry_run
    out("SearXNG for web_search" + (" (dry run: nothing will be changed)" if dry else ""))

    # 1. The config file, before anything else is touched.
    config_path = resolve_config_path(opts.config_path)
    if not config_path.is_file():
        out(f"  config: there is no prometheus.yaml at {config_path}.")
        out("  Run `oara setup` to make one, then run this again. Nothing was changed.")
        return 1
    try:
        current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        out(f"  config: {config_path} cannot be read ({type(exc).__name__}: {exc}).")
        out("  Fix it and run this again. Nothing was changed.")
        return 1
    if not isinstance(current, dict):
        out(f"  config: {config_path} is not a YAML mapping. Nothing was changed.")
        return 1

    # 2. Docker.
    if docker is None:
        out("  docker: Docker is not installed (no `docker` on PATH).")
        out("  `oara search setup` runs SearXNG in Docker. Install Docker and run this")
        out("  again, or run SearXNG yourself with JSON output enabled and set")
        out(f"  web_search.searxng_url in {config_path} (see {DOCS}).")
        out("  Nothing was changed.")
        return 1
    version, error = docker.daemon_version()
    if version is None:
        out(f"  docker: installed, but its daemon is not answering: {error}")
        out("  Start Docker and run this again. Nothing was changed.")
        return 1
    out(f"  docker: {version}")

    # 3. The container.
    try:
        container = docker.inspect(opts.name)
    except DockerError as exc:
        out(f"  docker: could not inspect a container named {opts.name!r}: {exc}")
        out("  Nothing was changed.")
        return 1

    port = opts.port
    settings_dir = opts.settings_dir or _default_settings_dir()
    settings_file = settings_dir / "settings.yml"
    if container is not None:
        image = str((container.get("Config") or {}).get("Image") or "")
        if IMAGE not in image:
            out(f"  container: {opts.name!r} already exists and runs {image or 'an unknown image'},")
            out(f"  not {IMAGE}. Pass --name to use another container name. Nothing was changed.")
            return 1
        bound = _published_port(container)
        if bound is not None and bound != port:
            out(f"  container: {opts.name!r} publishes port {bound}; using that, not {port}")
            port = bound
        action = "reuse" if (container.get("State") or {}).get("Running") else "start"
    else:
        action = "run"

    url = f"http://127.0.0.1:{port}"
    run_argv = [
        "--name", opts.name, "--restart", "unless-stopped",
        "-p", f"127.0.0.1:{port}:{CONTAINER_PORT}",
        "-e", "FORCE_OWNERSHIP=false",
        "-v", f"{settings_dir}:/etc/searxng",
        IMAGE,
    ]
    if action == "reuse":
        out(f"  container: {opts.name!r} is already running; reusing it")
    elif action == "start":
        out(f"  container: {opts.name!r} exists but is stopped; "
            f"{'would start' if dry else 'starting'} it (docker start {opts.name})")
    else:
        if settings_file.exists():
            out(f"  settings: keeping the existing {settings_file}")
        else:
            out(f"  settings: {'would write' if dry else 'writing'} {settings_file} "
                f"(JSON output on, limiter off, a random secret_key)")
        out(f"  container: {'would run' if dry else 'running'}: "
            f"docker run -d {shlex.join(run_argv)}")
    out(f"  ready: {'would wait' if dry else 'waiting'} until {url}/search?q=test&format=json "
        f"answers JSON (up to {opts.timeout:.0f}s)")

    section = current.get("web_search")
    already = isinstance(section, dict) and section.get("searxng_url") == url
    if already:
        out(f"  config: web_search.searxng_url is already {url} in {config_path}")
    else:
        out(f"  config: {'would set' if dry else 'will set'} "
            f'web_search.searxng_url: "{url}" in {config_path} (backed up first)')

    if dry:
        out("Dry run: nothing was changed.")
        return 0

    # 4. Act.
    try:
        if action == "run":
            if not settings_file.exists():
                _write_settings(settings_file)
            docker.run_detached(run_argv)
        elif action == "start":
            docker.start(opts.name)
    except (DockerError, OSError) as exc:
        out(f"  FAILED: {exc}")
        out(f"  The config was not changed.{_settings_note(action, settings_file)}")
        return 1

    # 5. Wait for JSON.
    ok, detail = _wait_for_json(url, opts.timeout, transport, sleep, clock)
    if not ok:
        out(f"  FAILED: {detail}")
        out(f"  Check `docker logs {opts.name}`. The config was not changed.")
        return 1
    out(f"  ready: {url} answers JSON")

    # 6. The config.
    if already:
        out("Done. web_search.searxng_url was already set; the config is unchanged.")
        return 0
    try:
        text = config_path.read_text(encoding="utf-8")
        new_text = set_searxng_url(text, url)
        backup = _free_backup_path(config_path)
        shutil.copy2(config_path, backup)
        _atomic_write(config_path, new_text)
    except (ConfigEditError, OSError) as exc:
        out(f"  FAILED to edit {config_path}: {exc}")
        out(f'  SearXNG is running. Add this yourself under web_search: searxng_url: "{url}"')
        return 1
    out(f"  config: backed up to {backup}")
    out(f"  config: web_search.searxng_url = {url}")
    out("Done. Restart the daemon so web_search picks it up; its boot log then says")
    out("`web_search: backends searxng -> ...`.")
    return 0


def _default_settings_dir() -> Path:
    from prometheus.config.paths import config_dir_path

    return config_dir_path() / "searxng"


def _settings_note(action: str, settings_file: Path) -> str:
    if action == "run" and settings_file.exists():
        return f" {settings_file} was written and is kept for the next run."
    return ""


def _published_port(container: dict[str, Any]) -> int | None:
    """The host port bound to the container's 8080, when there is one."""
    for source in (
        (container.get("HostConfig") or {}).get("PortBindings"),
        (container.get("NetworkSettings") or {}).get("Ports"),
    ):
        bindings = (source or {}).get(f"{CONTAINER_PORT}/tcp") if isinstance(source, dict) else None
        for binding in bindings or []:
            try:
                return int(binding.get("HostPort"))
            except (TypeError, ValueError, AttributeError):
                continue
    return None


def render_settings(secret_key: str) -> str:
    return (
        "# Written by `oara search setup` for Prometheus's web_search tool.\n"
        "#   search.formats - json is what web_search reads; SearXNG ships without it\n"
        "#   server.limiter - off: it rate-limits API clients like web_search\n"
        "#   secret_key     - random, generated for this instance\n"
        "use_default_settings: true\n"
        "server:\n"
        f'  secret_key: "{secret_key}"\n'
        "  limiter: false\n"
        "search:\n"
        "  formats:\n"
        "    - html\n"
        "    - json\n"
    )


def _write_settings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    path.write_text(render_settings(secrets.token_hex(32)), encoding="utf-8")
    # Readable by the container's own user, which does not own it (see the
    # module docstring on FORCE_OWNERSHIP).
    path.chmod(0o644)


def _wait_for_json(
    url: str, timeout: float, transport: httpx.AsyncBaseTransport | None,
    sleep: Callable[[float], None], clock: Callable[[], float],
) -> tuple[bool, str]:
    from prometheus.tools.builtin import web_search_backends as wsb

    async def probe() -> None:
        # Wall clock, not just httpx's timeout, which is per read: a server
        # trickling bytes would otherwise hold one probe, and the wait, forever.
        async with httpx.AsyncClient(transport=transport) as client:
            await wsb.bounded(
                wsb.fetch_searxng_json(client, url, "test", timeout=PROBE_TIMEOUT),
                PROBE_TIMEOUT,
            )

    started = clock()
    last = "no answer yet"
    while True:
        try:
            asyncio.run(probe())
            return True, ""
        except wsb.BackendFailure as exc:
            if exc.kind == wsb.KIND_JSON_DISABLED:
                return False, exc.reason
            last = exc.reason
        if clock() - started >= timeout:
            return False, f"{url} did not answer JSON within {timeout:.0f}s (last: {last})"
        sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# The config edit: one key, nothing else
# ---------------------------------------------------------------------------

_SECTION = re.compile(r"^web_search:(?P<rest>[^\n]*)$")


def set_searxng_url(text: str, url: str) -> str:
    """``text`` with ``web_search.searxng_url`` set to ``url``, comments kept.

    Raises :class:`ConfigEditError` when the result would differ from the
    original in anything but that key (checked by parsing both).
    """
    if any(c in url for c in "\"'\n#"):
        raise ConfigEditError(f"refusing a URL with quotes, '#' or a newline: {url!r}")
    value = f'"{url}"'
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if _SECTION.match(line.rstrip("\n"))), None)

    if start is None:
        prefix = text if (not text or text.endswith("\n")) else text + "\n"
        new = prefix + (
            "\n# web_search backends (docs/guide/web-search.md); set by `oara search setup`\n"
            f"web_search:\n  searxng_url: {value}\n"
        )
    else:
        rest = _SECTION.match(lines[start].rstrip("\n")).group("rest")  # type: ignore[union-attr]
        inline, comment = _split_comment(rest)
        try:
            inline_value = yaml.safe_load(inline) if inline.strip() else None
        except yaml.YAMLError as exc:
            raise ConfigEditError(f"cannot read the web_search line: {exc}") from exc
        if inline_value is not None:
            new = _rewrite_flow_section(lines, start, inline, comment, url)
        else:
            if inline.strip():  # `web_search: null` / `~`: a block from here on
                lines = list(lines)
                lines[start] = "web_search:" + (f" {comment.strip()}" if comment.strip() else "") + "\n"
            new = _edit_block_section(lines, start, value)

    _verify(text, new, url)
    return new


def _split_comment(rest: str) -> tuple[str, str]:
    """(value, comment) for the remainder of a ``key:`` line. A '#' after
    whitespace starts a comment; one inside a quoted string does not."""
    quote = ""
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i], rest[i:]
    return rest, ""


def _rewrite_flow_section(
    lines: list[str], start: int, inline: str, comment: str, url: str,
) -> str:
    section = yaml.safe_load(inline)
    if not isinstance(section, dict):
        raise ConfigEditError("web_search is not a mapping")
    section = {**section, "searxng_url": url}
    body = yaml.safe_dump(section, default_flow_style=False, sort_keys=False)
    header = "web_search:" + (f" {comment.strip()}" if comment.strip() else "") + "\n"
    block = header + "".join(f"  {line}\n" for line in body.splitlines())
    return "".join(lines[:start]) + block + "".join(lines[start + 1:])


def _edit_block_section(lines: list[str], start: int, value: str) -> str:
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
        end += 1
    children = [
        i for i in range(start + 1, end)
        if lines[i].strip() and not lines[i].lstrip().startswith("#")
    ]
    indent = (
        lines[children[0]][: len(lines[children[0]]) - len(lines[children[0]].lstrip())]
        if children else "  "
    )
    key = re.compile(rf"^{re.escape(indent)}searxng_url:(?P<rest>[^\n]*)$")
    for i in children:
        m = key.match(lines[i].rstrip("\n"))
        if m is None:
            continue
        old_value, comment = _split_comment(m.group("rest"))
        gap = old_value[len(old_value.rstrip()):]
        newline = "\n" if lines[i].endswith("\n") else ""
        lines = list(lines)
        lines[i] = f"{indent}searxng_url: {value}{gap if comment else ''}{comment}{newline}"
        return "".join(lines)
    insert_at = (children[-1] + 1) if children else start + 1
    if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
        lines = list(lines)
        lines[insert_at - 1] += "\n"
    return "".join(lines[:insert_at] + [f"{indent}searxng_url: {value}\n"] + lines[insert_at:])


def _verify(before: str, after: str, url: str) -> None:
    try:
        old = yaml.safe_load(before) or {}
        new = yaml.safe_load(after)
    except yaml.YAMLError as exc:
        raise ConfigEditError(f"the edited config does not parse: {exc}") from exc
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise ConfigEditError("the config is not a mapping")
    section = old.get("web_search")
    if section is not None and not isinstance(section, dict):
        raise ConfigEditError("web_search is not a mapping")
    expected = {**old, "web_search": {**(section or {}), "searxng_url": url}}
    if new != expected:
        raise ConfigEditError("the edit would change more than web_search.searxng_url")


def _free_backup_path(path: Path) -> Path:
    """``<config>.bak``, or ``.bak.1``, ``.bak.2``… — never over an older backup."""
    base = path.with_suffix(path.suffix + ".bak")
    if not base.exists():
        return base
    for n in range(1, 1000):
        candidate = Path(f"{base}.{n}")
        if not candidate.exists():
            return candidate
    raise OSError(f"could not find a free backup name beside {base}")


def _atomic_write(path: Path, text: str) -> None:
    mode = path.stat().st_mode & 0o777
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{value} is not a TCP port")
    return port


def add_search_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``oara search`` on the main CLI parser."""
    parser = subparsers.add_parser("search", help="Set up web_search's backends")
    sub = parser.add_subparsers(dest="search_action")
    setup = sub.add_parser(
        "setup",
        help="Run SearXNG in Docker with JSON output on and point "
             "web_search.searxng_url at it",
    )
    setup.add_argument("--port", type=_port, default=DEFAULT_PORT,
                       help=f"host port on 127.0.0.1 (default {DEFAULT_PORT})")
    setup.add_argument("--name", default=DEFAULT_NAME,
                       help=f"container name (default {DEFAULT_NAME})")
    setup.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                       help=f"seconds to wait for JSON (default {DEFAULT_TIMEOUT:.0f})")
    setup.add_argument("--dry-run", action="store_true",
                       help="show what would be done; change nothing")


def run_search_command(args: argparse.Namespace) -> int:
    if getattr(args, "search_action", None) != "setup":
        print("Usage: oara search setup [--port N] [--name NAME] [--timeout S] [--dry-run]")
        return 2
    options = SetupOptions(
        config_path=getattr(args, "config", None), port=args.port, name=args.name,
        dry_run=args.dry_run, timeout=args.timeout,
    )
    return run_setup(options, docker=Docker.find())
