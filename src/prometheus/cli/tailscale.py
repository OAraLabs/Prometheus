"""``oara tailscale``: set up, check or turn off HTTPS over Tailscale with ``tailscale serve``.

The daemon side is ``web/tailscale_serve.py``: a daemon bound to loopback answers what Serve relays to it. This sets
the two Serve entries and the ``web.tailscale`` block, or checks them, or removes them.

* ``check`` (the default): read-only. What is there, what setup would do, and what stops it.
* ``setup``: prints the plan and changes nothing without ``--apply``. With it, runs the two Serve commands it needs
  and writes ``web.tailscale`` (comment-preserving, atomic, like the network settings).
* ``off``: removes only the two Serve entries this set up (that relay to this daemon's own ports) and sets
  ``web.tailscale.enabled: false``. Needs ``--apply`` too.

What it never does: run ``sudo``, or ``tailscale cert``, ``funnel``, ``up``, ``login``, ``logout``, ``down`` or
``set``; replace a Serve entry it did not create (a port another app serves is reported, not touched); or turn
Funnel on. When Serve refuses for permission, it stops and prints the one command to run as the right user.

The only Tailscale commands it runs: ``status --json``, ``serve status --json``,
``serve --bg --https=<port> http://127.0.0.1:<port>`` and ``serve --https=<port> off``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: The only first words of a Tailscale command this module runs.
_ALLOWED_VERBS = frozenset({"status", "serve"})
#: Words that never appear anywhere in a command this module runs.
_FORBIDDEN = frozenset({"sudo", "cert", "funnel", "--funnel", "up", "login", "logout", "down", "set"})
#: Where the macOS app keeps its command-line tool when it is not on PATH.
_MAC_APP_CLI = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
TIMEOUT_S = 15.0


class TailscaleCommandRefused(RuntimeError):
    """A command this module must never run was asked for (a bug, never a user's input)."""


def tailscale_binary() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    if sys.platform == "darwin" and os.path.exists(_MAC_APP_CLI):
        return _MAC_APP_CLI
    return None


def run_tailscale(binary: str, *args: str, timeout: float = TIMEOUT_S) -> tuple[int, str, str]:
    """Run ``tailscale <args>`` (only the commands listed in the module docstring). Never through a shell."""
    if not args or args[0] not in _ALLOWED_VERBS or any(a.lower() in _FORBIDDEN for a in args):
        raise TailscaleCommandRefused(f"refusing to run tailscale {' '.join(args)}")
    try:
        done = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 124, "", f"tailscale {' '.join(args)} did not answer within {timeout:.0f} s"
    except OSError as exc:
        return 127, "", f"could not run {binary}: {exc}"
    return done.returncode, done.stdout or "", done.stderr or ""


# ── what is there ────────────────────────────────────────────────────────────

@dataclass
class ServeEntry:
    """What Serve does on one HTTPS port."""

    port: int
    proxies: dict[str, str] = field(default_factory=dict)   # mount path -> target
    other: bool = False                                     # a TCP forward, a file, a text handler: not ours to judge
    funnel: bool = False


@dataclass
class TailscaleState:
    binary: str | None = None
    running: bool = False
    backend_state: str = ""
    name: str = ""                  # this node's MagicDNS name, lowercase, no trailing dot
    cert_domains: list[str] = field(default_factory=list)
    entries: dict[int, ServeEntry] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _parse_serve(config: Any, name: str) -> dict[int, ServeEntry]:
    """The entries of a ``tailscale serve status --json`` answer, foreground sessions included."""
    entries: dict[int, ServeEntry] = {}
    if not isinstance(config, Mapping):
        return entries
    layers = [config]
    foreground = config.get("Foreground")
    if isinstance(foreground, Mapping):
        layers.extend(v for v in foreground.values() if isinstance(v, Mapping))

    def entry(port: int) -> ServeEntry:
        return entries.setdefault(port, ServeEntry(port=port))

    for layer in layers:
        tcp = layer.get("TCP")
        if isinstance(tcp, Mapping):
            for port_text, handler in tcp.items():
                if not str(port_text).isdigit():
                    continue
                e = entry(int(port_text))
                if isinstance(handler, Mapping) and (handler.get("TCPForward") or handler.get("TerminateTLS")):
                    e.other = True
        web = layer.get("Web")
        if isinstance(web, Mapping):
            for host_port, site in web.items():
                host, _, port_text = str(host_port).rpartition(":")
                if not port_text.isdigit():
                    continue
                e = entry(int(port_text))
                if name and host.lower().rstrip(".") != name:
                    e.other = True
                handlers = site.get("Handlers") if isinstance(site, Mapping) else None
                for path, handler in (handlers or {}).items() if isinstance(handlers, Mapping) else ():
                    target = handler.get("Proxy") if isinstance(handler, Mapping) else None
                    if isinstance(target, str) and target:
                        e.proxies[str(path)] = target
                    else:
                        e.other = True
        funnel = layer.get("AllowFunnel")
        if isinstance(funnel, Mapping):
            for host_port, on in funnel.items():
                _, _, port_text = str(host_port).rpartition(":")
                if port_text.isdigit() and on:
                    entry(int(port_text)).funnel = True
    return entries


def read_state(binary: str | None = None) -> TailscaleState:
    """Ask Tailscale what is there. Read-only."""
    state = TailscaleState(binary=binary or tailscale_binary())
    if state.binary is None:
        state.problems.append("the tailscale command was not found (install Tailscale, or put its CLI on PATH)")
        return state
    rc, out, err = run_tailscale(state.binary, "status", "--json")
    if rc != 0:
        state.problems.append(f"tailscale status failed: {(err or out).strip()[:200] or f'exit {rc}'}")
        return state
    try:
        status = json.loads(out)
    except ValueError:
        state.problems.append("tailscale status --json did not answer JSON")
        return state
    state.backend_state = str(status.get("BackendState") or "") if isinstance(status, Mapping) else ""
    state.running = state.backend_state == "Running"
    me = status.get("Self") if isinstance(status, Mapping) else None
    state.name = str((me or {}).get("DNSName") or "").rstrip(".").lower() if isinstance(me, Mapping) else ""
    domains = status.get("CertDomains") if isinstance(status, Mapping) else None
    state.cert_domains = [str(d).rstrip(".").lower() for d in domains] if isinstance(domains, list) else []
    rc, out, err = run_tailscale(state.binary, "serve", "status", "--json")
    if rc != 0:
        state.problems.append(f"tailscale serve status failed: {(err or out).strip()[:200] or f'exit {rc}'}")
        return state
    try:
        serve = json.loads(out) if out.strip() else {}
    except ValueError:
        state.problems.append("tailscale serve status --json did not answer JSON")
        return state
    state.entries = _parse_serve(serve, state.name)
    return state


def _web(config: Mapping[str, Any]) -> Mapping[str, Any]:
    web = config.get("web")
    return web if isinstance(web, Mapping) else {}


def _local_ports(config: Mapping[str, Any]) -> tuple[int, int]:
    web = _web(config)
    try:
        api = int(web.get("api_port", 8005) or 8005)
    except (TypeError, ValueError):
        api = 8005
    try:
        ws = int(web.get("ws_port", 8010) or 8010)
    except (TypeError, ValueError):
        ws = 8010
    return api, ws


def _relays_to(target: str, local_port: int) -> bool:
    text = target if "://" in target else f"http://{target}"
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return False
    return parts.scheme == "http" and (parts.hostname or "") in ("127.0.0.1", "localhost") and port == local_port


def classify(entry: ServeEntry | None, local_port: int) -> str:
    """``free``, ``ours`` (exactly ``/`` relayed to this daemon's local port) or ``taken``."""
    if entry is None or (not entry.proxies and not entry.other):
        return "free"
    if not entry.other and set(entry.proxies) == {"/"} and _relays_to(entry.proxies["/"], local_port):
        return "ours"
    return "taken"


# ── the plan ─────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    name: str
    rest_port: int
    ws_port: int
    blockers: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)   # tailscale arguments
    notes: list[str] = field(default_factory=list)


def _has_token(config: Mapping[str, Any]) -> bool:
    from prometheus.config.api_token import ApiTokenFileError, resolve_api_token

    try:
        token, _source = resolve_api_token(dict(config))
    except ApiTokenFileError:
        return False
    return bool(token)


def plan_setup(config: Mapping[str, Any], state: TailscaleState, *, rest_port: int, ws_port: int) -> Plan:
    from prometheus.web.bind import BindError, parse_bind
    from prometheus.web.loopback import is_loopback_address
    from prometheus.web.tailscale_serve import _trusts_loopback_proxy, _valid_name, _valid_port

    plan = Plan(name=state.name, rest_port=rest_port, ws_port=ws_port, blockers=list(state.problems))
    if state.problems:
        return plan
    if not state.running:
        plan.blockers.append(f"Tailscale is not running here (BackendState {state.backend_state or 'unknown'})")
    if not _valid_name(state.name):
        plan.blockers.append(f"this node's MagicDNS name {state.name!r} is not a <machine>.<tailnet>.ts.net name "
                             "(turn MagicDNS on in the admin console)")
    elif state.name not in state.cert_domains:
        plan.blockers.append("HTTPS certificates are off for this tailnet: in the admin console, DNS page, turn on "
                             "HTTPS Certificates. The machine name then goes into public certificate logs.")
    if not (_valid_port(rest_port) and _valid_port(ws_port)) or rest_port == ws_port:
        plan.blockers.append("--rest-port and --ws-port must be two different ports from 1 to 65535")
    web = _web(config)
    try:
        bind = parse_bind(web.get("bind", "127.0.0.1"), "web.bind")
    except BindError as exc:
        bind = None
        plan.blockers.append(str(exc))
    if bind is not None and not is_loopback_address(bind):
        plan.blockers.append(f"web.bind is {bind}, not loopback: Serve in front of a wide bind exposes the daemon "
                             "both ways. Set web.bind to 127.0.0.1 first (keep it only once your devices re-pair).")
    if not _has_token(config):
        plan.blockers.append("this daemon has no API token, and an open daemon must not answer the tailnet "
                             "(run `oara token rotate`)")
    if _trusts_loopback_proxy(dict(config)):
        plan.blockers.append("web.trusted_proxies covers this machine: remove the loopback entry (the tailscale "
                             "block replaces it)")
    api, ws_local = _local_ports(config)
    for port, local, what in ((rest_port, api, "REST"), (ws_port, ws_local, "WebSocket")):
        entry = state.entries.get(port)
        kind = classify(entry, local)
        if entry is not None and entry.funnel:
            plan.blockers.append(f"Funnel is on for port {port}: that serves the public internet. Turn it off "
                                 f"yourself (`tailscale funnel --https={port} off`); this never touches Funnel.")
        if kind == "taken":
            plan.blockers.append(f"port {port} already serves something else; it is left alone. Pick another port "
                                 f"(--{'rest' if what == 'REST' else 'ws'}-port).")
        elif kind == "ours":
            plan.notes.append(f"port {port} already relays to the daemon's {what} port {local}: kept")
        else:
            plan.commands.append(["serve", "--bg", f"--https={port}", f"http://127.0.0.1:{local}"])
    return plan


def _block_values(plan: Plan) -> dict[tuple[str, ...], Any]:
    return {("web", "tailscale", "enabled"): True, ("web", "tailscale", "name"): plan.name,
            ("web", "tailscale", "rest_port"): plan.rest_port, ("web", "tailscale", "ws_port"): plan.ws_port}


def _print_plan(plan: Plan, out: Any) -> None:
    print(f"HTTPS over Tailscale for {plan.name or '(unknown name)'}:", file=out)
    rest = f"https://{plan.name}" + ("" if plan.rest_port == 443 else f":{plan.rest_port}")
    print(f"  REST      {rest}", file=out)
    print(f"  WebSocket wss://{plan.name}:{plan.ws_port}", file=out)
    for note in plan.notes:
        print(f"  - {note}", file=out)
    for command in plan.commands:
        print(f"  will run: tailscale {' '.join(command)}", file=out)
    print("  will write: web.tailscale {enabled: true, name, rest_port, ws_port}", file=out)
    print("  will NOT: run sudo, cert, funnel, up or set; replace another app's Serve entry; turn Funnel on", file=out)
    for blocker in plan.blockers:
        print(f"  BLOCKED: {blocker}", file=out)


_PERMISSION_HINTS = ("access denied", "permission denied", "not permitted", "must be root", "operator")


def run_tailscale_command(args: argparse.Namespace, config: dict[str, Any]) -> int:
    """Entry point for ``oara tailscale``. 0 when done (or nothing to do), 1 when blocked or failed."""
    from prometheus.web.tailscale_serve import serve_settings

    out = sys.stdout
    action = getattr(args, "tailscale_action", None) or "check"
    state = read_state()
    block = _web(config).get("tailscale")
    current: Mapping[str, Any] = block if isinstance(block, Mapping) else {}
    rest_port = int(getattr(args, "rest_port", None) or current.get("rest_port") or 443)
    ws_port = int(getattr(args, "ws_port", None) or current.get("ws_port") or 8443)

    if action == "off":
        return _off(args, config, state, rest_port, ws_port, out)

    plan = plan_setup(config, state, rest_port=rest_port, ws_port=ws_port)
    _print_plan(plan, out)
    if action == "check":
        configured = serve_settings(config, has_token=_has_token(config))
        print(f"  web.tailscale: {'on' if configured else 'off'} in the config", file=out)
        return 1 if plan.blockers else 0
    if plan.blockers:
        print("Nothing was changed.", file=out)
        return 1
    if not getattr(args, "apply", False):
        print("Dry run: nothing was changed. Run again with --apply to do it.", file=out)
        return 0
    for command in plan.commands:
        rc, so, se = run_tailscale(state.binary or "tailscale", *command)
        if rc != 0:
            said = (se or so).strip()
            print(f"tailscale {' '.join(command)} failed: {said[:300] or f'exit {rc}'}", file=out)
            if any(h in said.lower() for h in _PERMISSION_HINTS):
                print("Serve needs this user to be Tailscale's operator. Run this yourself, once, then try again:\n"
                      "  sudo tailscale set --operator=$USER", file=out)
            print("Stopped. The config was not changed.", file=out)
            return 1
        print(f"ran: tailscale {' '.join(command)}", file=out)
    return _write(args, _block_values(plan), out)


def _write(args: argparse.Namespace, values: dict[tuple[str, ...], Any], out: Any) -> int:
    from prometheus.config.defaults import resolve_config_path
    from prometheus.web.network import PersistError, persist_values

    try:
        wrote = persist_values(resolve_config_path(getattr(args, "config", None)), values, "web.tailscale")
    except PersistError as exc:
        print(f"Could not write the config: {exc}", file=out)
        return 1
    print("wrote web.tailscale to the config. Restart the daemon to apply it." if wrote
          else "web.tailscale was already set that way.", file=out)
    return 0


def _off(args: argparse.Namespace, config: Mapping[str, Any], state: TailscaleState, rest_port: int, ws_port: int,
         out: Any) -> int:
    api, ws_local = _local_ports(config)
    commands: list[list[str]] = []
    for port, local in ((rest_port, api), (ws_port, ws_local)):
        kind = classify(state.entries.get(port), local)
        if kind == "ours":
            commands.append(["serve", f"--https={port}", "off"])
        elif kind == "taken":
            print(f"port {port} serves something else: left alone", file=out)
    for problem in state.problems:
        print(f"note: {problem}", file=out)
    for command in commands:
        print(f"will run: tailscale {' '.join(command)}", file=out)
    print("will write: web.tailscale.enabled: false", file=out)
    if not getattr(args, "apply", False):
        print("Dry run: nothing was changed. Run again with --apply to do it.", file=out)
        return 0
    for command in commands:
        rc, so, se = run_tailscale(state.binary or "tailscale", *command)
        if rc != 0:
            print(f"tailscale {' '.join(command)} failed: {(se or so).strip()[:300] or f'exit {rc}'}", file=out)
            print("Stopped. The config was not changed.", file=out)
            return 1
        print(f"ran: tailscale {' '.join(command)}", file=out)
    return _write(args, {("web", "tailscale", "enabled"): False}, out)


# ── doctor rows ──────────────────────────────────────────────────────────────

def doctor_rows(config: Mapping[str, Any], *, state: TailscaleState | None = None,
                handshake: Any = None) -> list[Any]:
    """The "Tailscale HTTPS" rows: none at all when ``web.tailscale`` is absent. Each says what it could not check."""
    from prometheus.infra.doctor import DiagnosticCheck
    from prometheus.web.bind import BindError, parse_bind
    from prometheus.web.loopback import is_loopback_address
    from prometheus.web.tailscale_serve import serve_settings

    web = _web(config)
    if web.get("tailscale") is None:
        return []
    block = web.get("tailscale")
    if isinstance(block, Mapping) and block.get("enabled") is False:
        return [DiagnosticCheck("Tailscale HTTPS", "connectivity", "info", "off (web.tailscale.enabled: false)")]
    rows: list[Any] = []
    settings = serve_settings(dict(config), has_token=_has_token(config))
    if settings is None:
        return [DiagnosticCheck("Tailscale HTTPS", "connectivity", "error",
                                "web.tailscale is set but the daemon ignores it (bad name or ports, no API token, or "
                                "trusted_proxies covering loopback; the daemon log says which)",
                                fix="Run `oara tailscale check`.")]
    state = state if state is not None else read_state()
    if state.problems:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "error", "; ".join(state.problems),
                                    fix="Serve, Funnel and the certificate were not checked."))
        return rows
    if not state.running:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "error",
                                    f"not running (BackendState {state.backend_state or 'unknown'})"))
    elif state.name != settings.name:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "error",
                                    f"this node is {state.name!r}, but web.tailscale.name is {settings.name!r}",
                                    fix="Run `oara tailscale setup` again."))
    else:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "ok", f"running as {state.name}"))
    api, ws_local = _local_ports(config)
    serve_bad: list[str] = []
    serve_missing: list[str] = []
    funnel: list[int] = []
    for port, local in ((settings.rest_port, api), (settings.ws_port, ws_local)):
        entry = state.entries.get(port)
        kind = classify(entry, local)
        if kind == "taken":
            serve_bad.append(f"port {port} serves something else")
        elif kind == "free":
            serve_missing.append(f"port {port} serves nothing")
        if entry is not None and entry.funnel:
            funnel.append(port)
    if serve_bad:
        rows.append(DiagnosticCheck("Tailscale Serve", "connectivity", "error", "; ".join(serve_bad),
                                    fix="Pick free ports with `oara tailscale setup --rest-port … --ws-port …`."))
    elif serve_missing:
        rows.append(DiagnosticCheck("Tailscale Serve", "connectivity", "warning", "; ".join(serve_missing),
                                    fix="Run `oara tailscale setup --apply`."))
    else:
        rows.append(DiagnosticCheck("Tailscale Serve", "connectivity", "ok",
                                    f"{settings.rest_port} -> 127.0.0.1:{api}, {settings.ws_port} -> 127.0.0.1:{ws_local}"))
    rows.append(DiagnosticCheck("Tailscale Funnel", "connectivity", "error" if funnel else "ok",
                                f"ON for port(s) {', '.join(map(str, funnel))}: the public internet reaches them"
                                if funnel else "off on both ports",
                                fix=(f"tailscale funnel --https={funnel[0]} off" if funnel else None)))
    try:
        bind = parse_bind(web.get("bind", "127.0.0.1"), "web.bind")
    except BindError:
        bind = None
    if bind is None or not is_loopback_address(bind):
        rows.append(DiagnosticCheck("Tailscale bind", "connectivity", "error",
                                    f"web.bind is {web.get('bind')!r}, not loopback: Serve plus a wide bind exposes "
                                    "the daemon both ways", fix="Set web.bind: 127.0.0.1."))
    else:
        rows.append(DiagnosticCheck("Tailscale bind", "connectivity", "ok", f"web.bind is {bind}"))
    rows.append(_certificate_row(settings.name, settings.rest_port, handshake or tls_days_left))
    return rows


def tls_days_left(name: str, port: int, timeout: float = 5.0) -> float:
    """Days until the certificate *name*:*port* serves expires, verified against the default trust store for that
    name. Raises on any failure (no answer, an untrusted or wrong-name certificate)."""
    import socket
    import ssl
    import time

    context = ssl.create_default_context()
    with socket.create_connection((name, port), timeout=timeout) as raw, \
            context.wrap_socket(raw, server_hostname=name) as tls:
        cert = tls.getpeercert()
    not_after = ssl.cert_time_to_seconds(str((cert or {}).get("notAfter")))
    return (not_after - time.time()) / 86400.0


def _certificate_row(name: str, port: int, handshake: Any) -> Any:
    from prometheus.infra.doctor import DiagnosticCheck

    try:
        days = float(handshake(name, port))
    except Exception as exc:                                   # noqa: BLE001 - every failure is the row's answer
        return DiagnosticCheck("Tailscale certificate", "connectivity", "error",
                               f"https://{name}:{port} did not verify ({type(exc).__name__}: {str(exc)[:120]})",
                               fix="Check Serve is on (`oara tailscale check`) and HTTPS Certificates are enabled.")
    if days <= 0:
        return DiagnosticCheck("Tailscale certificate", "connectivity", "error", "the certificate has expired")
    if days < 14:
        return DiagnosticCheck("Tailscale certificate", "connectivity", "warning",
                               f"verifies, but expires in {days:.0f} days (Serve renews it; check it renewed)")
    return DiagnosticCheck("Tailscale certificate", "connectivity", "ok", f"verifies; {days:.0f} days left")


def add_tailscale_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tailscale", help="HTTPS over Tailscale with tailscale serve: check, setup, off")
    sub = p.add_subparsers(dest="tailscale_action")
    for name, help_text in (("check", "read-only: what is there and what setup would do (the default)"),
                            ("setup", "set the two Serve entries and web.tailscale (dry run without --apply)"),
                            ("off", "remove the two Serve entries this set up and turn web.tailscale off")):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--config", default=argparse.SUPPRESS, help="the prometheus.yaml to read and write")
        if name != "check":
            q.add_argument("--apply", action="store_true", help="do it (without this, only print the plan)")
        q.add_argument("--rest-port", type=int, default=None, help="the Serve HTTPS port for REST (default 443)")
        q.add_argument("--ws-port", type=int, default=None, help="the Serve HTTPS port for the WebSocket (default 8443)")
