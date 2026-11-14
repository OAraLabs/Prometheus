"""``oara tailscale``: set up, check or turn off HTTPS over Tailscale with ``tailscale serve``.

The daemon side is ``web/tailscale_serve.py``: a daemon bound to loopback answers what Serve relays to it. This sets
the two Serve entries and the ``web.tailscale`` block, or checks them, or removes them.

* ``check`` (the default): read-only. What is there, what setup would do, and what stops it.
* ``setup``: prints the plan and changes nothing without ``--apply``. With it, runs the two Serve commands it needs
  and writes ``web.tailscale`` (comment-preserving, atomic, like the network settings).
* ``off``: removes only the Serve entries that relay exactly to this daemon's own ports (``/`` on this node's
  name, over HTTPS, to its loopback port) and sets ``web.tailscale.enabled: false``. Needs ``--apply`` too.

What it never does: run ``sudo``, or ``tailscale cert``, ``funnel``, ``up``, ``login``, ``logout``, ``down`` or
``set``; replace a Serve entry it did not create (a port another app serves is reported, not touched); or turn
Funnel on. When Serve refuses for permission, it stops and prints the one command to run as the right user.

The only Tailscale commands it runs, checked shape by shape (:func:`run_tailscale`): ``status --json``,
``serve status --json``, ``serve --bg --https=<port> http://<loopback>:<port>`` and ``serve --https=<port> off``.
The ``tailscale`` it runs is the one on PATH (or the macOS app's), the same one typing ``tailscale`` would run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

_PORT = re.compile(r"[1-9][0-9]{0,4}")
#: The relay target ``setup`` writes: plain HTTP to a loopback literal (or localhost) on a port.
_TARGET = re.compile(r"http://(?:127(?:\.[0-9]{1,3}){3}|localhost|\[::1\]):([1-9][0-9]{0,4})")
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


def _port_text_ok(text: str) -> bool:
    return bool(_PORT.fullmatch(text)) and int(text) <= 65535


def allowed_command(args: tuple[str, ...]) -> bool:
    """Is *args* exactly one of the four commands this module runs? Every word and every port is checked."""
    if args in (("status", "--json"), ("serve", "status", "--json")):
        return True
    if len(args) == 4 and args[:2] == ("serve", "--bg") and args[2].startswith("--https="):
        target = _TARGET.fullmatch(args[3])
        return _port_text_ok(args[2][len("--https="):]) and target is not None and _port_text_ok(target.group(1))
    if len(args) == 3 and args[0] == "serve" and args[1].startswith("--https=") and args[2] == "off":
        return _port_text_ok(args[1][len("--https="):])
    return False


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def run_tailscale(binary: str, *args: str, timeout: float = TIMEOUT_S) -> tuple[int, str, str]:
    """Run ``tailscale <args>``: only the four commands in the module docstring (:func:`allowed_command`), never
    through a shell, with no terminal input (a prompt cannot wait unseen) and UTF-8 whatever the locale."""
    if not allowed_command(tuple(args)):
        raise TailscaleCommandRefused(f"refusing to run tailscale {' '.join(args)}")
    try:
        done = subprocess.run([binary, *args], capture_output=True, stdin=subprocess.DEVNULL, encoding="utf-8",
                              errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        said = (_text(exc.stderr) or _text(exc.stdout)).strip()
        return 124, "", f"tailscale {' '.join(args)} did not answer within {timeout:.0f} s" + (
            f" (it said: {said[:300]})" if said else "")
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
    #: Serve terminates TLS on this port (``TCP[port].HTTPS``). Parsing sets it from the config; a plain-HTTP
    #: entry is never ours.
    https: bool = True


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

    seen_in: dict[int, set[int]] = {}
    https_on: set[int] = set()

    def entry(port: int, layer_index: int) -> ServeEntry:
        seen_in.setdefault(port, set()).add(layer_index)
        return entries.setdefault(port, ServeEntry(port=port, https=False))

    for index, layer in enumerate(layers):
        tcp = layer.get("TCP")
        if isinstance(tcp, Mapping):
            for port_text, handler in tcp.items():
                if not str(port_text).isdigit():
                    continue
                e = entry(int(port_text), index)
                if isinstance(handler, Mapping) and (handler.get("TCPForward") or handler.get("TerminateTLS")):
                    e.other = True
                if isinstance(handler, Mapping) and handler.get("HTTPS") is True:
                    https_on.add(e.port)
        web = layer.get("Web")
        if isinstance(web, Mapping):
            for host_port, site in web.items():
                host, _, port_text = str(host_port).rpartition(":")
                if not port_text.isdigit():
                    continue
                e = entry(int(port_text), index)
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
                    entry(int(port_text), index).funnel = True
    for port, e in entries.items():
        e.https = port in https_on
        if len(seen_in.get(port, ())) > 1:
            e.other = True                                # one port in two layers: which one answers is not ours
    return entries


def read_state(binary: str | None = None, *, timeout: float = TIMEOUT_S) -> TailscaleState:
    """Ask Tailscale what is there. Read-only."""
    state = TailscaleState(binary=binary or tailscale_binary())
    if state.binary is None:
        state.problems.append("the tailscale command was not found (install Tailscale, or put its CLI on PATH)")
        return state
    rc, out, err = run_tailscale(state.binary, "status", "--json", timeout=timeout)
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
    rc, out, err = run_tailscale(state.binary, "serve", "status", "--json", timeout=timeout)
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
    from prometheus.web.loopback import is_loopback_address

    text = target if "://" in target else f"http://{target}"
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return False
    return parts.scheme == "http" and is_loopback_address(parts.hostname or "") and port == local_port


def classify(entry: ServeEntry | None, local_port: int) -> str:
    """``free``, ``ours`` (exactly ``/``, over HTTPS, relayed to this daemon's loopback port, in one layer) or
    ``taken``."""
    if entry is None or (not entry.proxies and not entry.other):
        return "free"
    if (not entry.other and entry.https and set(entry.proxies) == {"/"}
            and _relays_to(entry.proxies["/"], local_port)):
        return "ours"
    return "taken"


def effective_bind(config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(address, None)``: the address the daemon will listen on, resolved the way it does (the environment and the
    env file over ``web.bind``, an unset bind meaning every interface, the Mac app's LAN guard); or
    ``(None, why)`` when it cannot be resolved."""
    from prometheus.config.api_token import parse_env_file
    from prometheus.web.bind import BindError, app_install_lan_guard, resolve_bind

    env = {**parse_env_file(), **os.environ}
    try:
        resolved = resolve_bind(dict(config), env=env)
    except BindError as exc:
        return None, str(exc)
    return app_install_lan_guard(resolved, dict(config), env=env, quiet=True).address, None


def _relay_host(bind: str | None) -> str:
    """The host the relay target names: the daemon's own loopback address (``[::1]`` for IPv6; an IPv4-mapped
    address as its IPv4 form)."""
    import ipaddress

    if bind is None or bind == "localhost":
        return "127.0.0.1"
    try:
        ip = ipaddress.ip_address(bind)
    except ValueError:
        return "127.0.0.1"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return f"[{ip}]" if ip.version == 6 else str(ip)


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
    bind, why = effective_bind(config)
    if bind is None:
        plan.blockers.append(f"the daemon's bind cannot be resolved: {why}")
    elif not is_loopback_address(bind):
        plan.blockers.append(f"the daemon will listen on {bind}, not loopback (web.bind, PROMETHEUS_WEB_BIND, or "
                             "unset, which is every interface): Serve in front of a wide bind exposes the daemon both "
                             "ways. Set web.bind: 127.0.0.1 first, once your devices can re-pair.")
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
            plan.commands.append(["serve", "--bg", f"--https={port}", f"http://{_relay_host(bind)}:{local}"])
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
    if not plan.blockers:
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
    block = _web(config).get("tailscale")
    current: Mapping[str, Any] = block if isinstance(block, Mapping) else {}
    try:
        rest_port = _pick_port(getattr(args, "rest_port", None), current.get("rest_port"), 443, "rest-port")
        ws_port = _pick_port(getattr(args, "ws_port", None), current.get("ws_port"), 8443, "ws-port")
    except ValueError as exc:
        print(f"{exc}. Nothing was changed.", file=out)
        return 1
    state = read_state()

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
    # The config edit is checked BEFORE anything runs: a file this cannot edit must not be left behind live relays.
    if _write(args, _block_values(plan), out, check_only=True) != 0:
        print("Nothing was changed.", file=out)
        return 1
    added: list[list[str]] = []
    for command in plan.commands:
        rc, so, se = run_tailscale(state.binary or "tailscale", *command)
        if rc != 0:
            said = (se or so).strip()
            print(f"tailscale {' '.join(command)} failed: {said[:300] or f'exit {rc}'}", file=out)
            if any(h in said.lower() for h in _PERMISSION_HINTS):
                print("Serve needs this user to be Tailscale's operator. Run this yourself, once, then try again:\n"
                      "  sudo tailscale set --operator=$USER", file=out)
            for done in added:
                print(f"already added: tailscale {' '.join(done)} (`oara tailscale off --apply`, with the same "
                      "--rest-port and --ws-port, removes it)", file=out)
            print("Stopped. The config was not changed.", file=out)
            return 1
        added.append(command)
        print(f"ran: tailscale {' '.join(command)}", file=out)
    return _write(args, _block_values(plan), out)


def _pick_port(flag: Any, configured: Any, default: int, label: str) -> int:
    """The port a flag, else the config, else the default names; ValueError (said plainly) for anything that is not
    a port from 1 to 65535 (0, true, a word, a list)."""
    from prometheus.web.tailscale_serve import _valid_port

    for value, source in ((flag, f"--{label}"), (configured, f"web.tailscale.{label.replace('-', '_')}")):
        if value is None:
            continue
        if not _valid_port(value):
            raise ValueError(f"{source} must be a port from 1 to 65535, not {value!r}")
        return int(value)
    return default


def _write(args: argparse.Namespace, values: dict[tuple[str, ...], Any], out: Any, *, check_only: bool = False) -> int:
    from prometheus.config.defaults import resolve_config_path
    from prometheus.web.network import PersistError, persist_values

    try:
        wrote = persist_values(resolve_config_path(getattr(args, "config", None)), values, "web.tailscale",
                               check_only=check_only)
    except PersistError as exc:
        print(f"Could not write the config: {exc}", file=out)
        return 1
    if check_only:
        return 0
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
        print(f"WARNING: {problem}. No Serve entry can be seen, so none is removed; turning web.tailscale off still "
              "makes the daemon answer no relayed request.", file=out)
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
                handshake: Any = None, timeout: float = 5.0) -> list[Any]:
    """The "Tailscale HTTPS" rows: none at all when ``web.tailscale`` is absent. Each says what it could not check."""
    from prometheus.infra.doctor import DiagnosticCheck
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
    state = state if state is not None else read_state(timeout=max(timeout, 1.0) * 3)
    if state.problems:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "error", "; ".join(state.problems),
                                    fix="Serve, Funnel and the certificate were not checked."))
        return rows
    if not state.running:
        rows.append(DiagnosticCheck("Tailscale", "connectivity", "error",
                                    f"not running (BackendState {state.backend_state or 'unknown'})",
                                    fix="Start Tailscale. Serve, Funnel and the certificate were not checked."))
        return rows
    if state.name != settings.name:
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
    bind, why = effective_bind(config)
    if bind is None or not is_loopback_address(bind):
        rows.append(DiagnosticCheck("Tailscale bind", "connectivity", "error",
                                    f"the daemon will listen on {bind or why}, not loopback (web.bind, "
                                    "PROMETHEUS_WEB_BIND, or unset, which is every interface): Serve plus a wide bind "
                                    "exposes the daemon both ways", fix="Set web.bind: 127.0.0.1."))
    else:
        rows.append(DiagnosticCheck("Tailscale bind", "connectivity", "ok", f"the daemon listens on {bind}"))
    shake = handshake or (lambda name, port: tls_days_left(name, port, timeout=timeout))
    rows.append(_certificate_row(settings.name, settings.rest_port, shake))
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
