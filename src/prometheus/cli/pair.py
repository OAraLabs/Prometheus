"""``oara pair`` — see and decide the devices that are asking to join this Prometheus.

    oara pair list
    oara pair approve <id> [--name NAME] [--code NNNN]
    oara pair deny <id>
    oara pair --owner

``--owner`` makes a Beacon (on this computer or another) the owner's own device (an owner device: the operator, and a person
for a worker's approval). It asks the daemon for a one-time code (8 digits, 5 minutes, once; only the API token from
this machine may), prints it, and waits to say what became of it: who used it and from which address, or that it
expired or was replaced, and how many wrong tries there were. The report is the alarm: a code someone else typed
first shows up here at once. Ctrl-C cancels the code. It prints no token. It asks for a code only when its output is
a terminal: piped, captured, or run by the agent's own shell, the code would land in a transcript or a log.

For a headless box with no Beacon open and no Telegram (docs/PAIRING-APPROVAL-API.md, 5.3). It is the same
REST routes the other channels use, with the global token read the way ``oara token show`` reads it, and it
sends ``X-Pairing-Via: cli`` so the owner's other screens say "approved in the terminal". The daemon decides;
this only asks. An id may be given as a unique prefix of at least 6 characters, and ambiguity or an unknown id
is refused rather than guessed, because approving the wrong device hands it a credential.

It prints neither a token (the approve response has none; the sealed copy goes to the new device) nor a poll
secret (the list does not carry it).

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from prometheus.config.api_token import resolve_api_token
from prometheus.config.env_file import parse_env_file
from prometheus.config.owner_codes import format_code
from prometheus.web.bind import BindError, format_host_port, is_all_interfaces, resolve_bind

MIN_PREFIX = 6
_TIMEOUT_SECONDS = 10.0
#: How often ``--owner`` asks what became of its code, and how many times at most (past the code's own life, the
#: daemon has answered "expired" long before).
OWNER_POLL_SECONDS = 1.0
_OWNER_MAX_POLLS = 400


def add_pair_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``pair`` subcommand on the main CLI parser."""
    pair = subparsers.add_parser("pair", help="List, approve or deny devices asking to join this Prometheus")
    pair.add_argument("--owner", action="store_true",
                      help="print a one-time code that makes a Beacon (on this computer or another) your own (owner) device")
    sub = pair.add_subparsers(dest="pair_action")
    sub.add_parser("list", help="Show the devices waiting for a decision")
    approve = sub.add_parser("approve", help="Let a waiting device join (it gets its own scoped token)")
    approve.add_argument("request", help="the request id, or a unique prefix of at least 6 characters")
    approve.add_argument("--name", help="enrol it under this name instead of the one it typed")
    approve.add_argument("--code", help="the 4-digit code shown on the new device; approve only if it matches")
    deny = sub.add_parser("deny", help="Refuse a waiting device")
    deny.add_argument("request", help="the request id, or a unique prefix of at least 6 characters")


def _port(config: dict[str, Any] | None) -> int:
    try:
        return int(((config or {}).get("web") or {}).get("api_port") or 8005)
    except (TypeError, ValueError):
        return 8005


def daemon_base_url(config: dict[str, Any] | None, env: Mapping[str, str] | None = None) -> str:
    """Where this machine reaches the daemon: the address it listens on, or loopback when that is every interface.

    In home-network mode plain HTTP listens on loopback only (TLS takes the bind address), so that mode is reached
    over loopback. Otherwise, a daemon bound to ONE address (a LAN or tailnet address, how an owner narrows a headless
    box) does not answer on 127.0.0.1, so the terminal route used to report "could not reach the daemon" about a daemon that
    was running. The address is resolved the way the daemon resolves it (``PROMETHEUS_WEB_BIND``, from the
    environment or the env file the daemon loads, then ``web.bind``). A ``--bind`` flag given to a running
    daemon cannot be seen from here, which is why an unreachable daemon is reported with the address tried.
    A bind that cannot be honoured is the daemon's complaint (it refuses to start); here it falls back to
    loopback.
    """
    environment = {**parse_env_file(), **os.environ} if env is None else env
    try:
        address = resolve_bind(config, env=environment).address
    except BindError:
        address = "127.0.0.1"
    if is_all_interfaces(address):
        address = "::1" if ":" in address else "127.0.0.1"
    elif _plain_http_on_loopback(config, address):
        address = "127.0.0.1"
    return f"http://{format_host_port(address, _port(config))}"


def _plain_http_on_loopback(config: dict[str, Any] | None, address: str) -> bool:
    """Home-network mode keeps plain HTTP on this machine only and puts TLS on the LAN bind, so a terminal here must
    use loopback. The daemon's own decision (``web.tls.decide_lan``), with "is there a certificate" answered the
    read-only way (``tls_ready``): only the owner's ``network.allow_plaintext_lan`` opt-out without one serves plain
    HTTP on the bind address."""
    from prometheus.web import tls

    lan = tls.decide_lan(config or {}, address, cert_ok=tls.tls_ready())
    return lan.plan is not None or lan.loopback_only


def run_pair_command(
    args: argparse.Namespace,
    config: dict[str, Any] | None = None,
    *,
    client: Any = None,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    interactive: bool | None = None,
) -> int:
    """Execute ``oara pair <list|approve|deny>`` or ``oara pair --owner``. Returns an exit code.

    *interactive* (``--owner`` only): whether the output reaches a person's terminal; None asks stdout.
    """
    action = getattr(args, "pair_action", None)
    owner = bool(getattr(args, "owner", False))
    if owner and action is not None:
        out("Usage: oara pair --owner  (on its own), or oara pair <list|approve <id>|deny <id>>")
        return 2
    if owner and not (sys.stdout.isatty() if interactive is None else interactive):
        # Before any request, so no code exists at all. An accident guard (asking the agent to run it, a pipe into a
        # log), not a wall: a process set on it can fake a terminal, and the shell floors are what stop that.
        out("oara pair --owner shows its code only on a terminal, so the code cannot end up in a transcript or a "
            "log. Run it yourself in a terminal on the machine Prometheus runs on (over ssh: ssh -t <host> oara "
            "pair --owner).")
        return 1
    if not owner and action not in ("list", "approve", "deny"):
        out("Usage: oara pair <list|approve <id>|deny <id>>, or oara pair --owner")
        return 2
    prefix = getattr(args, "request", None)
    if action in ("approve", "deny") and len(prefix or "") < MIN_PREFIX:
        out(f"Give at least {MIN_PREFIX} characters of the request id (see: oara pair list).")
        return 1
    token, _source = resolve_api_token(config)
    if not token:
        out("There is no web API token, so this daemon has no way to pair a new device: no web API token is set "
            "(the API is open).\nSet one with: oara token rotate")
        return 1
    base_url = daemon_base_url(config)
    own = client is None
    if own:
        client = httpx.Client(base_url=base_url, timeout=_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {token}", "X-Pairing-Via": "cli"}
    try:
        if owner:
            return _owner_code(client, headers, out, sleep)
        return _run(str(action), args, client, headers, out)     # one of list, approve, deny (checked above)
    except httpx.TransportError:
        out(f"Could not reach the daemon at {base_url}. Is it running? (start it with: oara daemon)\n"
            "If it was started with --bind ADDRESS, that address is not visible from here: run this with "
            "PROMETHEUS_WEB_BIND=ADDRESS set.")
        return 1
    finally:
        if own:
            client.close()


def _waiting_quietly(client: Any, headers: dict[str, str]) -> list[dict[str, Any]]:
    """The waiting list for a name to show, or nothing: the decision itself reports any real failure."""
    try:
        response = client.get("/api/pair/requests", headers=headers)
        return response.json().get("requests", []) if response.status_code == 200 else []
    except (httpx.TransportError, ValueError):
        return []


def _waiting(client: Any, headers: dict[str, str], out: Callable[[str], None]) -> list[dict[str, Any]] | None:
    response = client.get("/api/pair/requests", headers=headers)
    if response.status_code == 401:
        out("The daemon rejected the API token. Check it with: oara token show")
        return None
    if response.status_code != 200:
        out(f"The daemon answered {response.status_code} to the list of waiting devices.")
        return None
    return response.json().get("requests", [])


def _run(action: str, args: argparse.Namespace, client: Any, headers: dict[str, str],
         out: Callable[[str], None]) -> int:
    full_id = action != "list" and re.fullmatch(r"[0-9a-f]{32}", args.request.lower())
    if full_id:
        # A whole id goes straight to the daemon, which knows the truth: a request someone else already
        # decided is not in the waiting list, and "no match" would hide that it was approved a moment ago.
        request_id = args.request.lower()
        known = _waiting_quietly(client, headers)
        target = next((r for r in known if r["request_id"] == request_id),
                      {"request_id": request_id, "device_name": None})
        return _decide(action, args, target, client, headers, out)
    waiting = _waiting(client, headers, out)
    if waiting is None:
        return 1
    if action == "list":
        if not waiting:
            out("No devices are waiting to join.")
            return 0
        out(f"{'ID':<9}{'NAME':<26}{'PLATFORM':<10}{'FROM':<17}{'CODE':<6}WINDOW")
        for item in waiting:
            out(f"{item['request_id'][:8]:<9}{item['device_name'][:25]:<26}{item['platform']:<10}"
                f"{item['source_ip']:<17}{item['match_code']:<6}{max(1, item['ttl_seconds'] // 60)} min")
        out("\nApprove only the one whose code matches the code on the new device's screen.")
        return 0
    matches = [r for r in waiting if r["request_id"].startswith(args.request.lower())]
    if not matches:
        out(f"No waiting request matches {args.request!r}. See: oara pair list")
        return 1
    if len(matches) > 1:
        out(f"{args.request!r} matches more than one waiting request; give more of the id.")
        return 1
    return _decide(action, args, matches[0], client, headers, out)


def _decide(action: str, args: argparse.Namespace, target: dict[str, Any], client: Any,
            headers: dict[str, str], out: Callable[[str], None]) -> int:
    body: dict[str, str] | None = None
    if action == "approve":
        body = {k: v for k, v in (("name", args.name), ("match_code", args.code)) if v}
    response = client.post(f"/api/pair/requests/{target['request_id']}/{action}", headers=headers, json=body or None)
    return _report(action, target, response, out)


def _report(action: str, target: dict[str, Any], response: Any, out: Callable[[str], None]) -> int:
    status = response.status_code
    if status == 200:
        if action == "approve":
            body = response.json()
            out(f"Approved {body['name']!r}. It is now a device on this Prometheus (id {body['device_id'][:8]}), "
                "scoped to its own conversations. Its token went to the device, sealed; nothing is shown here.")
        else:
            name = target.get("device_name")
            out(f"Denied {name!r}." if name else "Denied that device.")
        return 0
    if status == 422:
        out("The code does not match the one on this request. Nothing was decided: check the new device's screen.")
    elif status == 409:
        out(f"That request is no longer waiting: it was already {response.json().get('status', 'decided')}.")
    elif status == 410:
        out("That request has expired. The new device has to ask again.")
    elif status == 404:
        out("That request is gone.")
    elif status == 401:
        out("The daemon rejected the API token. Check it with: oara token show")
    elif status == 403:
        out("This token may not decide pairing requests.")
    else:
        out(f"The daemon answered {status}.")
    return 1


# ── oara pair --owner ────────────────────────────────────────────────────────

def _owner_code(client: Any, headers: dict[str, str], out: Callable[[str], None],
                sleep: Callable[[float], None]) -> int:
    response = client.post("/api/pair/owner-codes", headers=headers)
    if response.status_code != 201:
        return _owner_refused(response, out)
    issued = response.json()
    minutes = max(1, int(issued["ttl_seconds"]) // 60)
    until = time.strftime("%H:%M", time.localtime(int(issued["expires_at"])))
    out(f"Owner code:  {format_code(issued['code'])}")
    out("")
    out(f"Type it into the Beacon you want as your own, on this computer or another. It works once, for {minutes} "
        f"minutes (until {until}).")
    out("That Beacon becomes an OWNER device: it sees every conversation and can approve tool calls, workers "
        "and new devices. Give the code to no one else.")
    out("")
    out("Waiting for it to be used (Ctrl-C cancels it)...")
    code_id = issued["code_id"]
    warned = 0
    try:
        for _ in range(_OWNER_MAX_POLLS):
            sleep(OWNER_POLL_SECONDS)
            state = client.get(f"/api/pair/owner-codes/{code_id}", headers=headers)
            if state.status_code != 200:
                out(f"The daemon answered {state.status_code} about the code; it may have restarted, which ends "
                    "every code. Run oara pair --owner again.")
                return 1
            body = state.json()
            wrong = int(body.get("wrong_attempts") or 0)
            if wrong > warned:
                # The alarm. Wrong guesses no longer burn the code (a stranger could switch it off); they are
                # reported here instead, as they happen.
                out(f"Warning: {wrong} wrong {'try' if wrong == 1 else 'tries'} at this code so far. If that was not "
                    "you, someone may be guessing: Ctrl-C cancels the code.")
                warned = wrong
            if body["status"] != "pending":
                return _owner_outcome(body, out)
        out("Stopped waiting. The code may still work until it expires.")
        return 1
    except KeyboardInterrupt:
        try:
            cancelled = client.delete(f"/api/pair/owner-codes/{code_id}", headers=headers)
        except httpx.TransportError:
            out(f"\nStopped waiting, but the daemon could not be told: the code works until {until}.")
            return 130
        if cancelled.status_code == 200:
            out("\nCancelled: the code no longer works.")
            return 130
        try:
            body = cancelled.json()
        except ValueError:
            body = {}
        if cancelled.status_code == 409 and body.get("status"):
            # It finished before the cancel landed, perhaps in the second between two polls: say how, above all
            # if someone redeemed it, because that report is the alarm.
            out("")
            return _owner_outcome(body, out)
        out(f"\nThe daemon answered {cancelled.status_code} to the cancel: the code may still work until {until}.")
        return 130


def _where(device: dict[str, Any]) -> str:
    """Where the Beacon that used the code is, from what the daemon saw at redeem time. A daemon that predates
    ``this_host`` gets the address alone, as before."""
    source = device.get("source")
    if "this_host" not in device:
        return f"from {source}"
    if device["this_host"]:
        return "on this Mac" if sys.platform == "darwin" else "on this computer"
    from prometheus.web.loopback import is_loopback_address

    if isinstance(source, str) and is_loopback_address(source):
        # Loopback, but not this host: a relay on this machine (Tailscale serve, cloudflared) brought it in.
        return f"through a relay on this machine from {source}, so on another computer"
    return f"another computer at {source}"


def _owner_outcome(body: dict[str, Any], out: Callable[[str], None]) -> int:
    status = body["status"]
    device = body.get("redeemed_by") or {}
    if status == "redeemed":
        out(f"Done: {device.get('name')!r} ({device.get('platform')}, {_where(device)}) is now an owner "
            f"device (id {str(device.get('id'))[:8]}).")
        out("If that was not you, revoke it now: in Beacon's device list, or "
            f"DELETE /api/devices/{device.get('id')} with the API token.")
        return 0
    if status == "expired":
        out("The code expired unused. Run oara pair --owner again when that Beacon is ready.")
    elif status == "replaced":
        out("A newer owner code replaced this one (oara pair --owner was run again). Only the newest works.")
    elif status == "canceled":
        out("The code was cancelled.")
    else:
        out(f"The code is {status}.")
    return 1


def _owner_refused(response: Any, out: Callable[[str], None]) -> int:
    status = response.status_code
    try:
        error = (response.json() or {}).get("error")
    except ValueError:
        error = None
    if status == 404:
        out("This daemon does not issue owner codes: it is older than this command, or still in setup mode. "
            "Update Prometheus (or finish setup) and try again.")
    elif status == 401:
        out("The daemon rejected the API token. Check it with: oara token show")
    elif error == "not_this_machine":
        out("The daemon issues owner codes only to a terminal on the machine it runs on, and this request did not "
            "reach it that way (it went through a proxy, or the daemon is on another machine). Run this on that "
            "machine, against the address the daemon listens on.")
    elif error == "pairing_unavailable":
        out("This daemon has no API token (the API is open), so it has no owner devices. "
            "Set one with: oara token rotate")
    else:
        out(f"The daemon answered {status}{f' ({error})' if error else ''}.")
    return 1
