"""``oara pair`` — see and decide the devices that are asking to join this Prometheus.

    oara pair list
    oara pair approve <id> [--name NAME] [--code NNNN]
    oara pair deny <id>

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
import re
from collections.abc import Callable
from typing import Any

import httpx

from prometheus.config.api_token import resolve_api_token

MIN_PREFIX = 6
_TIMEOUT_SECONDS = 10.0


def add_pair_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``pair`` subcommand on the main CLI parser."""
    pair = subparsers.add_parser("pair", help="List, approve or deny devices asking to join this Prometheus")
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


def run_pair_command(
    args: argparse.Namespace,
    config: dict[str, Any] | None = None,
    *,
    client: Any = None,
    out: Callable[[str], None] = print,
) -> int:
    """Execute ``oara pair <list|approve|deny>``. Returns an exit code."""
    action = getattr(args, "pair_action", None)
    if action not in ("list", "approve", "deny"):
        out("Usage: oara pair <list|approve <id>|deny <id>>")
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
    port = _port(config)
    own = client is None
    if own:
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {token}", "X-Pairing-Via": "cli"}
    try:
        return _run(action, args, client, headers, out)
    except httpx.TransportError:
        out(f"Could not reach the daemon at http://127.0.0.1:{port}. Is it running? (start it with: oara daemon)")
        return 1
    finally:
        if own:
            client.close()


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
        target = {"request_id": args.request.lower(), "device_name": "that device"}
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
            out(f"Denied {target['device_name']!r}.")
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
