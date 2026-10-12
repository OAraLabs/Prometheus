"""Promoting a REMOTE Beacon to an owner device with a one-time code from ``oara pair --owner``.

Since #744 a worker's request is approved only by a device token: the API token is refused (401
``not_a_person``) because a model can read it. Owner device tokens came only from a client on the daemon's own
machine, so a Beacon on another computer, connected over the tailnet with the API token, could not approve a
worker at all: every Approve was a 401 and the card came back.

The fix, and what these tests pin:

* ``oara pair --owner``, run ON the daemon's machine, prints a one-time code: 12 characters, ten minutes, one
  use, and the fifth wrong guess burns it. It writes the code's SHA-256 straight into ``devices.db``; there is
  NO route that issues a code, so nothing that only reaches the daemon over the network can make one. The
  API token alone, from anywhere, mints no owner device.
* ``POST /api/pair/owner`` trades a code for an owner device. It needs a bearer (the API token or a device
  token: the Beacon is already connected), a public key, and the code. The token comes back SEALED to that
  key, the way an approved pairing's does, because the owner token can do one thing the API token cannot
  (approve a worker) and must not cross a plaintext link in clear.
* Wrong guesses are limited per source, and every wrong guess counts against the code, so a short code cannot
  be ground out; a malformed body is refused before the code is looked at, and spends nothing.
"""

from __future__ import annotations

import ast
import json
import logging
import sqlite3
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.cli import pair as pair_cli  # noqa: E402
from prometheus.config import owner_codes  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, ApprovalResult, PendingAction  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from tests.support.pairing_world import GLOBAL, LOW_ORDER, T0, Requester, World  # noqa: E402

REMOTE = "192.0.2.77"          # the MacBook, as the daemon sees it (TEST-NET-1)
OTHER = "203.0.113.9"          # somebody else on the network (TEST-NET-3)
SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "no-env-file"))


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def issue(world: World) -> str:
    """What the person at the daemon's machine does: the code store, on its OWN connection to the file."""
    return owner_codes.issue_owner_code(DeviceStore(world.path), by="test", clock=world.clock).code


def redeem(world: World, code: str, *, who: str = "global", source: str = REMOTE, requester=None,
           headers=None, **body):
    requester = requester or Requester()
    payload = {"code": code, "device_name": "Will's MacBook", "platform": "macos",
               "public_key": requester.public_b64, **body}
    client = TestClient(world.app, client=(source, 50000))
    response = client.post("/api/pair/owner", json=payload, headers={**world.hdr(who), **(headers or {})})
    return response, requester


def opened(response, requester) -> dict:
    assert response.status_code == 201, response.text
    body = response.json()
    return requester.unseal(body["device_id"], body["sealed"])


def owner_rows(world: World) -> dict[str, str]:
    return DeviceStore(world.path).owner_sources()


def table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}


# ── the code ─────────────────────────────────────────────────────────────────

def test_a_code_is_twelve_unambiguous_characters_in_three_groups(world):
    code = issue(world)
    groups = code.split("-")
    assert [len(g) for g in groups] == [4, 4, 4], code
    assert set("".join(groups)) <= set(owner_codes.ALPHABET)
    assert not set("01ILOU") & set(owner_codes.ALPHABET), "no characters a person can misread"
    assert len(owner_codes.ALPHABET) ** 12 > 2 ** 58


def test_two_codes_are_not_the_same(world):
    assert len({issue(world) for _ in range(20)}) == 20


def test_the_code_lasts_ten_minutes(world):
    issued = owner_codes.issue_owner_code(DeviceStore(world.path), by="test", clock=world.clock)
    assert issued.expires_at == T0 + 600 == T0 + owner_codes.CODE_TTL_SECONDS


def test_only_the_codes_digest_is_stored(world):
    code = issue(world)
    with sqlite3.connect(world.path) as conn:
        rows = conn.execute("select * from owner_codes").fetchall()
    assert len(rows) == 1
    flat = json.dumps(rows)
    for spelling in (code, code.replace("-", ""), code.lower(), code.replace("-", "").lower()):
        assert spelling not in flat


def test_a_typed_code_may_be_lower_case_and_spaced(world):
    code = issue(world)
    typed = "  " + code.replace("-", " ").lower() + " "
    response, requester = redeem(world, typed)
    assert opened(response, requester)["device_id"] == response.json()["device_id"]


# ── redeeming ────────────────────────────────────────────────────────────────

def test_the_api_token_on_another_computer_trades_a_code_for_an_owner_device(world):
    code = issue(world)
    response, requester = redeem(world, code, who="global", source=REMOTE)
    body = response.json()
    assert response.status_code == 201, response.text
    assert body["owner"] is True and body["owner_source"] == "owner-code"
    assert body["name"] == "Will's MacBook" and body["platform"] == "macos"
    assert "token" not in body, "the token travels sealed, never in clear"
    secret = opened(response, requester)
    assert secret["device_id"] == body["device_id"] and secret["name"] == "Will's MacBook"
    assert secret["token"] not in response.text
    assert owner_rows(world)[body["device_id"]] == "owner-code"

    # The new token authenticates, from the other computer, as an operator who knows it is itself.
    client = TestClient(world.app, client=(REMOTE, 50000))
    listing = client.get("/api/devices", headers={"Authorization": f"Bearer {secret['token']}"})
    assert listing.status_code == 200
    me = next(d for d in listing.json() if d["is_self"])
    assert me["id"] == body["device_id"] and me["owner"] is True and me["owner_source"] == "owner-code"
    assert len(listing.json()) > 1, "an owner device sees every device"


def test_the_sealed_token_opens_only_with_the_requesters_key(world):
    code = issue(world)
    response, _ = redeem(world, code)
    assert response.status_code == 201
    with pytest.raises(Exception):
        Requester().unseal(response.json()["device_id"], response.json()["sealed"])


def test_the_promoted_beacon_approves_a_workers_request_and_the_api_token_still_cannot(world):
    """The point of the change: the card that looped on 401 not_a_person now resolves."""
    queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    world.app.state.approval_queue = queue
    client = TestClient(world.app, client=(REMOTE, 50000))

    def worker_request(request_id: str) -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="write_file", description="write ./notes.md",
                               session_id="telegram:42", once_only=True, worker_task_id="a0000worker")
        queue.pending[request_id] = action
        return action

    before = worker_request("0e0e0001")
    refused = client.post("/api/approvals/0e0e0001/approve", headers=world.hdr("global"), json={"scope": "once"})
    assert refused.status_code == 401 and refused.json()["code"] == "not_a_person"
    assert "0e0e0001" in queue.pending and before._result == ApprovalResult.TIMEOUT

    response, requester = redeem(world, issue(world))
    token = opened(response, requester)["token"]
    approved = client.post("/api/approvals/0e0e0001/approve", headers={"Authorization": f"Bearer {token}"},
                           json={"scope": "once"})
    assert approved.status_code == 200 and approved.json()["ok"] is True, approved.text
    assert before._result == ApprovalResult.APPROVED


def test_a_scoped_device_may_redeem_a_code_it_was_given(world):
    """A Beacon that joined by approve-to-pair holds a scoped token; promoting it is the same step."""
    response, requester = redeem(world, issue(world), who="scoped")
    assert response.status_code == 201, response.text
    assert owner_rows(world)[response.json()["device_id"]] == "owner-code"
    assert not DeviceStore(world.path).is_owner(world.scoped["id"]), "the scoped device itself is not changed"


def test_a_code_works_once(world):
    code = issue(world)
    first, _ = redeem(world, code)
    second, _ = redeem(world, code, source=OTHER)
    assert first.status_code == 201
    assert second.status_code == 422 and second.json()["error"] == "invalid_code"
    assert sum(1 for s in owner_rows(world).values() if s == "owner-code") == 1


def test_a_code_works_once_even_when_two_connections_race(world):
    code = issue(world)
    stores = [owner_codes.OwnerCodeStore(DeviceStore(world.path), clock=world.clock) for _ in range(8)]
    start = threading.Barrier(len(stores))
    results: list[object] = []

    def go(store):
        start.wait()
        results.append(store.redeem(code, name="racer", platform="macos"))

    threads = [threading.Thread(target=go, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for r in results if r is not None) == 1, results
    assert sum(1 for s in owner_rows(world).values() if s == "owner-code") == 1


def test_a_code_expires_after_ten_minutes(world):
    code = issue(world)
    world.clock.now = T0 + 600
    response, _ = redeem(world, code)
    assert response.status_code == 422 and response.json()["error"] == "invalid_code"
    assert "owner-code" not in owner_rows(world).values()


def test_a_code_one_second_before_expiry_still_works(world):
    code = issue(world)
    world.clock.now = T0 + 599
    response, _ = redeem(world, code)
    assert response.status_code == 201, response.text


def test_a_new_code_replaces_the_one_before(world):
    old = issue(world)
    new = issue(world)
    refused, _ = redeem(world, old)
    assert refused.status_code == 422
    accepted, _ = redeem(world, new, source=OTHER)
    assert accepted.status_code == 201, accepted.text


def test_redeeming_replaces_no_other_owner_device(world):
    same_mac = world.owner["id"]
    first, _ = redeem(world, issue(world))
    second, _ = redeem(world, issue(world), source=OTHER)
    store = DeviceStore(world.path)
    live = {d.id for d in store.list_devices() if d.revoked_at is None}
    assert {same_mac, world.scoped["id"], first.json()["device_id"], second.json()["device_id"]} <= live
    assert store.is_owner(same_mac)


# ── the API token alone mints nothing, and nothing remote makes a code ───────

def test_the_api_token_with_no_code_issued_mints_nothing(world):
    for guess in ("2345-6789-ABCD", "ZZZZ-ZZZZ-ZZZZ", "MNPQ-RSTV-WXYZ"):
        response, _ = redeem(world, guess, source=OTHER)
        assert response.status_code in (422, 429), response.text
    assert "owner-code" not in owner_rows(world).values()


def test_redeeming_with_no_code_ever_issued_creates_no_table(world):
    before = table_names(world.path)
    response, _ = redeem(world, "2345-6789-ABCD")
    assert response.status_code == 422
    assert table_names(world.path) == before
    assert "owner_codes" not in before


def test_the_api_token_from_another_computer_still_cannot_mint_an_owner_device_directly(world):
    client = TestClient(world.app, client=(REMOTE, 50000))
    response = client.post("/api/devices", headers=world.hdr("global"),
                           json={"name": "sneaky", "platform": "macos", "owner": True})
    assert response.status_code == 403
    assert owner_rows(world) == {world.owner["id"]: "same-mac-pairing"}


def test_no_bearer_no_redeem(world):
    code = issue(world)
    client = TestClient(world.app, client=(REMOTE, 50000))
    response = client.post("/api/pair/owner", json={
        "code": code, "device_name": "x", "platform": "macos", "public_key": Requester().public_b64})
    assert response.status_code == 401
    still, _ = redeem(world, code)
    assert still.status_code == 201, "a refused bearer spends nothing"


def test_a_wrong_bearer_does_not_redeem(world):
    response, _ = redeem(world, issue(world), who="not-a-token")
    assert response.status_code == 401


def _referrers(name: str) -> set[str]:
    """Every source file that names *name* (a call, an import or an attribute)."""
    found = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == name) or \
               (isinstance(node, ast.Attribute) and node.attr == name) or \
               (isinstance(node, ast.alias) and node.name == name) or \
               (isinstance(node, ast.FunctionDef) and node.name == name):
                found.add(path.relative_to(SRC).as_posix())
    return found


def test_only_the_terminal_command_issues_a_code():
    """No route, no tool, no channel: a code exists only because someone ran ``oara pair --owner`` there."""
    assert _referrers("issue_owner_code") == {"config/owner_codes.py", "cli/pair.py"}


def test_no_registered_route_issues_a_code(world):
    """Belt to the scan above: every route the app has, with the API token, leaves no code behind."""
    paths = sorted({r.path for r in world.app.routes if "owner" in getattr(r, "path", "")})
    assert paths == ["/api/pair/owner"], paths
    for method in ("GET", "PUT", "DELETE"):
        response = world.as_("global", method, "/api/pair/owner")
        assert response.status_code in (404, 405), (method, response.status_code)
    assert "owner_codes" not in table_names(world.path)


# ── guessing ─────────────────────────────────────────────────────────────────

def wrong_for(code: str) -> str:
    """A well-formed code that is not *code*."""
    flat = code.replace("-", "")
    swap = "2" if flat[0] != "2" else "3"
    return owner_codes.format_code(swap + flat[1:])


def test_a_wrong_code_is_422_invalid_code_never_401(world):
    """401 tells a client its token is dead; a wrong code is not that."""
    code = issue(world)
    response, _ = redeem(world, wrong_for(code))
    assert response.status_code == 422 and response.json()["error"] == "invalid_code"


def test_four_typos_then_the_right_code_still_works(world):
    code = issue(world)
    for i in range(4):
        response, _ = redeem(world, wrong_for(code), source=f"192.0.2.{10 + i}")
        assert response.status_code == 422
    response, _ = redeem(world, code)
    assert response.status_code == 201, response.text


def test_the_fifth_wrong_guess_burns_the_code_wherever_the_guesses_came_from(world):
    code = issue(world)
    for i in range(owner_codes.MAX_FAILURES):
        response, _ = redeem(world, wrong_for(code), source=f"192.0.2.{10 + i}")
        assert response.status_code == 422
    response, _ = redeem(world, code)
    assert response.status_code == 422 and response.json()["error"] == "invalid_code"
    assert "owner-code" not in owner_rows(world).values()
    with sqlite3.connect(world.path) as conn:
        assert conn.execute("select count(*) from owner_codes").fetchone()[0] == 0


def test_a_source_that_keeps_guessing_is_told_to_wait_and_spends_nothing_more(world):
    code = issue(world)
    for _ in range(owner_codes.WRONG_CODES_PER_MINUTE):
        response, _ = redeem(world, wrong_for(code), source=OTHER)
        assert response.status_code == 422
    limited, _ = redeem(world, code, source=OTHER)
    assert limited.status_code == 429 and limited.json()["error"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    # Refused before the code was looked at: the code is alive, and the owner's own Beacon gets it.
    accepted, _ = redeem(world, code, source=REMOTE)
    assert accepted.status_code == 201, accepted.text


def test_the_limited_source_may_try_again_after_the_window(world):
    code = issue(world)
    for _ in range(owner_codes.WRONG_CODES_PER_MINUTE):
        redeem(world, wrong_for(code), source=OTHER)
    world.clock.now = T0 + 61
    response, _ = redeem(world, code, source=OTHER)
    assert response.status_code == 201, response.text


def test_a_malformed_code_is_400_and_spends_nothing(world):
    code = issue(world)
    for bad in ("", "1234-5678-9ABC", "OOOO-OOOO-OOOO", "2345-6789-ABC", "2345-6789-ABCDE", 12345678, None,
                "2345­6789ABCD"):
        response, _ = redeem(world, bad, source=OTHER)
        assert response.status_code == 400 and response.json()["fields"] == ["code"], (bad, response.text)
    response, _ = redeem(world, code, source=OTHER)
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("field, value", [
    ("public_key", None), ("public_key", "not-a-key"), ("public_key", LOW_ORDER),
    ("device_name", None), ("device_name", ""), ("device_name", "a\nb"), ("device_name", "x" * 65),
])
def test_a_bad_body_is_refused_before_the_code_is_looked_at(world, field, value):
    code = issue(world)
    response, _ = redeem(world, code, **{field: value})
    assert response.status_code == 400 and field in response.json()["fields"], response.text
    accepted, _ = redeem(world, code)
    assert accepted.status_code == 201, "the code was neither used nor counted"


def test_a_browser_is_refused_and_spends_nothing(world):
    code = issue(world)
    response, _ = redeem(world, code, headers={"Origin": "https://evil.example"})
    assert response.status_code == 400 and response.json()["error"] == "browser_not_allowed"
    accepted, _ = redeem(world, code)
    assert accepted.status_code == 201


def test_a_body_that_is_not_json_is_refused(world):
    code = issue(world)
    client = TestClient(world.app, client=(REMOTE, 50000))
    response = client.post("/api/pair/owner", content=b"code=" + code.encode(),
                           headers={**world.hdr("global"), "Content-Type": "application/x-www-form-urlencoded"})
    assert response.status_code == 400
    accepted, _ = redeem(world, code)
    assert accepted.status_code == 201


def test_a_huge_body_is_refused(world):
    response, _ = redeem(world, issue(world), padding="x" * 5000)
    assert response.status_code == 413


def test_an_open_daemon_promotes_nobody(tmp_path):
    """No API token: nobody can be told from a model, and an owner device would mean nothing."""
    devices = DeviceStore(tmp_path / "devices.db")
    from prometheus.web.server import create_app

    app = create_app({"web": {}}, device_store=devices)
    code = owner_codes.issue_owner_code(DeviceStore(tmp_path / "devices.db"), by="test").code
    client = TestClient(app, client=(REMOTE, 50000))
    response = client.post("/api/pair/owner", json={
        "code": code, "device_name": "x", "platform": "macos", "public_key": Requester().public_b64})
    assert response.status_code == 403 and response.json()["error"] == "pairing_unavailable"
    assert DeviceStore(tmp_path / "devices.db").owner_sources() == {}


def test_neither_the_code_nor_the_token_is_logged(world, caplog):
    caplog.set_level(logging.DEBUG)
    code = issue(world)
    redeem(world, wrong_for(code), source=OTHER)
    response, requester = redeem(world, code)
    token = opened(response, requester)["token"]
    text = caplog.text
    for secret in (code, code.replace("-", ""), wrong_for(code), token):
        assert secret not in text
    assert "owner code" in text, "issue, refusal and redemption each leave an audit line"


# ── the terminal command ─────────────────────────────────────────────────────

def _args(*argv: str):
    import argparse

    parser = argparse.ArgumentParser(prog="oara")
    pair_cli.add_pair_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["pair", *argv])


class NoNetwork:
    """The command must not need the daemon or its token on the wire: it writes the code where the daemon reads it."""

    def __getattr__(self, name):
        raise AssertionError(f"oara pair --owner made a network call ({name})")


def run_cli(*argv: str, config=None) -> tuple[int, str]:
    lines: list[str] = []
    code = pair_cli.run_pair_command(_args(*argv), config if config is not None else {"web": {"api_token": GLOBAL}},
                                     client=NoNetwork(), out=lines.append)
    return code, "\n".join(lines)


def _printed_code(out: str) -> str:
    import re

    found = re.findall(r"\b[2-9A-Z]{4}-[2-9A-Z]{4}-[2-9A-Z]{4}\b", out)
    assert len(found) == 1, out
    return found[0]


def test_oara_pair_owner_prints_a_code_the_remote_beacon_redeems(world):
    status, out = run_cli("--owner")
    assert status == 0, out
    code = _printed_code(out)
    assert "10 minutes" in out and "once" in out
    response, requester = redeem(world, code)
    assert opened(response, requester)["token"]


def test_running_it_again_replaces_the_first_code(world):
    _, first = run_cli("--owner")
    _, second = run_cli("--owner")
    refused, _ = redeem(world, _printed_code(first))
    assert refused.status_code == 422
    accepted, _ = redeem(world, _printed_code(second), source=OTHER)
    assert accepted.status_code == 201


def test_it_refuses_where_there_is_no_daemon_registry_and_creates_none(world, tmp_path, monkeypatch):
    empty = tmp_path / "elsewhere"
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(empty))
    status, out = run_cli("--owner")
    assert status == 1
    assert "devices.db" in out and str(empty) in out
    assert not empty.exists()


def test_it_finds_the_registry_the_daemon_uses_through_the_env_file(world, tmp_path, monkeypatch):
    """The daemon loads ~/.config/prometheus/env at start; a data directory set there is the daemon's."""
    monkeypatch.delenv("PROMETHEUS_DATA_DIR")
    env_file = tmp_path / "envfile"
    env_file.write_text(f"PROMETHEUS_DATA_DIR={tmp_path}\n")
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_file))
    status, out = run_cli("--owner")
    assert status == 0, out
    response, _ = redeem(world, _printed_code(out))
    assert response.status_code == 201


def test_it_refuses_when_the_api_is_open(world):
    status, out = run_cli("--owner", config={"web": {}})
    assert status == 1 and "open" in out
    assert "owner_codes" not in table_names(world.path)


def test_owner_takes_no_action(world):
    status, out = run_cli("--owner", "list")
    assert status == 2 and "--owner" in out
    assert "owner_codes" not in table_names(world.path)


def test_the_plain_pair_command_is_unchanged():
    args = _args("approve", "abc123", "--code", "4821")
    assert (args.pair_action, args.request, args.code, args.owner) == ("approve", "abc123", "4821", False)
    assert _args("--owner").owner is True
