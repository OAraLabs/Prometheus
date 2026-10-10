"""``oara pair list | approve | deny`` — the route for a headless box with no Beacon open and no Telegram.

It is the same REST routes with the global token read the way ``oara token show`` reads it, and it says it
is the terminal (``X-Pairing-Via: cli``) so the owner's other screens read "approved in the terminal". What is
pinned: it never prints a secret or a token (neither is in any response it reads, and the poll secret never
leaves the requester), an id may be given as a unique prefix, ambiguity and unknowns are refused rather than
guessed, and every way it can fail (daemon down, no token, wrong token) says what to do.

The HTTP client is injectable; the tests hand it the app's own test client, so the routes are the real ones.
"""

from __future__ import annotations

import argparse

import httpx
import pytest

from prometheus.cli import pair as pair_cli
from prometheus.cli.pair import add_pair_subparser, daemon_base_url, run_pair_command
from tests.support.pairing_world import GLOBAL, World

CONFIG = {"web": {"api_token": GLOBAL, "api_port": 8005}}


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="oara")
    add_pair_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["pair", *argv])


def run(world: World, *argv: str, config=CONFIG, client=None) -> tuple[int, str]:
    lines: list[str] = []
    code = run_pair_command(_args(*argv), config, client=client or world.client, out=lines.append)
    return code, "\n".join(lines)


# ── registration ─────────────────────────────────────────────────────────────

def test_the_subcommand_has_list_approve_and_deny():
    args = _args("approve", "abc123", "--name", "Kitchen iPad", "--code", "4821")
    assert (args.command, args.pair_action, args.request, args.name, args.code) == (
        "pair", "approve", "abc123", "Kitchen iPad", "4821")
    assert _args("list").pair_action == "list"
    assert _args("deny", "abc123").pair_action == "deny"


def test_no_action_prints_usage_and_exits_2(world):
    parser = argparse.ArgumentParser(prog="oara")
    add_pair_subparser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["pair"])
    lines: list[str] = []
    assert run_pair_command(args, CONFIG, client=world.client, out=lines.append) == 2
    assert "Usage" in "\n".join(lines)


# ── list ─────────────────────────────────────────────────────────────────────

def test_list_says_nothing_is_waiting(world):
    code, out = run(world, "list")
    assert code == 0 and "No devices are waiting" in out


def test_list_shows_what_the_owner_needs_to_decide(world):
    created, _, _ = world.created(source="192.0.2.42")
    code, out = run(world, "list")
    assert code == 0
    for needle in (created["request_id"][:8], "Jennifer's MacBook", "macos", "192.0.2.42", created["match_code"], "5 min"):
        assert needle in out, needle
    assert created["poll_secret"] not in out


# ── approve and deny ─────────────────────────────────────────────────────────

def test_approve_by_full_id_mints_a_scoped_device_and_prints_no_token(world):
    created, requester, _ = world.created()
    code, out = run(world, "approve", created["request_id"])
    assert code == 0 and "Approved" in out and "Jennifer's MacBook" in out
    polled = world.poll(created).json()          # once: a second poll in the same second is refused
    token = requester.unseal(created["request_id"], polled["sealed"])["token"]
    assert token not in out and created["poll_secret"] not in out
    assert world.devices.is_owner(polled["device_id"]) is False


def test_approve_by_a_unique_prefix(world):
    created, _, _ = world.created()
    assert run(world, "approve", created["request_id"][:6])[0] == 0
    assert world.poll(created).json()["status"] == "approved"


def test_a_prefix_that_matches_two_is_refused_not_guessed(world):
    a, _, _ = world.created(source="192.0.2.1")
    b, _, _ = world.created(source="192.0.2.2")
    # Ids are random, so give the two rows ids that share a prefix: the ambiguity is the thing under test.
    one, two = "abc123" + "0" * 26, "abc123" + "1" * 26
    conn = world.devices.connection
    conn.execute("UPDATE pair_requests SET id = ? WHERE id = ?", (one, a["request_id"]))
    conn.execute("UPDATE pair_requests SET id = ? WHERE id = ?", (two, b["request_id"]))
    conn.commit()
    code, out = run(world, "approve", "abc123")
    assert code == 1 and "more than one" in out
    assert len(world.devices.list_devices()) == 2, "nothing decided: only the world's own two devices"
    code, out = run(world, "deny", "abc1230")           # one more character makes it unique
    assert code == 0 and "Denied" in out


def test_a_too_short_prefix_is_refused(world):
    created, _, _ = world.created()
    code, out = run(world, "approve", created["request_id"][:3])
    assert code == 1 and "at least 6" in out


def test_an_unknown_id_is_refused(world):
    world.created()
    code, out = run(world, "approve", "ffffff")
    assert code == 1 and "No waiting request" in out


def test_name_and_code_are_forwarded(world):
    created, requester, _ = world.created()
    code, out = run(world, "approve", created["request_id"], "--name", "Kitchen iPad", "--code", created["match_code"])
    assert code == 0
    assert requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["name"] == "Kitchen iPad"


def test_a_code_that_does_not_match_decides_nothing(world):
    created, _, _ = world.created()
    wrong = "0000" if created["match_code"] != "0000" else "1111"
    code, out = run(world, "approve", created["request_id"], "--code", wrong)
    assert code == 1 and "does not match" in out
    assert world.poll(created).json()["status"] == "pending"


def test_deny(world):
    created, _, _ = world.created()
    code, out = run(world, "deny", created["request_id"][:8])
    assert code == 0 and "Denied" in out
    assert world.poll(created).json() == {"status": "denied"}


def test_a_whole_id_still_shows_the_name_when_it_is_waiting(world):
    created, _, _ = world.created()
    code, out = run(world, "deny", created["request_id"])
    assert code == 0 and "Jennifer's MacBook" in out


def test_a_whole_id_that_is_no_longer_waiting_is_denied_by_the_daemon_not_by_us(world):
    created, _, _ = world.created()
    world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/deny")
    code, out = run(world, "approve", created["request_id"])
    assert code == 1 and "no longer waiting" in out and "denied" in out


def test_it_says_it_is_the_terminal(world):
    created, _, _ = world.created()               # the world already records what the channels are told
    run(world, "approve", created["request_id"])
    resolved = [p for k, p in world.events if k == "resolved"]
    assert [p["by"] for p in resolved] == ["cli"]


def test_a_decision_someone_else_made_first_is_reported(world):
    created, _, _ = world.created()
    world.approve(created)
    code, out = run(world, "deny", created["request_id"])
    assert code == 1 and "no longer waiting" in out and "approved" in out


# ── when it cannot work ──────────────────────────────────────────────────────

def test_a_daemon_that_is_not_running_says_how_to_start_it(world):
    class Down:
        def request(self, *a, **kw):
            raise httpx.ConnectError("refused")

        def get(self, *a, **kw):
            raise httpx.ConnectError("refused")

        post = get

    code, out = run(world, "list", client=Down())
    assert code == 1 and "daemon" in out and "8005" in out


def test_with_no_token_it_explains_there_is_nothing_to_pair_into(world):
    code, out = run(world, "list", config={"web": {"api_port": 8005}})
    assert code == 1 and "no web API token" in out


def test_a_wrong_token_points_at_oara_token_show(world):
    code, out = run(world, "list", config={"web": {"api_token": "not-the-token", "api_port": 8005}})
    assert code == 1 and "oara token show" in out



# ── where the daemon is ──────────────────────────────────────────────────────
# `oara pair` used to knock on http://127.0.0.1:<port> whatever the daemon listened on. A daemon bound to one
# address (a LAN or tailnet address, the way an owner narrows a headless box) does not answer on loopback, so
# the terminal route for exactly that box said "could not reach the daemon" about a daemon that was running.

LAN = "192.0.2.20"


@pytest.mark.parametrize("config, expected", [
    ({}, "http://127.0.0.1:8005"),
    ({"web": {"api_port": 9001}}, "http://127.0.0.1:9001"),
    ({"web": {"api_port": "not a port"}}, "http://127.0.0.1:8005"),
    ({"web": {"bind": "0.0.0.0"}}, "http://127.0.0.1:8005"),        # every interface includes loopback
    ({"web": {"bind": "127.0.0.1"}}, "http://127.0.0.1:8005"),
    ({"web": {"bind": "localhost"}}, "http://127.0.0.1:8005"),
    ({"web": {"bind": "::"}}, "http://[::1]:8005"),                  # every interface, IPv6: bracketed in a URL
    ({"web": {"bind": "::1"}}, "http://[::1]:8005"),
    ({"web": {"bind": LAN}}, f"http://{LAN}:8005"),                  # one address: ONLY that one answers
    ({"web": {"bind": LAN, "api_port": 9001}}, f"http://{LAN}:9001"),
    ({"web": {"bind": "fe80::1"}}, "http://[fe80::1]:8005"),
])
def test_the_daemon_is_reached_where_it_listens(config, expected):
    assert daemon_base_url(config, env={}) == expected


def test_the_environment_outranks_the_config_the_way_it_does_for_the_daemon():
    config = {"web": {"bind": "127.0.0.1"}}
    assert daemon_base_url(config, env={"PROMETHEUS_WEB_BIND": LAN}) == f"http://{LAN}:8005"


def test_a_bind_that_cannot_be_honoured_falls_back_to_loopback_and_leaves_the_complaint_to_the_daemon():
    """The daemon refuses to start on it; the CLI's job is only to try somewhere sensible and say what it tried."""
    assert daemon_base_url({"web": {"bind": "not an address"}}, env={}) == "http://127.0.0.1:8005"


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_WEB_BIND", raising=False)
    monkeypatch.setattr(pair_cli, "parse_env_file", lambda: {})


def _recording_client(monkeypatch, handler):
    seen: dict = {}
    real = httpx.Client

    def factory(**kw):
        seen.update(kw)
        return real(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(pair_cli.httpx, "Client", factory)
    return seen


def test_the_client_it_builds_talks_to_that_address(monkeypatch, isolated_env):
    seen = _recording_client(monkeypatch, lambda request: httpx.Response(200, json={"requests": []}))
    lines: list[str] = []
    config = {"web": {"api_token": GLOBAL, "bind": LAN, "api_port": 8123}}
    assert run_pair_command(_args("list"), config, out=lines.append) == 0
    assert seen["base_url"] == f"http://{LAN}:8123", "it used to be http://127.0.0.1:8123 whatever the bind"


def test_the_bind_in_the_env_file_counts_when_the_environment_does_not_say(monkeypatch):
    """The daemon loads ~/.config/prometheus/env itself; the terminal route must read the same address."""
    monkeypatch.delenv("PROMETHEUS_WEB_BIND", raising=False)
    monkeypatch.setattr(pair_cli, "parse_env_file", lambda: {"PROMETHEUS_WEB_BIND": LAN})
    seen = _recording_client(monkeypatch, lambda request: httpx.Response(200, json={"requests": []}))
    run_pair_command(_args("list"), {"web": {"api_token": GLOBAL, "bind": "127.0.0.1"}}, out=lambda _: None)
    assert seen["base_url"] == f"http://{LAN}:8005"
    monkeypatch.setenv("PROMETHEUS_WEB_BIND", "127.0.0.1")           # the real environment wins over the file
    run_pair_command(_args("list"), {"web": {"api_token": GLOBAL}}, out=lambda _: None)
    assert seen["base_url"] == "http://127.0.0.1:8005"


def test_an_unreachable_daemon_names_the_address_it_tried_and_the_ways_to_change_it(monkeypatch, isolated_env):
    def refuse(request):
        raise httpx.ConnectError("refused")

    _recording_client(monkeypatch, refuse)
    lines: list[str] = []
    config = {"web": {"api_token": GLOBAL, "bind": LAN, "api_port": 8123}}
    assert run_pair_command(_args("list"), config, out=lines.append) == 1
    text = "\n".join(lines)
    assert f"http://{LAN}:8123" in text and "oara daemon" in text
    assert "--bind" in text and "PROMETHEUS_WEB_BIND" in text, "a --bind given to a running daemon is invisible here"
