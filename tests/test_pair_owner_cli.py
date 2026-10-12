"""``oara pair --owner`` — the person at the machine makes a remote Beacon their own device.

It asks the daemon for a one-time code with the global token (read the way ``oara token show`` reads it, from
this machine), prints it in two groups of four, then WAITS and says what became of it: who redeemed it and from
which address, or that it expired, was replaced, or was locked by wrong guesses. The report is the point: a code
someone else typed first shows up here, at once, with the address it came from. Ctrl-C cancels the code rather
than leave it live.

The HTTP client and the sleep are injectable; the tests hand it the real app over a loopback test client and a
sleep that plays the remote Beacon's part.
"""

from __future__ import annotations

import argparse
import re

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.cli.pair import add_pair_subparser, run_pair_command  # noqa: E402
from tests.support.pairing_world import GLOBAL, World  # noqa: E402

CONFIG = {"web": {"api_token": GLOBAL, "api_port": 8005}}
MACBOOK = "192.0.2.77"


@pytest.fixture
def world(tmp_path) -> World:
    w = World(tmp_path)
    w.app.state.owner_codes.clock = w.clock
    w.app.state.owner_codes.wall = w.clock
    return w


def loopback(world: World) -> TestClient:
    return TestClient(world.app, base_url="http://127.0.0.1:8005", client=("127.0.0.1", 50123))


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="oara")
    add_pair_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["pair", *argv])


def printed_code(lines: list[str]) -> str:
    for line in lines:
        match = re.search(r"\b([0-9]{4})-([0-9]{4})\b", line)
        if match:
            return match[1] + match[2]
    raise AssertionError(f"no code was printed: {lines}")


def beacon_redeems(world: World, lines: list[str], peer: str = MACBOOK, name: str = "Will's MacBook Pro"):
    client = TestClient(world.app, base_url="http://192.0.2.50:8005", client=(peer, 50000))
    return client.post("/api/pair/owner", json={"code": printed_code(lines), "device_name": name, "platform": "macos"})


def run(world: World, *argv: str, sleep=None, config=CONFIG, client=None) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = run_pair_command(_args(*argv), config, client=client or loopback(world), out=lines.append,
                            sleep=sleep or (lambda _s: None))
    return code, lines


# ── the flag ─────────────────────────────────────────────────────────────────

def test_the_flag_parses_without_an_action():
    args = _args("--owner")
    assert (args.command, args.owner, args.pair_action) == ("pair", True, None)


def test_the_flag_with_an_action_is_a_usage_error(world):
    code, lines = run(world, "--owner", "list")
    assert code == 2 and "Usage" in "\n".join(lines)


def test_without_the_flag_nothing_changes(world):
    code, lines = run(world, "list")
    assert code == 0 and "No devices are waiting" in "\n".join(lines)


# ── the happy path ───────────────────────────────────────────────────────────

def test_it_prints_the_code_waits_and_reports_who_used_it(world):
    lines: list[str] = []
    redeemed: list[dict] = []

    def sleep(_seconds: float) -> None:
        if not redeemed:
            response = beacon_redeems(world, lines)
            assert response.status_code == 200, response.text
            redeemed.append(response.json())

    code = run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append, sleep=sleep)
    text = "\n".join(lines)
    assert code == 0, text
    assert "Will's MacBook Pro" in text and MACBOOK in text and redeemed[0]["device_id"][:8] in text
    assert "owner device" in text
    assert redeemed[0]["token"] not in text and GLOBAL not in text


def test_the_code_is_printed_once_in_two_groups_of_four(world):
    lines: list[str] = []
    run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append,
                     sleep=lambda _s: beacon_redeems(world, lines))
    groups = [line for line in lines if re.search(r"\b[0-9]{4}-[0-9]{4}\b", line)]
    assert len(groups) == 1
    code = printed_code(lines)
    assert not any(code in line for line in lines)         # never ungrouped (the device id may well be digits)


def test_it_says_what_an_owner_device_can_do_and_how_long_the_code_lasts(world):
    lines: list[str] = []
    run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append,
                     sleep=lambda _s: beacon_redeems(world, lines))
    text = "\n".join(lines)
    assert "5 minutes" in text and "once" in text
    assert "approve" in text.lower()


# ── what else can become of it ───────────────────────────────────────────────

def test_an_unused_code_is_reported_expired(world):
    def sleep(_seconds: float) -> None:
        world.clock.now += 61

    code, lines = run(world, "--owner", sleep=sleep)
    assert code == 1 and "expired" in "\n".join(lines).lower()


def test_wrong_guesses_are_reported_as_a_warning(world):
    lines: list[str] = []

    def sleep(_seconds: float) -> None:
        wrong = f"{(int(printed_code(lines)) + 1) % 10**8:08d}"
        for n in range(25):                                   # five addresses, five wrong tries each
            client = TestClient(world.app, base_url="http://192.0.2.50:8005", client=(f"198.51.100.{n // 5}", 1))
            client.post("/api/pair/owner", json={"code": wrong, "device_name": "guesser"})

    code = run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append, sleep=sleep)
    text = "\n".join(lines).lower()
    assert code == 1 and "locked" in text and "wrong" in text


def test_a_newer_code_from_another_terminal_is_reported(world):
    def sleep(_seconds: float) -> None:
        loopback(world).post("/api/pair/owner-codes", headers={"Authorization": f"Bearer {GLOBAL}"})

    code, lines = run(world, "--owner", sleep=sleep)
    assert code == 1 and "replaced" in "\n".join(lines).lower()


def test_ctrl_c_after_someone_used_the_code_reports_who_and_never_says_cancelled(world):
    """The review's case: a code redeemed in the second between two polls, then Ctrl-C. The alarm must still sound."""
    lines: list[str] = []

    def sleep(_seconds: float) -> None:
        assert beacon_redeems(world, lines, peer="198.51.100.66", name="not you").status_code == 200
        raise KeyboardInterrupt

    code = run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append, sleep=sleep)
    text = "\n".join(lines)
    assert "Cancelled" not in text and "cancelled" not in text.lower().split("was cancelled")[0]
    assert "not you" in text and "198.51.100.66" in text and "revoke" in text.lower()
    assert code == 0


def test_ctrl_c_when_the_daemon_cannot_be_told_says_how_long_the_code_lasts(world):
    lines: list[str] = []
    calls = {"n": 0}
    real = loopback(world)

    class Flaky:
        def post(self, *a, **kw):
            return real.post(*a, **kw)

        def get(self, *a, **kw):
            return real.get(*a, **kw)

        def delete(self, *a, **kw):
            calls["n"] += 1
            raise httpx.ConnectError("gone")

    def sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    code = run_pair_command(_args("--owner"), CONFIG, client=Flaky(), out=lines.append, sleep=sleep)
    assert code == 130 and calls["n"] == 1 and "until" in "\n".join(lines)


def test_ctrl_c_cancels_the_code(world):
    lines: list[str] = []

    def sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    code = run_pair_command(_args("--owner"), CONFIG, client=loopback(world), out=lines.append, sleep=sleep)
    assert code == 130 and "cancel" in "\n".join(lines).lower()
    assert beacon_redeems(world, lines).status_code == 401


# ── when it cannot ───────────────────────────────────────────────────────────

def test_no_token_says_so(world):
    code, lines = run(world, "--owner", config={"web": {"api_token": ""}})
    assert code == 1 and "token" in "\n".join(lines).lower()


def test_a_daemon_reached_through_a_relay_refuses_and_the_terminal_says_why(world):
    relayed = TestClient(world.app, base_url="http://127.0.0.1:8005", client=("127.0.0.1", 50123),
                         headers={"X-Forwarded-For": MACBOOK})
    code, lines = run(world, "--owner", client=relayed)
    assert code == 1 and "machine" in "\n".join(lines).lower()


def test_a_daemon_that_predates_owner_codes_says_so(world):
    class Old:
        def post(self, *a, **kw):
            return httpx.Response(404, json={"detail": "Not Found"})

    code, lines = run(world, "--owner", client=Old())
    assert code == 1 and "owner codes" in "\n".join(lines).lower()


def test_an_unreachable_daemon_says_where_it_looked(world):
    class Down:
        def post(self, *a, **kw):
            raise httpx.ConnectError("refused")

    code, lines = run(world, "--owner", client=Down())
    assert code == 1 and "Could not reach the daemon" in "\n".join(lines)


# ── where the terminal finds the daemon (independent review, 2026-10-11) ─────
#
# In home-network mode plain HTTP listens on this machine only and TLS takes the LAN address, so a terminal on the
# machine must use loopback, not the bind address (whose plain port answers nothing). `oara pair approve` shares it.

LAN = "192.168.1.10"


def test_a_home_network_daemon_is_reached_over_loopback(monkeypatch):
    from prometheus.cli.pair import daemon_base_url
    from prometheus.web import tls

    monkeypatch.setattr(tls, "tls_ready", lambda: True)
    config = {"web": {"bind": LAN, "api_port": 8005}, "network": {"home_network": True}}
    assert daemon_base_url(config, env={}) == "http://127.0.0.1:8005"


def test_home_network_asked_for_without_tls_keeps_plain_http_on_loopback_too(monkeypatch):
    from prometheus.cli.pair import daemon_base_url
    from prometheus.web import tls

    monkeypatch.setattr(tls, "tls_ready", lambda: False)
    config = {"web": {"bind": LAN, "api_port": 8005}, "network": {"home_network": True}}
    assert daemon_base_url(config, env={}) == "http://127.0.0.1:8005"


def test_the_owners_plain_http_opt_out_serves_the_lan_address(monkeypatch):
    from prometheus.cli.pair import daemon_base_url
    from prometheus.web import tls

    monkeypatch.setattr(tls, "tls_ready", lambda: False)
    config = {"web": {"bind": LAN, "api_port": 8005},
              "network": {"home_network": True, "allow_plaintext_lan": True}}
    assert daemon_base_url(config, env={}) == f"http://{LAN}:8005"


def test_a_specific_bind_without_home_network_is_reached_at_that_address():
    from prometheus.cli.pair import daemon_base_url

    assert daemon_base_url({"web": {"bind": LAN, "api_port": 8005}}, env={}) == f"http://{LAN}:8005"
