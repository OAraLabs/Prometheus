"""docs/reference/pairing-join-requests.md says what the code does.

That file is the contract the iOS and desktop clients are built to. A contract that drifts from the daemon is worse than
none, so each claim that can be read back out of it is: the numbers, the routes and who may call them, the statuses, the
platform names, the frame types, the error codes and the sealing recipe's names. Change the code and this fails until the
file is changed with it, which is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prometheus.config import owner_codes as OC
from prometheus.config import pair_requests as PR
from prometheus.config import pair_seal
from prometheus.config.device_store import registry_platform
from prometheus.web import pairing_routes as PRT
from prometheus.web.public_routes import PUBLIC_ROUTES
from prometheus.web.route_access import OPERATOR_ONLY
from tests.support.pairing_world import GLOBAL, World

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "reference" / "pairing-join-requests.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a)]


# ── the numbers ──────────────────────────────────────────────────────────────

def _expected_numbers() -> dict[tuple[str, str | None], int]:
    settings = PR.PairingSettings()
    return {
        ("pairing.request_ttl_seconds", "default"): PR.TTL_DEFAULT,
        ("pairing.request_ttl_seconds", "minimum"): PR.TTL_MIN,
        ("pairing.request_ttl_seconds", "maximum"): PR.TTL_MAX,
        ("pairing.max_pending", None): settings.max_pending,
        ("pairing.max_pending_per_source", None): settings.max_pending_per_source,
        ("pairing.max_requests_per_source_per_hour", None): settings.max_requests_per_source_per_hour,
        ("DELIVERY_WINDOW_SECONDS", None): PR.DELIVERY_WINDOW_SECONDS,
        ("PURGE_AFTER_SECONDS", None): PR.PURGE_AFTER_SECONDS,
        ("MAX_BODY_BYTES", None): PRT.MAX_BODY_BYTES,
        ("POLL_INTERVAL_SECONDS", None): PRT.POLL_INTERVAL_SECONDS,
        ("MIN_POLL_GAP_SECONDS", None): int(PRT.MIN_POLL_GAP_SECONDS),
        ("WRONG_SECRETS_PER_MINUTE", None): PRT.WRONG_SECRETS_PER_MINUTE,
        ("OWNER_CODE_DIGITS", None): OC.CODE_DIGITS,
        ("OWNER_CODE_TTL_SECONDS", None): OC.TTL_SECONDS,
        ("OWNER_CODE_MAX_WRONG_PER_SOURCE", None): OC.MAX_WRONG_PER_SOURCE,
        ("OWNER_CODE_MAX_WRONG_PER_MINUTE", None): OC.MAX_WRONG_PER_MINUTE,
        ("DEVICE_NAME_MAX_CHARS", None): PRT._MAX_NAME_CHARS,
    }


def _documented_numbers(text: str) -> dict[tuple[str, str | None], int]:
    table = _between(text, "## 8. The numbers", "## 9. Configuration")
    rows = re.findall(r"^\| `([^`]+)`(?: (default|minimum|maximum))? \| (\d+) \|", table, re.M)
    return {(name, kind or None): int(value) for name, kind, value in rows}


def check_numbers(text: str) -> None:
    assert _documented_numbers(text) == _expected_numbers()


def test_every_number_in_the_table_is_the_codes():
    check_numbers(_doc())


def test_the_check_can_fail():
    # A contract test that cannot fail is decoration: change one number and it must notice.
    text = _doc()
    assert "| `OWNER_CODE_DIGITS` | 8 |" in text
    with pytest.raises(AssertionError):
        check_numbers(text.replace("| `OWNER_CODE_DIGITS` | 8 |", "| `OWNER_CODE_DIGITS` | 6 |"))


# ── the routes and who may call them ─────────────────────────────────────────

def _documented_routes(text: str) -> dict[tuple[str, str], str]:
    table = _between(text, "## 1. Who may call what", "The four public routes")
    rows = re.findall(r"^\| `(GET|POST|DELETE|PUT) (/api/[^`]+)` \| ([^|]+?) \|", table, re.M)
    return {(method, path): who.strip() for method, path, who in rows}


def test_every_documented_route_is_mounted_and_classified_as_documented():
    routes_md = (REPO / "docs" / "reference" / "routes.md").read_text(encoding="utf-8")
    documented = _documented_routes(_doc())
    assert documented, "the route table was not found"
    for (method, path), who in documented.items():
        assert path in routes_md, f"{path} is documented but the app does not mount it"
        if who.startswith("anyone"):
            assert (method, path) in PUBLIC_ROUTES, f"{method} {path} is documented as public but is behind the gate"
        else:
            assert (method, path) not in PUBLIC_ROUTES, f"{method} {path} is documented as private but is public"
            assert (method, path) in OPERATOR_ONLY, f"{method} {path} is not an operator-only route"


def test_no_pairing_route_is_left_out_of_the_table():
    documented = set(_documented_routes(_doc()))
    actual = {(m, p) for (m, p) in PUBLIC_ROUTES | OPERATOR_ONLY
              if p.startswith("/api/pair/") and p != "/api/pair/local"}
    assert actual == documented, f"missing from the doc: {sorted(actual - documented)}; not real: {sorted(documented - actual)}"


# ── statuses, platforms, frames, errors, the sealing recipe ──────────────────

def test_the_poll_statuses_are_the_stores():
    table = _between(_doc(), "### 2.2", "### 2.3")
    shown = set(re.findall(r"^\| `([a-z]+)` \|", table, re.M)) - {"status"}   # "status" is the column's own heading
    assert shown == {PR.PENDING, PR.APPROVED, PR.DENIED, PR.CANCELED, PR.EXPIRED, PR.DELIVERED}


def test_the_owner_code_statuses_are_the_stores():
    text = _doc()
    for status in (OC.PENDING, OC.REDEEMED, OC.EXPIRED, OC.REPLACED, OC.CANCELED):
        assert f"`{status}`" in text


def test_the_platform_names_are_the_routes():
    text = _doc()
    for name in PRT._PLATFORMS:
        assert f"`{name}`" in text
    # The registry narrows it again, and the doc says so.
    assert registry_platform("windows") == "other" and registry_platform("ios") == "ios"
    assert "`ios`, `macos` or `other`" in text


def test_the_frame_types_are_the_daemons():
    text = _doc()
    ws = (REPO / "src" / "prometheus" / "web" / "ws_server.py").read_text(encoding="utf-8")
    owner = (REPO / "src" / "prometheus" / "web" / "owner_code_routes.py").read_text(encoding="utf-8")
    for frame, source in (("pairing_pending", ws), ("pairing_resolved", ws), ("owner_code", owner)):
        assert f'"{frame}"' in source and f'"type": "{frame}"' in text, frame


def test_every_error_code_in_the_table_is_one_the_daemon_sends():
    text = _doc()
    table = text[text.index("## 10. Error codes"):]
    codes = set(re.findall(r"`([a-z_]+)`", "".join(re.findall(r"^\| \d{3} \| ([^|]+) \|", table, re.M))))
    source = "".join((REPO / "src" / "prometheus" / "web" / f).read_text(encoding="utf-8")
                     for f in ("pairing_routes.py", "owner_code_routes.py"))
    assert codes, "the error table was not found"
    for code in sorted(codes):
        assert f'"{code}"' in source or f"'{code}'" in source, f"{code} is documented but never sent"


def test_the_sealing_recipe_names_are_the_codes():
    text = _doc()
    assert pair_seal.ALG in text
    assert pair_seal._SEAL_INFO.decode() in text
    assert "pairing_seal_v1.json" in text and (REPO / "tests" / "vectors" / "pairing_seal_v1.json").is_file()
    assert pair_seal._MATCH_DOMAIN.decode() in text and pair_seal._MATCH_DOMAIN_V2.decode() in text
    assert pair_seal.MATCH_CODE_VERSIONS == (1, 2)


# ── the response shapes, from a live flow on the real app ────────────────────

def test_the_documented_response_fields_are_the_ones_the_daemon_sends(tmp_path):
    """Every field name in the tables is read from a real response, and is in the doc in backticks."""
    text = _doc()
    world = World(tmp_path)
    created, requester, client = world.created()
    world.clock.now += 2
    shapes: dict[str, tuple[set[str], set[str]]] = {}

    def expect(label: str, body: dict, keys: set[str]) -> None:
        shapes[label] = (set(body), keys)

    expect("create", created, {"request_id", "poll_secret", "match_code", "match_code_version", "instance_public_key",
                               "instance_id", "node_pub", "expires_at", "ttl_seconds", "poll_interval_seconds", "notified"})
    expect("poll pending", world.poll(created, client).json(), {"status", "expires_at", "notified"})
    world.clock.now += 2
    listed = world.as_("owner", "GET", "/api/pair/requests").json()["requests"][0]
    expect("list item", listed, {"request_id", "device_name", "platform", "source_ip", "match_code", "created_at",
                                 "expires_at", "ttl_seconds"})
    world.clock.now += 2
    expect("approve", world.approve(created, who="owner").json(), {"request_id", "status", "device_id", "name", "platform"})
    world.clock.now += 2
    approved = world.poll(created, client).json()
    expect("poll approved", approved, {"status", "device_id", "sealed", "endpoints", "tls", "approved_at", "instance_id",
                                       "node_pub"})
    expect("sealed", approved["sealed"], {"alg", "ephemeral_public_key", "nonce", "ciphertext"})
    expect("sealed payload", requester.unseal(created["request_id"], approved["sealed"]), {"token", "device_id", "name"})
    expect("endpoints", approved["endpoints"], {"rest", "ws"})
    other, _, _ = world.created(source="192.0.2.31")
    expect("deny", world.as_("global", "POST", f"/api/pair/requests/{other['request_id']}/deny").json(),
           {"request_id", "status"})

    host = TestClient(world.app, client=("127.0.0.1", 50000))
    bearer = {"Authorization": f"Bearer {GLOBAL}"}
    minted = host.post("/api/pair/owner-codes", headers=bearer).json()
    expect("owner mint", minted, {"code_id", "code", "expires_at", "ttl_seconds"})
    expect("owner status", host.get(f"/api/pair/owner-codes/{minted['code_id']}", headers=bearer).json(),
           {"code_id", "status", "expires_at", "redeemed_by", "wrong_attempts"})
    redeemed = TestClient(world.app, client=("192.0.2.70", 50000)).post(
        "/api/pair/owner", json={"code": minted["code"], "device_name": "Alex's iPhone", "platform": "ios"}).json()
    expect("owner redeem", redeemed, {"token", "device_id", "name", "owner", "endpoints", "tls", "instance_id", "node_pub"})
    done = host.get(f"/api/pair/owner-codes/{minted['code_id']}", headers=bearer).json()
    expect("redeemed_by", done["redeemed_by"], {"id", "name", "platform", "source", "this_host"})

    for label, (actual, documented) in shapes.items():
        assert actual == documented, f"{label}: the daemon sends {sorted(actual)}, this test (and the doc) say {sorted(documented)}"
        for key in documented:
            assert f"`{key}`" in text, f"{label}: `{key}` is not in docs/reference/pairing-join-requests.md"


def test_an_owner_device_cannot_mint_an_owner_code_and_an_approval_is_never_an_owner(tmp_path):
    """The two sentences the iOS and desktop clients most need to believe, run against the real routes."""
    world = World(tmp_path)
    refused = world.as_("owner", "POST", "/api/pair/owner-codes")
    assert refused.status_code == 403 and refused.json()["error"] == "global_token_only"
    relayed = TestClient(world.app, client=("192.0.2.50", 50000)).post(
        "/api/pair/owner-codes", headers={"Authorization": f"Bearer {GLOBAL}"})
    assert relayed.status_code == 403 and relayed.json()["error"] == "not_this_machine"

    created, _, client = world.created()
    world.clock.now += 2
    assert world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/approve",
                     json={"owner": True}).status_code == 400, "an approval has no field that asks for more"
    assert world.approve(created, who="owner").status_code == 200
    world.clock.now += 2
    assert world.as_("scoped", "GET", "/api/pair/requests").status_code == 403
