"""Pairing PR 5: TLS on the home network (docs/design/lan-tls-pr5.md; contract docs/PAIRING-APPROVAL-API.md §7.3).

What a client pins and checks, and so what must hold:

* the pin is SHA-256 of the certificate's SPKI DER, all 32 bytes, read from the handshake;
* hello's ``fp`` is its first 16 hex characters, so the certificate is made OVER THE INSTANCE KEY;
* hello's ``tls`` says whether the port that was asked speaks TLS (per listener, not per daemon);
* TLS serves the LAN on its own ports (8006/8011 by default) only in home-network mode, and plain HTTP stays on
  loopback, so this Mac's own clients are unchanged.

The listener tests run real uvicorn and websockets servers on 127.0.0.1 (a LAN bind in a test would raise the macOS
firewall prompt); what they prove is the TLS, not the address.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import ssl
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from prometheus.config.instance_key import ensure_instance_key, instance_fingerprint

TOKEN = "lan-tls-test-token-0123456789abcdef"


@pytest.fixture
def node(tmp_path, monkeypatch):
    path = tmp_path / "node"
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(path))
    return path


class _LazyCert:
    """The module under test, imported on use, so that before it exists each test fails on its own (a failure
    at collection would prove nothing)."""

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module("prometheus.config.instance_cert"), name)


instance_cert = _LazyCert()


def _spki(cert: x509.Certificate) -> bytes:
    return cert.public_key().public_bytes(serialization.Encoding.DER,
                                          serialization.PublicFormat.SubjectPublicKeyInfo)


def _load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


# ── the certificate ──────────────────────────────────────────────────────────

def test_the_certificate_is_made_over_the_instance_key(node):
    key_der = ensure_instance_key()
    path = instance_cert.ensure_instance_certificate()
    assert path == node / "instance.crt"
    assert _spki(_load(path)) == key_der, "the handshake SPKI must be the instance key, or fp is not its prefix"
    full = hashlib.sha256(key_der).hexdigest()
    assert instance_cert.instance_spki_sha256() == full
    assert full[:16] == instance_fingerprint()


def test_names_dates_and_mode(node):
    key_der = ensure_instance_key()
    fp = hashlib.sha256(key_der).hexdigest()
    cert = _load(instance_cert.ensure_instance_certificate())
    names = set(cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
        x509.DNSName))
    assert {"localhost", f"prometheus-{fp[:8]}.local"} <= names
    assert any(n.endswith(".local") and not n.startswith("prometheus-") for n in names), names
    now = datetime.now(timezone.utc)
    assert cert.not_valid_before_utc <= now - timedelta(hours=12)
    assert timedelta(days=5 * 365 - 2) <= cert.not_valid_after_utc - now <= timedelta(days=5 * 365 + 2)
    assert cert.issuer == cert.subject, "self-signed"
    assert stat.S_IMODE((node / "instance.crt").stat().st_mode) == 0o644, "a certificate is public"


def test_it_is_made_once_and_the_key_is_never_touched(node):
    ensure_instance_key()
    key_before = (node / "instance.key").read_bytes()
    first = instance_cert.ensure_instance_certificate().read_bytes()
    assert instance_cert.ensure_instance_certificate().read_bytes() == first
    assert (node / "instance.key").read_bytes() == key_before


def test_it_is_reissued_over_the_same_key_near_expiry(node):
    key_der = ensure_instance_key()
    old = _load(instance_cert.ensure_instance_certificate())
    later = old.not_valid_after_utc - timedelta(days=20)
    new = _load(instance_cert.ensure_instance_certificate(now=later))
    assert new.serial_number != old.serial_number
    assert _spki(new) == key_der, "a renewal must not move the pin"
    assert new.not_valid_after_utc > old.not_valid_after_utc


def test_a_certificate_for_another_key_is_replaced(node):
    key_der = ensure_instance_key()
    other = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "impostor")])
    now = datetime.now(timezone.utc)
    stray = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(other.public_key())
             .serial_number(1).not_valid_before(now).not_valid_after(now + timedelta(days=3650))
             .sign(other, __import__("cryptography.hazmat.primitives.hashes", fromlist=["SHA256"]).SHA256()))
    (node / "instance.crt").write_bytes(stray.public_bytes(serialization.Encoding.PEM))
    assert _spki(_load(instance_cert.ensure_instance_certificate())) == key_der


# ── when TLS runs ────────────────────────────────────────────────────────────

def test_no_plan_on_loopback_or_without_home_network(node):
    from prometheus.web.tls import lan_tls_plan

    assert lan_tls_plan({"network": {"home_network": True}}, "127.0.0.1") is None
    assert lan_tls_plan({"network": {"home_network": True}}, "::1") is None
    assert lan_tls_plan({}, "0.0.0.0") is None
    assert lan_tls_plan({"network": {"home_network": False}}, "0.0.0.0") is None
    assert lan_tls_plan({"network": {"home_network": "yes"}}, "0.0.0.0") is None, "not a boolean is off, as in #700"


def test_the_plan_names_the_lan_address_the_ports_and_the_files(node):
    from prometheus.web.tls import lan_tls_plan

    plan = lan_tls_plan({"network": {"home_network": True}}, "0.0.0.0")
    assert (plan.host, plan.api_port, plan.ws_port) == ("0.0.0.0", 8006, 8011)
    assert (plan.certfile, plan.keyfile) == (node / "instance.crt", node / "instance.key")
    custom = lan_tls_plan({"network": {"home_network": True}, "web": {"tls_api_port": 9443, "tls_ws_port": 9444}},
                          "192.0.2.10")
    assert (custom.host, custom.api_port, custom.ws_port) == ("192.0.2.10", 9443, 9444)


@pytest.mark.parametrize("bad", ["8006", 0, 70000, True, -1])
def test_an_invalid_tls_port_falls_back_to_the_default_and_says_so(node, caplog, bad):
    from prometheus.web.tls import lan_tls_plan

    plan = lan_tls_plan({"network": {"home_network": True}, "web": {"tls_api_port": bad}}, "0.0.0.0")
    assert plan.api_port == 8006
    assert any("tls_api_port" in r.getMessage() for r in caplog.records)


def test_with_a_plan_plain_http_stays_on_loopback(node):
    from prometheus.web.tls import lan_tls_plan, listener_hosts

    assert listener_hosts("0.0.0.0", None) == ("0.0.0.0", None)
    plan = lan_tls_plan({"network": {"home_network": True}}, "0.0.0.0")
    assert listener_hosts("0.0.0.0", plan) == ("127.0.0.1", "0.0.0.0")


# ── real listeners ───────────────────────────────────────────────────────────

def _client_ctx() -> ssl.SSLContext:
    """A client that accepts any certificate, as a first contact does: the test checks the key itself."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _presented_spki(port: int) -> bytes:
    _reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=_client_ctx())
    try:
        der = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
    finally:
        writer.close()
    return _spki(x509.load_der_x509_certificate(der))


async def _wait_started(app, attr):
    for _ in range(300):
        server = getattr(app.state, attr, None)
        if server is not None and server.started:
            return server
        await asyncio.sleep(0.02)
    raise AssertionError(f"{attr} did not start")


@pytest.mark.asyncio
async def test_the_tls_rest_listener_presents_the_instance_key_and_hello_says_tls(node):
    pytest.importorskip("fastapi")
    import httpx

    from prometheus.web.server import create_app, start_web
    from prometheus.web.serving import bound_port
    from prometheus.web.tls import LanTlsPlan

    key_der = ensure_instance_key()
    instance_cert.ensure_instance_certificate()
    plan = LanTlsPlan(host="127.0.0.1", api_port=0, ws_port=0,
                      certfile=node / "instance.crt", keyfile=node / "instance.key")
    app = create_app({"web": {"api_token": TOKEN}})
    plain = asyncio.create_task(start_web(app, host="127.0.0.1", port=0))
    secure = asyncio.create_task(start_web(app, host="127.0.0.1", port=0, tls=plan))
    try:
        plain_port = bound_port(await _wait_started(app, "http_server"))
        tls_port = bound_port(await _wait_started(app, "https_server"))
        assert plain_port and tls_port and plain_port != tls_port
        assert hashlib.sha256(await _presented_spki(tls_port)).digest() == hashlib.sha256(key_der).digest()
        async with httpx.AsyncClient(verify=False) as client:
            over_tls = (await client.get(f"https://127.0.0.1:{tls_port}/api/hello")).json()
            over_plain = (await client.get(f"http://127.0.0.1:{plain_port}/api/hello")).json()
        assert over_tls["tls"] is True and over_plain["tls"] is False
        assert over_tls["fp"] == hashlib.sha256(key_der).hexdigest()[:16] == over_plain["fp"]
    finally:
        for name in ("http_server", "https_server"):
            server = getattr(app.state, name, None)
            if server is not None:
                server.should_exit = True
        await asyncio.wait_for(asyncio.gather(plain, secure), 10)


@pytest.mark.asyncio
async def test_the_tls_websocket_listener_presents_the_instance_key_and_upgrades(node):
    pytest.importorskip("websockets")
    import websockets

    from prometheus.web.tls import LanTlsPlan, server_ssl_context
    from prometheus.web.ws_server import WebSocketBridge

    key_der = ensure_instance_key()
    instance_cert.ensure_instance_certificate()
    plan = LanTlsPlan(host="127.0.0.1", api_port=0, ws_port=0,
                      certfile=node / "instance.crt", keyfile=node / "instance.key")
    bridge = WebSocketBridge()
    await bridge.start(host="127.0.0.1", port=0)
    await bridge.start(host="127.0.0.1", port=0, ssl=server_ssl_context(plan))
    try:
        assert bridge.bound_port and bridge.tls_bound_port and bridge.bound_port != bridge.tls_bound_port
        assert await _presented_spki(bridge.tls_bound_port) == key_der
        async with websockets.connect(f"wss://127.0.0.1:{bridge.tls_bound_port}", ssl=_client_ctx()) as ws:
            first = await asyncio.wait_for(ws.recv(), 5)
        assert '"connected"' in first or "connected" in first
    finally:
        await bridge.stop()


# ── the daemon ───────────────────────────────────────────────────────────────

def test_the_daemon_makes_the_certificate_after_the_key_and_hands_the_plan_to_the_listeners():
    import prometheus.daemon as daemon

    tree = ast.parse(Path(daemon.__file__).read_text(encoding="utf-8"))
    run_daemon = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_daemon")

    def calls(name):
        return [n for n in ast.walk(run_daemon) if isinstance(n, ast.Call)
                and (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)) == name]

    key_line = min(c.lineno for c in calls("ensure_instance_key"))
    cert_calls = calls("ensure_instance_certificate")
    assert cert_calls and min(c.lineno for c in cert_calls) > key_line
    launches = calls("launch_web")
    assert launches and all(any(k.arg == "lan_tls" for k in c.keywords) for c in launches)
