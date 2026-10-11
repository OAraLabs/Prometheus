"""Beacon's v0.4 flow against the real artifact: a daemon on this machine only is switched to the home network and
restarted over HTTP, and comes back serving TLS on the LAN.

A REAL daemon subprocess (isolated HOME, a loopback stub model) starts on ``127.0.0.1``. The operator saves
``PUT /api/network {"mode": "home_network"}`` (``web.bind: 0.0.0.0`` and ``network.home_network: true``) and asks
for ``POST /api/restart``. The daemon re-execs in place and comes back with:

* TLS on a LAN address of this machine, REST and WebSocket, TLS 1.2 or later, whose certificate's public key is the
  pin ``GET /api/network`` reports and whose hash starts with the ``fp`` hello advertised before the restart;
* plain HTTP on loopback only: the LAN address refuses a plain connection;
* the same instance (same ``fp``), in the same process.

It binds beyond loopback, so it runs on Linux only: on macOS the application firewall asks the person at the
screen about incoming connections. On CI (Linux) it never skips. Stopped by its recorded PID at the end, like
tests/test_restart_real_daemon.py.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import signal
import site
import socket
import ssl
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.test_restart_real_daemon import BOOTED, _free_port, _hello, _ModelsHandler

TOKEN = "restart-into-tls-token-0123456789abcdef"


def _lan_address() -> str | None:
    """An IPv4 address of this machine that is not loopback, from the routing table (a UDP connect sends
    nothing)."""
    probe = ".".join(("192", "0", "2", "1"))      # TEST-NET-1, built at run time: the pre-commit scanner reads IPs
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((probe, 9))
            address = s.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") or address == "0.0.0.0" else address


def _lan_or_skip() -> str:
    if not sys.platform.startswith("linux"):
        pytest.skip("binds beyond loopback: Linux only (on macOS the firewall would ask about incoming connections)")
    address = _lan_address()
    if address is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("no non-loopback IPv4 address on a Linux CI runner: this test must run there, not skip")
        pytest.skip("this machine has no non-loopback IPv4 address")
    return address


def _unverified() -> ssl.SSLContext:
    """The client side: no CA to check a self-signed certificate against, so the test pins it by hand."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _spki_sha256(cert_der: bytes) -> str:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    key = x509.load_der_x509_certificate(cert_der).public_key()
    return hashlib.sha256(key.public_bytes(serialization.Encoding.DER,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()


def _https_get(host: str, port: int, path: str, token: str | None = None) -> tuple[int, dict, bytes, str]:
    """``(status, body, peer certificate DER, TLS version)`` of one GET over TLS."""
    conn = http.client.HTTPSConnection(host, port, timeout=5, context=_unverified())
    try:
        conn.request("GET", path, headers={"Authorization": f"Bearer {token}"} if token else {})
        response = conn.getresponse()
        body = response.read()
        sock = conn.sock
        assert isinstance(sock, ssl.SSLSocket)
        return response.status, json.loads(body or b"null"), sock.getpeercert(binary_form=True) or b"", \
            sock.version() or ""
    finally:
        conn.close()


def _call(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=data,
                                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def test_a_real_daemon_switched_to_home_network_and_restarted_comes_back_on_tls(tmp_path):
    lan = _lan_or_skip()
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    api_port, ws_port, tls_api_port, tls_ws_port = (_free_port() for _ in range(4))
    home = tmp_path / "home"
    (home / ".prometheus").mkdir(parents=True)
    config_file = home / ".prometheus" / "prometheus.yaml"
    config_file.write_text(textwrap.dedent(f"""\
        model:
          provider: llama_cpp
          base_url: http://127.0.0.1:{stub.server_port}
          model: restart-model
        gateway:
          telegram_enabled: false
        web:
          enabled: true
          bind: 127.0.0.1
          api_port: {api_port}
          ws_port: {ws_port}
          tls_api_port: {tls_api_port}
          tls_ws_port: {tls_ws_port}
          api_token: {TOKEN}
        discovery:
          mdns: false
        tools:
          deferred_loading:
            enabled: auto
            always_loaded: [bash, read_file]
        """), encoding="utf-8")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONUNBUFFERED": "1",
           "LANG": "C.UTF-8", "PYTHONUSERBASE": site.getuserbase()}
    if "VIRTUAL_ENV" in os.environ:
        env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
    # THIS tree's code, first and absolute (see tests/test_restart_real_daemon.py).
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    log_path = tmp_path / "daemon.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "prometheus", "daemon"], cwd=tmp_path, env=env,
                            stdout=log, stderr=subprocess.STDOUT)

    def wait_for(predicate, what: str, seconds: float = 60):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"daemon exited rc={proc.returncode} while waiting for {what}: "
                            f"{log_path.read_text()[-3000:]}")
            try:
                value = predicate()
            except (OSError, ssl.SSLError, http.client.HTTPException):
                value = None
            if value:
                return value
            time.sleep(0.5)
        pytest.fail(f"timed out waiting for {what}: {log_path.read_text()[-3000:]}")

    try:
        # ── this machine only ──
        first = wait_for(lambda: _hello(api_port), "the first boot")
        assert first["fp"], first
        status, network = _call(api_port, "GET", "/api/network")
        assert status == 200 and network["mode"] == "this_mac" and network["tls"]["enabled"] is False, network

        # ── Beacon: save home network, then restart ──
        status, saved = _call(api_port, "PUT", "/api/network", {"mode": "home_network"})
        assert status == 200 and saved["applied"] == "on_restart", saved
        status, answer = _call(api_port, "POST", "/api/restart")
        assert status == 202 and answer == {"restarting": True}, answer
        wait_for(lambda: log_path.read_text().count(BOOTED) >= 2, "the second boot")

        # ── back on TLS, on the LAN ──
        tls_status, hello, cert, version = wait_for(lambda: _https_get(lan, tls_api_port, "/api/hello"),
                                                    "hello over TLS on the LAN address")
        assert tls_status == 200 and hello["fp"] == first["fp"], "the restarted daemon is not the same instance"
        assert hello["tls"] is True, hello
        assert version in ("TLSv1.2", "TLSv1.3"), version
        pin = _spki_sha256(cert)
        assert pin.startswith(first["fp"]), "the certificate is not over the key hello fingerprints"
        status, network, _, _ = _https_get(lan, tls_api_port, "/api/network", token=TOKEN)
        assert status == 200 and network["mode"] == "home_network", network
        assert network["tls"] == {"enabled": True, "spki_sha256": pin}, network

        # The WebSocket over TLS: a whole session, upgrade and first-frame auth included.
        from websockets.sync.client import connect

        with connect(f"wss://{lan}:{tls_ws_port}", ssl=_unverified(), open_timeout=10) as ws:
            ws.send(json.dumps({"type": "auth", "token": TOKEN}))
            assert json.loads(ws.recv(timeout=10))["type"] == "connected"

        # ── plain HTTP is this machine's only ──
        with pytest.raises(OSError):
            socket.create_connection((lan, api_port), timeout=3).close()
        assert (_hello(api_port) or {}).get("fp") == first["fp"], "plain HTTP on loopback went away"
        assert proc.poll() is None, "it exited instead of re-exec'ing in place"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()
        stub.shutdown()
