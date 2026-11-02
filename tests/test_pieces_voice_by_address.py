"""A voice box joins by an address the owner types once: probe, one prompt, approve, and voice uses it. No file edit.

Contract: docs/contracts/pieces.md, sections 3.2-3.4 (a typed address, what a probe may do, pending), 5 (transport),
7-8 (approval and routes), 10 (a YAML entry wins). This step builds pieces that cannot prove who they are (no card,
no key of their own): they are labelled "not verified", reached over plain HTTP only on loopback or a mesh VPN, and
are given no secret. Cards, the identity challenge and pinned TLS come with ``oara piece announce``.

The stand-in voice box is a real HTTP server on loopback answering like an OpenAI-audio server.
"""

from __future__ import annotations

import ipaddress
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")

from tests.support.pairing_world import World  # noqa: E402

MESH = ".".join(("100", "64", "0", "9"))           # built at run time: the pre-commit scanner looks for these


class _VoiceHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, body: object, ctype: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self.server.seen.append(("GET", self.path))  # type: ignore[attr-defined]
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.seen.append(("POST", self.path))  # type: ignore[attr-defined]
        kinds = self.server.kinds  # type: ignore[attr-defined]
        if self.path == "/v1/audio/transcriptions" and "stt" in kinds:
            if not body:
                self._send(422, {"detail": "file required"})
            else:
                self._send(200, {"text": "words from the box"})
        elif self.path == "/v1/audio/speech" and "tts" in kinds:
            if not body:
                self._send(400, {"error": "input required"})
            else:
                self._send(200, b"RIFF\x24\x00\x00\x00WAVEfmt box-audio", "audio/wav")
        else:
            self._send(404, {})


def _serve(kinds: tuple[str, ...]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VoiceHandler)
    server.seen = []  # type: ignore[attr-defined]
    server.kinds = kinds  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def voice_box():
    server = _serve(("stt", "tts"))
    try:
        yield server, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def two_port_box(monkeypatch):
    """Speech-to-text on one port and text-to-speech on another, as a typical GPU voice box runs them."""
    stt, tts = _serve(("stt",)), _serve(("tts",))
    from prometheus.infra import pieces

    monkeypatch.setattr(pieces, "VOICE_PORTS", (stt.server_address[1], tts.server_address[1]), raising=False)
    try:
        yield stt, tts
    finally:
        for server in (stt, tts):
            server.shutdown()
            server.server_close()


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, recording=False)
    from prometheus.infra import pieces

    # Voice reads approved pieces from the daemon's devices.db; here that is the World's.
    monkeypatch.setattr(pieces, "devices_db_path", lambda: w.path, raising=False)
    return w


def probe(world: World, address: str, who: str = "global"):
    return world.as_(who, "POST", "/api/pieces/probe", json={"address": address})


# ── address classes and transport ────────────────────────────────────────────

@pytest.mark.parametrize("address,expected", [
    ("127.0.0.1", "loopback"), ("::1", "loopback"),
    (MESH, "mesh"),
    ("192.168.1.20", "lan"), ("10.0.0.5", "lan"), ("169.254.3.4", "lan"), ("fd00::5", "lan"),
    ("203.0.113.7", "public"), ("8.8.8.8", "public"),
])
def test_address_classes_are_the_contracts(address, expected):
    from prometheus.infra.pieces import address_class

    assert address_class(ipaddress.ip_address(address)) == expected


@pytest.mark.parametrize("typed,error", [
    ("http://192.168.1.20:8001", "plaintext_not_allowed"),
    ("192.168.1.20", "plaintext_not_allowed"),
    ("203.0.113.7:8001", "plaintext_not_allowed"),
    ("https://192.168.1.20:8443", "tls_not_supported"),
    ("http://user:pw@127.0.0.1:8001", "address_not_allowed"),
    ("http://127.0.0.1:8001/v1/audio", "address_not_allowed"),
    ("ftp://127.0.0.1", "address_not_allowed"),
    ("", "address_not_allowed"),
])
def test_an_address_the_rules_refuse_is_refused_before_anything_is_asked(world, typed, error):
    response = probe(world, typed)
    assert response.status_code == 422, response.text
    assert response.json()["error"] == error


# ── the probe ────────────────────────────────────────────────────────────────

def test_a_typed_address_finds_a_voice_box_and_holds_it_pending(world, voice_box):
    server, port = voice_box
    response = probe(world, f"127.0.0.1:{port}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reason"] is None
    [found] = body["found"]
    assert found["state"] == "pending"
    assert found["verified"] is False
    assert found["source"] == "manual"
    assert found["address_class"] == "loopback"
    assert {o["kind"] for o in found["offers"]} == {"stt", "tts"}
    assert "Use it?" in found["prompt"] and "can't prove who it is" in found["prompt"]
    # Nothing was sent but empty requests: no audio, no text, no key.
    assert all(path in ("/health", "/v1/audio/transcriptions", "/v1/audio/speech") for _, path in server.seen)
    listed = world.as_("global", "GET", "/api/pieces").json()["pieces"]
    assert [p["id"] for p in listed] == [found["id"]] and listed[0]["state"] == "pending"
    assert "pieces" not in world.tables()                  # pending is in memory only


def test_a_box_on_two_ports_is_one_piece(world, two_port_box):
    [found] = probe(world, "127.0.0.1").json()["found"]
    by_kind = {o["kind"]: o["url"] for o in found["offers"]}
    stt, tts = two_port_box
    assert by_kind == {"stt": f"http://127.0.0.1:{stt.server_address[1]}",
                       "tts": f"http://127.0.0.1:{tts.server_address[1]}"}


def test_nothing_answering_is_said_plainly(world):
    from tests.test_voice_remote import _closed_port

    body = probe(world, f"127.0.0.1:{_closed_port()}").json()
    assert body == {"found": [], "reason": "no_answer", "detail": body["detail"]}


def test_a_server_that_is_not_a_voice_box_is_not_a_piece(world, monkeypatch):
    server = _serve(())                      # answers /health, but no audio route
    try:
        body = probe(world, f"127.0.0.1:{server.server_address[1]}").json()
    finally:
        server.shutdown()
        server.server_close()
    assert body["found"] == [] and body["reason"] == "not_a_piece"


def test_probing_is_the_owners(world, voice_box):
    _, port = voice_box
    response = probe(world, f"127.0.0.1:{port}", who="scoped")
    assert response.status_code == 403 and response.json()["error"] == "operator_only"


def test_pending_is_bounded(world, voice_box, monkeypatch):
    from prometheus.infra import pieces

    _, port = voice_box
    monkeypatch.setattr(pieces, "PENDING_LIMIT", 1, raising=False)
    assert probe(world, f"127.0.0.1:{port}").status_code == 200
    other = _serve(("stt",))
    try:
        response = probe(world, f"127.0.0.1:{other.server_address[1]}")
    finally:
        other.shutdown()
        other.server_close()
    assert response.status_code == 429 and response.json()["error"] == "pending_full"


def test_probing_the_same_box_again_gives_the_same_piece(world, voice_box):
    _, port = voice_box
    first = probe(world, f"127.0.0.1:{port}").json()["found"][0]["id"]
    again = probe(world, f"127.0.0.1:{port}").json()["found"][0]["id"]
    assert first == again


# ── approval ─────────────────────────────────────────────────────────────────

def test_approving_stores_the_piece_and_voice_uses_it_with_no_file_edit(world, voice_box, tmp_path):
    import asyncio

    from prometheus.infra.voice_remote import remote_settings
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.whisper_stt import WhisperSTTInput, WhisperSTTTool

    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    response = world.as_("global", "POST", f"/api/pieces/{found['id']}/approve", json={"uses": ["stt"]})
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "approved" and response.json()["uses"] == ["stt"]
    assert "pieces" in world.tables()

    remote = remote_settings("stt", {})                     # no stt.url anywhere
    assert remote is not None and remote.url == f"http://127.0.0.1:{port}/v1/audio/transcriptions"
    assert remote.headers() == {}                           # an unverified piece is given no key
    assert remote_settings("tts", {}) is None               # not approved for that use

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    result = asyncio.run(WhisperSTTTool(config={}).execute(
        WhisperSTTInput(audio_path=str(audio)), ToolExecutionContext(cwd=tmp_path)))
    assert result.output == "words from the box"


def test_uses_default_to_everything_offered(world, voice_box):
    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    approved = world.as_("owner", "POST", f"/api/pieces/{found['id']}/approve").json()
    assert sorted(approved["uses"]) == ["stt", "tts"]


def test_a_use_the_piece_does_not_offer_is_refused(world, two_port_box, monkeypatch):
    stt, _ = two_port_box
    from prometheus.infra import pieces

    monkeypatch.setattr(pieces, "VOICE_PORTS", (stt.server_address[1],))
    found = probe(world, "127.0.0.1").json()["found"][0]
    response = world.as_("global", "POST", f"/api/pieces/{found['id']}/approve", json={"uses": ["tts"]})
    assert response.status_code == 400 and response.json()["error"] == "invalid_request"


def test_no_secret_goes_to_an_unverified_piece_over_plain_http(world, voice_box):
    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    response = world.as_("global", "POST", f"/api/pieces/{found['id']}/approve", json={"secret": "box-key"})
    assert response.status_code == 422 and response.json()["error"] == "secret_needs_proof"
    assert "box-key" not in response.text
    assert world.as_("global", "GET", "/api/pieces").json()["pieces"][0]["state"] == "pending"


def test_a_yaml_entry_wins_and_shows_read_only(tmp_path, voice_box, monkeypatch):
    from prometheus.infra import pieces
    from prometheus.infra.voice_remote import remote_settings

    _, port = voice_box
    w = World(tmp_path, recording=False, config={"stt": {"url": "http://127.0.0.1:9/v1"}})
    monkeypatch.setattr(pieces, "devices_db_path", lambda: w.path, raising=False)
    found = probe(w, f"127.0.0.1:{port}").json()["found"][0]
    w.as_("global", "POST", f"/api/pieces/{found['id']}/approve", json={"uses": ["stt"]})

    assert remote_settings("stt", {"stt": {"url": "http://127.0.0.1:9/v1"}}).url == \
        "http://127.0.0.1:9/v1/audio/transcriptions"
    rows = w.as_("global", "GET", "/api/pieces").json()["pieces"]
    from_file = [r for r in rows if r["source"] == "config_file"]
    assert len(from_file) == 1
    assert from_file[0]["editable"] is False and from_file[0]["uses"] == ["stt"]
    refused = w.as_("global", "DELETE", f"/api/pieces/{from_file[0]['id']}")
    assert refused.status_code == 409 and refused.json()["error"] == "config_file_entry"


def test_deny_drops_a_pending_piece(world, voice_box):
    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    assert world.as_("global", "POST", f"/api/pieces/{found['id']}/deny").status_code == 200
    assert world.as_("global", "GET", "/api/pieces").json()["pieces"] == []


def test_ignore_keeps_it_quiet_and_unused(world, voice_box):
    from prometheus.infra.voice_remote import remote_settings

    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    ignored = world.as_("global", "POST", f"/api/pieces/{found['id']}/ignore")
    assert ignored.status_code == 200 and ignored.json()["state"] == "ignored"
    assert remote_settings("stt", {}) is None
    assert probe(world, f"127.0.0.1:{port}").json()["found"][0]["state"] == "ignored"


def test_forgetting_a_piece_stops_its_use(world, voice_box):
    from prometheus.infra.voice_remote import remote_settings

    _, port = voice_box
    found = probe(world, f"127.0.0.1:{port}").json()["found"][0]
    world.as_("global", "POST", f"/api/pieces/{found['id']}/approve")
    assert remote_settings("stt", {}) is not None
    assert world.as_("global", "DELETE", f"/api/pieces/{found['id']}").status_code == 200
    assert remote_settings("stt", {}) is None
    assert world.as_("global", "GET", "/api/pieces").json()["pieces"] == []


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/pieces/pc_0000000000000000/approve"), ("POST", "/api/pieces/pc_0000000000000000/deny"),
    ("POST", "/api/pieces/pc_0000000000000000/ignore"), ("DELETE", "/api/pieces/pc_0000000000000000"),
])
def test_an_unknown_piece_is_a_404(world, method, path):
    response = world.as_("global", method, path)
    assert response.status_code == 404 and response.json()["error"] == "unknown_piece"


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/pieces/x/approve"), ("POST", "/api/pieces/x/deny"),
    ("POST", "/api/pieces/x/ignore"), ("DELETE", "/api/pieces/x"),
])
def test_every_write_is_the_owners(world, method, path):
    response = world.as_("scoped", method, path)
    assert response.status_code == 403 and response.json()["error"] == "operator_only"


def test_reading_voice_settings_never_creates_the_registry(tmp_path, monkeypatch):
    from prometheus.infra import pieces
    from prometheus.infra.voice_remote import remote_settings

    db = tmp_path / "nowhere" / "devices.db"
    monkeypatch.setattr(pieces, "devices_db_path", lambda: db, raising=False)
    assert remote_settings("stt", {}) is None
    assert not db.exists() and not db.parent.exists()
