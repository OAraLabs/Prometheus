"""An owner-approved voice box that asks for a key gets it, once, from the owner; and nothing else ever does.

Contract: docs/contracts/pieces.md, sections 5.2 (rule 3) and 7.3. The flow: a probe finds a box whose audio routes
answer 401, so it needs a key; the owner is asked once and sends it with the approval (an owner-only route, never the
model or a tool); it is written to the env file and voice sends it as a bearer token. The rules each test pins:

* never sent before approval (a probe sends no key, whatever its body says);
* never sent to a LAN or public address: only loopback or a mesh VPN address;
* never in an answer, a log line or ``GET /api/pieces``;
* the agent's tools cannot read it (the env file is under the gate's floor, and the shell's environment is scrubbed).

The stand-in is a real HTTP server on loopback that requires the key on its audio routes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")

from tests.support.pairing_world import World  # noqa: E402

KEY = "box-key-" + "q7Z" * 8


class _KeyedVoice(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self.server.seen.append((self.path, self.headers.get("Authorization")))  # type: ignore[attr-defined]
        self._send(200 if self.path == "/health" else 404, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        auth = self.headers.get("Authorization")
        self.server.seen.append((self.path, auth))  # type: ignore[attr-defined]
        if self.path != "/v1/audio/transcriptions":
            self._send(404, {})
        elif auth != f"Bearer {KEY}":
            self._send(401, {"error": "a key is required"})
        elif not body:
            self._send(422, {"detail": "file required"})
        else:
            self._send(200, {"text": "words from the keyed box"})


@pytest.fixture
def keyed_box():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeyedVoice)
    server.seen = []  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "piece-env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    return path


@pytest.fixture
def world(tmp_path, monkeypatch, env_file):
    w = World(tmp_path, recording=False)
    from prometheus.infra import pieces

    monkeypatch.setattr(pieces, "devices_db_path", lambda: w.path)
    return w


def found(world: World, port: int, **extra) -> dict:
    response = world.as_("global", "POST", "/api/pieces/probe", json={"address": f"127.0.0.1:{port}", **extra})
    assert response.status_code == 200, response.text
    [piece] = response.json()["found"]
    return piece


def approve(world: World, piece_id: str, **body):
    return world.as_("global", "POST", f"/api/pieces/{piece_id}/approve", json=body)


def transcribe(tmp_path) -> str:
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.whisper_stt import WhisperSTTInput, WhisperSTTTool

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 64)
    return asyncio.run(WhisperSTTTool(config={}).execute(
        WhisperSTTInput(audio_path=str(audio)), ToolExecutionContext(cwd=tmp_path))).output


# ── asked once, before anything is sent ──────────────────────────────────────

def test_a_box_that_wants_a_key_says_so_and_is_sent_none_by_the_probe(world, keyed_box):
    server, port = keyed_box
    piece = found(world, port, secret=KEY)                 # a probe's body cannot carry a key to the box
    assert piece["needs_key"] is True
    assert "key" in piece["prompt"]
    assert server.seen and all(auth is None for _, auth in server.seen)


def test_approving_a_box_that_wants_a_key_without_one_is_refused(world, keyed_box):
    _, port = keyed_box
    piece = found(world, port)
    response = approve(world, piece["id"])
    assert response.status_code == 422 and response.json()["error"] == "key_required"
    assert world.as_("global", "GET", "/api/pieces").json()["pieces"][0]["state"] == "pending"


def test_the_key_goes_to_the_env_file_and_voice_sends_it(world, keyed_box, env_file, tmp_path, monkeypatch):
    server, port = keyed_box
    piece = found(world, port)
    server.seen.clear()
    response = approve(world, piece["id"], secret=KEY)
    assert response.status_code == 200, response.text
    assert server.seen == []                               # approving sends nothing to the box
    stored = env_file.read_text()
    assert KEY in stored and "PROMETHEUS_PIECE_" in stored
    assert env_file.stat().st_mode & 0o077 == 0            # owner-only, as every key in this file
    import os

    assert not any(value == KEY for value in os.environ.values())   # not in the daemon's environment either

    assert transcribe(tmp_path) == "words from the keyed box"
    assert ("/v1/audio/transcriptions", f"Bearer {KEY}") in server.seen


def test_the_key_is_never_in_an_answer(world, keyed_box):
    _, port = keyed_box
    piece = found(world, port)
    answers = [approve(world, piece["id"], secret=KEY).text,
               world.as_("global", "GET", "/api/pieces").text,
               world.as_("global", "POST", "/api/pieces/probe", json={"address": f"127.0.0.1:{port}"}).text]
    for text in answers:
        assert KEY not in text
        assert "PROMETHEUS_PIECE_" not in text             # nor the name of the variable that holds it


def test_the_key_is_never_in_a_log_line(world, keyed_box, tmp_path, caplog):
    _, port = keyed_box
    with caplog.at_level(logging.DEBUG):
        piece = found(world, port)
        approve(world, piece["id"], secret=KEY)
        transcribe(tmp_path)
        approve(world, piece["id"], secret=KEY + "\n")     # refused, and still not logged
    assert caplog.records
    assert not any(KEY in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("bad", ["", "has space", "line\nbreak", "quote'd", 'quote"d',
                                 pytest.param("x" * 4097, id="too-long")])
def test_a_key_that_cannot_sit_on_one_env_file_line_is_refused(world, keyed_box, env_file, bad):
    _, port = keyed_box
    piece = found(world, port)
    response = approve(world, piece["id"], secret=bad)
    assert response.status_code in (400, 422)
    assert response.json()["error"] in ("invalid_request", "key_required")
    if bad:
        assert bad not in response.text
    assert not env_file.exists() or "PROMETHEUS_PIECE_" not in env_file.read_text()


# ── never to a LAN or public address ─────────────────────────────────────────

def _stored_row(world: World) -> tuple[str, str]:
    with sqlite3.connect(world.path) as conn:
        return conn.execute("SELECT id, card_json FROM pieces").fetchone()


def test_a_stored_piece_on_a_lan_address_is_given_no_key(world, keyed_box):
    """A row that says LAN (written by an older build, or by hand) never gets the key, wherever its url points."""
    from prometheus.infra.voice_remote import remote_settings

    _, port = keyed_box
    piece = found(world, port)
    approve(world, piece["id"], secret=KEY)
    with sqlite3.connect(world.path) as conn:
        conn.execute("UPDATE pieces SET address_class = 'lan'")
    remote = remote_settings("stt", {})
    assert remote is None or remote.headers() == {}


def test_a_stored_piece_whose_url_is_a_public_address_is_given_no_key(world, keyed_box):
    from prometheus.infra.voice_remote import remote_settings

    _, port = keyed_box
    piece = found(world, port)
    approve(world, piece["id"], secret=KEY)
    piece_id, card_json = _stored_row(world)
    card = json.loads(card_json)
    for offer in card["offers"]:
        offer["url"] = "http://198.51.100.7:8001"          # documentation range: a public address
    with sqlite3.connect(world.path) as conn:
        conn.execute("UPDATE pieces SET card_json = ?", (json.dumps(card),))
    remote = remote_settings("stt", {})
    assert remote is None or remote.headers() == {}


# ── forgetting, and who can read it ──────────────────────────────────────────

def test_forgetting_the_piece_removes_its_key(world, keyed_box, env_file):
    from prometheus.infra.voice_remote import remote_settings

    _, port = keyed_box
    piece = found(world, port)
    approve(world, piece["id"], secret=KEY)
    assert world.as_("global", "DELETE", f"/api/pieces/{piece['id']}").status_code == 200
    assert KEY not in env_file.read_text() and "PROMETHEUS_PIECE_" not in env_file.read_text()
    assert remote_settings("stt", {}) is None


def test_only_the_owner_can_give_a_key(world, keyed_box, env_file):
    _, port = keyed_box
    piece = found(world, port)
    response = world.as_("scoped", "POST", f"/api/pieces/{piece['id']}/approve", json={"secret": KEY})
    assert response.status_code == 403
    assert not env_file.exists() or KEY not in env_file.read_text()


def test_the_agents_file_tools_cannot_read_the_env_file(tmp_path, monkeypatch, keyed_box):
    """Through the real dispatch path: the model's read_file of an env file at the floor's shape
    (``<home>/.config/<app>/env``) is refused before the prompt is asked, and the key never reaches the result."""
    import asyncio

    from prometheus.__main__ import create_tool_registry
    from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
    from prometheus.infra import pieces
    from prometheus.permissions.checker import SecurityGate

    env = tmp_path / "home" / ".config" / "prometheus" / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env))
    w = World(tmp_path, recording=False)
    monkeypatch.setattr(pieces, "devices_db_path", lambda: w.path)
    _, port = keyed_box
    assert approve(w, found(w, port)["id"], secret=KEY).status_code == 200
    assert KEY in env.read_text()

    asked: list[str] = []

    async def yes(tool_name, reason):
        asked.append(reason)
        return True

    gate = SecurityGate(workspace_root=[str(tmp_path)])
    ctx = LoopContext(provider=None, model="t", system_prompt="", max_tokens=512,
                      tool_registry=create_tool_registry({}, gate), permission_checker=gate, permission_prompt=yes)
    result = asyncio.run(_execute_tool_call(ctx, "read_file", "t1", {"path": str(env)}))
    assert result.is_error
    assert KEY not in result.content
    assert not asked


def test_the_agents_shell_never_sees_the_key_variable(world, keyed_box, env_file, monkeypatch):
    """The daemon loads the env file into its environment at start; the shell tool's environment drops the key."""
    from prometheus.config.env_file import parse_env_file
    from prometheus.security.env_scrub import scrubbed_env

    _, port = keyed_box
    approve(world, found(world, port)["id"], secret=KEY)
    [(name, value)] = [(k, v) for k, v in parse_env_file(env_file).items() if k.startswith("PROMETHEUS_PIECE_")]
    monkeypatch.setenv(name, value)                        # as after a restart
    child = scrubbed_env()
    assert name not in child
    assert KEY not in child.values()
