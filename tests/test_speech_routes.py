"""The speech routes (docs/design/speech-route.md, sections 2 to 4, 7 and 8), on the real app.

Real throughout: ``create_app``, the bearer middleware and its device tokens (the global token, an owner device, two
chat-only devices), the session path guard, ``route_access``, the reply holder fed by a real ``WebSocketBridge``. The
one stand-in is the voice box: the fake built from its contract (``tests/support/fake_voice_box.py``). Requests go
through ``httpx.ASGITransport`` on the test's own event loop, so the bridge's frames and the routes share it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.speech import voice as voice_mod  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.bridge_frames import Bridge, text  # noqa: E402
from tests.support.fake_voice_box import FakeVoiceBox  # noqa: E402

GLOBAL = "speech-global-" + secrets.token_hex(8)
KEY_ENV = "PROMETHEUS_SPEECH_ROUTES_KEY"
KEY = "speech-routes-key-" + secrets.token_hex(6)
BOX = "http://127.0.0.1:8003"
OGG = b"OggS" + b"\x00" * 60
ROUTES = {
    ("GET", "/api/speech"),
    ("POST", "/api/speech/transcribe"),
    ("POST", "/api/speech/speak"),
    ("GET", "/api/sessions/{session_id}/speech"),
}


def _config(*kinds: str) -> dict:
    cfg: dict = {"web": {"api_token": GLOBAL}}
    for kind in kinds:
        cfg[kind] = {"url": BOX, "api_key_env": KEY_ENV}
    return cfg


class World:
    def __init__(self, tmp_path: Path, monkeypatch, *kinds: str) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.app = create_app(_config(*kinds), session_mgr=SessionManager(), device_store=self.devices)
        self.speech = self.app.state.speech
        self.bridge = Bridge(monkeypatch, self.speech.tap.observe)
        self.app.state.ws_bridge = self.bridge.bridge
        self.a = self.devices.mint("device-a", "ios")
        self.b = self.devices.mint("device-b", "ios")
        self.owner = self.devices.mint_owner("owner-mac", "macos", by="test")
        self.extra = [self.devices.mint(f"device-{i}", "ios") for i in range(4)]

    def token(self, who: str) -> str:
        if who == "op":
            return GLOBAL
        if who.startswith("x"):
            return self.extra[int(who[1:])]["token"]
        return {"a": self.a, "b": self.b, "owner": self.owner}[who]["token"]

    def client(self, who: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {self.token(who)}"}, timeout=10)

    async def session(self, who: str) -> str:
        async with self.client(who) as c:
            return (await c.post("/api/sessions")).json()["session_id"]


@pytest.fixture(autouse=True)
def _key(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    Path(tmp_path / "prom-env").write_text(f"{KEY_ENV}={KEY}\n")


@pytest.fixture
def box(monkeypatch):
    fake = FakeVoiceBox(key=KEY, transcript="turn on the lights")
    real = httpx.AsyncClient
    monkeypatch.setattr(voice_mod, "_client",
                        lambda timeout: real(transport=httpx.MockTransport(fake.handler), timeout=timeout))
    return fake


@pytest.fixture
def world(tmp_path, monkeypatch, box):
    return World(tmp_path, monkeypatch, "stt", "tts")


def _lines(resp: httpx.Response) -> list[dict]:
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in resp.text.splitlines() if line]


# --------------------------------------------------------------------------- access


def test_the_routes_are_open_to_chat_only_devices():
    assert ROUTES <= route_access.SCOPED_ALLOWED


@pytest.mark.asyncio
async def test_no_token_is_refused(world):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=world.app), base_url="http://test") as c:
        assert (await c.get("/api/speech")).status_code == 401
        assert (await c.post("/api/speech/speak", json={"text": "Hi."})).status_code == 401


# --------------------------------------------------------------------------- status


@pytest.mark.asyncio
@pytest.mark.parametrize("who", ["op", "owner", "a"])
async def test_status(world, who):
    async with world.client(who) as c:
        r = await c.get("/api/speech")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert body["stt"] == {"configured": True, "state": "ok"}
    assert body["tts"] == {"configured": True, "state": "ok"}
    assert body["audio"] == {"mime": "audio/wav"}
    assert body["limits"] == {"upload_bytes": 2 * 1024 * 1024, "phrase_chars": 200, "spoken_chars": 5000,
                              "speak_chars": 20000, "streams": 4}
    assert "127.0.0.1" not in r.text and KEY not in r.text and KEY_ENV not in r.text


@pytest.mark.asyncio
async def test_status_when_nothing_is_configured(tmp_path, monkeypatch, box):
    w = World(tmp_path, monkeypatch)
    async with w.client("a") as c:
        body = (await c.get("/api/speech")).json()
    assert body["stt"] == {"configured": False} and body["tts"] == {"configured": False}


# --------------------------------------------------------------------------- transcribe


@pytest.mark.asyncio
async def test_transcribe(world, box):
    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe?language=en", content=OGG, headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 200 and r.json() == {"text": "turn on the lights"}
    sent = box.requests[-1]
    assert sent.url.path == "/v1/audio/transcriptions" and b"OggS" in sent.content


@pytest.mark.asyncio
async def test_a_transcript_never_starts_with_a_slash(world, box):
    box.transcript = "  / /reset everything"
    async with world.client("owner") as c:
        r = await c.post("/api/speech/transcribe", content=OGG, headers={"Content-Type": "audio/ogg"})
    assert r.json() == {"text": "reset everything"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ctype", [None, "application/octet-stream", "image/png", "text/plain"])
async def test_transcribe_refuses_a_type_not_on_the_list(world, ctype):
    headers = {"Content-Type": ctype} if ctype else {}
    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=OGG, headers=headers)
    assert r.status_code == 415 and r.json()["error"] == "type_not_allowed"


@pytest.mark.asyncio
async def test_transcribe_refuses_bytes_that_are_not_the_declared_type(world, box):
    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=b"not audio at all" * 4,
                         headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 400 and r.json()["error"] == "bad_audio"
    assert not box.requests


@pytest.mark.asyncio
async def test_transcribe_refuses_an_empty_body(world):
    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=b"", headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 400 and r.json()["error"] == "bad_audio"


@pytest.mark.asyncio
async def test_a_chunked_upload_over_the_cap_is_refused_while_reading(world, box):
    sent = {"bytes": 0}

    async def chunks():
        yield OGG
        for _ in range(64):
            sent["bytes"] += 64 * 1024
            yield b"\x00" * (64 * 1024)  # 4 MB in all, with no Content-Length

    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=chunks(), headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 413 and r.json()["error"] == "too_large"
    assert not box.requests
    assert sent["bytes"] < 4 * 1024 * 1024  # reading stopped at the cap, not at the end


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["e", "english-us", "en_US", "e1"])
async def test_transcribe_refuses_a_bad_language(world, language):
    async with world.client("a") as c:
        r = await c.post(f"/api/speech/transcribe?language={language}", content=OGG,
                         headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 400 and r.json()["error"] == "bad_language"


@pytest.mark.asyncio
async def test_transcribe_when_the_box_is_down(world, box):
    box.down = True
    async with world.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=OGG, headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 503
    assert r.json() == {"error": "speech_unavailable", "voice": "unreachable",
                        "message": "The voice server isn't answering."}


@pytest.mark.asyncio
async def test_transcribe_when_nothing_is_configured(tmp_path, monkeypatch, box):
    w = World(tmp_path, monkeypatch, "tts")
    async with w.client("a") as c:
        r = await c.post("/api/speech/transcribe", content=OGG, headers={"Content-Type": "audio/ogg"})
    assert r.status_code == 503 and r.json()["voice"] == "not_configured"


# --------------------------------------------------------------------------- speak


@pytest.mark.asyncio
async def test_speak(world, box):
    async with world.client("a") as c:
        r = await c.post("/api/speech/speak", json={"text": "Hello there. Bye now."})
    lines = _lines(r)
    assert [line["type"] for line in lines] == ["start", "phrase", "phrase", "done"]
    assert base64.b64decode(lines[1]["audio"])[:4] == b"RIFF"
    assert lines[-1] == {"type": "done", "reason": "complete"}
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize(("body", "status", "error"), [
    ({}, 400, "text_required"), ({"text": ""}, 400, "text_required"), ({"text": 5}, 400, "text_required"),
    ({"text": "x " * 10_001}, 413, "too_long"),
])
async def test_speak_refuses_bad_text(world, body, status, error):
    async with world.client("a") as c:
        r = await c.post("/api/speech/speak", json=body)
    assert r.status_code == status and r.json()["error"] == error


@pytest.mark.asyncio
async def test_speak_when_tts_is_not_configured(tmp_path, monkeypatch, box):
    w = World(tmp_path, monkeypatch, "stt")
    async with w.client("a") as c:
        r = await c.post("/api/speech/speak", json={"text": "Hi."})
    assert r.status_code == 503 and r.json()["voice"] == "not_configured"
    assert r.json()["message"] == "No voice server is set up."


@pytest.mark.asyncio
async def test_the_box_failing_mid_stream(world, box):
    original = box.handler
    calls = {"n": 0}

    def flaky(request):
        if request.url.path == "/v1/audio/speech":
            calls["n"] += 1
            if calls["n"] == 2:
                box.down = True
        return original(request)

    box.handler = flaky
    async with world.client("a") as c:
        r = await c.post("/api/speech/speak", json={"text": "One is fine. Two fails."})
    done = _lines(r)[-1]
    assert done["reason"] == "unavailable" and done["voice"] == "unreachable"
    assert done["resume_at"] == len("One is fine. ")


# --------------------------------------------------------------------------- a reply


@pytest.mark.asyncio
async def test_a_device_hears_its_own_reply(world):
    sid = await world.session("a")
    async with world.client("a") as c:
        req = asyncio.ensure_future(c.get(f"/api/sessions/{sid}/speech", params={"wait": 5}))
        await asyncio.sleep(0.05)
        await world.bridge.start(sid, [text("First part. "), text("Second part.")]).task
        r = await req
    lines = _lines(r)
    assert [line["type"] for line in lines][:2] == ["start", "reply"]
    assert [line["text"] for line in lines if line["type"] == "phrase"] == ["First part.", "Second part."]
    assert lines[-1]["reason"] == "complete"


@pytest.mark.asyncio
async def test_a_reply_named_by_its_message_id(world):
    sid = await world.session("a")
    await world.bridge.start(sid, [text("Already said.")]).task
    mid = next(f["payload"]["message_id"] for f in world.bridge.frames if f["type"] == "chat_delta")
    async with world.client("a") as c:
        r = await c.get(f"/api/sessions/{sid}/speech", params={"message_id": mid})
    assert [line["text"] for line in _lines(r) if line["type"] == "phrase"] == ["Already said."]


@pytest.mark.asyncio
async def test_no_reply(world):
    sid = await world.session("a")
    async with world.client("a") as c:
        r = await c.get(f"/api/sessions/{sid}/speech", params={"wait": 1})
    assert _lines(r)[-1] == {"type": "done", "reason": "no_reply"}


@pytest.mark.asyncio
async def test_another_devices_session_is_not_there(world):
    sid = await world.session("b")
    await world.bridge.start(sid, [text("B's private words.")]).task
    mid = next(f["payload"]["message_id"] for f in world.bridge.frames if f["type"] == "chat_delta")
    async with world.client("a") as c:
        r = await c.get(f"/api/sessions/{sid}/speech", params={"message_id": mid})
    assert r.status_code == 404 and "private" not in r.text


@pytest.mark.asyncio
async def test_an_unclaimed_session_is_not_there_for_a_device(world):
    async with world.client("a") as c:
        r = await c.get("/api/sessions/desktop:never-made/speech", params={"wait": 1})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_the_owner_hears_any_session(world):
    sid = await world.session("b")
    await world.bridge.start(sid, [text("Owner may hear.")]).task
    mid = next(f["payload"]["message_id"] for f in world.bridge.frames if f["type"] == "chat_delta")
    async with world.client("owner") as c:
        r = await c.get(f"/api/sessions/{sid}/speech", params={"message_id": mid})
    assert r.status_code == 200 and _lines(r)[-1]["reason"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"wait": 0}, {"wait": 121}, {"wait": "soon"}, {"message_id": "x" * 200}])
async def test_a_bad_query_is_refused(world, params):
    sid = await world.session("a")
    async with world.client("a") as c:
        r = await c.get(f"/api/sessions/{sid}/speech", params=params)
    assert r.status_code == 400 and r.json()["error"] == "bad_query"


# --------------------------------------------------------------------------- limits


@pytest.mark.asyncio
async def test_two_streams_per_caller(world):
    sids = [await world.session("a") for _ in range(3)]
    async with world.client("a") as c:
        held = [asyncio.ensure_future(c.get(f"/api/sessions/{s}/speech", params={"wait": 1})) for s in sids[:2]]
        await asyncio.sleep(0.1)
        third = await c.get(f"/api/sessions/{sids[2]}/speech", params={"wait": 1})
        await asyncio.gather(*held)
    assert third.status_code == 429 and third.json()["error"] == "speech_busy"


@pytest.mark.asyncio
async def test_four_device_streams_and_the_owner_is_never_refused(world):
    held = []
    clients = []
    for i in range(4):
        sid = await world.session(f"x{i}")
        c = world.client(f"x{i}")
        clients.append(c)
        held.append(asyncio.ensure_future(c.get(f"/api/sessions/{sid}/speech", params={"wait": 1})))
    await asyncio.sleep(0.1)
    fifth_sid = await world.session("a")
    async with world.client("a") as c:
        fifth = await c.get(f"/api/sessions/{fifth_sid}/speech", params={"wait": 1})
    async with world.client("owner") as c:
        owner = await c.post("/api/speech/speak", json={"text": "Owner speaks."})
    await asyncio.gather(*held)
    for c in clients:
        await c.aclose()
    assert fifth.status_code == 429
    assert owner.status_code == 200 and _lines(owner)[-1]["reason"] == "complete"


@pytest.mark.asyncio
async def test_a_newer_stream_for_the_same_session_replaces_the_older(world):
    sid = await world.session("a")
    async with world.client("a") as c:
        first = asyncio.ensure_future(c.get(f"/api/sessions/{sid}/speech", params={"wait": 3}))
        await asyncio.sleep(0.1)
        second = asyncio.ensure_future(c.get(f"/api/sessions/{sid}/speech", params={"wait": 1}))
        r1 = await first
        await second
    assert _lines(r1)[-1] == {"type": "done", "reason": "replaced"}


@pytest.mark.asyncio
async def test_a_revoked_device_stream_ends(world):
    sid = await world.session("a")
    async with world.client("a") as c:
        req = asyncio.ensure_future(c.get(f"/api/sessions/{sid}/speech", params={"wait": 5}))
        await asyncio.sleep(0.1)
        world.devices.revoke(world.a["id"])
        r = await asyncio.wait_for(req, 3)
    assert _lines(r)[-1] == {"type": "done", "reason": "revoked"}


# --------------------------------------------------------------------------- the bridge hooks


def test_the_capability_follows_the_configuration(tmp_path, monkeypatch):
    from prometheus.web.ws_server import WebSocketBridge

    with_box = WebSocketBridge(loop_context=object(), config=_config("tts"))
    without = WebSocketBridge(loop_context=object(), config=_config())
    assert "speech" in with_box._server_capabilities()
    assert "speech" not in without._server_capabilities()


@pytest.mark.asyncio
async def test_observers_hear_every_frame_even_with_no_client_and_never_break_it():
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge(loop_context=object())
    heard: list = []

    def broken(frame):
        raise RuntimeError("observer bug")

    bridge.frame_observers.append(broken)
    bridge.frame_observers.append(heard.append)
    await bridge.broadcast({"type": "chat_delta", "payload": {"session_id": "s", "content": "x"}})
    assert heard and heard[0]["type"] == "chat_delta"


def test_attach_registers_the_reply_holder_on_the_bridge(tmp_path, monkeypatch, box):
    from prometheus.web.speech_routes import attach_bridge
    from prometheus.web.ws_server import WebSocketBridge

    w = World(tmp_path, monkeypatch, "tts")
    bridge = WebSocketBridge(loop_context=object())
    attach_bridge(w.app, bridge)
    assert w.speech.tap.observe in bridge.frame_observers


# --------------------------------------------------------------------------- review 1 of #927


@pytest.mark.asyncio
async def test_a_chunked_speak_body_is_capped(world, box):
    async def chunks():
        yield b'{"text": "'
        for _ in range(64):
            yield b"a" * (64 * 1024)
        yield b'"}'

    async with world.client("a") as c:
        r = await c.post("/api/speech/speak", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["error"] == "too_long"
    assert not box.spoken


@pytest.mark.asyncio
async def test_deeply_nested_json_is_a_400(world):
    async with world.client("a") as c:
        r = await c.post("/api/speech/speak", content=b"[" * 50_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "text_required"


@pytest.mark.asyncio
async def test_transcriptions_count_against_the_callers_slots(world, box, monkeypatch):
    gate = asyncio.Event()
    original = box.handler

    async def slow(request):
        if request.url.path == "/v1/audio/transcriptions":
            await gate.wait()
        return original(request)

    real = httpx.AsyncClient
    monkeypatch.setattr(voice_mod, "_client",
                        lambda timeout: real(transport=httpx.MockTransport(slow), timeout=timeout))
    async with world.client("a") as c:
        held = [asyncio.ensure_future(c.post("/api/speech/transcribe", content=OGG,
                                             headers={"Content-Type": "audio/ogg"})) for _ in range(2)]
        await asyncio.sleep(0.1)
        third = await c.post("/api/speech/transcribe", content=OGG, headers={"Content-Type": "audio/ogg"})
        gate.set()
        done = await asyncio.gather(*held)
    assert third.status_code == 429
    assert all(r.status_code == 200 for r in done)
    assert world.speech._streams == []  # every slot released


def test_a_bad_timeout_never_breaks_the_capability(tmp_path):
    from prometheus.web.ws_server import WebSocketBridge

    cfg = _config("tts")
    cfg["tts"]["timeout_seconds"] = "zz"
    assert "speech" not in WebSocketBridge(loop_context=object(), config=cfg)._server_capabilities()


@pytest.mark.asyncio
async def test_a_revocation_from_another_thread_ends_an_owner_stream(world):
    import threading

    owner_sid = await world.session("owner")
    async with world.client("owner") as c:
        req = asyncio.ensure_future(c.get(f"/api/sessions/{owner_sid}/speech", params={"wait": 5}))
        await asyncio.sleep(0.1)
        t = threading.Thread(target=world.devices.revoke, args=(world.owner["id"],))
        t.start()
        t.join()
        r = await asyncio.wait_for(req, 3)
    assert _lines(r)[-1] == {"type": "done", "reason": "revoked"}


@pytest.mark.asyncio
async def test_auth_off_is_open_like_every_route(tmp_path, monkeypatch, box):
    app = create_app({"tts": {"url": BOX, "api_key_env": KEY_ENV}}, session_mgr=SessionManager(),
                     device_store=DeviceStore(tmp_path / "devices.db"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/speech/speak", json={"text": "Open daemon."})
    assert r.status_code == 200 and _lines(r)[-1]["reason"] == "complete"
    assert app.state.speech._streams == []
