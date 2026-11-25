"""The voice event route (Phase 1): the envelope, the credential-derived origin, the loud failures.

Real app, real bearer middleware, real device tokens; the voice box is the
contract fake (tests/support/fake_voice_box.py) and the bridge records what
was dispatched. Nothing here waits on an edge: a ptt_start answers at once.
"""

from __future__ import annotations

import base64
import secrets
import struct
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.speech import voice as voice_mod  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.fake_voice_box import FakeVoiceBox  # noqa: E402

GLOBAL = "voice-event-global-" + secrets.token_hex(8)
KEY_ENV = "PROMETHEUS_VOICE_EVENT_KEY"
BOX = "http://127.0.0.1:8003"


def pcm16_b64(level: int = 5000, seconds: float = 0.3) -> str:
    import math
    n = int(16000 * seconds)
    samples = [int(level * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(n)]
    return base64.b64encode(struct.pack(f"<{n}h", *samples)).decode()


def silence_b64() -> str:
    return base64.b64encode(b"\x00\x00" * 4800).decode()


class World:
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.box = FakeVoiceBox(transcript="what is the weather")
        cfg = {"web": {"api_token": GLOBAL}, "stt": {"url": BOX, "api_key_env": KEY_ENV}}
        self.app = create_app(cfg, session_mgr=SessionManager(), device_store=self.devices)
        self.dispatched: list[dict] = []
        world = self

        class _Bridge:
            async def dispatch_user_message(self, session_id, message, **kw):
                world.dispatched.append({"session_id": session_id, "message": message, **kw})

        self.app.state.ws_bridge = _Bridge()
        self.box_device = self.devices.mint("kitchen-box", "other")
        self.ptt_device = self.devices.mint("sonim", "other")
        self.plain_device = self.devices.mint("plain-phone", "ios")
        monkeypatch.setattr(voice_mod, "_client", lambda timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(self.box.handler), timeout=timeout))

    def client(self, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"}, timeout=10)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    return World(tmp_path, monkeypatch)


async def test_wake_dispatches_with_credential_origin(world):
    async with world.client(world.box_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "wake", "node_id": "kitchen-1", "audio_base64": pcm16_b64(), "ts": 123.5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "dispatched"
    assert body["origin"] == "voice_box"          # unmarked scoped credential fails closed
    assert body["transcript"] == "what is the weather"
    assert body["node_ts"] == 123.5
    assert len(world.dispatched) == 1
    d = world.dispatched[0]
    assert d["origin"] == "voice_box"
    assert d["operator"] is False
    assert d["session_id"] == f"voice:{world.box_device['id']}:kitchen-1"
    assert "what is the weather" in d["message"]


async def test_marked_ptt_device_gets_its_origin(world):
    world.devices.set_origin(world.ptt_device["id"], "android_ptt", by="test")
    async with world.client(world.ptt_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "ptt_start", "node_id": "sonim-1"})
        assert r.status_code == 200 and r.json()["status"] == "listening"
        assert r.json()["origin"] == "android_ptt"
        r = await c.post("/api/speech/event", json={
            "event": "ptt_end", "node_id": "sonim-1", "audio_base64": pcm16_b64()})
    assert r.status_code == 200
    assert world.dispatched[0]["origin"] == "android_ptt"


async def test_ptt_end_without_start_is_409(world):
    async with world.client(world.ptt_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "ptt_end", "node_id": "sonim-1", "audio_base64": pcm16_b64()})
    assert r.status_code == 409 and r.json()["error"] == "no_capture"
    assert not world.dispatched


async def test_another_device_cannot_close_the_capture(world):
    async with world.client(world.ptt_device["token"]) as c:
        await c.post("/api/speech/event", json={"event": "ptt_start", "node_id": "n1"})
    async with world.client(world.box_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "ptt_end", "node_id": "n1", "audio_base64": pcm16_b64()})
    assert r.status_code == 403
    assert not world.dispatched


async def test_origin_in_the_body_is_refused_by_name(world):
    async with world.client(world.box_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "wake", "node_id": "k1", "audio_base64": pcm16_b64(), "origin": "terminal"})
    assert r.status_code == 400 and r.json()["error"] == "origin_not_a_field"
    assert not world.dispatched


async def test_silence_is_no_speech_with_levels(world):
    async with world.client(world.box_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "wake", "node_id": "k1", "audio_base64": silence_b64()})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_speech"
    assert body["level_rms"] == 0.0
    assert body["silence_floor_rms"] == 32.0
    assert not world.dispatched


async def test_bad_envelope_fields_are_named(world):
    async with world.client(world.box_device["token"]) as c:
        for body, kind in [
            ({"event": "shout", "node_id": "k1", "audio_base64": pcm16_b64()}, "bad_event"),
            ({"event": "wake", "node_id": "../etc", "audio_base64": pcm16_b64()}, "bad_node_id"),
            ({"event": "wake", "node_id": "k1"}, "audio_required"),
            ({"event": "wake", "node_id": "k1", "audio_base64": "!!!"}, "bad_audio"),
            ({"event": "ptt_start", "node_id": "k1", "audio_base64": pcm16_b64()}, "audio_on_ptt_start"),
            ({"event": "wake", "node_id": "k1", "audio_base64": pcm16_b64(), "ts": "noon"}, "bad_ts"),
        ]:
            r = await c.post("/api/speech/event", json=body)
            assert r.status_code == 400 and r.json()["error"] == kind, (body, r.text)


async def test_box_down_is_503_named(world):
    world.box.down = True
    async with world.client(world.box_device["token"]) as c:
        r = await c.post("/api/speech/event", json={
            "event": "wake", "node_id": "k1", "audio_base64": pcm16_b64()})
    assert r.status_code == 503 and r.json()["error"] == "speech_unavailable"
    assert not world.dispatched


async def test_the_origin_mark_is_operator_only(world):
    async with world.client(world.plain_device["token"]) as c:
        r = await c.put(f"/api/devices/{world.plain_device['id']}/origin", json={"origin": "android_ptt"})
    assert r.status_code == 403
    async with world.client(GLOBAL) as c:
        r = await c.put(f"/api/devices/{world.plain_device['id']}/origin", json={"origin": "android_ptt"})
        assert r.status_code == 200
        r = await c.put(f"/api/devices/{world.plain_device['id']}/origin", json={"origin": "root"})
        assert r.status_code == 400
        r = await c.put(f"/api/devices/{world.plain_device['id']}/origin", json={"origin": None})
        assert r.status_code == 200
    assert world.devices.device_origin(world.plain_device["id"]) is None


async def test_a_marked_devices_chat_turn_carries_the_origin(world):
    """The mark declares what the node IS: its typed chat turns run under the
    same held profile (the send_chat seam), not only its voice events."""
    world.devices.set_origin(world.ptt_device["id"], "android_ptt", by="test")
    seen: list[dict] = []

    class _Bridge:
        async def dispatch_user_message(self, session_id, message, **kw):
            seen.append(kw)

    world.app.state.ws_bridge = _Bridge()
    async with world.client(world.ptt_device["token"]) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
        r = await c.post("/api/chat/send", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 200
    assert seen and seen[0].get("origin") == "android_ptt"


async def test_an_unmarked_devices_chat_turn_is_unchanged(world):
    seen: list[dict] = []

    class _Bridge:
        async def dispatch_user_message(self, session_id, message, **kw):
            seen.append(kw)

    world.app.state.ws_bridge = _Bridge()
    async with world.client(world.plain_device["token"]) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
        r = await c.post("/api/chat/send", json={"session_id": sid, "message": "hello"})
    assert r.status_code == 200
    assert seen and "origin" not in seen[0]
