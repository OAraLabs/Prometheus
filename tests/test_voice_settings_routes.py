"""The voice-settings routes (docs/design/voice-settings.md), on the real app.

Real throughout: ``create_app``, the bearer middleware and its device tokens (the global token, an owner
device, a chat-only device), ``route_access``, the comment-preserving config editor, the security audit. The
one stand-in is the voice box: the fake built from its contract (``tests/support/fake_voice_box.py``), with the
settings surface enabled. Requests go through ``httpx.ASGITransport`` on the test's own event loop.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path

import httpx
import pytest
import yaml

pytest.importorskip("fastapi")

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.speech import voice as voice_mod  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from prometheus.web import voice_settings as vs  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.fake_voice_box import FakeVoiceBox  # noqa: E402

GLOBAL = "voice-settings-global-" + secrets.token_hex(8)
KEY_ENV = "PROMETHEUS_VOICE_SETTINGS_KEY"
KEY = "voice-settings-key-" + secrets.token_hex(6)
BOX = "http://127.0.0.1:8015"

CONFIG_FILE = """\
# Prometheus config. Comments are the reasoning; they must survive an edit.
stt:
  url: "http://127.0.0.1:8015"   # the voice box
  api_key_env: "PROMETHEUS_VOICE_SETTINGS_KEY"
  model: "large-v3-turbo"        # the GPU tier
  timeout_seconds: 60

tts:
  url: "http://127.0.0.1:8015"
  api_key_env: "PROMETHEUS_VOICE_SETTINGS_KEY"
  model: "kokoro-v1.0"
  voice: "af_heart"              # the default voice
  timeout_seconds: 60

web:
  enabled: true
"""


def comment_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


class World:
    def __init__(self, tmp_path: Path, monkeypatch, *, surface: bool = True, text: str = CONFIG_FILE) -> None:
        self.path = tmp_path / "prometheus.yaml"
        self.path.write_text(text, encoding="utf-8")
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
        (tmp_path / "env").write_text(f"{KEY_ENV}={KEY}\n", encoding="utf-8")
        self.devices = DeviceStore(tmp_path / "devices.db")
        config = yaml.safe_load(text)
        config["web"]["api_token"] = GLOBAL
        self.app = create_app(config, session_mgr=SessionManager(), device_store=self.devices)
        self.app.state.config_path = str(self.path)
        self.owner = self.devices.mint_owner("owner-mac", "macos", by="test")
        self.scoped = self.devices.mint("approved phone", "ios")
        self.box = FakeVoiceBox(key=KEY)
        self.box.settings_surface = surface
        real = httpx.AsyncClient
        monkeypatch.setattr(voice_mod, "_client",
                            lambda timeout: real(transport=httpx.MockTransport(self.box.handler), timeout=timeout))
        monkeypatch.setattr(vs, "_client",
                            lambda timeout: real(transport=httpx.MockTransport(self.box.handler), timeout=timeout))

    def client(self, who: str = "op") -> httpx.AsyncClient:
        token = {"op": GLOBAL, "owner": self.owner["token"], "scoped": self.scoped["token"]}[who]
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"}, timeout=10)

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def running(self) -> dict:
        return self.app.state.config


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


# ── access ───────────────────────────────────────────────────────────────────

def test_the_routes_are_classified_operator_only():
    assert {("GET", "/api/speech/settings"), ("PUT", "/api/speech/settings")} <= route_access.OPERATOR_ONLY


@pytest.mark.asyncio
async def test_a_scoped_device_is_refused_on_get_and_put(world):
    async with world.client("scoped") as c:
        for method in ("GET", "PUT"):
            r = await c.request(method, "/api/speech/settings", json={"voice": "af_bella"})
            assert r.status_code == 403 and r.json()["error"] == "operator_only", method


@pytest.mark.asyncio
async def test_the_owner_device_may_read_and_change(world):
    async with world.client("owner") as c:
        assert (await c.get("/api/speech/settings")).status_code == 200
        r = await c.put("/api/speech/settings", json={"voice": "af_bella"})
        assert r.status_code == 200


# ── GET ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_reports_config_options_and_health(world):
    async with world.client() as c:
        body = (await c.get("/api/speech/settings")).json()
    assert body["version"] == 1
    assert body["stt"]["model"] == "large-v3-turbo" and body["stt"]["tier"] == "gpu"
    assert body["stt"]["configured"] is True and body["stt"]["health"]["state"] == "ok"
    assert body["tts"]["voice"] == "af_heart" and body["tts"]["engine"] == "kokoro"
    assert body["tts"]["speed"] == 1.0
    assert body["box"]["address"] == BOX and body["box"]["discovery"] == "live"
    assert body["options"]["voices"] == ["af_heart", "af_bella", "am_michael"]
    assert {m["id"] for m in body["options"]["stt_models"]} == {"base", "small", "large-v3-turbo"}
    assert body["options"]["speed"] == {"min": 0.5, "max": 2.0}
    assert body["vad"]["sensitivity"] == 0.5 and body["vad"]["health"]["state"] == "ok"
    assert body["silence_gate"]["threshold_rms"] == 32.0
    assert body["applied"] == "live" and body["pending"] is False


@pytest.mark.asyncio
async def test_get_reports_the_key_env_and_presence_never_the_value(world):
    async with world.client() as c:
        r = await c.get("/api/speech/settings")
    assert r.json()["stt"]["key"] == {"env": KEY_ENV, "present": True}
    assert KEY not in r.text


@pytest.mark.asyncio
async def test_get_on_an_older_box_says_discovery_unavailable(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, surface=False)
    async with w.client() as c:
        body = (await c.get("/api/speech/settings")).json()
    assert body["box"]["discovery"] == "unavailable"
    assert body["options"] is None
    assert any(w["code"] == "discovery_unavailable" for w in body["warnings"])
    # the config itself is still reported
    assert body["tts"]["voice"] == "af_heart"


@pytest.mark.asyncio
async def test_get_with_no_box_says_not_configured(tmp_path, monkeypatch):
    text = "web:\n  enabled: true\n"
    w = World(tmp_path, monkeypatch, text=text)
    async with w.client() as c:
        body = (await c.get("/api/speech/settings")).json()
    assert body["box"] is None and body["options"] is None
    assert body["stt"]["configured"] is False
    assert any(x["code"] == "not_configured" for x in body["warnings"])


# ── PUT: live apply ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_put_voice_saves_applies_live_and_pushes(world):
    before = comment_count(world.text)
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={"voice": "af_bella", "speed": 1.2})
        assert r.status_code == 200
        body = r.json()
    assert body["changed"] == ["speed", "voice"]
    assert body["tts"]["voice"] == "af_bella" and body["tts"]["speed"] == 1.2
    assert body["applied"] == "live" and body["pending"] is False
    # saved to the file, comments intact
    saved = yaml.safe_load(world.text)
    assert saved["tts"]["voice"] == "af_bella" and saved["tts"]["speed"] == 1.2
    assert comment_count(world.text) == before
    # applied to the running config the speech route reads per call
    assert world.running["tts"]["voice"] == "af_bella"
    # pushed to the box
    assert world.box.settings_pushes[-1]["voice"] == "af_bella"
    assert world.box.settings_pushes[-1]["speed"] == 1.2


@pytest.mark.asyncio
async def test_put_gate_and_vad_push(world):
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={"silence_gate_rms": 64.0, "vad_sensitivity": 0.3})
    assert r.status_code == 200
    push = world.box.settings_pushes[-1]
    assert push["silence_gate_rms"] == 64.0 and push["vad_sensitivity"] == 0.3
    assert yaml.safe_load(world.text)["stt"]["silence_gate_rms"] == 64.0


@pytest.mark.asyncio
async def test_put_engine_switch(world):
    world.box.engines[1]["available"] = True  # chatterbox now running
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={"tts_engine": "chatterbox"})
    assert r.status_code == 200
    assert world.box.settings_pushes[-1]["engine"] == "chatterbox"
    assert yaml.safe_load(world.text)["tts"]["provider"] == "chatterbox"


# ── PUT: the address waits for a restart ─────────────────────────────────────

@pytest.mark.asyncio
async def test_put_address_saves_but_waits_for_restart(world):
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={"box_address": "http://127.0.0.1:9015"})
        body = r.json()
    assert r.status_code == 200
    assert body["pending"] is True and body["applied"] == "on_restart"
    saved = yaml.safe_load(world.text)
    assert saved["stt"]["url"] == "http://127.0.0.1:9015"
    assert saved["tts"]["url"] == "http://127.0.0.1:9015"
    # the running config is NOT moved for the address (the health record is bound to the old box)
    assert world.running["stt"]["url"] == BOX
    # the address is never pushed to the box
    assert all("url" not in p and "box_address" not in p for p in world.box.settings_pushes)


# ── PUT: refusals write nothing ──────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("body,code,status", [
    ({"voice": "not_a_voice"}, "unknown_voice", 400),
    ({"speed": 3.0}, "bad_speed", 400),
    ({"speed": "fast"}, "invalid_request", 400),
    ({"vad_sensitivity": 0.99}, "bad_sensitivity", 400),
    ({"silence_gate_rms": -1}, "bad_threshold", 400),
    ({"tts_engine": "nope"}, "unsupported_engine", 400),
    ({"tts_engine": "system"}, "unsupported_engine", 400),
    ({"stt_model": "base"}, "tier_change", 409),
    ({"stt_model": "not_a_model"}, "unknown_model", 400),
    ({"box_address": "ftp://x"}, "bad_address", 400),
    ({"box_address": "http://user:pass@host"}, "bad_address", 400),
    ({"volume": 11}, "invalid_request", 400),
    ({"api_key": "sk-secret"}, "invalid_request", 400),
    ({"api_key_env": "X"}, "invalid_request", 400),
])
async def test_put_refusals(world, body, code, status):
    before = world.text
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json=body)
    assert r.status_code == status, (body, r.text)
    assert r.json()["error"] == code
    assert world.text == before  # nothing written
    assert world.box.settings_pushes == []  # nothing pushed


@pytest.mark.asyncio
async def test_put_empty_body_is_refused(world):
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={})
    assert r.status_code == 400 and r.json()["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_put_on_an_older_box_skips_membership_but_checks_range(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch, surface=False)
    async with w.client() as c:
        # a voice the box cannot confirm is accepted (range/format checks still run)
        r = await c.put("/api/speech/settings", json={"voice": "some_voice"})
        assert r.status_code == 200
        # but an out-of-range speed is still refused
        r = await c.put("/api/speech/settings", json={"speed": 9.0})
        assert r.status_code == 400 and r.json()["error"] == "bad_speed"


@pytest.mark.asyncio
async def test_put_push_failure_is_announced_not_swallowed(world):
    world.box.status["/v1/settings"] = (500, b'{"error":"boom"}')
    async with world.client() as c:
        r = await c.put("/api/speech/settings", json={"voice": "af_bella"})
        body = r.json()
    assert r.status_code == 200
    assert body["tts"]["voice"] == "af_bella"  # the daemon-side save stands
    assert any(w["code"] == "push_failed" for w in body["warnings"])


@pytest.mark.asyncio
async def test_put_with_no_config_file_is_refused(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    w.app.state.config_path = None
    async with w.client() as c:
        r = await c.put("/api/speech/settings", json={"voice": "af_bella"})
    assert r.status_code == 409 and r.json()["error"] == "no_config_file"


# ── audit ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_accepted_change_writes_an_allow_audit_row(world, monkeypatch, tmp_path):
    rows: list[dict] = []

    class _Audit:
        def __init__(self, data_dir):
            pass

        def log(self, **kw):
            rows.append(kw)

    monkeypatch.setattr("prometheus.permissions.audit.AuditLogger", _Audit)
    async with world.client() as c:
        await c.put("/api/speech/settings", json={"voice": "af_bella"})
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "speech.settings"
    assert rows[0]["decision"].value == "allow"
    assert rows[0]["tool_input"]["fields"] == {"voice": "af_bella"}
    assert KEY not in json.dumps(rows[0], default=str)


@pytest.mark.asyncio
async def test_a_refused_change_writes_a_deny_audit_row(world, monkeypatch):
    rows: list[dict] = []

    class _Audit:
        def __init__(self, data_dir):
            pass

        def log(self, **kw):
            rows.append(kw)

    monkeypatch.setattr("prometheus.permissions.audit.AuditLogger", _Audit)
    async with world.client() as c:
        await c.put("/api/speech/settings", json={"voice": "not_a_voice"})
    assert len(rows) == 1 and rows[0]["decision"].value == "deny"