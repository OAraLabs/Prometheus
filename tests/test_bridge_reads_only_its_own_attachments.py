"""The bridge analyses an image only when it received the image itself, as an attachment it stored.

Before, ``WebSocketBridge._handle_send_message`` searched the whole text of every message for ``[Image: /path]``
and ran vision analysis on any file it named (GHSA-v65x-3634-7pcm). A paired scoped device could name any picture
the daemon's user can read and get its description back; an uploaded document written by someone else could name
one too, and so could a vision description that began with "/". The read ran in the daemon, outside the floors.

Now no text, from anyone, becomes a file read: an image is analysed only when it arrived as an attachment
(``chat_upload``) and the bridge stored it in the media cache, and the analysis refuses a path outside that cache.

These tests run through the real app and bridge, with spies on every way an image is read: the bridge's vision
call, the shared vision service, the image-block preparer, and Python's own ``open`` audit event for the file.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from tests.support.device_world import GLOBAL, World  # noqa: E402

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_OPENED: list[str] = []


def _audit(event: str, args: tuple) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        _OPENED.append(args[0].decode() if isinstance(args[0], bytes) else args[0])


sys.addaudithook(_audit)


@pytest.fixture
def w(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))
    world = World(tmp_path)
    world.secret = tmp_path / "private" / "passport.png"  # outside every upload directory
    world.secret.parent.mkdir()
    world.secret.write_bytes(_PNG)
    world.described: list[str] = []

    async def fake_vision(path, *a, **kw):
        world.described.append(str(path))
        return "a passport photo"

    import prometheus.gateway.media_services as media_services
    import prometheus.gateway.image_prep as image_prep

    monkeypatch.setattr(media_services, "describe_image", fake_vision)
    real_prepare = image_prep.prepare_image_block

    def spy_prepare(path):
        world.described.append(f"prepare:{path}")
        return real_prepare(path)

    monkeypatch.setattr(image_prep, "prepare_image_block", spy_prepare)
    _OPENED.clear()
    return world


def _touched_secret(w) -> list[str]:
    secret = str(w.secret)
    return [p for p in w.described if secret in p] + [p for p in _OPENED if secret in p]


async def _send(w, token: str, message: str) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=w.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"}) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
        r = await c.post("/api/chat/send", json={"session_id": sid, "message": message})
        assert r.status_code == 200, r.text
    for _ in range(20):
        await asyncio.sleep(0)


async def _upload(w, *, filename: str, mime: str, data: bytes, caption: str = "") -> None:
    """The owner's upload, through the bridge's own chat_upload handler."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=w.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {GLOBAL}"}) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
    await w.bridge._handle_file_upload(sid, filename, base64.b64encode(data).decode(), mime, caption)
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_document_naming_a_local_picture_reads_nothing(w):
    doc = f"Quarterly notes.\n[Image: {w.secret}]\nEnd of notes.".encode()
    await _upload(w, filename="notes.txt", mime="text/plain", data=doc, caption="summarise this")
    assert _touched_secret(w) == []


@pytest.mark.asyncio
async def test_a_scoped_device_cannot_name_a_picture(w):
    await _send(w, w.a["token"], f"what is in [Image: {w.secret}] ?")
    assert _touched_secret(w) == []


@pytest.mark.asyncio
async def test_the_owners_own_text_does_not_name_a_file_either(w):
    # By structure, not by who wrote it: text never becomes a file read.
    await _send(w, GLOBAL, f"[Image: {w.secret}]")
    assert _touched_secret(w) == []


@pytest.mark.asyncio
async def test_a_description_that_starts_with_a_path_is_not_read_again(w, monkeypatch):
    import prometheus.gateway.media_services as media_services

    async def vision_says_a_path(path, *a, **kw):
        w.described.append(str(path))
        return str(w.secret)

    monkeypatch.setattr(media_services, "describe_image", vision_says_a_path)
    monkeypatch.setattr(w.bridge, "_turn_supports_vision", lambda sid: False)
    await _upload(w, filename="photo.png", mime="image/png", data=_PNG)
    assert len(w.described) == 1 and str(w.secret) not in w.described[0]  # the upload itself, nothing more
    assert [p for p in _OPENED if str(w.secret) in p] == []


@pytest.mark.asyncio
async def test_a_picture_the_bridge_received_is_still_described(w, monkeypatch):
    # The control: a real attachment, stored in the media cache by the bridge, is analysed as before.
    from prometheus.gateway.media_cache import image_cache_dir

    monkeypatch.setattr(w.bridge, "_turn_supports_vision", lambda sid: False)
    await _upload(w, filename="photo.png", mime="image/png", data=_PNG, caption="what is this?")
    assert len(w.described) == 1
    assert Path(w.described[0]).resolve().is_relative_to(image_cache_dir().resolve())


@pytest.mark.asyncio
async def test_the_bridges_vision_call_refuses_a_path_outside_the_media_cache(w):
    assert await w.bridge._describe_image(str(w.secret)) is None
    assert _touched_secret(w) == []


# --------------------------------------------------------------------------- #
# Review round 4: the readers behind the bridge are confined too
# --------------------------------------------------------------------------- #


def test_a_stored_image_block_outside_the_cache_is_not_read(w):
    # A stored row's image block names its file. One planted with a path outside the cache must not be read back.
    from prometheus.engine.messages import ImageBlock

    block = ImageBlock(media_type="image/png", data="", source_path=str(w.secret), description="a photo")
    out = block.rehydrate()
    assert not out.data
    assert [p for p in _OPENED if str(w.secret) in p] == []


@pytest.mark.asyncio
async def test_the_shared_vision_service_refuses_a_path_outside_the_cache(tmp_path, monkeypatch):
    # Telegram and Discord call it directly; the guard can't live only in the bridge.
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))
    secret = tmp_path / "private" / "scan.png"
    secret.parent.mkdir()
    secret.write_bytes(_PNG)
    from prometheus.gateway import media_services
    from prometheus.tools.builtin import vision

    ran: list = []

    async def fake_execute(self, arguments, context):
        ran.append(arguments)
        raise AssertionError("vision ran on a path outside the cache")

    monkeypatch.setattr(vision.VisionTool, "execute", fake_execute)
    _OPENED.clear()
    assert await media_services.describe_image(str(secret)) is None
    assert ran == [] and [p for p in _OPENED if str(secret) in p] == []


# ---------------------------------------------------------------------------
# A path the gate's own floor denies (Beacon iOS sprint item, 2026-11-11)
# ---------------------------------------------------------------------------


@pytest.fixture
def denied(w, tmp_path, monkeypatch):
    """A picture at a path SecurityGate refuses to read, so the test shows the marker does not get around it."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    path = home / ".ssh" / "id_photo.png"
    path.write_bytes(_PNG)
    from prometheus.permissions.checker import SecurityGate

    decision = SecurityGate().evaluate("read_file", is_read_only=True, file_path=str(path))
    assert not decision.allowed, "the control failed: the gate allows this path, so the test proves nothing"
    w.denied = path
    _OPENED.clear()      # creating the file above is itself an open of it
    return w


@pytest.mark.asyncio
async def test_the_marker_does_not_read_a_path_the_gate_denies(denied):
    await _send(denied, GLOBAL, f"[Image: {denied.denied}]")
    assert [p for p in denied.described + _OPENED if str(denied.denied) in p] == []


@pytest.mark.asyncio
async def test_a_scoped_device_cannot_read_a_denied_path_through_the_marker(denied):
    await _send(denied, denied.a["token"], f"look at [Image: {denied.denied}] please")
    assert [p for p in denied.described + _OPENED if str(denied.denied) in p] == []
