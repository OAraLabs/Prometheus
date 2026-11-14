"""``speech/stream.py``: a reply (or any text) as the NDJSON stream's lines (docs/design/speech-route.md, 2.5 and 5).

Against the fake voice box (``tests/support/fake_voice_box.py``) through the real ``Voice`` client, and, for a reply,
the real reply holder fed by a real bridge. Pinned: the line order (``start``, ``reply``, phrases, exactly one
``done``), phrases sent while the reply is still being written, a stopped turn, the voice box failing mid-stream
(``unavailable`` with ``resume_at``), a dropped round, keepalives, the idle timeout, the cap, a stop from outside, and
a client that goes away (synthesis stops).
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
import pytest

from prometheus.speech import voice as voice_mod
from prometheus.speech.phrases import TOO_LONG, PhraseCutter
from prometheus.speech.replies import ReplyTap
from prometheus.speech.stream import speak, text_source
from prometheus.speech.voice import Voice
from tests.support.bridge_frames import Bridge, text
from tests.support.fake_voice_box import FakeVoiceBox

KEY_ENV = "PROMETHEUS_SPEECH_TEST_KEY"
KEY = "speech-stream-key-" + "y" * 10
CFG = {"tts": {"url": "http://127.0.0.1:8003", "api_key_env": KEY_ENV}}
S = "desktop:speech-stream"


@pytest.fixture(autouse=True)
def _key(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    Path(tmp_path / "prom-env").write_text(f"{KEY_ENV}={KEY}\n")


class _GatedBox(FakeVoiceBox):
    """The fake box, answering each speech request only once ``gate`` is open (to order events in a test)."""

    def __init__(self) -> None:
        super().__init__(key=KEY)
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail_after: int | None = None

    async def ahandler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/audio/speech":
            await self.gate.wait()
            if self.fail_after is not None and len(self.spoken) >= self.fail_after:
                self.down = True
        return self.handler(request)


@pytest.fixture
def box(monkeypatch):
    fake = _GatedBox()
    real = httpx.AsyncClient
    monkeypatch.setattr(voice_mod, "_client",
                        lambda timeout: real(transport=httpx.MockTransport(fake.ahandler), timeout=timeout))
    return fake


@pytest.fixture
def voice():
    return Voice(lambda: CFG)


async def _collect(gen) -> list[dict]:
    return [line async for line in gen]


def _phrases(lines) -> list[str]:
    return [line["text"] for line in lines if line["type"] == "phrase"]


async def _queue_source(q: asyncio.Queue):
    while True:
        item = await q.get()
        yield item
        if item[0] == "end":
            return


@pytest.mark.asyncio
async def test_text_is_spoken_in_order(box, voice):
    lines = await _collect(speak(text_source("Hello there. Bye now."), voice))
    assert [line["type"] for line in lines] == ["start", "phrase", "phrase", "done"]
    assert lines[0] == {"type": "start", "version": 1, "mime": "audio/wav"}
    first, second = lines[1], lines[2]
    assert (first["index"], first["text"], first["start"], first["end"]) == (0, "Hello there.", 0, 12)
    assert (second["index"], second["text"]) == (1, "Bye now.")
    audio = base64.b64decode(first["audio"])
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    assert lines[-1] == {"type": "done", "reason": "complete"}
    assert box.spoken == ["Hello there.", "Bye now."]


@pytest.mark.asyncio
async def test_a_reply_from_a_real_bridge(monkeypatch, box, voice):
    tap = ReplyTap()
    bridge = Bridge(monkeypatch, tap.observe)
    finding = asyncio.ensure_future(tap.find(S, wait=2.0))
    await asyncio.sleep(0.01)
    await bridge.start(S, [text("First part. "), text("Second part.")]).task
    reply = await finding
    lines = await _collect(speak(reply.follow(), voice))
    types = [line["type"] for line in lines]
    assert types[:2] == ["start", "reply"] and types[-1] == "done"
    assert lines[1]["message_id"] == reply.message_id
    assert _phrases(lines) == ["First part.", "Second part."]
    assert lines[-1]["reason"] == "complete"


@pytest.mark.asyncio
async def test_a_phrase_is_sent_before_the_reply_ends(box, voice):
    q: asyncio.Queue = asyncio.Queue()
    gen = speak(_queue_source(q), voice)
    assert (await gen.__anext__())["type"] == "start"
    await q.put(("text", "Early sentence. And more"))
    line = await asyncio.wait_for(gen.__anext__(), 2)
    assert line["type"] == "phrase" and line["text"] == "Early sentence."
    await q.put(("end", "complete"))
    rest = [x async for x in gen]
    assert _phrases(rest) == ["And more"] and rest[-1]["reason"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["interrupted", "failed", "ended", "no_reply"])
async def test_a_turn_that_does_not_complete_ends_with_its_reason(box, voice, reason):
    async def source():
        yield ("text", "Some words without an end")
        yield ("end", reason)

    lines = await _collect(speak(source(), voice))
    assert lines[-1] == {"type": "done", "reason": reason}
    assert sum(line["type"] == "done" for line in lines) == 1
    assert "Some words without an end" not in _phrases(lines)  # not flushed: the turn did not complete


@pytest.mark.asyncio
async def test_the_box_failing_mid_stream_says_where_to_resume(box, voice):
    box.fail_after = 1
    source_text = "One is fine. Two fails. Three never."
    lines = await _collect(speak(text_source(source_text), voice))
    assert _phrases(lines) == ["One is fine."]
    done = lines[-1]
    assert done["type"] == "done" and done["reason"] == "unavailable"
    assert done["voice"] == "unreachable"
    assert done["message"] == "The voice server isn't answering."
    assert done["resume_at"] == source_text.index("Two")


@pytest.mark.asyncio
async def test_a_dropped_round_is_not_spoken(box, voice):
    box.gate.clear()
    q: asyncio.Queue = asyncio.Queue()
    gen = speak(_queue_source(q), voice)
    lines = [await gen.__anext__()]
    await q.put(("text", "Alpha one. Beta two. "))
    await asyncio.sleep(0.05)  # both cut; the first waits on the box
    await q.put(("discard", 11))
    await asyncio.sleep(0.02)
    box.gate.set()
    await q.put(("text", "Gamma. "))
    await q.put(("end", "complete"))
    lines += [x async for x in gen]
    assert {"type": "discard", "keep_chars": 11} in lines
    assert _phrases(lines) == ["Alpha one.", "Gamma."]
    gamma = next(line for line in lines if line.get("text") == "Gamma.")
    assert gamma["start"] == 11


@pytest.mark.asyncio
async def test_keepalives_then_the_idle_timeout(box, voice):
    async def silent():
        await asyncio.sleep(10)
        yield ("end", "complete")

    lines = await _collect(speak(silent(), voice, keepalive=0.03, idle_timeout=0.2))
    assert any(line["type"] == "keepalive" for line in lines)
    assert lines[-1] == {"type": "done", "reason": "timeout"}


@pytest.mark.asyncio
async def test_the_cap_ends_with_one_sentence(box, voice):
    lines = await _collect(speak(text_source("A first sentence here. A second one is too much. "), voice,
                                 cutter=PhraseCutter(spoken_cap=30)))
    assert _phrases(lines) == ["A first sentence here.", TOO_LONG]
    assert lines[-1] == {"type": "done", "reason": "complete", "truncated": True}


@pytest.mark.asyncio
async def test_a_stop_from_outside(box, voice):
    q: asyncio.Queue = asyncio.Queue()
    stop: asyncio.Future = asyncio.get_running_loop().create_future()
    gen = speak(_queue_source(q), voice, stop=stop)
    await gen.__anext__()
    stop.set_result("replaced")
    rest = [x async for x in gen]
    assert rest[-1] == {"type": "done", "reason": "replaced"}


@pytest.mark.asyncio
async def test_a_client_that_goes_away_stops_synthesis(box, voice):
    q: asyncio.Queue = asyncio.Queue()
    gen = speak(_queue_source(q), voice)
    await gen.__anext__()
    await q.put(("text", "One. "))
    assert (await asyncio.wait_for(gen.__anext__(), 2))["text"] == "One."
    await gen.aclose()
    spoken = len(box.spoken)
    await q.put(("text", "Two. Three. "))
    await asyncio.sleep(0.05)
    assert len(box.spoken) == spoken


# --------------------------------------------------------------------------- review of #926


@pytest.mark.asyncio
async def test_an_unexpected_voice_error_still_ends_the_stream():
    class Broken:
        async def synthesize(self, text):
            raise RuntimeError("a bug")

    lines = await asyncio.wait_for(_collect(speak(text_source("Hello there. Bye."), Broken(), keepalive=0.05)), 2)
    assert lines[-1] == {"type": "done", "reason": "unavailable", "voice": "failed",
                         "message": "The voice server sent an answer that could not be used.", "resume_at": 0}


@pytest.mark.asyncio
async def test_a_source_that_raises_still_ends_the_stream(voice, box):
    async def broken():
        yield ("text", "Some ")
        raise RuntimeError("source bug")

    lines = await asyncio.wait_for(_collect(speak(broken(), voice, keepalive=0.05)), 2)
    assert lines[-1] == {"type": "done", "reason": "ended"}


@pytest.mark.asyncio
async def test_a_cancelled_stop_ends_the_stream(box, voice):
    q: asyncio.Queue = asyncio.Queue()
    stop: asyncio.Future = asyncio.get_running_loop().create_future()
    gen = speak(_queue_source(q), voice, stop=stop)
    await gen.__anext__()
    stop.cancel()
    rest = [x async for x in gen]
    assert rest[-1] == {"type": "done", "reason": "ended"}
