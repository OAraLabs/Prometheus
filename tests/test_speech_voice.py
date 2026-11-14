"""``speech/voice.py``: the voice box as the speech route uses it (docs/design/speech-route.md, sections 6 and 7).

Against a fake voice box built from the box's own contract (``tests/support/fake_voice_box.py``):

* configured or not, from the settings Prometheus already has (``stt.url`` / ``tts.url``, or an approved piece);
* WAV bytes out and text back, under the key from the env file, which never reaches a log line;
* every failure becomes a ``VoiceUnavailable`` with its reason and the sentence for the person, and one log line;
* health: checked only when asked, kept 30 s, one check at a time, at most 3 s, and every real call counts;
* kept apart from the chat notices (``voice_remote._down`` never moves).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import pytest

from prometheus.speech import voice as voice_mod
from prometheus.speech.voice import SENTENCES, Voice, VoiceUnavailable
from tests.support.fake_voice_box import FakeVoiceBox

KEY_ENV = "PROMETHEUS_SPEECH_TEST_KEY"
KEY = "speech-test-key-" + "x" * 12
URL = "http://127.0.0.1:8003"


def _cfg(*kinds: str, url: str = URL) -> dict:
    return {kind: {"url": url, "api_key_env": KEY_ENV} for kind in kinds}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _key(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    for name in ("PROMETHEUS_STT_API_KEY", "PROMETHEUS_TTS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    Path(tmp_path / "prom-env").write_text(f"{KEY_ENV}={KEY}\n")


@pytest.fixture
def box(monkeypatch):
    fake = FakeVoiceBox(key=KEY)
    real = httpx.AsyncClient

    def _client(timeout):
        return real(transport=httpx.MockTransport(fake.handler), timeout=timeout)

    monkeypatch.setattr(voice_mod, "_client", _client)
    return fake


@pytest.fixture
def clock():
    return _Clock()


def _voice(config: dict, clock: _Clock) -> Voice:
    return Voice(lambda: config, clock=clock)


def _run(coro):
    return asyncio.run(coro)


def _unavailable(coro) -> VoiceUnavailable:
    with pytest.raises(VoiceUnavailable) as info:
        _run(coro)
    return info.value


class TestNotConfigured:
    def test_nothing_set_is_not_configured(self, box, clock):
        v = _voice({}, clock)
        assert not v.configured("tts") and not v.configured("stt")
        err = _unavailable(v.ready("tts"))
        assert err.reason == "not_configured"
        assert err.message == SENTENCES["not_configured"] == "No voice server is set up."
        assert _unavailable(v.synthesize("Hi.")).reason == "not_configured"
        assert box.requests == []

    def test_status_says_not_configured(self, box, clock):
        assert _run(_voice({}, clock).status()) == {"stt": {"configured": False}, "tts": {"configured": False}}

    def test_one_half_only(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        assert v.configured("tts") and not v.configured("stt")


class TestCalls:
    def test_synthesize_returns_the_boxes_wav(self, box, clock):
        audio = _run(_voice(_cfg("tts"), clock).synthesize("Hello there."))
        assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
        req = box.requests[-1]
        assert req.url.path == "/v1/audio/speech"
        assert req.headers["authorization"] == f"Bearer {KEY}"
        assert box.spoken == ["Hello there."]

    def test_synthesize_asks_for_wav_and_sends_the_configured_voice(self, box, clock):
        import json

        cfg = _cfg("tts")
        cfg["tts"]["voice"] = "warm"
        _run(_voice(cfg, clock).synthesize("Hi."))
        body = json.loads(box.requests[-1].content)
        assert body["response_format"] == "wav" and body["voice"] == "warm" and body["input"] == "Hi."

    def test_transcribe_returns_the_text(self, box, clock):
        box.transcript = "  turn on the lights  "
        text = _run(_voice(_cfg("stt"), clock).transcribe(b"OggS" + b"\x00" * 40, "audio/ogg", language="en"))
        assert text == "turn on the lights"
        req = box.requests[-1]
        assert req.url.path == "/v1/audio/transcriptions"
        assert b'name="file"' in req.content and b'filename="speech.ogg"' in req.content
        assert b'name="language"' in req.content and b"\r\n\r\nen\r\n" in req.content

    def test_no_language_sends_none(self, box, clock):
        _run(_voice(_cfg("stt"), clock).transcribe(b"RIFF....WAVEfmt ", "audio/wav"))
        assert b'name="language"' not in box.requests[-1].content


class TestFailures:
    @pytest.mark.parametrize(
        ("setup", "reason"),
        [
            (lambda b: setattr(b, "down", True), "unreachable"),
            (lambda b: b.status.update({"/v1/audio/speech": (500, b"boom")}), "unreachable"),
            (lambda b: setattr(b, "key", "another-key"), "refused"),
            (lambda b: b.status.update({"/v1/audio/speech": (403, b"no")}), "refused"),
            (lambda b: b.status.update({"/v1/audio/speech": (400, b"unknown voice")}), "rejected"),
            (lambda b: b.status.update({"/v1/audio/speech": (422, b"bad")}), "rejected"),
            (lambda b: b.status.update({"/v1/audio/speech": (200, b"not audio")}), "failed"),
            (lambda b: b.status.update({"/v1/audio/speech": (200, b"")}), "failed"),
        ],
    )
    def test_each_failure_has_its_reason_and_sentence(self, box, clock, caplog, setup, reason):
        setup(box)
        v = _voice(_cfg("tts"), clock)
        with caplog.at_level(logging.WARNING, logger="prometheus.speech.voice"):
            err = _unavailable(v.synthesize("Hi."))
        assert err.reason == reason
        assert err.message == SENTENCES[reason]
        records = [r for r in caplog.records if r.name == "prometheus.speech.voice"]
        assert len(records) == 1
        assert records[0].levelno == (logging.ERROR if reason in ("refused", "rejected") else logging.WARNING)
        assert "127.0.0.1:8003" in records[0].getMessage()
        assert KEY not in caplog.text

    def test_transcription_without_text_failed(self, box, clock):
        box.status["/v1/audio/transcriptions"] = (200, b'{"words": []}')
        assert _unavailable(_voice(_cfg("stt"), clock).transcribe(b"x", "audio/ogg")).reason == "failed"

    def test_a_large_upload_the_box_refuses_is_rejected(self, box, clock):
        box.status["/v1/audio/transcriptions"] = (413, b"too large")
        assert _unavailable(_voice(_cfg("stt"), clock).transcribe(b"x", "audio/ogg")).reason == "rejected"

    def test_the_chat_notices_never_move(self, box, clock, monkeypatch):
        from prometheus.infra import voice_remote

        monkeypatch.setattr(voice_remote, "_down", set())
        box.down = True
        _unavailable(_voice(_cfg("tts", "stt"), clock).synthesize("Hi."))
        assert voice_remote._down == set()


class TestHealth:
    def test_ready_when_up_makes_no_request(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        _run(v.ready("tts"))
        assert box.requests == []

    def test_a_failed_call_then_ready_checks_the_box_first(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.down = True
        _unavailable(v.synthesize("Hi."))
        box.down = False
        box.requests.clear()
        _run(v.ready("tts"))  # one failed call does not refuse: the box is checked and is back
        assert [r.url.path for r in box.requests][0] == "/health"
        assert _run(v.status())["tts"] == {"configured": True, "state": "ok"}

    def test_a_failed_check_refuses_for_30_seconds_then_checks_again(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.down = True
        _unavailable(v.synthesize("Hi."))
        assert _unavailable(v.ready("tts")).reason == "unreachable"  # checked, still down
        box.requests.clear()
        clock.now += 29
        assert _unavailable(v.ready("tts")).reason == "unreachable"
        assert box.requests == []  # within 30 s of a failed check: no new request
        clock.now += 2
        box.down = False
        _run(v.ready("tts"))
        assert box.requests and box.requests[0].url.path == "/health"

    def test_a_refused_key_shows_in_the_check(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.key = "another-key"
        assert _run(v.status())["tts"] == {"configured": True, "state": "down", "reason": "refused"}

    def test_rejected_is_not_cleared_by_a_good_check(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.status["/v1/audio/speech"] = (400, b"unknown voice")
        _unavailable(v.synthesize("Hi."))
        clock.now += 31
        assert _run(v.status())["tts"] == {"configured": True, "state": "down", "reason": "rejected"}

    def test_rejected_refuses_for_30_seconds_then_lets_a_call_try(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.status["/v1/audio/speech"] = (400, b"unknown voice")
        _unavailable(v.synthesize("Hi."))
        assert _unavailable(v.ready("tts")).reason == "rejected"
        clock.now += 31
        _run(v.ready("tts"))

    def test_a_good_call_clears_any_down_state(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.status["/v1/audio/speech"] = (400, b"unknown voice")
        _unavailable(v.synthesize("Hi."))
        del box.status["/v1/audio/speech"]
        _run(v.synthesize("Hi."))
        assert _run(v.status())["tts"]["state"] == "ok"

    def test_status_checks_when_stale_and_keeps_the_answer_30_seconds(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        assert _run(v.status())["tts"] == {"configured": True, "state": "ok"}
        n = len(box.requests)
        assert n >= 1
        clock.now += 10
        _run(v.status())
        assert len(box.requests) == n
        clock.now += 25
        _run(v.status())
        assert len(box.requests) > n

    def test_unknown_before_any_check(self, box, clock):
        assert _voice(_cfg("tts"), clock).state("tts") == {"configured": True, "state": "unknown"}

    def test_status_never_carries_the_address_or_key(self, box, clock):
        box.key = "another-key"
        text = repr(_run(_voice(_cfg("tts", "stt"), clock).status()))
        assert "127.0.0.1" not in text and "8003" not in text and KEY not in text and KEY_ENV not in text

    def test_one_check_at_a_time(self, box, clock):
        v = _voice(_cfg("tts"), clock)

        async def both():
            await asyncio.gather(v.status(), v.status(), v.status())

        _run(both())
        assert sum(r.url.path == "/health" for r in box.requests) == 1

    def test_a_check_takes_at_most_its_limit(self, clock, monkeypatch):
        async def slow(request):
            await asyncio.sleep(5)
            return httpx.Response(200)

        real = httpx.AsyncClient
        monkeypatch.setattr(voice_mod, "_client",
                            lambda timeout: real(transport=httpx.MockTransport(slow), timeout=timeout))
        monkeypatch.setattr(voice_mod, "CHECK_SECONDS", 0.2)

        async def timed():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            result = await _voice(_cfg("tts"), clock).status()
            return result, loop.time() - t0

        result, took = _run(timed())
        assert took < 2
        assert result["tts"] == {"configured": True, "state": "down", "reason": "unreachable"}

    def test_a_new_address_starts_fresh(self, box, clock):
        cfg = _cfg("tts")
        v = _voice(cfg, clock)
        box.down = True
        _unavailable(v.synthesize("Hi."))
        cfg["tts"]["url"] = "http://127.0.0.1:8004"
        assert v.state("tts") == {"configured": True, "state": "unknown"}


class TestReviewFixes:
    """Review 1 of #924."""

    def _socket_box(self, monkeypatch, handler):
        real = httpx.AsyncClient
        monkeypatch.setattr(voice_mod, "_client",
                            lambda timeout: real(transport=httpx.MockTransport(handler), timeout=timeout))

    def test_a_call_has_a_total_time_limit(self, clock, monkeypatch):
        class Dribble(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"RIFF\x00\x00\x00\x00WAVE"
                while True:
                    await asyncio.sleep(0.05)
                    yield b"\x00"

        self._socket_box(monkeypatch, lambda request: httpx.Response(200, stream=Dribble()))
        cfg = _cfg("tts")
        cfg["tts"]["timeout_seconds"] = 0.3

        async def timed():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            with pytest.raises(VoiceUnavailable) as info:
                await _voice(cfg, clock).synthesize("Hi.")
            return info.value, loop.time() - t0

        err, took = _run(timed())
        assert err.reason == "unreachable" and took < 2

    def test_an_oversized_answer_is_failed_without_reading_it_all(self, clock, monkeypatch):
        read = {"n": 0}

        class Flood(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"RIFF\x00\x00\x00\x00WAVE"
                for _ in range(10_000):
                    read["n"] += 1
                    yield b"\x00" * 65536

        self._socket_box(monkeypatch, lambda request: httpx.Response(200, stream=Flood()))
        assert _unavailable(_voice(_cfg("tts"), clock).synthesize("Hi.")).reason == "failed"
        assert read["n"] * 65536 <= voice_mod.MAX_AUDIO_BYTES + 2 * 65536

    def test_a_bad_url_is_a_failure_not_an_exception(self, box, clock):
        v = _voice(_cfg("tts", "stt", url="http://[::1"), clock)
        assert _unavailable(v.synthesize("Hi.")).reason == "unreachable"
        assert _run(v.status())["stt"]["state"] == "down"

    def test_a_key_a_header_cannot_carry_is_refused(self, box, clock, tmp_path):
        Path(tmp_path / "prom-env").write_text(f"{KEY_ENV}=pasted“key”\n")
        v = _voice(_cfg("tts"), clock)
        assert _unavailable(v.synthesize("Hi.")).reason == "refused"
        assert _run(v.status())["tts"]["state"] == "down"

    def test_a_deeply_nested_transcript_answer_is_failed(self, box, clock):
        box.status["/v1/audio/transcriptions"] = (200, b"[" * 100_000)
        assert _unavailable(_voice(_cfg("stt"), clock).transcribe(b"x", "audio/ogg")).reason == "failed"

    def test_a_refused_key_seen_by_a_call_is_not_cleared_by_a_check(self, box, clock):
        box.status["/v1/audio/speech"] = (401, b"no")  # /v1/models still answers: a check sees nothing wrong
        v = _voice(_cfg("tts"), clock)
        _unavailable(v.synthesize("Hi."))
        box.requests.clear()
        assert _unavailable(v.ready("tts")).reason == "refused"
        assert box.requests == []  # refused for 30 s with no check
        clock.now += 31
        assert _run(v.status())["tts"] == {"configured": True, "state": "down", "reason": "refused"}
        _run(v.ready("tts"))  # after 30 s a call may try again

    def test_a_check_that_began_before_a_call_failed_does_not_overwrite_it(self, box, clock):
        v = _voice(_cfg("tts"), clock)

        async def race():
            gate = asyncio.Event()
            real = voice_mod.Voice._probe

            async def slow_probe(settings):
                await gate.wait()
                return await real(settings)

            v._probe = slow_probe  # type: ignore[method-assign]
            checking = asyncio.ensure_future(v.status())
            await asyncio.sleep(0)
            clock.now += 1
            box.key = "another-key"
            with pytest.raises(VoiceUnavailable):
                await v.synthesize("Hi.")  # refused, after the check began
            box.key = KEY
            gate.set()
            await checking

        _run(race())
        assert v.state("tts") == {"configured": True, "state": "down", "reason": "refused"}

    def test_a_failed_check_keeps_what_a_call_saw(self, box, clock):
        v = _voice(_cfg("tts"), clock)
        box.status["/v1/audio/speech"] = (400, b"unknown voice")
        _unavailable(v.synthesize("Hi."))
        clock.now += 31
        box.down = True
        _run(v.status())
        box.down = False
        clock.now += 31
        assert _run(v.status())["tts"]["reason"] == "rejected"

    def test_the_boxes_text_is_one_log_line(self, box, clock, caplog):
        box.status["/v1/audio/speech"] = (400, b"bad voice\nERROR forged line")
        with caplog.at_level(logging.ERROR, logger="prometheus.speech.voice"):
            _unavailable(_voice(_cfg("tts"), clock).synthesize("Hi."))
        assert "\n" not in caplog.records[-1].getMessage()

    def test_status_checks_both_halves_at_once(self, clock, monkeypatch):
        async def slow(request):
            await asyncio.sleep(0.3)
            return httpx.Response(200, json={})

        self._socket_box(monkeypatch, slow)

        async def timed():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await _voice(_cfg("tts", "stt"), clock).status()
            return loop.time() - t0

        assert _run(timed()) < 1.0  # two requests per check, both halves at once: ~0.6 s, not ~1.2 s
