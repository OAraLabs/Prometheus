"""The voice box, as the speech route uses it (docs/design/speech-route.md, sections 6 and 7).

**Which box.** The one Prometheus already uses for each half (``infra/voice_remote.remote_settings``):
``stt.url`` / ``tts.url`` in prometheus.yaml with the key in the env file, else a box the owner approved by its
address. Nothing here is configured separately; only the URL, key, model, voice and timeout are reused.

**Its own request code.** ``voice_remote``'s calls are for Prometheus's own voice surfaces: a failure there logs
"falling back to the local engine" and moves the "isn't answering" line Telegram and the ``tts`` tool send. The speech
route never falls back to the local engines (the app has a better voice of its own), so it makes its own requests and
keeps its own record of which half is down. Every failure is a :class:`VoiceUnavailable` with a reason and the
sentence for the person, and leaves one log line (WARNING when the box is unreachable or sent something unusable,
ERROR when it refused its key or rejected the request), naming the box's redacted URL and never its key.

**Health.** Checked only when asked: ``GET /health`` at the box's root, then ``GET /v1/models`` to tell a refused key
from a healthy box (the order ``oara doctor`` uses), at most :data:`CHECK_SECONDS` for the whole check, one check at a
time per half. :meth:`Voice.status` checks a half whose last check is older than :data:`CHECK_TTL`.

Only what is about the box or its key changes a half's health for everyone: ``unreachable`` and ``refused``. A call
that fails that way marks the half down at once; a good call marks it up. ``rejected`` (the box refused this request:
a phrase it will not take, audio it cannot decode, an upload too big) and ``failed`` (this answer was unusable) end
that one request and are logged, and nothing else changes: otherwise any paired device could switch a half off for
every device by sending something the box refuses. A misconfigured model or voice name therefore costs one failed
request per stream, each telling the app why, rather than a refusal that a caller's bad audio could also cause.

A check cannot see a key refused only on the audio routes, so a ``refused`` seen by a call is not cleared by a check:
it refuses new streams for :data:`CHECK_TTL` and then lets a call try. A check that began before a call answered never
overwrites the call's answer, good or bad. The record is per box, model, voice and key: changing any of them starts
afresh.

**Bounds.** A call has ``timeout_seconds`` in all (not per read), and reads at most :data:`MAX_AUDIO_BYTES` of audio or
:data:`MAX_JSON_BYTES` of a transcript; a check reads at most :data:`MAX_PROBE_BYTES` of each answer; a compressed
answer is refused, never unzipped. Proxy variables in the environment are ignored (the key goes only to the box), and
redirects are not followed. A bad URL, a bad ``timeout_seconds`` or a key a header cannot carry is a failure like any
other, never an exception out of here.

:meth:`Voice.ready` is the gate a stream passes before it starts: refused at once when nothing is configured, when a
check less than :data:`CHECK_TTL` ago failed, or when a call's ``refused`` is less than that old; a half that is down
only because a call could not reach the box is checked first, so one timeout does not refuse every stream for 30 s.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

import httpx

from prometheus.infra.voice_remote import RemoteVoice, probe_urls, remote_settings

logger = logging.getLogger(__name__)

KINDS = ("stt", "tts")
#: How long a check's answer is kept, and how long a failure refuses a new stream.
CHECK_TTL = 30.0
#: The most one check may take, both requests together.
CHECK_SECONDS = 3.0

#: The sentence for the person, by reason. The app adds its own part ("so this phone's voice is reading the rest").
SENTENCES = {
    "not_configured": "No voice server is set up.",
    "unreachable": "The voice server isn't answering.",
    "refused": "The voice server refused its key.",
    "rejected": "The voice server rejected the request.",
    "failed": "The voice server sent an answer that could not be used.",
}

_LABEL = {"stt": "speech-to-text", "tts": "text-to-speech"}
_EXT = {"audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/mp4": "m4a", "audio/webm": "webm"}
# A request the box understood and refused as wrong: a configuration error, not an outage.
_REJECTED = frozenset({400, 404, 413, 415, 422})
# What a call may say about the box itself, for everyone. Anything else (rejected, failed) is about one request.
_HALF_REASONS = frozenset({"unreachable", "refused"})
#: The most a response may carry, by what is read. A broken or hostile box must not fill the daemon's memory.
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_PROBE_BYTES = 64 * 1024


class _TooLarge(Exception):
    pass


class _Encoded(Exception):
    pass


class VoiceUnavailable(Exception):
    """The voice box cannot do this now. ``reason`` is a key of :data:`SENTENCES`; ``message`` its sentence."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.message = SENTENCES[reason]
        super().__init__(self.message)


@dataclass
class _Health:
    state: str = "unknown"            # ok | down | unknown
    reason: str | None = None         # why it is down
    failed_at: float | None = None    # last failure, by a call or a check
    checked_at: float | None = None   # last check
    check_ok: bool | None = None      # how the last check went
    by_call: bool = False             # the reason came from a call, not a check
    call_at: float | None = None      # when a call last changed the record
    check_started: float | None = None
    inflight: asyncio.Task[str] | None = None


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    # trust_env=False: a proxy variable must never carry the key off the box's loopback or mesh address.
    return httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False)


def _printable(text: str) -> str:
    """For a log line: control characters shown as '?', so a value cannot forge another line."""
    return "".join(ch if ch.isprintable() else "?" for ch in text)


def _reason_for(status: int) -> str:
    if status in (401, 403):
        return "refused"
    if status in _REJECTED:
        return "rejected"
    return "unreachable"


def _headers(settings: RemoteVoice) -> dict[str, str]:
    """The bearer header, and no compression (a size cap must count the bytes that arrive, not what they unzip to).

    UnicodeError for a key a header cannot carry (a curly quote or a line break pasted in, say).
    """
    headers = settings.headers()
    for value in headers.values():
        value.encode("ascii")
        if any(ord(ch) < 32 or ch == "\x7f" for ch in value):
            raise UnicodeError("a control character in a header")
    headers["Accept-Encoding"] = "identity"
    return headers


def _encoded(resp: httpx.Response) -> bool:
    """True when the answer is compressed anyway: it is refused rather than unzipped past the cap."""
    return resp.headers.get("content-encoding", "identity").strip().lower() not in ("", "identity")


def _is_wav(audio: bytes) -> bool:
    return len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


class Voice:
    """The voice box for the speech route. *config* returns the loaded prometheus.yaml each time it is asked."""

    def __init__(self, config: Callable[[], Mapping[str, Any]], *, clock: Callable[[], float] = time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._health_by: dict[tuple[str, ...], _Health] = {}

    # ------------------------------------------------------------ settings
    def settings(self, kind: str) -> RemoteVoice | None:
        try:
            return remote_settings(kind, self._config())
        except (ValueError, TypeError) as exc:  # a bad timeout_seconds, say: fail closed, as not configured
            logger.error("speech: the %s settings cannot be read (%s); treating it as not configured",
                         _LABEL.get(kind, kind), type(exc).__name__)
            return None

    def configured(self, kind: str) -> bool:
        return self.settings(kind) is not None

    @staticmethod
    def _key(settings: RemoteVoice) -> tuple[str, ...]:
        import hashlib

        try:
            key = hashlib.sha256(settings.api_key().encode("utf-8", "replace")).hexdigest()[:16]
        except Exception:  # noqa: BLE001 - an unreadable env file is just "no key" here
            key = ""
        return (settings.kind, settings.url, settings.model, settings.voice, settings.api_key_env, key)

    def _health(self, settings: RemoteVoice) -> _Health:
        return self._health_by.setdefault(self._key(settings), _Health())

    # -------------------------------------------------------------- health
    def state(self, kind: str) -> dict[str, Any]:
        """What ``GET /api/speech`` says about *kind*, without checking. Never the address, the key or its name."""
        settings = self.settings(kind)
        if settings is None:
            return {"configured": False}
        h = self._health(settings)
        out: dict[str, Any] = {"configured": True, "state": h.state}
        if h.state == "down" and h.reason:
            out["reason"] = h.reason
        return out

    async def status(self) -> dict[str, dict[str, Any]]:
        """Both halves, checking (at once) any whose last check is older than :data:`CHECK_TTL`."""
        stale = []
        current = {self._key(s) for kind in KINDS if (s := self.settings(kind)) is not None}
        for old in [k for k in self._health_by if k not in current]:
            del self._health_by[old]  # settings changed: the old record is gone
        for kind in KINDS:
            settings = self.settings(kind)
            if settings is None:
                continue
            h = self._health(settings)
            if h.checked_at is None or self._clock() - h.checked_at >= CHECK_TTL:
                stale.append(self._check(settings, h))
        await asyncio.gather(*stale)
        return {kind: self.state(kind) for kind in KINDS}

    async def ready(self, kind: str) -> RemoteVoice:
        """The settings for *kind* if a stream may start; :class:`VoiceUnavailable` if not (see the module text)."""
        settings = self.settings(kind)
        if settings is None:
            raise VoiceUnavailable("not_configured")
        h = self._health(settings)
        if h.state != "down":
            return settings
        now = self._clock()
        if h.by_call and h.reason == "refused":
            if h.failed_at is not None and now - h.failed_at < CHECK_TTL:
                raise VoiceUnavailable(h.reason)
            return settings  # a check cannot see it; let a call try again
        if h.check_ok is False and h.checked_at is not None and now - h.checked_at < CHECK_TTL:
            raise VoiceUnavailable(h.reason or "unreachable")
        await self._check(settings, h)
        if h.state == "down":
            raise VoiceUnavailable(h.reason or "unreachable")
        return settings

    async def _check(self, settings: RemoteVoice, h: _Health) -> None:
        """One check at a time: a caller arriving while one runs shares its answer."""
        loop = asyncio.get_running_loop()
        if h.inflight is None or h.inflight.done() or h.inflight.get_loop() is not loop:
            h.check_started = self._clock()
            h.inflight = loop.create_task(self._timed_probe(settings))
        started = h.check_started
        result = await asyncio.shield(h.inflight)
        now = self._clock()
        h.checked_at = now  # kept 30 s even when a call's newer answer wins below
        if h.call_at is not None and started is not None and h.call_at > started:
            return  # a call answered after this check began: the call is newer
        sticky = h.state == "down" and h.by_call and h.reason == "refused"
        if result == "ok":
            h.check_ok = True
            if not sticky:
                h.state, h.reason, h.by_call = "ok", None, False
            return
        h.check_ok = False
        if sticky:
            return  # what a call saw stays until a call clears it
        if h.state != "down" or h.reason != result:
            logger.warning("%s at %s: check says %s", _label(settings), _printable(settings.shown_url()), result)
        h.state, h.reason, h.failed_at, h.by_call = "down", result, now, False

    async def _timed_probe(self, settings: RemoteVoice) -> str:
        try:
            return await asyncio.wait_for(self._probe(settings), CHECK_SECONDS)
        except TimeoutError:
            return "unreachable"
        except Exception:  # noqa: BLE001 - a failed check must never leave an exception in the shared task
            logger.warning("%s at %s: the check failed", _label(settings), _printable(settings.shown_url()),
                           exc_info=True)
            return "unreachable"

    @staticmethod
    async def _get_status(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> int:
        """The status of a GET, reading at most :data:`MAX_PROBE_BYTES` of its body."""
        async with client.stream("GET", url, headers=headers) as resp:
            if _encoded(resp):
                return resp.status_code  # only the status matters; a compressed body is never unzipped
            read = 0
            async for chunk in resp.aiter_bytes():
                read += len(chunk)
                if read > MAX_PROBE_BYTES:
                    break
            return resp.status_code

    @classmethod
    async def _probe(cls, settings: RemoteVoice) -> str:
        """``ok`` / ``refused`` / ``unreachable``, in ``voice_remote.probe``'s order."""
        health_url, models_url = probe_urls(settings)
        try:
            headers = _headers(settings)
        except UnicodeError:
            return "refused"
        try:
            async with _client(settings.timeout(CHECK_SECONDS)) as client:
                try:
                    health = await cls._get_status(client, health_url, headers)
                except httpx.HTTPError:
                    return "unreachable"
                if health in (401, 403):
                    return "refused"
                if health >= 500:
                    return "unreachable"
                healthy = 200 <= health < 300
                try:
                    models = await cls._get_status(client, models_url, headers)
                except httpx.HTTPError:
                    return "ok" if healthy else "unreachable"
        except (httpx.InvalidURL, UnicodeError):
            return "unreachable"
        if models in (401, 403):
            return "refused"
        if healthy:
            return "ok"
        return "unreachable" if models >= 500 else "ok"

    # --------------------------------------------------------------- calls
    async def synthesize(self, text: str) -> bytes:
        """The box's WAV for *text*. :class:`VoiceUnavailable` on any failure."""
        settings = self.settings("tts")
        if settings is None:
            raise VoiceUnavailable("not_configured")
        payload = {"model": settings.model, "input": text, "response_format": "wav"}
        if settings.voice:
            payload["voice"] = settings.voice
        audio = await self._post(settings, MAX_AUDIO_BYTES, json=payload)
        if not _is_wav(audio):
            self._failed(settings, "failed", "the answer is not WAV audio")
        self._ok(settings)
        return audio

    async def transcribe(self, audio: bytes, mime: str, language: str | None = None) -> str:
        """The box's transcript of *audio* (declared as *mime*), stripped. :class:`VoiceUnavailable` on any failure."""
        settings = self.settings("stt")
        if settings is None:
            raise VoiceUnavailable("not_configured")
        data = {"model": settings.model, "response_format": "json"}
        if language:
            data["language"] = language
        mime = mime.split(";", 1)[0].strip().lower()  # "audio/ogg; codecs=opus" is audio/ogg
        if mime not in _EXT:
            mime = "application/octet-stream"  # never a caller's string in the multipart headers
        files = {"file": (f"speech.{_EXT.get(mime, 'bin')}", audio, mime)}
        body = await self._post(settings, MAX_JSON_BYTES, data=data, files=files)
        try:
            answer = json.loads(body)
            text = answer.get("text") if isinstance(answer, dict) else None
        except (ValueError, RecursionError):
            text = None
        if not isinstance(text, str):
            self._failed(settings, "failed", "the answer carried no transcript")
        self._ok(settings)
        return text.strip()

    async def _post(self, settings: RemoteVoice, limit: int, **kwargs: Any) -> bytes:
        """The body of a 200 answer, read within ``timeout_seconds`` in all and at most *limit* bytes."""
        try:
            headers = _headers(settings)
        except UnicodeError:
            self._failed(settings, "refused", "the key has characters a header cannot carry")
        try:
            async with asyncio.timeout(max(settings.timeout_seconds, 1.0)):
                async with _client(settings.timeout()) as client:
                    async with client.stream("POST", settings.url, headers=headers, **kwargs) as resp:
                        if _encoded(resp):
                            raise _Encoded
                        body = bytearray()
                        read = 0
                        async for chunk in resp.aiter_bytes():  # identity only, checked above
                            read += len(chunk)
                            if read > limit:
                                raise _TooLarge
                            body.extend(chunk)
                        status = resp.status_code
        except TimeoutError:
            self._failed(settings, "unreachable", f"no full answer within {settings.timeout_seconds:g} s")
        except _TooLarge:
            self._failed(settings, "failed", f"the answer is over {limit} bytes")
        except _Encoded:
            self._failed(settings, "failed", "the answer is compressed, though it was asked not to be")
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
            self._failed(settings, "unreachable", type(exc).__name__)
        if status != 200:
            detail = f"HTTP {status}"
            if status in _REJECTED and body:
                # The box's reason ("unknown voice") is the useful part; repr keeps it on one log line.
                detail += f": {body[:200].decode('utf-8', 'replace').strip()!r}"
            self._failed(settings, _reason_for(status), detail)
        return bytes(body)

    def _ok(self, settings: RemoteVoice) -> None:
        h = self._health(settings)
        h.state, h.reason, h.by_call, h.call_at = "ok", None, False, self._clock()

    def _failed(self, settings: RemoteVoice, reason: str, detail: str) -> NoReturn:
        """Mark the half down, leave the one log line, and raise."""
        if reason in _HALF_REASONS:
            h = self._health(settings)
            now = self._clock()
            h.state, h.reason, h.failed_at, h.by_call, h.call_at = "down", reason, now, True, now
        where = f"{_label(settings)} at {_printable(settings.shown_url())}"
        if reason == "refused":
            logger.error("%s refused the speech route's request (%s); check $%s", where, detail, settings.api_key_env)
        elif reason == "rejected":
            logger.error("%s rejected the speech route's request (%s); check %s.model%s", where, detail,
                         settings.kind, " and tts.voice" if settings.kind == "tts" else "")
        else:
            logger.warning("%s: %s (%s); the app is told and uses its own voice", where,
                           "unreachable" if reason == "unreachable" else "unusable answer", detail)
        raise VoiceUnavailable(reason)


def _label(settings: RemoteVoice) -> str:
    return f"voice box {_LABEL[settings.kind]}"
