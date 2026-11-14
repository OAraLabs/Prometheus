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
time per half. :meth:`Voice.status` checks a half whose last check is older than :data:`CHECK_TTL`. Every real call
counts too: a failure marks the half down at once, a success marks it up. A check cannot see that the box would
reject a request (a bad model or voice name), so a good check never clears ``rejected``; only a good call does.

:meth:`Voice.ready` is the gate a stream passes before it starts: refused at once when nothing is configured, when a
check less than :data:`CHECK_TTL` ago failed, or when the box rejected a request less than that ago; a half that is down
only because a call failed is checked first, so one timeout does not refuse every stream for 30 s.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import asyncio
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
    inflight: asyncio.Task[str] | None = None


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def _reason_for(status: int) -> str:
    if status in (401, 403):
        return "refused"
    if status in _REJECTED:
        return "rejected"
    return "unreachable"


def _is_wav(audio: bytes) -> bool:
    return len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


class Voice:
    """The voice box for the speech route. *config* returns the loaded prometheus.yaml each time it is asked."""

    def __init__(self, config: Callable[[], Mapping[str, Any]], *, clock: Callable[[], float] = time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._health_by: dict[tuple[str, str], _Health] = {}

    # ------------------------------------------------------------ settings
    def settings(self, kind: str) -> RemoteVoice | None:
        return remote_settings(kind, self._config())

    def configured(self, kind: str) -> bool:
        return self.settings(kind) is not None

    def _health(self, settings: RemoteVoice) -> _Health:
        return self._health_by.setdefault((settings.kind, settings.url), _Health())

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
        """Both halves, checking any whose last check is older than :data:`CHECK_TTL`."""
        for kind in KINDS:
            settings = self.settings(kind)
            if settings is None:
                continue
            h = self._health(settings)
            if h.checked_at is None or self._clock() - h.checked_at >= CHECK_TTL:
                await self._check(settings, h)
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
        if h.reason == "rejected":
            if h.failed_at is not None and now - h.failed_at < CHECK_TTL:
                raise VoiceUnavailable("rejected")
            return settings  # a check cannot see it; let a call try again
        if h.check_ok is False and h.checked_at is not None and now - h.checked_at < CHECK_TTL:
            raise VoiceUnavailable(h.reason or "unreachable")
        await self._check(settings, h)
        if h.state == "down":
            raise VoiceUnavailable(h.reason or "unreachable")
        return settings

    async def _check(self, settings: RemoteVoice, h: _Health) -> None:
        """One check at a time: a caller arriving while one runs shares its answer."""
        if h.inflight is None or h.inflight.done():
            h.inflight = asyncio.ensure_future(self._timed_probe(settings))
        result = await asyncio.shield(h.inflight)
        now = self._clock()
        h.checked_at = now  # every caller that shared the check applies the same answer
        if result == "ok":
            h.check_ok = True
            if h.reason != "rejected":
                h.state, h.reason = "ok", None
            return
        h.check_ok = False
        if h.state != "down" or h.reason != result:
            logger.warning("%s at %s: check says %s", _label(settings), settings.shown_url(), result)
        h.state, h.reason, h.failed_at = "down", result, now

    async def _timed_probe(self, settings: RemoteVoice) -> str:
        try:
            return await asyncio.wait_for(self._probe(settings), CHECK_SECONDS)
        except TimeoutError:
            return "unreachable"

    @staticmethod
    async def _probe(settings: RemoteVoice) -> str:
        """``ok`` / ``refused`` / ``unreachable``, in ``voice_remote.probe``'s order."""
        health_url, models_url = probe_urls(settings)
        headers = settings.headers()
        async with _client(settings.timeout(CHECK_SECONDS)) as client:
            try:
                health = await client.get(health_url, headers=headers)
            except httpx.HTTPError:
                return "unreachable"
            if health.status_code in (401, 403):
                return "refused"
            if health.status_code >= 500:
                return "unreachable"
            healthy = health.is_success
            try:
                models = await client.get(models_url, headers=headers)
            except httpx.HTTPError:
                return "ok" if healthy else "unreachable"
        if models.status_code in (401, 403):
            return "refused"
        if healthy:
            return "ok"
        return "unreachable" if models.status_code >= 500 else "ok"

    # --------------------------------------------------------------- calls
    async def synthesize(self, text: str) -> bytes:
        """The box's WAV for *text*. :class:`VoiceUnavailable` on any failure."""
        settings = self.settings("tts")
        if settings is None:
            raise VoiceUnavailable("not_configured")
        payload = {"model": settings.model, "input": text, "response_format": "wav"}
        if settings.voice:
            payload["voice"] = settings.voice
        resp = await self._post(settings, json=payload)
        if not _is_wav(resp.content):
            self._failed(settings, "failed", "the answer is not WAV audio")
        self._ok(settings)
        return resp.content

    async def transcribe(self, audio: bytes, mime: str, language: str | None = None) -> str:
        """The box's transcript of *audio* (declared as *mime*), stripped. :class:`VoiceUnavailable` on any failure."""
        settings = self.settings("stt")
        if settings is None:
            raise VoiceUnavailable("not_configured")
        data = {"model": settings.model, "response_format": "json"}
        if language:
            data["language"] = language
        files = {"file": (f"speech.{_EXT.get(mime, 'bin')}", audio, mime)}
        resp = await self._post(settings, data=data, files=files)
        try:
            text = resp.json().get("text")
        except (ValueError, AttributeError):
            text = None
        if not isinstance(text, str):
            self._failed(settings, "failed", "the answer carried no transcript")
        self._ok(settings)
        return text.strip()

    async def _post(self, settings: RemoteVoice, **kwargs: Any) -> httpx.Response:
        try:
            async with _client(settings.timeout()) as client:
                resp = await client.post(settings.url, headers=settings.headers(), **kwargs)
        except httpx.HTTPError as exc:
            self._failed(settings, "unreachable", type(exc).__name__)
        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}"
            if resp.status_code in _REJECTED and resp.text:
                detail += f": {resp.text.strip()[:200]}"  # the box's reason ("unknown voice") is the useful part
            self._failed(settings, _reason_for(resp.status_code), detail)
        return resp

    def _ok(self, settings: RemoteVoice) -> None:
        h = self._health(settings)
        h.state, h.reason = "ok", None

    def _failed(self, settings: RemoteVoice, reason: str, detail: str) -> NoReturn:
        """Mark the half down, leave the one log line, and raise."""
        h = self._health(settings)
        h.state, h.reason, h.failed_at = "down", reason, self._clock()
        where = f"{_label(settings)} at {settings.shown_url()}"
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
