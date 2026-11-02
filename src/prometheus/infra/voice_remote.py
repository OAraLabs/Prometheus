"""Optional remote voice servers — ``stt.url`` and ``tts.url``.

Speech-to-text runs a local Whisper engine and text-to-speech runs local
piper/espeak. Either can instead be pointed at an OpenAI-compatible server on
another machine:

* ``stt.url`` — a ``POST /v1/audio/transcriptions`` endpoint (multipart upload)
* ``tts.url`` — a ``POST /v1/audio/speech`` endpoint (JSON in, audio out)

Either url may also name just the server (``http://127.0.0.1:8001`` or
``.../v1``); the route is added. ``whisper.url`` is accepted for ``stt.url``.

The remote is a PREFERENCE, never a dependency. Every caller tries it first
and keeps its local path as the fallback:

* unset (``url: ""``) — no request is made; local runs exactly as before
* unreachable, timed out, or any other failure — local runs, one WARNING line
* 401 / 403 — local runs, but the refusal is an ERROR: a wrong key does not
  fix itself, and quietly speaking with the local voice would hide it
* 400 — the same: the server rejected what was configured (an unknown voice
  or model), which is a configuration error, not an outage

The bearer key is read from the environment or the env file
(``~/.config/prometheus/env``) under the name ``api_key_env`` gives, the same
way model providers name theirs. Unset, it defaults to
``PROMETHEUS_STT_API_KEY`` / ``PROMETHEUS_TTS_API_KEY``. It is NEVER read from prometheus.yaml: an
``api_key`` written there is ignored, so a config file that is shared or
committed cannot carry the credential.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

KINDS = ("stt", "tts")
_LABEL = {"stt": "speech-to-text", "tts": "text-to-speech"}
_ROUTE = {"stt": "/audio/transcriptions", "tts": "/audio/speech"}
# Like a cloud provider's default_env: the name looked up when api_key_env
# is unset.
DEFAULT_KEY_ENV = {"stt": "PROMETHEUS_STT_API_KEY", "tts": "PROMETHEUS_TTS_API_KEY"}

# A refused connection should fail over to the local engine at once, so the
# connect phase gets its own short limit; timeout_seconds bounds the rest.
_CONNECT_TIMEOUT = 5.0

# Same shape test doctor applies to a provider's api_key_env.
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,30}")

_MIME = {".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
         ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
         ".webm": "audio/webm", ".flac": "audio/flac"}


@dataclass(frozen=True)
class RemoteVoice:
    """One configured remote voice server."""

    kind: str
    url: str
    api_key_env: str
    model: str
    voice: str
    timeout_seconds: float

    @property
    def label(self) -> str:
        return f"remote {_LABEL[self.kind]}"

    def api_key(self) -> str:
        """The key the daemon would send: process env first, then the env file."""
        if not self.api_key_env:
            return ""
        value = os.environ.get(self.api_key_env, "")
        if value:
            return value
        try:
            from prometheus.config.env_file import parse_env_file
            return parse_env_file().get(self.api_key_env, "")
        except OSError as exc:
            logger.warning("%s: cannot read the env file for $%s (%s)",
                           self.label, self.api_key_env, exc)
            return ""

    def headers(self) -> dict[str, str]:
        key = self.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def timeout(self, total: float | None = None) -> httpx.Timeout:
        return httpx.Timeout(total or self.timeout_seconds, connect=_CONNECT_TIMEOUT)

    def shown_url(self) -> str:
        from prometheus.api.turn_errors import redact_url
        return redact_url(self.url)


def _load_config() -> dict[str, Any]:
    from prometheus.config.load import load_config_resolved

    return load_config_resolved(
        None, subsystem="voice_remote",
        substituting="the local voice engines", strict=False,
    )


def _endpoint(kind: str, url: str) -> str:
    """The full route for *url*, which may name only the server or its /v1."""
    url = url.rstrip("/")
    if url.endswith(_ROUTE[kind]):
        return url
    if not url.endswith("/v1"):
        url += "/v1"
    return url + _ROUTE[kind]


def remote_settings(
    kind: str, config: Mapping[str, Any] | None = None,
) -> RemoteVoice | None:
    """The configured remote for *kind* (``"stt"``/``"tts"``), or None when unset.

    *config* is the loaded prometheus.yaml; None reads it from the usual search
    path (the ``tts`` tool and the STT tool are built without one).
    """
    if config is None:
        config = _load_config()
    block = config.get(kind, {})
    if not isinstance(block, Mapping):
        block = {}
    url = str(block.get("url", "") or "").strip()
    if not url and kind == "stt":
        # whisper: is where local speech-to-text is configured, so whisper.url
        # is the name an operator reaches for first. The stt: block keeps
        # every other setting.
        whisper = config.get("whisper", {})
        if isinstance(whisper, Mapping):
            url = str(whisper.get("url", "") or "").strip()
    if not url:
        return None
    if block.get("api_key"):
        logger.warning(
            "%s.api_key is set in prometheus.yaml and is ignored — the key is "
            "read only from the environment or the env file. Put it there and "
            "name the variable in %s.api_key_env.", kind, kind,
        )
    key_env = str(block.get("api_key_env", "") or "").strip()
    if key_env and not _ENV_NAME_RE.fullmatch(key_env):
        # The value is not echoed: a key pasted here is the likely cause.
        logger.error(
            "%s.api_key_env does not look like a variable name (value not "
            "shown) — put the NAME of the variable that holds the key there. "
            "Using the local engine.", kind,
        )
        return None
    default_model = "whisper-1" if kind == "stt" else "tts-1"
    return RemoteVoice(
        kind=kind,
        url=_endpoint(kind, url),
        api_key_env=key_env or DEFAULT_KEY_ENV[kind],
        model=str(block.get("model", default_model) or default_model),
        # Empty: no voice is sent and the server uses its own default.
        voice=str(block.get("voice", "") or ""),
        timeout_seconds=float(block.get("timeout_seconds", 60) or 60),
    )


def _async_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def _sync_client(timeout: httpx.Timeout) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def _log_failure(settings: RemoteVoice, detail: str, status: int | None = None) -> None:
    """The one line a failed remote call leaves behind."""
    if status in (401, 403):
        logger.error(
            "%s at %s refused the request (HTTP %s) — check $%s; falling back "
            "to the local engine", settings.label, settings.shown_url(), status,
            settings.api_key_env,
        )
        return
    if status == 400:
        logger.error(
            "%s at %s rejected the request (%s) — check %s.model%s; falling "
            "back to the local engine", settings.label, settings.shown_url(),
            detail, settings.kind, " and tts.voice" if settings.kind == "tts" else "",
        )
        return
    logger.warning("%s at %s unavailable (%s); falling back to the local engine",
                   settings.label, settings.shown_url(), detail)


async def _post(settings: RemoteVoice, **kwargs: Any) -> httpx.Response | None:
    try:
        async with _async_client(settings.timeout()) as client:
            resp = await client.post(settings.url, headers=settings.headers(), **kwargs)
    except httpx.HTTPError as exc:
        _log_failure(settings, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)
        return None
    if resp.status_code != 200:
        detail = f"HTTP {resp.status_code}"
        if resp.status_code == 400 and resp.text:
            # The server's reason ("unknown voice") is the useful part.
            detail += f": {resp.text.strip()[:200]}"
        _log_failure(settings, detail, resp.status_code)
        return None
    return resp


async def transcribe_remote(
    settings: RemoteVoice, audio_path: str | Path, language: str | None = None,
) -> str | None:
    """Transcribe *audio_path* on the remote server. None means use local."""
    path = Path(audio_path)
    try:
        audio = path.read_bytes()
    except OSError as exc:
        _log_failure(settings, f"cannot read {path.name}: {exc}")
        return None
    data = {"model": settings.model, "response_format": "json"}
    if language:
        data["language"] = language
    files = {"file": (path.name, audio, _MIME.get(path.suffix.lower(), "application/octet-stream"))}
    resp = await _post(settings, data=data, files=files)
    if resp is None:
        return None
    try:
        text = resp.json().get("text")
    except ValueError:
        text = resp.text
    if not isinstance(text, str):
        _log_failure(settings, "response carried no transcription text")
        return None
    return text.strip()


async def synthesize_remote(
    settings: RemoteVoice, text: str, out_path: str | Path,
) -> bool:
    """Write a WAV of *text* to *out_path* from the remote server. False means use local."""
    payload = {"model": settings.model, "input": text, "response_format": "wav"}
    if settings.voice:
        payload["voice"] = settings.voice
    resp = await _post(settings, json=payload)
    if resp is None:
        return False
    if not resp.content:
        _log_failure(settings, "response carried no audio")
        return False
    try:
        Path(out_path).write_bytes(resp.content)
    except OSError as exc:
        _log_failure(settings, f"cannot write {out_path}: {exc}")
        return False
    return True


def probe_urls(settings: RemoteVoice) -> tuple[str, str]:
    """Where doctor checks the server: ``(health, models)``.

    ``/health`` sits at the server root, beside ``/v1`` — it is what voice
    servers reliably serve, and it answers without touching a model.
    ``/v1/models`` is optional: an OpenAI-compatible server that has it
    authenticates it like the audio routes, so it is the one read that can
    tell a refused key apart from a healthy server.
    """
    api_base = settings.url.rstrip("/")
    for suffix in ("/audio/transcriptions", "/audio/speech"):
        if api_base.endswith(suffix):
            api_base = api_base[: -len(suffix)]
            break
    root = api_base[: -len("/v1")] if api_base.endswith("/v1") else api_base
    return root + "/health", api_base + "/models"


def probe(settings: RemoteVoice, timeout: float = 5.0) -> tuple[str, int | None]:
    """``("ok"|"refused"|"unreachable", status)`` for a doctor row.

    ``/health`` first. A 2xx there means the server is up whatever
    ``/v1/models`` says, except that a 401/403 from it still reports the key
    as refused. Without ``/health`` (404 or similar), the models answer
    decides alone, as any reply from a listening server does.
    """
    health_url, models_url = probe_urls(settings)
    headers = settings.headers()
    with _sync_client(settings.timeout(timeout)) as client:
        try:
            health = client.get(health_url, headers=headers)
        except httpx.HTTPError:
            return "unreachable", None
        if health.status_code in (401, 403):
            return "refused", health.status_code
        if health.status_code >= 500:
            return "unreachable", health.status_code
        healthy = health.is_success
        try:
            models = client.get(models_url, headers=headers)
        except httpx.HTTPError:
            return ("ok", health.status_code) if healthy else ("unreachable", None)
    if models.status_code in (401, 403):
        return "refused", models.status_code
    if healthy:
        return "ok", health.status_code
    if models.status_code >= 500:
        return "unreachable", models.status_code
    # Any other answer — even a 404 from a server with neither route — means
    # something is listening there.
    return "ok", models.status_code
