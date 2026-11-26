"""The voice-settings API's logic: read the voice config + live options, change it (docs/design/voice-settings.md).

The HTTP shape is ``web/voice_settings_routes.py``; this module knows nothing of HTTP. It sits on the speech
route's own voice-box client (``speech/voice.py``) and the config editor ``/api/model`` uses (``web/network.py``).

What a change is:

* **Saved to the file** as text, comments and all, verified before writing (the document must read back as the
  old one plus exactly the intended keys). A failed verification writes nothing.
* **Applied live** for every field except the box's address: the speech route reads its config on every call,
  so updating the running config in the same request makes the next phrase use it. Voice/speed/engine/VAD/gate
  are also PUSHED to the box (``PUT /v1/settings``); a push that fails is announced, never swallowed.
* **The address waits for a restart** (``applied: "on_restart"``): the health record and the key's meaning are
  bound to the configured box.

Options are DISCOVERED live from the box (``GET /v1/models``, ``/v1/voices``, ``/v1/settings``), never hardcoded.
A box that predates those reads is still usable: discovery is ``unavailable`` and only range/format checks run.

The box's key never appears in an answer, the yaml or a log line; the address appears only in this operator-only
answer. A cross-tier STT model change is refused (the whisper server runs one model per process).

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from prometheus.infra.voice_remote import RemoteVoice, probe_urls, remote_settings
from prometheus.web.network import (
    _edit_text,
    _set_nested,
    read_config_text,
    replace_config,
    verify_edit,
)

logger = logging.getLogger("prometheus.voice_settings")

VERSION = 1

#: Stable error codes (a code is a promise; the English detail is for logs and clients with no copy yet).
INVALID_REQUEST = "invalid_request"
UNKNOWN_VOICE = "unknown_voice"
UNKNOWN_MODEL = "unknown_model"
UNSUPPORTED_ENGINE = "unsupported_engine"
BAD_SPEED = "bad_speed"
BAD_SENSITIVITY = "bad_sensitivity"
BAD_THRESHOLD = "bad_threshold"
BAD_ADDRESS = "bad_address"
TIER_CHANGE = "tier_change"
NO_CONFIG_FILE = "no_config_file"
PERSIST_FAILED = "persist_failed"

#: The fields a PUT may carry, and the config key path each saves to.
FIELDS = {
    "stt_model": ("stt", "model"),
    "tts_engine": ("tts", "provider"),
    "voice": ("tts", "voice"),
    "speed": ("tts", "speed"),
    "vad_sensitivity": ("vad", "sensitivity"),
    "silence_gate_rms": ("stt", "silence_gate_rms"),
}
#: The box address saves to BOTH halves and is the one on_restart field.
ADDRESS_FIELD = "box_address"

#: Knob ranges (the box reports its own; these are the fallback when discovery is unavailable).
SPEED_RANGE = (0.5, 2.0)
VAD_RANGE = (0.1, 0.9)
GATE_RANGE = (0.0, 1000.0)

#: A model id / voice name / engine id: no whitespace, quote, ``#`` or newline (it goes into a YAML scalar).
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,99}")
#: A box address: http(s)://host[:port][/path]. No userinfo, no query (the key is never in the URL).
_ADDRESS = re.compile(r"https?://[A-Za-z0-9._-]+(:\d{1,5})?(/[A-Za-z0-9._/-]*)?")

#: The most a discovery read may take, and the most of its body it may carry.
DISCOVER_SECONDS = 4.0
MAX_DISCOVER_BYTES = 256 * 1024


class VoiceSettingsRefusal(Exception):
    """A request this module will not act on: HTTP *status*, stable *code*, English *detail*."""

    def __init__(self, status: int, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail, self.extra = status, code, detail, extra

    def as_json(self) -> dict[str, Any]:
        return {"error": self.code, "detail": self.detail, **self.extra}


def _literal(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else json.dumps(value)


def _client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    # trust_env=False: a proxy variable must never carry the key off the box's address.
    return httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False)


def _box_root(settings: RemoteVoice) -> str:
    """The box's root URL (``/health``'s parent), from either half's configured endpoint."""
    health, _models = probe_urls(settings)
    return health[: -len("/health")]


def _settings_for(config: Mapping[str, Any], kind: str) -> RemoteVoice | None:
    try:
        return remote_settings(kind, config)
    except (ValueError, TypeError):
        return None


def any_box(config: Mapping[str, Any]) -> RemoteVoice | None:
    """The first configured half — the box the settings reads talk to (both halves share one box)."""
    for kind in ("tts", "stt"):
        s = _settings_for(config, kind)
        if s is not None:
            return s
    return None


# ── discovery ────────────────────────────────────────────────────────────────

async def _get_json(settings: RemoteVoice, url: str) -> dict[str, Any] | None:
    """One bearer-authed JSON read from the box, bounded; None on any failure (announced by the caller)."""
    try:
        headers = settings.headers()
        headers["Accept-Encoding"] = "identity"
        async with _client(settings.timeout(DISCOVER_SECONDS)) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code != 200:
                    return None
                if resp.headers.get("content-encoding", "identity").strip().lower() not in ("", "identity"):
                    return None
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_DISCOVER_BYTES:
                        return None
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except (httpx.HTTPError, ValueError, UnicodeError, TimeoutError):
        return None


@dataclass
class Discovery:
    """What the box says it has. ``live`` False when it could not answer the settings reads."""

    live: bool = False
    stt_models: list[dict[str, Any]] | None = None
    voices: list[str] | None = None
    engines: list[dict[str, Any]] | None = None
    vad_health: str = "unknown"
    ranges: dict[str, Any] | None = None
    box_settings: dict[str, Any] | None = None


async def discover(config: Mapping[str, Any]) -> Discovery:
    """Read the box's models, voices and settings. A box that predates the reads yields ``live=False``."""
    settings = any_box(config)
    if settings is None:
        return Discovery()
    root = _box_root(settings)
    models = await _get_json(settings, root + "/v1/models")
    voices = await _get_json(settings, root + "/v1/voices")
    box = await _get_json(settings, root + "/v1/settings")
    if box is None:
        # An older box: /v1/models may still answer, but without the settings reads there are no live options.
        return Discovery(live=False)
    out = Discovery(live=True, box_settings=box)
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        out.stt_models = [
            {"id": m.get("id"), "tier": m.get("tier"), "active": bool(m.get("active"))}
            for m in models["data"]
            if isinstance(m, dict) and m.get("kind") == "stt" and isinstance(m.get("id"), str)
        ]
    if isinstance(voices, dict) and isinstance(voices.get("voices"), list):
        out.voices = [v for v in voices["voices"] if isinstance(v, str)]
    tts = box.get("tts") if isinstance(box.get("tts"), dict) else {}
    if isinstance(tts.get("engines"), list):
        out.engines = [e for e in tts["engines"] if isinstance(e, dict) and isinstance(e.get("id"), str)]
    vad = box.get("vad") if isinstance(box.get("vad"), dict) else {}
    out.vad_health = vad.get("health") if vad.get("health") in ("ok", "down") else "unknown"
    if isinstance(box.get("ranges"), dict):
        out.ranges = box["ranges"]
    return out


# ── the GET view ─────────────────────────────────────────────────────────────

def _half_view(config: Mapping[str, Any], kind: str, health: Mapping[str, Any]) -> dict[str, Any]:
    settings = _settings_for(config, kind)
    block = config.get(kind) if isinstance(config.get(kind), Mapping) else {}
    if settings is None:
        return {"configured": False, "health": {"state": "unknown"}}
    out: dict[str, Any] = {"configured": True, "health": dict(health)}
    if kind == "stt":
        out["model"] = settings.model
        out["tier"] = _tier_of(settings.model)
    else:
        out["engine"] = str(block.get("provider") or "kokoro")
        out["model"] = settings.model
        out["voice"] = settings.voice or None
        out["speed"] = float(block.get("speed", 1.0) or 1.0)
    out["key"] = {"env": settings.api_key_env, "present": bool(settings.api_key())}
    return out


def _tier_of(model: str) -> str | None:
    return {"base": "cpu", "small": "cpu", "large-v3-turbo": "gpu"}.get(model)


def _options(disc: Discovery) -> dict[str, Any] | None:
    if not disc.live:
        return None
    ranges = disc.ranges or {}

    def _range(name: str, fallback: tuple[float, float]) -> dict[str, float]:
        r = ranges.get(name)
        if isinstance(r, (list, tuple)) and len(r) == 2:
            return {"min": float(r[0]), "max": float(r[1])}
        return {"min": fallback[0], "max": fallback[1]}

    return {
        "stt_models": disc.stt_models or [],
        "tts_engines": disc.engines or [],
        "voices": disc.voices or [],
        "speed": _range("speed", SPEED_RANGE),
        "vad_sensitivity": _range("vad_sensitivity", VAD_RANGE),
        "silence_gate_rms": _range("silence_gate_rms", GATE_RANGE),
    }


def view(config: Mapping[str, Any], health: Mapping[str, Any], disc: Discovery, *,
         pending: bool = False, warnings: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """The state both GET and PUT answer with. ``health`` is ``Voice.status()``'s per-half record."""
    notices = list(warnings or [])
    box = any_box(config)
    vad_block = config.get("vad") if isinstance(config.get("vad"), Mapping) else {}
    stt_block = config.get("stt") if isinstance(config.get("stt"), Mapping) else {}
    if box is None:
        notices.append({"code": "not_configured", "message": "No voice box is set up."})
    elif not disc.live:
        notices.append({"code": "discovery_unavailable",
                        "message": "The voice box did not answer the settings reads; options are not available."})
    return {
        "version": VERSION,
        "stt": _half_view(config, "stt", health.get("stt", {})),
        "tts": _half_view(config, "tts", health.get("tts", {})),
        "vad": {"sensitivity": float(vad_block.get("sensitivity", 0.5) or 0.5),
                "health": {"state": disc.vad_health}},
        "silence_gate": {"threshold_rms": float(stt_block.get("silence_gate_rms", 32.0) or 32.0)},
        "box": {"address": _box_root(box), "discovery": "live" if disc.live else "unavailable"} if box else None,
        "options": _options(disc),
        "pending": pending,
        "applied": "on_restart" if pending else "live",
        "warnings": notices,
    }


# ── reading a PUT ────────────────────────────────────────────────────────────

def _number(value: Any, name: str, lo: float, hi: float, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceSettingsRefusal(400, INVALID_REQUEST, f"{name} must be a number")
    value = float(value)
    if not lo <= value <= hi:
        raise VoiceSettingsRefusal(400, code, f"{name} must be between {lo} and {hi}")
    return value


def _name(value: Any, what: str, code: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value.strip()):
        raise VoiceSettingsRefusal(400, code, f"{what} must be a name (letters, digits, . _ : @ + -)")
    return value.strip()


def clean(body: Any, disc: Discovery, config: Mapping[str, Any]) -> dict[str, Any]:
    """The validated change *body* asks for. Raises VoiceSettingsRefusal; nothing is written.

    Membership checks (a voice the box has, a model it runs, an engine it can run) run only when discovery
    worked; range/format checks always run. A cross-tier STT model change is refused regardless.
    """
    if not isinstance(body, dict):
        raise VoiceSettingsRefusal(400, INVALID_REQUEST, "expected a JSON object")
    for forbidden in ("api_key", "api_key_env", "token"):
        if forbidden in body:
            raise VoiceSettingsRefusal(400, INVALID_REQUEST,
                                       f"{forbidden} is not a field; keys live in the env file only")
    allowed = set(FIELDS) | {ADDRESS_FIELD}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise VoiceSettingsRefusal(400, INVALID_REQUEST, f"unknown field(s): {', '.join(unknown)}")

    out: dict[str, Any] = {}
    ranges = (disc.ranges or {}) if disc.live else {}

    def _range(name: str, fallback: tuple[float, float]) -> tuple[float, float]:
        r = ranges.get(name)
        if isinstance(r, (list, tuple)) and len(r) == 2:
            return float(r[0]), float(r[1])
        return fallback

    if "stt_model" in body:
        model = _name(body["stt_model"], "stt_model", UNKNOWN_MODEL)
        if disc.live and disc.stt_models:
            row = next((m for m in disc.stt_models if m["id"] == model), None)
            if row is None:
                raise VoiceSettingsRefusal(400, UNKNOWN_MODEL, f"the box does not run a model named {model!r}")
            current_tier = _tier_of(str(config.get("stt", {}).get("model", "")))
            if row.get("tier") and current_tier and row["tier"] != current_tier:
                raise VoiceSettingsRefusal(
                    409, TIER_CHANGE,
                    f"{model} runs in the {row['tier']} tier; the box is running {current_tier}. "
                    "Switching tiers is a compose-profile change, not a settings change",
                    tier=row["tier"])
        out["stt_model"] = model
    if "tts_engine" in body:
        engine = _name(body["tts_engine"], "tts_engine", UNSUPPORTED_ENGINE)
        if disc.live and disc.engines:
            row = next((e for e in disc.engines if e["id"] == engine), None)
            if row is None:
                raise VoiceSettingsRefusal(400, UNSUPPORTED_ENGINE, f"the box does not know an engine named {engine!r}")
            if not row.get("available"):
                raise VoiceSettingsRefusal(400, UNSUPPORTED_ENGINE,
                                           f"{engine} is not available: {row.get('reason', 'unknown')}")
        out["tts_engine"] = engine
    if "voice" in body:
        voice = _name(body["voice"], "voice", UNKNOWN_VOICE)
        if disc.live and disc.voices and voice not in disc.voices:
            raise VoiceSettingsRefusal(400, UNKNOWN_VOICE, f"the box has no voice named {voice!r}")
        out["voice"] = voice
    if "speed" in body:
        lo, hi = _range("speed", SPEED_RANGE)
        out["speed"] = _number(body["speed"], "speed", lo, hi, BAD_SPEED)
    if "vad_sensitivity" in body:
        lo, hi = _range("vad_sensitivity", VAD_RANGE)
        out["vad_sensitivity"] = _number(body["vad_sensitivity"], "vad_sensitivity", lo, hi, BAD_SENSITIVITY)
    if "silence_gate_rms" in body:
        lo, hi = _range("silence_gate_rms", GATE_RANGE)
        out["silence_gate_rms"] = _number(body["silence_gate_rms"], "silence_gate_rms", lo, hi, BAD_THRESHOLD)
    if ADDRESS_FIELD in body:
        addr = body[ADDRESS_FIELD]
        if not isinstance(addr, str) or not _ADDRESS.fullmatch(addr.strip()):
            raise VoiceSettingsRefusal(400, BAD_ADDRESS,
                                       "box_address must be http(s)://host[:port][/path], no userinfo or query")
        out[ADDRESS_FIELD] = addr.strip().rstrip("/")
    return out


# ── saving ───────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    target: Path
    updated: str
    expected: dict[str, Any]
    edits: list[tuple[tuple[str, ...], Any]]


def plan_change(path: str | Path, change: Mapping[str, Any]) -> Plan:
    """The verified text edit *change* needs. Raises PersistError / NoConfigFile; the file is not touched."""
    target, original, document = read_config_text(path)
    edits: list[tuple[tuple[str, ...], Any]] = []
    for field, value in change.items():
        if field == ADDRESS_FIELD:
            edits.append((("stt", "url"), value))
            edits.append((("tts", "url"), value))
        else:
            edits.append((FIELDS[field], value))
    expected = copy.deepcopy(document)
    for keys, value in edits:
        _set_nested(expected, list(keys), value)
    updated = _edit_text(original, [(list(keys), _literal(value)) for keys, value in edits]) if edits else original
    verify_edit(original, updated, expected, target, "voice settings")
    return Plan(target=target, updated=updated, expected=expected, edits=edits)


def save(plan: Plan) -> None:
    replace_config(plan.target, plan.updated)


def apply_running(config: dict[str, Any], change: Mapping[str, Any]) -> list[str]:
    """Update the RUNNING config in place for every field except the address (which waits for a restart).

    Returns the fields applied live. The speech route reads this config on every call, so the next phrase
    uses the change at once.
    """
    applied = []
    for field, value in change.items():
        if field == ADDRESS_FIELD:
            continue
        keys = FIELDS[field]
        node = config
        for key in keys[:-1]:
            nxt = node.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                node[key] = nxt
            node = nxt
        node[keys[-1]] = value
        applied.append(field)
    return applied


# ── pushing to the box ───────────────────────────────────────────────────────

async def push(config: Mapping[str, Any], change: Mapping[str, Any]) -> list[dict[str, str]]:
    """Tell the box its new voice/speed/engine/gate/VAD. Returns warnings for what did not reach it.

    The daemon-side save stands regardless (the daemon sends ``voice`` on every call); a push failure is
    announced, never swallowed. The address is never pushed — it is a restart change.
    """
    pushable = {k: v for k, v in change.items() if k != ADDRESS_FIELD and k != "stt_model"}
    if not pushable:
        return []
    settings = any_box(config)
    if settings is None:
        return []
    root = _box_root(settings)
    body: dict[str, Any] = {}
    if "voice" in pushable:
        body["voice"] = pushable["voice"]
    if "speed" in pushable:
        body["speed"] = pushable["speed"]
    if "tts_engine" in pushable:
        body["engine"] = pushable["tts_engine"]
    if "silence_gate_rms" in pushable:
        body["silence_gate_rms"] = pushable["silence_gate_rms"]
    if "vad_sensitivity" in pushable:
        body["vad_sensitivity"] = pushable["vad_sensitivity"]
    try:
        headers = settings.headers()
        headers["Accept-Encoding"] = "identity"
        async with _client(settings.timeout(DISCOVER_SECONDS)) as client:
            resp = await client.put(root + "/v1/settings", headers=headers, json=body)
        if resp.status_code != 200:
            return [{"code": "push_failed",
                     "message": f"The voice box did not take the change (HTTP {resp.status_code}); "
                                "the daemon still sends the new voice on every call."}]
        return []
    except (httpx.HTTPError, UnicodeError, TimeoutError) as exc:
        logger.warning("voice settings: the push to the box failed (%s)", type(exc).__name__)
        return [{"code": "push_failed",
                 "message": "The voice box could not be reached to apply the change; "
                            "the daemon still sends the new voice on every call."}]