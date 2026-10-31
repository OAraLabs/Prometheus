"""Shadow mirroring: send each primary model request to a second, local model too, and only log it.

Distillation Phase 7 (``docs/sprints/DISTILL-FLYWHEEL-RUNBOOK.md``). A tuned student serves beside the
primary, and this measures on live traffic how often it would have done what the primary did. Nothing it
answers ever reaches a turn:

* **The primary path is untouched.** With ``shadow.enabled: false`` (the default) no mirror is built and
  ``LoopContext.shadow`` is None, so a round streams exactly as before. With it on, the primary's events
  pass through by identity, and the shadow request runs in a task of its own that the round never awaits.
  A shadow error or timeout is caught, logged and written to the row; it is never raised.
* **Shadow tool calls are never executed.** The shadow's reply is read for its calls and text, compared
  with the primary's, written down and dropped. No registry, gate or hook ever sees it.
* **One JSONL row per mirrored round** (``shadow-<UTC date>.jsonl`` under ``shadow.log_dir``): request
  id, category, both sides' tool names, whether the arguments are equivalent, both final texts through
  the capture scrubber (``security.log_redaction.redact_capture``), latencies (the time spent
  waiting on each provider, and on its first event) and errors.
  ``scripts/distill/shadow_report.py`` turns the rows into per-category agreement.
* **Auto-disable.** While mirroring is on, the mirror watches the primary itself: when the last
  ``auto_disable.window`` primary rounds show an error rate, or a p95 time to the first streamed event,
  past its threshold, mirroring turns off for the life of the process, says so in the log, records a
  ``shadow_auto_disabled`` event and calls the notify hook (the daemon wires the owner's Telegram chat).
  A restart re-arms it. Time to the first event, not the whole round: a long reply is not a slow
  primary. A backend's "prompt over the window" refusal is not counted at all: the loop answers it
  itself, and it says nothing about the primary's health.

Only the PRIMARY attempt of a round is mirrored; a fallback attempt is not (it is not the model being
measured against). Forced rounds (``tool_choice`` naming a tool or ``required``) are not mirrored either:
llama-server ignores the OpenAI forcing field, so the student would be answering a different question.
Nor is an ephemeral turn (``/ephemeral on``): the loop passes no mirror for it, because a row is
conversation content written to disk.

Shadow targets are local by rule, by provider (``llama_cpp``, ``ollama``, ``lm_studio``, ``vllm``,
``openai_compatible``) AND by address (``base_url`` must be loopback, a private or tailnet address, or a
host name that only resolves on the operator's own network): mirroring to a hosted API would double the
spend and send every conversation to a second provider. Nor may it be a server the daemon already
serves turns from (the primary's, the fallback's, a ``backends:`` entry), whichever way loopback, a
``/v1`` path or a trailing slash is written (a name and its address still differ: nothing is looked up).

All state lives on one event loop (the daemon's), so the counters and the in-flight set need no lock;
only the file append runs in a worker thread, under its own lock.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import logging
import os
import random
import re
import socket
import threading
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Mapping
from urllib.parse import urlparse
from uuid import uuid4

from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest, ApiStreamEvent

log = logging.getLogger(__name__)

ROW_SCHEMA = "shadow-row/1"

#: Request categories, in the order the classifier tries them (the Phase 1 audit's, simplified).
CATEGORIES = ("chat-mode", "sentinel/daemon", "memory/wiki", "coding", "other")

#: Longest final text a row keeps per side. Both sides are cut at the same length, and the row says so.
TEXT_CAP = 20_000
#: Arguments whose JSON is longer than this are stored as a size and a hash, not inline.
ARGS_CAP = 4_000

# The keys a ``shadow:`` block may carry. Anything else is refused at load: a misspelt key would
# otherwise be a setting that silently does nothing.
_KEYS = frozenset({
    "enabled", "provider", "base_url", "model", "sample_rate", "max_concurrency",
    "timeout_seconds", "log_dir", "auto_disable",
})
_AUTO_DISABLE_KEYS = frozenset({"primary_error_rate", "primary_latency_p95_ms", "window"})


class ShadowConfigError(ValueError):
    """The ``shadow:`` block cannot be used, with the reason."""


@dataclass(frozen=True)
class ShadowConfig:
    """The validated ``shadow:`` block. See ``config/prometheus.yaml.default``."""

    enabled: bool
    provider: str
    base_url: str
    model: str
    sample_rate: float
    max_concurrency: int
    timeout_seconds: float
    log_dir: Path
    primary_error_rate: float
    primary_latency_p95_ms: float
    window: int


def _number(value: Any, key: str, *, low: float, high: float | None = None,
            low_open: bool = False) -> float:
    # bool is an int to Python; `sample_rate: true` is a typo, not 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShadowConfigError(f"shadow.{key} must be a number, got {value!r}")
    if (value <= low) if low_open else (value < low):
        raise ShadowConfigError(
            f"shadow.{key} must be {'>' if low_open else '>='} {low:g}, got {value!r}")
    if high is not None and value > high:
        raise ShadowConfigError(f"shadow.{key} must be <= {high:g}, got {value!r}")
    return float(value)


def _integer(value: Any, key: str, *, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowConfigError(f"shadow.{key} must be an integer, got {value!r}")
    if value < low:
        raise ShadowConfigError(f"shadow.{key} must be >= {low}, got {value!r}")
    return value


def resolve_shadow_config(config: Mapping[str, Any] | None) -> ShadowConfig:
    """Validate the ``shadow:`` block and fill in its defaults; raise ShadowConfigError on any problem.

    Validated whether or not it is enabled, so a block that is wrong is reported the day it is
    written rather than the day someone switches it on. An absent block is the defaults: off.

    Each fallback is a literal equal to the template value (tests/test_config_defaults_equality).
    """
    shadow_cfg = (config or {}).get("shadow")
    if shadow_cfg is None:
        shadow_cfg = {}
    if not isinstance(shadow_cfg, Mapping):
        raise ShadowConfigError(f"shadow: expected a mapping, got {type(shadow_cfg).__name__}")
    unknown = sorted(str(k) for k in shadow_cfg if k not in _KEYS)
    if unknown:
        raise ShadowConfigError(f"shadow: unknown key(s) {unknown}; known keys are {sorted(_KEYS)}")

    enabled = shadow_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ShadowConfigError(f"shadow.enabled must be true or false, got {enabled!r}")

    from prometheus.providers.registry import _LOCAL_PROVIDERS

    provider = str(shadow_cfg.get("provider", "llama_cpp") or "")
    if provider not in _LOCAL_PROVIDERS:
        raise ShadowConfigError(
            f"shadow.provider must be a local provider ({', '.join(sorted(_LOCAL_PROVIDERS))}), got "
            f"{provider!r}: mirroring to a hosted API doubles the spend and sends every conversation "
            "to a second provider")

    base_url = str(shadow_cfg.get("base_url", "") or "").strip().rstrip("/")
    if base_url and not base_url.startswith(("http://", "https://")):
        raise ShadowConfigError(f"shadow.base_url must be an http(s) URL, got {base_url!r}")
    # The provider label alone proves nothing: vllm or openai_compatible pointed at a hosted
    # endpoint is a hosted API. So the address must be local too, judged by its spelling (no lookup).
    if base_url and not _is_local_host(_host_of(base_url)):
        raise ShadowConfigError(
            f"shadow.base_url {base_url!r} is not on this machine or its own network: it must be "
            "loopback, a private or tailnet address, or a host name that only resolves locally "
            "(gpu-box, gpu-box.lan, gpu-box.<tailnet>.ts.net). Mirroring to a hosted API doubles "
            "the spend and sends every conversation to a second provider")
    model = str(shadow_cfg.get("model", "") or "").strip()

    sample_rate = _number(shadow_cfg.get("sample_rate", 1.0), "sample_rate", low=0.0, high=1.0)
    max_concurrency = _integer(shadow_cfg.get("max_concurrency", 1), "max_concurrency", low=1)
    timeout_seconds = _number(shadow_cfg.get("timeout_seconds", 300), "timeout_seconds",
                              low=0.0, low_open=True)
    log_dir_raw = shadow_cfg.get("log_dir", "")
    if log_dir_raw is not None and not isinstance(log_dir_raw, str):
        raise ShadowConfigError(f"shadow.log_dir must be a path string, got {log_dir_raw!r}")

    auto_disable = shadow_cfg.get("auto_disable", {})
    if auto_disable is None:
        auto_disable = {}
    if not isinstance(auto_disable, Mapping):
        raise ShadowConfigError(
            f"shadow.auto_disable: expected a mapping, got {type(auto_disable).__name__}")
    unknown = sorted(str(k) for k in auto_disable if k not in _AUTO_DISABLE_KEYS)
    if unknown:
        raise ShadowConfigError(
            f"shadow.auto_disable: unknown key(s) {unknown}; known keys are {sorted(_AUTO_DISABLE_KEYS)}")
    error_rate = _number(auto_disable.get("primary_error_rate", 0.2),
                         "auto_disable.primary_error_rate", low=0.0, high=1.0, low_open=True)
    latency_p95 = _number(auto_disable.get("primary_latency_p95_ms", 120000),
                          "auto_disable.primary_latency_p95_ms", low=0.0, low_open=True)
    window = _integer(auto_disable.get("window", 20), "auto_disable.window", low=1)

    if enabled:
        if not base_url:
            raise ShadowConfigError("shadow.base_url is required when shadow.enabled is true")
        if not model:
            raise ShadowConfigError(
                "shadow.model is required when shadow.enabled is true (it labels every row, and a "
                "round the primary serves with the same model is not mirrored)")
        shadow_server = _server_key(base_url)
        for what, url in _servers_in_use(config):
            if shadow_server is not None and _server_key(url) == shadow_server:
                raise ShadowConfigError(
                    f"shadow.base_url {base_url!r} is {what} ({url}): every mirrored round would "
                    "queue behind (or ahead of) the requests that server is there for")

    if log_dir_raw:
        log_dir = Path(log_dir_raw).expanduser()
    else:
        from prometheus.config.paths import data_dir_path

        log_dir = data_dir_path() / "shadow"

    return ShadowConfig(
        enabled=enabled, provider=provider, base_url=base_url, model=model,
        sample_rate=sample_rate, max_concurrency=max_concurrency, timeout_seconds=timeout_seconds,
        log_dir=log_dir, primary_error_rate=error_rate, primary_latency_p95_ms=latency_p95,
        window=window,
    )


# Host-name suffixes that only resolve on the operator's own network: mDNS, the usual home-router
# and private zones, and Tailscale's MagicDNS names.
_LOCAL_NAME_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal", ".localhost", ".ts.net")


def _host_of(url: str) -> str:
    """The URL's host, lower-cased, without brackets or a trailing dot; "" when it has none."""
    try:
        return (urlparse(url).hostname or "").rstrip(".")
    except ValueError:  # an unclosed IPv6 bracket
        return ""


def _ip_of(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``host`` as an address when it is one, else None.

    In any spelling the resolver takes as one (``2130706433`` and ``0x7f.1`` are 127.0.0.1 to
    getaddrinfo), and with an IPv4-mapped IPv6 address unwrapped to the IPv4 inside it.
    """
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return None
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def _is_local_host(host: str) -> bool:
    """Is ``host`` this machine or a box on its own network, judged by its spelling alone?

    An address: loopback, private, link-local or the tailnet range. A name: a single label
    (``localhost``, ``gpu-box``) or one under a suffix only a local resolver answers. Nothing is
    looked up: the check runs at load, offline, and gives the same answer every time. A public
    name for a LAN box is refused; name that box by its address instead.
    """
    if not host:
        return False
    ip = _ip_of(host)
    if ip is None:
        return "." not in host or host.endswith(_LOCAL_NAME_SUFFIXES)
    from prometheus.security.url_guard import TAILNET_RANGE

    return ip.is_loopback or ip.is_private or ip.is_link_local or ip in TAILNET_RANGE


def _server_key(url: str) -> tuple[str, int] | None:
    """Which server a URL reaches, as (host, port); None when it names none.

    The path (``/v1``), a trailing slash, the host's case and the spelling of loopback
    (``localhost``, ``127.0.0.1``, ``[::1]``, ``0.0.0.0``) do not change the server, so they do not
    change the key. A name and the address it resolves to still differ: nothing is looked up.
    """
    host = _host_of(url)
    if not host:
        return None
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return None
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    ip = _ip_of(host)
    if ip is not None:
        host = "loopback" if ip.is_loopback or ip.is_unspecified else str(ip)
    elif host == "localhost" or host.endswith(".localhost"):
        host = "loopback"
    return host, port


def _servers_in_use(config: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The servers the daemon already serves turns from, as (what it is, its URL).

    The primary; the fallback when ``model.fallback`` places it (absent, the primary IS the
    fallback; one naming a ``backend`` is that entry, compared below); and every ``backends:``
    entry, which a session can be switched to. A hosted provider is left out: it is no server of
    ours for a mirrored round to queue on.
    """
    cfg = config or {}
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    candidates: list[tuple[str, Mapping[str, Any]]] = [("the primary's own server", model_cfg)]
    # The fallback, resolved the way engine/fallback.build_fallback_target resolves it: its own
    # section, else the model config itself (then it is the primary, already listed above).
    section = model_cfg.get("fallback") or model_cfg
    fallback = dict(section) if isinstance(section, Mapping) else {}
    if fallback.get("enabled") is not False and not fallback.get("backend"):
        candidates.append(("the fallback's server", fallback))
    backends = cfg.get("backends")
    if isinstance(backends, Mapping):
        candidates += [(f"backends.{name}", entry) for name, entry in backends.items()
                       if isinstance(entry, Mapping)]

    from prometheus.context.prompt_assembler import is_hosted_provider
    from prometheus.providers.registry import resolve_model_base_url

    servers: list[tuple[str, str]] = []
    for what, entry in candidates:
        if is_hosted_provider(str(entry.get("provider") or "llama_cpp")):
            continue
        try:
            url = resolve_model_base_url(dict(entry))
        except Exception:  # noqa: BLE001 — a server we cannot place cannot be compared; not a refusal
            continue
        if url:
            servers.append((what, url))
    return servers


def _create_provider(config: ShadowConfig) -> Any:
    from prometheus.providers.registry import ProviderRegistry

    return ProviderRegistry.create({
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "timeout": config.timeout_seconds,
    })


def build_shadow_mirror(
    config: Mapping[str, Any] | None,
    *,
    telemetry: Any = None,
    provider_factory: Callable[[ShadowConfig], Any] | None = None,
) -> "ShadowMirror | None":
    """The daemon's mirror, or None: off, or a block that does not validate (logged, never raised).

    A bad block leaves mirroring OFF rather than stopping the daemon: this is a measurement, and no
    measurement is worth a boot failure. It is an ERROR line naming the key, not a silent default.
    """
    try:
        cfg = resolve_shadow_config(config)
    except ShadowConfigError as exc:
        log.error("%s — shadow mirroring stays OFF", exc)
        return None
    if not cfg.enabled:
        log.info("Shadow mirroring: OFF (shadow.enabled: false)")
        return None
    try:
        provider = (provider_factory or _create_provider)(cfg)
    except Exception:  # noqa: BLE001 — an unbuildable shadow is "no shadow", not a crash
        log.error("shadow: could not build the %s provider for %s — shadow mirroring stays OFF",
                  cfg.provider, cfg.base_url, exc_info=True)
        return None
    log.info(
        "Shadow mirroring: ON — %s (%s at %s), sample %.2f, %d in flight at most, %gs timeout, "
        "rows in %s; auto-off past %.0f%% primary errors or %.0f ms p95 over %d rounds",
        cfg.model, cfg.provider, cfg.base_url, cfg.sample_rate, cfg.max_concurrency,
        cfg.timeout_seconds, cfg.log_dir, cfg.primary_error_rate * 100,
        cfg.primary_latency_p95_ms, cfg.window,
    )
    return ShadowMirror(cfg, provider, telemetry=telemetry)


# ── Reading a reply ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Call:
    """One tool call as a reply made it: never executed, only compared."""

    name: str
    arguments: dict[str, Any]


class _OfferedSchemas:
    """The round's offered tools, as the registry the extractor types XML values with.

    ``extract_tool_calls`` reads a ``<parameter=K>`` value as JSON only when the tool's schema says
    K is not a string; with no registry every value is text, so ``limit`` 50 would be ``"50"`` and
    never equal a structured call's 50. The loop passes its own registry; the mirror has the
    request's schemas, which are the same ones. Never a filter: ``get`` answers for any name, so a
    call to a tool the round did not offer is still read as the call the reply made.
    """

    def __init__(self, tools: Any) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        for tool in tools or []:
            schema = tool.get("input_schema") if isinstance(tool, Mapping) else None
            if isinstance(schema, Mapping):
                self._schemas[str(tool.get("name"))] = dict(schema)

    def get(self, name: str) -> Any:
        schema = self._schemas.get(name, {})
        return SimpleNamespace(input_model=SimpleNamespace(model_json_schema=lambda: schema))


def reply_calls(message: Any, tools: Any = None) -> tuple[list[Call], bool]:
    """The calls in a reply, and whether they were read from its text.

    Structured ``tool_uses`` first. When there are none, the reply's text is read with the adapter's
    own extractor, but only when the text carries tool-call markup (``<tool_call`` / ``<function=``):
    a local model with the tools withheld writes its calls that way, and the loop runs them. Prose
    from a cloud model that merely contains a JSON example is not a call, so it is never read as one.
    ``tools`` (the request's offered tools) types each value read from text by its schema.
    """
    if message is None:
        return [], False
    uses = list(getattr(message, "tool_uses", None) or [])
    if uses:
        return [Call(name=u.name, arguments=dict(u.input or {})) for u in uses], False
    text = getattr(message, "text", "") or ""
    if "<tool_call" not in text and "<function=" not in text:
        return [], False
    try:
        from prometheus.adapter.enforcer import StructuredOutputEnforcer

        found = StructuredOutputEnforcer().extract_tool_calls(text, _OfferedSchemas(tools))
    except Exception:  # noqa: BLE001 — an unreadable reply has no calls; the row says so
        log.debug("shadow: tool-call markup could not be read", exc_info=True)
        return [], False
    return [Call(name=b.name, arguments=dict(b.input or {})) for b in found], bool(found)


def reply_text(message: Any) -> str:
    """The reply's prose, tool-call markup removed (what a gateway would show)."""
    text = (getattr(message, "text", "") or "") if message is not None else ""
    if "<" in text:
        from prometheus.adapter.formatter import strip_tool_call_markup

        text = strip_tool_call_markup(text)
    return text.strip()


#: Top-level arguments whose surrounding whitespace means nothing: a shell command line (bash,
#: cron_create and task_create all name theirs ``command``). Everywhere else whitespace IS the
#: argument: an edit's ``old_string`` re-indented, a file's ``content``, YAML.
_TRIMMED_ARGS = frozenset({"command"})


def _canonical(value: Any) -> Any:
    """A value as compared: keys in order, a null key absent, strings exactly as written.

    A null argument is dropped before validation anyway (the adapter's null_drop).
    """
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                if v is not None}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _canonical_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """A call's arguments as compared: ``_canonical``, and a command line trimmed (a trailing
    newline on a command is not a different command)."""
    out = _canonical(dict(args or {}))
    for key in _TRIMMED_ARGS:
        if isinstance(out.get(key), str):
            out[key] = out[key].strip()
    return out


def _strict_eq(a: Any, b: Any) -> bool:
    """JSON equality without bool/int coercion: ``true`` is not ``1``; ``2`` equals ``2.0``."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_strict_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def args_equivalent(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Are two calls' arguments the same arguments? See ``_canonical_args`` for what is ignored."""
    return _strict_eq(_canonical_args(a), _canonical_args(b))


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def normalize_answer_text(text: str | None) -> str:
    """A final answer reduced to what an exact-match judge should compare.

    Unicode-normalised, thinking blocks and tool-call markup removed, case-folded, whitespace
    collapsed, a trailing full stop dropped. Deliberately no further: deciding that two differently
    worded answers mean the same is a judge's job, not a normaliser's.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _THINK_BLOCK.sub(" ", out)
    if "<" in out:
        from prometheus.adapter.formatter import strip_tool_call_markup

        out = strip_tool_call_markup(out)
    out = _SPACE.sub(" ", out.casefold()).strip()
    return out.rstrip(".").rstrip()


# ── Categories ─────────────────────────────────────────────────────────────────────────────

_SENTINEL_TOOL_PREFIXES = ("cron_", "sentinel", "symbiote_")
_MEMORY_TOOLS = frozenset({"memory", "vault_search"})
_MEMORY_TOOL_PREFIXES = ("wiki_", "lcm_")
_CODING_TOOLS = frozenset({
    "edit_file", "write_file", "read_file", "grep", "glob", "lsp", "todo_write", "notebook_edit",
})
_INJECTED_PROVENANCE = frozenset({"cron", "task_supervisor", "orchestrator"})
_DAEMON_COMMAND = re.compile(
    r"\b(?:systemctl|journalctl|crontab|nvidia-smi|llama-server|ollama|docker)\b"
    r"|\.prometheus/|prometheus\.yaml"
    r"|\b(?:oara|prometheus)\s+(?:daemon|status|doctor|setup|pair|logs)\b"
)
_MEMORY_PATH = re.compile(r"wiki/|MEMORY\.md|USER\.md|memory\.db|lcm\.db")


def _turn_context(request: ApiMessageRequest) -> tuple[list[Call], str | None]:
    """The calls made so far in this turn, and who wrote the message that opened it."""
    calls: list[Call] = []
    for message in reversed(request.messages or []):
        if message.role == "assistant":
            calls[:0] = [Call(name=u.name, arguments=dict(u.input or {})) for u in message.tool_uses]
            continue
        if message.text.strip():
            return calls, getattr(message, "provenance", None)
    return calls, None


def classify_round(request: ApiMessageRequest, primary_calls: list[Call],
                   session_id: str | None) -> str:
    """The request category a round is reported under; the first rule that matches wins.

    chat-mode: the round offered no tools by request (``suppress_tools`` / ``tool_choice: none``).
    sentinel/daemon: no session, or the turn was opened by the runtime (cron, a worker's report),
    or a cron/SENTINEL/symbiote tool, or a shell command on the daemon or the host's services.
    memory/wiki: a memory, wiki or LCM tool, or a file or command on the memory stores.
    coding: a file or code tool, any other shell command, or a ``coding:`` session.
    other: everything else (web, tasks, a turn with no tool activity yet).

    The calls judged are the primary's in this round, else the ones made earlier in the turn.
    """
    if request.suppress_tools or request.tool_choice == "none":
        return "chat-mode"
    history, opened_by = _turn_context(request)
    calls = primary_calls or history
    names = [c.name for c in calls]
    blob = json.dumps([c.arguments for c in calls], default=str, ensure_ascii=False)
    shell = [str(c.arguments.get("command", "")) for c in calls if c.name == "bash"]
    sid = session_id or ""
    if (
        sid in ("", "system")
        or opened_by in _INJECTED_PROVENANCE
        or any(n.startswith(_SENTINEL_TOOL_PREFIXES) for n in names)
        or any(_DAEMON_COMMAND.search(cmd) for cmd in shell)
    ):
        return "sentinel/daemon"
    if (
        any(n in _MEMORY_TOOLS or n.startswith(_MEMORY_TOOL_PREFIXES) for n in names)
        or _MEMORY_PATH.search(blob)
    ):
        return "memory/wiki"
    if any(n in _CODING_TOOLS or n == "bash" for n in names) or sid.startswith("coding:"):
        return "coding"
    return "other"


# ── The row ────────────────────────────────────────────────────────────────────────────────


def _describe_error(exc: BaseException) -> str:
    from prometheus.security.log_redaction import redact_secrets

    detail = redact_secrets(str(exc))[:300]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _side(outcome: "_Outcome", *, model: str,
          tools: Any) -> tuple[dict[str, Any], list[Call]]:
    from prometheus.security.log_redaction import redact_capture

    message = outcome.message
    calls, from_text = reply_calls(message, tools)
    # Scrubbed BEFORE it is cut: a cut through a credential leaves a prefix no pattern matches.
    text = redact_capture(reply_text(message))
    args: list[Any] = []
    for call in calls:
        redacted = redact_capture(call.arguments)
        encoded = json.dumps(redacted, sort_keys=True, default=str, ensure_ascii=False)
        if len(encoded) > ARGS_CAP:
            import hashlib

            args.append({"_omitted_chars": len(encoded),
                         "_sha256": hashlib.sha256(encoded.encode()).hexdigest()})
        else:
            args.append(redacted)
    side = {
        "model": model,
        "tools": [c.name for c in calls],
        "args": args,
        "calls_from_text": from_text,
        "text": text[:TEXT_CAP],
        "text_truncated": len(text) > TEXT_CAP,
        "latency_ms": round(outcome.latency_ms, 1),
        "first_event_ms": (round(outcome.first_event_ms, 1)
                           if outcome.first_event_ms is not None else None),
        "error": outcome.error,
        "stop_reason": outcome.stop_reason,
    }
    return side, calls


def build_row(
    *,
    request_id: str,
    request: ApiMessageRequest,
    session_id: str | None,
    round_index: int | None,
    shadow_model: str,
    shadow_provider: str,
    primary: "_Outcome",
    shadow: "_Outcome",
    ts: float,
) -> dict[str, Any]:
    """One comparison row. Pure: everything it needs is passed in."""
    p_side, p_calls = _side(primary, model=request.model, tools=request.tools)
    s_side, s_calls = _side(shadow, model=shadow_model, tools=request.tools)
    s_side["provider"] = shadow_provider
    comparable = primary.error is None and shadow.error is None
    same_tool: bool | None = None
    args_eq: bool | None = None
    answer_match: bool | None = None
    if comparable:
        same_tool = [c.name for c in p_calls] == [c.name for c in s_calls]
        if p_calls:
            args_eq = same_tool and all(
                args_equivalent(a.arguments, b.arguments) for a, b in zip(p_calls, s_calls))
        elif not s_calls:
            answer_match = (normalize_answer_text(reply_text(primary.message))
                            == normalize_answer_text(reply_text(shadow.message)))
    return {
        "schema": ROW_SCHEMA,
        "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "session_id": session_id,
        "round_index": round_index,
        "category": classify_round(request, p_calls, session_id),
        "tools_offered": len(request.tools or []),
        "primary": p_side,
        "shadow": s_side,
        "comparable": comparable,
        "same_tool": same_tool,
        "args_equivalent": args_eq,
        "answer_match": answer_match,
    }


# ── The mirror ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Outcome:
    message: Any = None
    error: str | None = None
    # Time spent waiting on the provider, start to end of its stream; for the primary, the time its
    # consumer spent between events is not in it (it is not the provider's).
    latency_ms: float = 0.0
    # Of that, the wait for the first event; None when none came.
    first_event_ms: float | None = None
    stop_reason: str | None = None


@dataclass
class _Pending:
    """One mirrored round, until both sides are in."""

    request_id: str
    request: ApiMessageRequest
    session_id: str | None
    round_index: int | None
    started_at: float
    task: "asyncio.Task[None] | None" = None
    primary: _Outcome | None = None
    shadow: _Outcome | None = None
    finished: bool = False


def _forced(tool_choice: Any) -> bool:
    return tool_choice not in (None, "auto", "none")


def _is_context_overflow(exc: BaseException) -> bool:
    """Is this the backend refusing a prompt over its window? The loop's own test
    (``_overflow_guard``), so the two cannot disagree about which failures those are."""
    from prometheus.api.turn_errors import KIND_CONTEXT_OVERFLOW, classify_turn_error

    return classify_turn_error(exc).get("kind") == KIND_CONTEXT_OVERFLOW


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, -(-95 * len(ordered) // 100))  # nearest rank: ceil(0.95 n)
    return ordered[rank - 1]


class ShadowMirror:
    """Mirrors primary rounds to the shadow provider. One per daemon, shared by every session.

    ``notify`` is the notification hook auto-disable calls with one line of text (sync or async);
    ``signal_bus`` is late-wired like the compactor's. Neither is required. ``rng`` and the clocks
    are injectable for tests.
    """

    def __init__(
        self,
        config: ShadowConfig,
        provider: Any,
        *,
        telemetry: Any = None,
        notify: Callable[[str], Any] | None = None,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.provider = provider
        self.telemetry = telemetry
        self.notify = notify
        self.signal_bus: Any = None
        self.rng = rng
        self.clock = clock
        self.wall_clock = wall_clock
        self.counters: Counter[str] = Counter()
        self.disabled_reason: str | None = None
        self._enabled = True
        self._inflight: set[asyncio.Task[None]] = set()
        self._background: set[asyncio.Task[None]] = set()
        # The primary's last `window` rounds: (ended well, ms to its first event).
        self._outcomes: deque[tuple[bool, float]] = deque(maxlen=config.window)
        self._write_lock = threading.Lock()
        self._failure_streak = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def snapshot(self) -> dict[str, Any]:
        """Counters and state, for a status line or a test."""
        return {
            "enabled": self._enabled,
            "disabled_reason": self.disabled_reason,
            "in_flight": len(self._inflight),
            "counters": dict(self.counters),
        }

    # -- the primary side ---------------------------------------------------------------

    def mirror(
        self,
        stream: AsyncIterator[ApiStreamEvent],
        request: ApiMessageRequest,
        *,
        session_id: str | None = None,
        round_index: int | None = None,
    ) -> AsyncIterator[ApiStreamEvent]:
        """``stream``, observed; the same object back when mirroring is off.

        Every primary round is observed while mirroring is on (auto-disable judges the primary on
        all of them, not on the sampled share); only a sampled round with a free slot also goes to
        the shadow.
        """
        if not self._enabled:
            return stream
        return self._tap(stream, request, session_id, round_index)

    async def _tap(
        self,
        stream: AsyncIterator[ApiStreamEvent],
        request: ApiMessageRequest,
        session_id: str | None,
        round_index: int | None,
    ) -> AsyncIterator[ApiStreamEvent]:
        pending = self._start(request, session_id, round_index)
        iterator = stream.__aiter__()
        # Only the time spent awaiting the provider is counted. Between two events the consumer
        # has the round (a websocket broadcast, a Telegram edit), and that time is not the
        # primary's: counted, a slow reader would look like a slow model.
        waited = 0.0
        first_event: float | None = None
        final: ApiMessageCompleteEvent | None = None
        error: BaseException | None = None
        stopped = False
        try:
            while True:
                pulled = self.clock()
                try:
                    event = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                finally:
                    waited += self.clock() - pulled
                if first_event is None:
                    first_event = waited
                if isinstance(event, ApiMessageCompleteEvent):
                    final = event
                yield event
        except (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt):
            # The consumer stopped, or the turn was: not the primary failing.
            stopped = True
            raise
        except Exception as exc:
            error = exc
            raise
        finally:
            self._primary_done(
                pending, final=final, error=error, stopped=stopped, latency_ms=waited * 1000.0,
                first_event_ms=first_event * 1000.0 if first_event is not None else None)

    def _start(self, request: ApiMessageRequest, session_id: str | None,
               round_index: int | None) -> _Pending | None:
        """Send this round to the shadow, or say why not. Never raises, never awaits."""
        try:
            if request.model == self.config.model:
                self.counters["skipped_same_model"] += 1
                return None
            if _forced(request.tool_choice):
                self.counters["skipped_forced"] += 1
                return None
            if self.rng() >= self.config.sample_rate:
                self.counters["skipped_sample"] += 1
                return None
            if len(self._inflight) >= self.config.max_concurrency:
                # Skipped, never queued: a queue behind a slow shadow grows without bound.
                self.counters["skipped_busy"] += 1
                return None
            pending = _Pending(
                request_id=uuid4().hex, request=request, session_id=session_id,
                round_index=round_index, started_at=self.wall_clock(),
            )
            # The same request with the shadow's model name. The message list is copied: the loop
            # appends to its own after the round, and the provider reads this one when it runs.
            shadow_request = replace(request, model=self.config.model,
                                     messages=list(request.messages))
            task = asyncio.get_running_loop().create_task(
                self._run_shadow(pending, shadow_request), name=f"shadow-{pending.request_id[:8]}")
            pending.task = task
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            self.counters["mirrored"] += 1
            return pending
        except Exception:  # noqa: BLE001 — a mirror that cannot start is no mirror, never a failed round
            log.debug("shadow: could not start a mirror", exc_info=True)
            return None

    def _primary_done(self, pending: _Pending | None, *, final: ApiMessageCompleteEvent | None,
                      error: BaseException | None, stopped: bool, latency_ms: float,
                      first_event_ms: float | None) -> None:
        """Record how the primary round ended. Runs in the round's own ``finally``: never raises."""
        try:
            if stopped:
                # A stopped turn has no answer to compare against, and says nothing about the
                # primary's health. Its shadow request is dropped with it.
                if pending is not None and pending.task is not None:
                    pending.task.cancel()
                return
            if error is not None:
                described: str | None = _describe_error(error)
            elif final is None:
                described = "stream ended without a final message"
            else:
                described = None
            if error is None or not _is_context_overflow(error):
                # A "prompt over the window" refusal is the conversation's size, not the primary
                # failing: the loop answers it itself (_overflow_guard), writes no provider-error
                # row for it, and neither does this health window. Its row is still written.
                self._observe_primary(described is None, first_event_ms or 0.0)
            if pending is not None:
                pending.primary = _Outcome(
                    message=final.message if final is not None else None,
                    error=described, latency_ms=latency_ms, first_event_ms=first_event_ms,
                    stop_reason=final.stop_reason if final is not None else None,
                )
                self._maybe_finish(pending)
        except Exception:  # noqa: BLE001 — bookkeeping must not reach the round
            log.debug("shadow: primary bookkeeping failed", exc_info=True)

    def _observe_primary(self, ok: bool, first_event_ms: float) -> None:
        """One primary round into the health window: whether it ended well, and how long the
        provider took to send its first event. Not the whole round: a long reply (a big file
        written at a hosted model's tokens per second) is not a slow primary, while a queue or
        a stalled prefill shows up here first."""
        if not self._enabled:
            return
        self._outcomes.append((ok, first_event_ms))
        if len(self._outcomes) < self.config.window:
            return
        errors = sum(1 for good, _ in self._outcomes if not good)
        rate = errors / len(self._outcomes)
        if rate > self.config.primary_error_rate:
            self.disable(
                f"primary error rate {rate:.0%} over the last {len(self._outcomes)} rounds is above "
                f"{self.config.primary_error_rate:.0%}")
            return
        latencies = [ms for good, ms in self._outcomes if good]
        if latencies:
            p95 = _p95(latencies)
            if p95 > self.config.primary_latency_p95_ms:
                self.disable(
                    f"primary p95 latency (time to its first event) {p95:.0f} ms over the last "
                    f"{len(self._outcomes)} rounds is above {self.config.primary_latency_p95_ms:.0f} ms")

    def disable(self, reason: str) -> None:
        """Turn mirroring off for the life of the process, and say so everywhere it can be heard."""
        if not self._enabled:
            return
        self._enabled = False
        self.disabled_reason = reason
        self.counters["auto_disabled"] += 1
        for task in list(self._inflight):
            task.cancel()
        log.warning("Shadow mirroring turned OFF: %s. The primary is unaffected; restart the daemon "
                    "to re-arm it.", reason)
        self._spawn(self._announce(reason))

    async def _announce(self, reason: str) -> None:
        payload = {"reason": reason, "shadow_model": self.config.model, "window": self.config.window,
                   "counters": dict(self.counters)}
        try:
            if self.signal_bus is not None:
                from prometheus.sentinel.signals import ActivitySignal

                await self.signal_bus.emit(ActivitySignal(
                    kind="shadow_auto_disabled", payload=payload, source="shadow_mirror"))
            elif self.telemetry is not None and hasattr(self.telemetry, "record_signal_event"):
                self.telemetry.record_signal_event(
                    signal_type="shadow_auto_disabled", payload=payload,
                    source_subsystem="shadow_mirror")
        except Exception:  # noqa: BLE001
            log.exception("shadow: the auto-disable event could not be recorded")
        if self.notify is None:
            return
        try:
            result = self.notify(
                f"Shadow mirroring turned itself off: {reason}. The primary is unaffected; restart "
                "the daemon to re-arm it.")
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001
            log.exception("shadow: the auto-disable notification failed")

    # -- the shadow side ----------------------------------------------------------------

    async def _run_shadow(self, pending: _Pending, request: ApiMessageRequest) -> None:
        started = self.clock()
        first: list[float] = []  # the clock at the first event, kept through a timeout
        final: ApiMessageCompleteEvent | None = None
        error: str | None = None
        try:
            final = await asyncio.wait_for(self._consume(request, first),
                                           timeout=self.config.timeout_seconds)
        except TimeoutError:
            error = f"timeout after {self.config.timeout_seconds:g}s"
        except Exception as exc:  # noqa: BLE001 — a shadow failure is a row, never an exception
            error = _describe_error(exc)
        latency_ms = (self.clock() - started) * 1000.0
        if error is not None:
            self._failure_streak += 1
            self.counters["shadow_errors"] += 1
            # Loud on the first failure and then every 50th: a shadow box that is down would
            # otherwise write one warning per primary round.
            level = (logging.WARNING if self._failure_streak == 1 or self._failure_streak % 50 == 0
                     else logging.DEBUG)
            log.log(level, "shadow: request to %s failed (%s); %d in a row. The primary is "
                    "unaffected.", self.config.model, error, self._failure_streak)
        else:
            self._failure_streak = 0
        pending.shadow = _Outcome(
            message=final.message if final is not None else None, error=error,
            latency_ms=latency_ms,
            first_event_ms=(first[0] - started) * 1000.0 if first else None,
            stop_reason=final.stop_reason if final is not None else None,
        )
        self._maybe_finish(pending)

    async def _consume(self, request: ApiMessageRequest,
                       first: list[float]) -> ApiMessageCompleteEvent:
        """Read the shadow's reply to the end. Its tool calls are data here, and nothing else."""
        final: ApiMessageCompleteEvent | None = None
        async for event in self.provider.stream_message(request):
            if not first:
                first.append(self.clock())
            if isinstance(event, ApiMessageCompleteEvent):
                final = event
        if final is None:
            raise RuntimeError("stream ended without a final message")
        return final

    # -- the row ------------------------------------------------------------------------

    def _maybe_finish(self, pending: _Pending) -> None:
        if pending.finished or pending.primary is None or pending.shadow is None:
            return
        pending.finished = True
        # Built and written in a task of its own, so whichever side finished last (the primary's
        # round included) only schedules it.
        self._spawn(self._write_row(pending))

    async def _write_row(self, pending: _Pending) -> None:
        try:
            assert pending.primary is not None and pending.shadow is not None
            row = build_row(
                request_id=pending.request_id, request=pending.request,
                session_id=pending.session_id, round_index=pending.round_index,
                shadow_model=self.config.model, shadow_provider=self.config.provider,
                primary=pending.primary, shadow=pending.shadow, ts=pending.started_at,
            )
            line = json.dumps(row, ensure_ascii=False, default=str)
            day = datetime.fromtimestamp(pending.started_at, tz=timezone.utc).strftime("%Y-%m-%d")
            await asyncio.to_thread(self._append, self.config.log_dir / f"shadow-{day}.jsonl", line)
            self.counters["rows_written"] += 1
        except Exception:  # noqa: BLE001
            self.counters["write_errors"] += 1
            log.warning("shadow: comparison row %s could not be written", pending.request_id,
                        exc_info=True)

    def _append(self, path: Path, line: str) -> None:
        # Owner-only: a row holds conversation text (scrubbed of credentials, not of content).
        with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def aclose(self) -> None:
        """Cancel whatever is in flight and wait for it (tests, shutdown)."""
        tasks = list(self._inflight) + list(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Wait until every shadow request and every row write in flight has finished (tests)."""
        while True:
            live = [t for t in (*self._inflight, *self._background) if not t.done()]
            if not live:
                return
            await asyncio.gather(*live, return_exceptions=True)
