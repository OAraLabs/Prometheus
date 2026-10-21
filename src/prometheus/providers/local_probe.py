"""What a local AI app at an address is, and whether Prometheus can think with it (setup's look and probe, Beacon's A2).

One definition of "usable", shared by ``GET /api/setup/detect`` (its ``local`` answer), ``POST /api/setup/probe`` and
``POST /api/setup/configure`` for a local provider. The contract is ``docs/contracts/setup-probe.md``.

What a probe may do, so the route cannot be used to read pages it was not built for:

* GET (and Ollama's ``POST /api/show``) to fixed paths only; no redirect followed; at most ``MAX_ANSWER_BYTES`` read;
* the app is identified by what it answers, never by its port (an Ollama also answers ``/v1/models``, and the old
  detector called every typed address llama.cpp);
* only parsed fields come back, never an answer's body: ``detail`` is Prometheus's own sentence.

No LAN scan: a look is this machine's known places, a probe is the one computer someone typed (its port, or the known
places on it). Every place is asked in parallel.

Source: novel code for Prometheus, 2026-10-20.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: The places looked at on a computer: the port each app listens on by default. A test or a recorder points these at
#: stand-ins on other ports.
KNOWN_PLACES: list[dict[str, Any]] = [
    {"app": "llama_cpp", "port": 8080},
    {"app": "ollama", "port": 11434},
    {"app": "lm_studio", "port": 1234},
    {"app": "vllm", "port": 8000},
]
#: The host that is "this machine" for a look.
THIS_MACHINE = "localhost"

#: The most of one answer a probe reads. An answer that is longer is not read as an app.
MAX_ANSWER_BYTES = 256 * 1024
#: The most models reported per app (as the old detector).
MAX_MODELS = 20
#: How long a name may take to resolve (a macOS ``.local`` lookup can stall on the family that is missing).
RESOLVE_TIMEOUT_S = 3.0

#: Why nothing usable was found. When no app answered, the place that says most wins (``PLACE_ORDER``); when an app
#: answered, its models' problems decide (``MODEL_ORDER``).
REASONS = (
    "bad_target",
    "name_not_found",
    "no_answer",
    "refused",
    "needs_key",
    "not_ai",
    "short_context",
    "none_loaded",
    "no_chat",
    "no_models",
)
PLACE_ORDER = ("needs_key", "not_ai", "no_answer", "refused")
MODEL_ORDER = ("short_context", "none_loaded", "no_chat", "no_models")

#: Display names, for ``detail`` only (a client names apps itself).
APP_NAMES = {
    "ollama": "Ollama",
    "lm_studio": "LM Studio",
    "llama_cpp": "llama.cpp",
    "vllm": "vLLM",
    "openai_compatible": "an OpenAI-compatible app",
}
#: Providers setup may write for an app that answers only as OpenAI-compatible (every one of these is the same
#: OpenAI-shaped client; the old wizard called every typed address llama_cpp).
OPENAI_FAMILY = frozenset({"llama_cpp", "lm_studio", "vllm", "openai_compatible"})

#: The smallest window Prometheus can hold a conversation in: round 0 of a fresh install's first turn + the local reply
#: cap (4,096) + 4,096 of conversation, rounded up to 8,192 / 16,384 / 32,768 / 65,536. Set only from
#: MIN_WINDOW_MEASURED, which records how it was measured (``scripts/local_gate/measure_floor.py``).
MIN_WINDOW: int = 0
MIN_WINDOW_MEASURED: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProbeModel:
    id: str
    loaded: bool | None
    chat: bool | None
    window: int | None
    window_source: str | None
    usable: bool
    problem: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "loaded": self.loaded, "chat": self.chat, "window": self.window,
                "window_source": self.window_source, "usable": self.usable, "problem": self.problem}


@dataclass(frozen=True)
class Found:
    app: str
    provider: str
    base_url: str
    version: str | None
    latency_ms: float
    models: list[ProbeModel] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"app": self.app, "provider": self.provider, "base_url": self.base_url, "version": self.version,
                "latency_ms": round(self.latency_ms, 1), "models": [m.as_dict() for m in self.models]}


@dataclass(frozen=True)
class ProbeAnswer:
    target: str | None
    found: list[Found]
    reason: str | None
    detail: str
    min_window: int

    def as_dict(self) -> dict[str, Any]:
        return {"target": self.target, "found": [f.as_dict() for f in self.found], "reason": self.reason,
                "detail": self.detail, "min_window": self.min_window}


# ---------------------------------------------------------------------------
# A typed computer
# ---------------------------------------------------------------------------

_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")


def _host_ok(host: str) -> bool:
    if not host or len(host) > 253:
        return False
    try:
        ipaddress.IPv4Address(host)
        return True
    except ValueError:
        pass
    return all(_LABEL.match(label) for label in host.split("."))


def _port_ok(text: str) -> int | None:
    if not text.isdigit() or len(text) > 5:
        return None
    port = int(text)
    return port if 1 <= port <= 65535 else None


def parse_target(text: str) -> tuple[str | None, str, int | None] | None:
    """(scheme or None, host, port or None) for a typed computer; None when it is not one (``bad_target``).

    A name (``studio-mac``, ``studio-mac.local``), an IPv4 address, a bracketed IPv6 address, any of them with
    ``:port``, or an ``http://`` / ``https://`` URL with no path beyond ``/``. Never userinfo, a query, a fragment,
    a path or another scheme.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t or len(t) > 253 + 14 or any(ord(c) < 0x21 or ord(c) == 0x7F for c in t):
        return None
    scheme: str | None = None
    if "://" in t:
        try:
            parts = urlsplit(t)
            port_text = parts.port  # raises ValueError when out of range
        except ValueError:
            return None
        if parts.scheme.lower() not in ("http", "https") or parts.username is not None or parts.password is not None:
            return None
        if parts.path not in ("", "/") or parts.query or parts.fragment or "@" in parts.netloc:
            return None
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        if port_text is not None and not 1 <= port_text <= 65535:
            return None
        port = port_text
        if ":" in host:
            try:
                ipaddress.IPv6Address(host)
            except ValueError:
                return None
            return scheme, host, port
        return (scheme, host, port) if _host_ok(host) else None
    if any(c in t for c in "/?#@"):
        return None
    if t.startswith("["):
        end = t.find("]")
        if end < 0:
            return None
        host, rest = t[1:end].lower(), t[end + 1:]
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return None
        if not rest:
            return None, host, None
        if not rest.startswith(":") or (port := _port_ok(rest[1:])) is None:
            return None
        return None, host, port
    host, _, port_text = t.lower().partition(":")
    if ":" in port_text:
        return None
    port = None
    if port_text or t.endswith(":"):
        if (port := _port_ok(port_text)) is None:
            return None
    return (None, host, port) if _host_ok(host) else None


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------


@dataclass
class _Reply:
    status: int
    json: Any  # None when the answer was not JSON, or too long


class _Place:
    """One host:port, asked over one resolved address. Each failure is a reason, never an exception."""

    def __init__(self, scheme: str, name: str, addrs: list[tuple[int, str]], port: int, timeout: float,
                 deadline: float) -> None:
        self.scheme, self.name, self.addrs, self.port = scheme, name, addrs, port
        self.timeout, self.deadline = timeout, deadline
        self.addr: tuple[int, str] | None = None
        self.failure: str | None = None  # "refused" | "no_answer" | "needs_key"
        self.asked = 0
        self.first_ms: float | None = None

    @property
    def base_url(self) -> str:
        host = f"[{self.name}]" if ":" in self.name else self.name
        return f"{self.scheme}://{host}:{self.port}"

    def _connect(self, addr: tuple[int, str], timeout: float) -> http.client.HTTPConnection:
        family, ip = addr
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((ip, self.port))
            if self.scheme == "https":
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(sock, server_hostname=self.name)
        except BaseException:
            sock.close()
            raise
        conn = http.client.HTTPConnection(ip, self.port, timeout=timeout)
        conn.sock = sock
        return conn

    def ask(self, path: str, body: dict[str, Any] | None = None) -> _Reply | None:
        """The parsed answer, or None (and ``failure`` set) when the place cannot be asked any more."""
        if self.failure is not None:
            return None
        left = self.deadline - time.monotonic()
        if left <= 0:
            self.failure = self.failure or "no_answer"
            return None
        timeout = min(self.timeout, left)
        candidates = [self.addr] if self.addr else list(self.addrs)
        t0 = time.monotonic()
        last = "refused"
        for addr in candidates:
            try:
                conn = self._connect(addr, timeout)
            except ConnectionRefusedError:
                last = "refused"
                continue
            except (TimeoutError, socket.timeout):
                last = "no_answer"
                continue
            except ssl.SSLError:
                last = "not_ai"
                continue
            except OSError:
                last = "no_answer"
                continue
            self.addr = addr
            try:
                data = json.dumps(body).encode() if body is not None else None
                headers = {"Host": self.base_url.split("://", 1)[1], "Accept": "application/json",
                           "User-Agent": "Prometheus-setup-probe"}
                if data is not None:
                    headers["Content-Type"] = "application/json"
                conn.request("POST" if data is not None else "GET", path, body=data, headers=headers)
                resp = conn.getresponse()
                raw = resp.read(MAX_ANSWER_BYTES + 1)
                status = resp.status
            except (TimeoutError, socket.timeout):
                self.failure = "no_answer"
                return None
            except (OSError, http.client.HTTPException):
                return _Reply(0, None)
            finally:
                conn.close()
            self.asked += 1
            if self.first_ms is None:
                self.first_ms = (time.monotonic() - t0) * 1000.0
            if status in (401, 403):
                self.failure = "needs_key"
                return None
            parsed: Any = None
            if status == 200 and len(raw) <= MAX_ANSWER_BYTES:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
            return _Reply(status, parsed)
        # No address took the connection: the place cannot be asked (refused, no answer, or TLS that is not ours).
        self.failure = last
        return None


def _dict(v: Any) -> dict[str, Any] | None:
    return v if isinstance(v, dict) else None


def _list_of_dicts(v: Any) -> list[dict[str, Any]] | None:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else None


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


# ---------------------------------------------------------------------------
# What an app is, and its models
# ---------------------------------------------------------------------------


def _model(mid: str, *, loaded: bool | None, chat: bool | None, window: int | None, source: str | None,
           needs_loading: bool) -> ProbeModel:
    problem: str | None = None
    if chat is False:
        problem = "no_chat"
    elif needs_loading and loaded is False:
        problem = "none_loaded"
    elif window is not None and window < MIN_WINDOW:
        problem = "short_context"
    return ProbeModel(id=mid[:200], loaded=loaded, chat=chat, window=window, window_source=source if window else None,
                      usable=problem is None, problem=problem)


def _ollama(place: _Place, version: str | None) -> Found | None:
    tags = place.ask("/api/tags")
    entries = _list_of_dicts((_dict(tags.json) or {}).get("models")) if tags else None
    if entries is None:
        return None
    names = [str(e.get("name") or e.get("model") or "") for e in entries][:MAX_MODELS]
    names = [n for n in names if n]
    ps = place.ask("/api/ps")
    loaded = {str(e.get("name") or e.get("model") or "") for e in (_list_of_dicts((_dict(ps.json) or {}).get("models")) or [])} if ps else set()
    models = []
    for name in names:
        reply = place.ask("/api/show", {"model": name})
        show = (_dict(reply.json) if reply else None) or {}
        caps = show.get("capabilities")
        chat: bool | None = None
        if isinstance(caps, list):
            chat = "completion" in caps
        window = None
        info = _dict(show.get("model_info")) or {}
        for key, value in info.items():
            if isinstance(key, str) and key.endswith(".context_length") and _int(value):
                window = value
                break
        models.append(_model(name, loaded=name in loaded, chat=chat, window=window, source="trained",
                             needs_loading=False))
    return Found("ollama", "ollama", place.base_url, version, place.first_ms or 0.0, models)


def _lm_studio(place: _Place, entries: list[dict[str, Any]]) -> Found:
    models = []
    for e in entries[:MAX_MODELS]:
        mid = str(e.get("id") or "")
        if not mid:
            continue
        kind = e.get("type")
        chat = None if not isinstance(kind, str) else kind in ("llm", "vlm")
        state = e.get("state")
        loaded = None if not isinstance(state, str) else state == "loaded"
        window = _int(e.get("loaded_context_length")) if loaded else None
        models.append(_model(mid, loaded=loaded, chat=chat, window=window, source="loaded", needs_loading=True))
    return Found("lm_studio", "lm_studio", place.base_url, None, place.first_ms or 0.0, models)


def _openai_models(place: _Place) -> list[dict[str, Any]] | None:
    reply = place.ask("/v1/models")
    return _list_of_dicts((_dict(reply.json) or {}).get("data")) if reply else None


def identify(place: _Place) -> Found | None:
    """What answers at *place*, by what it answers; None when nothing there is an app (``place.failure`` says why)."""
    ov = place.ask("/api/version")
    if ov is None:
        return None
    version = (_dict(ov.json) or {}).get("version")
    if isinstance(version, str):
        found = _ollama(place, version[:40])
        if found is not None:
            return found
    lm = place.ask("/api/v0/models")
    entries = _list_of_dicts((_dict(lm.json) or {}).get("data")) if lm else None
    if entries is not None and any("type" in e for e in entries):
        return _lm_studio(place, entries)
    props = place.ask("/props")
    settings = _dict((_dict(props.json) or {}).get("default_generation_settings")) if props else None
    if settings is not None:
        n_ctx = _int(settings.get("n_ctx")) or _int((_dict(props.json) or {}).get("n_ctx"))
        data = _openai_models(place) or []
        ids = [str(d.get("id") or "") for d in data if d.get("id")][:MAX_MODELS]
        if not ids:
            path = (_dict(props.json) or {}).get("model_path")
            ids = [path.rsplit("/", 1)[-1]] if isinstance(path, str) and path else []
        models = [_model(i, loaded=True, chat=None, window=n_ctx, source="served", needs_loading=False) for i in ids]
        return Found("llama_cpp", "llama_cpp", place.base_url, None, place.first_ms or 0.0, models)
    vv = place.ask("/version")
    vllm_version = (_dict(vv.json) or {}).get("version") if vv else None
    data = _openai_models(place)
    if data is None:
        return None
    if isinstance(vllm_version, str):
        models = [_model(str(d["id"]), loaded=True, chat=None, window=_int(d.get("max_model_len")), source="served",
                         needs_loading=False) for d in data[:MAX_MODELS] if d.get("id")]
        return Found("vllm", "vllm", place.base_url, vllm_version[:40], place.first_ms or 0.0, models)
    models = [_model(str(d["id"]), loaded=None, chat=None, window=None, source=None, needs_loading=False)
              for d in data[:MAX_MODELS] if d.get("id")]
    return Found("openai_compatible", "openai_compatible", place.base_url, None, place.first_ms or 0.0, models)


# ---------------------------------------------------------------------------
# Looking and probing
# ---------------------------------------------------------------------------


def _resolve(host: str) -> list[tuple[int, str]] | None:
    """Addresses for *host*, IPv4 first; None when it does not resolve within RESOLVE_TIMEOUT_S."""
    try:
        ip = ipaddress.ip_address(host)
        return [(socket.AF_INET6 if ip.version == 6 else socket.AF_INET, str(ip))]
    except ValueError:
        pass

    def lookup(family: int) -> list[tuple[int, str]]:
        try:
            return [(family, info[4][0]) for info in socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)]
        except (socket.gaierror, OSError):
            return []

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        v4, v6 = pool.submit(lookup, socket.AF_INET), pool.submit(lookup, socket.AF_INET6)
        deadline = time.monotonic() + RESOLVE_TIMEOUT_S
        addrs: list[tuple[int, str]] = []
        for fut in (v4, v6):
            try:
                addrs += fut.result(timeout=max(0.0, deadline - time.monotonic()))
            except FutureTimeout:
                pass
    finally:
        pool.shutdown(wait=False)
    seen: set[str] = set()
    out = [(f, a) for f, a in addrs if not (a in seen or seen.add(a))]
    return out or None


def _summary(found: list[Found], places: list[_Place]) -> tuple[str | None, str]:
    usable = [(f, m) for f in found for m in f.models if m.usable]
    parts = []
    for f in found:
        n = sum(1 for m in f.models if m.usable)
        line = f"{APP_NAMES[f.app]} at {f.base_url}: {n} of {len(f.models)} usable"
        short = [m for m in f.models if m.problem == "short_context"]
        if short:
            line += (f" ({', '.join(m.id for m in short[:3])} with a window under {MIN_WINDOW:,};"
                     f" Prometheus needs {MIN_WINDOW:,})")
        parts.append(line)
    if usable:
        return None, "; ".join(parts)
    if found:
        problems = {m.problem for f in found for m in f.models}
        if not any(f.models for f in found):
            problems.add("no_models")
        reason = next(r for r in MODEL_ORDER if r in problems)
        return reason, "; ".join(parts)
    failures = {p.failure or "not_ai" for p in places}
    reason = next(r for r in PLACE_ORDER if r in failures)
    ports = ", ".join(str(p.port) for p in places)
    said = {"needs_key": "asked for a key (401 or 403)", "not_ai": "answered, but not like an AI app Prometheus knows",
            "no_answer": "did not answer in time", "refused": "refused the connection"}
    return reason, f"{places[0].name if places else '?'} on {ports}: " + "; ".join(
        f"{p.port} {said[p.failure or 'not_ai']}" for p in places)


def _probe_places(scheme: str, name: str, addrs: list[tuple[int, str]], ports: list[int], timeout: float,
                  target: str | None) -> ProbeAnswer:
    deadline = time.monotonic() + max(timeout * 4, 1.5)
    places = [_Place(scheme, name, addrs, port, timeout, deadline) for port in ports]
    pool = ThreadPoolExecutor(max_workers=max(1, len(places)))
    try:
        futures = [pool.submit(identify, p) for p in places]
        found: list[Found] = []
        for p, fut in zip(places, futures):
            try:
                f = fut.result(timeout=max(0.0, deadline - time.monotonic()) + 0.5)
            except FutureTimeout:
                p.failure = p.failure or "no_answer"
                continue
            except Exception:  # an app that answers in a shape nobody expected is not an app
                continue
            if f is not None:
                found.append(f)
    finally:
        pool.shutdown(wait=False)
    reason, detail = _summary(found, places)
    return ProbeAnswer(target=target, found=found, reason=reason, detail=detail, min_window=MIN_WINDOW)


def legacy_candidates() -> list[dict[str, str]]:
    """The old detector's candidates (``cli.init.detect_local_servers``), at the same places a look uses, so the two
    answers in ``GET /api/setup/detect`` describe the same ports. With the defaults this is exactly
    ``KNOWN_LOCAL_SERVERS``."""
    from prometheus.cli.init import KNOWN_LOCAL_SERVERS

    by_provider = {spec["provider"]: spec for spec in KNOWN_LOCAL_SERVERS}
    out = []
    for place in KNOWN_PLACES:
        spec = by_provider.get(place["app"])
        if spec is not None:
            out.append({**spec, "url": f"http://{THIS_MACHINE}:{int(place['port'])}"})
    return out


def look_here(timeout: float = 1.0) -> ProbeAnswer:
    """This machine's known places, in parallel."""
    addrs = _resolve(THIS_MACHINE) or [(socket.AF_INET, "127.0.0.1")]
    return _probe_places("http", THIS_MACHINE, addrs, [int(p["port"]) for p in KNOWN_PLACES], timeout, None)


def probe_target(text: str, timeout: float = 3.0) -> ProbeAnswer:
    """The one computer someone typed: its port, or every known place on it, in parallel."""
    parsed = parse_target(text)
    shown = text.strip()[:260] if isinstance(text, str) else ""
    if parsed is None:
        return ProbeAnswer(shown, [], "bad_target", "a target names one computer: a name or an address, with an "
                           "optional port, or an http(s) address with no path", MIN_WINDOW)
    scheme, host, port = parsed
    addrs = _resolve(host)
    if addrs is None:
        return ProbeAnswer(shown, [], "name_not_found", f"{host} did not resolve within {RESOLVE_TIMEOUT_S:g} s",
                           MIN_WINDOW)
    # A URL means its scheme's port; a bare name means the known places on that computer.
    if port is None and scheme is not None:
        port = 443 if scheme == "https" else 80
    ports = [port] if port else [int(p["port"]) for p in KNOWN_PLACES]
    return _probe_places(scheme or "http", host, addrs, ports, timeout, shown)


def probe_choice(provider: str, base_url: str, model: str, timeout: float = 3.0
                 ) -> tuple[bool, str | None, Found | None]:
    """Configure's re-probe: is *model* usable at *base_url* as *provider*?

    (True, None, found) when it is; (False, reason, found) when it is not, with a probe reason, or with None when
    the app answered and does not serve *model* (``model_not_served``).
    """
    answer = probe_target(base_url, timeout=timeout)
    if not answer.found:
        return False, answer.reason or "not_ai", None
    found = answer.found[0]
    # Ollama's own client needs an Ollama. Every other local provider is the same OpenAI-shaped client, which any of
    # these apps answers (the old wizard called every typed address llama_cpp, an Ollama included).
    if provider == "ollama" and found.app != "ollama":
        return False, "not_ai", found
    for m in found.models:
        if m.id == model:
            return (True, None, found) if m.usable else (False, m.problem, found)
    return False, None, found
