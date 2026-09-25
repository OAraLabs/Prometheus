"""The model at the provider boundary: a recording proxy, or a replayer.

One stdlib HTTP server, two modes:

RECORD — every request is forwarded to a real upstream model server
(llama.cpp / Ollama, both OpenAI-compatible; or, for a hosted route, a hosted
API such as Anthropic's) and the exchange is saved. The
UPSTREAM's response is sanitized before the daemon sees it (see
``sanitize_upstream``), so a private identifier the model server publishes —
the GGUF's path under someone's home, say — never enters the daemon, and so
never enters a recorded request either. Recorded requests are stored
NORMALIZED (``normalize.normalize_request``).

REPLAY — answers from the recording. Each incoming ``POST`` is normalized and
matched against the recorded requests by fingerprint:

* an exact match is served, whatever its position (background calls such as
  session titles are fire-and-forget, and their order relative to the next
  turn is scheduling, not behavior);
* no exact match → the next UNCONSUMED recorded exchange for that path is
  served anyway and the pair is filed as a REQUEST MISMATCH. Serving it keeps
  the turn going, so the diff also shows what the divergence did downstream
  instead of stopping at the first symptom;
* nothing left to serve → HTTP 400 (non-retryable on purpose: a 5xx would be
  retried with backoff and look like a hang) and an EXTRA REQUEST entry.

Every served completion is timestamped on the monotonic clock (receipt of the
full request, and the moment the last byte was flushed) — the overhead
benchmark subtracts model time using exactly these.

``GET`` probes (``/props``, ``/v1/models``, ``/api/tags`` …) are answered from
the most recent recording of that path. They are not diffed: how often the
daemon probes a backend is a timer, not a turn.

A HOSTED upstream needs a real API key, and the daemon under test never holds
one: it is given an obviously fake key, and the recording proxy puts the
operator's key on the forwarded request in its place (``upstream_keys``). The
key lives in this process's memory for one recording and is written nowhere —
an ``Exchange`` carries no request headers, so no trace can contain it.
"""

from __future__ import annotations

import http.client
import json
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from parity.normalize import bind_response, fingerprint, normalize_request

# Model calls: matched, consumed, timed and DIFFED. Everything else is a probe.
# "/v1/messages" is Anthropic's Messages API — a turn routed to the hosted
# provider (a /claude override) is a model call like any other, and filing it as
# a probe would take it out of the diff entirely.
COMPLETIONS_PATHS = ("/v1/chat/completions", "/v1/messages")

# Request headers the proxy passes upstream besides Content-Type: the provider
# protocol version the hosted API requires. Local providers send none of them,
# so a local recording forwards exactly what it did before.
_FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")

# Upstream-published values that are private to the recording machine. The
# daemon reads some of them (the served model id is echoed into telemetry and
# the identity line), so they are rewritten BEFORE the daemon sees them —
# rewriting them afterwards, in the trace only, would make every recorded
# request disagree with every replayed one.
_HOME_PATH = re.compile(r"/(?:home|Users)/[^/\"\s]+/")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def sanitize_upstream(text: str) -> str:
    text = _HOME_PATH.sub("/models-root/", text)
    return _IPV4.sub("0.0.0.0", text)


@dataclass
class Exchange:
    """One recorded HTTP exchange at the provider boundary."""

    method: str
    path: str
    status: int
    content_type: str
    body: str                       # response body, verbatim (SSE text or JSON)
    request: Any = None             # NORMALIZED request JSON (POST only)
    upstream: str = ""              # which upstream answered (a label, never a URL)
    fp: str = ""                    # fingerprint(request), cached

    def __post_init__(self) -> None:
        if self.request is not None and not self.fp:
            self.fp = fingerprint(self.request)

    def to_json(self) -> dict:
        d = {
            "method": self.method, "path": self.path, "status": self.status,
            "content_type": self.content_type, "upstream": self.upstream,
        }
        if self.request is not None:
            d["request"] = self.request
            d["fingerprint"] = self.fp
        d["body"] = self.body
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Exchange":
        # Re-normalized on load: the rules only ever ADD placeholders and are
        # idempotent, so a trace recorded before a rule existed still matches.
        req = d.get("request")
        return cls(method=d["method"], path=d["path"], status=d["status"],
                   content_type=d["content_type"], body=d["body"],
                   request=normalize_request(req) if req is not None else None,
                   upstream=d.get("upstream", ""))


@dataclass
class Served:
    """What the replayer did with one incoming completion request."""

    index: int                      # position in arrival order
    recorded_index: int | None      # which recorded exchange answered (None = none left)
    matched: bool                   # fingerprint-exact?
    request: Any                    # the incoming request, normalized
    upstream: str = ""              # which listening port (backend) it arrived on
    t_recv_ns: int = 0
    t_sent_ns: int = 0


@dataclass
class ServerState:
    mode: str                                   # "record" | "replay"
    upstreams: dict[str, str] = field(default_factory=dict)  # label -> base URL
    # label -> the operator's API key for that upstream (record mode, hosted
    # upstreams only). Put on the forwarded request in place of the daemon's
    # fake key; never stored in an Exchange, never written anywhere.
    upstream_keys: dict[str, str] = field(default_factory=dict, repr=False)
    route: dict[int, str] = field(default_factory=dict)      # listen port -> upstream label
    recorded: list[Exchange] = field(default_factory=list)   # record: grows; replay: fixed
    consumed: set[int] = field(default_factory=set)
    served: list[Served] = field(default_factory=list)
    in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    # -- replay matching --------------------------------------------------
    def match(self, path: str, norm: Any, upstream: str) -> tuple[int | None, bool]:
        """Completions: exact fingerprint first, else the next unconsumed one."""
        fp = fingerprint(norm)
        with self.lock:
            candidates = [
                i for i, ex in enumerate(self.recorded)
                if ex.method == "POST" and ex.path == path and i not in self.consumed
            ]
            # Exact = same request on the same BACKEND: an identical body that
            # arrives at the other model is a routing change, not a match.
            for i in candidates:
                if self.recorded[i].fp == fp and self.recorded[i].upstream == upstream:
                    self.consumed.add(i)
                    return i, True
            same_backend = [i for i in candidates if self.recorded[i].upstream == upstream]
            pick = (same_backend or candidates or [None])[0]
            if pick is not None:
                self.consumed.add(pick)
            return pick, False

    def probe(self, method: str, path: str, upstream: str, fp: str = "") -> Exchange | None:
        """Probes (every GET, and POSTs that are not completions) are answered
        from the latest recording of that request and never consumed: how often
        the daemon probes a backend is a timer, not a turn."""
        hits = [ex for ex in self.recorded if ex.method == method and ex.path == path]
        for pool in ([ex for ex in hits if ex.upstream == upstream and (not fp or ex.fp == fp)],
                     [ex for ex in hits if ex.upstream == upstream], hits):
            if pool:
                return pool[-1]
        return None


def _make_handler(state: ServerState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # the harness reports; no stderr noise
            pass

        @property
        def upstream_label(self) -> str:
            return state.route.get(self.server.server_address[1], "")

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if content_type.startswith("text/event-stream"):
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        # -- RECORD ---------------------------------------------------------
        def _upstream_headers(self) -> dict[str, str]:
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            for name in _FORWARDED_HEADERS:
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            # The daemon's key is fake by construction; the operator's goes on
            # the forwarded request instead, in the scheme the daemon used.
            # Without a key for this upstream, no credential is forwarded at all.
            key = state.upstream_keys.get(self.upstream_label)
            if key:
                if self.headers.get("x-api-key"):
                    headers["x-api-key"] = key
                if self.headers.get("Authorization"):
                    headers["Authorization"] = f"Bearer {key}"
            return headers

        def _forward(self, method: str, raw: bytes) -> None:
            if self.upstream_label not in state.upstreams:
                self._send(502, "application/json", json.dumps({"error": {
                    "message": f"parity record: no upstream for {self.upstream_label!r} "
                               f"(this scenario needs --upstream-{self.upstream_label})",
                    "type": "parity_no_upstream"}}).encode())
                return
            base = urllib.parse.urlsplit(state.upstreams[self.upstream_label])
            conn_cls = (http.client.HTTPSConnection if base.scheme == "https"
                        else http.client.HTTPConnection)
            conn = conn_cls(base.hostname, base.port, timeout=900)
            try:
                headers = self._upstream_headers()
                conn.request(method, self.path, body=raw or None, headers=headers)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8", errors="replace")
                ctype = resp.getheader("Content-Type", "application/json")
                status = resp.status
            finally:
                conn.close()
            body = sanitize_upstream(body)
            req = None
            if method == "POST":
                try:
                    req = normalize_request(json.loads(raw or b"{}"))
                except json.JSONDecodeError:
                    req = {"__unparseable__": True}
            ex = Exchange(method=method, path=self.path.split("?")[0], status=status,
                          content_type=ctype, body=body, request=req,
                          upstream=self.upstream_label)
            with state.lock:
                state.recorded.append(ex)
            self._send(status, ctype, body.encode("utf-8"))

        # -- REPLAY ---------------------------------------------------------
        def _replay_probe(self, method: str, norm: Any = None) -> None:
            fp = fingerprint(norm) if norm is not None else ""
            ex = state.probe(method, self.path.split("?")[0], self.upstream_label, fp)
            if ex is None:
                self._send(404, "application/json", json.dumps(
                    {"error": f"parity: no recording of {method} {self.path}"}).encode())
                return
            self._send(ex.status, ex.content_type, ex.body.encode("utf-8"))

        def _replay_post(self, raw: bytes, t_recv: int) -> None:
            path = self.path.split("?")[0]
            try:
                norm = normalize_request(json.loads(raw or b"{}"))
            except json.JSONDecodeError:
                norm = {"__unparseable__": True}
            if path not in COMPLETIONS_PATHS:
                self._replay_probe("POST", norm)
                return
            idx, exact = state.match(path, norm, self.upstream_label)
            with state.lock:
                served = Served(index=len(state.served), recorded_index=idx,
                                matched=exact, request=norm, upstream=self.upstream_label,
                                t_recv_ns=t_recv)
                state.served.append(served)
            if idx is None:
                self._send(400, "application/json", json.dumps({
                    "error": {"message": "parity replay: the daemon sent a model "
                                         "request the recording has no response for",
                              "type": "parity_extra_request"}}).encode())
            else:
                ex = state.recorded[idx]
                body = bind_response(ex.body, raw.decode("utf-8", errors="replace"))
                self._send(ex.status, ex.content_type, body.encode("utf-8"))
            served.t_sent_ns = time.monotonic_ns()

        def do_GET(self):
            with state.lock:
                state.in_flight += 1
            try:
                if state.mode == "record":
                    self._forward("GET", b"")
                else:
                    self._replay_probe("GET")
            finally:
                with state.lock:
                    state.in_flight -= 1

        def do_POST(self):
            raw = self._read_body()
            t_recv = time.monotonic_ns()
            with state.lock:
                state.in_flight += 1
            try:
                if state.mode == "record":
                    self._forward("POST", raw)
                else:
                    self._replay_post(raw, t_recv)
            finally:
                with state.lock:
                    state.in_flight -= 1

    return Handler


class ModelServer:
    """One or more listening ports (one per upstream label) sharing a state."""

    def __init__(self, state: ServerState) -> None:
        self.state = state
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self.ports: dict[str, int] = {}

    def start(self, labels: list[str]) -> dict[str, int]:
        handler = _make_handler(self.state)
        for label in labels:
            srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            srv.daemon_threads = True
            port = srv.server_address[1]
            self.state.route[port] = label
            self.ports[label] = port
            t = threading.Thread(target=srv.serve_forever, name=f"parity-model-{label}",
                                 daemon=True)
            t.start()
            self._servers.append(srv)
            self._threads.append(t)
        return self.ports

    def stop(self) -> None:
        for srv in self._servers:
            srv.shutdown()
            srv.server_close()
        self._servers.clear()
