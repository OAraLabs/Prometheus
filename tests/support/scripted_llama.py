"""A scripted stand-in for the 4090's llama-server, with its real constraints.

The local gate (tests/test_local_gate_integration.py) boots the real daemon and
replaces only the model. This is the replacement. It serves the endpoints the
daemon uses (``/props``, ``/v1/models``, ``/health``, ``/tokenize`` and streamed
or plain ``/v1/chat/completions``) on a loopback port, in a thread of the test
process, so the daemon child reaches it over HTTP like the real server.

THE CONSTRAINTS IT KEEPS (each measured on the 4090, 2026-10-16: llama.cpp
build b1-9d57ce456, Qwen3.8-27B-UD-Q4_K_XL, ``--jinja``, one slot)

* **The window.** ``n_ctx`` 32768. Prompt tokens are counted from what the
  template renders (messages, tool schemas, framing) at ``chars_per_token``
  (3.73 makes Prometheus's chars/4 estimate read ~7% low, as live telemetry
  shows). A prompt over the window gets llama.cpp's own 400
  ``exceed_context_size_error`` with ``n_prompt_tokens`` and ``n_ctx``. A
  reply is cut at ``min(max_tokens, n_ctx - prompt)``.
* **Only offered tool names.** With native ``tools`` the server builds its
  tool-call grammar from them (common/chat.cpp, the Qwen3-Coder handler), so a
  call can name only an offered tool. The probes: with no grammar the model
  wrote ``<function=lcm_grep>``; with the tools it came back as ``tool_search``
  select (or ``memory``). Told to call ``agent`` or ``cron_create`` after
  select returned the schema, it selected again, grammar or not. So a scripted
  call to an unoffered tool becomes a ``tool_search`` select of that name when
  ``tool_search`` is offered, and a short refusal in text when it is not.
* **tool_calls on truncation.** A tool call cut at the output limit comes back
  with ``finish_reason: "tool_calls"`` and the partial arguments, not
  ``"length"`` (why #806 judges a cut-off call by its token count). Cut text
  reports ``"length"``.
* **One slot, one prompt cache.** ``cached_tokens`` is the common prefix with
  the previous chat request, in tokens.
* **No ``tools`` but a ``grammar``** (tier full, a forced round): the grammar
  Prometheus sends covers every registered tool, so a call there is produced as
  JSON text in the content, never as a native call.

THE SCRIPT

A conversation picks its script with a marker in its first user message,
``[gate:<name>]`` (``SCRIPTS`` below). A subagent's conversation carries the
marker the parent wrote into its prompt. A request with no system message and
no tools is a summarizer call; one asking for at most 32 tokens is a title.
"""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

MODEL_ID = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
_MARK = re.compile(r"\[gate:([a-z0-9_]+)\]")
_FRAMING_PER_MESSAGE = 12        # <|im_start|>role\n ... <|im_end|>\n
_TOOLS_WRAPPER = 907             # the template's "# Tools ... <IMPORTANT>" text, measured


@dataclass
class Call:
    name: str
    args: dict[str, Any]


@dataclass
class Text:
    text: str


@dataclass
class Exchange:
    """What the stand-in received and what it sent, for one chat request."""

    script: str
    request: dict[str, Any]
    prompt_tokens: int
    cached_tokens: int
    status: int
    finish_reason: str | None = None
    sent: dict[str, Any] = field(default_factory=dict)

    @property
    def offered(self) -> list[str]:
        return [(t.get("function") or {}).get("name", "") for t in self.request.get("tools") or []]

    @property
    def system(self) -> str:
        return next((m.get("content") or "" for m in self.request.get("messages", []) if m.get("role") == "system"), "")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def rendered_chars(payload: dict[str, Any]) -> int:
    """Characters the chat template renders for this request."""
    chars = 0
    for m in payload.get("messages", []):
        chars += _FRAMING_PER_MESSAGE + len(_content_text(m.get("content")))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            chars += 40 + len(fn.get("name", "")) + len(fn.get("arguments", ""))
    tools = payload.get("tools") or []
    if tools:
        chars += _TOOLS_WRAPPER + sum(len(json.dumps(t.get("function", t), ensure_ascii=False)) + 1 for t in tools)
    return chars


def _render_key(payload: dict[str, Any]) -> str:
    """A stand-in for the rendered prompt, for prefix-cache matching: tools first
    (the template puts ``# Tools`` ahead of the system text), then messages."""
    parts = [json.dumps(payload.get("tools") or [], ensure_ascii=False, sort_keys=True)]
    for m in payload.get("messages", []):
        parts.append(json.dumps({k: m.get(k) for k in ("role", "content", "tool_calls")}, ensure_ascii=False))
    return "\x1e".join(parts)


def tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(tool name, result text) for every tool result, in order."""
    names: dict[str, str] = {}
    out: list[tuple[str, str]] = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc.get("id", "")] = (tc.get("function") or {}).get("name", "")
        if m.get("role") == "tool":
            out.append((names.get(m.get("tool_call_id", ""), ""), _content_text(m.get("content"))))
    return out


def all_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(_content_text(m.get("content")) for m in messages)


# ---------------------------------------------------------------- the scripts


def want_then_answer(name: str, args: dict[str, Any]) -> Callable[[list[dict[str, Any]]], Call | Text]:
    """Call ``name`` once; after its result arrives, answer."""

    def script(messages: list[dict[str, Any]]) -> Call | Text:
        done = [r for n, r in tool_results(messages) if n == name]
        if done:
            return Text(f"Done. {name} said: {done[-1][:120]}")
        return Call(name, args)

    return script


LEDGER_MODULES = ("accounts", "accruals", "allocations", "assets", "audit", "billing", "budgets", "currency",
                  "deposits", "dividends", "equity", "expenses", "forecasts", "invoices", "journal", "liabilities",
                  "payments", "payroll", "receivables", "reconciliation", "refunds", "reports", "statements",
                  "taxes", "transfers", "ledger_io")


def _coding(messages: list[dict[str, Any]]) -> Call | Text:
    """Read every ledger module one call at a time, edit billing.py, answer.

    It keeps its place through compaction: the summarizer below writes the
    files read so far into the summary, and this reads them back from there.
    """
    text = all_text(messages)
    read = {m for m in LEDGER_MODULES if f"FILE ledger/{m}.py" in text}
    for m in re.findall(r"files read so far: ([a-z_, ]+)", text):
        read |= {x.strip() for x in m.split(",") if x.strip()}
    # The calls survive where their results were trimmed.
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            found = re.search(r"/src/ledger/([a-z_]+)\.py", (tc.get("function") or {}).get("arguments", ""))
            if found and (tc.get("function") or {}).get("name") == "read_file":
                read.add(found.group(1))
    edits = [r for n, r in tool_results(messages) if n == "edit_file"]
    if edits:
        return Text("I added a one-line docstring to compute_total in src/ledger/billing.py.")
    todo = [m for m in LEDGER_MODULES if m not in read]
    if todo:
        return Call("read_file", {"path": f"{{workspace}}/src/ledger/{todo[0]}.py"})
    return Call("edit_file", {
        "path": "{workspace}/src/ledger/billing.py",
        "old_str": "def compute_total(entries: list[dict]) -> int:\n",
        "new_str": "def compute_total(entries: list[dict]) -> int:\n    \"\"\"Sum one ledger's entries; the package is a small double-entry ledger.\"\"\"\n",
    })


def _repeat(messages: list[dict[str, Any]]) -> Call | Text:
    return Call("glob", {"pattern": "**/*.gate-nothing-matches"})


def _empty(messages: list[dict[str, Any]]) -> Call | Text:
    return Text("")


def _subagent(messages: list[dict[str, Any]]) -> Call | Text:
    results = tool_results(messages)
    if results:
        return Text(f"Files found: {results[-1][1][:200]}")
    return Call("glob", {"pattern": "*"})


def _cloud_three_rounds(messages: list[dict[str, Any]]) -> Call | Text:
    reads = [r for n, r in tool_results(messages) if n == "read_file"]
    if len(reads) >= 2:
        return Text("Read both files.")
    return Call("read_file", {"path": "{workspace}/src/ledger/" + ("accounts.py" if not reads else "audit.py")})


def _fallback_answer(messages: list[dict[str, Any]]) -> Call | Text:
    return Text("Answered by the fallback.")


SCRIPTS: dict[str, Callable[[list[dict[str, Any]]], Call | Text]] = {
    "fallback": _fallback_answer,
    "agent": want_then_answer("agent", {
        "description": "List workspace files",
        "prompt": "[gate:sub] List the files in the workspace and report them.",
        "subagent_type": "explorer",
    }),
    "sub": _subagent,
    "cron": want_then_answer("cron_create", {
        "name": "check-build", "schedule": "0 9 * * 1-5", "command": "echo check the build"}),
    "wiki": want_then_answer("wiki_query", {"query": "deploy checklist"}),
    "memory": want_then_answer("lcm_grep", {"query": "release checklist"}),
    "mcp": want_then_answer("mcp__docs__lookup", {"library": "fastapi", "topic": "dependency injection"}),
    "coding": _coding,
    "repeat": _repeat,
    "empty": _empty,
    "hello": lambda messages: Text("Hello."),
    "cloud": _cloud_three_rounds,
}


# ---------------------------------------------------------------- the server


class ScriptedLlamaServer:
    def __init__(
        self,
        *,
        props: dict[str, Any] | None = None,
        n_ctx: int = 32768,
        chars_per_token: float = 3.73,
        workspace: str = "/nonexistent",
        native_tool_grammar: bool = True,
        model_id: str = MODEL_ID,
        report_n_ctx: bool = True,
        extra_props: dict[str, Any] | None = None,
        fail_script: str | None = None,
    ) -> None:
        self.props = dict(props or {})
        #: A script whose conversations this server refuses with a 401 (an
        #: auth failure is terminal, so the daemon serves the round from its
        #: fallback): the fallback tests' failing side.
        self.fail_script = fail_script
        #: The id /v1/models reports (and llama.cpp echoes per chunk): renaming
        #: the GGUF changes it.
        self.model_id = model_id
        #: False: /props carries no n_ctx, as when detection cannot read it.
        self.report_n_ctx = report_n_ctx
        self.extra_props = dict(extra_props or {})
        self.n_ctx = n_ctx
        self.chars_per_token = chars_per_token
        self.workspace = workspace
        #: False for a cloud API stand-in: no llama.cpp tool grammar (a cloud
        #: model is still only shown the offered tools; nothing scripted asks
        #: a cloud turn for an unoffered one).
        self.native_tool_grammar = native_tool_grammar
        self.exchanges: list[Exchange] = []
        #: Per-test scripts, consulted before SCRIPTS (the reachability suite
        #: registers one per tool: "reach_<tool>").
        self.scripts: dict[str, Callable[[list[dict[str, Any]]], Call | Text]] = {}
        self._last_key = ""
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "ScriptedLlamaServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def tokens(self, chars: int) -> int:
        return math.ceil(chars / self.chars_per_token)

    def of(self, script: str) -> list[Exchange]:
        with self._lock:
            return [e for e in self.exchanges if e.script == script]

    # -- deciding a reply ----------------------------------------------------

    def _script_for(self, payload: dict[str, Any]) -> str:
        msgs = payload.get("messages", [])
        has_system = any(m.get("role") == "system" for m in msgs)
        if not payload.get("tools") and (payload.get("max_tokens") or 0) <= 32:
            return "title"  # session titles: 24 tokens, no tools, the boot grammar attached (#298)
        if not has_system and not payload.get("tools"):
            if "Reply with the single word: ready" in all_text(msgs):
                return "probe"
            return "summary"
        first_user = next((_content_text(m.get("content")) for m in msgs if m.get("role") == "user"), "")
        found = _MARK.search(first_user)
        return found.group(1) if found else "unscripted"

    def _decide(self, script: str, payload: dict[str, Any]) -> Call | Text:
        msgs = payload.get("messages", [])
        if script == "title":
            return Text("Gate session")
        if script == "probe":
            return Text("ready")
        if script == "summary":
            body = all_text(msgs)
            seen = [m for m in LEDGER_MODULES if f"FILE ledger/{m}.py" in body]
            for listed in re.findall(r"files read so far: ([a-z_, ]+)", body):
                seen += [x.strip() for x in listed.split(",") if x.strip() and x.strip() not in seen]
            note = f" files read so far: {', '.join(seen)}." if seen else ""
            return Text("Summary: the user asked for work in the ledger project." + note)
        fn = self.scripts.get(script) or SCRIPTS.get(script)
        return fn(msgs) if fn else Text("I have no script for this conversation.")

    def _constrain(self, intent: Call | Text, payload: dict[str, Any]) -> Call | Text:
        """What the server lets out: an unoffered name never leaves as a call,
        and on Qwen's XML format (``<parameter=NAME>``) the grammar admits only
        the property names the offered tool declares, so any other argument
        never leaves either. A property-less tool is called with no arguments."""
        if isinstance(intent, Text):
            return intent
        tools = {(t.get("function") or {}).get("name"): t.get("function") or {} for t in payload.get("tools") or []}
        offered = set(tools)
        if offered and self.native_tool_grammar and intent.name not in offered:
            if "tool_search" in offered:
                return Call("tool_search", {"query": intent.name, "action": "select"})
            return Text(f"I don't have a {intent.name} tool available in this session.")
        if not offered:
            if payload.get("grammar"):
                # Prometheus's own GBNF covers every registered tool; the call is text.
                return Text(json.dumps({"name": intent.name, "arguments": intent.args}))
            return Text(f"(no tools offered; would have called {intent.name})")
        if self.native_tool_grammar:
            declared = set(((tools[intent.name].get("parameters") or {}).get("properties") or {}))
            if set(intent.args) - declared:
                return Call(intent.name, {k: v for k, v in intent.args.items() if k in declared})
        return intent

    def _fill(self, intent: Call | Text) -> Call | Text:
        if isinstance(intent, Call):
            raw = json.dumps(intent.args).replace("{workspace}", self.workspace)
            return Call(intent.name, json.loads(raw))
        return intent

    # -- HTTP --------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        srv = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args: object) -> None:
                pass

            def _json(self, code: int, obj: dict[str, Any]) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/props"):
                    props = dict(srv.props)
                    gen = dict(props.get("default_generation_settings") or {})
                    if srv.report_n_ctx:
                        gen["n_ctx"] = srv.n_ctx
                    else:
                        gen.pop("n_ctx", None)
                    props["default_generation_settings"] = gen
                    props["total_slots"] = 1
                    props["model_path"] = srv.model_id
                    props.setdefault("modalities", {"vision": False})
                    props.setdefault("build_info", "b1-9d57ce456")
                    props.update(srv.extra_props)
                    return self._json(200, props)
                if self.path.rstrip("/").endswith("/models"):
                    return self._json(200, {"object": "list", "data": [{"id": srv.model_id, "object": "model"}]})
                if self.path.startswith("/slots") or self.path.startswith("/metrics"):
                    return self._json(501, {"error": {"code": 501, "type": "not_supported_error"}})
                return self._json(200, {"status": "ok"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/tokenize"):
                    content = payload.get("content", "")
                    return self._json(200, {"tokens": [0] * srv.tokens(len(content))})
                if not self.path.endswith("/chat/completions"):
                    return self._json(404, {"error": {"code": 404, "message": "not found"}})
                self._chat(payload)

            def _chat(self, payload: dict[str, Any]) -> None:
                script = srv._script_for(payload)
                prompt = srv.tokens(rendered_chars(payload))
                key = _render_key(payload)
                with srv._lock:
                    common = 0
                    for a, b in zip(srv._last_key, key):
                        if a != b:
                            break
                        common += 1
                    srv._last_key = key
                cached = min(prompt, srv.tokens(common))
                ex = Exchange(script=script, request=payload, prompt_tokens=prompt, cached_tokens=cached, status=200)
                if srv.fail_script and script == srv.fail_script and payload.get("tools"):
                    ex.status = 401
                    with srv._lock:
                        srv.exchanges.append(ex)
                    return self._json(401, {"error": {"code": 401, "message": "invalid api key",
                                                      "type": "authentication_error"}})
                if prompt > srv.n_ctx:
                    ex.status = 400
                    with srv._lock:
                        srv.exchanges.append(ex)
                    return self._json(400, {"error": {
                        "code": 400,
                        "message": f"request ({prompt} tokens) exceeds the available context size "
                                   f"({srv.n_ctx} tokens), try increasing it",
                        "type": "exceed_context_size_error",
                        "n_prompt_tokens": prompt, "n_ctx": srv.n_ctx}})
                intent = srv._fill(srv._constrain(srv._decide(script, payload), payload))
                budget = max(1, min(int(payload.get("max_tokens") or 4096), srv.n_ctx - prompt))
                delta: dict[str, Any]
                if isinstance(intent, Call):
                    args = json.dumps(intent.args)
                    used = srv.tokens(len(args) + len(intent.name) + 30)
                    if used > budget:
                        args = args[: max(0, int(budget * srv.chars_per_token) - len(intent.name) - 30)]
                        used = budget
                    ex.finish_reason = "tool_calls"  # llama.cpp reports this even when the call was cut
                    delta = {"tool_calls": [{"index": 0, "id": f"call_{len(srv.exchanges)}", "type": "function",
                                             "function": {"name": intent.name, "arguments": args}}]}
                    ex.sent = {"call": intent.name, "arguments": args}
                else:
                    text = intent.text
                    used = srv.tokens(len(text))
                    ex.finish_reason = "stop"
                    if used > budget:
                        text = text[: int(budget * srv.chars_per_token)]
                        used = budget
                        ex.finish_reason = "length"
                    delta = {"content": text} if text else {}
                    ex.sent = {"text": text}
                with srv._lock:
                    srv.exchanges.append(ex)
                usage = {"prompt_tokens": prompt, "completion_tokens": used, "total_tokens": prompt + used,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                if not payload.get("stream"):
                    message: dict[str, Any] = {"role": "assistant", "content": delta.get("content") or ""}
                    if "tool_calls" in delta:
                        message["tool_calls"] = delta["tool_calls"]
                    return self._json(200, {"model": srv.model_id, "choices": [
                        {"index": 0, "message": message, "finish_reason": ex.finish_reason}], "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"model": srv.model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {"model": srv.model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": ex.finish_reason}]},
                    {"model": srv.model_id, "choices": [], "usage": usage,
                     "timings": {"cache_n": cached, "prompt_n": prompt - cached}},
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
