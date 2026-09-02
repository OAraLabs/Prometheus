"""Item 6 — @-references, the daemon half.

The composer's ``@file`` / ``@diff`` / ``@url`` chips arrive as an explicit
``references`` list on ``POST /api/chat/send`` and the WS ``send_message``
frame. The daemon resolves them BEFORE the turn is queued, scoped to the
session's workspace (item W) or the ``/api/files`` browse root, through the
gate's own denied-path matcher and ``web_fetch``'s SSRF guard.

Load-bearing assertions:

* scope FOLLOWS THE SESSION — a bound workspace is the only root, and a file
  that exists only in the fallback root is *not found* from a bound session;
* every refusal is LOUD and typed (``kind`` + HTTP status), on both surfaces,
  and the turn is not queued;
* a reference never rides a slash command;
* the resolved blocks reach the session WITH the user turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from prometheus.web.references import (
    DIFF_CAP_CHARS,
    FILE_CAP_BYTES,
    MAX_REFERENCES,
    Reference,
    ReferenceRefused,
    ReferenceResolver,
    parse_references,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _resolver(ws: Path | None, files_root: Path, **kw) -> ReferenceResolver:
    return ReferenceResolver(
        security_cfg=kw.pop("security_cfg", {"workspace_root": str(files_root)}),
        files_root=lambda: files_root,
        workspace_resolver=(lambda _sid: str(ws)) if ws is not None else None,
        **kw,
    )


def _git(cwd: Path, *argv: str) -> None:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    subprocess.run(["git", "-C", str(cwd), *argv], check=True, capture_output=True, env=env)


@pytest.fixture
def scope(tmp_path: Path) -> tuple[Path, Path]:
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("print('hello')\n")
    root = tmp_path / "browse"
    root.mkdir()
    (root / "only-in-root.txt").write_text("root file\n")
    return ws, root


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# wire shape
# --------------------------------------------------------------------------- #


def test_parse_accepts_the_three_types_and_defaults_diff_target():
    refs = parse_references([
        {"type": "file", "target": " src/app.py "},
        {"type": "diff"},
        {"type": "url", "target": "https://example.com"},
    ])
    assert refs == [
        Reference("file", "src/app.py"), Reference("diff", ""), Reference("url", "https://example.com"),
    ]


@pytest.mark.parametrize("raw, fragment", [
    ("not-a-list", "must be a list"),
    (["x"], "must be an object"),
    ([{"type": "image", "target": "x"}], "expected one of"),
    ([{"type": "file"}], "needs a target"),
    ([{"type": "url", "target": ""}], "needs a target"),
    ([{"type": "file", "target": 7}], "must be a string"),
    ([{"type": "diff"}] * (MAX_REFERENCES + 1), "too many"),
])
def test_parse_refuses_bad_shapes_as_bad_request(raw, fragment):
    with pytest.raises(ReferenceRefused) as ei:
        parse_references(raw)
    assert ei.value.kind == "bad_request" and ei.value.status == 400
    assert fragment in str(ei.value)


# --------------------------------------------------------------------------- #
# file — scope follows the session
# --------------------------------------------------------------------------- #


def test_file_resolves_inside_the_bound_workspace(scope):
    ws, root = scope
    blocks = _run(_resolver(ws, root).resolve("s1", [Reference("file", "src/app.py")]))
    assert len(blocks) == 1 and blocks[0].type == "text"
    assert '<reference type="file" path="src/app.py"' in blocks[0].text
    assert "print('hello')" in blocks[0].text


def test_bound_workspace_is_the_only_root(scope):
    # The file exists in the fallback browse root — but this session has a
    # workspace, so the browse root is NOT consulted. Not "outside", not found.
    ws, root = scope
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(ws, root).resolve("s1", [Reference("file", "only-in-root.txt")]))
    assert ei.value.kind == "not_found"
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(ws, root).resolve("s1", [Reference("file", str(root / "only-in-root.txt"))]))
    assert ei.value.kind == "forbidden" and ei.value.status == 403


def test_without_a_workspace_the_browse_root_is_the_scope(scope):
    ws, root = scope
    blocks = _run(_resolver(None, root).resolve("s1", [Reference("file", "only-in-root.txt")]))
    assert "root file" in blocks[0].text
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(None, root).resolve("s1", [Reference("file", str(ws / "src" / "app.py"))]))
    assert ei.value.kind == "forbidden"


def test_dotdot_and_symlink_escapes_are_forbidden(scope, tmp_path):
    ws, root = scope
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (ws / "link").symlink_to(outside)
    r = _resolver(ws, root)
    for target in ("../outside.txt", "link", "src/../../outside.txt"):
        with pytest.raises(ReferenceRefused) as ei:
            _run(r.resolve("s1", [Reference("file", target)]))
        assert ei.value.kind == "forbidden", target


def test_denied_paths_use_the_gates_matcher(scope):
    ws, root = scope
    (ws / "secrets").mkdir()
    (ws / "secrets" / "key.txt").write_text("k\n")
    r = _resolver(ws, root, security_cfg={"denied_paths": [str(ws / "secrets")]})
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("file", "secrets/key.txt")]))
    assert ei.value.kind == "forbidden"
    assert "denied" in str(ei.value)


def test_missing_directory_binary_and_oversize_files(scope):
    ws, root = scope
    r = _resolver(ws, root)
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("file", "src/nope.py")]))
    assert ei.value.kind == "not_found" and ei.value.status == 404
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("file", "src")]))
    assert ei.value.kind == "bad_request"
    (ws / "blob.bin").write_bytes(b"\x89PNG\x00\x00binary")
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("file", "blob.bin")]))
    assert "binary" in str(ei.value)
    (ws / "big.txt").write_bytes(b"x" * (FILE_CAP_BYTES + 10))
    blocks = _run(r.resolve("s1", [Reference("file", "big.txt")]))
    assert 'truncated="true"' in blocks[0].text and "...[truncated" in blocks[0].text
    assert len(blocks[0].text) < FILE_CAP_BYTES + 400


def test_workspace_lookup_failure_is_surfaced_not_collapsed(scope):
    # A resolver that raises must NOT silently fall back to the browse root —
    # a reference resolved against the wrong root is the plausible-degraded-
    # state trap (Standing-Principles §23).
    ws, root = scope

    def boom(_sid):
        raise RuntimeError("store is down")

    r = ReferenceResolver(security_cfg=None, files_root=lambda: root, workspace_resolver=boom)
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("file", "only-in-root.txt")]))
    assert ei.value.kind == "unavailable" and ei.value.status == 503


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #


@pytest.fixture
def repo(scope) -> tuple[Path, Path]:
    ws, root = scope
    _git(ws, "init", "-q")
    _git(ws, "add", ".")
    _git(ws, "commit", "-q", "-m", "init")
    return ws, root


def test_diff_of_the_working_tree_in_the_session_workspace(repo):
    ws, root = repo
    (ws / "src" / "app.py").write_text("print('changed')\n")
    blocks = _run(_resolver(ws, root).resolve("s1", [Reference("diff", "")]))
    text = blocks[0].text
    assert '<reference type="diff" ref="worktree"' in text
    assert "+print('changed')" in text and "-print('hello')" in text


def test_diff_against_a_ref_and_no_changes(repo):
    ws, root = repo
    blocks = _run(_resolver(ws, root).resolve("s1", [Reference("diff", "HEAD")]))
    assert "(no changes)" in blocks[0].text
    (ws / "src" / "app.py").write_text("print('v2')\n")
    _git(ws, "commit", "-q", "-am", "v2")
    blocks = _run(_resolver(ws, root).resolve("s1", [Reference("diff", "HEAD~1...HEAD")]))
    assert "+print('v2')" in blocks[0].text


@pytest.mark.parametrize("ref", ["--output=/tmp/x", "-p", "HEAD --stat", "a;b", "$(x)"])
def test_option_shaped_or_spaced_refs_never_reach_git(repo, ref):
    ws, root = repo
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(ws, root).resolve("s1", [Reference("diff", ref)]))
    assert ei.value.kind == "bad_request" and "not a valid git revision" in str(ei.value)


def test_unknown_ref_and_non_repo_are_refused(repo, scope):
    ws, root = repo
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(ws, root).resolve("s1", [Reference("diff", "no-such-ref")]))
    assert ei.value.kind == "bad_request"
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(None, root).resolve("s1", [Reference("diff", "")]))
    assert ei.value.kind == "not_found" and "not inside a git repository" in str(ei.value)


def test_diff_is_capped(repo):
    ws, root = repo
    (ws / "src" / "app.py").write_text("\n".join(f"line {i}" for i in range(20000)) + "\n")
    blocks = _run(_resolver(ws, root).resolve("s1", [Reference("diff", "")]))
    assert 'truncated="true"' in blocks[0].text
    assert len(blocks[0].text) < DIFF_CAP_CHARS + 400


# --------------------------------------------------------------------------- #
# url
# --------------------------------------------------------------------------- #


class _Page:
    def __init__(self, url, body="page text", status=200, content_type="text/html", truncated=False):
        self.url, self.body, self.status, self.content_type, self.truncated = url, body, status, content_type, truncated


def test_url_goes_through_the_shared_fetcher(scope, monkeypatch):
    ws, root = scope
    seen = {}

    async def fake_fetch(url, *, max_chars):
        seen["url"], seen["max_chars"] = url, max_chars
        return _Page(url)

    monkeypatch.setattr("prometheus.tools.builtin.web_fetch._is_safe_url", lambda _u: True)
    blocks = _run(_resolver(ws, root, fetch_url=fake_fetch).resolve(
        "s1", [Reference("url", "https://example.com/doc")]))
    assert seen["url"] == "https://example.com/doc" and seen["max_chars"] > 0
    assert '<reference type="url" url="https://example.com/doc" status="200"' in blocks[0].text
    assert "page text" in blocks[0].text


def test_url_refusals(scope, monkeypatch):
    ws, root = scope

    async def never(url, *, max_chars):  # pragma: no cover — must not be reached
        raise AssertionError("fetch must not run")

    r = _resolver(ws, root, fetch_url=never)
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("url", "ftp://example.com/x")]))
    assert ei.value.kind == "bad_request"
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("url", "file:///etc/passwd")]))
    assert ei.value.kind == "bad_request"
    # Loopback resolves to a private address — the SSRF guard, not a mock.
    with pytest.raises(ReferenceRefused) as ei:
        _run(r.resolve("s1", [Reference("url", "http://127.0.0.1:8005/api/status")]))
    assert ei.value.kind == "forbidden" and ei.value.status == 403

    async def failing(url, *, max_chars):
        raise ConnectionError("boom")

    monkeypatch.setattr("prometheus.tools.builtin.web_fetch._is_safe_url", lambda _u: True)
    with pytest.raises(ReferenceRefused) as ei:
        _run(_resolver(ws, root, fetch_url=failing).resolve("s1", [Reference("url", "https://example.com")]))
    assert ei.value.kind == "fetch_failed" and ei.value.status == 502


# --------------------------------------------------------------------------- #
# REST — POST /api/chat/send
# --------------------------------------------------------------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402


@register_double("at_references._RefBridge", replaces="prometheus.web.ws_server.WebSocketBridge")
class _RefBridge:
    def __init__(self, fail: ReferenceRefused | None = None) -> None:
        self.dispatched: list[dict] = []
        self.resolved: list[tuple[str, list]] = []
        self.fail = fail

    async def resolve_references(self, session_id, refs):
        self.resolved.append((session_id, refs))
        if self.fail:
            raise self.fail
        from prometheus.engine.messages import TextBlock
        return [TextBlock(text=f"<reference type={r.type!r}>{r.target}</reference>") for r in refs]

    async def dispatch_user_message(self, session_id, content, **kw):
        self.dispatched.append({"session_id": session_id, "content": content, **kw})


def _client(bridge):
    app = create_app({})
    app.state.ws_bridge = bridge
    return TestClient(app)


def test_rest_resolves_then_dispatches_with_blocks():
    br = _RefBridge()
    resp = _client(br).post("/api/chat/send", json={
        "session_id": "web:t", "message": "explain this",
        "references": [{"type": "file", "target": "src/app.py"}, {"type": "diff"}],
    })
    assert resp.status_code == 200 and resp.json()["status"] == "sent"
    assert br.resolved == [("web:t", [Reference("file", "src/app.py"), Reference("diff", "")])]
    assert len(br.dispatched) == 1
    blocks = br.dispatched[0]["blocks"]
    assert [b.text for b in blocks] == [
        "<reference type='file'>src/app.py</reference>", "<reference type='diff'></reference>",
    ]


def test_rest_empty_references_dispatches_plainly():
    br = _RefBridge()
    resp = _client(br).post("/api/chat/send", json={"session_id": "web:t", "message": "hi", "references": []})
    assert resp.status_code == 200
    assert br.resolved == [] and "blocks" not in br.dispatched[0]


def test_rest_malformed_references_are_400_and_nothing_is_dispatched():
    br = _RefBridge()
    resp = _client(br).post("/api/chat/send", json={
        "session_id": "web:t", "message": "hi", "references": [{"type": "image", "target": "x"}],
    })
    assert resp.status_code == 400 and resp.json()["kind"] == "bad_request"
    assert br.dispatched == [] and br.resolved == []


def test_rest_refuses_references_on_a_slash_command():
    br = _RefBridge()
    resp = _client(br).post("/api/chat/send", json={
        "session_id": "web:t", "message": "/status", "references": [{"type": "diff"}],
    })
    assert resp.status_code == 400 and "slash command" in resp.json()["error"]
    assert br.dispatched == [] and br.resolved == []


@pytest.mark.parametrize("kind, status", [
    ("forbidden", 403), ("not_found", 404), ("fetch_failed", 502), ("unavailable", 503), ("bad_request", 400),
])
def test_rest_refusal_kinds_map_to_statuses_and_do_not_queue_a_turn(kind, status):
    br = _RefBridge(fail=ReferenceRefused(kind, f"@file x: {kind}"))
    resp = _client(br).post("/api/chat/send", json={
        "session_id": "web:t", "message": "hi", "references": [{"type": "file", "target": "x"}],
    })
    assert resp.status_code == status
    assert resp.json() == {"error": f"@file x: {kind}", "kind": kind}
    assert br.dispatched == []


def test_rest_bridge_without_resolver_is_503_when_references_are_sent():
    class _Legacy:
        def __init__(self):
            self.dispatched = []

        async def dispatch_user_message(self, session_id, content, **kw):
            self.dispatched.append(content)

    br = _Legacy()
    resp = _client(br).post("/api/chat/send", json={
        "session_id": "web:t", "message": "hi", "references": [{"type": "diff"}],
    })
    assert resp.status_code == 503 and br.dispatched == []


# --------------------------------------------------------------------------- #
# WS — send_message frame
# --------------------------------------------------------------------------- #

from prometheus.web.ws_server import WebSocketBridge  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def of(self, ftype: str) -> list[dict]:
        return [f["payload"] for f in self.frames if f["type"] == ftype]


class _FakeSession:
    def __init__(self) -> None:
        self.turns: list[tuple[str, list | None]] = []
        self.messages: list = []

    def add_user_message(self, content: str, blocks=None) -> int:
        self.turns.append((content, blocks))
        self.messages.append(content)
        return len(self.messages)

    def last_persisted_row_id(self) -> int:
        return 7


class _FakeMgr:
    def __init__(self) -> None:
        self.session = _FakeSession()

    def get(self, session_id):
        return self.session

    def get_or_create(self, session_id):
        return self.session


class _StubResolver:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    async def resolve(self, session_id, refs):
        self.calls.append((session_id, refs))
        if self.fail:
            raise self.fail
        from prometheus.engine.messages import TextBlock
        return [TextBlock(text=f"[{r.type}:{r.target}]") for r in refs]


def _ws_bridge(fail=None):
    mgr = _FakeMgr()
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=None)
    bridge.reference_resolver = _StubResolver(fail)
    rec = _Recorder()
    bridge._clients.add(rec)
    return bridge, rec, mgr.session


def _frame(payload: dict) -> str:
    return json.dumps({"type": "send_message", "payload": payload})


@pytest.mark.asyncio
async def test_ws_references_become_blocks_on_the_user_turn():
    bridge, rec, session = _ws_bridge()
    await bridge._handle_client_message(rec, _frame({
        "session_id": "web:t", "content": "look at this",
        "references": [{"type": "file", "target": "src/app.py"}],
    }))
    assert rec.of("error") == []
    assert len(session.turns) == 1
    content, blocks = session.turns[0]
    assert content == "look at this"
    assert [b.text for b in blocks] == ["[file:src/app.py]"]
    assert bridge.reference_resolver.calls == [("web:t", [Reference("file", "src/app.py")])]


@pytest.mark.asyncio
async def test_ws_unresolvable_reference_is_an_error_frame_and_no_turn():
    bridge, rec, session = _ws_bridge(fail=ReferenceRefused("forbidden", "@file x: outside"))
    await bridge._handle_client_message(rec, _frame({
        "session_id": "web:t", "content": "look", "references": [{"type": "file", "target": "x"}],
    }))
    errs = rec.of("error")
    assert errs and errs[0]["kind"] == "forbidden" and errs[0]["message"] == "@file x: outside"
    assert errs[0]["session_id"] == "web:t"
    assert session.turns == [] and rec.of("chat_message") == []


@pytest.mark.asyncio
async def test_ws_malformed_and_slash_references_are_bad_request():
    bridge, rec, session = _ws_bridge()
    await bridge._handle_client_message(rec, _frame({
        "session_id": "web:t", "content": "look", "references": "nope",
    }))
    await bridge._handle_client_message(rec, _frame({
        "session_id": "web:t", "content": "/status", "references": [{"type": "diff"}],
    }))
    kinds = [e["kind"] for e in rec.of("error")]
    assert kinds == ["bad_request", "bad_request"]
    assert "slash command" in rec.of("error")[1]["message"]
    assert session.turns == [] and bridge.reference_resolver.calls == []


@pytest.mark.asyncio
async def test_ws_without_references_is_unchanged():
    bridge, rec, session = _ws_bridge()
    await bridge._handle_client_message(rec, _frame({"session_id": "web:t", "content": "plain"}))
    assert session.turns == [("plain", None)] and bridge.reference_resolver.calls == []
