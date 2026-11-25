"""The shared ACP core (framing, negotiation, session map).

Fixtures are the other side's real code: wire lines captured
from the TCK's conforming v2 fixture agent (tests/acp_fixtures/tck_exchanges.json,
regenerate with capture.py) and shapes read from the pinned schema itself.
Nothing here hand-writes an ACP payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.acp import types
from prometheus.acp.framing import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
    FrameError,
    Notification,
    Request,
    Response,
    error_response,
    parse_frame,
    render,
)
from prometheus.acp.negotiation import PROTOCOL_VERSION, SCHEMA_TAG, negotiate
from prometheus.acp.sessions import SessionMap, message_update, replay_complete_update
from prometheus.memory.lcm_conversation_store import LCMConversationStore

from .acp_fixtures import pinned_schema, schema_def, tck_exchanges


# ── the pin ──────────────────────────────────────────────────────────────────


def test_schema_pin_is_the_committed_tag():
    import hashlib

    raw = (Path(__file__).parent.parent / "src" / "prometheus" / "acp" / "schema" / "schema.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "86dae71cce55d412b927383cb3838ea560bdbf683ea8903bede75785fc79c494"
    )
    assert SCHEMA_TAG == "schema-v2.0.0-beta.4"
    assert PROTOCOL_VERSION == 2


def test_generated_types_cover_every_object_def_in_the_pin():
    """Every concrete object in the pinned schema has a generated model whose
    required fields match. (Scalar aliases and unions are inlined by the
    generator; objects are the load-bearing shapes.)"""
    names = set(dir(types))
    for name, d in pinned_schema()["$defs"].items():
        if d.get("type") != "object" and "properties" not in d:
            continue
        cls = next((getattr(types, c) for c in (name, name + "Base", name + "2")
                    if c in names), None)
        if cls is None or not hasattr(cls, "model_fields"):
            continue  # union-wrapped objects are covered via their variants
        if "root" in cls.model_fields and len(cls.model_fields) == 1:
            continue  # a RootModel union wrapper — its variants carry the fields
        req = set(d.get("required", []))
        model_req = {(f.alias or n) for n, f in cls.model_fields.items() if f.is_required()}
        # A required field whose schema is a single const may be generated as a
        # defaulted Literal (the generator's one-literal-as-default rule): the
        # value is still pinned, omission still yields it. Accept that shape —
        # but only when the pin really is a single const.
        defs = pinned_schema()["$defs"]

        def _single_const(prop: dict) -> bool:
            node = prop
            if "allOf" in node and len(node["allOf"]) == 1:
                ref = node["allOf"][0].get("$ref", "")
                node = defs.get(ref.split("/")[-1], {})
            for kw in ("anyOf", "oneOf"):
                if kw in node:
                    return len(node[kw]) == 1 and "const" in node[kw][0]
            return "const" in node

        props = d.get("properties", {})
        defaulted_consts = {
            (f.alias or n) for n, f in cls.model_fields.items()
            if not f.is_required() and (f.alias or n) in req
            and _single_const(props.get(f.alias or n, {}))
        }
        assert model_req | defaulted_consts == req, f"{name}: required fields drifted from the pin"
        assert model_req <= req, f"{name}: model requires a field the pin does not"


# ── framing: judged against the TCK agent's real lines ──────────────────────


def test_every_captured_tck_line_parses_as_a_frame():
    for entry in tck_exchanges():
        frame = parse_frame(entry["response_raw"])
        assert isinstance(frame, (Response, Notification))


def test_captured_tck_results_validate_against_the_generated_types():
    by_method = {}
    for entry in tck_exchanges():
        msg = json.loads(entry["response_raw"])
        if "result" in msg:
            by_method.setdefault(entry["request"]["method"], msg["result"])
    types.InitializeResponse.model_validate(by_method["initialize"])
    types.NewSessionResponse.model_validate(by_method["session/new"])
    types.PromptResponse.model_validate(by_method["session/prompt"])
    types.ResumeSessionResponse.model_validate(by_method["session/resume"])
    types.ListSessionsResponse.model_validate(by_method["session/list"])


def test_captured_tck_notifications_validate_as_session_updates():
    seen_kinds = set()
    for entry in tck_exchanges():
        msg = json.loads(entry["response_raw"])
        if msg.get("method") == "session/update":
            n = types.UpdateSessionNotification.model_validate(msg["params"])
            seen_kinds.add(msg["params"]["update"]["sessionUpdate"])
    # The capture must actually exercise the lifecycle, not just parse.
    assert {"user_message", "agent_message", "state_update", "replay_complete"} <= seen_kinds


def test_tck_replay_ends_with_the_marker_before_the_response():
    resume_entries = [e for e in tck_exchanges() if e["request"]["method"] == "session/resume"]
    kinds = [json.loads(e["response_raw"]).get("params", {}).get("update", {}).get("sessionUpdate")
             for e in resume_entries]
    assert "replay_complete" in kinds
    marker_at = kinds.index("replay_complete")
    # every replayed message precedes the marker; the response comes last
    assert all(k in ("user_message", "agent_message", "agent_message_chunk")
               for k in kinds[:marker_at])
    assert "result" in json.loads(resume_entries[-1]["response_raw"])


def test_tck_idempotent_retry_returns_the_same_message_id():
    prompts = [json.loads(e["response_raw"]) for e in tck_exchanges()
               if e["request"]["method"] == "session/prompt" and "result" in json.loads(e["response_raw"])]
    assert len(prompts) == 2
    assert prompts[0]["result"]["messageId"] == prompts[1]["result"]["messageId"]


# ── framing: refusals (shapes from the schema's own rules) ──────────────────


def test_batching_is_refused_v2_has_no_arrays():
    with pytest.raises(FrameError) as ei:
        parse_frame("[]")
    assert ei.value.code == INVALID_REQUEST


def test_null_request_id_is_refused():
    with pytest.raises(FrameError):
        parse_frame('{"jsonrpc":"2.0","id":null,"method":"initialize"}')


def test_boolean_id_is_refused():
    with pytest.raises(FrameError):
        parse_frame('{"jsonrpc":"2.0","id":true,"method":"initialize"}')


def test_garbage_is_parse_error():
    with pytest.raises(FrameError) as ei:
        parse_frame("{not json")
    assert ei.value.code == PARSE_ERROR


def test_roundtrip_request_notification_response():
    for frame in (
        Request(id=7, method="session/prompt", params={"sessionId": "s"}),
        Notification(method="session/cancel", params={"sessionId": "s"}),
        Response(id=7, result={"messageId": "m"}),
        error_response(7, METHOD_NOT_FOUND, "no such method"),
    ):
        assert parse_frame(render(frame)) == frame


def test_error_response_may_carry_null_id():
    r = parse_frame(render(error_response(None, PARSE_ERROR, "bad json")))
    assert r.id is None and r.is_error


# ── negotiation ──────────────────────────────────────────────────────────────


def _init_params(version: int) -> dict:
    # required fields read from the pin, not guessed
    req = schema_def("InitializeRequest")
    assert set(req["required"]) == {"protocolVersion", "info"}
    return {"protocolVersion": version, "info": {"name": "test", "version": "0"}}


def test_negotiate_v2_answers_v2():
    resp = negotiate(_init_params(2), agent_name="prometheus", agent_version="0.9.24")
    assert resp.protocol_version == 2
    assert resp.info.name == "prometheus"


def test_v1_only_peer_is_refused_with_32022():
    with pytest.raises(FrameError) as ei:
        negotiate(_init_params(1), agent_name="prometheus", agent_version="0.9.24")
    assert ei.value.code == UNSUPPORTED_PROTOCOL_VERSION


def test_malformed_initialize_is_invalid_params():
    with pytest.raises(FrameError) as ei:
        negotiate({"protocolVersion": 2}, agent_name="p", agent_version="0")
    assert ei.value.code == -32602


# ── session map over a real store ────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path):
    return LCMConversationStore(db_path=tmp_path / "lcm.db")


def test_new_prompt_resume_delete_roundtrip(store):
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    assert sid.startswith("acp:")
    ins = m.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "hello"}]})
    resp, replay = m.resume_session({"sessionId": sid, "cwd": "/tmp", "replayFrom": {"type": "start"}})
    assert len(replay) == 1
    assert replay[0]["messageId"] == ins.message_id
    # our replay updates validate against the same union the TCK's lines do
    for u in replay + [replay_complete_update()]:
        types.UpdateSessionNotification.model_validate({"sessionId": sid, "update": u})
    listed = m.list_sessions({}).sessions
    assert [s.session_id for s in listed] == [sid]
    assert listed[0].cwd == "/tmp"
    m.delete_session({"sessionId": sid})
    assert m.list_sessions({}).sessions == []


def test_idempotency_is_durable_across_restart(store):
    m1 = SessionMap(store)
    sid = m1.new_session({"cwd": "/tmp"}).session_id
    first = m1.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}],
                              "idempotencyKey": "k1"})
    # a NEW map over the same store = a daemon restart
    m2 = SessionMap(store)
    again = m2.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}],
                              "idempotencyKey": "k1"})
    assert again.message_id == first.message_id
    assert len(store.get_all_messages(sid)) == 1


def test_overlong_idempotency_key_refused_never_truncated(store):
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    with pytest.raises(FrameError) as ei:
        m.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "x"}],
                         "idempotencyKey": "k" * 129})
    assert ei.value.code == -32602


def test_foreign_session_is_not_found(store):
    m = SessionMap(store)
    with pytest.raises(FrameError) as ei:
        m.insert_prompt({"sessionId": "telegram:123", "prompt": [{"type": "text", "text": "x"}]})
    assert ei.value.code == -32002


def test_unsupported_replay_cursor_is_invalid_params(store):
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    m.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]})
    with pytest.raises(FrameError) as ei:
        m.resume_session({"sessionId": sid, "cwd": "/tmp", "replayFrom": {"type": "_custom"}})
    assert ei.value.code == -32602


def test_message_update_uses_durable_rowid(store):
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    ins = m.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": "x"}]})
    part = store.messages_after_id(0, session_id=sid)[0]
    u = message_update(part)
    assert u["messageId"] == str(part.row_id) == ins.message_id
    assert u["content"] == [{"type": "text", "text": "x"}]

# ── the off-by-default gate ──────────────────────────────────────────────────


def test_acp_gate_defaults_off():
    from prometheus.acp.gate import acp_enabled

    assert acp_enabled(None) is False
    assert acp_enabled({}) is False
    assert acp_enabled({"acp": {}}) is False
    assert acp_enabled({"acp": {"enabled": False}}) is False
    assert acp_enabled({"acp": {"enabled": "yes"}}) is False  # only literal true
    assert acp_enabled({"acp": {"enabled": True}}) is True


def test_shipped_template_keeps_acp_off():
    import yaml

    template = Path(__file__).parent.parent / "config" / "prometheus.yaml.default"
    cfg = yaml.safe_load(template.read_text())
    from prometheus.acp.gate import acp_enabled

    assert acp_enabled(cfg) is False


def test_no_daemon_module_imports_the_acp_core():
    """With ACP off, nothing on any daemon path imports the core. A wiring
    point must go through the gate; a top-level import here would put the
    generated types on every boot's import path (overhead gate)."""
    src = Path(__file__).parent.parent / "src" / "prometheus"
    offenders = []
    for py in src.rglob("*.py"):
        if "acp" in py.relative_to(src).parts[:1]:
            continue
        text = py.read_text()
        if "from prometheus.acp" in text or "import prometheus.acp" in text:
            offenders.append(str(py.relative_to(src)))
    assert offenders == [], f"daemon modules import the ACP core unconditionally: {offenders}"


# ── review fixes: key-length boundary, atomicity, rowid survival ────────────


@pytest.mark.parametrize("n", [120, 121, 128])
def test_key_length_boundary_retry_is_idempotent_or_refused(store, n):
    """The claim is stored WITH the 8-char 'acp-key:' prefix in a 128-char
    column. A 120-char key fits (128 stored) and a retry must return the first
    messageId; 121 and 128 must be REFUSED by name — never accepted and then
    silently dropped by the store's length guard (review finding 1)."""
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    key = "k" * n
    params = {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}], "idempotencyKey": key}
    if n <= 120:
        first = m.insert_prompt(params)
        again = m.insert_prompt(params)
        assert again.message_id == first.message_id
        assert len(store.get_all_messages(sid)) == 1
        # the claim really is stored (not silently dropped)
        assert f"acp-key:{key}" in store.get_message_client_ids(sid).values()
    else:
        with pytest.raises(FrameError) as ei:
            m.insert_prompt(params)
        assert ei.value.code == -32602
        assert store.get_all_messages(sid) == []


def test_concurrent_same_key_prompts_insert_once(store):
    """Two threads, one key: the lookup-insert-record unit is serialized by
    the store transaction, so exactly one row exists and both callers get the
    same messageId (review finding 2)."""
    import threading

    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    params = {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}], "idempotencyKey": "race"}
    results: list = []
    barrier = threading.Barrier(2)

    def send():
        barrier.wait()
        results.append(m.insert_prompt(params).message_id)

    threads = [threading.Thread(target=send) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(store.get_all_messages(sid)) == 1


def test_crash_between_insert_and_record_rolls_back(store):
    """A failure of the claim record must not leave the inserted row behind:
    the whole unit rolls back, so a retry inserts cleanly (review finding 2)."""
    m = SessionMap(store)
    sid = m.new_session({"cwd": "/tmp"}).session_id
    params = {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}], "idempotencyKey": "crash"}

    def boom(*a, **k):
        raise RuntimeError("simulated crash after insert, before record")

    orig = store.set_message_client_id
    store.set_message_client_id = boom
    try:
        with pytest.raises(RuntimeError):
            m.insert_prompt(params)
    finally:
        store.set_message_client_id = orig
    assert store.get_all_messages(sid) == []  # insert rolled back with the unit
    # the retry now inserts exactly once
    first = m.insert_prompt(params)
    again = m.insert_prompt(params)
    assert again.message_id == first.message_id
    assert len(store.get_all_messages(sid)) == 1


def test_purge_vacuum_preserves_other_sessions_rowids(store):
    """messageId is str(rowid) and lcm_messages has no INTEGER PRIMARY KEY;
    the purge path runs VACUUM. SQLite only PROMISES rowid stability across
    VACUUM for INTEGER PRIMARY KEY tables — it holds in practice, and this
    test runs the REAL purge path so a SQLite change cannot silently break
    ACP message ids (session_forks.origin_rowid and FTS rely on the same
    property) (review finding 3)."""
    m = SessionMap(store)
    keep = m.new_session({"cwd": "/tmp"}).session_id
    drop = m.new_session({"cwd": "/tmp"}).session_id
    ids = {}
    for sid in (keep, drop):
        for i in range(3):
            ins = m.insert_prompt({"sessionId": sid, "prompt": [{"type": "text", "text": f"m{i}"}]})
            ids.setdefault(sid, []).append(ins.message_id)
    before = {p.row_id: p.content for p in store.messages_after_id(0, session_id=keep)}
    store.purge_session(drop)  # the real path: DELETE + fts cleanup + VACUUM
    after = {p.row_id: p.content for p in store.messages_after_id(0, session_id=keep)}
    assert after == before, "VACUUM renumbered surviving rows — ACP messageIds would dangle"
    assert [str(r) for r in after] == ids[keep]
